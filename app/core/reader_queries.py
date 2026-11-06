"""Read-only, on-demand public projections of Reader-v3's saved results.

No AnalysisStore, runner, agent, budget ledger, subprocess, or model client is
used here. Original files are opened only for reading. Public responses are
whitelists, never a dump of AnalysisDocument or a concatenation of export files.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
from threading import RLock
from typing import Any
from urllib.parse import quote, urlencode

from pydantic import ValidationError
from app.models.document_v2 import PaperDocument
from app.models.reader_v3 import AnalysisDocument
from app.core.mapping_views import check_saved_mapping, display_view, presentation_analysis

API_VERSION = "reader.read.v1"
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\Z")
CONTENT_PARTS = {"source_text", "source_context", "caption", "visual_observation", "table_cells", "equation"}
CELL_FIELDS = ("cell_id", "row", "col", "text", "role", "rowspan", "colspan", "image_asset_id")
IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml"}


class ReadError(Exception):
    def __init__(self, code: str, message: str, status: int = 400, **details: Any):
        super().__init__(message)
        self.code, self.message, self.status, self.details = code, message, status, details

    def envelope(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


def identifier(value: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value) or value in {".", ".."}:
        raise ReadError("INVALID_ID", "Use a canonical source ID, not a filesystem path.")
    return value


def unique(values):
    return list(dict.fromkeys(v for v in values if v is not None))


def pick(value: dict, keys) -> dict:
    return {key: deepcopy(value[key]) for key in keys if key in value}


def anchor(value, *, compact=False):
    result = value.model_dump(mode="json") if value is not None else None
    if result and compact:
        result = deepcopy(result)
        for key in ("exact", "prefix", "suffix"):
            result["selector"].pop(key, None)
    return result


def preview(text: str, size=240) -> dict:
    text = str(text or "")
    return {"text": text[:size], "is_complete": len(text) <= size}


def page(values: list, offset: int, limit: int, *, max_limit=100):
    if (type(offset) is not int or offset < 0 or type(limit) is not int
            or not 1 <= limit <= max_limit):
        raise ReadError("INVALID_PAGE", f"offset must be nonnegative and limit must be 1..{max_limit}.")
    selected = values[offset:offset + limit]
    stop = offset + len(selected)
    return selected, {"offset": offset, "limit": limit, "total": len(values),
                      "has_more": stop < len(values), "next_offset": stop if stop < len(values) else None}


def safe_note(message: Any) -> str:
    """Operational messages only; scientific quotations are never rewritten."""
    text = str(message or "")
    text = re.sub(r"[A-Za-z]:[\\/][^\n\r\"']+", "[local path omitted]", text)
    text = re.sub(r"(?:/home/|/mnt/|/Users/|file://)[^\s\"']+", "[local path omitted]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[credential omitted]", text)
    return text[:1200]


@dataclass
class _Cached:
    stamp: tuple
    digest: str
    raw_bytes: int
    payload: dict
    model: Any = None


class ReadRepository:
    """Read files on change; keep a bounded, process-local cache.

    Cache limits count source bytes, not the Python heap size. A cache miss still
    parses the complete internal JSON on the server; it is NOT sent to a browser.
    There is no persistent index, export, migration, or write side effect.
    """
    def __init__(self, project: str | Path, *, cache_entries=6,
                 cache_source_bytes=64 * 1024 * 1024, max_file_bytes=128 * 1024 * 1024):
        self.project = Path(project).resolve()
        if not (self.project / "app").is_dir():
            raise ReadError("INVALID_PROJECT", "The configured project must contain app/.", 500)
        self.cache_entries = cache_entries
        self.cache_source_bytes = cache_source_bytes
        self.max_file_bytes = max_file_bytes
        self._cache: OrderedDict[Path, _Cached] = OrderedDict()
        self._lock = RLock()

    def _path(self, area: str, paper_id: str, name: str) -> Path:
        identifier(paper_id)
        base = self.project / "data" / area
        target = base / paper_id / name
        if any(p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction())
               for p in (self.project / "data", base, base / paper_id, target)):
            raise ReadError("LINKED_SOURCE", "Linked source paths are not served.", 409)
        if not target.resolve().is_relative_to(base.resolve()):
            raise ReadError("INVALID_SOURCE_PATH", "Source path escapes its configured storage root.", 400)
        return target

    @staticmethod
    def _stamp(path: Path):
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino

    def _load(self, path: Path, model_type) -> _Cached:
        with self._lock:
            try:
                stamp = self._stamp(path)
                old = self._cache.get(path)
                if old and old.stamp == stamp:
                    self._cache.move_to_end(path)
                    return old
                for _ in range(2):
                    if stamp[2] > self.max_file_bytes:
                        raise ReadError("SOURCE_TOO_LARGE", "This local reader's source-file limit was exceeded.", 413)
                    with path.open("rb") as handle:
                        content = handle.read(self.max_file_bytes + 1)
                    if len(content) > self.max_file_bytes:
                        raise ReadError("SOURCE_TOO_LARGE", "This local reader's source-file limit was exceeded.", 413)
                    after = self._stamp(path)
                    if stamp != after:
                        stamp = after
                        continue
                    try:
                        payload = json.loads(content.decode("utf-8-sig"))
                        if not isinstance(payload, dict):
                            raise ValueError("Source root must be an object")
                        model = model_type.model_validate(payload)
                    except (ValueError, UnicodeError, ValidationError):
                        raise ReadError("INVALID_SAVED_SOURCE", "The saved JSON cannot be read with the installed schema; no file was changed.", 409) from None
                    entry = _Cached(stamp, sha256(content).hexdigest(), len(content), payload, model)
                    self._cache[path] = entry
                    self._cache.move_to_end(path)
                    while self._cache and (len(self._cache) > self.cache_entries or
                            sum(e.raw_bytes for e in self._cache.values()) > self.cache_source_bytes):
                        self._cache.popitem(last=False)
                    return entry
                raise ReadError("SOURCE_CHANGING", "Source changed while being read; retry the same read request.", 503)
            except FileNotFoundError:
                raise ReadError("SOURCE_NOT_FOUND", "The requested saved source does not exist.", 404) from None
            except OSError:
                raise ReadError("SOURCE_READ_FAILED", "The saved source is temporarily unreadable.", 503) from None

    def paper(self, paper_id: str):
        cached = self._load(self._path("standardized", paper_id, "paper.json"), PaperDocument)
        if cached.model.paper_id != paper_id:
            raise ReadError("PAPER_ID_MISMATCH", "Stored paper identity does not match the requested paper.", 409)
        return cached.model

    def snapshot(self, paper_id: str, expected: str | None = None):
        paper = self.paper(paper_id)
        path = self._path("analysis", paper_id, "analysis.json")
        if path.is_file():
            cached = self._load(path, AnalysisDocument)
            data, revision = cached.model, "sha256:" + cached.digest
            if data.paper_id != paper_id or data.paper_revision != paper.revision:
                raise ReadError("REVISION_MISMATCH", "Saved analysis and canonical paper have different identities or revisions.", 409)
        else:
            data = AnalysisDocument(paper_id=paper_id, paper_revision=paper.revision)
            revision = "empty:" + sha256(paper.revision.encode()).hexdigest()
        # Always hash the actual bytes, even when old JSON has a stale snapshot_id.
        if expected and expected != revision:
            raise ReadError("SNAPSHOT_CHANGED", "The saved result changed; reload the first page or current Claim view.", 409,
                            current_snapshot_id=revision)
        return paper, data, revision

    @staticmethod
    def meta(paper, revision):
        return {"api_version": API_VERSION, "paper_id": paper.paper_id,
                "paper_revision": paper.revision, "snapshot_id": revision}

    @staticmethod
    def base(paper_id: str):
        return "/api/papers/" + quote(identifier(paper_id), safe="")

    def _asset_info(self, paper, asset_id):
        if not asset_id:
            return None
        source = paper.assets.get(asset_id)
        available = bool(source and source.status == "available")
        return {"asset_id": asset_id, "status": "available" if available else "missing",
                "url": (self.base(paper.paper_id) + "/assets/" + quote(asset_id, safe="") + "?" + urlencode({"paper_revision": paper.revision})) if available else None}

    def _claim_card(self, paper, data, cid, *, full=False):
        claim = data.claim_pool[cid]
        result = {"claim_id": cid, "type": claim.type, "scope": claim.scope,
                  "source_element_id": claim.source_element_id, "source_status": claim.source_status,
                  "source_anchor": anchor(claim.source_anchor, compact=not full),
                  "current_mapping_id": data.mapping_by_claim.get(cid),
                  "detail_url": self.base(paper.paper_id) + "/claims/" + quote(cid, safe="")}
        text = claim.source_quote or claim.model_quote
        result.update({"text": text} if full else {"preview": preview(text)})
        return result

    def overview(self, paper_id):
        paper, data, revision = self.snapshot(paper_id)
        return {**self.meta(paper, revision), "metadata": paper.metadata.model_dump(mode="json"),
                "counts": {"sections": len(paper.sections), "elements": len(paper.elements),
                           "claims": len(data.claim_pool), "evidence": len(data.evidence_pool),
                           "current_mappings": len(data.mapping_by_claim)},
                "quality": {"body": paper.quality.body, "figures": paper.quality.figures,
                            "tables": paper.quality.tables, "references": paper.quality.references,
                            "warnings": [safe_note(x) for x in paper.quality.warnings]},
                "routes": {name: self.base(paper_id) + "/" + name for name in ("sections", "claims", "evidence", "response")},
                "capabilities": {"read_saved_results": True, "selection_lookup": True,
                                 "submit_analysis": False, "submit_chat": False, "upload": False}}

    def sections(self, paper_id, offset=0, limit=50):
        paper = self.paper(paper_id)
        rows, pagination = page(paper.sections, offset, limit)
        return {"api_version": API_VERSION, "paper_id": paper_id, "paper_revision": paper.revision,
                "items": [{"section_id": s.section_id, "title": s.title, "level": s.level,
                           "parent_section_id": s.parent_section_id, "block_count": len(s.block_ids),
                           "elements_url": self.base(paper_id) + "/sections/" + quote(s.section_id, safe="") + "/elements"}
                          for s in rows], "page": pagination}

    def _element(self, paper, eid, *, cells=False, row_offset=0, row_limit=20):
        element = paper.elements.get(eid)
        if element is None:
            raise ReadError("ELEMENT_NOT_FOUND", "The requested canonical element does not exist.", 404)
        raw = element.model_dump(mode="json")
        result = pick(raw, ("element_id", "section_id", "kind", "label", "text", "caption", "alt_text",
                            "format_spans", "inline_math_spans", "caption_format_spans", "caption_inline_assets",
                            "latex", "mathml", "plain_text", "representation", "recognition_status"))
        result["detail_url"] = self.base(paper.paper_id) + "/elements/" + quote(eid, safe="")
        if raw.get("image_asset_id"):
            result["image"] = self._asset_info(paper, raw["image_asset_id"])
        if element.kind == "figure":
            for span in result.get("caption_inline_assets", []):
                span["image"] = self._asset_info(paper, span.get("asset_id"))
        if element.kind == "table":
            grid = (raw.get("grid") or {}).get("cells") or []
            result["cell_count"] = len(grid)
            result["row_count"] = len({c["row"] for c in grid})
            result["cells_loaded"] = cells
            if cells:
                result["table"] = self._table_page(paper, grid, row_offset, row_limit)
        return result

    def elements(self, paper_id, section_id, offset=0, limit=20, paper_revision=None):
        paper = self.paper(paper_id)
        if paper_revision and paper_revision != paper.revision:
            raise ReadError("REVISION_MISMATCH", "Reload the canonical paper before continuing.", 409)
        section = next((s for s in paper.sections if s.section_id == section_id), None)
        if section is None:
            raise ReadError("SECTION_NOT_FOUND", "The requested section does not exist.", 404)
        ids, pagination = page(section.block_ids, offset, limit, max_limit=50)
        return {"api_version": API_VERSION, "paper_id": paper_id, "paper_revision": paper.revision,
                "section_id": section_id, "element_order": ids,
                "elements": {eid: self._element(paper, eid) for eid in ids}, "page": pagination,
                "text_offset_unit": "unicode_code_points"}

    def element(self, paper_id, element_id, row_offset=0, row_limit=20, paper_revision=None):
        paper = self.paper(paper_id)
        if paper_revision and paper_revision != paper.revision:
            raise ReadError("REVISION_MISMATCH", "The source changed; refresh the reader.", 409)
        return {"api_version": API_VERSION, "paper_id": paper_id, "paper_revision": paper.revision,
                "element": self._element(paper, element_id, cells=True, row_offset=row_offset, row_limit=row_limit),
                "rendering_note": "Render source text as text. Sanitize MathML; never inject publisher HTML."}

    def claims(self, paper_id, offset=0, limit=50, element_id=None, snapshot_id=None):
        if offset and not snapshot_id:
            raise ReadError("SNAPSHOT_REQUIRED", "Pass snapshot_id from the first page when reading subsequent pages.")
        paper, data, revision = self.snapshot(paper_id, snapshot_id)
        if element_id is not None and element_id not in paper.elements:
            raise ReadError("ELEMENT_NOT_FOUND", "The source element filter is unknown.", 404)
        ids = [cid for cid, c in data.claim_pool.items() if element_id is None or c.source_element_id == element_id]
        ids, pagination = page(ids, offset, limit)
        return {**self.meta(paper, revision), "items": [self._claim_card(paper, data, cid) for cid in ids], "page": pagination}

    def selection_claims(self, paper_id: str, body: dict):
        """Look up existing Claims fully inside exact text spans; never extract."""
        allowed = {"paper_revision", "segments", "offset", "limit", "snapshot_id"}
        if not isinstance(body, dict) or set(body) - allowed:
            raise ReadError("INVALID_SELECTION", "Unknown selection-lookup fields.")
        if body.get("offset", 0) and not body.get("snapshot_id"):
            raise ReadError("SNAPSHOT_REQUIRED", "Pass snapshot_id when continuing a selection lookup.")
        paper, data, revision = self.snapshot(paper_id, body.get("snapshot_id"))
        if body.get("paper_revision") != paper.revision:
            raise ReadError("REVISION_MISMATCH", "The selection must match the canonical paper revision.", 409)
        segments = body.get("segments")
        if not isinstance(segments, list) or not 1 <= len(segments) <= 32:
            raise ReadError("INVALID_SELECTION", "Supply 1..32 exact paragraph segments.")
        intervals = {}
        for segment in segments:
            if not isinstance(segment, dict) or set(segment) != {"element_id", "start", "end", "exact"}:
                raise ReadError("INVALID_SELECTION", "Each segment needs element_id, start, end, and exact.")
            eid, start, end = segment["element_id"], segment["start"], segment["end"]
            if not isinstance(eid, str):
                raise ReadError("INVALID_SELECTION", "element_id must be text.")
            element = paper.elements.get(eid)
            if (element is None or element.kind != "paragraph" or type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(element.text) or not isinstance(segment["exact"], str)
                    or element.text[start:end] != segment["exact"]):
                raise ReadError("SELECTION_MISMATCH", "Selection is not an exact span in the canonical paragraph text.", 409)
            intervals.setdefault(eid, []).append((start, end))
        merged = {}
        for eid, ranges in intervals.items():
            spans = []
            for start, end in sorted(ranges):
                if spans and start <= spans[-1][1]:
                    spans[-1] = (spans[-1][0], max(end, spans[-1][1]))
                else:
                    spans.append((start, end))
            merged[eid] = spans
        ids = []
        for cid, claim in data.claim_pool.items():
            loc = anchor(claim.source_anchor)
            selector = (loc or {}).get("selector") or {}
            if (claim.source_status == "located" and selector.get("type") == "text_span"
                    and any(a <= selector["start"] and selector["end"] <= b
                            for a, b in merged.get(claim.source_element_id, []))):
                ids.append(cid)
        ids, pagination = page(ids, body.get("offset", 0), body.get("limit", 50))
        return {**self.meta(paper, revision), "operation": "lookup_existing_selection_claims",
                "new_extraction": False, "selection": deepcopy(segments),
                "items": [self._claim_card(paper, data, cid, full=True) for cid in ids], "page": pagination,
                "coverage_note": "This lists saved Claims, not a guarantee that the selection was fully extracted."}

    def _mapping(self, data, cid, mapping_id=None):
        if cid not in data.claim_pool:
            raise ReadError("CLAIM_NOT_FOUND", "The requested Claim does not exist.", 404)
        mid = mapping_id or data.mapping_by_claim.get(cid)
        if mid is None:
            return None, {"usable": False, "errors": [], "warnings": []}
        record = data.mappings.get(mid)
        if not isinstance(record, dict) or record.get("claim_id") != cid:
            if mapping_id:
                raise ReadError("MAPPING_NOT_FOUND", "This Mapping is unavailable or does not belong to the requested Claim.", 404)
            return mid, {"usable": False, "errors": ["The current Mapping pointer is unavailable."], "warnings": []}
        # A failed/stale active result is never replaced with an older result silently.
        return mid, check_saved_mapping(data, cid, mid)

    def _evidence_card(self, paper, source, *, parts=None, claim_id=None, mapping_id=None, snapshot_id=None):
        content = source.source_content or {}
        text = source.source_quote or content.get("passage") or content.get("caption") or source.description
        if parts is not None:
            if "source_text" in parts:
                text = source.source_quote or content.get("passage") or ""
            elif "source_context" in parts:
                text = content.get("context") or ""
            elif "visual_observation" in parts:
                text = content.get("visual_observation") or ""
            elif "caption" in parts:
                text = content.get("caption") or ""
            else:
                text = content.get("label") or source.kind
        params = {k: v for k, v in {"claim_id": claim_id, "mapping_id": mapping_id,
                                    "snapshot_id": snapshot_id}.items() if v is not None}
        url = self.base(paper.paper_id) + "/evidence/" + quote(source.evidence_id, safe="")
        if params:
            url += "?" + urlencode(params)
        element = paper.elements.get(source.source_element_id)
        asset_id = source.source_asset_id or content.get("image_asset_id") or getattr(element, "image_asset_id", None)
        result = {"evidence_id": source.evidence_id, "kind": source.kind,
                  "source_element_id": source.source_element_id, "source_status": source.source_status,
                  "preview": preview(text), "detail_url": url, "image": self._asset_info(paper, asset_id),
                  "source_parts_used": list(parts) if parts is not None else None,
                  "pixel_input_recorded": bool(source.image_receipt),
                  "visual_reading_verified": content.get("visual_reading_verified") if content.get("visual_observation") else None}
        return result

    def claim(self, paper_id, claim_id, mapping_id=None, evidence_offset=0, evidence_limit=20, snapshot_id=None):
        if evidence_offset and not snapshot_id:
            raise ReadError("SNAPSHOT_REQUIRED", "Pass snapshot_id when continuing a Claim view.")
        paper, data, revision = self.snapshot(paper_id, snapshot_id)
        mid, check = self._mapping(data, claim_id, mapping_id)
        result = {**self.meta(paper, revision), "claim": self._claim_card(paper, data, claim_id, full=True),
                  "mapping": None, "mapping_status": "not_analyzed" if mid is None else "unavailable",
                  "evidence": {}, "display": None, "issues": list(check["errors"]), "warnings": list(check["warnings"])}
        if not check["usable"]:
            return result
        record = data.mappings[mid]
        row = presentation_analysis(data, claim_id, record)
        full = display_view(data, claim_id, mid)
        ids, pagination = page(full["evidence_ids"], evidence_offset, evidence_limit)
        selected = set(ids)
        # Only this Claim's adopted Evidence, not a pool/Link/task-wide union.
        filtered = pick(full, ("claim_id", "mapping_id", "status", "material_status", "claim_anchor",
                               "relation_review_required", "presentation_policy"))
        filtered["evidence_ids"] = ids
        filtered["evidence_highlights"] = [h for h in full["evidence_highlights"] if h["evidence_id"] in selected]
        filtered["nodes"] = [n for n in full["nodes"] if n["id"] == claim_id or n["id"] in selected]
        filtered["edges"] = [e for e in full["edges"] if e["from"] in selected]
        filtered["graph_complete"] = evidence_offset == 0 and not pagination["has_more"]
        result["mapping"] = {"mapping_id": mid, "claim_id": claim_id,
            "status": record["status"], "analysis_focus": record.get("analysis_focus"),
            "explanation": row["explanation"], "qualifications": list(row.get("qualifications") or []),
            "material_status": row.get("material_status", "limited"),
            "evidence_ids_used": list(full["evidence_ids"]),
            "historical_explanation_requires_review": row["historical_explanation_requires_review"],
            "presentation_review_notes": row["presentation_review_notes"]}
        parts = {e["from"]: e.get("source_parts_used") or [] for e in filtered["edges"]}
        result.update({"mapping_status": "available", "display": filtered, "evidence_page": pagination,
                       "evidence": {eid: self._evidence_card(paper, data.evidence_pool[eid], parts=parts[eid],
                            claim_id=claim_id, mapping_id=mid, snapshot_id=revision) for eid in ids}})
        result["next_evidence_url"] = (self.base(paper_id) + "/claims/" + quote(claim_id, safe="") + "?" + urlencode({
            "mapping_id": mid, "evidence_offset": pagination["next_offset"], "evidence_limit": evidence_limit,
            "snapshot_id": revision}) if pagination["has_more"] else None)
        return result

    def evidence_list(self, paper_id, offset=0, limit=50, element_id=None, kind=None, snapshot_id=None):
        if offset and not snapshot_id:
            raise ReadError("SNAPSHOT_REQUIRED", "Pass snapshot_id when continuing an Evidence list.")
        paper, data, revision = self.snapshot(paper_id, snapshot_id)
        if element_id is not None and element_id not in paper.elements:
            raise ReadError("ELEMENT_NOT_FOUND", "The source element filter is unknown.", 404)
        if kind is not None and kind not in {"paragraph", "figure", "table", "equation"}:
            raise ReadError("INVALID_KIND", "Unknown Evidence kind.")
        rows = [e for e in data.evidence_pool.values() if (kind is None or e.kind == kind)
                and (element_id is None or e.source_element_id == element_id)]
        rows, pagination = page(rows, offset, limit)
        return {**self.meta(paper, revision), "scope": "standalone_evidence_pool_not_support_relations",
                "items": [self._evidence_card(paper, row, snapshot_id=revision) for row in rows], "page": pagination}

    def _table_page(self, paper, cells, row_offset, row_limit):
        cells = [c for c in cells if isinstance(c, dict)]
        row_ids = sorted({c.get("row") for c in cells if type(c.get("row")) is int})
        rows, pagination = page(row_ids, row_offset, row_limit, max_limit=100)
        chosen = set(rows)
        def project(cell):
            item = pick(cell, CELL_FIELDS)
            if item.get("image_asset_id"):
                item["image"] = self._asset_info(paper, item["image_asset_id"])
            return item
        # Repeat original header records, never infer new header/dataset labels.
        headers = [project(c) for c in cells if c.get("role") == "column_header" and c.get("row") not in chosen]
        return {"cells": [project(c) for c in cells if c.get("row") in chosen],
                "header_context_cells": headers, "row_indices": rows, "page": pagination,
                "page_unit": "stored_rows", "context_is_not_additional_cited_evidence": True}

    def evidence(self, paper_id, evidence_id, claim_id=None, mapping_id=None,
                 row_offset=0, row_limit=20, snapshot_id=None):
        if row_offset and not snapshot_id:
            raise ReadError("SNAPSHOT_REQUIRED", "Pass snapshot_id when continuing Evidence table rows.")
        paper, data, revision = self.snapshot(paper_id, snapshot_id)
        source = data.evidence_pool.get(evidence_id)
        if source is None:
            raise ReadError("EVIDENCE_NOT_FOUND", "The requested Evidence does not exist.", 404)
        if mapping_id and not claim_id:
            raise ReadError("CLAIM_REQUIRED", "Provide claim_id when binding an Evidence read to a Mapping.")
        binding, parts = None, set(CONTENT_PARTS)
        if claim_id:
            mid, check = self._mapping(data, claim_id, mapping_id)
            if not check["usable"]:
                raise ReadError("MAPPING_UNAVAILABLE", "This Claim has no usable bound Mapping.", 409)
            row = presentation_analysis(data, claim_id, data.mappings[mid])
            edge = next((e for e in row["evidence_relations"] if e["evidence_id"] == evidence_id), None)
            if edge is None:
                raise ReadError("NOT_ADOPTED", "This Evidence was not used by this Claim's bound Mapping.", 404)
            parts = set(edge.get("source_parts_used") or [])
            binding = {"claim_id": claim_id, "mapping_id": mid,
                       "relation": edge.get("relation"), "relation_status": edge.get("relation_status"),
                       "source_parts_used": sorted(parts)}
        source_content = source.source_content or {}
        content = {}
        for name, value in (("source_text", source.source_quote or source_content.get("passage")),
                            ("source_context", source_content.get("context")),
                            ("caption", source_content.get("caption")),
                            ("visual_observation", source_content.get("visual_observation"))):
            if name in parts and value is not None:
                content[name] = value
        if "visual_observation" in content:
            content["visual_reading_verified"] = source_content.get("visual_reading_verified", False)
            content["model_reading_note"] = source.description
        if "equation" in parts and source.kind == "equation":
            content["equation"] = pick(source_content, ("latex", "mathml", "plain_text", "label"))
        if "table_cells" in parts and source.kind == "table":
            content["table"] = self._table_page(paper, source_content.get("cells") or [], row_offset, row_limit)
            content["table"].update(pick(source_content, ("row_group_context", "row_group_context_note", "representation", "recognition_status")))
            content["table"]["scope"] = "cells_stored_in_this_evidence_not_necessarily_the_entire_table"
        card = self._evidence_card(paper, source, parts=sorted(parts) if binding else None,
                                   claim_id=claim_id, mapping_id=binding["mapping_id"] if binding else None,
                                   snapshot_id=revision)
        result = {**self.meta(paper, revision), **card,
                  "source_anchor": anchor(source.source_anchor), "binding": binding, "content": content,
                  "usage_scope": "bound_mapping_components" if binding else "standalone_saved_evidence",
                  "is_new_analysis": False}
        if binding:
            result["source_parts_used"] = binding["source_parts_used"]
            result["usage_note"] = "Only recorded adopted components are returned. Unlisted components may still exist in the pool."
            result["standalone_detail_url"] = self.base(paper_id) + "/evidence/" + quote(evidence_id, safe="") + "?" + urlencode({"snapshot_id": revision})
            if not parts:
                result["usage_note"] = "Historical component usage is unknown. Use the standalone Evidence view to inspect stored material; do not infer support."
        else:
            result["extraction_note"] = source.description
            result["extraction_note_is_model_generated"] = True
        return result

    @staticmethod
    def _root_run(data, workflow_id=None):
        runs = data.runs
        if workflow_id:
            run = next((r for r in runs if r.get("run_id") == workflow_id), None)
            if run is None:
                raise ReadError("RESULT_NOT_RETAINED", "This operation is not retained in the current saved state.", 404)
            return run
        if not runs:
            return {}
        latest = max(enumerate(runs), key=lambda item: (str(item[1].get("finished_at") or item[1].get("started_at") or ""), item[0]))[1]
        parent_id = latest.get("parent_workflow_id") or latest.get("question_budget_id") or (latest.get("budget_report") or {}).get("question_budget_id")
        return next((r for r in reversed(runs) if r.get("stage") == "coordinator" and r.get("run_id") == parent_id), latest)

    def response(self, paper_id, workflow_id=None, snapshot_id=None):
        paper, data, revision = self.snapshot(paper_id, snapshot_id)
        run = self._root_run(data, workflow_id)
        states = run.get("task_states") or {}
        contract = (run.get("request_contract") or {}).get("request") or {}
        targets = (contract.get("target_claim_ids") or unique(cid for s in states.values() for cid in s.get("selected_claim_ids") or [])
                   or run.get("target_claim_ids") or run.get("claim_ids") or [])
        answer = run.get("answer")
        answer = pick(answer, ("answer", "used_source_ids", "limitations")) if isinstance(answer, dict) else None
        if run.get("status") in {"running", "queued"}:
            answer = None
        usage = run.get("token_usage_report") or run.get("api_cost_report") or {}
        budget = run.get("budget_report") or {}
        issues = []
        for issue in (run.get("issues") or [])[:20]:
            if isinstance(issue, dict):
                issues.append({**pick(issue, ("type", "stage", "claim_id")), "message": safe_note(issue.get("message") or issue.get("type"))})
            else:
                issues.append({"message": safe_note(issue)})
        return {**self.meta(paper, revision), "workflow_id": run.get("run_id"),
                "status": run.get("status", "ready"), "stage": run.get("stage"), "mode": run.get("mode"),
                "question": run.get("user_question") or run.get("source_question"), "target_claim_ids": targets,
                "answer_status": run.get("answer_status", "unavailable"), "answer": answer, "issues": issues,
                "more_issues": len(run.get("issues") or []) > 20,
                "unfinished_task_count": len(run.get("unfinished_tasks") or []),
                "token_usage": pick(usage, ("llm_calls_recorded", "input_tokens", "cached_input_tokens", "output_tokens", "total_tokens", "estimated_total_usd", "token_usage_completeness")),
                "budget": pick(budget, ("question_budget_id", "limit_usd", "operating_limit_usd", "committed_usd", "remaining_usd", "blocked", "estimate_only")),
                "note": "A saved result, not a new operation or chat-history listing. No model was invoked."}

    def asset(self, paper_id, asset_id):
        paper = self.paper(paper_id)
        identifier(asset_id)
        asset = paper.assets.get(asset_id)
        if asset is None or asset.status != "available":
            raise ReadError("ASSET_NOT_FOUND", "The requested image asset is not available.", 404)
        relative = asset.normalized_relative_path
        if not relative:
            raise ReadError("ASSET_NOT_LOCAL", "No standardized local image is registered for this asset.", 404)
        path = PurePosixPath(relative.replace("\\", "/"))
        if path.is_absolute() or any(p in {".", ".."} or ":" in p for p in path.parts):
            raise ReadError("INVALID_ASSET_PATH", "Unsafe asset path.", 409)
        root = self._path("standardized", paper_id, "paper.json").parent
        target = root.joinpath(*path.parts)
        checked = root
        for part in path.parts:
            checked = checked / part
            if checked.is_symlink() or (hasattr(checked, "is_junction") and checked.is_junction()):
                raise ReadError("LINKED_ASSET", "Linked image files are not served.", 409)
        if not target.resolve().is_relative_to(root.resolve()) or target.suffix.lower() not in IMAGE_TYPES:
            raise ReadError("UNSUPPORTED_ASSET", "Only registered local image assets are served.", 415)
        if not target.is_file():
            raise ReadError("ASSET_NOT_FOUND", "The registered local image file is missing.", 404)
        if target.stat().st_size > 32 * 1024 * 1024:
            raise ReadError("ASSET_TOO_LARGE", "The local image size limit was exceeded.", 413)
        return target, IMAGE_TYPES[target.suffix.lower()]
