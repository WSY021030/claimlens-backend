"""Local Evidence retrieval and bounded Mapping inputs; no Links or API calls.

This module NEVER edits Claim/Evidence records. It indexes all eligible saved
records locally, keeps source contents intact, and reports what a bounded packet
could not include. Retrieval completion is not proof of scientific completeness.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

POLICY = "evidence_packet_no_links_v4_origin_guard"
WORD = re.compile(r"[^\W_]+(?:[.-][^\W_]+)*", re.UNICODE)
STOP = frozenset("a an the and or of to for in on at by from with is are was were be been being it its this that these those as we our they their paper study result results model models data dataset datasets".split())
CONTRAST = re.compile(r"\b(?:however|although|except\w*|limitat\w*|fail\w*|worse|misinterpret\w*|inconsisten\w*|unlike|whereas|uncertain\w*|not|never|no)\b|但|例外|局限|未能|不能", re.I)
ANAPHORA = re.compile(r"\b(?:same direction|this effect|these results|such|respectively|former|latter)\b", re.I)


def encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def byte_size(value: Any) -> int:
    return len(encoded(value).encode("utf-8"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(encoded(value).encode("utf-8")).hexdigest()


def _limit(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer byte limit.") from exc
    if not 4000 <= value <= 60000:
        raise ValueError(f"{name} must be between 4000 and 60000 bytes.")
    return value


@dataclass(frozen=True)
class RetrievalLimits:
    packet_bytes: int = 18000
    expanded_packet_bytes: int = 30000

    @classmethod
    def from_environment(cls):
        normal = _limit("READER_V3_MAPPING_PACKET_BYTES", 18000)
        expanded = _limit("READER_V3_MAPPING_EXPANDED_BYTES", 30000)
        if expanded < normal:
            raise ValueError("Expanded Mapping limit must not be below the initial limit.")
        return cls(normal, expanded)


def terms(text: str) -> set[str]:
    found = set()
    for match in WORD.finditer(str(text)):
        word = match.group().casefold()
        # Keep conditions, negations, model versions and numbers distinguishable.
        variants = [word, *re.split(r"[-.]", word)]
        for term in variants:
            if term in STOP or not term:
                continue
            found.add(term)
            if len(term) > 5 and term.endswith("s") and not term.endswith("ss"):
                found.add(term[:-1])
    return found


def _text(record) -> str:
    source = record.source_content or {}
    parts = [record.source_quote or "", record.description or ""]
    for field in ("passage", "context", "caption", "visual_observation", "latex", "plain_text"):
        if isinstance(source.get(field), str):
            parts.append(source[field])
    parts.extend(str(c.get("text") or "") for c in source.get("cells", []) if isinstance(c, dict))
    return " ".join(parts)


def _body(record) -> str:
    source = record.source_content or {}
    return str(record.source_quote or source.get("passage") or "")


def _span(record):
    selector = getattr(record.source_anchor, "selector", None)
    if getattr(selector, "type", None) == "text_span":
        return selector.start, selector.end
    return None


def _contained(record, retained) -> bool:
    """Drop ONLY exact contained source text, never a fuzzy overlap threshold."""
    if record.kind != "paragraph" or not _body(record):
        return False
    for previous in retained:
        if previous.kind != record.kind or previous.source_element_id != record.source_element_id:
            continue
        a, b = _span(record), _span(previous)
        if a and b and b[0] <= a[0] and a[1] <= b[1] and _body(record) in _body(previous):
            return True
    return False


def _source_text_key(value) -> str:
    """Normalize whitespace only; do not discard negations, numbers or symbols."""
    return " ".join(str(value or "").split())


def same_claim_source(claim, record) -> bool:
    """Identify an exact, located source repetition for THIS target Claim.

    This is a provenance comparison, not semantic similarity or a scientific
    judgment. Longer passages, other modalities, and other Claims are unaffected.
    """
    if (getattr(record, "kind", None) != "paragraph"
            or getattr(claim, "source_status", None) != "located"
            or getattr(record, "source_status", None) != "located"
            or not getattr(claim, "source_element_id", None)
            or claim.source_element_id != record.source_element_id):
        return False
    claim_span, record_span = _span(claim), _span(record)
    if claim_span is None or claim_span != record_span:
        return False
    claim_text = _source_text_key(getattr(claim, "source_quote", None))
    return bool(claim_text and claim_text == _source_text_key(_body(record)))


def claim_origin_only(claim, record, *, source_parts=None, visible_view=None) -> bool:
    """True only if all actually used components are the target's exact sentence.

    A saved neighboring context is NOT implicitly treated as a used component.
    Conversely, an explicitly supplied/used longer context prevents this exact
    duplicate rule from discarding potentially useful additional information.
    """
    if not same_claim_source(claim, record):
        return False
    parts = list(source_parts) if source_parts is not None else ["source_text"]
    if not parts or not set(parts) <= {"source_text", "source_context"}:
        return False
    view = visible_view if visible_view is not None else evidence_view(
        record, include_context="source_context" in parts)
    target = _source_text_key(claim.source_quote)
    return all(isinstance(view.get(part), str)
               and _source_text_key(view[part]) == target for part in parts)


def available_source_parts(view: dict) -> list[str]:
    """Name actual source components in a model view, not evidential strength.

    Model-generated extraction notes explain reading limitations. They are not
    a separately citable source component or independent experimental evidence.
    """
    parts = []
    for name in ("source_text", "source_context", "caption"):
        value = view.get(name)
        if isinstance(value, str) and value.strip():
            parts.append(name)
    observation = view.get("visual_observation")
    if (view.get("pixels_were_submitted")
            and isinstance(observation, str) and observation.strip()):
        parts.append("visual_observation")
    if isinstance(view.get("cells"), list) and view["cells"]:
        parts.append("table_cells")
    if isinstance(view.get("equation"), dict) and any(view["equation"].values()):
        parts.append("equation")
    return parts


def evidence_view(record, *, include_context=False) -> dict:
    """Lossless material view; no arbitrary text/cell truncation.

    Long records are deferred at packet assembly, not silently cut. The stored
    anchor remains authoritative for UI highlighting. Compact cells preserve
    every stored cell ID and value; they do not invent missing comparison rows.
    """
    source = record.source_content or {}
    anchor = record.source_anchor.model_dump(mode="json") if record.source_anchor else None
    view = {
        "evidence_id": record.evidence_id,
        "kind": record.kind,
        "source_element_id": record.source_element_id,
        "source_group": record.source_element_id,
        "source_status": record.source_status,
        "observed_via": record.observed_via,
        "source_note": record.source_note,
    }
    if anchor:
        selector = dict(anchor["selector"])
        # The quotation is already included below, not repeated in its locator.
        for key in ("exact", "prefix", "suffix"):
            selector.pop(key, None)
        view["source_anchor"] = {"element_id": anchor["element_id"], "selector": selector}
    if record.kind == "paragraph":
        view["source_text"] = _body(record)
        view["quotation_confirmed"] = bool(record.source_quote)
        view["basis"] = "paper_text_not_independent_confirmation"
        context = source.get("context")
        if include_context and context and context != view["source_text"]:
            view["source_context"] = context
    elif record.kind in {"figure", "table"}:
        view.update({"label": source.get("label"), "caption": source.get("caption", "")})
        if source.get("alt_text"):
            view["alt_text"] = source["alt_text"]
        observation = source.get("visual_observation")
        view["pixels_were_submitted"] = bool(record.image_receipt)
        view["has_visual_observation"] = bool(observation)
        if (record.image_receipt or record.observed_via == "image_input") and record.description:
            # Preserve existing warnings such as unreadable axes or cropped labels.
            # This is an extraction-model note, NOT a new source observation.
            view["image_extraction_note"] = record.description
        if observation:
            view["visual_observation"] = observation
            view["visual_reading_verified"] = bool(source.get("visual_reading_verified", False))
            view["basis"] = "model_reading_of_image_not_independent_verification"
        else:
            view["basis"] = "caption_only" if record.kind == "figure" else "stored_table_cells"
        if record.image_receipt:
            view["image_identity"] = {k: record.image_receipt.get(k) for k in (
                "asset_id", "source_sha256", "sent_sha256", "width", "height", "conversion")}
        if record.kind == "table":
            fields = ("cell_id", "row", "col", "text", "role", "rowspan", "colspan")
            cells = [cell for cell in source.get("cells", []) if isinstance(cell, dict)]
            view["cell_fields"] = list(fields)
            view["cells"] = [[cell.get(k, 1 if k in {"rowspan", "colspan"} else None)
                              for k in fields] for cell in cells]
            view["representation"] = source.get("representation")
            view["recognition_status"] = source.get("recognition_status")
            view["row_group_context"] = source.get("row_group_context", [])
            view["table_scope"] = "all_cells_stored_in_this_evidence_not_necessarily_the_whole_table"
    elif record.kind == "equation":
        view["equation"] = {k: source[k] for k in ("latex", "mathml", "plain_text", "label") if source.get(k)}
    view["available_source_parts"] = available_source_parts(view)
    return view


def scope_element_ids(paper, data, task):
    """Match workflow evidence boundaries using real sections, never ID prefixes."""
    scope = task.scope
    if scope.mode == "whole_paper" or (task.evidence_scope == "whole_paper" and not scope.user_locked):
        return None
    elements = set(scope.element_ids)
    sections = set(scope.section_ids)
    if paper is None:
        if sections:
            raise ValueError("A standardized paper is required to honor a section-scoped Evidence request.")
        return elements
    while True:
        expanded = sections | {s.section_id for s in paper.sections if s.parent_section_id in sections}
        if expanded == sections:
            break
        sections = expanded
    elements.update(eid for eid, item in paper.elements.items() if item.section_id in sections)
    if not scope.user_locked:
        origins = {data.claim_pool[cid].source_element_id for cid in scope.claim_ids if cid in data.claim_pool}
        elements.update(m.target_id for m in paper.mentions
                        if m.source_anchor.element_id in origins and m.target_id in paper.elements)
    return elements


class EvidenceDataManager:
    """One local index per analysis snapshot; no persisted Claim-Evidence Links."""

    def __init__(self, data, paper=None, limits=None):
        if paper is not None and (paper.paper_id != data.paper_id or paper.revision != data.paper_revision):
            raise ValueError("Paper and analysis revisions must match for Evidence retrieval.")
        self.data, self.paper = data, paper
        self.limits = limits or RetrievalLimits.from_environment()
        self.records = {eid: e for eid, e in data.evidence_pool.items()
                        if e.source_status == "located" and e.source_anchor is not None
                        and (e.kind != "paragraph" or bool(_body(e)))}
        self.words = {eid: terms(_text(e)) for eid, e in self.records.items()}
        self.frequency = Counter(word for words in self.words.values() for word in words)
        self.by_element = defaultdict(list)
        for eid, record in self.records.items():
            self.by_element[record.source_element_id].append(eid)
        self.sections = {eid: e.section_id for eid, e in paper.elements.items()} if paper else {}
        self.references = defaultdict(set)
        if paper:
            for mention in paper.mentions:
                if mention.target_id in paper.elements:
                    self.references[mention.source_anchor.element_id].add(mention.target_id)

    def _rank(self, claim_id, allowed_element_ids=None):
        claim = self.data.claim_pool[claim_id]
        query = claim.source_quote or claim.model_quote
        wanted = terms(query)
        # Anaphoric Claims need source context to identify the comparison.
        if ANAPHORA.search(query):
            for eid in self.by_element.get(claim.source_element_id, []):
                context = (self.records[eid].source_content or {}).get("context", "")
                if query in context:
                    wanted.update(terms(context))
                    break
        targets = self.references.get(claim.source_element_id, set())
        n = max(1, len(self.records))
        ranked = []
        for eid, record in self.records.items():
            origin = record.source_element_id
            if allowed_element_ids is not None and origin not in allowed_element_ids:
                continue
            common = wanted & self.words[eid]
            same_origin = origin == claim.source_element_id
            direct_ref = origin in targets
            same_section = bool(self.sections.get(origin) and
                                self.sections.get(origin) == self.sections.get(claim.source_element_id))
            if not common and not same_origin and not direct_ref:
                continue
            lexical = sum(1.0 + math.log((n + 1) / (1 + self.frequency[t])) for t in common)
            score = lexical / max(1.0, math.sqrt(len(self.words[eid]) / 50))
            score += 8 * same_origin + 7 * direct_ref + 3 * same_section
            contrast = bool(CONTRAST.search(_text(record)) and (len(common) >= 2 or same_section))
            if contrast:
                score += 2.0
            observation = bool((record.source_content or {}).get("visual_observation"))
            if observation and (direct_ref or same_section):
                score += 2.0
            ranked.append({"id": eid, "score": score, "same_origin": same_origin,
                           "direct_reference": direct_ref, "same_section": same_section,
                           "contrast_candidate": contrast, "visual_observation": observation})
        ranked.sort(key=lambda row: (-row["score"], row["id"]))
        return ranked

    def packet(self, claim_id, *, expanded=False, allowed_element_ids=None,
               analysis_focus="general", context_claim_ids=(), required_evidence_ids=()):
        if claim_id not in self.data.claim_pool:
            raise ValueError(f"Unknown Claim ID: {claim_id}")
        claim = self.data.claim_pool[claim_id]
        ranked = self._rank(claim_id, allowed_element_ids)
        include_context = bool(ANAPHORA.search(claim.source_quote or ""))
        origin_only_ids = []
        for row in ranked:
            record = self.records[row["id"]]
            if not same_claim_source(claim, record):
                continue
            view = evidence_view(record, include_context=include_context)
            if claim_origin_only(claim, record,
                                 source_parts=view["available_source_parts"],
                                 visible_view=view):
                origin_only_ids.append(row["id"])
        # The Claim already supplies this sentence and its locator. Do not send
        # it again as a candidate argument. Records remain intact in the pool.
        origin_only_set = set(origin_only_ids)
        ranked = [row for row in ranked if row["id"] not in origin_only_set]
        peak = ranked[0]["score"] if ranked else 0
        # A relevance floor is not an absence claim; all excluded IDs stay in
        # the local receipt. Expansion lowers this floor without sending an index.
        fraction = 0.16 if expanded else 0.32
        eligible = [r for r in ranked if r["score"] >= peak * fraction or
                    r["direct_reference"] or r["same_origin"] or
                    (r["same_section"] and r["contrast_candidate"])]
        required = set(required_evidence_ids) - origin_only_set
        for row in ranked:
            if row["id"] in required and row not in eligible:
                eligible.append(row)
        protected = set(required)
        if eligible:
            protected.add(eligible[0]["id"])
        for row in eligible:
            if row["direct_reference"]:
                protected.add(row["id"])
        # Keep the best same-condition qualification and a real observation
        # ahead of multiple restatements. These are NOT scientific judgments.
        for predicate in (
            lambda r: r["contrast_candidate"] and (r["same_section"] or r["score"] >= peak * .35),
            lambda r: r["visual_observation"] and (r["same_section"] or r["direct_reference"]),
            lambda r: self.records[r["id"]].kind == "table" and r["score"] >= peak * .35,
        ):
            match = next((r for r in eligible if predicate(r)), None)
            if match:
                protected.add(match["id"])
        # Once a figure/table is a candidate, expose its available companion
        # modality records before deciding which record is actually useful.
        companion_sources = {self.records[r["id"]].source_element_id for r in eligible
                             if r["id"] in protected and self.records[r["id"]].kind in {"figure", "table"}}
        for r in ranked:
            if self.records[r["id"]].source_element_id in companion_sources:
                if r not in eligible:
                    eligible.append(r)
                if r["visual_observation"]:
                    protected.add(r["id"])
        eligible.sort(key=lambda r: (r["id"] not in required, r["id"] not in protected,
                                     -r["score"], r["id"]))
        cap = self.limits.expanded_packet_bytes if expanded else self.limits.packet_bytes
        claim_anchor = claim.source_anchor.model_dump(mode="json") if claim.source_anchor else None
        if claim_anchor:
            # The target quotation already appears in text; retain only its locator.
            for field in ("exact", "prefix", "suffix"):
                claim_anchor.get("selector", {}).pop(field, None)
        payload = {
            "paper_id": self.data.paper_id, "paper_revision": self.data.paper_revision,
            "retrieval_policy": POLICY, "analysis_focus": analysis_focus,
            "analysis_goals": (analysis_focus.split(":", 1)[1].split("+")
                               if analysis_focus.startswith("integrated:") else [analysis_focus]),
            "target_claim_ids": [claim_id],
            "target_claims": [{"claim_id": claim_id, "type": claim.type, "scope": claim.scope,
                               "text": claim.source_quote or claim.model_quote,
                               "source_element_id": claim.source_element_id,
                               "source_status": claim.source_status,
                               "source_anchor": claim_anchor}],
            "context_claims": [], "evidence_items": [],
            "source_contract": (
                "Use only supplied source records. Retrieval is bounded, not exhaustive. "
                "A source_group identifies shared provenance, not independent replication. "
                "Cell arrays follow cell_fields and preserve stored values. A caption is "
                "not a pixel reading. Missing material here is not absence from the paper. "
                "Use no candidate scores as support judgments. Do not invent Links. "
                "available_source_parts names present content, not verification. "
                "image_extraction_note is a model-generated reading note; preserve its "
                "limitations but never elevate it above source text or original cells. "
                "The target Claim and its anchor establish what the author said, not "
                "why it is justified. Exact origin-only duplicates are omitted as "
                "arguments; their absence here does not mean the paper lacks evidence. "
                "A longer excerpt is useful only for its additional premises."
            ),
        }
        # Explicit small peer context is optional; never add all paper Claims.
        for cid in dict.fromkeys(context_claim_ids):
            if cid == claim_id or cid not in self.data.claim_pool:
                continue
            c = self.data.claim_pool[cid]
            item = {"claim_id": cid, "text": c.source_quote or c.model_quote}
            if byte_size({**payload, "context_claims": [*payload["context_claims"], item]}) > cap // 4:
                break
            payload["context_claims"].append(item)
        selected, deferred, duplicates = [], [], []
        for row in eligible:
            record = self.records[row["id"]]
            if row["id"] not in required and _contained(record, selected):
                duplicates.append(row["id"])
                continue
            view = evidence_view(record, include_context=include_context)
            if same_claim_source(claim, record):
                view["source_text_role_for_target"] = "claim_origin_not_additional_support"
            tentative = {**payload, "evidence_items": [*payload["evidence_items"], view]}
            # Leave room for bounded coverage metadata; never truncate an item.
            if byte_size(tentative) > cap - 1200:
                deferred.append(row["id"])
                continue
            selected.append(record)
            payload["evidence_items"].append(view)
        missing_refs = sorted(target for target in self.references.get(claim.source_element_id, set())
                              if not self.by_element.get(target) and
                              (allowed_element_ids is None or target in allowed_element_ids))
        protected_deferred = sorted(protected & set(deferred))
        payload["retrieval_coverage"] = {
            "bounded_view": True, "exhaustive_recall_claimed": False,
            "structure_available": self.paper is not None,
            "candidate_records_not_included": len(deferred),
            "local_candidates_below_relevance_floor": len([r for r in ranked if r not in eligible]),
            "priority_records_not_included": len(protected_deferred),
            "direct_reference_sources_without_saved_evidence": missing_refs[:12],
            "additional_missing_reference_count": max(0, len(missing_refs) - 12),
            "expanded_view": expanded,
            "exact_claim_origin_records_omitted": len(origin_only_ids),
        }
        if byte_size(payload) > cap:
            raise ValueError("Claim and required packet metadata exceed the configured Mapping byte limit.")
        receipt = {
            "policy": POLICY, "claim_id": claim_id, "expanded": expanded,
            "packet_bytes": byte_size(payload), "packet_byte_limit": cap,
            "pool_records_considered_locally": len(self.records),
            "eligible_evidence_ids": [r["id"] for r in eligible],
            "selected_evidence_ids": [r.evidence_id for r in selected],
            "deferred_for_size": deferred, "priority_deferred": protected_deferred,
            "exact_containment_duplicates": duplicates,
            "claim_origin_only_evidence_ids": origin_only_ids,
            "below_relevance_floor": [r["id"] for r in ranked if r not in eligible],
            "missing_direct_source_ids": missing_refs,
            "scope_element_ids": sorted(allowed_element_ids) if allowed_element_ids is not None else None,
            "complete_recall_guaranteed": False,
        }
        return payload, receipt

    def evidence_ids(self, claim_ids, *, allowed_element_ids=None):
        result = []
        for cid in dict.fromkeys(claim_ids):
            packet, _ = self.packet(cid, allowed_element_ids=allowed_element_ids)
            result.extend(item["evidence_id"] for item in packet["evidence_items"])
        return list(dict.fromkeys(result))
