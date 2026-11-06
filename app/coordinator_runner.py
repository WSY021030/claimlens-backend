"""Task-scoped material selection, Mapping, and ONE final response.

Source collection is owned by workflow_runner. This module never reads the
paper to manufacture Evidence, never concatenates independent batch answers,
and never stores a source-only diagnostic as a successful final answer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from openai import APIConnectionError, APIStatusError, APITimeoutError
from app.agents.coordinator_agent import CoordinatorAgent
from app.agents.mapping_agent import MappingAgent
from app.budget import BudgetStop, PaperBudget, QUESTION_BUDGET_ENV, QUESTION_LIMIT_ENV
from app.core.analysis_store import AnalysisStore
from app.core.token_report import task_token_report, print_task_tokens
from app.core.table_context import (
    audits_for_packet, compact_audits_for_model, render_verified_numeric_checks,
)
from app.core.evidence_retrieval import EvidenceDataManager, scope_element_ids
from app.core.mapping_execution import run_claim_mappings
from app.core.request_contract import (
    get_request, locked_targets, saved_only, evidence_boundary, enforce_plan_targets,
    question_context_note,
)
from app.core.mapping_views import (
    bind_saved_task, display_view, mapping_bindings, argument_packet, canonical_references,
    presentation_analysis,
)
from app.core.coordinator_tools import check_answer_references, render_source_fallback
from app.llm import StructuredOutputParseError
from app.core.mapping_records import (
    completed_mapping_evidence,
)
from app.models.coordinator_v3 import CoordinatorPlan, required_capabilities
from app.models.reader_v3 import AnalysisDocument
from app.usage_cost import CostTracker

POLICY_VERSION = "unified_question_v1"
MODEL_ERRORS = (BudgetStop, StructuredOutputParseError, APIConnectionError,
                APIStatusError, APITimeoutError)
SOURCE_PAGE_BYTES = 50000
FULL_VIEW_BYTES = 150000


def now():
    return datetime.now(timezone.utc).isoformat()


def unique(values):
    return list(dict.fromkeys(v for v in values if v))


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def signature(value):
    return hashlib.sha256(encoded(value).encode("utf-8")).hexdigest()


def size(value):
    return len(encoded(value).encode("utf-8"))


def find_saved_run(data, run_id):
    for item in reversed(data.runs):
        if item.get("run_id") == run_id:
            return item
    raise ValueError(f"Saved workflow not found: {run_id}")


def error_message(exc):
    if isinstance(exc, (BudgetStop, StructuredOutputParseError)):
        return f"{type(exc).__name__}: {exc}"
    # Internal validation failures have no HTTP code. Preserve a bounded
    # diagnostic rather than hiding the only actionable information.
    if isinstance(exc, (ValueError, RuntimeError)):
        return f"{type(exc).__name__}: {str(exc)[:400]}"
    return f"{type(exc).__name__}; HTTP={getattr(exc, 'status_code', None)}"


def claim_view(claim):
    return {
        "claim_id": claim.claim_id, "type": claim.type, "scope": claim.scope,
        "source_element_id": claim.source_element_id,
        "source_quote": claim.source_quote, "source_status": claim.source_status,
        "source_anchor": claim.source_anchor.model_dump(mode="json") if claim.source_anchor else None,
        "unlocated_model_quote": claim.model_quote if claim.source_status == "unlocated" else None,
    }


def record_pages(records, byte_limit=SOURCE_PAGE_BYTES):
    """Partition all candidates, without a count cutoff or dropping a record."""
    pages, current = [], []
    for record in records:
        if current and size(current + [record]) > byte_limit:
            pages.append(current)
            current = []
        current.append(record)
    if current:
        pages.append(current)
    return pages


def select_task_claims(*, coordinator, data, run, task, candidate_ids, store):
    candidates = [claim_view(data.claim_pool[cid]) for cid in candidate_ids]
    state = run.setdefault("task_states", {}).setdefault(task.task_id, {})
    # Explicit cited Claim IDs require no paid semantic re-selection. This is
    # NOT applied to open-ended questions or Planner-added unrelated IDs.
    requested = set(re.findall(r"\bclaim_\d{6}\b", run["user_question"], flags=re.IGNORECASE))
    anchored = set(task.scope.claim_ids)
    if requested and anchored and anchored <= requested and anchored <= set(candidate_ids):
        chosen = [cid for cid in candidate_ids if cid in anchored]
        state["selected_claim_ids"] = chosen
        state["claim_selection"] = {
            "selection_mode": "explicit_grounded_claim_ids",
            "candidate_claim_ids": list(candidate_ids),
            "selected_claim_ids": chosen,
            "not_selected_claim_ids": [cid for cid in candidate_ids if cid not in chosen],
        }
        store.save(data)
        return chosen
    # Do not let an earlier selection act as a whitelist for later extraction.
    task_view = task.model_dump(mode="json")
    task_view["scope"]["claim_ids"] = []
    key = signature({"policy": POLICY_VERSION, "question": run["user_question"],
                     "task": task_view, "candidates": candidates})
    old = state.get("claim_selection") or {}
    if old.get("signature") == key:
        return list(old["selected_claim_ids"])
    pages = record_pages(candidates)
    selected, decisions = [], []
    for index, page in enumerate(pages, 1):
        print(f"Selecting task Claims: {task.task_id}, page {index}/{len(pages)}", flush=True)
        result = coordinator.select_claims(question=run["user_question"], task=task,
                    candidates=page, page=index, page_count=len(pages))
        allowed = {item["claim_id"] for item in page}
        unknown = set(result.selected_claim_ids) - allowed
        if unknown:
            raise ValueError(f"Claim selection returned unknown IDs: {sorted(unknown)}")
        selected.extend(result.selected_claim_ids)
        decisions.append({"candidate_claim_ids": sorted(allowed),
                          "selected_claim_ids": result.selected_claim_ids,
                          "reason": result.selection_reason})
    selected = unique(selected)
    state["claim_selection"] = {"signature": key, "candidate_claim_ids": list(candidate_ids),
        "selected_claim_ids": selected, "not_selected_claim_ids": [cid for cid in candidate_ids if cid not in selected],
        "page_decisions": decisions}
    state["selected_claim_ids"] = selected
    store.save(data)
    return selected


def compact_table_from_pool(data, element_id, evidence_ids=None):
    """One source-grounded table view shared by all Evidence from that table.

    The saved Evidence remains unchanged. Merge identical cell IDs once and
    expose conflicting originals instead of silently preferring one value.
    """
    by_id, conflicts, groups = {}, [], {}
    metadata = {}
    selected = set(evidence_ids) if evidence_ids is not None else None
    for record in data.evidence_pool.values():
        if selected is not None and record.evidence_id not in selected:
            continue
        if (record.kind != "table" or record.source_element_id != element_id
                or record.source_status != "located"):
            continue
        source = record.source_content or {}
        if not isinstance(source, dict):
            continue
        for field in ("label", "caption", "representation", "recognition_status"):
            if field not in metadata and source.get(field) is not None:
                metadata[field] = source[field]
        for row_group in source.get("row_group_context") or []:
            if not isinstance(row_group, dict):
                continue
            name, first, last = (row_group.get("dataset_label"),
                                 row_group.get("row_start"), row_group.get("row_end"))
            if name and isinstance(first, int) and isinstance(last, int):
                groups[(first, last, str(name))] = {
                    "dataset_label": name, "row_start": first, "row_end": last,
                }
        for cell in source.get("cells") or []:
            if not isinstance(cell, dict) or not cell.get("cell_id"):
                continue
            compact = {field: cell[field] for field in (
                "cell_id", "row", "col", "text") if field in cell}
            for field in ("role", "rowspan", "colspan", "image_asset_id"):
                value = cell.get(field)
                if value is not None and value not in ("data", 1):
                    compact[field] = value
            existing = by_id.get(cell["cell_id"])
            if existing is None:
                by_id[cell["cell_id"]] = compact
            elif existing != compact and not any(
                    conflict["cell_id"] == cell["cell_id"] for conflict in conflicts):
                conflicts.append({"cell_id": cell["cell_id"],
                                  "first": existing, "conflicting": compact})
    if not by_id:
        return None
    ordered = sorted(by_id.values(), key=lambda c: (
        c.get("row", -1), c.get("col", -1), c["cell_id"]))
    table = {
        **metadata,
        "source_element_id": element_id,
        "row_groups": [groups[k] for k in sorted(groups)],
        "cell_conflicts": conflicts,
        "source_view_note": (
            "Values are exact original located cell text. Null means a cell was "
            "not observed, not that its source is empty. Evidence source anchors "
            "identify the relevant cells. All original records remain on disk."
        ),
    }
    # Safe only if row/column/ID values exactly encode the original cells.
    # Irregular cell IDs, merged cells, images and nonstandard headers stay
    # expanded; do not silently discard data to force a compact grid.
    regular = not conflicts and all(
        isinstance(c.get("row"), int) and isinstance(c.get("col"), int)
        and c["row"] >= 0 and c["col"] >= 0
        and c["cell_id"] == f"{element_id}.R{c['row']}C{c['col']}"
        and c.get("rowspan", 1) == 1 and c.get("colspan", 1) == 1
        and c.get("image_asset_id") is None
        and c.get("role") in (None, "data", "column_header")
        and (c.get("role") != "column_header" or c["row"] == 0)
        and isinstance(c.get("text"), str)
        for c in ordered
    )
    if regular:
        values_by_row = {}
        for cell in ordered:
            values_by_row.setdefault(cell["row"], {})[cell["col"]] = cell["text"]
        width = max(cell["col"] for cell in ordered) + 1
        table["grid_cells"] = {
            "cell_id_rule": element_id + ".R{row}C{col}",
            "rows": [[row, [cols.get(col) for col in range(width)]]
                     for row, cols in sorted(values_by_row.items())],
            "column_count": width,
            "format": "[row_index, [column_0_text, ...]]. Null is absent. "
                "Substitute row and col into cell_id_rule for provenance. "
                "Original strings, including decimal precision, are unchanged.",
        }
    else:
        table["cells"] = ordered
    return table


def source_packet(data, *, claim_ids=(), evidence_ids=(), mapping_ids=(),
                  paper=None, question=""):
    """Source-grounded model view, with each table cell transmitted once.

    Original Evidence records are retained unchanged in analysis.json. Their
    source anchors and IDs are preserved for citation and integrity review.
    """
    cids, eids, mids = unique(claim_ids), unique(evidence_ids), unique(mapping_ids)
    selected_mappings = {}
    for mid in mids:
        record = data.mappings.get(mid)
        if not isinstance(record, dict):
            # Preserve explicit historical references from older workflow runs.
            record = data.mapping_batch_archive.get(mid)
        if (not isinstance(record, dict) or record.get("status") != "completed"
                or record.get("paper_revision") != data.paper_revision):
            continue
        selected_mappings[mid] = {key: deepcopy(record.get(key)) for key in (
            "status", "target_claim_ids", "claim_analyses", "cross_claim_explanation",
            "missing_claim_ids", "unresolved_evidence_ids", "analysis_focus")}
        cids.extend(record.get("target_claim_ids") or [])
        for analysis in record.get("claim_analyses") or []:
            eids.extend(analysis.get("evidence_ids_used") or [])
    cids = unique(cids)
    eids = unique(eids)
    claims = {cid: claim_view(data.claim_pool[cid]) for cid in cids if cid in data.claim_pool}
    evidence, shared, index = {}, {}, {}
    table_views = {}
    for cid, record in claims.items():
        index[cid] = {"kind": "claim", "source_element_id": record["source_element_id"],
                      "source_status": record["source_status"]}
    for eid in eids:
        record = data.evidence_pool.get(eid)
        if record is None:
            continue
        item = record.model_dump(mode="json")
        content = item.pop("source_content")
        if record.kind == "table":
            source_id = record.source_element_id
            if source_id not in table_views:
                table_views[source_id] = compact_table_from_pool(data, source_id, evidence_ids=eids)
            if table_views[source_id] is not None:
                # One full, source-grounded table across all selected Evidence.
                # An Evidence's own source_anchor still identifies its cells.
                content = table_views[source_id]
                item.pop("description", None)  # Repeated prose, not source data.
                # The source anchor remains fully recoverable from row/col
                # numbers when it follows the verified regular cell-ID rule.
                anchor = item.get("source_anchor") or {}
                selector = anchor.get("selector") or {}
                ids = selector.get("cell_ids") or []
                source_rule = source_id + ".R{row}C{col}"
                if content.get("grid_cells", {}).get("cell_id_rule") == source_rule and ids:
                    coordinates = []
                    for cell_id in ids:
                        # The plain prefix is checked below as well: no extra
                        # characters or alternate source IDs are ever accepted.
                        if not cell_id.startswith(source_id + ".R"):
                            coordinates = []
                            break
                        tail = cell_id[len(source_id) + 2:]
                        match = re.fullmatch(r"(\d+)C(\d+)", tail)
                        if not match:
                            coordinates = []
                            break
                        coordinates.append([int(match.group(1)), int(match.group(2))])
                    if len(coordinates) == len(ids):
                        item["source_anchor"] = {"element_id": source_id,
                            "selector": {"type": "table_coordinates", "positions": coordinates,
                                         "cell_id_rule": source_rule}}
                for field in ("source_asset_id", "image_receipt", "source_quote"):
                    if item.get(field) is None:
                        item.pop(field, None)
                if item.get("source_note") == "table_cells_located":
                    item.pop("source_note", None)
        ref = "content_" + signature(content)
        shared.setdefault(ref, content)
        item["content_ref"] = ref
        # Administrative IDs are not source contents. The original stays on disk.
        item.pop("created_run_id", None)
        evidence[eid] = item
        index[eid] = {"kind": "evidence", "source_element_id": record.source_element_id,
                      "source_status": record.source_status, "observed_via": record.observed_via}
    relations = {
        rid: deepcopy(rel) for rid, rel in data.mapping_relations.items()
        if (set(rel.get("claim_ids") or []) <= set(cids)
            and set(rel.get("mapping_ids") or []) <= set(selected_mappings)
            and rel.get("paper_revision") == data.paper_revision)
    }
    for mid in selected_mappings:
        index[mid] = {"kind": "mapping", "source_status": "analysis"}
    for rid in relations:
        index[rid] = {"kind": "mapping_relation", "source_status": "analysis"}
    full_checks = audits_for_packet(paper, data, question, evidence) if paper is not None else []
    # The full audit is derived from the original cells; do not send the same
    # per-row numbers and Evidence IDs twice to the model.
    checks = compact_audits_for_model(full_checks)
    return {"paper_id": data.paper_id, "paper_revision": data.paper_revision,
            "claims": claims, "evidence": evidence,
            "mappings": selected_mappings, "mapping_relations": relations,
            "shared_source_content": shared,
            "numeric_table_checks": checks,
            "source_index": index,
            "source_contract": "Resolve Evidence content_ref in shared_source_content. "
                "grid_cells.rows is a losslessly packed cell matrix. "
                "table_coordinates reconstruct original Cell IDs with cell_id_rule. "
                "numeric_table_checks require exact agreement with the standardized "
                "source grid; computed findings never replace original cells. "
                "Do not silently resolve cell conflicts. Retrieved Evidence is "
                "a candidate, not a logical support assertion. "
                "Mapping relations remain separate from each Claim Mapping."}


def packet_pages(packet):
    """Page unique source content once; never re-expand it for each Evidence.

    All Evidence IDs remain separate citable records. Shared content is read
    under a representative grounded Evidence ID and its referenced IDs are
    listed explicitly. Oversized content is partitioned without discarding rows.
    """
    records = []
    for collection in ("claims", "evidence", "mappings", "mapping_relations"):
        for sid, value in packet[collection].items():
            records.append({"source_id": sid, "kind": collection, "record": deepcopy(value)})
    refs = {}
    for eid, record in packet["evidence"].items():
        refs.setdefault(record["content_ref"], []).append(eid)
    for ref, content in packet["shared_source_content"].items():
        associated = refs.get(ref) or []
        if not associated:
            continue
        item = {"source_id": associated[0], "kind": "shared_source_content",
                "content_ref": ref, "referenced_evidence_ids": associated,
                "source_content": content}
        if size(item) <= SOURCE_PAGE_BYTES:
            records.append(item)
            continue
        # Keep each original cell exactly once even in the large-table fallback.
        content = deepcopy(content)
        grid = content.get("grid_cells") if isinstance(content, dict) else None
        if isinstance(grid, dict) and isinstance(grid.get("rows"), list):
            rows = grid.pop("rows")
            for part, page in enumerate(record_pages(rows, 16000), 1):
                fragment = deepcopy(item)
                fragment["source_content"] = deepcopy(content)
                fragment["source_content"]["grid_cells"]["rows"] = page
                fragment["fragment"] = {"part": part, "kind": "original_table_rows"}
                records.append(fragment)
        elif isinstance(content, dict) and isinstance(content.get("cells"), list):
            cells = content.pop("cells")
            for part, page in enumerate(record_pages(cells, 16000), 1):
                fragment = deepcopy(item)
                fragment["source_content"] = deepcopy(content)
                fragment["source_content"]["cells"] = page
                fragment["fragment"] = {"part": part, "kind": "original_table_cells"}
                records.append(fragment)
        else:
            text = encoded(item)
            for part, start in enumerate(range(0, len(text), 9000), 1):
                records.append({"source_id": associated[0], "kind": "serialized_source_fragment",
                    "content_ref": ref, "text": text[start:start + 9000],
                    "fragment": {"part": part, "kind": "exact_json_text"}})
    return record_pages(records, byte_limit=min(26000, SOURCE_PAGE_BYTES))


def _direct_material_limit():
    """Reserve room for system instructions, schemas, catalog and task text."""
    try:
        cap = int(os.environ.get("READER_V3_MAX_REQUEST_TEXT_BYTES", "110000"))
    except ValueError:
        cap = 80000
    return min(FULL_VIEW_BYTES, max(10000, cap * 55 // 100))


def _source_ids_visible_on_page(page, packet):
    """Citable IDs actually present on this page, including shared references.

    A shared-content record has one transport representative but can explicitly
    list other Evidence records that cite its original cells. Those IDs are
    valid if grounded in the full packet AND visible in this page's content.
    A fragmented table may cite only Evidence anchored to the fragment cells.
    """
    known = set(packet["source_index"])
    visible = {item["source_id"] for item in page
               if item.get("source_id") in known}
    for item in page:
        if item.get("kind") != "shared_source_content":
            continue
        candidates = set(item.get("referenced_evidence_ids") or []) & known
        fragment = item.get("fragment") or {}
        if not fragment:
            visible.update(candidates)
            continue
        content = item.get("source_content") or {}
        grid = content.get("grid_cells") if isinstance(content, dict) else None
        cells = set()
        if isinstance(grid, dict) and isinstance(grid.get("rows"), list):
            for row in grid["rows"]:
                if not isinstance(row, list) or len(row) != 2 or not isinstance(row[1], list):
                    continue
                cells.update((row[0], col) for col, value in enumerate(row[1])
                             if value is not None)
        elif isinstance(content, dict) and isinstance(content.get("cells"), list):
            cells.update((c.get("row"), c.get("col")) for c in content["cells"]
                         if isinstance(c, dict))
        # A serialized source fragment is not a license to cite every related ID.
        for eid in candidates:
            anchor = (packet["evidence"].get(eid) or {}).get("source_anchor") or {}
            selector = anchor.get("selector") or {}
            coordinates = selector.get("positions") or []
            if selector.get("type") == "table_cells":
                coordinates = []
                source_id = anchor.get("element_id") or ""
                for cell_id in selector.get("cell_ids") or []:
                    if not isinstance(cell_id, str):
                        continue
                    match = re.fullmatch(re.escape(source_id) + r"\.R(\d+)C(\d+)", cell_id)
                    if match:
                        coordinates.append([int(match.group(1)), int(match.group(2))])
            if any(isinstance(x, (list, tuple)) and len(x) == 2
                   and tuple(x) in cells for x in coordinates):
                visible.add(eid)
    return visible


def material_view(*, coordinator, packet, run, data, store, purpose):
    """Use the full verified source when it fits the REAL request-byte guard.

    Numeric data and Evidence remain exact. Long material still uses paged
    source notes, with page-local references validated against visible sources.
    """
    if purpose == "final_answer":
        unfinished = run.get("unfinished_tasks") or []
        execution_notes = (["The following entries describe unfinished PROCESSING, not omissions in the paper: "
                            + encoded(unfinished)] if unfinished else [])
        preflight = coordinator.preflight_answer(
            question=run["user_question"],
            plan=CoordinatorPlan.model_validate(run["task_plan"]),
            answer_pack=packet, execution_notes=execution_notes,
            language=run.get("language", "en"),
        )
        if preflight["fits"]:
            run.setdefault("source_reading", {})[purpose] = {
                "representation": "direct_preflight_verified_sources",
                "packet_bytes": size(packet),
                "request_bytes": preflight["request_bytes"],
                "pages": 0, "unprocessed_source_ids": [],
            }
            return packet
        # Only a size-related preflight stop can be helped by pagination.
        # Unsupported models or budget policy errors must stop immediately.
        if "exceeding the per-call limit" not in preflight["reason"]:
            raise BudgetStop(preflight["reason"])
    # A failed exact final-answer preflight must NOT fall back to a looser
    # packet-only size estimate. Review uses its existing strict threshold.
    if purpose != "final_answer" and size(packet) <= _direct_material_limit():
        run.setdefault("source_reading", {})[purpose] = {
            "representation": "direct_compact_sources", "packet_bytes": size(packet),
            "pages": 0, "unprocessed_source_ids": [],
        }
        return packet
    pages = packet_pages(packet)
    notes, considered = [], []
    cache = run.setdefault("source_page_cache", {})
    for number, page in enumerate(pages, 1):
        key = signature({"policy": "unique_content_pages_v2", "question": run["user_question"],
                         "page": page})
        entry = cache.get(key)
        if entry is None:
            result = coordinator.source_notes(question=run["user_question"], material=page,
                                              page=number, page_count=len(pages))
            # Coverage counts the page's primary source records; citations
            # additionally permit explicitly advertised, truly visible Evidence.
            page_primary = {item["source_id"] for item in page}
            page_citable = _source_ids_visible_on_page(page, packet)
            bad = sorted({sid for note in result.notes for sid in note.source_ids
                          if sid not in page_citable})
            if bad:
                raise ValueError(
                    f"Source notes on page {number}/{len(pages)} cite IDs "
                    f"not present in that page's grounded sources: {bad[:12]}"
                )
            entry = {"source_ids": sorted(page_primary),
                     "notes": result.model_dump(mode="json")["notes"]}
            cache[key] = entry
            store.save(data)
        considered.extend(entry["source_ids"])
        notes.extend(entry["notes"])
    if set(considered) != set(packet["source_index"]):
        raise ValueError("Source page coverage does not match selected source IDs.")
    run.setdefault("source_reading", {})[purpose] = {
        "pages": len(pages), "processed_source_ids": unique(considered),
        "unprocessed_source_ids": [], "representation": "unique_content_source_notes",
    }
    store.save(data)
    return {"paper_id": packet["paper_id"], "paper_revision": packet["paper_revision"],
        "source_index": packet["source_index"], "source_notes": notes,
        "numeric_table_checks": packet.get("numeric_table_checks", []),
        "claims": packet["claims"], "mappings": packet["mappings"],
        "mapping_relations": packet.get("mapping_relations", {}),
        "all_source_pages_processed": True,
        "source_contract": "Source notes are lossy observations, not an indication of "
            "source absence. Exact numeric_table_checks come from source-matched "
            "table cells and must be consulted for numeric conclusions."}


def _mapping_groups(plan, run):
    """Combine overlapping reasoning work, never independent source scopes."""
    groups = {}
    for task in plan.tasks:
        if "mapping" not in required_capabilities(task.intent):
            continue
        state = run["task_states"].get(task.task_id, {})
        claims = tuple(sorted(state.get("selected_claim_ids") or []))
        scope = task.scope.model_dump(mode="json")
        # The selected Evidence is combined; user-imposed source boundaries
        # and Claim sets MUST be identical to share an analysis call.
        key = signature({"claims": claims,
                         "mode": scope["mode"],
                         "section_ids": sorted(scope["section_ids"]),
                         "element_ids": sorted(scope["element_ids"]),
                         "user_locked": scope["user_locked"],
                         "evidence_scope": task.evidence_scope})
        groups.setdefault(key, []).append(task)
    return list(groups.values())


def _mapping_paper_from_disk(data):
    """Read standardized metadata for section/reference retrieval, not extraction."""
    from app.models.document_v2 import PaperDocument
    path = Path(__file__).resolve().parents[1] / "data" / "standardized" / data.paper_id / "paper.json"
    if not path.is_file():
        return None
    paper = PaperDocument.model_validate_json(path.read_text(encoding="utf-8-sig"))
    if paper.paper_id != data.paper_id or paper.revision != data.paper_revision:
        raise ValueError("Standardized paper and analysis revisions do not match.")
    return paper


def execute_mappings(*, coordinator, data, store, run, plan, model, language,
                     tracker, budget, paper=None):
    """Execute only requested targets, or read their explicitly saved results."""
    if paper is None:
        paper = _mapping_paper_from_disk(data)
    targets = locked_targets(run)
    request = get_request(run)
    enforce_plan_targets(plan, run, data)
    for tasks in _mapping_groups(plan, run):
        states = [run["task_states"][task.task_id] for task in tasks]
        cids = unique(states[0].get("selected_claim_ids") or [])
        if targets and not set(cids) <= set(targets):
            raise ValueError("Mapping targets exceed the application-owned Claim selection.")
        if not cids:
            continue
        if not saved_only(run) and any(state.get("mapping_blocked_for_incomplete_coverage") for state in states):
            for state in states:
                state["mapping_ids"] = []
                state["mapping_ids_by_claim"] = {}
                state["deferred_claim_ids"] = list(cids)
                state["mapping_errors"] = ["Strict Evidence coverage is incomplete for this task."]
            store.save(data)
            continue
        if saved_only(run):
            allowed = evidence_boundary(run, paper) if paper is not None else None
            for state in states:
                bindings = bind_saved_task(data, state, cids,
                    request.mapping_ids_by_claim if request else {}, allowed_element_ids=allowed)
                run["used_mapping_ids"] = unique(run.get("used_mapping_ids", []) + list(bindings.values()))
                run.setdefault("execution_decisions", []).append({
                    "stage": "mapping", "action": "read_saved", "claim_ids": list(cids),
                    "mapping_ids_by_claim": dict(bindings), "new_model_calls": 0})
            store.save(data)
            continue
        for state in states:
            state["mapping_ids"] = []
            state["mapping_ids_by_claim"] = {}
            state["mapping_errors"] = []
            state["analyzed_claim_ids"] = []
            state["deferred_claim_ids"] = list(cids)
        focuses = sorted({task.mapping_focus or "general" for task in tasks})
        focus = focuses[0] if len(focuses) == 1 else "integrated:" + "+".join(focuses)
        if targets:
            allowed = evidence_boundary(run, paper) if paper is not None else None
        else:
            scopes = [scope_element_ids(paper, data, task) for task in tasks]
            allowed = None if any(scope is None for scope in scopes) else set().union(*scopes)
        def record(cid, mid):
            for state in states:
                state["mapping_ids_by_claim"][cid] = mid
                state["mapping_ids"] = list(state["mapping_ids_by_claim"].values())
                state["analyzed_claim_ids"] = unique(state["analyzed_claim_ids"] + [cid])
                state["deferred_claim_ids"] = [x for x in cids if x not in state["analyzed_claim_ids"]]
        run.setdefault("execution_decisions", []).append({
            "stage": "mapping", "action": "compute_or_exact_cache", "claim_ids": list(cids),
            "reason": "Requested analysis, not a saved-result explanation.", "analysis_focus": focus})
        result = run_claim_mappings(
            data=data, store=store, run=run, claim_ids=cids, model=model,
            language=language, tracker=tracker, budget=budget, paper=paper,
            analysis_focus=focus, allowed_element_ids=allowed, on_saved=record,
            rerun=bool(request and request.operation == "refresh_claims"))
        for state in states:
            state["deferred_claim_ids"] = list(result["pending_claim_ids"])
            state["mapping_errors"] = [str(issue) for issue in result["issues"]]
            state["mapping_diagnostics"] = list(result.get("diagnostics") or [])
            state["mapping_input_policy"] = "bounded_no_links_v2"
        run.setdefault("mapping_attempts", []).extend(result["issues"])
        store.save(data)
        if run.get("interruption_reason") == "BudgetStop":
            break
    # Current task bindings, not a union carried over from an earlier plan.
    run["used_mapping_ids"] = unique(mid for task in plan.tasks
        for mid in run.get("task_states", {}).get(task.task_id, {}).get("mapping_ids", []))
    store.save(data)


def _saved_mapping_answer(data, claim_ids, mapping_ids):
    """Present saved analyses without new reasoning or another model call."""
    lines = ["# Analysis of selected author Claims"]
    material, limitations = [], []
    mapping_set = set(mapping_ids)
    for cid in unique(claim_ids):
        claim = data.claim_pool.get(cid)
        if claim is None:
            continue
        lines.extend(["", "## " + cid, claim.source_quote or claim.model_quote])
        material.append(cid)
        candidates = [(mid, data.mappings.get(mid) or {}) for mid in mapping_ids if mid in mapping_set]
        rows = [(mid, record, row) for mid, record in candidates
                if record.get("status") == "completed"
                and record.get("paper_revision") == data.paper_revision
                for row in record.get("claim_analyses") or [] if row.get("claim_id") == cid]
        if not rows:
            lines.append("No completed Mapping is available for this Claim.")
            limitations.append("Mapping incomplete for " + cid)
            continue
        for mid, record, row in rows:
            row = presentation_analysis(data, cid, record)
            review_notes = row.get("presentation_review_notes") or []
            lines.extend(["", "### " + mid])
            if review_notes:
                lines.append("Historical analysis requiring annotation review; the following saved explanation is not a newly verified result.")
                for note in review_notes:
                    lines.append("Review note: " + note)
                limitations.extend(review_notes)
            lines.append(canonical_references(row.get("explanation", ""), row.get("evidence_ids_used") or []))
            material.append(mid)
            material.extend(eid for eid in row.get("evidence_ids_used") or [] if eid in data.evidence_pool)
            relations = row.get("evidence_relations") or []
            for edge in relations:
                label = edge.get("relation") or edge.get("relation_status", "not_recorded")
                lines.append("- " + edge["evidence_id"] + " [" + label + "; parts=" + ", ".join(edge.get("source_parts_used") or ["not_recorded"]) + "]: " + edge["explanation"])
            for note in row.get("qualifications") or []:
                lines.append("Qualification: " + str(note))
            receipt = record.get("retrieval_receipt") or {}
            if row.get("material_status") == "limited":
                limitations.append("Material limitations are recorded for " + cid)
            if receipt.get("priority_deferred") or receipt.get("missing_direct_source_ids"):
                limitations.append("Some priority sources were not included for " + cid)
    return {"answer": "\n\n".join(lines), "used_source_ids": unique(material),
            "limitations": unique(limitations)}


def final_answer(*, coordinator, data, store, run, plan, language, paper=None):
    """Present bound results; use at most one synthesis call for argument views."""
    cids, direct_eids, unfinished, argument_states = [], [], [], []
    for task in plan.tasks:
        state = run.get("task_states", {}).get(task.task_id, {})
        cids.extend(state.get("selected_claim_ids") or [])
        if "mapping" in required_capabilities(task.intent):
            argument_states.append(state)
            if not state.get("selected_claim_ids") or state.get("deferred_claim_ids") or state.get("mapping_errors"):
                unfinished.append({"task_id": task.task_id, "stage": "mapping",
                    "pending_claim_ids": state.get("deferred_claim_ids") or [],
                    "issues": state.get("mapping_errors") or []})
        else:
            direct_eids.extend(state.get("selected_evidence_ids") or [])
        if not state.get("collection_ready"):
            unfinished.append({"task_id": task.task_id, "stage": "collection",
                               "missing": state.get("missing_information") or []})
    cids = unique(cids)
    targets = locked_targets(run)
    request = get_request(run)
    if targets and argument_states:
        cids = [cid for cid in cids if cid in targets]
    bindings, view_errors = mapping_bindings(
        data, argument_states, targets=set(targets) if targets else None,
        pins=request.mapping_ids_by_claim if request else None)
    if view_errors:
        unfinished.append({"stage": "display_binding", "issues": view_errors})
    mids = list(bindings.values())
    run["display_by_claim"] = {cid: display_view(data, cid, mid) for cid, mid in bindings.items()}
    for cid in cids:
        if argument_states and cid not in run["display_by_claim"]:
            run["display_by_claim"][cid] = {"claim_id": cid, "mapping_id": None,
                "status": "unavailable", "evidence_ids": [], "edges": [],
                "issues": ["No completed Mapping is bound to this requested Claim."]}
    run["display_evidence_ids"] = unique(eid for view in run["display_by_claim"].values()
                                        for eid in view.get("evidence_ids", []))
    run["display_policy"] = "per_claim_pinned_mapping_origin_guard_v2"
    run["display_evidence_ids_role"] = "aggregate_diagnostic_only_not_a_selected_claim_view"
    run["explicit_evidence_only_ids"] = unique(direct_eids)
    if not argument_states:
        run["display_evidence_ids"] = unique(direct_eids)
        run["display_policy"] = "explicit_claim_or_evidence_source_operation"
        run["display_evidence_ids_role"] = "explicitly_requested_source_records"
    run["used_mapping_ids"] = mids
    if argument_states:
        packet = argument_packet(data, cids, bindings)
        # Explicit Evidence-only subtasks stay separate from argument bindings.
        if direct_eids:
            extra = source_packet(data, evidence_ids=unique(direct_eids), paper=paper,
                                  question=run["user_question"])
            for field in ("evidence", "shared_source_content", "source_index"):
                for key, value in extra[field].items():
                    packet[field].setdefault(key, value)
            packet["explicit_evidence_only_ids"] = unique(direct_eids)
            packet["source_contract"] += " Evidence-only materials are for descriptive subtasks, not new Claim support."
    else:
        packet = source_packet(data, claim_ids=cids, evidence_ids=unique(direct_eids),
                               paper=paper, question=run["user_question"])
    run["answer_material_ids"] = list(packet["source_index"])
    run["unfinished_tasks"] = unfinished
    run["material_limitations_by_claim"] = {cid: list(view.get("qualifications") or [])
                                           for cid, view in run["display_by_claim"].items()
                                           if view.get("material_status") == "limited"}
    if not packet["source_index"]:
        run.update({"answer": None, "answer_available": False,
                    "answer_status": "unavailable", "status": "partial"})
        store.save(data)
        return
    answer_key = signature({"policy": "bound_answer_v2_origin_guard", "question": run["user_question"],
                            "contract": run.get("request_contract"), "material": packet,
                            "unfinished": unfinished, "language": language})
    previous = run.get("answer") or {}
    if (run.get("answer_input_signature") == answer_key and previous.get("answer")
            and run.get("answer_rendering_mode") != "saved_fallback"
            and not run.get("answer_reference_check", {}).get("unknown_ids")):
        reply = previous
        run["answer_reused"] = True
        rendering_mode = run.get("answer_rendering_mode", "model")
    else:
        if previous.get("answer"):
            run.setdefault("answer_history", []).append({"answer": previous,
                "input_signature": run.get("answer_input_signature"), "archived_at": now()})
        run["answer_reused"] = False
        rendering_mode = "model"
        direct = all(t.intent in {"list_claims", "show_evidence"} and t.output_style == "direct" for t in plan.tasks)
        if run.get("mode") == "selected_analysis":
            reply = _saved_mapping_answer(data, cids, mids)
            rendering_mode = "saved_results"
        elif direct:
            full = deepcopy(packet)
            for item in full["evidence"].values():
                item["source_content"] = full["shared_source_content"][item.pop("content_ref")]
            text = render_source_fallback(full)
            if not isinstance(text, str) or not text.strip():
                text = "Selected source records:\n\n```json\n" + json.dumps(full, ensure_ascii=False, indent=2) + "\n```"
            tables = {eid: item["source_content"] for eid, item in full["evidence"].items() if item.get("kind") == "table"}
            if tables:
                text += "\n\n### Structured table data\n```json\n" + json.dumps(tables, ensure_ascii=False, indent=2) + "\n```"
            reply = {"answer": text, "used_source_ids": list(packet["source_index"]), "limitations": []}
            rendering_mode = "source_listing"
        elif argument_states:
            notes = ["Only the bound Mapping results may supply argument judgments. Missing saved analysis must not trigger implicit extraction."]
            context_note = question_context_note(run)
            if context_note:
                notes.append(context_note)
            if unfinished:
                notes.append("Unfinished requested processing: " + encoded(unfinished))
            # No paid page-notes pipeline for an ordinary saved-Claim explanation.
            try:
                cap = int(os.environ.get("READER_V3_SAVED_ANSWER_BYTES", "48000"))
                if not 4000 <= cap <= 100000:
                    raise ValueError("READER_V3_SAVED_ANSWER_BYTES must be between 4000 and 100000.")
                if size(packet) > cap:
                    raise BudgetStop("Bound answer material exceeds the configured presentation size limit.")
                checked = coordinator.preflight_answer(question=run["user_question"], plan=plan,
                    answer_pack=packet, execution_notes=notes, language=language)
                if not checked["fits"]:
                    raise BudgetStop(checked["reason"])
                print("Generating one Coordinator answer from bound Mapping results.", flush=True)
                result = coordinator.answer(question=run["user_question"], plan=plan,
                    answer_pack=packet, execution_notes=notes, language=language)
                reply = result.model_dump(mode="json")
                run["source_reading"] = {"final_answer": {"representation": "bound_mapping_sources",
                    "packet_bytes": size(packet), "request_bytes": checked["request_bytes"], "pages": 0}}
            except MODEL_ERRORS as exc:
                # Preserve actual analyses without disguising a failed synthesis.
                reply = _saved_mapping_answer(data, cids, mids)
                reply["limitations"] = unique(list(reply.get("limitations") or []) + [
                    "The question-specific synthesis was not completed. Saved analyses are shown without new reasoning."])
                run.setdefault("presentation_diagnostics", []).append(error_message(exc))
                rendering_mode = "saved_fallback"
                unfinished.append({"stage": "answer_synthesis", "reason": error_message(exc)})
        else:
            # Preserve established Claim-only and Evidence-only presentation.
            view = material_view(coordinator=coordinator, packet=packet, run=run, data=data,
                                 store=store, purpose="final_answer")
            notes = ["Unfinished processing, not paper omissions: " + encoded(unfinished)] if unfinished else []
            context_note = question_context_note(run)
            if context_note:
                notes.append(context_note)
            result = coordinator.answer(question=run["user_question"], plan=plan, answer_pack=view,
                                        execution_notes=notes, language=language)
            reply = result.model_dump(mode="json")
    if (not argument_states and not unfinished and rendering_mode == "model"
            and run.get("mode") != "selected_analysis"):
        comparative = bool(re.search(r"\b(ties?|counterexamples?|compare|comparison|across|best)\b", run["user_question"], re.I)) or any(
            term in run["user_question"] for term in ("最低", "最高", "并列", "比较", "反例", "所有数据集"))
        if comparative and paper is not None:
            appendix = render_verified_numeric_checks(audits_for_packet(paper, data, run["user_question"], packet["evidence"]), language=language)
            if appendix and not run.get("answer_reused"):
                reply["answer"] = reply["answer"].rstrip() + "\n\n" + appendix
                run["deterministic_table_check_appended"] = True
    if not str(reply.get("answer") or "").strip():
        raise ValueError("Coordinator returned an empty final answer.")
    # A narrowly deterministic ID normalization does not alter scientific text.
    reply["answer"] = canonical_references(reply["answer"], packet["source_index"])
    check = check_answer_references(reply["answer"], packet)
    known = {sid.casefold() for sid in packet["source_index"]}
    unknown = unique(check["unknown_ids"] + [sid for sid in reply.get("used_source_ids", []) if sid.casefold() not in known])
    run["answer"] = reply
    run["answer_input_signature"] = answer_key
    run["answer_rendering_mode"] = rendering_mode
    run["answer_scope"] = "saved_analysis" if saved_only(run) else "requested_operation"
    run["answer_reference_check"] = {**check, "unknown_ids": unknown}
    run["answer_status"] = "needs_reference_review" if unknown else ("partial_generated" if unfinished else "generated")
    run["status"] = "partial" if unfinished or unknown else "completed"
    run["answer_available"] = True
    run["processing_status"] = "incomplete" if unfinished or unknown else "completed"
    run["analysis_material_status"] = "limited" if run["material_limitations_by_claim"] else "as_recorded"
    store.save(data)
    print("\n=== COORDINATOR ANSWER ===\n" + reply["answer"], flush=True)


def save_outputs(data, run, analysis_path):
    """Point callers at fixed current exports; AnalysisStore.save writes them.

    There are no automatic run-ID snapshots or historical processing reports.
    Error, completion and material-limitation state remain in response.json.
    """
    base = Path(analysis_path).parent
    run["display_file"] = str((base / "display.json").resolve())
    if (run.get("answer") or {}).get("answer"):
        run["answer_file"] = str((base / "answer.md").resolve())
    else:
        run.pop("answer_file", None)
        run["answer_available"] = False
    run.pop("processing_report_file", None)
    run.pop("historical_processing_reports", None)


def main():
    parser = argparse.ArgumentParser(description="Resume through the same request contract as the main workflow")
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--resume-run", required=True)
    parser.add_argument("--paper", type=Path)
    parser.add_argument("--model", default="gpt-5.4-mini", help="Compatibility only; resume retains its saved model")
    parser.add_argument("--language", choices=["en", "zh"], default="en")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    data = AnalysisDocument.model_validate_json(args.analysis.read_text(encoding="utf-8-sig"))
    root = Path(__file__).resolve().parents[1]
    from app.core.analysis_store import default_analysis_path
    expected = (root / default_analysis_path(data.paper_id)).resolve()
    if args.analysis.resolve() != expected:
        parser.error("Unified resume requires this paper's canonical analysis.json; it will not silently write elsewhere.")
    paper_path = args.paper or root / "data" / "standardized" / data.paper_id / "paper.json"
    from app.workflow_runner import run_workflow, run_selected_analysis
    run = find_saved_run(data, args.resume_run)
    settings = argparse.Namespace(
        paper=paper_path.resolve(), resume_workflow=args.resume_run, question=None,
        request_file=None, target_claim_id=[], analysis_operation="auto",
        selection_file=None, analyze_selection_file=None, fresh_analysis=False,
        question_limit_usd=None, model=run.get("model", args.model),
        language=run.get("language", args.language), image_mode=run.get("image_mode", "relevant"),
        focus_section=[], focus_element=[], collection_rounds=3,
        preview=args.preview, plan_only=False, use_saved_sources=True,
        strict_evidence_coverage=False)
    os.chdir(root)
    if run.get("mode") == "selected_analysis":
        if args.preview:
            print("Selected-analysis recovery uses its saved text selection and the same budget.")
            return
        run_selected_analysis(settings, resume_id=args.resume_run)
    else:
        run_workflow(settings)


if __name__ == "__main__":
    main()
