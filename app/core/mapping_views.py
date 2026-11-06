"""Read saved analyses and build per-Claim presentation bindings.

Reading a saved result is not a new Mapping cache hit. This module validates
provenance and reports limitations, but never asserts fresh scientific coverage.
No Links, candidate scores, other Claim targets, or new source extractions enter
the default argument display.
"""
from __future__ import annotations

from copy import deepcopy
import re

from app.core.evidence_retrieval import (
    available_source_parts, evidence_view, fingerprint, same_claim_source, claim_origin_only,
)

RELATIONS = {"supports", "contradicts", "qualifies", "context", "insufficient_support"}


def unique(values):
    return list(dict.fromkeys(value for value in values if value))


def record_fingerprint(record):
    value = record.model_dump(mode="json") if hasattr(record, "model_dump") else record
    return fingerprint(value)


def check_saved_mapping(data, claim_id, mapping_id, *, allowed_element_ids=None):
    """Validate readability, not scientific completeness or a new input cache."""
    errors, warnings = [], []
    claim = data.claim_pool.get(claim_id)
    record = data.mappings.get(mapping_id)
    if claim is None or not isinstance(record, dict):
        return {"usable": False, "errors": ["Missing Claim or saved Mapping."], "warnings": []}
    if record.get("claim_id") != claim_id or record.get("status") != "completed":
        errors.append("The Mapping is not a completed analysis of this Claim.")
    if record.get("paper_revision") != data.paper_revision:
        errors.append("The Mapping belongs to another paper revision.")
    if record.get("validation_issues") or record.get("unresolved_evidence_ids"):
        errors.append("The Mapping has unresolved source validation errors.")
    rows = record.get("claim_analyses") or []
    if len(rows) != 1 or rows[0].get("claim_id") != claim_id:
        errors.append("A readable per-Claim result must contain exactly its own analysis.")
        return {"usable": False, "errors": errors, "warnings": warnings}
    row = rows[0]
    if not str(row.get("explanation") or "").strip():
        errors.append("The saved analysis is empty.")
    used = unique(row.get("evidence_ids_used") or [])
    expected_claim = record.get("claim_snapshot_fingerprint")
    if expected_claim and expected_claim != record_fingerprint(claim):
        errors.append("The Claim changed after this Mapping was saved.")
    expected_sources = record.get("used_evidence_fingerprints") or {}
    if not expected_claim or set(expected_sources) != set(used):
        warnings.append("Historical snapshot: source fingerprints were not fully recorded; paper revision and source IDs were checked.")
    edges = row.get("evidence_relations") or []
    edge_ids = [edge.get("evidence_id") for edge in edges if isinstance(edge, dict)]
    legacy_unannotated = bool(used and not edges and
        record.get("record_policy") != "per_claim_no_links_relations_v4_origin_guard")
    if legacy_unannotated:
        warnings.append("Historical Mapping has no per-Evidence relation annotations; relation labels remain null, never inferred from use.")
    elif len(edge_ids) != len(set(edge_ids)) or set(edge_ids) != set(used):
        errors.append("Saved Evidence relations do not match the adopted Evidence IDs.")
    for eid in used:
        source = data.evidence_pool.get(eid)
        if source is None:
            errors.append(f"Missing adopted Evidence: {eid}.")
            continue
        if source.source_status != "located" or source.source_anchor is None:
            errors.append(f"Adopted Evidence is not source-located: {eid}.")
        if allowed_element_ids is not None and source.source_element_id not in allowed_element_ids:
            errors.append(f"Adopted Evidence is outside the requested source boundary: {eid}.")
        expected = expected_sources.get(eid)
        if expected and expected != record_fingerprint(source):
            errors.append(f"Adopted Evidence changed after analysis: {eid}.")
    for edge in edges:
        if not isinstance(edge, dict) or edge.get("relation") not in RELATIONS or not str(edge.get("explanation") or "").strip():
            errors.append("Invalid saved Evidence relation.")
            continue
        source = data.evidence_pool.get(edge.get("evidence_id"))
        parts = edge.get("source_parts_used") or []
        if not parts:
            warnings.append("Historical relation without source_parts_used; no caption/pixel/cell usage is inferred.")
        elif source is not None:
            supplied = set(available_source_parts(evidence_view(source, include_context=True)))
            if not set(parts) <= supplied:
                errors.append(f"Saved source component is unavailable: {edge.get('evidence_id')}.")
    if row.get("material_status") == "limited":
        warnings.append("The saved Mapping reports material limitations; do not present it as exhaustive verification.")
    review = presentation_analysis(data, claim_id, record)
    warnings.extend(review.get("presentation_review_notes") or [])
    return {"usable": not errors, "errors": unique(errors), "warnings": unique(warnings)}


def select_saved_mapping(data, claim_id, *, pinned_mapping_id=None, allowed_element_ids=None):
    """Honor an explicit version; otherwise use the active/latest valid version."""
    candidates = ([pinned_mapping_id] if pinned_mapping_id else
                  unique([data.mapping_by_claim.get(claim_id), *reversed(list(data.mappings))]))
    diagnostics = []
    for mid in candidates:
        record = data.mappings.get(mid) or {}
        if not pinned_mapping_id and record.get("claim_id") != claim_id:
            continue
        check = check_saved_mapping(data, claim_id, mid, allowed_element_ids=allowed_element_ids)
        if check["usable"]:
            return mid, check
        diagnostics.append({"mapping_id": mid, "errors": check["errors"]})
    return None, {"usable": False, "errors": ["No readable saved Mapping is available for this target."],
                  "warnings": [], "rejected_versions": diagnostics}


def bind_saved_task(data, state, claim_ids, pins=None, *, allowed_element_ids=None):
    """Read-only source reuse; a missing result never authorizes new extraction."""
    pins = pins or {}
    selected, checks = {}, {}
    for cid in unique(claim_ids):
        # Persist the selected version within the question, including on resume.
        existing = (state.get("mapping_ids_by_claim") or {}).get(cid)
        mid, check = select_saved_mapping(data, cid, pinned_mapping_id=pins.get(cid) or existing,
                                         allowed_element_ids=allowed_element_ids)
        checks[cid] = check
        if mid:
            selected[cid] = mid
    missing = [cid for cid in claim_ids if cid not in selected]
    state.update({"selected_claim_ids": list(claim_ids), "mapping_ids_by_claim": selected,
                  "mapping_ids": list(selected.values()), "analyzed_claim_ids": list(selected),
                  "deferred_claim_ids": missing, "mapping_errors": [],
                  "mapping_read_checks": checks, "mapping_execution": "read_saved_only",
                  "collection_review_mode": "saved_mapping_provenance_check",
                  "collection_ready": not missing,
                  "missing_information": [f"No usable saved Mapping for {cid}." for cid in missing],
                  "selected_evidence_ids": unique(eid for mid in selected.values()
                      for row in data.mappings[mid]["claim_analyses"] for eid in row.get("evidence_ids_used", [])),
                  "coverage": {"mode": "saved_analysis_only", "new_source_discovery": False,
                               "exhaustive_recall_claimed": False}})
    return selected


def canonical_references(text, allowed_ids):
    """Normalize only an unambiguous shorthand of an actually supplied ID."""
    allowed = set(allowed_ids)
    def replace(match):
        candidate = "evidence_" + match.group(1)
        return candidate if candidate in allowed else match.group(0)
    return re.sub(r"\be(\d{6})\b", replace, str(text or ""), flags=re.I)


def presentation_analysis(data, claim_id, record):
    """Project legacy annotations safely without editing stored Mapping results.

    A null relation is NOT a newly inferred context/contradiction. Its status
    explains whether the old record lacked a label or an exact self-support label
    was withheld. Original labels, explanations and IDs remain available for
    audit; new scientific judgments still require the Mapping Agent.
    """
    rows = record.get("claim_analyses") or []
    source_row = next((r for r in rows if r.get("claim_id") == claim_id), {})
    row = deepcopy(source_row)
    claim = data.claim_pool.get(claim_id)
    by_id = {edge.get("evidence_id"): edge for edge in row.get("evidence_relations") or []
             if isinstance(edge, dict)}
    projected, notes = [], []
    for eid in unique(row.get("evidence_ids_used") or []):
        edge = deepcopy(by_id.get(eid) or {})
        source = data.evidence_pool.get(eid)
        origin = bool(claim is not None and source is not None and same_claim_source(claim, source))
        parts = list(edge.get("source_parts_used") or [])
        if not edge:
            edge = {"evidence_id": eid, "relation": None, "source_parts_used": [],
                    "relation_status": "not_recorded",
                    "explanation": "This historical Mapping recorded use of the source, but no per-Evidence relation. No support label is inferred."}
            notes.append(f"{eid}: relation was not recorded; source use is not a support judgment.")
        else:
            edge["relation_status"] = "recorded"
        if origin:
            edge["source_text_role_for_target"] = "claim_origin"
            edge["origin_note"] = "The source text is the target Claim's identical source sentence; it establishes attribution, not an additional premise."
            # Missing historical component metadata cannot certify that a larger
            # neighboring context, rather than the identical sentence, was used.
            no_added_part = not parts or claim_origin_only(claim, source, source_parts=parts)
            if no_added_part and edge.get("relation") in {"supports", "contradicts", "qualifies"}:
                edge["recorded_relation"] = edge["relation"]
                edge["recorded_explanation"] = edge.get("explanation", "")
                edge["relation"] = None
                edge["relation_status"] = "needs_origin_review"
                edge["explanation"] = (
                    "The historical relation is withheld: the recorded text is the target Claim's exact source sentence, "
                    "with no recorded additional component establishing this relation. The original label is retained for audit; this is not a new analysis.")
                notes.append(f"{eid}: historical {edge['recorded_relation']} label requires origin review; do not repeat it as established support.")
        projected.append(edge)
    row["evidence_relations"] = projected
    row["presentation_review_notes"] = unique(notes)
    row["presentation_policy"] = "origin_guard_read_view_v1"
    row["historical_explanation_requires_review"] = bool(notes)
    return row


def display_view(data, claim_id, mapping_id):
    check = check_saved_mapping(data, claim_id, mapping_id)
    if not check["usable"]:
        return {"claim_id": claim_id, "mapping_id": mapping_id, "status": "unavailable",
                "evidence_ids": [], "edges": [], "issues": check["errors"]}
    claim = data.claim_pool[claim_id]
    record = data.mappings[mapping_id]
    row = presentation_analysis(data, claim_id, record)
    used = unique(row.get("evidence_ids_used") or [])
    by_evidence = {edge["evidence_id"]: edge for edge in row.get("evidence_relations") or []}
    highlights, edges, nodes = [], [], [{"id": claim_id, "kind": "claim"}]
    for eid in used:
        source, edge = data.evidence_pool[eid], by_evidence[eid]
        parts = list(edge.get("source_parts_used") or [])
        # Highlight records intentionally carry no scientific relation label.
        highlights.append({"evidence_id": eid, "kind": source.kind,
                           "source_element_id": source.source_element_id,
                           "source_anchor": source.source_anchor.model_dump(mode="json"),
                           "source_parts_used": parts,
                           "source_asset_id": source.source_asset_id,
                           "image_receipt_available": bool(source.image_receipt)})
        nodes.append({"id": eid, "kind": source.kind, "source_element_id": source.source_element_id})
        projected_edge = {"from": eid, "to": claim_id, "mapping_id": mapping_id,
                          "relation": edge["relation"], "explanation": edge["explanation"],
                          "source_parts_used": parts,
                          "relation_status": edge.get("relation_status", "recorded")}
        for field in ("recorded_relation", "recorded_explanation", "source_text_role_for_target", "origin_note"):
            if field in edge:
                projected_edge[field] = edge[field]
        edges.append(projected_edge)
    return {"claim_id": claim_id, "mapping_id": mapping_id, "status": "available",
            "material_status": row.get("material_status", "limited"),
            "claim_anchor": claim.source_anchor.model_dump(mode="json") if claim.source_anchor else None,
            "evidence_ids": used, "evidence_highlights": highlights,
            "nodes": nodes, "edges": edges, "qualifications": list(row.get("qualifications") or []),
            "provenance_warnings": check["warnings"],
            "presentation_policy": row["presentation_policy"],
            "relation_review_required": bool(row["presentation_review_notes"])}


def mapping_bindings(data, task_states, *, targets=None, pins=None):
    """Choose one explicit Mapping version per displayed Claim, never an ID union."""
    candidates = {}
    for state in task_states:
        target_set = set(state.get("selected_claim_ids") or [])
        for mid in state.get("mapping_ids") or []:
            record = data.mappings.get(mid) or {}
            cid = record.get("claim_id")
            if cid in target_set and (targets is None or cid in targets):
                candidates.setdefault(cid, []).append(mid)
    bindings, issues = {}, []
    for cid, mids in candidates.items():
        pin = (pins or {}).get(cid)
        preferred = pin or (data.mapping_by_claim.get(cid) if data.mapping_by_claim.get(cid) in mids else mids[-1])
        if preferred not in mids:
            issues.append(f"Requested Mapping is not bound to this task: {cid}/{preferred}.")
            continue
        check = check_saved_mapping(data, cid, preferred)
        if check["usable"]:
            bindings[cid] = preferred
        else:
            issues.extend(check["errors"])
    return bindings, unique(issues)


def adopted_evidence_view(record, source_parts):
    """Expose only the exact source components adopted by a saved Mapping."""
    parts = set(source_parts)
    full = evidence_view(record, include_context="source_context" in parts)
    result = {key: full.get(key) for key in ("evidence_id", "kind", "source_element_id", "source_group", "source_status")}
    # UI anchors stay in display_view; they need not repeat source text in the LLM.
    result["source_parts_used"] = sorted(parts)
    keys = {"source_text": ("source_text", "quotation_confirmed"),
            "source_context": ("source_context",), "caption": ("caption", "label"),
            "visual_observation": ("visual_observation", "visual_reading_verified", "pixels_were_submitted", "image_extraction_note"),
            "table_cells": ("cells", "cell_fields", "row_group_context", "table_scope", "recognition_status"),
            "equation": ("equation",)}
    for part in parts:
        for key in keys.get(part, ()):
            if key in full:
                result[key] = full[key]
    result["source_contract"] = "Only the listed components were adopted. Unlisted components are not absent from the paper."
    if not parts:
        result["historical_usage_note"] = "Component usage was not recorded; consult the saved explanation without claiming new pixel/cell verification."
    return result


def argument_packet(data, claim_ids, bindings):
    """A compact answer packet built from completed, specifically bound analyses."""
    claims, evidence, mappings, shared, index = {}, {}, {}, {}, {}
    adopted = {}
    warnings = []
    for cid in unique(claim_ids):
        claim = data.claim_pool.get(cid)
        if claim is None:
            continue
        claims[cid] = {"claim_id": cid, "type": claim.type, "source_quote": claim.source_quote or claim.model_quote,
                       "source_element_id": claim.source_element_id, "source_status": claim.source_status}
        index[cid] = {"kind": "claim", "source_status": claim.source_status, "source_element_id": claim.source_element_id}
        mid = bindings.get(cid)
        if not mid:
            continue
        check = check_saved_mapping(data, cid, mid)
        if not check["usable"]:
            warnings.extend(check["errors"])
            continue
        record = data.mappings[mid]
        row = presentation_analysis(data, cid, record)
        warnings.extend(row.get("presentation_review_notes") or [])
        permitted = set(row.get("evidence_ids_used") or [])
        row["explanation"] = canonical_references(row.get("explanation"), permitted)
        for edge in row.get("evidence_relations") or []:
            edge["explanation"] = canonical_references(edge.get("explanation"), permitted)
            adopted.setdefault(edge["evidence_id"], set()).update(edge.get("source_parts_used") or [])
        mappings[mid] = {"mapping_id": mid, "claim_id": cid, "status": "completed",
                         "target_claim_ids": [cid], "analysis_focus": record.get("analysis_focus"),
                         "claim_analyses": [row], "read_provenance_warnings": check["warnings"]}
        index[mid] = {"kind": "mapping", "source_status": "analysis"}
    for eid, parts in adopted.items():
        record = data.evidence_pool[eid]
        view = adopted_evidence_view(record, parts)
        # Shared caption/text content is transmitted once, while IDs stay citable.
        identity = {k: v for k, v in view.items() if k not in {"evidence_id"}}
        ref = "content_" + fingerprint(identity)
        shared.setdefault(ref, identity)
        evidence[eid] = {"evidence_id": eid, "kind": record.kind,
                         "source_element_id": record.source_element_id, "source_status": record.source_status,
                         "content_ref": ref, "source_parts_used": sorted(parts)}
        index[eid] = {"kind": "evidence", "source_element_id": record.source_element_id,
                      "source_status": record.source_status}
    relations = {rid: deepcopy(row) for rid, row in data.mapping_relations.items()
                 if set(row.get("claim_ids") or []) <= set(claims)
                 and set(row.get("mapping_ids") or []) <= set(mappings)
                 and row.get("paper_revision") == data.paper_revision}
    for rid in relations:
        index[rid] = {"kind": "mapping_relation", "source_status": "analysis"}
    return {"paper_id": data.paper_id, "paper_revision": data.paper_revision,
            "claims": claims, "evidence": evidence, "mappings": mappings,
            "mapping_relations": relations, "shared_source_content": shared,
            "numeric_table_checks": [], "source_index": index,
            "mapping_bindings": dict(bindings), "view_warnings": unique(warnings),
            "source_contract": "Saved Mapping is the source for each Claim's historical argument. Rephrase, do not add new reasoning. Only adopted Evidence components are supplied. A null relation has NOT been scientifically reclassified. Follow presentation_review_notes: do not count an identical Claim-origin sentence as additional support, even if historical prose did. Missing historical labels must not be filled from evidence_ids_used. Report the historical limitation rather than declaring the Claim true or false. A nonempty review list requires a visible caveat; unlisted source parts are not absent from the paper."}
