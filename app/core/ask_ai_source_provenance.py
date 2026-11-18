"""Validated Ask AI citation locations and scope ownership for saved/API output.

This projects only IDs actually reported by the final answer after source packet
validation. It is not a second analysis, a Claim-Evidence Link, or proof that
any scientific statement was independently verified.
"""
from __future__ import annotations


def _anchor(record):
    value = getattr(record, "source_anchor", None)
    return value.model_dump(mode="json") if value is not None else None


def used_source_provenance(*, sources: dict, packet: dict, paper, data) -> dict:
    """Build front-end locators from validated answer IDs, never invented IDs."""
    grouped = packet.get("candidate_source_groups") or {}
    original = grouped.get("original_selection") or {}
    approved = grouped.get("approved_scope") or {}

    def ownership(source_id: str, id_kind: str) -> str:
        column = {
            "claim": "claim_ids", "evidence": "evidence_ids",
            "mapping": "mapping_ids", "element": "source_element_ids",
        }[id_kind]
        if source_id in original.get(column, []):
            return "original_selection"
        if source_id in approved.get(column, []):
            return "approved_scope"
        # Normal selection-only Ask AI has no explicit comparison groups.
        # Recover exact selected-source roles from the validated packet.
        selected_ids = {item["element_id"]
                        for item in packet.get("selection", [])}
        direct_ids = {item["element_id"]
                      for item in packet.get("related_source_text", [])}
        if id_kind == "element" and source_id in selected_ids | direct_ids:
            return "original_selection"
        if id_kind == "claim":
            item = packet.get("claims", {}).get(source_id) or {}
            if item.get("source_element_id") in selected_ids:
                return "original_selection"
        if id_kind == "mapping":
            item = packet.get("mappings", {}).get(source_id) or {}
            bound_claim = data.claim_pool.get(item.get("claim_id"))
            if bound_claim and bound_claim.source_element_id in selected_ids:
                return "original_selection"
        if id_kind == "evidence":
            item = packet.get("evidence", {}).get(source_id) or {}
            if item.get("question_source_role") in {
                "selected_text_evidence", "direct_selection_reference",
            }:
                return "original_selection"
        # Evidence located elsewhere that supports a selected Claim is not a
        # newly authorized topic; its exact location is still reported.
        return "selected_topic_support"

    rows = []
    by_side = {"original_selection": [], "approved_scope": [],
               "selected_topic_support": []}
    fields = (
        ("claim", "used_claim_ids"),
        ("evidence", "used_evidence_ids"),
        ("mapping", "used_mapping_ids"),
        ("element", "used_source_element_ids"),
    )
    for source_type, key in fields:
        for source_id in dict.fromkeys(sources.get(key) or []):
            physical_id = None
            source_anchor = None
            is_derived = source_type == "mapping"
            if source_type == "claim":
                record = data.claim_pool[source_id]
                physical_id = record.source_element_id
                source_anchor = _anchor(record)
            elif source_type == "evidence":
                record = data.evidence_pool[source_id]
                physical_id = record.source_element_id
                source_anchor = _anchor(record)
            elif source_type == "mapping":
                record = data.mappings.get(source_id) or {}
                claim = data.claim_pool.get(record.get("claim_id"))
                if claim is not None:
                    physical_id = claim.source_element_id
                    # Only a navigation hint to the Claim: a Mapping has no
                    # independent source anchor in the paper.
                    source_anchor = _anchor(claim)
            else:
                physical_id = source_id
                source_anchor = {
                    "element_id": source_id,
                    "selector": {"type": "element"},
                }

            element = paper.elements.get(physical_id) if physical_id else None
            side = ownership(source_id, source_type)
            rows.append({
                "source_id": source_id,
                "source_type": source_type,
                "scope_role": side,
                "source_element_id": physical_id,
                "section_id": (element.section_id if element is not None else None),
                "element_kind": (element.kind if element is not None else None),
                "source_anchor": source_anchor,
                "is_derived_analysis": is_derived,
            })
            by_side[side].append(source_id)

    if sources.get("used_source_element_ids"):
        raise ValueError("Raw paper elements must not be direct answer sources.")

    return {
        "schema_version": "reader.qa.source_provenance.v1",
        "sources": rows,
        "used_source_ids_by_scope": by_side,
        "comparison_coverage": {
            "original_selection_cited": bool(by_side["original_selection"]),
            "approved_scope_cited": bool(by_side["approved_scope"]),
            "both_sides_cited": bool(by_side["original_selection"] and
                                     by_side["approved_scope"]),
        },
        "note": (
            "Validated answer-reported sources, not all available sources. "
            "Claim quotations are not independent confirmation, and a "
            "Mapping is derived analysis rather than original Evidence. "
            "Paper-element IDs are locations for highlighting, not direct answer citations."
        ),
    }
