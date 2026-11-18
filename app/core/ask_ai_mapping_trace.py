"""Render *used* per-Claim Mapping and its source-bound Evidence for the UI.

Nested records explain a Mapping's provenance. They are NOT independent direct
citations of the Ask AI answer, new Claims-Evidence Links, or factual verification.
"""
from __future__ import annotations

from copy import deepcopy

from app.core.ask_ai_scope import saved_mapping_view


def _anchor(record):
    anchor = getattr(record, "source_anchor", None)
    return anchor.model_dump(mode="json") if anchor is not None else None


def build_mapping_trace(*, data, used_mapping_ids: list[str], limit: int = 8) -> dict:
    rows = []
    for mapping_id in list(dict.fromkeys(used_mapping_ids))[:limit]:
        record = data.mappings.get(mapping_id) or {}
        claim_id = record.get("claim_id")
        if not claim_id or data.mapping_by_claim.get(claim_id) != mapping_id:
            raise ValueError("A cited Mapping has no valid single-Claim binding.")
        mapping = saved_mapping_view(data, claim_id)
        if mapping is None or mapping["mapping_id"] != mapping_id:
            raise ValueError("A cited Mapping is no longer readable.")
        claim = data.claim_pool[claim_id]
        relations = {
            edge.get("evidence_id"): edge
            for edge in mapping.get("evidence_relations") or []
            if isinstance(edge, dict) and edge.get("evidence_id")
        }
        adopted = []
        for eid in mapping.get("evidence_ids_used_by_mapping") or []:
            source = data.evidence_pool.get(eid)
            if source is None:
                raise ValueError("A cited Mapping has a missing adopted Evidence.")
            edge = relations.get(eid) or {}
            source_content = source.source_content or {}
            # Bounded excerpts, never full table grids or image bytes in response.
            excerpt = source.source_quote
            if not excerpt:
                excerpt = (source_content.get("caption") or
                           source_content.get("visual_observation") or
                           source_content.get("passage") or "")
            adopted.append({
                "evidence_id": eid,
                "source_element_id": source.source_element_id,
                "kind": source.kind,
                "description": str(source.description or "")[:500],
                "source_excerpt": str(excerpt or "")[:800],
                "source_anchor": _anchor(source),
                "relation": edge.get("relation"),
                "relation_explanation": edge.get("explanation"),
                "is_direct_answer_source": False,
            })
        rows.append({
            "mapping_id": mapping_id,
            "target_claim_id": claim_id,
            "mapping_explanation": str(mapping.get("explanation") or "")[:6000],
            "qualifications": list(mapping.get("qualifications") or [])[:15],
            "material_status": mapping.get("material_status"),
            "claim": {
                "claim_id": claim_id,
                "text": str(claim.source_quote or claim.model_quote or "")[:1500],
                "source_element_id": claim.source_element_id,
                "source_anchor": _anchor(claim),
                "is_direct_answer_source": False,
            },
            "adopted_evidence": adopted[:24],
        })
    return {
        "schema_version": "reader.qa.mapping_trace.v1",
        "direct_mapping_ids": [row["mapping_id"] for row in rows],
        "mappings": rows,
        "note": "Nested Claim/Evidence records explain saved Mapping, not additional direct answer citations.",
    }
