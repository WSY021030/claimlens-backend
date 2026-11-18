"""Bounded retrieval of canonical content from an explicitly approved paper scope.

Only requested sections (plus descendants) can supply new answer text.
Mapping remains a derived layer, not primary evidence.
"""
from __future__ import annotations

import re
from copy import deepcopy

from app.core.ask_ai_scope import (
    POLICY, MAX_PACKET_BYTES, add_bounded_source,
    evidence_view, packet_size, saved_mapping_view,
)
from app.core.ask_ai_source_plan import layers_for_kind


def approved_sections(paper, proposal: dict) -> set[str]:
    known = {s.section_id for s in paper.sections}
    if proposal.get("mode") == "whole_paper":
        return known
    selected = set(proposal.get("section_ids") or [])
    if proposal.get("mode") != "sections" or not selected or not selected <= known:
        raise ValueError("The saved approved section IDs are invalid.")
    permitted = set(selected)
    while True:
        grown = permitted | {s.section_id for s in paper.sections
                             if s.parent_section_id in permitted}
        if grown == permitted:
            return permitted
        permitted = grown


def source_text(element):
    raw = element.model_dump(mode="json")
    if element.kind == "paragraph":
        return {"text": raw.get("text") or ""}
    if element.kind == "figure":
        return {"caption": raw.get("caption") or "",
                "alt_text": raw.get("alt_text") or "",
                "note": "A caption is not a verified reading of image pixels."}
    if element.kind == "table":
        return {"caption": raw.get("caption") or "",
                "grid": raw.get("grid") or {},
                "note": "Standardized source cells, not independently verified visual pixels."}
    if element.kind == "equation":
        return {"latex": raw.get("latex"), "plain_text": raw.get("plain_text")}
    return None


# Ignore question boilerplate so a large approved scope prioritizes
# the requested scientific topic rather than "what", "paper" or "results".
QUERY_STOPWORDS = frozenset("""
    the and for from that this those these with without what which when where
    whose why how does did do was were are is be been being can could should
    would may might into about above below under over here there their its
    his her our your any all some more most less very much paper article
    authors author study studies task tasks experiment experiments result
    results finding findings claim claims compare comparison compared between
    show shows showing explain please evidence provide provided reported
    reporting using use used based versus vs different same related
""".split())


def relevance(question: str, text: str) -> int:
    body = text.casefold()
    terms = {
        term for term in re.findall(r"[a-z0-9][a-z0-9_-]{2,}",
                                   question.casefold())
        if term not in QUERY_STOPWORDS
    }
    # Overlapping CJK phrases also work for questions written in Chinese.
    for phrase in re.findall(r"[\u4e00-\u9fff]+", question):
        terms.update(phrase[i:i + 2] for i in range(len(phrase) - 1))
    return sum((2 if len(term) >= 6 else 1) * min(body.count(term), 3)
               for term in terms if term)


def build_approved_scope_packet(*, paper, data, selection_packet: dict,
                                proposal: dict, question: str,
                                answer_kind: str, requested_layers=(),
                                answer_parts=()) -> dict:
    if selection_packet.get("policy") != POLICY:
        raise ValueError("Invalid original selection packet.")
    if (selection_packet.get("paper_id") != paper.paper_id
            or selection_packet.get("paper_revision") != paper.revision
            or data.paper_id != paper.paper_id
            or data.paper_revision != paper.revision):
        raise ValueError("Paper changed after selection authorization.")
    from app.core.ask_ai_source_plan import answer_modes
    modes = answer_modes(answer_kind, answer_parts, requested_layers)
    layers = layers_for_kind(answer_kind, requested_layers, answer_parts)
    sections = approved_sections(paper, proposal)
    selected_elements = [e for e in paper.elements.values() if e.section_id in sections]
    element_ids = {e.element_id for e in selected_elements}
    section_titles = {s.section_id: s.title for s in paper.sections}
    packet = deepcopy(selection_packet)
    # The selection was read for scope planning. The final answer gets only
    # location metadata and extracted records, never raw paragraph text.
    packet["selection"] = [
        {k: v for k, v in span.items() if k != "text"}
        for span in packet["selection"]
    ]
    original_claims = deepcopy(packet["claims"])
    original_evidence = deepcopy(packet["evidence"])
    original_mappings = deepcopy(packet["mappings"])
    packet["authorized_text"] = []
    packet["claims"] = {}
    packet["evidence"] = {}
    packet["mappings"] = {}
    packet["evidence_candidates_by_claim"] = {}
    packet["deferred_source_ids"] = []
    # Candidate ownership is metadata; the final response will expose only
    # actually USED sources. It must not be confused with citation usage.
    columns = ("claim_ids", "evidence_ids", "mapping_ids", "source_element_ids")
    packet["candidate_source_groups"] = {
        side: {column: [] for column in columns}
        for side in ("original_selection", "approved_scope")
    }
    original_group = packet["candidate_source_groups"]["original_selection"]
    approved_group = packet["candidate_source_groups"]["approved_scope"]
    original_group["source_element_ids"] = list(dict.fromkeys(
        [span["element_id"] for span in packet["selection"]]
        + [item["element_id"] for item in packet.get("related_source_text", [])]
    ))
    # Original figures/tables were only retrieval/navigation hints. Do not
    # let their raw content consume source budget or reach the answer model.
    packet["related_source_text"] = []

    # Keep room for both sides and the small provenance bookkeeping. Selected
    # sources have first access to a bounded share, not all 48 KB. This is
    # one common budget for every paper and every cross-topic comparison.
    content_limit = MAX_PACKET_BYTES - 3000
    original_limit = min(content_limit - 10000, 22000)

    def add_grouped(collection: str, source_id: str, value: dict,
                    group: dict, *, limit: int = content_limit) -> bool:
        already = source_id in packet[collection]
        if not add_bounded_source(packet, collection, source_id, value):
            return False
        if packet_size(packet) > limit:
            if not already:
                del packet[collection][source_id]
                packet["deferred_source_ids"].append(source_id)
            return False
        column = {"claims": "claim_ids", "evidence": "evidence_ids",
                  "mappings": "mapping_ids"}[collection]
        if source_id not in group[column]:
            group[column].append(source_id)
        element_id = value.get("source_element_id")
        if element_id and element_id not in group["source_element_ids"]:
            group["source_element_ids"].append(element_id)
        return True

    if "claims" in layers:
        for claim_id, item in sorted(
            original_claims.items(),
            key=lambda pair: -relevance(question, pair[1].get("text", "")),
        )[:12]:
            add_grouped("claims", claim_id, item, original_group,
                        limit=original_limit)

    if "evidence" in layers:
        def original_rank(pair):
            item = pair[1]
            kind = item.get("kind") or item.get("source_kind")
            return (3 if kind in {"figure", "table"} else 0) + relevance(
                question, str(item.get("description") or "")
            )

        for evidence_id, item in sorted(
            original_evidence.items(), key=original_rank, reverse=True,
        )[:12]:
            add_grouped("evidence", evidence_id, item, original_group,
                        limit=original_limit)

    if "mapping" in layers:
        for mapping_id, item in list(original_mappings.items())[:6]:
            add_grouped("mappings", mapping_id, item, original_group,
                        limit=original_limit)
    packet["scope_used"] = ("approved_whole_paper" if proposal["mode"] == "whole_paper"
                            else "approved_sections")
    packet["scope_authorization"] = {
        "approved": True, "mode": proposal["mode"],
        "section_ids": list(proposal.get("section_ids") or []),
    }
    packet["source_policy"]["question_scope_expanded"] = True
    packet["source_policy"]["answer_kind"] = answer_kind
    packet["source_policy"]["answer_modes"] = list(modes)
    packet["source_policy"]["answer_parts"] = [
        p.model_dump(mode="json") if hasattr(p, "model_dump") else dict(p)
        for p in answer_parts
    ]
    packet["source_policy"]["selected_text_role"] = "routing_and_provenance_only"
    packet["source_policy"]["authorized_text_role"] = "routing_only_not_answer_material"
    packet["source_policy"]["direct_source_contract"] = "extracted_records_only_v1"
    packet["source_policy"]["visible_layers"] = sorted(layers)
    packet["source_policy"]["mapping_policy"] = "primary_mapping_with_nested_trace"

    # Approved sections supply candidate IDs for saved-pool retrieval only.
    # Their raw paragraphs, figures and table grids are NEVER answer sources.
    candidates = [element for element in selected_elements
                  if element.kind in {"paragraph", "figure", "table", "equation"}]
    deferred = []

    claims = [c for c in data.claim_pool.values() if c.source_element_id in element_ids]
    claims.sort(key=lambda c: -relevance(question, c.source_quote or c.model_quote))
    if "claims" in layers:
        for claim in claims[:16]:
            add_grouped("claims", claim.claim_id, {
                "claim_id": claim.claim_id, "type": claim.type,
                "scope": claim.scope, "source_element_id": claim.source_element_id,
                "text": claim.source_quote or claim.model_quote,
            }, approved_group)

    evidence = [e for e in data.evidence_pool.values() if e.source_element_id in element_ids]
    evidence.sort(key=lambda e: -relevance(
        question, (e.source_quote or "") + " " + e.description))

    if "mapping" in layers:
        for claim in claims[:8]:
            # A Mapping is the direct source in mapping_answer. Do not require
            # its underlying Claim to be separately added as a direct source.
            if claim.claim_id not in data.claim_pool:
                continue
            mapping = saved_mapping_view(data, claim.claim_id)
            if mapping is None:
                continue
            # Section approval does not authorize unrelated source passages.
            adopted = [data.evidence_pool.get(eid) for eid in
                       mapping["evidence_ids_used_by_mapping"]]
            if any(e is None or e.source_element_id not in element_ids
                   for e in adopted):
                continue
            if add_grouped("mappings", mapping["mapping_id"], mapping,
                           approved_group):
                # Adopted Evidence is represented only by nested mapping_trace;
                # it must not become a parallel direct citation candidate.
                pass

    if "evidence" in layers:
        for record in evidence[:16]:
            item = evidence_view(record)
            item["question_source_role"] = "approved_scope_evidence"
            add_grouped("evidence", record.evidence_id, item,
                        approved_group)


    # The two sides share an answer packet, but no record should silently
    # switch provenance. A record already admitted for the original selection
    # retains its original role even if its section overlaps the approval.
    for column in columns:
        original_ids = set(original_group[column])
        approved_group[column] = [
            value for value in dict.fromkeys(approved_group[column])
            if value not in original_ids
        ]
    packet["approved_scope_coverage"] = {
        "approved_section_ids": sorted(sections),
        "source_basis": "saved_extracted_records_only",
        "extraction_not_exhaustive": True,
        "candidate_element_count": len(candidates),
        "included_element_count": len(packet["authorized_text"]),
        "deferred_element_ids": deferred[:80],
        "deferred_element_count": len(deferred),
        "complete": (bool(packet["claims"] or packet["evidence"] or packet["mappings"])
                     and ("claims" not in layers or len(claims) <= 16)
                     and ("evidence" not in layers or len(evidence) <= 16)
                     and not packet["deferred_source_ids"]),
        "retrieval_is_exhaustive": False,
        "source_text_complete": False,
        "deferred_claim_candidate_count": (
            max(0, len(claims) - 16) if "claims" in layers else 0
        ),
        "deferred_evidence_candidate_count": (
            max(0, len(evidence) - 16) if "evidence" in layers else 0
        ),
    }
    packet["authorized_text"] = []
    packet["related_source_text"] = []
    return packet
