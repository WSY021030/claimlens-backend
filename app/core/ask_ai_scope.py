"""Selection-first question material with independent source layers.

This module reads saved Claims, Evidence and Mappings.
It never invokes an Agent, modifies analysis pools, or creates
persistent Claim-Evidence Links.

Question relevance must be confirmed before material outside
the exact selection is attached.
"""

from __future__ import annotations

from copy import deepcopy
import json

from app.core.evidence_retrieval import (
    EvidenceDataManager,
    evidence_view,
)
from app.core.mapping_views import (
    check_saved_mapping,
    presentation_analysis,
)
from app.core.request_contract import (
    SelectionReference,
    validate_selection_reference,
)


POLICY = "ask_ai_independent_sources_v1"
MAX_PACKET_BYTES = 48000
MAX_EVIDENCE_ITEMS = 24


def packet_size(value: dict) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def selection_ranges(selection: SelectionReference) -> dict:
    ranges = {}

    for span in selection.segments:
        ranges.setdefault(span.element_id, []).append(
            (span.start, span.end)
        )

    return ranges


def record_span(record):
    if getattr(record, "source_status", None) != "located":
        return None

    anchor = getattr(record, "source_anchor", None)

    if anchor is None:
        return None

    selector = anchor.selector

    if getattr(selector, "type", None) != "text_span":
        return None

    return (
        anchor.element_id,
        selector.start,
        selector.end,
    )


def fully_selected(record, ranges: dict) -> bool:
    location = record_span(record)

    if location is None:
        return False

    element_id, start, end = location

    return any(
        selected_start <= start
        and end <= selected_end
        for selected_start, selected_end
        in ranges.get(element_id, [])
    )


def add_bounded_source(
    packet: dict,
    collection: str,
    source_id: str,
    value: dict,
) -> bool:
    """Add a whole record or defer it without truncating content."""

    if source_id in packet[collection]:
        return True

    if (
        collection == "evidence"
        and len(packet["evidence"]) >= MAX_EVIDENCE_ITEMS
    ):
        packet["deferred_source_ids"].append(source_id)
        return False

    packet[collection][source_id] = value

    if packet_size(packet) > MAX_PACKET_BYTES:
        del packet[collection][source_id]
        packet["deferred_source_ids"].append(source_id)
        return False

    return True


def build_selection_seed(
    *,
    paper,
    data,
    selection: SelectionReference,
    question: str,
) -> dict:
    """Read only material grounded in the exact selected spans."""

    validate_selection_reference(selection, paper)

    if (
        data.paper_id != paper.paper_id
        or data.paper_revision != paper.revision
    ):
        raise ValueError(
            "Paper and saved analysis revisions differ."
        )

    ranges = selection_ranges(selection)

    packet = {
        "entry_mode": "ask_ai",
        "policy": POLICY,
        "paper_id": paper.paper_id,
        "paper_revision": paper.revision,
        "question": question,
        "scope_used": "selection",
        "selection": [
            {
                "element_id": span.element_id,
                "start": span.start,
                "end": span.end,
                "text": span.exact,
            }
            for span in selection.segments
        ],
        "claims": {},
        "evidence": {},
        "mappings": {},
        "evidence_candidates_by_claim": {},
        "unavailable_mapping_claim_ids": [],
        "deferred_source_ids": [],
        "source_policy": {
            "claims": "author_statements",
            "evidence": "paper_source_material",
            "mappings": "derived_argument_analysis",
            "question_scope_expanded": False,
            "new_agent_calls": 0,
        },
    }

    for claim_id, claim in data.claim_pool.items():
        if not fully_selected(claim, ranges):
            continue

        packet["claims"][claim_id] = {
            "claim_id": claim_id,
            "type": claim.type,
            "scope": claim.scope,
            "source_element_id": claim.source_element_id,
            "text": claim.source_quote or claim.model_quote,
        }

    if packet_size(packet) > MAX_PACKET_BYTES:
        raise ValueError(
            "Selected text and Claims exceed the question packet limit."
        )

    for evidence_id, record in data.evidence_pool.items():
        if not fully_selected(record, ranges):
            continue

        item = evidence_view(record)
        item["question_source_role"] = "selected_text_evidence"

        add_bounded_source(
            packet,
            "evidence",
            evidence_id,
            item,
        )

    return packet


def attach_direct_selection_references(*, paper, data, packet: dict) -> dict:
    """Reuse canonical objects explicitly referenced by selected paragraphs."""
    result = deepcopy(packet)
    selected = {}
    for span in result["selection"]:
        selected.setdefault(span["element_id"], []).append(
            (span["start"], span["end"])
        )

    def is_selected_mention(mention) -> bool:
        anchor = mention.source_anchor
        if anchor is None or anchor.element_id not in selected:
            return False
        selector = anchor.selector
        if getattr(selector, "type", None) == "text_span":
            return any(start <= selector.start and selector.end <= end
                       for start, end in selected[anchor.element_id])
        # With no exact mention position, only whole-paragraph selection
        # authorizes its reference; a partial span does not.
        element = paper.elements.get(anchor.element_id)
        text = getattr(element, "text", None)
        return bool(text) and any(
            start == 0 and end == len(text)
            for start, end in selected[anchor.element_id]
        )

    targets = list(dict.fromkeys(
        mention.target_id
        for mention in paper.mentions
        if is_selected_mention(mention)
        and mention.target_id in paper.elements
        and paper.elements[mention.target_id].kind in {"figure", "table", "equation"}
    ))
    result["related_source_text"] = []
    result["deferred_related_element_ids"] = []
    for target_id in targets:
        element = paper.elements[target_id]
        raw = element.model_dump(mode="json")
        if element.kind == "figure":
            content = {"caption": raw.get("caption"),
                       "alt_text": raw.get("alt_text"),
                       "note": "Caption/alt text do not verify image pixels."}
        elif element.kind == "table":
            content = {"caption": raw.get("caption"), "grid": raw.get("grid"),
                       "note": "Only standardized cells; verify labels/row groups."}
        else:
            content = {"latex": raw.get("latex"),
                       "plain_text": raw.get("plain_text")}
        result["related_source_text"].append({
            "element_id": target_id, "kind": element.kind, "content": content,
            "source_role": "direct_selection_reference",
        })
        if packet_size(result) > MAX_PACKET_BYTES:
            result["related_source_text"].pop()
            result["deferred_related_element_ids"].append(target_id)
    for record in data.evidence_pool.values():
        if (record.source_element_id not in targets
                or record.source_status != "located"):
            continue
        view = evidence_view(record)
        view["question_source_role"] = "direct_selection_reference"
        add_bounded_source(result, "evidence", record.evidence_id, view)
    return result


def saved_mapping_view(data, claim_id: str) -> dict | None:
    """Read a validated, single-Claim Mapping without reanalysis."""

    mapping_id = data.mapping_by_claim.get(claim_id)

    if not mapping_id:
        return None

    record = data.mappings.get(mapping_id)

    if not isinstance(record, dict):
        return None

    if (
        record.get("claim_id") != claim_id
        or record.get("target_claim_ids") != [claim_id]
        or record.get("paper_revision") != data.paper_revision
    ):
        return None

    validation = check_saved_mapping(
        data,
        claim_id,
        mapping_id,
    )

    if not validation["usable"]:
        return None

    row = presentation_analysis(
        data,
        claim_id,
        record,
    )

    return {
        "mapping_id": mapping_id,
        "claim_id": claim_id,
        "source_type": "derived_mapping",
        "analysis_focus": record.get("analysis_focus"),
        "explanation": row.get("explanation", ""),
        "qualifications": list(
            row.get("qualifications") or []
        ),
        "material_status": row.get(
            "material_status", "limited"
        ),
        "evidence_ids_used_by_mapping": list(
            row.get("evidence_ids_used") or []
        ),
        "evidence_relations": deepcopy(
            row.get("evidence_relations") or []
        ),
        "grounding_validation": deepcopy(
            row.get("grounding_validation")
        ),
        "presentation_review_notes": list(
            row.get("presentation_review_notes") or []
        ),
    }


def attach_saved_claim_sources(
    *,
    paper,
    data,
    selection_packet: dict,
    claim_ids: list[str],
    include_evidence: bool = False,
    include_mapping: bool = False,
    question_scope_confirmed: bool = False,
) -> dict:
    """Attach separately requested saved Evidence and/or Mapping.

    The caller must first establish that the user's question
    is about the selected material. This confirmation concerns
    question scope, not a new user permission dialog.

    Neither source type triggers a new Agent operation.
    """

    if selection_packet.get("policy") != POLICY:
        raise ValueError("Invalid Ask AI source packet.")

    if not question_scope_confirmed:
        raise ValueError(
            "Question relevance must be confirmed first."
        )

    if (
        selection_packet.get("paper_id") != paper.paper_id
        or selection_packet.get("paper_revision") != paper.revision
        or data.paper_id != paper.paper_id
        or data.paper_revision != paper.revision
    ):
        raise ValueError("Question packet revision mismatch.")

    packet = deepcopy(selection_packet)
    allowed_claim_ids = set(packet["claims"])
    targets = list(dict.fromkeys(claim_ids))

    if not set(targets) <= allowed_claim_ids:
        raise ValueError(
            "Question sources must belong to selected Claims."
        )

    manager = (
        EvidenceDataManager(data, paper)
        if include_evidence and targets
        else None
    )

    for claim_id in targets:
        if include_evidence:
            material, receipt = manager.packet(
                claim_id,
                analysis_focus="general",
            )

            accepted = []

            for item in material["evidence_items"]:
                evidence_id = item["evidence_id"]
                candidate = deepcopy(item)
                candidate["question_source_role"] = (
                    "selected_claim_related_evidence"
                )

                if add_bounded_source(
                    packet,
                    "evidence",
                    evidence_id,
                    candidate,
                ):
                    accepted.append(evidence_id)

            packet["evidence_candidates_by_claim"][
                claim_id
            ] = accepted

            packet["deferred_source_ids"].extend(
                receipt.get("deferred_for_size") or []
            )

        if include_mapping:
            mapping = saved_mapping_view(
                data,
                claim_id,
            )

            if mapping is None:
                packet["unavailable_mapping_claim_ids"].append(
                    claim_id
                )
            else:
                if add_bounded_source(
                    packet, "mappings", mapping["mapping_id"], mapping,
                ):
                    # Adopted Evidence remains distinct from the Mapping itself.
                    # Include only those exact sources, not the full Evidence pool.
                    for evidence_id in mapping["evidence_ids_used_by_mapping"]:
                        record = data.evidence_pool.get(evidence_id)
                        if record is None or record.source_status != "located":
                            continue
                        item = evidence_view(record)
                        item["question_source_role"] = "mapping_adopted_evidence"
                        add_bounded_source(packet, "evidence", evidence_id, item)

    packet["deferred_source_ids"] = list(
        dict.fromkeys(packet["deferred_source_ids"])
    )

    # The analysis target remains the selected text.
    # External supporting Evidence does not expand question scope.
    packet["scope_used"] = "selection"

    packet["source_policy"][
        "question_scope_expanded"
    ] = False

    return packet