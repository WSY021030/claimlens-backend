"""One primary answer-source route per Ask AI question part.

Scope permission is separate from answer mode. A Claim, Evidence, or Mapping is
not a new permission boundary. An answer can use text as routing context without
citing that text as a second direct source layer.
"""
from __future__ import annotations

from copy import deepcopy

from app.core.ask_ai_scope import (
    add_bounded_source,
    attach_direct_selection_references,
    attach_saved_claim_sources,
)
from app.core.evidence_retrieval import evidence_view, same_claim_source

PRIMARY_MODES = frozenset({
    "claim_answer", "evidence_answer", "mapping_answer",
})

MODE_LAYERS = {
    "claim_answer": frozenset({"claims"}),
    "evidence_answer": frozenset({"evidence"}),
    "mapping_answer": frozenset({"mapping"}),
}

LEGACY_MODES = {
    "explain_selection": "claim_answer",
    "selection_answer": "claim_answer",
    "report_claims": "claim_answer",
    "report_evidence": "evidence_answer",
    "evaluate_argument": "evidence_answer",
}


def normalize_mode(kind: str, requested_layers=()) -> str:
    """Permit already-pending legacy workflows to resume under the new contract."""
    if kind == "evaluate_argument" and "mapping" in requested_layers:
        return "mapping_answer"
    return LEGACY_MODES.get(kind, kind)


def answer_modes(answer_kind: str, answer_parts=(), requested_layers=()) -> tuple[str, ...]:
    kind = normalize_mode(answer_kind, requested_layers)
    if kind != "composite_answer":
        if kind not in PRIMARY_MODES:
            raise ValueError(f"Invalid Ask AI answer mode: {kind}")
        return (kind,)
    modes = []
    for part in answer_parts:
        mode = part.get("answer_mode") if isinstance(part, dict) else getattr(part, "answer_mode", None)
        mode = normalize_mode(mode or "")
        if mode not in PRIMARY_MODES:
            raise ValueError(f"Invalid composite question part: {mode}")
        if mode not in modes:
            modes.append(mode)
    if len(modes) < 2:
        raise ValueError("Composite Ask AI requires at least two different source modes.")
    return tuple(modes)


def layers_for_kind(answer_kind: str, requested_layers=(), answer_parts=()) -> frozenset[str]:
    """Only routing classification grants a direct source layer.

    Legacy requested_layers are read for saved-workflow compatibility only.
    Newly created decisions do not use free-form layers to combine sources.
    """
    modes = answer_modes(answer_kind, answer_parts, requested_layers)
    return frozenset().union(*(MODE_LAYERS[mode] for mode in modes))




def _supplement_same_claim_scan_evidence(*, data, packet: dict,
                                         claim_ids: list[str]) -> dict:
    """Use material already discovered for the SAME target Claims.

    Mapping uses a ranked, narrow argument packet. An experimental-results
    answer must not silently lose saved observations solely because its
    wording scores below a lexical cutoff. The saved Evidence RUN receipt is
    a provenance/eligibility boundary, not a permanent Claim-Evidence Link and
    not a guarantee that the extraction model judged relevance correctly.

    The existing shared Answer size/item cap remains authoritative; no new
    model, scan, index or persistent association is introduced here. Search
    receipts live in current state even after the historical run list is
    compacted, and their run_id matches each saved Evidence's created_run_id.
    """
    targets = {cid for cid in claim_ids if cid in data.claim_pool}
    eligible_run_ids: set[str] = set()
    partial_runs: set[str] = set()
    # The current-state store prunes older `runs` on subsequent questions.
    # Evidence search receipts, not historical run logs, are the durable
    # Claim-scoped discovery state and survive this compaction.
    receipts = data.evidence_search_by_claim or {}
    for claim_id in targets:
        receipt = receipts.get(claim_id) or {}
        if (receipt.get("paper_revision") != data.paper_revision
                or receipt.get("status") not in {"completed", "partial"}):
            continue
        run_id = receipt.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            continue
        eligible_run_ids.add(run_id)
        if receipt.get("status") == "partial":
            partial_runs.add(run_id)

    inspected = 0
    added = 0
    for evidence_id, record in sorted(data.evidence_pool.items()):
        if record.created_run_id not in eligible_run_ids:
            continue
        if record.source_status != "located" or record.source_anchor is None:
            continue
        if record.kind == "paragraph" and not (
                record.source_quote or (record.source_content or {}).get("passage")):
            continue
        # A selected Claim sentence is not independent experimental Evidence.
        if any(same_claim_source(data.claim_pool[cid], record) for cid in targets):
            continue
        inspected += 1
        if evidence_id in packet["evidence"]:
            continue
        view = evidence_view(record)
        view["question_source_role"] = "same_target_claim_scan"
        if add_bounded_source(packet, "evidence", evidence_id, view):
            added += 1
            # A source previously deferred from the smaller Mapping packet is
            # no longer deferred when it fits in the Answer packet.
            packet["deferred_source_ids"] = [
                sid for sid in packet["deferred_source_ids"] if sid != evidence_id
            ]

    packet["retrieval_coverage"] = {
        "source": "saved_same_target_claim_evidence_runs",
        "candidate_records": inspected,
        "additional_records_included": added,
        "direct_records_available": len(packet["evidence"]),
        "deferred_source_ids": list(dict.fromkeys(
            packet.get("deferred_source_ids") or []
        )),
        "complete_semantic_recall_guaranteed": False,
        "has_partial_claim_scans": bool(partial_runs),
        "meaning": "Source availability, not proof or independent verification.",
    }
    return packet

def selection_answer_packet(*, paper, data, selection_seed: dict,
                            claim_ids: list[str], answer_kind: str,
                            requested_layers=(), answer_parts=()) -> dict:
    """Expose only direct sources for the selected mode(s).

    Mapping's adopted Evidence stays in the persistent pool, not the answer
    candidate Evidence layer. Its trace is separately projected on output.
    """
    modes = answer_modes(answer_kind, answer_parts, requested_layers)
    layers = layers_for_kind(answer_kind, requested_layers, answer_parts)
    targets = list(dict.fromkeys(claim_ids))
    if not set(targets) <= set(selection_seed["claims"]):
        raise ValueError("Ask AI requested a Claim outside its selection.")

    packet = deepcopy(selection_seed)
    # The Coordinator already used the selected text to authorize the topic.
    # Answer generation receives only locators, never raw selected paragraphs.
    packet["selection"] = [
        {k: v for k, v in span.items() if k != "text"}
        for span in packet["selection"]
    ]
    packet["claims"] = {}
    packet["evidence"] = {}
    packet["mappings"] = {}
    packet["evidence_candidates_by_claim"] = {}
    packet["unavailable_mapping_claim_ids"] = []
    packet["deferred_source_ids"] = []
    packet["source_policy"]["answer_kind"] = (
        normalize_mode(answer_kind, requested_layers)
    )
    packet["source_policy"]["answer_modes"] = list(modes)
    packet["source_policy"]["answer_parts"] = [
        part.model_dump(mode="json") if hasattr(part, "model_dump") else dict(part)
        for part in answer_parts
    ]
    packet["source_policy"]["visible_layers"] = sorted(layers)
    packet["source_policy"]["mapping_policy"] = "primary_mapping_with_nested_trace"
    packet["source_policy"]["direct_source_contract"] = "extracted_records_only_v1"
    packet["source_policy"]["origin_evidence_rule"] = (
        "An author statement establishes attribution, not independent verification."
    )
    # The target Claim defines the question's semantic subject, not an extra
    # citable source layer. Never elevate this context into packet["claims"].
    if "evidence" in layers:
        packet["target_claim_context"] = [
            {
                "text": selection_seed["claims"][cid]["text"],
                "role": "author_statement_for_consistency_only_not_evidence",
            }
            for cid in targets
        ]
        packet["source_policy"]["target_claim_context_role"] = (
            "Topic/direction cross-check only; do not cite these Claim IDs "
            "in an evidence_answer. Evidence may contradict the Claim."
        )

    if "claims" in layers:
        packet["claims"] = {
            cid: deepcopy(selection_seed["claims"][cid]) for cid in targets
        }
    if "evidence" in layers:
        # Selection-local Evidence is valid in Evidence mode without Claim routing.
        packet["evidence"] = deepcopy(selection_seed["evidence"])
        packet = attach_direct_selection_references(
            paper=paper, data=data, packet=packet,
        )
        # Reference targets help find extracted Evidence. Their raw captions,
        # images and table grids do not belong in the answer model packet.
        packet["related_source_text"] = []
        if targets:
            staged = deepcopy(packet)
            staged["claims"] = {
                cid: deepcopy(selection_seed["claims"][cid]) for cid in targets
            }
            staged = attach_saved_claim_sources(
                paper=paper, data=data, selection_packet=staged,
                claim_ids=targets, include_evidence=True,
                question_scope_confirmed=True,
            )
            packet["evidence"] = staged["evidence"]
            packet["evidence_candidates_by_claim"] = staged["evidence_candidates_by_claim"]
            packet["deferred_source_ids"] = staged["deferred_source_ids"]
            # Answer-mode coverage is not Mapping-mode relevance ranking.
            # Preserve other source records from those Claim-focused scans
            # when the existing packet budget allows them.
            packet = _supplement_same_claim_scan_evidence(
                data=data, packet=packet, claim_ids=targets,
            )
    if "mapping" in layers and targets:
        staged = deepcopy(packet)
        staged["claims"] = {
            cid: deepcopy(selection_seed["claims"][cid]) for cid in targets
        }
        staged = attach_saved_claim_sources(
            paper=paper, data=data, selection_packet=staged,
            claim_ids=targets, include_mapping=True,
            question_scope_confirmed=True,
        )
        packet["mappings"] = staged["mappings"]
        packet["unavailable_mapping_claim_ids"] = staged["unavailable_mapping_claim_ids"]
        packet["deferred_source_ids"] = list(dict.fromkeys(
            packet["deferred_source_ids"] + staged["deferred_source_ids"]
        ))
        # Adopted Evidence is NOT an independent direct answer source here.
    # Raw selection/figure/table text may locate sources during retrieval,
    # but must not become a separate answer source. Keep provenance metadata.
    packet["related_source_text"] = []
    packet["authorized_text"] = []
    return packet
