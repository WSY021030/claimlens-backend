"""Selection-first Ask AI question execution.

This workflow uses saved source layers independently.
It does not call Claim, Evidence, or Mapping Agents.
It does not expand the user's question scope without approval.
"""

from __future__ import annotations

import os
import re
from time import perf_counter
from uuid import uuid4

from app.agents.coordinator_agent import CoordinatorAgent
from app.budget import (
    PaperBudget,
    QUESTION_BUDGET_ENV,
    QUESTION_LIMIT_ENV,
)
from app.coordinator_runner import (
    POLICY_VERSION,
    MODEL_ERRORS,
    error_message,
    now,
    save_outputs,
)
from app.core.ask_ai_expansion import build_approved_scope_packet
from app.core.ask_ai_source_provenance import used_source_provenance
from app.ask_ai_material import enrich_selected_material
from app.core.ask_ai_source_plan import (
    layers_for_kind, selection_answer_packet, answer_modes, normalize_mode,
)
from app.core.ask_ai_scope import saved_mapping_view
from app.core.ask_ai_mapping_trace import build_mapping_trace
from app.core.ask_ai_scope import (
    build_selection_seed,
    attach_saved_claim_sources,
    attach_direct_selection_references,
)
from app.core.request_contract import ReaderRequest
from app.core.token_report import (
    task_token_report,
    print_task_tokens,
)
from app.usage_cost import CostTracker


SOURCE_ID = re.compile(
    r"\b(?:claim|evidence|mapping)_\d{6}\b"
)


def paper_outline(paper) -> dict:
    """Expose document structure, not unapproved paper text."""

    return {
        "paper_id": paper.paper_id,
        "title": paper.metadata.title,
        "sections": [
            {
                "section_id": section.section_id,
                "title": section.title,
            }
            for section in paper.sections
        ],
        "role": "location_hints_only_not_answer_sources",
    }


def proposed_scope(decision, paper, *, reason=None,
                   selected_section_ids=(), question="") -> dict:
    """Propose narrow known section IDs without exposing answer material."""

    known = {
        section.section_id: section.title
        for section in paper.sections
    }

    selected_sections = set(selected_section_ids)
    chosen = list(dict.fromkeys(
        section_id
        for section_id in decision.suggested_section_ids
        if section_id in known and section_id not in selected_sections
    ))[:3]

    # When a planner recognizes a new topic but omits section IDs, titles can
    # still provide a conservative LOCATION HINT. This never fetches text or
    # automatically approves the scope. Without a clear match, request
    # whole-paper authorization instead of pretending a section was verified.
    if not chosen and question:
        from app.core.ask_ai_expansion import relevance
        ranked = sorted(
            ((relevance(question, title), section_id)
             for section_id, title in known.items()
             if section_id not in selected_sections),
            reverse=True,
        )
        best_score = ranked[0][0] if ranked else 0
        if best_score > 0:
            chosen = [section_id for score, section_id in ranked
                      if score == best_score][:2]

    if chosen:
        # When both a parent and its descendant are proposed, retain the
        # smaller authorized scope rather than all the parent's siblings.
        parents = {s.section_id: s.parent_section_id for s in paper.sections}

        def is_ancestor(ancestor: str, child: str) -> bool:
            current = parents.get(child)
            for _ in range(len(parents)):
                if current is None or current not in parents:
                    return False
                if current == ancestor:
                    return True
                current = parents[current]
            return False

        chosen = [section_id for section_id in chosen
                  if not any(is_ancestor(section_id, other)
                             for other in chosen if other != section_id)]
        return {
            "mode": "sections",
            "section_ids": chosen,
            "section_titles": [
                known[section_id]
                for section_id in chosen
            ],
            "approved": False,
            "reason": reason or decision.reason,
        }

    return {
        "mode": "whole_paper",
        "section_ids": [],
        "section_titles": [],
        "approved": False,
        "reason": reason or decision.reason,
    }

_INFORMATION_NEED_MODES = {
    "author_assertions": "claim_answer",
    "reported_findings": "evidence_answer",
    "argument_assessment": "mapping_answer",
    "multiple_operations": "composite_answer",
}


def align_information_need(decision):
    """Canonicalize a model-produced route before applying strict validation.

    Topic permission and Claim targets are never changed here. The requested
    information need is the authoritative primary mode. Parts represent
    independent, differently sourced subquestions ONLY for a composite route;
    two Claims assessed separately under mapping_answer are still one mode.
    """
    expected = _INFORMATION_NEED_MODES[decision.information_need]
    original = decision.answer_kind
    updates = {}
    if original != expected:
        updates["answer_kind"] = expected
    if expected != "composite_answer" and decision.answer_parts:
        # The model may include per-Claim parts even after choosing one mode.
        # Drop this incompatible presentation field, not the real Claim IDs.
        updates["answer_parts"] = []
    if decision.requested_layers:
        # Legacy hints cannot override the canonical source route.
        updates["requested_layers"] = []
    if updates:
        decision = decision.model_copy(update=updates)
    return decision, original


def validate_decision(decision, packet: dict) -> list[str]:
    """Reject invalid target bindings and malformed composite routing."""
    modes = answer_modes(decision.answer_kind, decision.answer_parts)
    if decision.answer_kind == "composite_answer":
        for part in decision.answer_parts:
            if not part.question_part.strip():
                raise ValueError("Composite subquestion cannot be blank.")
            if not set(part.selected_claim_ids) <= set(packet["claims"]):
                raise ValueError("Composite subquestion contains outside Claims.")
            if not set(part.selected_claim_ids) <= set(decision.selected_claim_ids):
                raise ValueError("Composite Claim targets must be declared globally.")
    elif decision.answer_parts:
        raise ValueError("Single-mode question must not contain answer_parts.")
    if "mapping_answer" in modes and not decision.selected_claim_ids:
        # Material enrichment can extract new selected Claims; do not invent IDs.
        pass

    selected = list(dict.fromkeys(
        decision.selected_claim_ids
    ))

    unknown = set(selected) - set(packet["claims"])

    if unknown:
        raise ValueError(
            "Ask AI selected Claims outside the text selection: "
            + ", ".join(sorted(unknown))
        )

    return selected


def validate_answer_sources(reply, packet: dict) -> dict:
    """Validate model-reported sources against supplied records."""

    modes = set(packet.get("source_policy", {}).get("answer_modes") or [])
    claims = list(dict.fromkeys(reply.used_claim_ids))
    evidence = list(dict.fromkeys(reply.used_evidence_ids))
    mappings = list(dict.fromkeys(reply.used_mapping_ids))
    if claims and "claim_answer" not in modes:
        raise ValueError("Ask AI cited Claims outside its direct answer mode.")
    if evidence and "evidence_answer" not in modes:
        raise ValueError("Ask AI cited Evidence outside its direct answer mode.")
    if mappings and "mapping_answer" not in modes:
        raise ValueError("Ask AI cited Mapping outside its direct answer mode.")
    elements = list(dict.fromkeys(
        reply.used_source_element_ids
    ))

    if not set(claims) <= set(packet["claims"]):
        raise ValueError("Ask AI cited unavailable Claims.")

    if not set(evidence) <= set(packet["evidence"]):
        raise ValueError("Ask AI cited unavailable Evidence.")

    if not set(mappings) <= set(packet["mappings"]):
        raise ValueError("Ask AI cited unavailable Mappings.")

    allowed_elements = {
        item["element_id"]
        for item in packet["selection"]
    }

    allowed_elements.update(
        item["source_element_id"]
        for item in packet["evidence"].values()
        if item.get("source_element_id")
    )

    allowed_elements.update(
        item["source_element_id"]
        for item in packet["claims"].values()
    )

    allowed_elements.update(
        item["element_id"]
        for item in packet.get("authorized_text", [])
    )
    allowed_elements.update(
        item["element_id"]
        for item in packet.get("related_source_text", [])
    )

    if not set(elements) <= allowed_elements:
        raise ValueError(
            "Ask AI cited paper elements outside its source packet."
        )
    if elements:
        raise ValueError(
            "Original paragraph/figure/table IDs are provenance locators, "
            "not direct Ask AI answer sources."
        )

    if reply.status == "answered" and not (claims or evidence or mappings):
        raise ValueError("Ask AI answered without citing extracted source records.")

    declared = set(claims + evidence + mappings)
    mentioned = set(SOURCE_ID.findall(reply.answer))

    if not mentioned <= declared:
        raise ValueError(
            "Ask AI answer contains undeclared source references."
        )

    return {
        "used_claim_ids": claims,
        "used_evidence_ids": evidence,
        "used_mapping_ids": mappings,
        "used_source_element_ids": elements,
    }


def resolve_answer_route(*, requested_kind: str, answer_parts, requested_layers,
                         claim_ids, data):
    """Use saved per-Claim Mapping when available; record transparent fallback."""
    modes = answer_modes(requested_kind, answer_parts, requested_layers)
    kind = normalize_mode(requested_kind, requested_layers)
    if "mapping_answer" not in modes:
        return kind, list(answer_parts), None
    missing = [cid for cid in claim_ids if saved_mapping_view(data, cid) is None]
    if missing and len(missing) < len(claim_ids):
        # Preserve available per-Claim Mappings for a partial assessment.
        # A missing Mapping for one Claim does not invalidate another.
        return kind, list(answer_parts), (
            "Some requested Claims have no validated Mapping: "
            + ", ".join(missing)
        )
    if not claim_ids or missing:
        # Never pretend to have used a Mapping that does not exist.
        reason = (
            "No selected Claim could be matched to a validated Mapping."
            if not claim_ids else
            "Validated Mapping unavailable for: " + ", ".join(missing)
        )
        if kind == "composite_answer":
            parts = []
            for part in answer_parts:
                obj = part.model_dump(mode="json") if hasattr(part, "model_dump") else dict(part)
                if obj["answer_mode"] == "mapping_answer":
                    obj["answer_mode"] = "evidence_answer"
                parts.append(obj)
            if len(set(p["answer_mode"] for p in parts)) >= 2:
                return kind, parts, reason
            return parts[0]["answer_mode"], [], reason
        return "evidence_answer", [], reason
    return kind, list(answer_parts), None



def material_inventory(packet: dict) -> dict:
    """Summarize actual direct records without treating locators as sources."""
    kinds = {"claim_answer": "claims", "evidence_answer": "evidence",
             "mapping_answer": "mappings"}
    modes = list(packet.get("source_policy", {}).get("answer_modes") or [])
    counts = {mode: len(packet.get(collection) or {})
              for mode, collection in kinds.items()}
    return {
        "required_modes": modes,
        "candidate_counts": counts,
        "missing_modes": [mode for mode in modes if counts.get(mode, 0) == 0],
        "direct_source_count": sum(counts.get(mode, 0) for mode in modes),
        "deferred_source_count": len(packet.get("deferred_source_ids") or []),
        "scope_used": packet.get("scope_used"),
    }


def insufficient_material_answer(*, run, packet, answer_mode, language,
                                 fallback_reason=None) -> bool:
    """End groundedly if extraction produced no usable direct records."""
    inventory = material_inventory(packet)
    run["qa_material_inventory"] = inventory
    if inventory["direct_source_count"]:
        return False
    if language == "zh":
        message = (
            "当前问答没有可用于回答的已提取 Claim、Evidence 或 Mapping。"
            "已按现有覆盖记录检查和尝试必要的资料收集；"
            "不能仅凭原文或图注补造分析结论。"
        )
    else:
        message = (
            "No extracted Claim, Evidence, or Mapping is available to ground "
            "this answer. The applicable saved coverage and collection steps "
            "were checked; raw paragraphs and captions cannot substitute "
            "for extracted analysis."
        )
    missing = ["Missing direct answer sources for: " + ", ".join(
        inventory["missing_modes"])]
    if fallback_reason:
        missing.append(fallback_reason)
    run["current_phase"] = "material_gap"
    run["qa_material_gap"] = {
        "reason": "no_extracted_direct_sources",
        "missing_modes": list(inventory["missing_modes"]),
        "collection_actions": list(run.get("qa_material_actions") or [])[-15:],
    }
    run["qa_answer_mode_fallback_reason"] = fallback_reason
    set_outcome(
        run, status="unanswerable", answer=message,
        answer_mode=answer_mode, missing_information=missing,
        limitations=missing,
    )
    return True


def finish_rejected_scope_request(*, run, answer_mode: str, language: str) -> None:
    """A missing same-topic source is never a new-topic authorization request."""
    message = (
        "目前已授权的资料仍不足以回答，不能自动扩大到其他研究主题。"
        if language == "zh" else
        "The currently authorized extracted material is insufficient to answer. "
        "Ask AI will not enlarge the research topic without a separate scope decision."
    )
    reason = "Answer model requested a scope change after the scope decision."
    run["qa_material_gap"] = {
        "reason": "answer_model_scope_request_rejected",
        "collection_actions": list(run.get("qa_material_actions") or [])[-15:],
    }
    run["current_phase"] = "material_gap"
    set_outcome(
        run, status="unanswerable", answer=message,
        answer_mode=answer_mode,
        limitations=[reason], missing_information=[reason],
    )


def set_outcome(
    run: dict,
    *,
    status: str,
    answer: str,
    sources: dict | None = None,
    answer_mode: str | None = None,
    limitations: list[str] | None = None,
    missing_information: list[str] | None = None,
) -> None:
    """Store response status separately from processing success."""

    sources = sources or {
        "used_claim_ids": [],
        "used_evidence_ids": [],
        "used_mapping_ids": [],
        "used_source_element_ids": [],
    }

    all_ids = list(dict.fromkeys(
        sources["used_claim_ids"]
        + sources["used_evidence_ids"]
        + sources["used_mapping_ids"]
        + sources["used_source_element_ids"]
    ))

    run["qa_outcome"] = status
    if answer_mode is not None:
        run["qa_answer_mode"] = answer_mode
    run.update(sources)
    run["used_mapping_ids"] = sources["used_mapping_ids"]
    run["answer_material_ids"] = all_ids

    run["answer"] = {
        "answer": answer,
        "answer_mode": run.get("qa_answer_mode"),
        "used_source_ids": all_ids,
        "used_claim_ids": sources["used_claim_ids"],
        "used_evidence_ids": sources["used_evidence_ids"],
        "used_mapping_ids": sources["used_mapping_ids"],
        "used_source_element_ids": sources[
            "used_source_element_ids"
        ],
        "limitations": limitations or [],
    }

    run["missing_information"] = (
        missing_information or []
    )

    run["answer_available"] = True

    if status == "needs_scope_approval":
        run["status"] = "partial"
        run["answer_status"] = "needs_scope_approval"
    else:
        run["status"] = "completed"
        run["answer_status"] = "generated"


def saved_evidence_scan_coverage(data, claim_ids):
    """Report persisted scan receipts; do not infer completeness from an answer.

    A completed answer can use adequate saved evidence even if discovery left
    unrelated source windows pending. Such coverage must remain visible rather
    than silently becoming an unqualified completed scan.
    """
    targets = list(dict.fromkeys(claim_ids))
    if not targets:
        return None
    receipts = data.evidence_search_by_claim
    detail = {}
    for cid in targets:
        saved = receipts.get(cid) or {}
        detail[cid] = {
            "status": saved.get("status", "not_recorded"),
            "text_scan_complete": saved.get("text_scan_complete"),
            "image_scan_complete": saved.get("image_scan_complete"),
            "run_id": saved.get("run_id"),
        }
    incomplete = [
        cid for cid, row in detail.items()
        if row["status"] != "completed"
        or row["text_scan_complete"] is not True
        or row["image_scan_complete"] is not True
    ]
    evidence_run_ids = {row["run_id"] for row in detail.values() if row["run_id"]}
    saved_runs = [
        item for item in data.runs
        if item.get("stage") == "evidence" and item.get("run_id") in evidence_run_ids
    ]
    pending = {
        (str(item.get("element_id")), str(item.get("scan_id")))
        for evidence_run in saved_runs
        for item in evidence_run.get("pending_text_source_windows", [])
        if isinstance(item, dict)
    }
    return {
        "policy": "saved_claim_evidence_scan_receipts_v1",
        "by_claim": detail,
        "incomplete_claim_ids": incomplete,
        "all_saved_scans_completed": not incomplete,
        "pending_text_source_window_count": (len(pending) if saved_runs else None),
        "note": "A source scan receipt is not a proof of semantic evidence completeness.",
    }


def save_grounded_ask_ai_answer(*, run, reply, packet, paper, data,
                                answer_mode, fallback_reason=None):
    sources = validate_answer_sources(reply, packet)
    run["answer_source_provenance"] = used_source_provenance(
        sources=sources, packet=packet, paper=paper, data=data,
    )
    run["answer_mapping_trace"] = build_mapping_trace(
        data=data, used_mapping_ids=sources["used_mapping_ids"],
    ) if sources["used_mapping_ids"] else None
    run["qa_answer_mode_fallback_reason"] = fallback_reason
    limitations = list(reply.limitations)
    if answer_mode in {"evidence_answer", "mapping_answer", "composite_answer"}:
        selected_claims = (
            (run.get("source_plan") or {}).get("selected_claim_ids")
            or (run.get("selection_decision") or {}).get("selected_claim_ids")
            or []
        )
        coverage = saved_evidence_scan_coverage(data, selected_claims)
        if coverage is not None:
            inventory = run.get("qa_material_inventory") or material_inventory(packet)
            run["qa_material_inventory"] = {
                **inventory, "evidence_scan_coverage": coverage,
            }
            if coverage["incomplete_claim_ids"]:
                limitations.append(
                    "Evidence discovery is not fully complete for selected Claims: "
                    + ", ".join(coverage["incomplete_claim_ids"])
                    + ". The answer uses available saved material; unprocessed "
                    "sources have not been ruled out as relevant."
                )
    set_outcome(
        run, status=reply.status, answer=reply.answer, sources=sources,
        answer_mode=answer_mode,
        limitations=list(dict.fromkeys(limitations)),
        missing_information=reply.missing_information,
    )


def run_ask_ai(
    *,
    args,
    paper,
    store,
    data,
    contract: dict,
    old_run: dict | None = None,
) -> None:
    """Run one bounded Ask AI attempt with the existing budget ledger."""

    request = ReaderRequest.model_validate(
        contract["request"]
    )

    if request.resolved_entry_mode() != "ask_ai":
        raise ValueError("Expected an Ask AI request.")

    if request.selection_context is None:
        raise ValueError("Ask AI requires selected text.")

    if old_run is not None:
        if old_run.get("mode") != "ask_ai":
            raise ValueError(
                "This workflow is not an Ask AI question."
            )

        if old_run.get("paper_revision") != paper.revision:
            raise ValueError("Paper revision changed.")

        if old_run.get("status") == "completed":
            print(
                "Ask AI question already completed:",
                old_run["run_id"],
                flush=True,
            )
            return

        pending = old_run.get("qa_outcome") == "needs_scope_approval"
        already_approved = bool((old_run.get("scope_approval") or {}).get("approved"))
        if already_approved and getattr(args, "reject_scope", False):
            raise ValueError("An already approved scope cannot be rejected.")

        if pending and getattr(args, "reject_scope", False):
            old_run["scope_decision"] = "rejected"
            set_outcome(
                old_run,
                status="scope_declined",
                answer=(
                    "The proposed scope expansion was declined."
                ),
            )
            old_run["finished_at"] = now()
            store.save(data)
            print("Workflow ID:", old_run["run_id"], flush=True)
            print("Status: completed", flush=True)
            print("Answer status: generated", flush=True)
            return

        if pending and not (getattr(args, "approve_scope", False) or already_approved):
            print(
                "Ask AI is waiting for scope approval:",
                old_run["run_id"],
                flush=True,
            )
            return

        if getattr(args, "approve_scope", False) and not pending:
            raise ValueError(
                "The saved workflow is not awaiting scope approval."
            )

        # Reuse the original question ID and budget.
        from app.workflow_runner import resume_question_limit

        limit = resume_question_limit(
            old_run,
            args.question_limit_usd,
            store,
            data,
        )

        run = old_run
        args.model = run["model"]
        args.language = run["language"]
        run["attempt_count"] = run.get(
            "attempt_count", 1
        ) + 1

    else:
        limit = (
            0.50
            if args.question_limit_usd is None
            else args.question_limit_usd
        )

        run = {
            "run_id": uuid4().hex[:12],
            "stage": "coordinator",
            "mode": "ask_ai",
            "workflow_policy": POLICY_VERSION,
            "budget_policy": "per_question_v1",
            "paper_revision": paper.revision,
            "user_question": request.question,
            "model": args.model,
            "language": args.language,
            "image_mode": args.image_mode,
            "question_limit_usd": limit,
            "status": "running",
            "answer_status": "pending",
            "answer_available": False,
            "started_at": now(),
            "attempt_count": 1,
            "current_phase": "selection",
            "task_states": {},
            "stage_results": [],
            "issues": [],
            "answer": None,
            "used_mapping_ids": [],
        }

        run["question_budget_id"] = run["run_id"]
        data.runs.append(run)

    run["request_contract"] = contract
    run["scope_policy"] = "ask_ai_selection_first_v1"
    run["status"] = "running"
    run["answer_status"] = "pending"
    run["answer_available"] = False
    run["current_phase"] = "selection"
    run["answer_source_provenance"] = None
    run["answer_mapping_trace"] = None
    run["qa_answer_mode"] = None
    run["qa_answer_mode_fallback_reason"] = None
    run["qa_material_actions"] = []
    run["qa_material_inventory"] = None
    run["qa_material_gap"] = None

    store.save(data)
    print("Workflow ID:", run["run_id"], flush=True)
    print("Question mode: ask_ai", flush=True)

    previous_id = os.environ.get(QUESTION_BUDGET_ENV)
    previous_limit = os.environ.get(QUESTION_LIMIT_ENV)

    os.environ[QUESTION_BUDGET_ENV] = run["run_id"]
    os.environ[QUESTION_LIMIT_ENV] = str(limit)

    budget = None
    tracker = CostTracker()
    started = perf_counter()

    try:
        budget = PaperBudget(
            paper.paper_id,
            question_id=run["run_id"],
            limit_usd=limit,
        )

        coordinator = CoordinatorAgent(
            model=args.model,
            tracker=tracker,
            budget=budget,
        )

        seed = build_selection_seed(
            paper=paper,
            data=data,
            selection=request.selection_context,
            question=request.question,
        )

        if getattr(args, "approve_scope", False) or (
            old_run is not None
            and (run.get("scope_approval") or {}).get("approved")
        ):
            proposal = run.get("scope_proposal") or {}

            if (
                proposal.get("approved") not in {False, True}
                or (proposal.get("approved") is True and not (
                    run.get("scope_approval") or {}
                ).get("approved"))
                or proposal.get("mode")
                not in {"sections", "whole_paper"}
            ):
                raise ValueError("No valid pending scope proposal.")

            # A previously authorized scope remains authoritative, while
            # its saved answer-kind controls which layers become visible.
            saved_decision = run.get("selection_decision") or {}
            saved_kind = saved_decision.get("answer_kind", "evidence_answer")
            saved_layers = saved_decision.get("requested_layers") or []
            saved_parts = saved_decision.get("answer_parts") or []
            saved_kind = normalize_mode(saved_kind, saved_layers)
            route_fallback = None
            approved_layers = layers_for_kind(saved_kind, answer_parts=saved_parts)
            # Find SAVED Evidence on directly referenced figures/tables.
            # Canonical objects themselves are never direct answer sources.
            anchored_seed = attach_direct_selection_references(
                paper=paper, data=data, packet=seed,
            )
            # Reuse already-saved support for selected Claims even if a figure
            # is physically located outside the selected paragraph. That does
            # not authorize a different scientific topic.
            selected_claims = list(anchored_seed["claims"])[:12]
            if selected_claims and (
                "evidence" in approved_layers or "mapping" in approved_layers
            ):
                anchored_seed = attach_saved_claim_sources(
                    paper=paper, data=data, selection_packet=anchored_seed,
                    claim_ids=selected_claims,
                    include_evidence="evidence" in approved_layers,
                    include_mapping="mapping" in approved_layers,
                    question_scope_confirmed=True,
                )
            packet = build_approved_scope_packet(
                paper=paper,
                data=data,
                selection_packet=anchored_seed,
                proposal=proposal,
                question=request.question,
                answer_kind=saved_kind,
                answer_parts=saved_parts,
            )
            # Only now may the authorized scope supply NEW-topic Mappings.
            if "mapping" in approved_layers and not packet["mappings"]:
                route_fallback = (
                    "No validated Mapping was available in the approved "
                    "answer scope; answering from Evidence instead."
                )
                if saved_kind == "composite_answer":
                    revised = []
                    for part in saved_parts:
                        part = dict(part)
                        if part["answer_mode"] == "mapping_answer":
                            part["answer_mode"] = "evidence_answer"
                        revised.append(part)
                    if len({p["answer_mode"] for p in revised}) >= 2:
                        saved_parts = revised
                    else:
                        saved_kind, saved_parts = revised[0]["answer_mode"], []
                else:
                    saved_kind, saved_parts = "evidence_answer", []
                approved_layers = layers_for_kind(saved_kind, answer_parts=saved_parts)
                support_seed = attach_saved_claim_sources(
                    paper=paper, data=data, selection_packet=anchored_seed,
                    claim_ids=selected_claims, include_evidence=True,
                    question_scope_confirmed=True,
                ) if selected_claims else anchored_seed
                packet = build_approved_scope_packet(
                    paper=paper, data=data, selection_packet=support_seed,
                    proposal=proposal, question=request.question,
                    answer_kind=saved_kind, answer_parts=saved_parts,
                )
            if route_fallback:
                packet["source_policy"]["route_fallback_reason"] = route_fallback

            run["scope_approval"] = {
                "approved": True,
                "mode": proposal["mode"],
                "section_ids": list(
                    proposal.get("section_ids") or []
                ),
                "approved_at": (
                    (run.get("scope_approval") or {}).get("approved_at") or now()
                ),
            }
            run["scope_proposal"]["approved"] = True
            run["scope_used"] = packet["scope_used"]
            run["selected_source_layers"] = sorted(approved_layers)
            run["source_plan"] = {
                "answer_kind": saved_kind,
                "resolved_layers": list(run["selected_source_layers"]),
                "answer_modes": list(answer_modes(saved_kind, saved_parts)),
                "answer_parts": saved_parts,
                "route_fallback_reason": route_fallback,
                "approved_scope": {
                    "mode": proposal["mode"],
                    "section_ids": list(proposal.get("section_ids") or []),
                },
            }
            run["candidate_source_ids"] = {
                "claims": list(packet["claims"]),
                "evidence": list(packet["evidence"]),
                "mappings": list(packet["mappings"]),
            }

            if insufficient_material_answer(
                run=run, packet=packet, answer_mode=saved_kind,
                language=args.language, fallback_reason=route_fallback,
            ):
                return
            run["qa_material_inventory"] = material_inventory(packet)
            run["current_phase"] = "answer"
            store.save(data)

            reply = coordinator.answer_ask_ai(
                question=request.question,
                selection_material=packet,
                language=args.language,
            )

            if reply.status == "needs_scope_approval":
                finish_rejected_scope_request(
                    run=run, answer_mode=saved_kind, language=args.language,
                )
                return

            save_grounded_ask_ai_answer(
                run=run, reply=reply, packet=packet, paper=paper, data=data,
                answer_mode=saved_kind, fallback_reason=route_fallback,
            )
            return

        run["current_phase"] = "scope_decision"
        store.save(data)

        decision = coordinator.decide_ask_ai(
            question=request.question,
            selection_material=seed,
            paper_outline=paper_outline(paper),
        )
        raw_answer_parts_count = len(decision.answer_parts)
        decision, original_kind = align_information_need(decision)
        run["qa_route_history"] = [{
            "phase": "initial",
            "relation": decision.relation,
            "information_need": decision.information_need,
            "model_answer_kind": original_kind,
            "effective_answer_kind": decision.answer_kind,
            "suppressed_answer_parts_count": (
                raw_answer_parts_count - len(decision.answer_parts)
            ),
        }]

        claim_ids = validate_decision(
            decision,
            seed,
        )

        run["selection_decision"] = (
            decision.model_dump(mode="json")
        )
        store.save(data)

        if decision.relation == "unrelated_to_paper":
            message = (
                "无法根据当前选中的论文内容回答这个问题。"
                if args.language == "zh"
                else
                "This question cannot be answered from the selected paper content."
            )

            set_outcome(
                run,
                status="unanswerable",
                answer=message,
                limitations=[decision.reason],
            )
            return

        if decision.relation == "other_paper_topic":
            # No full-paper content is retrieved at this stage.
            selected_sections = {
                paper.elements[span.element_id].section_id
                for span in request.selection_context.segments
            }
            run["scope_proposal"] = proposed_scope(
                decision, paper,
                selected_section_ids=selected_sections,
                question=request.question,
            )

            proposal = run["scope_proposal"]
            if proposal["mode"] == "sections":
                names = ", ".join(proposal["section_titles"])
                message = (
                    f"这个问题涉及选区之外的主题，是否允许检索以下章节：{names}？"
                    if args.language == "zh" else
                    f"This question is outside the selection. May I search: {names}?"
                )
            else:
                message = (
                    "这个问题超出了选区主题。是否允许检索整篇论文？"
                    if args.language == "zh" else
                    "This question is outside the selection. May I search the full paper?"
                )

            set_outcome(
                run,
                status="needs_scope_approval",
                answer=message,
                missing_information=[decision.reason],
            )
            return

        # Same-subject supporting Evidence may be elsewhere in the paper;
        # it does not change the user's research topic or require approval.
        run["scope_used"] = "selection"
        # Resolve one authoritative source plan for both the worker and
        # final-answer paths. Planner-suggested layers cannot override it.
        requested_layers = set(layers_for_kind(
            decision.answer_kind, answer_parts=decision.answer_parts,
        ))
        data, run, seed, decision, claim_ids = enrich_selected_material(
            args=args, paper=paper, store=store, data=data, run=run,
            request=request, coordinator=coordinator, decision=decision,
            requested_layers=requested_layers, tracker=tracker, budget=budget,
        )
        requested_layers = set(layers_for_kind(
            decision.answer_kind, answer_parts=decision.answer_parts,
        ))
        actual_kind, actual_parts, route_fallback = resolve_answer_route(
            requested_kind=decision.answer_kind,
            answer_parts=decision.answer_parts,
            requested_layers=decision.requested_layers,
            claim_ids=claim_ids, data=data,
        )
        requested_layers = set(layers_for_kind(
            actual_kind, answer_parts=actual_parts,
        ))
        packet = selection_answer_packet(
            paper=paper, data=data, selection_seed=seed,
            claim_ids=claim_ids, answer_kind=actual_kind,
            answer_parts=actual_parts,
        )
        if route_fallback:
            packet["source_policy"]["route_fallback_reason"] = route_fallback
        run["source_plan"] = {
            "answer_kind": actual_kind,
            "answer_modes": list(answer_modes(actual_kind, actual_parts)),
            "answer_parts": [p.model_dump(mode="json") if hasattr(p, "model_dump")
                             else p for p in actual_parts],
            "resolved_layers": sorted(requested_layers),
            "route_fallback_reason": route_fallback,
            "selected_claim_ids": list(claim_ids),
        }
        run["scope_used"] = packet["scope_used"]
        run["selected_source_layers"] = sorted(requested_layers)
        run["candidate_source_ids"] = {
            "claims": list(packet["claims"]),
            "evidence": list(packet["evidence"]),
            "mappings": list(packet["mappings"]),
        }

        if insufficient_material_answer(
            run=run, packet=packet, answer_mode=actual_kind,
            language=args.language, fallback_reason=route_fallback,
        ):
            return
        run["qa_material_inventory"] = material_inventory(packet)
        run["current_phase"] = "answer"
        store.save(data)

        reply = coordinator.answer_ask_ai(
            question=request.question,
            selection_material=packet,
            language=args.language,
        )

        # A populated Evidence pool may still lack the observation requested by
        # the question. If the answer model explicitly reports that gap, one
        # unfinished same-Claim discovery is allowed before giving up. Never
        # repeat completed receipts or searches already attempted this time.
        if actual_kind == "evidence_answer" and reply.status == "unanswerable":
            from app.ask_ai_material import recover_incomplete_evidence_after_unanswerable
            previous_ids = set(packet["evidence"])
            data, run, attempted = recover_incomplete_evidence_after_unanswerable(
                args=args, paper=paper, store=store, data=data, run=run,
                coordinator=coordinator, claim_ids=claim_ids,
            )
            if attempted:
                refreshed_seed = build_selection_seed(
                    paper=paper, data=data, selection=request.selection_context,
                    question=request.question,
                )
                refreshed = selection_answer_packet(
                    paper=paper, data=data, selection_seed=refreshed_seed,
                    claim_ids=claim_ids, answer_kind=actual_kind,
                    answer_parts=actual_parts,
                )
                if set(refreshed["evidence"]) - previous_ids:
                    packet = refreshed
                    if route_fallback:
                        packet["source_policy"]["route_fallback_reason"] = route_fallback
                    run["qa_material_inventory"] = material_inventory(packet)
                    run["candidate_source_ids"] = {
                        "claims": list(packet["claims"]),
                        "evidence": list(packet["evidence"]),
                        "mappings": list(packet["mappings"]),
                    }
                    run["current_phase"] = "answer"
                    store.save(data)
                    reply = coordinator.answer_ask_ai(
                        question=request.question,
                        selection_material=packet,
                        language=args.language,
                    )

        if reply.status == "needs_scope_approval":
            finish_rejected_scope_request(
                run=run, answer_mode=actual_kind, language=args.language,
            )
            return
        if reply.status == "unanswerable":
            run["qa_material_gap"] = {
                "reason": "extracted_sources_insufficient_for_question",
                "missing_information": list(reply.missing_information),
                "collection_actions": list(run.get("qa_material_actions") or [])[-15:],
            }

        save_grounded_ask_ai_answer(
            run=run, reply=reply, packet=packet, paper=paper, data=data,
            answer_mode=actual_kind, fallback_reason=route_fallback,
        )

    except (MODEL_ERRORS + (ValueError, RuntimeError)) as exc:
        # Child workers may have committed a newer analysis snapshot before
        # throwing. Recover it BEFORE recording the current error outcome.
        from app.coordinator_runner import find_saved_run
        data = store.load(paper)
        run = find_saved_run(data, run["run_id"])
        run["status"] = "partial"
        run["answer_status"] = "unavailable"
        run["answer_available"] = False
        run["answer"] = None
        run.setdefault("issues", []).append(
            error_message(exc)
        )
        print(
            "Ask AI stopped:",
            error_message(exc),
            flush=True,
        )

    except KeyboardInterrupt:
        from app.coordinator_runner import find_saved_run
        data = store.load(paper)
        run = find_saved_run(data, run["run_id"])
        run["status"] = "interrupted"
        run["answer_available"] = False
        raise

    except Exception:
        from app.coordinator_runner import find_saved_run
        data = store.load(paper)
        run = find_saved_run(data, run["run_id"])
        run["status"] = "failed"
        run["answer_available"] = False
        raise

    finally:
        run["finished_at"] = now()
        run["duration_seconds_last_attempt"] = round(
            perf_counter() - started, 3
        )

        run.setdefault(
            "attempt_cost_reports", []
        ).append(tracker.report())

        if budget is not None:
            run["budget_report"] = budget.report()

        run["token_usage_report"] = task_token_report(
            run,
            data.runs,
        )

        if budget is not None:
            print_task_tokens(
                run["token_usage_report"],
                run["budget_report"],
            )

        from app.core.analysis_store import default_analysis_path

        save_outputs(
            data,
            run,
            default_analysis_path(paper.paper_id),
        )

        store.save(data)

        if budget is not None:
            budget.close()

        for key, value in (
            (QUESTION_BUDGET_ENV, previous_id),
            (QUESTION_LIMIT_ENV, previous_limit),
        ):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        print(
            "=== ASK AI FINISHED ===",
            flush=True,
        )
        print(
            "Workflow ID:",
            run["run_id"],
            flush=True,
        )
        print(
            "Status:",
            run["status"],
            flush=True,
        )
        print(
            "Answer status:",
            run["answer_status"],
            flush=True,
        )
