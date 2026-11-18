"""On-demand Claim/Evidence/Mapping reuse for selected-paper questions.

Only the selected text can become a NEW Claim target. Evidence discovery for
those Claims can search the paper. Do not invoke Evidence without Claims.
The standard workers, their receipts and the existing question-budget ledger
remain authoritative; no parallel extraction or Mapping implementation exists.
"""
from __future__ import annotations

from app.core.ask_ai_scope import (
    build_selection_seed,
    attach_direct_selection_references,
    attach_saved_claim_sources,
    saved_mapping_view,
)
from app.core.evidence_retrieval import EvidenceDataManager
from app.models.coordinator_v3 import CoordinatorPlan, CoordinatorTask, TaskScope
from app.selection_claims import _completed_ranges, _is_fully_covered


def _session(*, args, paper, store, data, run, coordinator, claim_ids,
             task_intent="show_evidence"):
    # Reuse the same child-process/checkpoint machinery as selected analysis.
    from app.workflow_runner import Session

    selection = run["request_contract"]["request"]["selection_context"]
    selected_elements = list(dict.fromkeys(
        item["element_id"] for item in selection["segments"]
    ))
    task = CoordinatorTask(
        task_id="ask_ai_material",
        intent=task_intent,
        goal=run["user_question"],
        scope=TaskScope(
            mode="selection", section_ids=[], element_ids=selected_elements,
            claim_ids=list(claim_ids), user_locked=False,
        ),
        evidence_scope="whole_paper",
        mapping_focus="claim_support",
        output_style="structured",
    )
    plan = CoordinatorPlan(
        paper_id=paper.paper_id, user_message=run["user_question"],
        language=run["language"], tasks=[task],
    )
    return Session(paper, args.paper, store, data, run, plan, args, coordinator), task


def _choose_evidence_search_targets(*, seed: dict, question: str, selected_ids: list[str]) -> list[str]:
    """Find a bounded Claim focus for internal Evidence discovery only.

    A claim used to guide retrieval does not become an Ask AI answer source.
    Never infer Mapping targets from this helper.
    """
    if selected_ids:
        return list(dict.fromkeys(selected_ids))
    from app.core.ask_ai_expansion import relevance
    ranked = sorted(seed["claims"].items(), key=lambda item: (
        relevance(question, str(item[1].get("text") or "")), item[0]
    ), reverse=True)
    if not ranked:
        return []
    best = relevance(question, str(ranked[0][1].get("text") or ""))
    if best == 0:
        return [ranked[0][0]]
    return [cid for cid, item in ranked[:3]
            if relevance(question, str(item.get("text") or "")) > 0]


def _pending_evidence_claims(session, task, claim_ids: list[str]) -> list[str]:
    """Respect the existing completion receipt of EACH Claim independently."""
    from app.workflow_runner import completed_evidence
    original_targets = list(task.scope.claim_ids)
    pending = []
    try:
        for cid in list(dict.fromkeys(claim_ids)):
            task.scope.claim_ids = [cid]
            if not completed_evidence(session, task):
                pending.append(cid)
    finally:
        task.scope.claim_ids = original_targets
    return pending


def enrich_selected_material(*, args, paper, store, data, run, request,
                             coordinator, decision, requested_layers,
                             tracker, budget):
    """Fill actual material gaps through existing workers in the same workflow.

    A question about the selection never requests new-topic approval merely
    because its Claim/Evidence/Mapping pool is incomplete. Completed extraction
    windows and search receipts prevent repeating already completed work.
    """
    from app.ask_ai_runner import validate_decision, align_information_need
    from app.core.ask_ai_source_plan import layers_for_kind

    actions = run.setdefault("qa_material_actions", [])
    seed = build_selection_seed(
        paper=paper, data=data, selection=request.selection_context,
        question=request.question,
    )
    selected_ids = validate_decision(decision, seed)
    wants_argument = "mapping" in requested_layers
    wants_evidence_answer = "evidence" in requested_layers
    wants_claim_answer = "claims" in requested_layers

    # A raw paragraph/figure/table is never a direct answer source. The
    # attached references below may only contribute pre-extracted Evidence.
    direct = attach_direct_selection_references(paper=paper, data=data, packet=seed)
    has_evidence = bool(direct["evidence"])

    # Inspect actual span coverage, not only whether some unrelated Claim
    # happens to exist in the selected paragraph.
    need_target = (wants_claim_answer or wants_argument or
                   (wants_evidence_answer and not has_evidence))
    if need_target and not selected_ids:
        missing_spans = [
            span for span in request.selection_context.segments
            if not _is_fully_covered(
                _completed_ranges(data, span.element_id), span.start, span.end
            )
        ]
        if missing_spans:
            session, _task = _session(
                args=args, paper=paper, store=store, data=data, run=run,
                coordinator=coordinator, claim_ids=[],
            )
            command = ["app.claim_runner", "--paper", str(args.paper),
                       "--model", args.model]
            for span in missing_spans:
                command += ["--span", span.element_id, str(span.start), str(span.end)]
            success = session.child("claim", command)
            data, run = session.data, session.run
            actions = run.setdefault("qa_material_actions", [])
            actions.append({"stage": "claim", "action": "extract_uncovered_selection",
                            "result": "completed" if success else "incomplete",
                            "selected_span_count": len(missing_spans)})
            seed = build_selection_seed(
                paper=paper, data=data, selection=request.selection_context,
                question=request.question,
            )
            if seed["claims"]:
                # Refresh Claim bindings only. The already-authorized topic and
                # source mode are immutable across an extraction retry.
                new_decision = coordinator.decide_ask_ai(
                    question=request.question,
                    selection_material=seed,
                    paper_outline={
                        "paper_id": paper.paper_id,
                        "title": paper.metadata.title,
                        "sections": [
                            {"section_id": section.section_id, "title": section.title}
                            for section in paper.sections
                        ],
                        "role": "location_hints_only_not_answer_sources",
                    },
                )
                raw_answer_parts_count = len(new_decision.answer_parts)
                new_decision, proposed_kind = align_information_need(new_decision)
                route_trace = run.setdefault("qa_route_history", [])
                route_trace.append({
                    "phase": "after_claim_extraction",
                    "relation": new_decision.relation,
                    "information_need": new_decision.information_need,
                    "model_answer_kind": proposed_kind,
                    "effective_answer_kind": new_decision.answer_kind,
                    "suppressed_answer_parts_count": (
                        raw_answer_parts_count - len(new_decision.answer_parts)
                    ),
                })
                if new_decision.relation == "about_selection":
                    rebound_ids = validate_decision(new_decision, seed)
                    selected_ids = rebound_ids
                    updates = {"selected_claim_ids": list(selected_ids)}
                    old_kind, new_kind = decision.answer_kind, new_decision.answer_kind
                    if {old_kind, new_kind} <= {"claim_answer", "evidence_answer"}:
                        # A retrieved Claim is not grounds to downgrade a user's
                        # empirical-information request into Claim-only output.
                        chosen = ("evidence_answer" if "evidence_answer" in
                                  {old_kind, new_kind} else "claim_answer")
                        updates["answer_kind"] = chosen
                        updates["information_need"] = (
                            "reported_findings" if chosen == "evidence_answer"
                            else "author_assertions"
                        )
                        updates["answer_parts"] = []
                    elif (old_kind == "composite_answer"
                            and new_kind == "composite_answer"
                            and len(new_decision.answer_parts) == len(decision.answer_parts)
                            and [p.answer_mode for p in new_decision.answer_parts]
                            == [p.answer_mode for p in decision.answer_parts]):
                        updates["answer_parts"] = new_decision.answer_parts
                    # Mapping and composite routes do not silently downgrade.
                    decision = decision.model_copy(update=updates)
                    route_trace.append({
                        "phase": "reconciled_after_claim",
                        "relation": decision.relation,
                        "information_need": decision.information_need,
                        "effective_answer_kind": decision.answer_kind,
                        "policy": "scope_locked_evidence_not_downgraded",
                    })
                    # Critical: execute the REVISED route during this call, not
                    # only when building the final answer packet afterward.
                    requested_layers = set(layers_for_kind(
                        decision.answer_kind, answer_parts=decision.answer_parts,
                    ))
                    wants_argument = "mapping" in requested_layers
                    wants_evidence_answer = "evidence" in requested_layers
                    wants_claim_answer = "claims" in requested_layers
                else:
                    actions.append({"stage": "claim_binding",
                                    "result": "kept_original_scope_decision"})
        else:
            actions.append({"stage": "claim", "action": "reuse_completed_windows",
                            "result": "no_uncovered_selection"})

    # A broad Claim-only question must not appear unanswerable just because
    # the planner returned no particular Claim ID. All records here are already
    # anchored inside the exact user selection.
    if wants_claim_answer and not selected_ids and seed["claims"]:
        selected_ids = list(seed["claims"])[:12]
        actions.append({"stage": "claim", "action": "use_selected_claim_pool",
                        "claim_ids": list(selected_ids)})
    elif wants_argument and not selected_ids and len(seed["claims"]) == 1:
        selected_ids = list(seed["claims"])
        actions.append({"stage": "mapping", "action": "single_selected_claim_target",
                        "claim_ids": list(selected_ids)})

    # Evidence Agent requires Claims. A selected Claim may retrieve support
    # from elsewhere in the paper without changing the question's subject.
    # For Evidence-only, internal Claim targets are retrieval context, never
    # independent citations in the final answer.
    evidence_targets = list(selected_ids)
    if wants_evidence_answer and not evidence_targets and not has_evidence:
        evidence_targets = _choose_evidence_search_targets(
            seed=seed, question=request.question, selected_ids=[])
        if evidence_targets:
            actions.append({"stage": "evidence", "action": "internal_claim_focus",
                            "claim_ids": list(evidence_targets)})
            # Evidence-only packets need the same internal Claim targets to
            # attach newly discovered supporting Evidence, but the Claims are
            # NOT exposed as direct answer sources.
            if requested_layers == {"evidence"}:
                selected_ids = list(evidence_targets)

    missing_mappings = [cid for cid in selected_ids
                        if wants_argument and saved_mapping_view(data, cid) is None]
    # No Evidence discovery is needed for a Claim whose valid Mapping already
    # contains adopted evidence and conclusions.
    required_evidence_targets = list(dict.fromkeys(
        ([cid for cid in evidence_targets] if wants_evidence_answer else [])
        + missing_mappings
    ))

    if required_evidence_targets:
        direct = attach_direct_selection_references(
            paper=paper, data=data, packet=seed)
        prior = attach_saved_claim_sources(
            paper=paper, data=data, selection_packet=direct,
            claim_ids=required_evidence_targets, include_evidence=True,
            question_scope_confirmed=True,
        )
        missing_candidates = [
            cid for cid in required_evidence_targets
            if not prior["evidence_candidates_by_claim"].get(cid)
        ]
        reusable = [cid for cid in required_evidence_targets
                    if cid not in missing_candidates]
        if reusable:
            actions.append({"stage": "evidence", "action": "reuse_saved_candidates",
                            "claim_ids": reusable})
        if missing_candidates:
            session, task = _session(
                args=args, paper=paper, store=store, data=data, run=run,
                coordinator=coordinator, claim_ids=missing_candidates,
            )
            from app.workflow_runner import collect_evidence
            pending = _pending_evidence_claims(session, task, missing_candidates)
            completed = [cid for cid in missing_candidates if cid not in pending]
            if completed:
                actions.append({"stage": "evidence", "action": "reuse_completed_search",
                                "claim_ids": completed, "result": "no_new_discovery"})
            if pending:
                task.scope.claim_ids = pending
                collect_evidence(session, task)
                data, run = session.data, session.run
                actions = run.setdefault("qa_material_actions", [])
                actions.append({"stage": "evidence", "action": "discover_for_selected_claims",
                                "claim_ids": pending, "result": "attempted"})

    # Each Mapping is for one selected Claim. Do not force Mapping for a
    # descriptive question, or create new Mapping when one already exists.
    if wants_argument and selected_ids:
        missing_mappings = [cid for cid in selected_ids
                            if saved_mapping_view(data, cid) is None]
        if missing_mappings:
            manager = EvidenceDataManager(data, paper)
            supported = [cid for cid in missing_mappings
                         if manager.evidence_ids([cid])]
            unsupported = [cid for cid in missing_mappings if cid not in supported]
            if unsupported:
                actions.append({"stage": "mapping", "action": "skip_without_evidence",
                                "claim_ids": unsupported, "result": "insufficient_material"})
            if supported:
                from app.coordinator_runner import execute_mappings
                from app.agents.mapping_agent import (
                    mapping_model_for_run, resolve_mapping_model,
                )
                session, task = _session(
                    args=args, paper=paper, store=store, data=data, run=run,
                    coordinator=coordinator, claim_ids=supported,
                    task_intent="explain_argument",
                )
                state = session.state(task)
                state["selected_claim_ids"] = list(supported)
                state["selected_evidence_ids"] = manager.evidence_ids(supported)
                state["collection_ready"] = True
                session.save()
                # A new Ask AI run has only the Coordinator model in run.model.
                # Mapping must use its own configured default, not that model.
                requested_mapping_model = getattr(args, "mapping_model", None)
                mapping_model = (
                    mapping_model_for_run(session.run, requested_mapping_model)
                    if session.run.get("mapping_model")
                    else resolve_mapping_model(requested_mapping_model)
                )
                execute_mappings(
                    coordinator=coordinator, data=session.data, store=store,
                    run=session.run, plan=session.plan, model=mapping_model,
                    language=args.language, tracker=tracker, budget=budget,
                    paper=paper,
                )
                session.refresh()
                data, run = session.data, session.run
                actions = run.setdefault("qa_material_actions", [])
                actions.append({"stage": "mapping", "action": "analyze_missing_claims",
                                "claim_ids": list(supported), "result": "attempted",
                                "mapping_model": mapping_model})
        else:
            actions.append({"stage": "mapping", "action": "reuse_saved_mappings",
                            "claim_ids": list(selected_ids)})

    seed = build_selection_seed(
        paper=paper, data=data, selection=request.selection_context,
        question=request.question,
    )
    # Persist the final target binding, including internal Evidence-only
    # retrieval focus, without changing the answer route or topic permission.
    decision = decision.model_copy(update={"selected_claim_ids": list(selected_ids)})
    run["selection_decision"] = decision.model_dump(mode="json")
    return data, run, seed, decision, selected_ids


def recover_incomplete_evidence_after_unanswerable(*, args, paper, store, data,
                                                    run, coordinator, claim_ids):
    """One bounded same-topic scan only after a genuinely unanswered Evidence query.

    Never repeat a completed search or a discovery already attempted in the
    current Ask AI attempt. This function is internal to Evidence-only answers.
    """
    targets = list(dict.fromkeys(claim_ids))
    actions = run.setdefault("qa_material_actions", [])
    if not targets:
        return data, run, False
    attempted = {
        cid for action in actions
        if action.get("stage") == "evidence"
        and action.get("action") in {
            "discover_for_selected_claims", "retry_after_unanswerable"
        }
        for cid in action.get("claim_ids") or []
    }
    candidates = [cid for cid in targets if cid not in attempted]
    if not candidates:
        return data, run, False

    session, task = _session(
        args=args, paper=paper, store=store, data=data, run=run,
        coordinator=coordinator, claim_ids=candidates,
    )
    from app.workflow_runner import collect_evidence
    pending = _pending_evidence_claims(session, task, candidates)
    completed = [cid for cid in candidates if cid not in pending]
    if completed:
        actions.append({
            "stage": "evidence", "action": "reuse_completed_search",
            "claim_ids": completed, "result": "already_complete",
        })
    if not pending:
        return data, run, False

    task.scope.claim_ids = pending
    collect_evidence(session, task)
    data, run = session.data, session.run
    run.setdefault("qa_material_actions", []).append({
        "stage": "evidence", "action": "retry_after_unanswerable",
        "claim_ids": pending, "result": "attempted",
    })
    return data, run, True
