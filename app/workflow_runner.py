"""One question lifecycle: plan -> collect/reuse -> review -> Mapping -> answer.

The same lifecycle handles a new paper and later questions. Agent results live
in document pools; question-specific selections and coverage live in its run.
Raw source reports are never substituted for the final answer.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from app.agents.coordinator_agent import CoordinatorAgent
from app.budget import HARD_LIMIT, PaperBudget, QUESTION_BUDGET_ENV, QUESTION_LIMIT_ENV, money
from app.core.token_report import task_token_report, print_task_tokens
from app.core.table_context import audits_for_packet
from app.claim_runner import make_windows
from app.coordinator_runner import (
    POLICY_VERSION, MODEL_ERRORS, claim_view, error_message, execute_mappings,
    final_answer, find_saved_run, material_view, now, save_outputs,
    select_task_claims, signature, source_packet, unique,
)
from app.core.analysis_store import AnalysisStore, default_analysis_path
from app.core.retired_associations import SOURCE_SCAN_POLICY, scan_policy
from app.core.evidence_retrieval import EvidenceDataManager, scope_element_ids
from app.core.request_contract import (
    ReaderRequest, REQUEST_POLICY, SAVED_OPERATIONS, FIXED_OPERATIONS,
    resolve_request, deterministic_plan, get_request, locked_targets, saved_only,
    enforce_plan_targets, evidence_boundary,
)
from app.core.mapping_views import bind_saved_task, display_view
from app.direct_evidence_runner import direct_search_key
from app.table_fact_lookup import (
    covering_evidence_ids, render_exact_table_answer, resolve_exact_table_lookup,
)
from app.models.coordinator_v3 import (CoordinatorPlan, CoordinatorTask, TaskScope, required_capabilities)
from app.models.document_v2 import PaperDocument, Paragraph
from app.usage_cost import CostTracker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_COLLECTION_ROUNDS = 3


def section_ids_with_children(paper, section_ids):
    selected = set(section_ids)
    while True:
        grown = selected | {s.section_id for s in paper.sections if s.parent_section_id in selected}
        if grown == selected:
            return selected
        selected = grown


def source_scope(paper, task):
    scope = task.scope
    if scope.mode == "whole_paper":
        return set(), set(), True
    sections = section_ids_with_children(paper, scope.section_ids)
    elements = set(scope.element_ids)
    if not sections and not elements:
        raise ValueError(f"Task {task.task_id} has no grounded source scope.")
    return sections, elements, False


def in_scope(paper, element_id, sections, elements, whole):
    if element_id not in paper.elements:
        return False
    return whole or element_id in elements or paper.elements[element_id].section_id in sections


def paper_catalog(paper, question):
    words = set(re.findall(r"[\w-]+", question.casefold()))
    sections, candidates, figures = [], [], []
    for section in paper.sections:
        paragraphs = [paper.elements[eid] for eid in section.block_ids
                      if isinstance(paper.elements[eid], Paragraph)]
        sections.append({"section_id": section.section_id, "title": section.title,
                         "parent_section_id": section.parent_section_id,
                         "paragraph_count": len(paragraphs)})
        # These are localization hints only. Selection may expand beyond them.
        for paragraph in paragraphs:
            score = len(words & set(re.findall(r"[\w-]+", paragraph.text.casefold())))
            candidates.append({"element_id": paragraph.element_id, "section_id": section.section_id,
                               "score": score, "excerpt": paragraph.text[:400]})
    for element in paper.elements.values():
        if element.kind in {"table", "figure", "equation"}:
            figures.append({"element_id": element.element_id, "section_id": element.section_id,
                            "kind": element.kind, "label": getattr(element, "label", None),
                            "caption": getattr(element, "caption", "")})
    return {"paper_id": paper.paper_id, "title": paper.metadata.title,
            "sections": sections, "figures_tables_equations": figures,
            "localization_hints": sorted(candidates, key=lambda x: -x["score"])[:15],
            "note": "Hints are not exhaustive. Use real IDs; an absent hint is not absent paper content."}


def validate_plan(plan, paper, data, *, focus_sections=(), focus_elements=()):
    known_sections, known_elements = {s.section_id for s in paper.sections}, set(paper.elements)
    ids = [t.task_id for t in plan.tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Task IDs must be unique.")
    for task in plan.tasks:
        if not required_capabilities(task.intent):
            continue
        if focus_sections or focus_elements:
            task.scope.mode = "selection"
            task.scope.section_ids = list(focus_sections)
            task.scope.element_ids = list(focus_elements)
            task.scope.user_locked = True
        task.scope.section_ids = unique(str(x).strip().strip('"\'') for x in task.scope.section_ids)
        task.scope.element_ids = unique(str(x).strip().strip('"\'') for x in task.scope.element_ids)
        bad = set(task.scope.section_ids) - known_sections
        bad_elements = set(task.scope.element_ids) - known_elements
        if bad or bad_elements:
            raise ValueError(f"Unknown plan sources: sections={sorted(bad)}, elements={sorted(bad_elements)}")
        task.scope.claim_ids = [cid for cid in unique(task.scope.claim_ids) if cid in data.claim_pool]
        # Existing explicitly chosen Claims remain anchored. They do not force
        # whole-section expansion or erase user source boundaries.
        if task.scope.mode != "whole_paper":
            sections = section_ids_with_children(paper, task.scope.section_ids)
            for cid in task.scope.claim_ids:
                element_id = data.claim_pool[cid].source_element_id
                if element_id not in known_elements:
                    continue
                if paper.elements[element_id].section_id not in sections and element_id not in task.scope.element_ids:
                    if task.scope.user_locked:
                        continue
                    task.scope.element_ids.append(element_id)
            source_scope(paper, task)
    plan.paper_id = paper.paper_id


class Session:
    """Single owner of question state; reload after every child process."""
    def __init__(self, paper, paper_path, store, data, run, plan, args, coordinator):
        self.paper, self.paper_path, self.store = paper, paper_path, store
        self.data, self.run, self.plan, self.args = data, run, plan, args
        self.run_id = run["run_id"]
        self.coordinator = coordinator
        self.catalog = paper_catalog(paper, run["user_question"])
        self.local_attempts = {}

    def save(self):
        self.run["task_plan"] = self.plan.model_dump(mode="json")
        self.store.save(self.data)

    def refresh(self):
        self.data = self.store.load(self.paper)
        self.run = find_saved_run(self.data, self.run_id)

    def state(self, task):
        return self.run.setdefault("task_states", {}).setdefault(task.task_id, {})

    def child(self, stage, arguments):
        if saved_only(self.run):
            raise ValueError("A saved-analysis explanation cannot invoke extraction or another child workflow.")
        if stage == "claim" and locked_targets(self.run):
            raise ValueError("Explicit existing Claim targets do not authorize new Claim extraction.")
        self.run["current_phase"] = stage
        self.save()
        before = {r.get("run_id") for r in self.data.runs}
        print(f"\n=== {stage.upper()} ===", flush=True)
        result = subprocess.run([sys.executable, "-u", "-m", *arguments],
            cwd=PROJECT_ROOT, env=os.environ.copy(), check=False)
        self.refresh()
        new = [r for r in self.data.runs if r.get("run_id") not in before]
        for record in new:
            record["parent_workflow_id"] = self.run_id
            record["question_budget_id"] = self.run_id
        self.run.setdefault("stage_results", []).append({
            "stage": stage, "exit_code": result.returncode,
            "child_run_ids": [r.get("run_id") for r in new],
            "status": new[-1].get("status", "unknown") if new else ("reused" if result.returncode == 0 else "failed"),
            "time": now()})
        if result.returncode:
            self.run.setdefault("issues", []).append(f"{stage} exited with code {result.returncode}; saved receipts retained.")
        self.save()
        return result.returncode == 0

    def progress(self):
        return signature({"claims": list(self.data.claim_pool), "evidence": list(self.data.evidence_pool),
            "claim_windows": self.data.completed_claim_windows,
            "evidence_scans": self.data.completed_evidence_scans})


def claim_coverage(session, task):
    sections, elements, whole = source_scope(session.paper, task)
    windows = make_windows(session.paper, None if whole else sections,
                           selected_elements=None if whole else elements)
    missing = [w for w in windows if session.data.completed_claim_windows.get(w.window_id) != "completed"]
    return windows, missing


def collect_claims(session, task):
    windows, missing = claim_coverage(session, task)
    if not missing:
        return
    sections, elements, whole = source_scope(session.paper, task)
    cmd = ["app.claim_runner", "--paper", str(session.paper_path), "--model", session.args.model]
    if whole:
        cmd.append("--all")
    else:
        for sid in sorted(sections):
            cmd += ["--section", sid]
        for eid in sorted(elements):
            if isinstance(session.paper.elements[eid], Paragraph):
                cmd += ["--element-id", eid]
    if not whole and "--section" not in cmd and "--element-id" not in cmd:
        return
    session.child("claim", cmd)


def evidence_scope(session, task):
    if locked_targets(session.run):
        allowed = evidence_boundary(session.run, session.paper)
        return (set(), set(), True) if allowed is None else (set(), set(allowed), False)
    sections, elements, whole = source_scope(session.paper, task)
    if task.evidence_scope == "whole_paper" and not task.scope.user_locked:
        whole, sections, elements = True, set(), set()
    if not whole and not task.scope.user_locked:
        # Follow explicit source links using the actual document index, not ID
        # prefixes. This collects referenced figures/tables outside a section.
        sources = {session.data.claim_pool[cid].source_element_id for cid in task.scope.claim_ids
                   if cid in session.data.claim_pool}
        for mention in session.paper.mentions:
            if mention.source_anchor.element_id in sources and mention.target_id in session.paper.elements:
                elements.add(mention.target_id)
    return sections, elements, whole


def candidate_evidence(session, task):
    sections, elements, whole = evidence_scope(session, task)
    if "mapping" in required_capabilities(task.intent) and task.scope.claim_ids:
        return EvidenceDataManager(session.data, session.paper).evidence_ids(
            task.scope.claim_ids,
            allowed_element_ids=(evidence_boundary(session.run, session.paper) if locked_targets(session.run)
                                 else scope_element_ids(session.paper, session.data, task)),
        )
    scoped = [eid for eid, record in session.data.evidence_pool.items()
              if in_scope(session.paper, record.source_element_id, sections, elements, whole)]

    # For precise table facts, prefer Evidence discovered for THIS question.
    # Older table Evidence may have correct cells but the wrong dataset label.
    # If there is no question-scoped scan from the revised source pack, let the
    # ordinary Evidence Agent collect one. Never delete old records or block
    # the fallback: unsuccessful/empty new scans still expose saved material.
    if task.intent == "fact_lookup" and any(
        session.paper.elements[eid].kind == "table"
        for eid in elements if eid in session.paper.elements
    ):
        key = direct_search_key(
            session.paper, question=task.goal, model=session.args.model,
            image_mode=session.args.image_mode, section_ids=sections,
            element_ids=elements,
        )
        receipt = session.data.direct_evidence_searches.get(key) or {}
        fresh = [eid for eid in receipt.get("evidence_ids", [])
                 if eid in session.data.evidence_pool]
        if fresh:
            return unique(fresh + (scoped if receipt.get("status") != "completed" else []))
        if receipt.get("status") != "completed" and not session.args.use_saved_sources:
            return []
    return scoped


def completed_evidence(session, task):
    sections, elements, whole = evidence_scope(session, task)
    for cid in task.scope.claim_ids:
        receipt = session.data.evidence_search_by_claim.get(cid, {})
        if (receipt.get("status") != "completed"
                or receipt.get("paper_revision") != session.paper.revision
                or (receipt.get("discovery_policy") != "claim_focused_discovery_no_links_v2"
                    and scan_policy(receipt) != SOURCE_SCAN_POLICY)):
            return False
        levels = {"none": 0, "relevant": 1, "all": 2}
        if levels.get(receipt.get("image_mode"), -1) < levels[session.args.image_mode]:
            return False
        if receipt.get("search_scope") != "whole_paper":
            if whole or not set(receipt.get("source_section_ids") or []) >= sections:
                return False
            if not set(receipt.get("source_element_ids") or []) >= elements:
                return False
    return bool(task.scope.claim_ids)


def strict_evidence_required(session, task, requirements):
    """Optional safety gate for argument-analysis tasks with selected Claims.

    Ordinary runs, including existing ACM/IEEE workflows, are unchanged unless
    --strict-evidence-coverage is explicitly enabled. Direct Evidence and
    Claim-only operations do not require a Claim-focused Evidence receipt.
    """
    return bool(
        not saved_only(session.run) and
        getattr(session.args, "strict_evidence_coverage", False)
        and "mapping" in requirements
        and "evidence" in requirements
        and task.scope.claim_ids
    )


def strict_evidence_complete(session, task, requirements):
    """Check finished source scans, not the LLM's material-ready opinion.

    Old completed receipts without per-modality flags remain valid. A known
    false scan flag is not treated as completed. Source-matched, fully verified
    table cells remain a narrow exception for explicitly locked Claim tasks.
    This checks provenance/coverage, not whether a scientific Claim is true.
    """
    if not strict_evidence_required(session, task, requirements):
        return True
    state = session.state(task)
    if (task.scope.user_locked and state.get("collection_ready")
            and state.get("collection_strategy") == "original_cells_verified_and_explicit_claims"
            and state.get("numeric_table_audit")):
        return True
    if not completed_evidence(session, task):
        return False
    for claim_id in task.scope.claim_ids:
        receipt = session.data.evidence_search_by_claim.get(claim_id) or {}
        if receipt.get("text_scan_complete") is False:
            return False
        if (session.args.image_mode != "none"
                and receipt.get("image_scan_complete") is False):
            return False
    return True


def collect_evidence(session, task, missing=()):
    sections, elements, whole = evidence_scope(session, task)
    cids = task.scope.claim_ids
    if cids and not completed_evidence(session, task):
        attempts = session.local_attempts.get(task.task_id, 0)
        session.local_attempts[task.task_id] = attempts + 1
        batch_size = max(1, 6 // (2 ** attempts))
        cmd = ["app.evidence_runner", "--paper", str(session.paper_path),
               "--model", session.args.model, "--image-mode", session.args.image_mode,
               "--claim-batch-size", str(batch_size)]
        for cid in cids:
            cmd += ["--claim-id", cid]
        if not whole:
            for sid in sorted(sections):
                cmd += ["--source-section", sid]
            for eid in sorted(elements):
                cmd += ["--source-element-id", eid]
        session.child("evidence", cmd)
    else:
        # A direct focused lookup supplements saved material without fabricating
        # new Claims or candidate-association records. The ordinary Evidence Agent runs it.
        focus = task.goal
        if missing:
            focus += "\nLocate the following requested source material: " + "; ".join(missing)
        key = direct_search_key(session.paper, question=focus, model=session.args.model,
            image_mode=session.args.image_mode, section_ids=sections, element_ids=elements)
        if (session.data.direct_evidence_searches.get(key) or {}).get("status") == "completed":
            return
        cmd = ["app.direct_evidence_runner", "--paper", str(session.paper_path),
               "--question", focus, "--model", session.args.model,
               "--image-mode", session.args.image_mode]
        if not whole:
            for sid in sorted(sections):
                cmd += ["--section", sid]
            for eid in sorted(elements):
                cmd += ["--element-id", eid]
        session.child("direct_evidence", cmd)


def expand_requested_scope(session, task, review):
    section_ids, element_ids = review.additional_section_ids, review.additional_element_ids
    known_sections = {s.section_id for s in session.paper.sections}
    if not set(section_ids) <= known_sections or not set(element_ids) <= set(session.paper.elements):
        raise ValueError("Collection review requested an unknown paper source ID.")
    if task.scope.user_locked:
        session.state(task)["scope_expansion_blocked_by_user"] = bool(section_ids or element_ids or review.search_whole_paper)
        return False
    previous = (task.scope.mode, set(task.scope.section_ids), set(task.scope.element_ids))
    if review.search_whole_paper:
        task.scope.mode = "whole_paper"
        task.scope.section_ids, task.scope.element_ids = [], []
    elif task.scope.mode != "whole_paper":
        task.scope.section_ids = unique(task.scope.section_ids + section_ids)
        task.scope.element_ids = unique(task.scope.element_ids + element_ids)
    return previous != (task.scope.mode, set(task.scope.section_ids), set(task.scope.element_ids))



def reconcile_collection_review_references(session, task, requirements, review, packet):
    """Ground model-selected IDs in the sources actually supplied for this review.

    The Evidence Pool can contain valid records from other tasks. Their IDs
    are not proof for this task unless they were included in its source packet.
    This shared post-review boundary is independent of publisher/HTML adapter.
    """
    allowed_by_field = {
        "selected_evidence_ids": set(packet["evidence"]),
        "additional_section_ids": {s.section_id for s in session.paper.sections},
        "additional_element_ids": set(session.paper.elements),
    }
    rejected = {}
    for field, allowed in allowed_by_field.items():
        requested = unique(getattr(review, field) or [])
        accepted = [source_id for source_id in requested if source_id in allowed]
        excluded = [source_id for source_id in requested if source_id not in allowed]
        setattr(review, field, accepted)
        if excluded:
            rejected[field] = excluded

    if not rejected:
        return review

    out_of_packet = [source_id for source_id in rejected.get("selected_evidence_ids", [])
                     if source_id in session.data.evidence_pool]
    absent_from_pool = [source_id for source_id in rejected.get("selected_evidence_ids", [])
                        if source_id not in session.data.evidence_pool]
    warning = {
        "stage": "collection_review",
        "task_id": task.task_id,
        "policy": "source_packet_boundary_v2",
        "rejected_ids": rejected,
        "evidence_in_pool_but_not_in_packet": out_of_packet,
        "evidence_not_in_pool": absent_from_pool,
        "note": "Excluded IDs cannot become Evidence or Mapping support.",
    }
    session.state(task).setdefault("collection_reference_warnings", []).append(warning)
    session.run.setdefault("collection_reference_warnings", []).append(warning)

    # The reviewer must not certify an Evidence-requiring operation as ready
    # if every Evidence ID it selected was outside the reviewed packet.
    if ("evidence" in requirements
            and rejected.get("selected_evidence_ids")
            and not review.selected_evidence_ids):
        review.ready_for_requested_operation = False
        review.missing_information = unique(list(review.missing_information) + [
            "The Coordinator selected no Evidence from this task's reviewed "
            "source packet. References to other tasks or unknown IDs were excluded."
        ])

    print("Collection review excluded out-of-packet source IDs:",
          task.task_id, rejected, flush=True)
    return review


def reuse_ready_sources_before_new_claim_extraction(session, task, requirements):
    """Check existing sources before scanning newly planned background sections.

    This is a QUESTION-level readiness check, not a claim that all selected
    sections were completely extracted. Exhaustive Claim listings/summaries
    always extract their requested source scope before returning.
    """
    if (session.args.use_saved_sources or "claim" not in requirements
            or task.intent in {"list_claims", "summarize_claims", "summarize"}):
        return False
    _, pending = claim_coverage(session, task)
    if not pending:
        return False
    sections, elements, whole = source_scope(session.paper, task)
    candidates = [cid for cid, record in session.data.claim_pool.items()
                  if in_scope(session.paper, record.source_element_id, sections, elements, whole)]
    if not candidates:
        return False
    selected = select_task_claims(
        coordinator=session.coordinator, data=session.data, run=session.run,
        task=task, candidate_ids=candidates, store=session.store,
    )
    if not selected:
        return False
    task.scope.claim_ids = selected
    eids = candidate_evidence(session, task) if "evidence" in requirements else []
    if "evidence" in requirements and not eids:
        return False

    packet = source_packet(session.data, claim_ids=selected,
                           evidence_ids=eids,
                           paper=session.paper, question=session.run["user_question"])
    # A complete source-matched numeric table needs no LLM collection review
    # merely because unrelated Claim extraction windows remain unprocessed.
    # The narrower readiness guard still requires explicit, located Claims.
    if _ready_from_exact_saved_table(session, task, requirements, packet, pending):
        print("Reusing verified Claims + complete table before extra extraction:",
              task.task_id, flush=True)
        return True
    if not strict_evidence_complete(session, task, requirements):
        # A relevant old Evidence item is not the same as a completed scan.
        return False
    view = material_view(coordinator=session.coordinator, packet=packet,
                         run=session.run, data=session.data, store=session.store,
                         purpose=f"reuse_before_expansion_{task.task_id}")
    coverage = {
        "claim_windows": len(pending),
        "pending_claim_windows": [w.window_id for w in pending],
        "evidence_search_by_claim": {
            cid: session.data.evidence_search_by_claim.get(cid, {"status": "not_scanned"})
            for cid in selected
        },
        "cached_sources_only": True,
        "scope_is_user_locked": task.scope.user_locked,
        "search_is_not_proof_of_absence": True,
    }
    review = session.coordinator.review(
        question=session.run["user_question"], task=task,
        material=view, document_catalog=session.catalog, coverage=coverage,
    )
    review = reconcile_collection_review_references(
        session, task, requirements, review, packet
    )
    state = session.state(task)
    state.setdefault("collection_reviews", []).append(review.model_dump(mode="json"))
    ready = bool(review.ready_for_requested_operation)
    ready = (ready and not review.missing_information and
             not review.additional_section_ids and not review.additional_element_ids and
             not review.search_whole_paper)
    if "evidence" in requirements and not review.selected_evidence_ids:
        ready = False
    if ready:
        state["selected_claim_ids"] = list(selected)
        state["selected_evidence_ids"] = unique(review.selected_evidence_ids)
        state["missing_information"] = []
        state["collection_ready"] = True
        state["coverage"] = coverage
        state["collection_strategy"] = "reused_question_material_before_extra_extraction"
        state["unprocessed_scope_windows"] = [w.window_id for w in pending]
        session.save()
        print(f"Task material ready from existing pools before expanding extraction: {task.task_id}", flush=True)
        return True
    session.save()
    return False


def reuse_verified_table_cells(session, task):
    """Use authoritative cells when every requested fact has existing Evidence.

    This is a narrow Coordinator response-builder optimization, not a new
    extraction requirement. Ambiguous/non-structured tables take the full
    legacy collection and review path without losing any source records.
    """
    if task.intent != "fact_lookup" or task.scope.mode == "whole_paper":
        return False
    sections, elements, _ = evidence_scope(session, task)
    table_ids = sorted(
        {eid for eid in elements if eid in session.paper.elements
         and session.paper.elements[eid].kind == "table"}
        | {eid for eid, element in session.paper.elements.items()
           if element.kind == "table" and element.section_id in sections}
    )
    lookup = resolve_exact_table_lookup(session.paper, session.run["user_question"], table_ids)
    if lookup is None:
        return False
    source_ids = covering_evidence_ids(session.data, session.paper, lookup)
    if not source_ids:
        return False
    state = session.state(task)
    state["selected_claim_ids"] = []
    state["selected_evidence_ids"] = source_ids
    state["collection_ready"] = True
    state["missing_information"] = []
    state["verified_table_lookup"] = {**lookup, "evidence_ids": source_ids,
                                       "verification_basis": "original_table_cells"}
    state["coverage"] = {"claim_windows": 0, "pending_claim_windows": [],
                         "evidence_search_by_claim": {},
                         "scope_is_user_locked": task.scope.user_locked,
                         "search_is_not_proof_of_absence": True}
    session.save()
    print("Reusing verified table cells:", lookup["table_id"], source_ids, flush=True)
    return True


def deliver_verified_table_answer(session):
    """Generate an exact lookup response without trusting model descriptions."""
    if len(session.plan.tasks) != 1:
        return False
    task = session.plan.tasks[0]
    state = session.state(task)
    lookup = state.get("verified_table_lookup")
    if task.intent != "fact_lookup" or not state.get("collection_ready") or not lookup:
        return False
    evidence_ids = list(lookup.get("evidence_ids") or [])
    if not evidence_ids or any(eid not in session.data.evidence_pool for eid in evidence_ids):
        return False
    answer = render_exact_table_answer(lookup, evidence_ids, language=session.args.language)
    session.run["answer_material_ids"] = evidence_ids
    session.run["unfinished_tasks"] = []
    session.run["answer"] = {"answer": answer, "used_source_ids": evidence_ids,
                             "limitations": []}
    session.run["answer_reference_check"] = {
        "mentioned_ids": evidence_ids, "unknown_ids": [], "references_recognized": True,
    }
    session.run["answer_status"] = "generated"
    session.run["status"] = "completed"
    session.run["answer_available"] = True
    session.save()
    print("\n=== VERIFIED TABLE ANSWER ===\n" + answer, flush=True)
    return True


def _ready_from_exact_saved_table(session, task, requirements, packet, pending):
    """Skip paid collection review only when source-completeness is proven.

    This does not judge whether the author's argument is valid. Mapping still
    analyzes reasoning. Ambiguous, partial, or non-tabular sources fall back to
    the usual Coordinator review and incremental extraction.
    """
    if ("mapping" not in requirements or not task.scope.user_locked
            or not task.scope.claim_ids or not packet["evidence"]):
        return False
    explicitly_requested = set(re.findall(r"\bclaim_\d{6}\b",
                                          session.run["user_question"], flags=re.IGNORECASE))
    if not explicitly_requested or not set(task.scope.claim_ids) <= explicitly_requested:
        return False
    # Incomplete EXHAUSTIVE extraction is not a blocker when the user explicitly
    # selected already-located Claims for this one question. Retain the pending
    # scope windows as an explicit coverage note, not as paper-wide absence.
    if any(session.data.claim_pool[cid].source_status != "located"
           for cid in task.scope.claim_ids if cid in session.data.claim_pool):
        return False
    table_ids = {eid for eid in task.scope.element_ids
                 if eid in session.paper.elements
                 and session.paper.elements[eid].kind == "table"}
    if not table_ids:
        return False
    present = {item["source_element_id"] for item in packet["evidence"].values()}
    if not table_ids <= present:
        return False
    audits = {item["table_id"]: item for item in packet.get("numeric_table_checks", [])}
    if any(audits.get(eid, {}).get("status") != "verified_complete"
           or not audits[eid].get("metric_checks")
           or any(m["status"] != "verified" or m["direction"] == "unranked"
                  for m in audits[eid]["metric_checks"])
           for eid in table_ids):
        return False
    expected_metrics = {metric.casefold() for metric in ("MD", "MSE", "AuRC")
                        if re.search(r"\b" + metric + r"\b",
                                     session.run["user_question"], flags=re.IGNORECASE)}
    for eid in table_ids:
        audited = {m["metric"].casefold() for m in audits[eid]["metric_checks"]}
        if expected_metrics and not expected_metrics <= audited:
            return False
    state = session.state(task)
    state["selected_claim_ids"] = list(task.scope.claim_ids)
    state["selected_evidence_ids"] = list(packet["evidence"])
    state["missing_information"] = []
    state["collection_ready"] = True
    state["collection_strategy"] = "original_cells_verified_and_explicit_claims"
    state["unprocessed_scope_windows"] = [w.window_id for w in pending]
    state["coverage_scope"] = "explicit_claim_ids_and_source_table_not_exhaustive_paper"
    state["source_packet_bytes"] = len(json.dumps(packet, ensure_ascii=False).encode("utf-8"))
    state["numeric_table_audit"] = {
        eid: {"status": audits[eid]["status"], "cell_count": audits[eid]["cell_count"],
              "metrics": [m["metric"] for m in audits[eid]["metric_checks"]]}
        for eid in sorted(table_ids)}
    session.save()
    return True


def prepare_mapping_task(session, task):
    """Keep analysis targets fixed; source context never creates extra targets."""
    state = session.state(task)
    request = get_request(session.run)
    targets = locked_targets(session.run)
    if targets:
        selected = [cid for cid in task.scope.claim_ids if cid in targets] or list(targets)
        pending = []
        task.scope.claim_ids = list(selected)
        state["claim_selection_policy"] = "application_owned_targets"
        if saved_only(session.run):
            bindings = bind_saved_task(session.data, state, selected,
                request.mapping_ids_by_claim, allowed_element_ids=evidence_boundary(session.run, session.paper))
            session.run.setdefault("execution_decisions", []).append({
                "stage": "collection", "action": "read_saved_analysis", "claim_ids": selected,
                "mapping_ids_by_claim": dict(bindings), "claim_calls": 0, "evidence_calls": 0})
            session.save()
            print("Saved analysis targets:", selected, "Mappings:", bindings,
                  "New extraction: disabled", flush=True)
            return
    else:
        # Free paper questions retain their established extraction behavior.
        requested = {cid.casefold() for cid in re.findall(r"\bclaim_\d{6}\b", session.run["user_question"], flags=re.I)}
        sections, elements, whole = source_scope(session.paper, task)
        explicit = [cid for cid in task.scope.claim_ids if cid in requested and cid in session.data.claim_pool
                    and in_scope(session.paper, session.data.claim_pool[cid].source_element_id, sections, elements, whole)]
        if explicit and set(task.scope.claim_ids) <= requested:
            selected, pending = explicit, []
            state["claim_selection_policy"] = "explicit_existing_claims"
        else:
            if not session.args.use_saved_sources:
                collect_claims(session, task)
            candidates = [cid for cid, claim in session.data.claim_pool.items()
                          if in_scope(session.paper, claim.source_element_id, sections, elements, whole)]
            selected = select_task_claims(coordinator=session.coordinator, data=session.data, run=session.run,
                task=task, candidate_ids=candidates, store=session.store)
            _, pending = claim_coverage(session, task)
    task.scope.claim_ids = list(selected)
    state = session.state(task)
    state["selected_claim_ids"] = list(selected)
    session.save()
    if selected and not completed_evidence(session, task) and not session.args.use_saved_sources:
        session.run.setdefault("execution_decisions", []).append({
            "stage": "evidence", "action": "collect_missing_coverage", "claim_ids": list(selected),
            "reason": "The requested analysis lacks a completed Evidence discovery receipt in its permitted source scope."})
        collect_evidence(session, task)
    complete = completed_evidence(session, task) if selected else False
    eids = candidate_evidence(session, task) if selected else []
    state = session.state(task)
    state.update({"selected_claim_ids": list(selected), "selected_evidence_ids": list(eids),
        "collection_review_mode": "local_evidence_index_no_llm_links",
        "collection_ready": bool(selected) and not pending and complete,
        "coverage": {"claim_extraction_complete": not bool(pending),
            "pending_claim_windows": [w.window_id for w in pending],
            "evidence_scan_complete": complete,
            "evidence_search_scope": "whole_paper" if evidence_scope(session, task)[2] else "selected_sources",
            "candidate_count": len(eids), "search_is_not_proof_of_absence": True}})
    state["missing_information"] = ([] if selected else ["No target Claims were selected."])
    if pending:
        state["missing_information"].append("Some requested Claim extraction windows remain unprocessed.")
    if selected and not complete:
        state["missing_information"].append("Evidence discovery coverage is incomplete; only saved sources are available.")
    session.save()
    print("Argument material:", task.task_id, "Targets:", selected,
          "Local candidates:", len(eids), "Discovery completed:", complete, flush=True)


def prepare_task(session, task):
    requirements = required_capabilities(task.intent)
    state = session.state(task)
    if not requirements:
        state.update({"collection_ready": False, "missing_information": ["This request requires external information or clarification outside the opened paper."],
                      "selected_claim_ids": [], "selected_evidence_ids": []})
        session.save()
        return
    if "mapping" in requirements:
        prepare_mapping_task(session, task)
        return
    # Reuse already located exact cells before a paid Evidence scan or review.
    if "evidence" in requirements and "claim" not in requirements:
        if reuse_verified_table_cells(session, task):
            return
    # A follow-up may already have enough grounded results, even when the
    # Planner mentioned extra method sections. Do not extract those sections
    # until the existing material has been checked for THIS question.
    if reuse_ready_sources_before_new_claim_extraction(session, task, requirements):
        return
    attempted = set()
    for round_number in range(1, session.args.collection_rounds + 1):
        if "claim" in requirements:
            if not session.args.use_saved_sources:
                collect_claims(session, task)
            sections, elements, whole = source_scope(session.paper, task)
            candidates = [cid for cid, c in session.data.claim_pool.items()
                          if in_scope(session.paper, c.source_element_id, sections, elements, whole)]
            # Reselect after EVERY pool change. The previous requested IDs never
            # suppress newly extracted relevant Claims.
            selected = select_task_claims(coordinator=session.coordinator, data=session.data,
                run=session.run, task=task, candidate_ids=candidates, store=session.store)
            task.scope.claim_ids = selected
        else:
            task.scope.claim_ids = []
            session.state(task)["selected_claim_ids"] = []
        session.save()
        eids = candidate_evidence(session, task) if "evidence" in requirements else []
        # An explicit Claim/table request can already be resolved by exact,
        # verified original cells. Preserve that older ACM/IEEE fast path even
        # when the optional strict scan gate is enabled.
        if (strict_evidence_required(session, task, requirements)
                and task.scope.user_locked and eids
                and not strict_evidence_complete(session, task, requirements)
                and any(session.paper.elements[eid].kind == "table"
                        for eid in task.scope.element_ids
                        if eid in session.paper.elements)):
            _, pre_pending = claim_coverage(session, task)
            exact_packet = source_packet(
                session.data, claim_ids=task.scope.claim_ids,
                evidence_ids=eids, paper=session.paper,
                question=session.run["user_question"])
            if _ready_from_exact_saved_table(
                    session, task, requirements, exact_packet, pre_pending):
                print("Reusing explicitly verified source table:",
                      task.task_id, flush=True)
                return
        needs_evidence = (
            "evidence" in requirements and not session.args.use_saved_sources
            and (not eids or not strict_evidence_complete(session, task, requirements))
        )
        if needs_evidence:
            collect_evidence(session, task)
            eids = candidate_evidence(session, task)
        # Newly collected structured Evidence may now cover the exact lookup.
        if "evidence" in requirements and "claim" not in requirements:
            if reuse_verified_table_cells(session, task):
                return
        windows, pending = claim_coverage(session, task) if "claim" in requirements else ([], [])
        coverage = {"claim_windows": len(windows), "pending_claim_windows": [w.window_id for w in pending],
            "evidence_search_by_claim": {cid: session.data.evidence_search_by_claim.get(cid, {"status": "not_scanned"})
                                         for cid in task.scope.claim_ids},
            "scope_is_user_locked": task.scope.user_locked,
            "search_is_not_proof_of_absence": True}
        # Claim-only tasks have already selected author statements. Evidence
        # selection is not part of their contract and must not gate an answer.
        # Use the actual extraction coverage, not a second paid LLM review.
        if requirements == ("claim",):
            state = session.state(task)
            state["coverage"] = coverage
            state["selected_claim_ids"] = list(task.scope.claim_ids)
            state["selected_evidence_ids"] = []
            missing_information = []
            if pending:
                missing_information.append(
                    "Claim extraction remains incomplete in the selected source scope."
                )
            if not task.scope.claim_ids:
                missing_information.append(
                    "No relevant author Claims were selected from the requested scope."
                )
            state["missing_information"] = missing_information
            state["collection_ready"] = bool(task.scope.claim_ids) and not pending
            state["collection_review_mode"] = "deterministic_claim_only"
            if missing_information:
                state["collection_stop_reason"] = "Claim-only collection incomplete; preserve the available statements."
            session.save()
            print(
                "Claim-only material:", task.task_id,
                "selected=", len(task.scope.claim_ids),
                "pending_windows=", len(pending),
                "ready=", state["collection_ready"],
                flush=True,
            )
            return

        packet = source_packet(session.data, claim_ids=task.scope.claim_ids,
                               evidence_ids=eids,
                               paper=session.paper, question=session.run["user_question"])
        if _ready_from_exact_saved_table(session, task, requirements, packet, pending):
            print("Reusing verified Claims + complete table without paid review:",
                  task.task_id, flush=True)
            return
        if not strict_evidence_complete(session, task, requirements):
            # Preserve usable partial Evidence and its genuine scan receipts.
            # Do not ask the Coordinator to approve data it never searched.
            state = session.state(task)
            state["coverage"] = coverage
            state["selected_claim_ids"] = list(task.scope.claim_ids)
            state["selected_evidence_ids"] = list(eids)
            state["collection_ready"] = False
            state["missing_information"] = [
                "Claim-focused Evidence scanning is incomplete for this task's "
                "source scope. Mapping must wait for completed coverage."
            ]
            state["collection_stop_reason"] = "Strict Evidence scan gate: incomplete."
            session.save()
            print("Strict Evidence scan incomplete:", task.task_id, flush=True)
            return
        review_view = material_view(coordinator=session.coordinator, packet=packet,
            run=session.run, data=session.data, store=session.store, purpose=f"review_{task.task_id}_{round_number}")
        print(f"Reviewing accumulated task material: {task.task_id}, round {round_number}", flush=True)
        review = session.coordinator.review(question=session.run["user_question"], task=task,
            material=review_view,
            document_catalog=(session.catalog if not task.scope.user_locked else {
                "paper_id": session.paper.paper_id,
                "explicit_locked_scope": task.scope.model_dump(mode="json"),
                "note": "Do not request out-of-scope material."}),
            coverage=coverage)
        review = reconcile_collection_review_references(
            session, task, requirements, review, packet
        )
        state = session.state(task)
        state["coverage"] = coverage
        state["selected_evidence_ids"] = unique(review.selected_evidence_ids)
        state["missing_information"] = list(review.missing_information)
        state["collection_ready"] = bool(review.ready_for_requested_operation)
        if "claim" in requirements and not task.scope.claim_ids:
            state["collection_ready"] = False
        if "evidence" in requirements and not state["selected_evidence_ids"]:
            state["collection_ready"] = False
        state.setdefault("collection_reviews", []).append(review.model_dump(mode="json"))
        session.save()
        if state["collection_ready"]:
            print("Task material ready:", task.task_id, flush=True)
            return
        if session.args.use_saved_sources:
            state["collection_stop_reason"] = "Explicit saved-source-only recovery; no new retrieval requested."
            session.save()
            return
        expanded = expand_requested_scope(session, task, review)
        attempt_key = signature({"scope": task.scope.model_dump(mode="json"),
                                 "missing": review.missing_information, "pool": session.progress()})
        if attempt_key in attempted:
            state["collection_stop_reason"] = "No new source or unresolved operation to execute in this scope."
            session.save()
            return
        attempted.add(attempt_key)
        if round_number == session.args.collection_rounds:
            state["collection_stop_reason"] = "Collection-round limit reached; essential unresolved material is recorded."
            session.save()
            return
        session.save()
        if expanded:
            # The next round extracts any new Claims BEFORE reselecting them.
            continue
        before = session.progress()
        if "evidence" in requirements:
            collect_evidence(session, task, review.missing_information)
        elif "claim" in requirements and pending:
            collect_claims(session, task)
        if session.progress() == before:
            session.state(task)["collection_stop_reason"] = "Requested scope was processed without additional usable material; not a paper-wide absence claim."
            session.save()
            return


def reset_analysis(path: Path):
    """Delete only this paper's previous analysis and archived answers.

    Standardized paper files, other papers, and global budget logs are untouched.
    Caller must explicitly request a fresh analysis.
    """
    analysis_file = Path(path).resolve()
    analysis_root = (PROJECT_ROOT / "data" / "analysis").resolve()
    paper_dir = analysis_file.parent

    if (analysis_file.name != "analysis.json"
            or paper_dir.parent != analysis_root
            or paper_dir.name == "_archives"):
        raise ValueError(f"Refusing to clear unexpected analysis path: {analysis_file}")

    archive_dir = analysis_root / "_archives" / paper_dir.name
    for directory in (paper_dir, archive_dir):
        if directory.is_symlink():
            raise ValueError(f"Refusing to delete a linked directory: {directory}")
        if directory.is_dir():
            shutil.rmtree(directory)
            print("Deleted previous analysis:", directory, flush=True)



def resume_question_limit(run, requested_limit_usd, store, data):
    """Explicitly increase a saved question's ceiling without resetting spending.

    The workflow ID remains the SQLite question-budget ID. Existing settled
    costs and saved Claim/Evidence/Mapping pools are not modified. An omitted
    limit on resume preserves the saved original limit and behavior.
    """
    current = money(run["question_limit_usd"])
    if requested_limit_usd is None:
        return float(current)
    proposed = money(requested_limit_usd)
    if proposed == current:
        return float(current)
    if proposed < current:
        raise ValueError("Resume cannot lower a question budget; use its saved limit.")
    if proposed > HARD_LIMIT:
        raise ValueError(f"Explicit budget increases cannot exceed USD {HARD_LIMIT}.")
    if (run.get("budget_policy") != "per_question_v1"
            or run.get("question_budget_id") != run.get("run_id")):
        raise ValueError("Cannot amend a workflow without a persisted per-question budget ID.")
    run.setdefault("question_limit_amendments", []).append({
        "previous_limit_usd": float(current),
        "new_limit_usd": float(proposed),
        "recorded_at": now(),
        "authorization": "explicit --question-limit-usd on --resume-workflow",
    })
    run["question_limit_usd"] = float(proposed)
    store.save(data)
    print(f"Question budget ceiling increased from ${current} to ${proposed}; "
          f"all earlier spending remains charged to {run['run_id']}.", flush=True)
    return float(proposed)


def preview_request_contract(args, paper, data, contract, old_run=None):
    """Read-only request preview, including recovery without touching a ledger."""
    request = ReaderRequest.model_validate(contract["request"])
    plan = deterministic_plan(contract, paper, data, args.language)
    print("Paper:", paper.paper_id, flush=True)
    print("Existing pools:", len(data.claim_pool), len(data.evidence_pool), len(data.mappings), flush=True)
    print("Operation:", request.operation, flush=True)
    print("Target Claims:", request.target_claim_ids, flush=True)
    print("Request origin:", contract["origin"], flush=True)
    summary = {"request_contract": contract, "resume_workflow": (old_run or {}).get("run_id"),
               "preview_only": True, "no_api_calls": True, "no_analysis_writes": True}
    if request.operation in SAVED_OPERATIONS:
        state = {}
        bindings = bind_saved_task(data, state, request.target_claim_ids, request.mapping_ids_by_claim,
                                  allowed_element_ids=evidence_boundary({"request_contract": contract}, paper))
        summary["mapping_ids_by_claim"] = bindings
        summary["missing_saved_claim_ids"] = state["deferred_claim_ids"]
        summary["display_evidence_ids_by_claim"] = {
            cid: display_view(data, cid, mid).get("evidence_ids", []) for cid, mid in bindings.items()}
        summary["planned_new_calls"] = {"planning": 0, "claim": 0, "evidence": 0,
                                         "mapping": 0, "coordinator_answer": "at most 1; 0 if an identical answer is reusable"}
        summary["budget_id_on_resume"] = (old_run or {}).get("question_budget_id")
    elif plan is not None:
        summary["task_plan"] = plan.model_dump(mode="json")
        summary["note"] = "Only explicit targets may be analyzed. Actual source reuse and cost are determined during execution."
    else:
        summary["note"] = "A free paper question still needs planning; this preview makes no planning call."
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("Preview only: no API calls, source extraction, analysis writes, or budget changes.", flush=True)


def adopt_request_contract(run, contract, plan, paper, data, language):
    """Migrate execution state, not source pools or historical paid records."""
    fixed_plan = deterministic_plan(contract, paper, data, language)
    if fixed_plan is not None and run.get("orchestration_policy") != REQUEST_POLICY:
        if run.get("task_plan") or run.get("task_states"):
            run.setdefault("orchestration_history", []).append({
                "archived_at": now(), "reason": "Apply the original request's explicit target/result-reading contract.",
                "task_plan": deepcopy(run.get("task_plan")), "task_states": deepcopy(run.get("task_states", {})),
                "used_mapping_ids": list(run.get("used_mapping_ids") or []),
                "unfinished_tasks": deepcopy(run.get("unfinished_tasks") or [])})
        run["task_states"] = {}
        run["used_mapping_ids"] = []
        run["unfinished_tasks"] = []
        run.pop("answer_input_signature", None)
        plan = fixed_plan
    elif plan is None and fixed_plan is not None:
        plan = fixed_plan
    run["request_contract"] = contract
    run["orchestration_policy"] = REQUEST_POLICY
    if plan is not None:
        enforce_plan_targets(plan, run, data)
        run["task_plan"] = plan.model_dump(mode="json")
    return plan


def attempt_effects(data, before, tracker):
    """Separate the current attempt from cumulative same-question accounting."""
    children = [r for r in data.runs if r.get("run_id") not in before["run_ids"]
                and r.get("stage") in {"claim", "evidence"}]
    reports = [tracker.report(), *[r.get("api_cost_report") or {} for r in children]]
    by_agent = {}
    for report in reports:
        for item in report.get("by_agent") or []:
            name = item.get("agent", "unknown")
            by_agent[name] = by_agent.get(name, 0) + int(item.get("llm_calls") or 0)
    return {"new_claim_ids": sorted(set(data.claim_pool) - before["claims"]),
            "new_evidence_ids": sorted(set(data.evidence_pool) - before["evidence"]),
            "new_mapping_ids": sorted(set(data.mappings) - before["mappings"]),
            "recorded_calls_by_agent": by_agent,
            "estimated_attempt_usd": sum(float(r.get("estimated_total_usd") or 0) for r in reports),
            "budget_accounting": "The existing per-question ledger remains authoritative; this is an attempt diagnostic."}


def run_workflow(args):
    paper_path = args.paper.resolve()
    paper = PaperDocument.model_validate_json(paper_path.read_text(encoding="utf-8-sig"))
    path = default_analysis_path(paper.paper_id)
    if args.fresh_analysis:
        reset_analysis(path)
    store = AnalysisStore(path)
    data = store.load(paper)
    old_run = find_saved_run(data, args.resume_workflow) if args.resume_workflow else None
    if old_run and old_run.get("paper_revision") != paper.revision:
        raise ValueError("Saved workflow and open paper revisions differ.")
    if args.preview and not (old_run or args.question or getattr(args, "request_file", None)):
        print("Paper:", paper.paper_id)
        print("Existing pools:", len(data.claim_pool), len(data.evidence_pool), len(data.mappings))
        for section in paper.sections:
            print(section.section_id, section.title)
        return
    contract = resolve_request(args, paper, data, saved_run=old_run)
    request = ReaderRequest.model_validate(contract["request"])
    if not args.resume_workflow:
        args.question = request.question
    if args.preview:
        preview_request_contract(args, paper, data, contract, old_run)
        return
    print("Analysis pools: Claims:", len(data.claim_pool), "Evidence:", len(data.evidence_pool), "Mappings:", len(data.mappings), flush=True)
    start = perf_counter()
    if args.resume_workflow:
        run = find_saved_run(data, args.resume_workflow)
        if run.get("workflow_policy") != POLICY_VERSION:
            raise ValueError("The saved run uses legacy orchestration. Start the requested fresh-analysis test, not a legacy resume.")
        if run.get("paper_revision") != paper.revision:
            raise ValueError("Cannot resume a workflow with a different paper revision.")
        limit = resume_question_limit(run, args.question_limit_usd, store, data)
        plan = CoordinatorPlan.model_validate(run["task_plan"])
        args.strict_evidence_coverage = bool(
            args.strict_evidence_coverage or run.get("strict_evidence_coverage", False)
        )
        if args.strict_evidence_coverage:
            run["strict_evidence_coverage"] = True
        # Resume uses the original model/language and the same question budget.
        args.model, args.language, args.image_mode = run["model"], run["language"], run["image_mode"]
        print("Resuming workflow:", run["run_id"], flush=True)
    else:
        limit = 0.50 if args.question_limit_usd is None else args.question_limit_usd
        run = {"run_id": uuid4().hex[:12], "stage": "coordinator", "mode": "full_pipeline",
               "workflow_policy": POLICY_VERSION, "budget_policy": "per_question_v1",
               "paper_revision": paper.revision, "user_question": args.question,
               "model": args.model, "language": args.language, "image_mode": args.image_mode,
               "question_limit_usd": limit, "status": "running", "started_at": now(),
               "current_phase": "planning", "task_states": {}, "stage_results": [],
               "issues": [], "created_mapping_ids": [], "used_mapping_ids": [],
               "answer": None, "answer_status": "unavailable"}
        run["question_budget_id"] = run["run_id"]
        if args.strict_evidence_coverage:
            run["strict_evidence_coverage"] = True
        data.runs.append(run)
        store.save(data)
        plan = None
    plan = adopt_request_contract(run, contract, plan, paper, data, args.language)
    before = {"claims": set(data.claim_pool), "evidence": set(data.evidence_pool),
              "mappings": set(data.mappings), "run_ids": {r.get("run_id") for r in data.runs}}
    store.save(data)
    print("Workflow ID:", run["run_id"], flush=True)
    print("Operation:", request.operation, "Target Claims:", request.target_claim_ids, flush=True)
    original_budget_id = os.environ.get(QUESTION_BUDGET_ENV)
    original_budget_limit = os.environ.get(QUESTION_LIMIT_ENV)
    os.environ[QUESTION_BUDGET_ENV] = run["run_id"]
    os.environ[QUESTION_LIMIT_ENV] = str(limit)
    budget = PaperBudget(paper.paper_id, question_id=run["run_id"], limit_usd=limit)
    tracker = CostTracker()
    coordinator = CoordinatorAgent(model=args.model, tracker=tracker, budget=budget)
    session = None
    run["status"], run["answer_status"] = "running", "pending"
    run["answer_available"] = False
    run.pop("interruption_reason", None)
    try:
        if plan is None:
            catalog = paper_catalog(paper, args.question)
            catalog["existing_claims"] = [claim_view(c) for c in data.claim_pool.values()
                if not request.target_claim_ids or c.claim_id in request.target_claim_ids]
            catalog["request_contract"] = contract
            catalog["existing_evidence_count"] = len(data.evidence_pool)
            catalog["existing_mapping_count"] = len(data.mappings)
            catalog["language"] = args.language
            catalog["recent_questions"] = [r.get("user_question") for r in data.runs
                                          if r.get("mode") == "full_pipeline" and r.get("run_id") != run["run_id"]][-3:]
            catalog["explicit_user_scope"] = {"section_ids": args.focus_section, "element_ids": args.focus_element}
            print("\n=== COORDINATOR PLANNING ===", flush=True)
            plan = coordinator.plan(args.question, catalog)
            enforce_plan_targets(plan, run, data)
            try:
                validate_plan(plan, paper, data, focus_sections=[] if request.target_claim_ids else args.focus_section, focus_elements=[] if request.target_claim_ids else args.focus_element)
            except ValueError as exc:
                # One schema/ID repair, not an implicit whole-paper fallback.
                catalog["previous_plan_error"] = str(exc)
                plan = coordinator.plan(args.question, catalog)
                enforce_plan_targets(plan, run, data)
                validate_plan(plan, paper, data, focus_sections=[] if request.target_claim_ids else args.focus_section, focus_elements=[] if request.target_claim_ids else args.focus_element)
            plan.language = args.language
            run["task_plan"] = plan.model_dump(mode="json")
            store.save(data)
            print(json.dumps(run["task_plan"], ensure_ascii=False, indent=2), flush=True)
        else:
            enforce_plan_targets(plan, run, data)
            validate_plan(plan, paper, data)
        session = Session(paper, paper_path, store, data, run, plan, args, coordinator)
        if args.plan_only:
            run["status"] = "paused_after_planning"
            session.save()
            return
        for task in plan.tasks:
            prepare_task(session, task)
        data, run = session.data, session.run
        if args.strict_evidence_coverage:
            pending = [
                task for task in plan.tasks
                if strict_evidence_required(session, task, required_capabilities(task.intent))
                and not strict_evidence_complete(
                    session, task, required_capabilities(task.intent))
            ]
            for task in plan.tasks:
                session.state(task)["mapping_blocked_for_incomplete_coverage"] = task in pending
            if pending:
                session.save()
                print("Strict Evidence gate deferred only incomplete tasks:",
                      [task.task_id for task in pending], flush=True)
        run["current_phase"] = "mapping"
        session.save()
        execute_mappings(coordinator=coordinator, data=data, store=store, run=run, plan=plan,
                         model=args.model, language=args.language, tracker=tracker, budget=budget,
                         paper=paper)
        run["current_phase"] = "answer"
        session.save()
        if not deliver_verified_table_answer(session):
            final_answer(coordinator=coordinator, data=data, store=store, run=run, plan=plan,
                         language=args.language, paper=paper)
    except (MODEL_ERRORS + (ValueError, RuntimeError)) as exc:
        if session is not None:
            session.refresh()
            data, run = session.data, session.run
        if run.get("answer"):
            run.setdefault("answer_history", []).append({"answer": run["answer"], "archived_at": now(), "reason": "Attempt interrupted before successful presentation."})
        run["answer"] = None
        run["status"], run["answer_status"] = "partial", "unavailable"
        run.setdefault("issues", []).append(error_message(exc))
        print("Workflow stopped:", error_message(exc), flush=True)
    except KeyboardInterrupt:
        data = store.load(paper)
        run = find_saved_run(data, run["run_id"])
        run["status"] = "interrupted"
        run["answer"] = None
        run["answer_status"] = "unavailable"
        raise
    except Exception:
        data = store.load(paper)
        run = find_saved_run(data, run["run_id"])
        run["status"] = "failed"
        run["answer"] = None
        run["answer_status"] = "unavailable"
        raise
    finally:
        run["last_attempt_effects"] = attempt_effects(data, before, tracker)
        run["finished_at"] = now()
        run["duration_seconds_last_attempt"] = round(perf_counter() - start, 3)
        run.setdefault("attempt_cost_reports", []).append(tracker.report())
        run["budget_report"] = budget.report()
        run["token_usage_report"] = task_token_report(run, data.runs)
        print_task_tokens(run["token_usage_report"], run["budget_report"])
        save_outputs(data, run, path)
        store.save(data)
        budget.close()
        for key, value in ((QUESTION_BUDGET_ENV, original_budget_id), (QUESTION_LIMIT_ENV, original_budget_limit)):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        print("Attempt effects:", json.dumps(run["last_attempt_effects"], ensure_ascii=False), flush=True)
        print("\n=== FULL WORKFLOW FINISHED ===")
        print("Workflow ID:", run["run_id"])
        print("Status:", run["status"])
        print("Answer status:", run.get("answer_status"))
        print("Analysis JSON:", path.resolve())
        print("Answer file:", run.get("answer_file"))



def run_selected_analysis(args, resume_id: str | None = None):
    """One budget and one run: selected Claims -> broad Evidence -> Mapping -> answer.

    The selection limits CLAIM extraction, not where supporting Evidence may
    appear. Each Claim receives a bounded packet from the shared local Evidence index.
    """
    from app.selection_claims import (
        _validated_segments, _completed_ranges, _is_fully_covered,
        _selected_claims,
    )

    paper = PaperDocument.model_validate_json(args.paper.read_text(encoding="utf-8-sig"))
    analysis_path = default_analysis_path(paper.paper_id)
    store = AnalysisStore(analysis_path)
    data = store.load(paper)
    if resume_id:
        run = find_saved_run(data, resume_id)
        if run.get("mode") != "selected_analysis" or run.get("workflow_policy") != POLICY_VERSION:
            raise ValueError("This is not a resumable selected-analysis run.")
        if run.get("paper_revision") != paper.revision:
            raise ValueError("Paper revision changed; highlighted source spans cannot be reused.")
        limit = resume_question_limit(run, args.question_limit_usd, store, data)
        args.model, args.language, args.image_mode = (
            run["model"], run["language"], run["image_mode"]
        )
        segments = list(run["selection"]["segments"])
        plan = CoordinatorPlan.model_validate(run["task_plan"])
        run["attempt_count"] = run.get("attempt_count", 1) + 1
    else:
        payload = json.loads(args.analyze_selection_file.read_text(encoding="utf-8-sig"))
        segments = _validated_segments(paper, payload)
        limit = 0.50 if args.question_limit_usd is None else args.question_limit_usd
        task = CoordinatorTask(
            task_id="t1",
            intent="explain_argument",
            goal=("Analyze EVERY substantive author Claim in the selected passages. "
                  "Locate relevant paper Evidence, explain each Claim, and provide "
                  "a grounded synthesis of meaningful cross-Claim relationships."),
            scope=TaskScope(
                mode="selection", section_ids=[],
                element_ids=unique(item["element_id"] for item in segments),
                claim_ids=[], user_locked=False,
            ),
            evidence_scope="whole_paper",
            mapping_focus="claim_support",
            output_style="structured",
        )
        plan = CoordinatorPlan(paper_id=paper.paper_id,
                               user_message=task.goal, language=args.language,
                               tasks=[task])
        run = {
            "run_id": uuid4().hex[:12], "stage": "coordinator",
            "mode": "selected_analysis", "workflow_policy": POLICY_VERSION,
            "paper_revision": paper.revision,
            "user_question": task.goal,
            "model": args.model, "language": args.language,
            "image_mode": args.image_mode, "question_limit_usd": limit,
            "budget_policy": "per_question_v1", "status": "running",
            "started_at": now(), "current_phase": "selection_claim",
            "task_states": {}, "stage_results": [], "issues": [],
            "created_mapping_ids": [], "used_mapping_ids": [],
            "answer": None, "answer_status": "unavailable",
            "selection": {"segments": segments},
            "task_plan": plan.model_dump(mode="json"),
        }
        run["question_budget_id"] = run["run_id"]
        if args.preview:
            missing = [item for item in segments if not _is_fully_covered(
                _completed_ranges(data, item["element_id"]),
                item["start"], item["end"])]
            print(json.dumps({
                "paper_id": paper.paper_id, "mode": "selected_analysis",
                "selection": segments, "pending_claim_spans": missing,
                "existing_selected_claim_ids": [c.claim_id for c in _selected_claims(data, paper, segments)],
                "evidence_scope": "whole_paper", "preview_only": True,
            }, ensure_ascii=False, indent=2), flush=True)
            return
        data.runs.append(run)
        store.save(data)

    original_id = os.environ.get(QUESTION_BUDGET_ENV)
    original_limit = os.environ.get(QUESTION_LIMIT_ENV)
    os.environ[QUESTION_BUDGET_ENV] = run["run_id"]
    os.environ[QUESTION_LIMIT_ENV] = str(limit)
    budget = PaperBudget(paper.paper_id, question_id=run["run_id"], limit_usd=limit)
    tracker = CostTracker()
    coordinator = CoordinatorAgent(model=args.model, tracker=tracker, budget=budget)
    session = Session(paper, args.paper, store, data, run, plan, args, coordinator)
    started = perf_counter()
    run["status"] = "running"
    run["answer_status"] = "unavailable"
    run["answer"] = None
    try:
        task = plan.tasks[0]
        missing = [item for item in segments if not _is_fully_covered(
            _completed_ranges(session.data, item["element_id"]),
            item["start"], item["end"])]
        if missing:
            cmd = ["app.claim_runner", "--paper", str(args.paper), "--model", args.model]
            for item in missing:
                cmd += ["--span", item["element_id"], str(item["start"]), str(item["end"])]
            session.child("claim", cmd)

        # Collect EVERY active Claim located wholly in the selected spans.
        # Do not call select_task_claims(): this is an Analyze-all operation.
        claims = _selected_claims(session.data, paper, segments)
        selected_ids = [claim.claim_id for claim in claims]
        claim_complete = all(_is_fully_covered(
            _completed_ranges(session.data, item["element_id"]),
            item["start"], item["end"]) for item in segments)
        task.scope.claim_ids = selected_ids
        state = session.state(task)
        state["selected_claim_ids"] = selected_ids
        state["claim_selection_policy"] = "all_substantive_claims_in_highlight"
        session.run["selection"]["claim_ids"] = selected_ids
        session.run["selection"]["claim_complete"] = claim_complete
        session.save()

        if not selected_ids:
            # Zero substantive Claims is a valid result of focused extraction,
            # NOT an instruction to invent or expand Claims outside the highlight.
            text = ("No independent substantive research Claims were identified "
                    "in the selected text under the current extraction policy.")
            if not claim_complete:
                text = ("Claim extraction was incomplete; no qualifying Claim "
                        "is available from the processed selection yet.")
            session.run["answer"] = {"answer": text, "used_source_ids": [],
                                     "limitations": [] if claim_complete else [
                                         "Some selected source spans were not processed."]}
            session.run["answer_status"] = "generated" if claim_complete else "partial_generated"
            session.run["status"] = "completed" if claim_complete else "partial"
            session.run["answer_available"] = True
            session.run["answer_material_ids"] = []
            session.run["unfinished_tasks"] = [] if claim_complete else [
                {"task_id": task.task_id, "stage": "claim"}]
            session.save()
            return

        # Evidence is allowed throughout the paper; selected Claims remain fixed.
        if not completed_evidence(session, task):
            collect_evidence(session, task)
        evidence_complete = completed_evidence(session, task)
        evidence_ids = candidate_evidence(session, task)
        state = session.state(task)
        state["selected_evidence_ids"] = evidence_ids
        state["missing_information"] = []
        if not claim_complete:
            state["missing_information"].append("Some selected text spans are not fully extracted.")
        if not evidence_complete:
            state["missing_information"].append("Evidence scan is incomplete; existing findings remain available.")
        # No retrieved candidate is not equivalent to a proven absence in the paper.
        state["collection_ready"] = bool(claim_complete and evidence_complete)
        state["coverage"] = {
            "selected_claim_count": len(selected_ids),
            "claim_extraction_complete": claim_complete,
            "evidence_scan_complete": evidence_complete,
            "evidence_search_scope": "whole_paper",
            "selected_evidence_count": len(evidence_ids),
            "search_is_not_proof_of_absence": True,
        }
        session.save()
        session.run["current_phase"] = "mapping"
        session.save()
        execute_mappings(
            coordinator=coordinator, data=session.data, store=store,
            run=session.run, plan=plan, model=args.model, language=args.language,
            tracker=tracker, budget=budget, paper=paper,
        )
        session.run["current_phase"] = "answer"
        session.save()
        final_answer(coordinator=coordinator, data=session.data, store=store,
                     run=session.run, plan=plan, language=args.language, paper=paper)
    except (MODEL_ERRORS + (ValueError, RuntimeError)) as exc:
        session.refresh()
        session.run["status"] = "partial"
        session.run["answer_status"] = "unavailable"
        session.run.setdefault("issues", []).append(error_message(exc))
        print("Selected analysis stopped:", error_message(exc), flush=True)
    except KeyboardInterrupt:
        session.refresh()
        session.run["status"] = "interrupted"
        raise
    except Exception:
        session.refresh()
        session.run["status"] = "failed"
        raise
    finally:
        session.run["finished_at"] = now()
        session.run["duration_seconds_last_attempt"] = round(perf_counter() - started, 3)
        session.run.setdefault("attempt_cost_reports", []).append(tracker.report())
        session.run["budget_report"] = budget.report()
        session.run["token_usage_report"] = task_token_report(session.run, session.data.runs)
        print_task_tokens(session.run["token_usage_report"], session.run["budget_report"])
        save_outputs(session.data, session.run, analysis_path)
        session.save()
        budget.close()
        for key, value in ((QUESTION_BUDGET_ENV, original_id),
                           (QUESTION_LIMIT_ENV, original_limit)):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        print("\n=== SELECTED ANALYSIS FINISHED ===", flush=True)
        print("Workflow ID:", session.run["run_id"], flush=True)
        print("Status:", session.run["status"], flush=True)
        print("Answer status:", session.run.get("answer_status"), flush=True)
        print("Analysis JSON:", analysis_path.resolve(), flush=True)
        print("Answer file:", session.run.get("answer_file"), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Reader-v3 unified paper-question workflow")
    parser.add_argument("--paper", type=Path, required=True)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--question")
    operation.add_argument("--request-file", type=Path, help="Versioned frontend request with explicit operation, Claim targets, and optional Mapping bindings")
    parser.add_argument("--target-claim-id", action="append", default=[], help="Explicit Claim target; repeat for a selected set")
    parser.add_argument("--analysis-operation", choices=["auto", "explain_saved", "compare_saved", "analyze_claims", "refresh_claims"], default="auto")
    operation.add_argument("--resume-workflow")
    operation.add_argument("--selection-file", type=Path,
                           help="JSON with exact highlighted Paragraph.text ranges (Claim-only)")
    operation.add_argument("--analyze-selection-file", type=Path,
                           help="Analyze all Claims in selected text with whole-paper Evidence and Mapping")
    parser.add_argument("--fresh-analysis", action="store_true",
                        help="Permanently clear this paper's previous analysis before a new question")
    parser.add_argument("--question-limit-usd", type=float, default=None)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--language", choices=["en", "zh"], default="en")
    parser.add_argument("--image-mode", choices=["none", "relevant", "all"], default="relevant")
    parser.add_argument("--focus-section", action="append", default=[])
    parser.add_argument("--focus-element", action="append", default=[])
    parser.add_argument("--collection-rounds", type=int, default=MAX_COLLECTION_ROUNDS)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--use-saved-sources", action="store_true")
    parser.add_argument("--strict-evidence-coverage", action="store_true",
                        help="Opt-in: require completed scoped Evidence scans before Mapping")
    args = parser.parse_args()
    if args.request_file:
        if not args.request_file.is_file():
            parser.error("The request JSON does not exist.")
        if (args.target_claim_id or args.analysis_operation != "auto" or args.focus_section or args.focus_element or args.fresh_analysis):
            parser.error("A request file owns its operation and scope; do not combine competing scope flags.")
        args.request_file = args.request_file.resolve()
    if (args.target_claim_id or args.analysis_operation != "auto") and not args.question:
        parser.error("Target/operation CLI flags require --question. Resume retains its stored request.")
    if args.fresh_analysis and (args.target_claim_id or args.analysis_operation != "auto"):
        parser.error("Do not delete source pools for an operation on existing Claims.")
    if args.strict_evidence_coverage and (args.selection_file or args.analyze_selection_file):
        parser.error("--strict-evidence-coverage applies to question workflows only.")
    if not args.paper.is_file():
        parser.error("Standardized paper.json not found.")
    if not args.preview and not (args.question or args.request_file or args.resume_workflow or args.selection_file or args.analyze_selection_file):
        parser.error("Specify --question, --resume-workflow, --selection-file or --analyze-selection-file.")
    if args.selection_file:
        if not args.selection_file.is_file():
            parser.error("The --selection-file JSON does not exist.")
        if args.fresh_analysis or args.plan_only or args.use_saved_sources or args.focus_section or args.focus_element:
            parser.error("Text selection is explicit Claim-only scope; do not combine it with workflow scope/recovery flags.")
    if args.analyze_selection_file:
        if not args.analyze_selection_file.is_file():
            parser.error("The --analyze-selection-file JSON does not exist.")
        if (args.fresh_analysis or args.plan_only or args.use_saved_sources
                or args.focus_section or args.focus_element):
            parser.error("Selected Analyze uses its own Claim scope; do not combine it with recovery/scope flags.")
    if args.fresh_analysis and (not args.question or args.preview or args.plan_only):
        parser.error("--fresh-analysis requires a new --question, not preview/plan-only/resume.")
    if args.use_saved_sources and not args.resume_workflow:
        parser.error("--use-saved-sources is an explicit recovery mode, requiring --resume-workflow.")
    if args.question is not None and not args.question.strip():
        parser.error("--question must contain text.")
    if args.collection_rounds < 1:
        parser.error("--collection-rounds must be positive.")
    if args.question_limit_usd is not None and (not 0 < args.question_limit_usd < float('inf')):
        parser.error("--question-limit-usd must be finite and positive.")
    # Resolve a relative input before changing the working directory.
    args.paper = args.paper.resolve()
    if args.selection_file:
        args.selection_file = args.selection_file.resolve()
    if args.analyze_selection_file:
        args.analyze_selection_file = args.analyze_selection_file.resolve()
    os.chdir(PROJECT_ROOT)
    if args.analyze_selection_file:
        run_selected_analysis(args)
        return
    if args.resume_workflow and not args.preview:
        paper = PaperDocument.model_validate_json(args.paper.read_text(encoding="utf-8-sig"))
        old = find_saved_run(AnalysisStore(default_analysis_path(paper.paper_id)).load(paper),
                             args.resume_workflow)
        if old.get("mode") == "selected_analysis":
            run_selected_analysis(args, resume_id=args.resume_workflow)
            return
    if args.selection_file:
        from app.selection_claims import run_selection
        selection = json.loads(args.selection_file.read_text(encoding="utf-8-sig"))
        result = run_selection(
            paper_path=args.paper, selection=selection,
            model=args.model, limit_usd=args.question_limit_usd or 0.50,
            preview=args.preview,
        )
        print("\n=== SELECTED TEXT CLAIMS ===", flush=True)
        print("Status:", result.get("status", "preview"), flush=True)
        print("Cache hit:", result["cache_hit"], flush=True)
        print("Claims:", len(result.get("claim_ids", [])), flush=True)
        print("Analysis JSON:", default_analysis_path(result["paper_id"]), flush=True)
        if result.get("result_file"):
            print("Export JSON:", result["result_file"], flush=True)
        if result.get("answer"):
            print("\n" + result["answer"])
        return
    run_workflow(args)


if __name__ == "__main__":
    main()
