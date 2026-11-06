"""Application-owned request boundaries; never inferred from paper content.

A target Claim set, a retrieval boundary, and a display binding are different
objects. The planner may propose work but cannot enlarge an explicit target set.
This module makes no API calls and never edits extracted source records.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.coordinator_v3 import CoordinatorPlan, CoordinatorTask, TaskScope, required_capabilities

REQUEST_POLICY = "explicit_targets_saved_results_v1"
SAVED_OPERATIONS = frozenset({"explain_saved", "compare_saved"})
FIXED_OPERATIONS = SAVED_OPERATIONS | {"analyze_claims", "refresh_claims"}
CLAIM_ID = re.compile(r"\bclaim_\d{6}\b", re.I)
SAVED_INTENT = re.compile(
    r"\b(?:using|use|explain|summarize|describe|show|compare)\b.{0,100}"
    r"\b(?:existing|saved|completed)\s+(?:mapping|analysis|analyses|results?)\b"
    r"|\b(?:existing|saved)\s+mapping\b|(?:解释|说明|总结|比较|展示).{0,20}(?:已有|现有|已保存).{0,12}(?:分析|结果|mapping)",
    re.I | re.S,
)
REFRESH_INTENT = re.compile(
    r"\b(?:reanalyze|re-analyze|recompute|re-compute|reassess|re-assess)\b"
    r"|(?:重新分析|重新评估|重新论证)", re.I,
)
BROAD_INTENT = re.compile(
    r"\b(?:all|other|additional)\s+(?:claims|sections)\b|\b(?:whole|entire)\s+paper\b"
    r"|(?:其他观点|所有观点|整篇|全文)", re.I,
)


def unique(values):
    return list(dict.fromkeys(value for value in values if value))


class RetrievalBoundary(BaseModel):
    """A user-imposed Evidence boundary, not a Claim selection."""
    model_config = ConfigDict(extra="forbid")
    mode: Literal["whole_paper", "selection"] = "whole_paper"
    section_ids: list[str] = Field(default_factory=list)
    element_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent(self):
        self.section_ids = unique(self.section_ids)
        self.element_ids = unique(self.element_ids)
        if self.mode == "whole_paper" and (self.section_ids or self.element_ids):
            raise ValueError("whole_paper must not carry a narrower source list.")
        if self.mode == "selection" and not (self.section_ids or self.element_ids):
            raise ValueError("A selected retrieval boundary needs actual source IDs.")
        return self


class ReaderRequest(BaseModel):
    """Frontend/CLI input. Operation permissions are owned by the application."""
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["reader.request.v1"] = "reader.request.v1"
    paper_id: str
    paper_revision: str
    question: str
    operation: Literal["auto", "explain_saved", "compare_saved", "analyze_claims", "refresh_claims"] = "auto"
    target_claim_ids: list[str] = Field(default_factory=list)
    mapping_ids_by_claim: dict[str, str] = Field(default_factory=dict)
    retrieval_scope: RetrievalBoundary = Field(default_factory=RetrievalBoundary)

    @model_validator(mode="after")
    def consistent(self):
        self.question = self.question.strip()
        if not self.question:
            raise ValueError("The request question must not be empty.")
        self.target_claim_ids = unique(cid.casefold() for cid in self.target_claim_ids)
        if any(CLAIM_ID.fullmatch(cid) is None for cid in self.target_claim_ids):
            raise ValueError("target_claim_ids must contain canonical Claim IDs.")
        if self.operation in FIXED_OPERATIONS and not self.target_claim_ids:
            raise ValueError("This operation requires explicit target_claim_ids.")
        if self.operation == "compare_saved" and len(self.target_claim_ids) < 2:
            raise ValueError("compare_saved requires at least two target Claims.")
        if not set(self.mapping_ids_by_claim) <= set(self.target_claim_ids):
            raise ValueError("Pinned Mapping keys must belong to the target Claim set.")
        if self.mapping_ids_by_claim and self.operation not in SAVED_OPERATIONS:
            raise ValueError("Pin Mapping versions only when reading saved analyses.")
        return self


def validate_request(request, paper, data):
    if request.paper_id != paper.paper_id or request.paper_revision != paper.revision:
        raise ValueError("Request and open paper revisions differ; refresh the reader.")
    unknown = set(request.target_claim_ids) - set(data.claim_pool)
    if unknown:
        raise ValueError(f"Unknown selected Claims: {sorted(unknown)}")
    boundary = request.retrieval_scope
    if set(boundary.section_ids) - {s.section_id for s in paper.sections}:
        raise ValueError("Unknown user-selected Evidence section.")
    if set(boundary.element_ids) - set(paper.elements):
        raise ValueError("Unknown user-selected Evidence element.")
    for cid, mid in request.mapping_ids_by_claim.items():
        record = data.mappings.get(mid)
        if not isinstance(record, dict) or record.get("claim_id") != cid:
            raise ValueError(f"Pinned Mapping {mid} does not belong to {cid}.")
    return request


def resolve_request(args, paper, data, *, saved_run=None):
    """Prefer explicit interface state. Text fallback is deliberately narrow."""
    if saved_run and saved_run.get("request_contract"):
        contract = saved_run["request_contract"]
        request = ReaderRequest.model_validate(contract["request"])
        validate_request(request, paper, data)
        if request.question != str(saved_run.get("user_question") or "").strip():
            raise ValueError("Stored request and original budgeted question differ.")
        return contract
    request_file = getattr(args, "request_file", None) if saved_run is None else None
    origin = "structured_request" if request_file else "question"
    if request_file:
        request = ReaderRequest.model_validate_json(Path(request_file).read_text(encoding="utf-8-sig"))
    else:
        question = (saved_run or {}).get("user_question") or getattr(args, "question", "") or ""
        targets = unique(getattr(args, "target_claim_id", []) or []) if saved_run is None else []
        operation = getattr(args, "analysis_operation", "auto") if saved_run is None else "auto"
        if targets:
            origin = "explicit_cli_targets"
        negative_saved = bool(re.search(r"(?:ignore|disregard|do not use|don't use).{0,30}(?:existing|saved).{0,15}(?:mapping|analysis)|不(?:要|再)使用.{0,10}(?:已有|现有)", question, re.I))
        if operation == "auto" and SAVED_INTENT.search(question) and not REFRESH_INTENT.search(question) and not BROAD_INTENT.search(question) and not negative_saved:
            named = list(targets) or unique(match.group().casefold() for match in CLAIM_ID.finditer(question))
            if named:
                targets = named
                operation = "compare_saved" if len(named) > 1 and re.search(r"compare|比较|对比", question, re.I) else "explain_saved"
                origin = "explicit_saved_analysis_phrase"
        boundary = {"mode": "whole_paper"}
        if saved_run is None and (getattr(args, "focus_section", []) or getattr(args, "focus_element", [])):
            boundary = {"mode": "selection", "section_ids": args.focus_section, "element_ids": args.focus_element}
        request = ReaderRequest(paper_id=paper.paper_id, paper_revision=paper.revision,
                                question=question, operation=operation,
                                target_claim_ids=targets, retrieval_scope=boundary)
    validate_request(request, paper, data)
    return {"policy": REQUEST_POLICY, "origin": origin,
            "targets_locked": bool(request.target_claim_ids),
            "request": request.model_dump(mode="json")}


def get_request(run):
    contract = run.get("request_contract") or {}
    payload = contract.get("request")
    return ReaderRequest.model_validate(payload) if payload else None


def locked_targets(run):
    request = get_request(run)
    return list(request.target_claim_ids) if request else []


def saved_only(run):
    request = get_request(run)
    return bool(request and request.operation in SAVED_OPERATIONS)


def deterministic_plan(contract, paper, data, language):
    request = ReaderRequest.model_validate(contract["request"])
    if request.operation not in FIXED_OPERATIONS:
        return None
    task = CoordinatorTask(
        task_id="t1", intent="explain_argument", goal=request.question,
        scope=TaskScope(mode="selection", section_ids=[],
                        element_ids=unique(data.claim_pool[cid].source_element_id for cid in request.target_claim_ids),
                        claim_ids=list(request.target_claim_ids), user_locked=False),
        evidence_scope="whole_paper", mapping_focus="claim_support", output_style="structured")
    return CoordinatorPlan(paper_id=paper.paper_id, user_message=request.question,
                           language=language, tasks=[task])


def enforce_plan_targets(plan, run, data):
    """An explicit interface target cannot be enlarged by a planner decision."""
    targets = locked_targets(run)
    if not targets:
        return
    assigned = set()
    argument_tasks = [task for task in plan.tasks if "mapping" in required_capabilities(task.intent)]
    for task in argument_tasks:
        chosen = [cid for cid in task.scope.claim_ids if cid in targets]
        if len(argument_tasks) == 1 or not chosen:
            chosen = list(targets)
        task.scope.claim_ids = unique(chosen)
        task.scope.mode = "selection"
        task.scope.section_ids = []
        task.scope.element_ids = unique(data.claim_pool[cid].source_element_id for cid in chosen)
        # Target locking is not a prohibition on reading Evidence elsewhere.
        task.scope.user_locked = False
        assigned.update(chosen)
    if argument_tasks and assigned != set(targets):
        raise ValueError("The plan does not cover every explicitly selected Claim.")


def evidence_boundary(run, paper):
    """Return None for whole paper, a set for a real user source restriction."""
    request = get_request(run)
    if request is None:
        return None
    boundary = request.retrieval_scope
    if boundary.mode == "whole_paper":
        return None
    sections = set(boundary.section_ids)
    while True:
        grown = sections | {s.section_id for s in paper.sections if s.parent_section_id in sections}
        if grown == sections:
            break
        sections = grown
    return set(boundary.element_ids) | {eid for eid, element in paper.elements.items() if element.section_id in sections}
