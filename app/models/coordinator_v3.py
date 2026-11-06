from typing import Literal

from pydantic import BaseModel, Field


TaskIntent = Literal[
    "fact_lookup",
    "list_claims",
    "summarize_claims",
    "summarize",
    "show_evidence",
    "summarize_evidence",
    "explain_argument",
    "evaluate_argument",
    "check_conflicts",
    "check_assumptions",
    "cross_section",
    "external_question",
    "other",
]

ScopeMode = Literal[
    "auto",
    "selection",
    "section",
    "related_sections",
    "whole_paper",
]

EvidenceScope = Literal[
    "auto",
    "none",
    "selection",
    "related_sections",
    "whole_paper",
]

MappingFocus = Literal[
    "general",
    "claim_support",
    "data_consistency",
    "experimental_comparability",
    "scope_and_generalization",
    "method_result_relationship",
]


class TaskScope(BaseModel):
    """Defines the document scope of a user task."""

    mode: ScopeMode = "auto"

    section_ids: list[str] = Field(default_factory=list)
    element_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)

    # Prevent automatic scope expansion when the user sets a strict boundary.
    user_locked: bool = False


class CoordinatorTask(BaseModel):
    """Represents one independently executable user task."""

    task_id: str
    intent: TaskIntent
    goal: str

    scope: TaskScope = Field(default_factory=TaskScope)
    evidence_scope: EvidenceScope = "auto"

    # Optional focus for future Mapping orchestration.
    # It does not grant Mapping access to document retrieval.
    mapping_focus: MappingFocus | None = None

    output_style: Literal[
        "direct",
        "summary",
        "detailed",
        "structured",
    ] = "direct"


class CoordinatorPlan(BaseModel):
    """Represents the plan for one user message."""

    paper_id: str | None = None
    user_message: str

    language: Literal["zh", "en"] = "zh"

    # Multiple tasks support mixed or multi-part user requests.
    tasks: list[CoordinatorTask] = Field(min_length=1)


class StageResult(BaseModel):
    """Records the execution outcome of one pipeline stage."""

    stage: Literal[
        "ingestion",
        "claim",
        "evidence",
        "mapping",
    ]

    status: Literal[
        "completed",
        "partial",
        "reused",
        "blocked",
        "failed",
    ]

    run_id: str | None = None

    created_ids: list[str] = Field(default_factory=list)
    reused_ids: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


def required_capabilities(
    intent: TaskIntent,
) -> tuple[str, ...]:
    """
    Return the possible capabilities required by a task.

    This function does not execute any pipeline stage.
    Actual execution depends on existing data, coverage,
    task requirements, and budget availability.
    """

    if intent == "fact_lookup":
        # Direct Evidence discovery also handles simple source fact lookups.
        return ("evidence",)

    if intent in {"list_claims", "summarize_claims"}:
        return ("claim",)

    if intent in {"show_evidence", "summarize_evidence"}:
        return ("evidence",)

    if intent == "summarize":
        return ("claim", "evidence")

    if intent in {
        "explain_argument",
        "evaluate_argument",
        "check_conflicts",
        "check_assumptions",
        "cross_section",
    }:
        return ("claim", "evidence", "mapping")

    return ()