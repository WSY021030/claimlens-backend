

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator, model_serializer

from app.models.document_v2 import SourceAnchor
from app.core.retired_associations import clean_analysis_payload


ClaimType = Literal[
    "assertion",
    "method",
    "result",
    "interpretation",
    "limitation",
]

ClaimScope = Literal["background", "this_study"]

EvidenceKind = Literal[
    "paragraph",
    "table",
    "figure",
    "equation",
]


class RawClaim(BaseModel):
    source_element_id: str
    type: ClaimType
    scope: ClaimScope
    source_quote: str


class ClaimExtraction(BaseModel):
    claims: list[RawClaim]


class ClaimWindow(BaseModel):
    window_id: str
    element_id: str
    section_id: str
    section_title: str
    start: int
    end: int
    text: str


class ClaimRecord(BaseModel):
    claim_id: str
    type: ClaimType
    scope: ClaimScope | None = None

    source_element_id: str
    model_quote: str
    source_quote: str
    source_anchor: SourceAnchor | None = None

    source_status: Literal[
        "located", "unlocated"
    ] = "unlocated"

    alignment_note: str | None = None
    created_run_id: str




class RawEvidence(BaseModel):
    kind: EvidenceKind
    source_element_id: str

    source_quote: str | None = None
    cell_ids: list[str] = Field(default_factory=list)

    description: str

    # Only populated when actual image pixels were supplied.
    visual_observation: str | None = None



class EvidenceExtraction(BaseModel):
    evidence: list[RawEvidence]

    # Figures or image tables worth checking visually.
    image_candidates: list[str] = Field(
        default_factory=list
    )


class EvidenceRecord(BaseModel):
    evidence_id: str
    kind: EvidenceKind

    source_element_id: str
    source_anchor: SourceAnchor | None = None

    source_status: Literal[
        "located", "unlocated"
    ]

    source_note: str | None = None
    source_quote: str | None = None

    # Actual paper content assembled by Python.
    source_content: dict[str, Any] = Field(
        default_factory=dict
    )

    # Model-generated description, separate from source data.
    description: str

    observed_via: Literal[
        "text",
        "structured_table",
        "caption_only",
        "image_input",
        "equation",
    ]

    source_asset_id: str | None = None
    image_receipt: dict[str, Any] | None = None

    created_run_id: str




class AnalysisDocument(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def discard_retired_candidate_fields(cls, value):
        # Old files remain readable; the removed relation pool is never hydrated.
        return clean_analysis_payload(value) if isinstance(value, dict) else value

    @model_serializer(mode="wrap")
    def serialize_current_contract(self, handler):
        # Applies to every model_dump/model_dump_json caller, not one CLI only.
        return clean_analysis_payload(handler(self))

    schema_version: Literal["reader.v3"] = "reader.v3"

    paper_id: str
    paper_revision: str

    # Persistence metadata; not model-generated scientific content.
    storage_policy: str | None = None
    snapshot_id: str | None = None
    id_counters: dict[str, int] = Field(default_factory=dict)
    evidence_scan_batches: list[dict[str, Any]] = Field(default_factory=list)

    claim_pool: dict[str, ClaimRecord] = Field(
        default_factory=dict
    )

    evidence_pool: dict[str, EvidenceRecord] = Field(
        default_factory=dict
    )


    completed_claim_windows: dict[str, str] = Field(
        default_factory=dict
    )

    completed_evidence_scans: dict[str, str] = Field(
        default_factory=dict
    )

    # Preserve image candidates across interrupted runs.
    evidence_image_candidates: dict[str, list[str]] = Field(
        default_factory=dict
    )

    evidence_search_by_claim: dict[
        str, dict[str, Any]
    ] = Field(default_factory=dict)

    direct_evidence_searches: dict[str, dict[str, Any]] = Field(
        default_factory=dict
    )

    runs: list[dict[str, Any]] = Field(
        default_factory=list
    )

    summaries: dict[str, Any] = Field(
        default_factory=dict
    )

    mappings: dict[str, Any] = Field(default_factory=dict)

    # One current Mapping pointer per Claim. Superseded unbound versions are pruned.
    mapping_by_claim: dict[str, str] = Field(default_factory=dict)

    # Cross-Claim interpretations are not part of a single Claim Mapping.
    mapping_relations: dict[str, Any] = Field(default_factory=dict)

    # Only an actively referenced legacy batch remains until its request ends.
    mapping_batch_archive: dict[str, Any] = Field(default_factory=dict)

MappingEvidenceRelationType = Literal[
    "supports", "contradicts", "qualifies", "context", "insufficient_support"
]


MappingSourcePart = Literal[
    "source_text", "source_context", "caption", "visual_observation",
    "table_cells", "equation",
]


class MappingEvidenceRelation(BaseModel):
    evidence_id: str
    relation: MappingEvidenceRelationType
    explanation: str = Field(description="Source-grounded role of this Evidence for this Claim.")
    source_parts_used: list[MappingSourcePart] = Field(
        default_factory=list,
        description=(
            "Nonempty list of the actual components used for this relation, chosen "
            "from that Evidence's available_source_parts. A caption is not a pixel "
            "observation. Name only components used, not everything available."
        ),
    )


class MappingClaimAnalysis(BaseModel):
    claim_id: str
    explanation: str
    evidence_ids_used: list[str]
    qualifications: list[str]
    evidence_relations: list[MappingEvidenceRelation] = Field(default_factory=list)
    material_status: Literal[
        "sufficient_for_scoped_analysis", "limited"
    ] = "limited"


class MappingExtraction(BaseModel):
    claim_analyses: list[MappingClaimAnalysis]
    cross_claim_explanation: str
