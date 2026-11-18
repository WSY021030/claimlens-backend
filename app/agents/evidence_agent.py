import json


from pydantic import BaseModel, Field

from app.budget import PaperBudget
from app.llm import LazyOpenAI
from app.models.reader_v3 import (
    ClaimRecord,
    EvidenceExtraction,
)
from app.tools.image_inputs import PreparedImage
from app.usage_cost import CostTracker, MeteredClient


EVIDENCE_INSTRUCTIONS = """
You are an Evidence Discovery Agent for academic papers.

You receive:
1. A batch of target Claims.
2. A chunk of source material from the same paper.
3. Optionally, one actual source image.

Your sole responsibility is to DISCOVER source material that is specifically
relevant to at least one input Claim. Claims define the scientific target;
source chunks and the rest of the paper are search locations, NOT a request
for general extraction. Do not return Claim/Evidence pairs, relevance labels,
support judgments, or separate relevance explanations.

For each target Claim, pay attention to its specific task, dataset/population,
conditions, comparators, measurements and conclusion scope. Examine each
source chunk using that focus. Extract only findings that inform those details:
- Results, observations or measurements of the targeted task and conditions.
- Method definitions, datasets, controls and assumptions needed to interpret
  the targeted result, even if described in a shared Methods section.
- Relevant qualifications, limitations, counterexamples and contradictory
  results that concern that same target, including its stated generality.
- Cross-task findings ONLY if the Claim explicitly compares tasks, or the paper
  explicitly uses that finding to qualify the targeted result.

Exclude a finding if its only connection is a shared model, method, metric,
plot type, research area, paper or section. Experiments on OTHER tasks must
not be collected as background merely because they appear nearby. In mixed
paragraphs, quote only the directly relevant contiguous portion. In multi-
panel figures or tables, inspect and extract only the relevant panels/rows.
An empty evidence list is the correct output for an unrelated chunk.
The section_title can resolve context-dependent sentences such as 'unlike
previous settings' or 'this result'; do not discard a target-task result
merely because the individual sentence does not repeat the task name.
When several input elements are in one chunk, do not extract an unrelated
element just because a nearby element is relevant.
When only one source element is supplied, examine its entire content before
returning. Do not overlook a distinct task-specific result or comparison
merely because the preceding Claim already states a broad conclusion.
Anaphoric wording can still contain independent experimental information
when the supplied section title and target context resolve its subject.
An empty result is valid when the element has no such material.

Preserve relevant contrary evidence just as carefully as favorable evidence.
Do not decide whether any Evidence actually proves the Claim: Mapping alone
performs that analysis. No Claim-Evidence Links are created or stored.

For each finding:
- Return its source element_id and kind.
- For paragraph findings, source_quote must be an exact,
  contiguous substring of the supplied source text.
  Never paraphrase, normalize, combine fragments, or
  modify numbers, negations, metrics, or conditions.
  If no exact quotation is identifiable, use null.
  Python recovers the source text and records whether
  precise quotation alignment was established.
- For tables, identify relevant cell_ids.
- Preserve one factual source record per relevant substantive finding.
- Include material qualifications and potentially opposing results concerning
  the target Claims, not unrelated negative findings from other tasks.
- Do not omit relevant material because it is unfavorable to the target Claims.
- Treat target Claims as search requests, never as source evidence.
  Cite an element only if its ID is present in the supplied source_elements.
  Do not duplicate a target Claim's own quoted statement as if it were
  an independent source of experimental Evidence. Other source material,
  including non-overlapping details from its paragraph, may qualify.
- For a table, favor one compact finding with relevant source cell IDs
  over repeated records for individual cells. Preserve necessary comparisons.

STRICT DIVISION OF RESPONSIBILITIES:

You DO NOT:
- Evaluate scientific correctness.
- Determine whether the evidence proves a Claim.
- Construct reasoning chains.
- Evaluate causal or logical validity.
- Reject related material because it is indirect,
  uncertain, contradictory, or incomplete.
- Invent source IDs, cell IDs, or quotations.

The Mapping Agent is solely responsible for
reasoning about the collected evidence.

Your description must summarize source contents,
not claim that the source proves a conclusion.

Candidate-association output is not part of the extraction schema.

IMAGE RULES:

If actual image pixels are NOT supplied:
- Use only captions and alt text.
- Do not invent visual observations.
- Suggest image element IDs only when their depicted data could inform an
  input Claim, not just because they contain a similar type of chart.

If actual image pixels ARE supplied:
- Inspect only panels relevant to the target Claims; do not report data from
  unrelated tasks, even when those panels are clearly readable.
- Identify each relevant panel, metric, axis, unit and legend before comparing results.
- Keep methods, datasets and panels separate.
- Distinguish printed numeric labels from visually estimated positions.
- Report exact numbers only when they are clearly readable.
- Distinguish the numeric direction of an axis or measurement from the
  semantic direction of difficulty, risk or subtlety. State observable axis
  changes; do not invent a higher/lower difficulty interpretation if the
  image does not establish it.
- For plots without readable numbers, describe only defensible qualitative patterns.
- Do not infer overall rankings from one axis, color or polygon area.
- State important unreadable or ambiguous details briefly.
- Put source observations in visual_observation, without evaluating Claim truth.

EFFICIENCY:

Return short source quotations and relevant cell IDs.
Do not reproduce entire tables or lengthy passages.
Python will retrieve source content after this call.

Return all identifiable relevant findings in the
current chunk, not just the strongest finding.

If no material is related, return empty evidence.
"""


class EvidenceSourceSelection(BaseModel):
    """Transient source-element IDs selected before Evidence extraction."""
    element_ids: list[str] = Field(default_factory=list)


class EvidenceTopicReview(BaseModel):
    """Transient admission decision; never stored as Claim-Evidence relations."""
    accepted_finding_indices: list[int]
    image_element_ids: list[str]


SOURCE_SELECTION_INSTRUCTIONS = """
You are the source-locating stage of the Evidence Agent.
You receive target academic Claims and a catalog of actual paper elements.
Choose source ELEMENT IDs that are potentially useful for investigating the
specific scientific content of at least one target Claim. You are not asked to
extract Evidence, create Claim/Evidence relations, or evaluate support.

Read the experimental subject carefully: task, dataset/population, outcome,
compared conditions, qualifiers and limitations. Include measurements,
original figures/tables, relevant methods, and actual counterexamples for that
same subject. Include shared methods when necessary to interpret the result.
Include summaries spanning multiple tasks if they explicitly cover the target.
Include same-topic passages with anaphora (e.g. 'unlike the other settings')
when their section title and neighboring context identify the target task.

Do NOT select an experiment about another task merely because it uses the same
models, metrics, visualizations, or broad research theme. Other-task results
are relevant only if the Claim compares tasks or the paper explicitly uses
those results to qualify the targeted Claim. Do not select general background
with no concrete bearing on the Claim.

Source selection is a HIGH-RECALL screening operation, not admission into the
Evidence Pool. If unsure about a potentially relevant method or target-task
result, inspect it. But do NOT select a clearly different experimental task
merely because it reports a similar visualization benefit or uses the same
models. The target Claim's source section and full paragraph in target_context
specify what the Claim is actually about. Return only existing element IDs.
"""


TOPIC_REVIEW_INSTRUCTIONS = """
You are the final topic-admission step WITHIN an academic Evidence Agent.
You are NOT a scientific correctness judge, Mapping Agent, or link builder.

Inputs contain (1) target Claims with their original paragraph and section,
(2) the source elements just inspected and their section titles, and
(3) zero-indexed candidate findings from an Evidence extraction.

Decide for EACH finding independently whether it belongs to the SPECIFIC
scientific investigation represented by at least one target Claim.

Admit a finding if its source concerns:
- The Claim's actual task/population/dataset, compared conditions, measurements,
  experimental results, or relevant contradicting observations;
- Shared methodology or definitions genuinely needed to interpret that task;
- A paper-wide summary explicitly including the target task, or a limitation
  or counterexample that explicitly applies to it.

Reject standalone results from another experiment, even when the same model,
plot type, metric, condition names, and trend occur in both experiments.
A general thematic analogy is NOT evidence for a task-specific Claim.
A source from a different section can be admitted when it actually discusses
or qualifies the target task. Do not discard relevant contrary evidence.
Do not require evidence to agree with the Claim; do not evaluate its truth.

For image_element_ids: choose only the actual figure/table source elements
whose pixels could provide material about the target investigation. A source
being selected for text inspection is NOT sufficient reason to inspect its
image. Use source headings and captions to distinguish other experiments.

Return accepted_finding_indices as indices into candidate_findings (starting
at 0), and image_element_ids from the supplied source_elements only. Return
empty lists for unrelated content. Do not output explanations or Claim links.
"""


class EvidenceAgent:
    def __init__(
        self,
        model: str,
        tracker: CostTracker,
        budget: PaperBudget,
    ):
        self.model = model

        self.client = MeteredClient(
            LazyOpenAI(),
            tracker,
            budget,
        )


    def select_sources(
        self,
        claims: list[ClaimRecord],
        source_catalog: list[dict],
        target_context: list[dict] | None = None,
    ) -> EvidenceSourceSelection:
        """Search the source index for Claims; does not extract Evidence."""
        if not claims:
            raise ValueError("Source selection requires target Claims.")
        if not source_catalog:
            return EvidenceSourceSelection(element_ids=[])
        payload = {
            "target_claims": [
                {"claim_id": c.claim_id, "type": c.type, "scope": c.scope,
                 "text": c.source_quote or c.model_quote}
                for c in claims
            ],
            "source_catalog": source_catalog,
            "target_context": target_context or [],
        }
        request = {
            "model": self.model,
            "instructions": SOURCE_SELECTION_INSTRUCTIONS,
            "input": json.dumps(payload, ensure_ascii=False),
            "text_format": EvidenceSourceSelection,
            "max_output_tokens": 2500,
            "store": False,
        }
        if self.model.startswith("gpt-5.4-mini"):
            request["reasoning"] = {"effort": "none"}
        response = self.client.responses.parse(**request)
        result = response.output_parsed
        if result is None:
            raise RuntimeError("Source selection returned no parsed result.")
        return result

    def extract(
        self,
        claims: list[ClaimRecord],
        source_elements: list[dict],
        image: PreparedImage | None = None,
        target_context: list[dict] | None = None,
    ) -> tuple[EvidenceExtraction, object]:

        if not claims:
            raise ValueError(
                "EvidenceAgent requires at least one Claim."
            )

        if not source_elements:
            raise ValueError(
                "EvidenceAgent requires source material."
            )

        if any(
            not isinstance(claim, ClaimRecord)
            or not claim.claim_id
            or not claim.source_element_id
            for claim in claims
        ):
            raise ValueError(
                "EvidenceAgent received invalid Claims."
            )

        claim_data = [
            {
                "claim_id": c.claim_id,
                "type": c.type,
                "scope": c.scope,
                "text": c.source_quote or c.model_quote,
                "source_element_id": c.source_element_id,
            }
            for c in claims
        ]

        payload = {
            "target_claims": claim_data,
            "target_context": target_context or [],
            "source_elements": source_elements,
            "has_image_pixels": image is not None,
        }

        content = [
            {
                "type": "input_text",
                "text": json.dumps(
                    payload,
                    ensure_ascii=False,
                ),
            }
        ]

        if image is not None:
            content.append({
                "type": "input_text",
                "text": (
                    "Inspect the attached source image. "
                    f"element_id={image.element_id}; "
                    f"asset_id={image.asset_id}; "
                    f"cell_id={image.cell_id}. "
                    "Report only visible information relevant to the "
                    "target Claims."
                ),
            })

            content.append(image.input_part())

        request = {
            "model": self.model,
            "instructions": EVIDENCE_INSTRUCTIONS,
            "input": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "text_format": EvidenceExtraction,
            "max_output_tokens": 4000,
            "store": False,
        }

        if self.model.startswith("gpt-5.4-mini"):
            request["reasoning"] = {"effort": "none"}

        response = self.client.responses.parse(**request)

        result = response.output_parsed

        if result is None:
            raise RuntimeError(
                "Evidence extraction returned no parsed data."
            )

        return result, response

    def review_topic(
        self,
        claims: list[ClaimRecord],
        source_elements: list[dict],
        extraction: EvidenceExtraction,
        target_context: list[dict],
    ) -> tuple[EvidenceExtraction, dict]:
        """Topic admission after extraction; does not decide support or make Links.

        Invalid/missing decisions raise an error. The runner must then leave
        this chunk unscanned so a resumed request can retry it.
        """
        if not claims or not source_elements:
            raise ValueError("Topic review requires Claims and source elements.")
        media_ids = {
            item["element_id"] for item in source_elements
            if item.get("kind") in {"figure", "table"}
        }
        count = len(extraction.evidence)
        if not count and not media_ids:
            return extraction, {
                "candidate_findings": 0,
                "accepted_findings": 0,
                "excluded_source_elements": [],
                "reviewed_image_candidates": 0,
                "model_call": False,
            }
        compact_sources = [{
            "element_id": item["element_id"],
            "kind": item.get("kind"),
            "section_title": item.get("section_title"),
            "section_id": item.get("section_id"),
            "label": item.get("label"),
            "caption": str(item.get("caption") or "")[:700],
            "text_preview": str(item.get("text") or "")[:1100],
            "row_group_context": item.get("row_group_context"),
        } for item in source_elements]
        findings = [{
            "index": i,
            "element_id": raw.source_element_id,
            "kind": raw.kind,
            "source_quote": (raw.source_quote or "")[:1300],
            "description": raw.description[:850],
            "visual_observation": (raw.visual_observation or "")[:850],
        } for i, raw in enumerate(extraction.evidence)]
        payload = {
            "target_claims": [{
                "claim_id": claim.claim_id,
                "type": claim.type,
                "scope": claim.scope,
                "text": claim.source_quote or claim.model_quote,
            } for claim in claims],
            "target_context": target_context,
            "source_elements": compact_sources,
            "candidate_findings": findings,
            "extraction_image_candidates": extraction.image_candidates,
        }
        request = {
            "model": self.model,
            "instructions": TOPIC_REVIEW_INSTRUCTIONS,
            "input": json.dumps(payload, ensure_ascii=False),
            "text_format": EvidenceTopicReview,
            "max_output_tokens": 800,
            "store": False,
        }
        if self.model.startswith("gpt-5.4-mini"):
            request["reasoning"] = {"effort": "none"}
        response = self.client.responses.parse(**request)
        decision = response.output_parsed
        if decision is None:
            raise RuntimeError("Evidence topic review returned no parsed data.")
        indices = decision.accepted_finding_indices
        image_ids = decision.image_element_ids
        if (len(indices) != len(set(indices))
                or any(type(i) is not int or i < 0 or i >= count for i in indices)):
            raise ValueError("Topic review returned invalid finding indices.")
        if len(image_ids) != len(set(image_ids)) or not set(image_ids) <= media_ids:
            raise ValueError("Topic review returned invalid image element IDs.")
        keep = set(indices)
        admitted = EvidenceExtraction(
            evidence=[raw for i, raw in enumerate(extraction.evidence) if i in keep],
            image_candidates=list(image_ids),
        )
        return admitted, {
            "candidate_findings": count,
            "accepted_findings": len(admitted.evidence),
            "excluded_source_elements": sorted({
                raw.source_element_id for i, raw in enumerate(extraction.evidence)
                if i not in keep
            }),
            "reviewed_image_candidates": len(image_ids),
            "model_call": True,
        }
