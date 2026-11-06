
import json


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

Your sole responsibility is to FIND source material relevant to the target
Claims. Claims define a discovery focus only. Return source records,
not Claim/Evidence pairs, relevance relations, or support labels.
The standard workflow does not run a separate relevance-matching operation.

Search for concrete connections including:
- The same method, variable, experiment, or dataset
- The same measurement or reported result
- Relevant method details or assumptions
- Explicit references to tables, figures, equations
- Important context or potentially contrasting data

For each finding:
- Return its source element_id and kind.
- For text, copy a short exact source_quote when possible.
- For tables, identify relevant cell_ids.
- Preserve one factual source record per substantive finding.
- Include observations, methods, qualifications and potentially opposing results.
- Do not omit material merely because it is not favorable to the target Claims.
- Treat target Claims as search requests, never as source evidence.
  Cite an element only if its ID is present in the supplied source_elements.
  Do not copy a target Claim's own text as Evidence when that text does not
  actually appear in the supplied source_elements or attached image.
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
- You may use the caption and alt text.
- Do not invent visual observations.
- Suggest image element IDs in image_candidates
  when visual inspection could find relevant material.

If actual image pixels ARE supplied:
- You may describe visible data or structures.
- Put those observations in visual_observation.
- State unreadable values as unreadable.
- Do not invent precise plot values.

EFFICIENCY:

Return short source quotations and relevant cell IDs.
Do not reproduce entire tables or lengthy passages.
Python will retrieve source content after this call.

Return all identifiable relevant findings in the
current chunk, not just the strongest finding.

If no material is related, return empty evidence.
"""


DIRECT_EVIDENCE_INSTRUCTIONS = """
You are the Evidence Agent in direct source-discovery mode.
The user requests paper-source material, not an evaluation of author Claims.
Find source excerpts, tables, figures, methods, measurements, or observed
results relevant to the supplied source_focus, using ONLY the supplied chunk
and any attached original source image.

Return EvidenceExtraction with source element IDs, brief factual descriptions,
short exact quotations for text and relevant table cell_ids when present.
Return source records only; no target Claims have been provided.
Return all relevant source records, including conflicting or uncertain ones.
Do not create Claims, do not judge support or correctness, and do not infer
unobserved results. If there is no relevant source, return empty evidence.
When no image pixels are attached, never fabricate visual readings; add
candidate figure/table IDs to image_candidates when needed. When an image
is attached, describe only what is clearly visible in visual_observation.
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


    def extract(
        self,
        claims: list[ClaimRecord],
        source_elements: list[dict],
        image: PreparedImage | None = None,
        source_focus: str | None = None,
    ) -> tuple[EvidenceExtraction, object]:

        claim_data = [
            {
                "claim_id": c.claim_id,
                "type": c.type,
                "scope": c.scope,
                "text": c.source_quote,
                "source_element_id": c.source_element_id,
            }
            for c in claims
        ]

        payload = {
            "target_claims": claim_data,
            "source_elements": source_elements,
            "has_image_pixels": image is not None,
        }
        if source_focus is not None:
            payload["source_focus"] = source_focus

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
                    "supplied source focus."
                ),
            })

            content.append(image.input_part())

        request = {
            "model": self.model,
            "instructions": (
                DIRECT_EVIDENCE_INSTRUCTIONS
                if source_focus is not None
                else EVIDENCE_INSTRUCTIONS
            ),
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
