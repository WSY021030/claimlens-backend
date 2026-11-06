
import json

from app.budget import PaperBudget
from app.llm import LazyOpenAI
from app.models.reader_v3 import (
    ClaimExtraction,
    ClaimWindow,
)
from app.usage_cost import CostTracker, MeteredClient


CLAIM_INSTRUCTIONS = """
You are a Claim Extraction Agent for an academic paper understanding system.

Extract the author's DISTINCT, MEANINGFUL RESEARCH PROPOSITIONS. A Claim should
help a reader understand a research question, central idea, important method,
empirical finding, interpretation, or limitation. Do NOT annotate every sentence.

VALUE TEST
- Keep a proposition when removing it would hide an important scientific
  contribution, consequential design choice, substantive finding, meaningful
  trade-off or exception, interpretation, or boundary of applicability.
- Include meaningful negative results, qualifications, and comparisons even
  when they weaken or complicate the paper's main conclusion.
- Do not keep a statement merely because it is factual, technical, or contains
  a number. The question is whether it contributes to understanding the paper.

Classify each Claim into one of these five types:
- assertion: a consequential thesis, hypothesis, or scientific judgment.
- method: a central method, meaningful design choice, or important rationale.
- result: a substantive empirical finding, comparison, or negative result.
- interpretation: the author's significant explanation of a finding or its
  scientific implications.
- limitation: a consequential limitation, assumption, uncertainty, or boundary.

GRANULARITY: COHERENT, NOT ATOMIC
- Prefer ONE Claim for a coherent finding even when it includes several
  related metrics, datasets, baselines, or qualifications. Keep necessary
  specifics that make the finding scientifically meaningful.
- Split only when the passage makes DIFFERENT, INDEPENDENTLY IMPORTANT research
  propositions that deserve separate explanation or evidence. For example,
  an overall advantage and a consequential opposing finding may be distinct.
- Do NOT split simply because multiple numbers, metrics, methods, or sentences
  appear. Do NOT combine unrelated findings merely because they share a paragraph.
- If an author explains an empirical result, retain the explanation as its
  own Claim only when it adds a substantive causal or theoretical insight;
  otherwise keep it with the finding or omit a redundant restatement.
- Choose the type that best describes the main proposition. If a method and a
  hypothesis are both independently important, they may be separate Claims;
  do not split them automatically.

EXCLUDE AS STANDALONE CLAIMS
- Routine implementation steps, component descriptions, parameter definitions,
  common knowledge, procedural setup, generic motivation, and section pointers.
- Mere figure/table announcements or restatements of an already extracted
  proposition that add no independent scientific insight.
- Vague statements such as 'the results support our hypothesis' when the
  underlying result is already extracted and no new conclusion is given.
- Generic background facts unless essential to the paper's scientific argument.

SOURCE ACCURACY AND RESPONSIBILITY
- For each Claim, provide its original source_element_id and ONE verbatim,
  contiguous source_quote from the supplied passage. Prefer the shortest
  self-contained quotation that preserves the meaningful proposition.
- Never invent, paraphrase, or stitch source quotes across separate spans.
  Preserve numbers, units, comparison direction, negations, and qualifications.
- Do not repeat the same long paragraph as multiple Claims with different
  labels; use distinct, appropriately scoped source quotations.
- Extract Claims whether or not the supplied passage contains adequate proof.
  Do NOT evaluate their truth or sufficiency and do NOT retrieve Evidence.
  Evidence and Mapping stages handle those responsibilities.
- A passage may contain ZERO Claims; there is no numerical target, quota,
  or fixed maximum. Favor significance and coverage over Claim count.

Return ClaimExtraction with only the meaningful distinct propositions.
"""

CLAIM_SCOPE_INSTRUCTIONS = """
Classify each retained Claim by scope:

background:
  A substantive research premise, prior-work finding, or theoretical
  assertion needed to understand the present paper's contribution.

this_study:
  The current paper's core methods, design rationale, hypotheses, findings,
  interpretations, and acknowledged limitations.

Scope is a classification, not an Evidence quality test.
Do not automatically include generic background definitions or textbook facts.
Do not automatically exclude a consequential premise merely because it is
from background material.
"""


class ClaimAgent:

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
        windows: list[ClaimWindow],
    ) -> ClaimExtraction:

        payload = {
            "task": "Extract meaningful, distinct research propositions at a coherent scientific granularity",
            "passages": [
                window.model_dump()
                for window in windows
            ],
        }

        request = {
            "model": self.model,
            "instructions": (
                CLAIM_INSTRUCTIONS
                + "\n"
                + CLAIM_SCOPE_INSTRUCTIONS
            ),
            "input": json.dumps(
                payload,
                ensure_ascii=False,
            ),
            "text_format": ClaimExtraction,
            "max_output_tokens": 4096,
            "store": False,
        }

        if self.model.startswith("gpt-5.4-mini"):
            request["reasoning"] = {"effort": "none"}

        response = self.client.responses.parse(**request)

        result = response.output_parsed

        if result is None:
            raise RuntimeError(
                "Claim extraction returned no structured result."
            )

        return result

