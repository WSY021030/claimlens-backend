"""Read-only full-workflow budget, cache and evidence-quality preflight.

Usage: python -m app.workflow_audit --paper path/to/paper.json \
       --analysis path/to/analysis.json --workflow-id RUN_ID [--claim-id CLAIM_ID ...]

The forecast is deliberately labeled: future Evidence, Mapping and Answer
outputs cannot be known before execution; exact billing is never promised.
"""
from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from app.agents.coordinator_agent import ANSWER_INSTRUCTIONS, CoordinatorReply
from app.agents.evidence_agent import EVIDENCE_INSTRUCTIONS
from app.agents.mapping_agent import MAPPING_INSTRUCTIONS
from app.budget import BudgetStop, prepare_request
from app.coordinator_runner import find_saved_run
from app.claim_runner import make_windows
from app.evidence_runner import (
    batches, historical_scan_index, make_chunks, source_scoped_chunks,
)
from app.mapping_runner import make_mapping_input
from app.models.coordinator_v3 import CoordinatorPlan
from app.models.document_v2 import PaperDocument
from app.models.reader_v3 import AnalysisDocument, EvidenceExtraction, MappingExtraction
from app.usage_cost import _prices_for


def approx_reservation(request):
    """Apply the exact local guard's reservation formula without writing."""
    _, amount, _ = prepare_request(request)
    return amount


def mapping_reservation(payload, model, language):
    instructions = MAPPING_INSTRUCTIONS + ("\nWrite explanations in clear English."
        if language == "en" else "\nWrite explanations in clear Chinese.")
    request = {
        "model": model, "instructions": instructions,
        "input": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        "text_format": MappingExtraction, "max_output_tokens": 4096,
        "store": False,
    }
    if model.startswith("gpt-5.4-mini"):
        request["reasoning"] = {"effort": "none"}
    return approx_reservation(request)


def evidence_reservation(claims, elements, model):
    content = [{"type": "input_text", "text": json.dumps({
        "target_claims": [{
            "claim_id": c.claim_id, "type": c.type, "scope": c.scope,
            "text": c.source_quote, "source_element_id": c.source_element_id,
        } for c in claims],
        "source_elements": elements, "has_image_pixels": False,
    }, ensure_ascii=False)}]
    request = {
        "model": model, "instructions": EVIDENCE_INSTRUCTIONS,
        "input": [{"role": "user", "content": content}],
        "text_format": EvidenceExtraction, "max_output_tokens": 4000,
        "store": False,
    }
    if model.startswith("gpt-5.4-mini"):
        request["reasoning"] = {"effort": "none"}
    return approx_reservation(request)


def answer_reservation_range(model, question, language, plan, max_pack_bytes):
    request = {
        "model": model, "instructions": ANSWER_INSTRUCTIONS,
        "input": json.dumps({
            "user_message": question, "language": language,
            "plan": plan.model_dump(mode="json"),
            "available_material": {}, "execution_notes": [],
        }, ensure_ascii=False, separators=(",", ":")),
        "text_format": CoordinatorReply, "max_output_tokens": 2600,
        "store": False,
    }
    if model.startswith("gpt-5.4-mini"):
        request["reasoning"] = {"effort": "none"}
    minimum = approx_reservation(request)
    rates = _prices_for(model)
    # The answer pack may not yet exist. This is an input-budget envelope,
    # not a quotation and not proof that only one answer job will be needed.
    indicative = minimum + Decimal(max_pack_bytes * 2) * rates[0] / 1000000
    return minimum, indicative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--claim-id", action="append", default=[])
    parser.add_argument("--claim-batch-size", type=int, default=15)
    parser.add_argument("--mapping-batch-size", type=int, default=5)
    parser.add_argument("--chunk-chars", type=int, default=6000)
    parser.add_argument("--language", choices=["en", "zh"], default="en")
    parser.add_argument("--model", default="gpt-5.4-mini")
    args = parser.parse_args()
    if args.claim_batch_size < 1 or args.mapping_batch_size < 1:
        parser.error("Batch sizes must be positive")
    paper = PaperDocument.model_validate_json(args.paper.read_text(encoding="utf-8"))
    data = AnalysisDocument.model_validate_json(args.analysis.read_text(encoding="utf-8"))
    if paper.paper_id != data.paper_id or paper.revision != data.paper_revision:
        parser.error("Paper and analysis refer to different source revisions")
    run = find_saved_run(data, args.workflow_id)
    plan = CoordinatorPlan.model_validate(run.get("task_plan"))
    if args.claim_id:
        claim_ids = list(dict.fromkeys(args.claim_id))
        print("Mode: selected hypothetical Claim subset, NOT a modified saved TaskPlan")
    else:
        claim_ids = list(dict.fromkeys(
            cid for task in plan.tasks for cid in task.scope.claim_ids
        ))
    unknown = set(claim_ids) - set(data.claim_pool)
    if unknown:
        parser.error(f"Unknown Claims: {sorted(unknown)}")
    scope_sections = set()
    scope_elements = set()
    whole_paper = False
    for task in plan.tasks:
        whole_paper = whole_paper or task.scope.mode == "whole_paper"
        scope_sections.update(task.scope.section_ids)
        scope_elements.update(task.scope.element_ids)
    if not whole_paper and not (scope_sections or scope_elements):
        parser.error("Saved TaskPlan has no usable grounded source scope")
    if whole_paper:
        scope_sections.clear()
        scope_elements.clear()
    claim_windows = make_windows(
        paper, None if whole_paper else scope_sections,
        selected_elements=None if whole_paper else scope_elements,
    )
    pending_claim_windows = [
        window.window_id for window in claim_windows
        if data.completed_claim_windows.get(window.window_id) != "completed"
    ]
    chunks = source_scoped_chunks(
        paper, make_chunks(paper, chunk_chars=args.chunk_chars),
        scope_sections, scope_elements,
    )
    coverage, _, _ = historical_scan_index(
        paper, data, args.model, chunks, args.chunk_chars,
    )
    selected = [data.claim_pool[cid] for cid in claim_ids]
    grouped = list(batches(selected, args.claim_batch_size))
    text_required = 0
    evidence_reserve = Decimal(0)
    for group in grouped:
        for chunk in chunks:
            missing = [c for c in group if c.claim_id not in coverage[chunk["chunk_id"]]]
            if missing:
                text_required += 1
                evidence_reserve += evidence_reservation(missing, chunk["elements"], args.model)
    mapping_reserved = Decimal(0)
    input_chars = 0
    unlinked_candidates = set()
    shared_table_sources = set()
    for group in batches(claim_ids, args.mapping_batch_size):
        payload = make_mapping_input(data, group)
        input_chars += len(json.dumps(payload, ensure_ascii=False))
        mapping_reserved += mapping_reservation(payload, args.model, args.language)
        unlinked_candidates.update((item["claim_id"], item["evidence_id"])
            for item in payload.get("supplemental_evidence_candidates", []))
        shared_table_sources.update(item["source_element_id"]
            for item in payload.get("shared_source_tables", []))
    reply_min, reply_envelope = answer_reservation_range(
        args.model, run.get("user_question") or "", args.language, plan, 32000,
    )
    hist_budget = run.get("budget_report") or {}
    print("Paper:", data.paper_id)
    print("Saved workflow:", args.workflow_id)
    print("Source sections:", sorted(scope_sections) if not whole_paper else "whole paper")
    print("Target Claims:", len(claim_ids))
    print("Pending NEW Claim windows:", len(pending_claim_windows))
    print("Source chunks:", len(chunks))
    print("Historical text scans reusable:", len(grouped) * len(chunks) - text_required)
    print("Remaining text API calls (estimated):", text_required)
    print("Mapping batches:", len(list(batches(claim_ids, args.mapping_batch_size))))
    print("Mapping input total chars:", input_chars)
    print("Shared table sources:", len(shared_table_sources))
    print("Supplemental UNLINKED Claim/Evidence candidates:", len(unlinked_candidates))
    print("Evidence text maximum reservations USD:", round(evidence_reserve, 6))
    print("Mapping maximum reservations USD:", round(mapping_reserved, 6))
    print("Answer 1-job illustrative reserve USD:", round(reply_min, 6), "-", round(reply_envelope, 6))
    remaining_stage_reservations = evidence_reserve + mapping_reserved + reply_envelope
    print("Known post-planning reserve envelope USD:",
          round(remaining_stage_reservations, 6))
    spent = hist_budget.get("committed_usd")
    print("Selected workflow committed USD:", spent if spent is not None else "unknown")
    print("Selected workflow operating balance USD:",
          hist_budget.get("remaining_usd", "unknown"))
    if spent is not None:
        stress_total = Decimal(str(spent)) + remaining_stage_reservations
        print("Stress estimate incl. committed spending USD:", round(stress_total, 6))
        if stress_total > Decimal("0.18"):
            print("COST WARNING: sum of maximum request reservations exceeds"
                  " this question's operating budget. These estimates are"
                  " deliberately conservative, not predicted actual bills.")
    print("Per-question operating limit USD: 0.18; hard budget USD: 0.20")
    if pending_claim_windows:
        print("BUDGET GAP: pending Claim extraction is NOT priced in this snapshot; "
              "do not regard this estimate as a full-workflow bound.")
    print("Budget gaps: planning calls, additional image scans, newly found sources,"
          " multiple answer jobs, actual output lengths and retries are NOT forecast.")
    print("Coverage gaps: previous incomplete scans remain incomplete until covered;"
          " unlinked candidates require Mapping evaluation.")
    print("Read-only audit finished. No API calls, SQLite writes or file changes.")


if __name__ == "__main__":
    main()
