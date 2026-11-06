"""Read-only, API-free source packet preview for an existing Reader-v3 Workflow.

Usage: uv run python -m app.context_preview --paper path/to/paper.json
       --workflow cef1816322ab
No model clients, tasks or saved analysis are invoked or mutated.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from app.coordinator_runner import find_saved_run, size, source_packet
from app.core.analysis_store import AnalysisStore, default_analysis_path
from app.models.coordinator_v3 import CoordinatorPlan
from app.models.document_v2 import PaperDocument
from app.workflow_runner import in_scope, source_scope


def main():
    parser = argparse.ArgumentParser(description="Read-only source context inspection")
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--workflow", required=True)
    args = parser.parse_args()
    paper = PaperDocument.model_validate_json(args.paper.read_text(encoding="utf-8-sig"))
    data = AnalysisStore(default_analysis_path(paper.paper_id)).load(paper)
    run = find_saved_run(data, args.workflow)
    plan = CoordinatorPlan.model_validate(run["task_plan"])
    print("Workflow:", run["run_id"])
    print("Read-only preview: no LLM calls or changes to stored analysis")
    for task in plan.tasks:
        state = run.get("task_states", {}).get(task.task_id, {})
        claim_ids = list(state.get("selected_claim_ids") or task.scope.claim_ids)
        sections, elements, whole = source_scope(paper, task)
        evidence_ids = list(state.get("selected_evidence_ids") or [
            eid for eid, evidence in data.evidence_pool.items()
            if in_scope(paper, evidence.source_element_id, sections, elements, whole)
        ])
        packet = source_packet(data, claim_ids=claim_ids, evidence_ids=evidence_ids,
                               paper=paper, question=run["user_question"])
        print("\nTask:", task.task_id, "intent:", task.intent)
        print("Claim IDs:", claim_ids)
        print("Evidence count:", len(packet["evidence"]))
        print("Unique source content:", len(packet["shared_source_content"]))
        print("Compact packet bytes:", size(packet))
        for check in packet.get("numeric_table_checks") or []:
            print("Table:", check["table_id"], "status:", check["status"],
                  "source cells:", check.get("cell_count", "unknown"))
            for metric in check.get("metric_checks") or []:
                print("  Metric:", metric["metric"], "status:", metric["status"],
                      "dataset groups:", len(metric.get("datasets") or []))
            for issue in check.get("issues") or []:
                print("  Source issue:", issue)


if __name__ == "__main__":
    main()
