"""Standalone per-Claim Mapping using the same Data Manager as workflows."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from app.budget import PaperBudget
from app.core.analysis_store import AnalysisStore
from app.core.evidence_retrieval import EvidenceDataManager, byte_size
from app.core.mapping_execution import run_claim_mappings
from app.core.mapping_records import claim_cache_key, find_cached_claim
from app.core.token_report import task_token_report, print_task_tokens
from app.models.document_v2 import PaperDocument
from app.models.reader_v3 import AnalysisDocument
from app.usage_cost import CostTracker


def make_batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def select_claim_ids(args, data):
    ids = list(data.claim_pool) if args.all_claims else list(dict.fromkeys(args.claim_id or []))
    unknown = [cid for cid in ids if cid not in data.claim_pool]
    if not ids or unknown:
        raise ValueError(f"No valid Claim selection or unknown IDs: {unknown}")
    return ids


def load_mapping_paper(analysis_path, data, explicit_path=None):
    if explicit_path:
        path = Path(explicit_path)
    else:
        root = Path(__file__).resolve().parents[1]
        path = root / "data" / "standardized" / data.paper_id / "paper.json"
    if not path.is_file():
        if explicit_path:
            raise ValueError(f"Standardized paper not found: {path}")
        return None
    paper = PaperDocument.model_validate_json(path.read_text(encoding="utf-8-sig"))
    if paper.paper_id != data.paper_id or paper.revision != data.paper_revision:
        raise ValueError("Mapping paper metadata and analysis revision do not match.")
    return paper


def make_mapping_input(data, claim_ids, paper=None):
    """Compatibility helper. Execution always uses one Claim per packet."""
    ids = list(dict.fromkeys(claim_ids))
    if len(ids) != 1:
        raise ValueError("Build one bounded Mapping input per Claim, not a task-wide Evidence union.")
    packet, _ = EvidenceDataManager(data, paper).packet(ids[0])
    return packet


def main():
    parser = argparse.ArgumentParser(description="Link-free, bounded per-Claim Mapping")
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--paper", type=Path, help="Read source structure only; never extract new Evidence")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--claim-id", action="append")
    selection.add_argument("--all-claims", action="store_true")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--language", choices=["zh", "en"], default="en")
    parser.add_argument("--focus", default="claim_support",
                        choices=["general", "claim_support", "data_consistency",
                                 "experimental_comparability", "scope_and_generalization",
                                 "method_result_relationship"])
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Accepted for CLI compatibility; execution is bounded per Claim")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or not args.analysis.is_file():
        parser.error("A valid analysis.json and positive batch size are required.")
    data = AnalysisDocument.model_validate_json(args.analysis.read_text(encoding="utf-8-sig"))
    paper = load_mapping_paper(args.analysis, data, args.paper)
    ids = select_claim_ids(args, data)
    manager = EvidenceDataManager(data, paper)
    print("Paper:", data.paper_id)
    print("Claims:", len(ids), "Evidence pool:", len(data.evidence_pool))
    print("Legacy Links are not read. Mapping input is independent per Claim.")
    print("Real document structure available:", paper is not None)
    for cid in ids:
        packet, receipt = manager.packet(cid, analysis_focus=args.focus)
        expanded, expanded_receipt = manager.packet(
            cid, expanded=True, analysis_focus=args.focus,
            required_evidence_ids=receipt["selected_evidence_ids"],
        )
        if receipt["priority_deferred"]:
            packet, receipt = expanded, expanded_receipt
        key = claim_cache_key(data, cid, args.model, args.language, args.focus, input_packet=packet)
        cached = find_cached_claim(data, cid, key)
        expanded_key = claim_cache_key(data, cid, args.model, args.language, args.focus, input_packet=expanded)
        expanded_cached = find_cached_claim(data, cid, expanded_key)
        print(json.dumps({
            "claim_id": cid, "evidence_ids": receipt["selected_evidence_ids"],
            "packet_bytes": byte_size(packet), "byte_limit": receipt["packet_byte_limit"],
            "priority_deferred": receipt["priority_deferred"],
            "missing_direct_sources": receipt["missing_direct_source_ids"],
            "other_candidates_deferred": len(receipt["deferred_for_size"]),
            "expanded_view_bytes": expanded_receipt["packet_bytes"],
            "cached_mapping": None if args.rerun else expanded_cached or cached,
        }, ensure_ascii=False), flush=True)
    if args.preview:
        print("Preview only: no API requests and no analysis file writes.")
        return
    tracker = CostTracker()
    budget = PaperBudget(data.paper_id)
    store = AnalysisStore(args.analysis)
    started = perf_counter()
    run = {"run_id": uuid4().hex[:12], "stage": "mapping", "model": args.model,
           "language": args.language, "target_claim_ids": ids, "status": "running",
           "execution_policy": "bounded_no_links_v2", "planned_batches": len(ids),
           "created_mapping_ids": [], "reused_mapping_ids": [], "issues": [],
           "started_at": datetime.now(timezone.utc).isoformat()}
    data.runs.append(run)
    store.save(data)
    try:
        result = run_claim_mappings(
            data=data, store=store, run=run, claim_ids=ids, model=args.model,
            language=args.language, tracker=tracker, budget=budget, paper=paper,
            rerun=args.rerun, analysis_focus=args.focus)
        run["issues"].extend(result["issues"])
        run["pending_claim_ids"] = result["pending_claim_ids"]
        run["new_mappings"] = len(run["created_mapping_ids"])
        run["status"] = "partial" if result["pending_claim_ids"] or result["issues"] else "completed"
    except BaseException as exc:
        run["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        run["issues"].append({"type": type(exc).__name__, "message": str(exc)[:500]})
        raise
    finally:
        run["finished_at"] = datetime.now(timezone.utc).isoformat()
        run["duration_seconds"] = round(perf_counter() - started, 3)
        run["api_cost_report"] = tracker.report()
        run["budget_report"] = budget.report()
        run["question_budget_id"] = run["budget_report"].get("question_budget_id")
        run["token_usage_report"] = task_token_report(run)
        print_task_tokens(run["token_usage_report"], run["budget_report"])
        store.save(data)
        budget.close()
    print("Mapping status:", run["status"])
    print("Saved:", args.analysis.resolve())


if __name__ == "__main__":
    main()
