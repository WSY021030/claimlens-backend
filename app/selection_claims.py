"""User-highlight Claim-only workflow for a standardized paper.

Frontend sends exact paragraph element IDs and [start, end) offsets measured in
PaperDocument Paragraph.text. Existing completed Claim windows are reused; only
uncovered highlighted spans are sent to the existing Claim Runner/Agent.
No Evidence or Mapping invocation is made for a "list all author claims" action.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from app.budget import PaperBudget, QUESTION_BUDGET_ENV, QUESTION_LIMIT_ENV
from app.core.analysis_store import AnalysisStore, default_analysis_path
from app.models.document_v2 import PaperDocument, Paragraph
from app.usage_cost import CostTracker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTION = "List all author claims in the highlighted text."


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _validated_segments(paper: PaperDocument, payload: dict) -> list[dict]:
    """Reject stale/ambiguous frontend offsets before any model is called."""
    if not isinstance(payload, dict):
        raise ValueError("Selection payload must be a JSON object.")
    if payload.get("paper_id") not in (None, paper.paper_id):
        raise ValueError("Selection paper ID does not match the open paper.")
    if payload.get("paper_revision") != paper.revision:
        raise ValueError("Selected text belongs to another paper revision; refresh the reader.")
    items = list(payload.get("segments", [])) if isinstance(payload.get("segments", []), list) else payload.get("segments")
    paragraph_ids = payload.get("element_ids", [])
    if not isinstance(items, list) or not isinstance(paragraph_ids, list):
        raise ValueError("segments and element_ids must be arrays.")
    # Full-paragraph clicks are the exact same scope as a [0,len(text)) span.
    for element_id in paragraph_ids:
        paragraph = paper.elements.get(element_id) if isinstance(element_id, str) else None
        if not isinstance(paragraph, Paragraph):
            raise ValueError(f"Selected element is not a paragraph: {element_id!r}.")
        if paragraph.text.strip():
            items.append({"element_id": element_id, "start": 0,
                          "end": len(paragraph.text), "exact": paragraph.text})
    if not items:
        raise ValueError("At least one selected paragraph text span is required.")

    order = {eid: index for index, eid in enumerate(
        block for section in paper.sections for block in section.block_ids)}
    checked = []
    for segment in items:
        if not isinstance(segment, dict):
            raise ValueError("Each selected segment must be an object.")
        eid = segment.get("element_id")
        element = paper.elements.get(eid) if isinstance(eid, str) else None
        if not isinstance(element, Paragraph):
            raise ValueError(f"Selected element is not a source paragraph: {eid!r}.")
        start, end = segment.get("start"), segment.get("end")
        if (not isinstance(start, int) or isinstance(start, bool)
                or not isinstance(end, int) or isinstance(end, bool)
                or not 0 <= start < end <= len(element.text)):
            raise ValueError(f"Invalid character offsets for paragraph {eid!r}.")
        exact = segment.get("exact")
        if not isinstance(exact, str) or element.text[start:end] != exact:
            raise ValueError(f"Selected text does not match standardized paragraph {eid!r}; refresh the reader.")
        checked.append({"element_id": eid, "start": start, "end": end})

    # Merge only overlapping or adjacent selections within the SAME paragraph.
    # This preserves user-defined boundaries while eliminating redundant work.
    checked.sort(key=lambda s: (order[s["element_id"]], s["start"], s["end"]))
    normalized = []
    for item in checked:
        if (normalized and normalized[-1]["element_id"] == item["element_id"]
                and item["start"] <= normalized[-1]["end"]):
            normalized[-1]["end"] = max(normalized[-1]["end"], item["end"])
        else:
            normalized.append(item.copy())
    return normalized


def _completed_ranges(data, element_id: str) -> list[tuple[int, int]]:
    """Read completed extraction windows for the requested paragraph."""
    ranges = []
    for window_id, status in data.completed_claim_windows.items():
        if status != "completed":
            continue
        parts = window_id.rsplit(":", 2)
        if len(parts) != 3 or parts[0] != element_id:
            continue
        try:
            start, end = int(parts[1]), int(parts[2])
        except ValueError:
            continue
        if 0 <= start < end:
            ranges.append((start, end))
    ranges.sort()
    return ranges


def _is_fully_covered(ranges: list[tuple[int, int]], start: int, end: int) -> bool:
    cursor = start
    for left, right in ranges:
        if right <= cursor:
            continue
        if left > cursor:
            return False
        cursor = max(cursor, right)
        if cursor >= end:
            return True
    return False


def _claim_is_inside_selection(claim, segments: list[dict]) -> bool:
    """Do not return claims whose quoted passage extends outside the highlight."""
    if claim.source_status != "located" or claim.source_anchor is None:
        return False
    selector = claim.source_anchor.selector
    if not hasattr(selector, "start") or not hasattr(selector, "end"):
        return False
    for item in segments:
        if (claim.source_element_id == item["element_id"]
                and item["start"] <= selector.start
                and selector.end <= item["end"]):
            return True
    return False


def _selected_claims(data, paper, segments):
    order = {eid: index for index, eid in enumerate(
        block for section in paper.sections for block in section.block_ids)}
    claims = [c for c in data.claim_pool.values()
              if _claim_is_inside_selection(c, segments)]
    claims.sort(key=lambda c: (order.get(c.source_element_id, len(order)),
               c.source_anchor.selector.start, c.source_anchor.selector.end, c.claim_id))
    return claims


def _as_answer(claims, complete: bool) -> str:
    heading = f"Author claims in the selected text ({len(claims)})"
    lines = [f"## {heading}", ""]
    if not claims:
        lines.append("No substantive author claims were extracted from this selection.")
    else:
        for claim in claims:
            lines.append(f"- **{claim.claim_id}** ({claim.type}): {claim.source_quote}")
    if not complete:
        lines += ["", "**Incomplete extraction:** Some selected text has not been successfully processed."]
    return "\n".join(lines)


def run_selection(*, paper_path: Path, selection: dict, model: str = "gpt-5.4-mini",
                  limit_usd: float = 0.5, preview: bool = False,
                  export_result: bool = False) -> dict:
    """List selected Claims, reusing existing extractions across user questions.

    Each run is an independent Claim-only user operation, sharing the existing
    paper AnalysisStore, and never clears document pools or spending history.
    """
    paper_path = Path(paper_path).resolve()
    paper = PaperDocument.model_validate_json(paper_path.read_text(encoding="utf-8-sig"))
    segments = _validated_segments(paper, selection)
    path = PROJECT_ROOT / default_analysis_path(paper.paper_id)
    store = AnalysisStore(path)
    data = store.load(paper)
    missing = [item for item in segments if not _is_fully_covered(
        _completed_ranges(data, item["element_id"]), item["start"], item["end"])]
    if preview:
        return {"paper_id": paper.paper_id, "selected_segments": segments,
                "uncovered_segments": missing, "cache_hit": not missing,
                "claim_ids": [c.claim_id for c in _selected_claims(data, paper, segments)],
                "preview_only": True}

    start_clock = perf_counter()
    retry = next((r for r in reversed(data.runs)
                  if r.get("mode") == "selected_claims"
                  and r.get("paper_revision") == paper.revision
                  and r.get("selection", {}).get("segments") == segments
                  and r.get("status") in {"partial", "interrupted", "failed"}), None)
    if retry is not None:
        if float(retry.get("question_limit_usd", limit_usd)) != float(limit_usd):
            raise ValueError("Retry must retain the original selection-question budget limit.")
        run_id = retry["run_id"]
        run = retry
        run["status"] = "running"
        run["attempt_count"] = run.get("attempt_count", 1) + 1
    else:
        run_id = uuid4().hex[:12]
        run = {"run_id": run_id, "stage": "coordinator", "mode": "selected_claims",
           "workflow_policy": "selection_claims_v1", "paper_revision": paper.revision,
           "user_question": ACTION, "question_budget_id": run_id,
           "budget_policy": "per_question_v1", "question_limit_usd": limit_usd,
           "status": "running", "answer_status": "unavailable",
           "selection": {"segments": segments}, "cache_hit": not missing,
           "stage_results": [], "issues": [], "started_at": _now(), "answer": None}
        data.runs.append(run)
    run["cache_hit"] = not missing
    store.save(data)

    claim_child = None
    try:
        if missing:
            before_child_ids = {r.get("run_id") for r in data.runs}
            args = [sys.executable, "-u", "-m", "app.claim_runner",
                    "--paper", str(paper_path), "--model", model]
            for item in missing:
                args.extend(["--span", item["element_id"],
                             str(item["start"]), str(item["end"])])
            env = os.environ.copy()
            env[QUESTION_BUDGET_ENV] = run_id
            env[QUESTION_LIMIT_ENV] = str(limit_usd)
            result = subprocess.run(args, cwd=PROJECT_ROOT, env=env, check=False)
            data = store.load(paper)
            run = next(r for r in data.runs if r.get("run_id") == run_id)
            children = [r for r in data.runs if r.get("stage") == "claim"
                        and r.get("run_id") not in before_child_ids]
            if children:
                claim_child = children[-1]
                claim_child["parent_workflow_id"] = run_id
                claim_child["question_budget_id"] = run_id
            run["stage_results"].append({"stage": "claim", "exit_code": result.returncode,
                                         "run_id": claim_child.get("run_id") if claim_child else None})
            if result.returncode:
                run["issues"].append(f"Claim extraction exited with code {result.returncode}.")

        claims = _selected_claims(data, paper, segments)
        complete = all(_is_fully_covered(
            _completed_ranges(data, item["element_id"]), item["start"], item["end"])
            for item in segments)
        run["selection"]["claim_ids"] = [claim.claim_id for claim in claims]
        run["selection"]["uncovered_segments"] = [item for item in segments
            if not _is_fully_covered(_completed_ranges(data, item["element_id"]), item["start"], item["end"])]
        run["status"] = "completed" if complete else "partial"
        run["answer_status"] = "generated" if complete else "partial_generated"
        answer = _as_answer(claims, complete)
        run["answer"] = {"answer": answer, "used_source_ids": [c.claim_id for c in claims],
                         "limitations": [] if complete else ["Some selected source spans remain unprocessed."]}
        run["answer_available"] = True
        run["selected_claim_ids"] = [c.claim_id for c in claims]
        run["task_states"] = {"selection": {"selected_claim_ids": [c.claim_id for c in claims],
             "selected_evidence_ids": [], "mapping_ids": [], "collection_ready": complete}}
        run["claim_count"] = len(claims)
        run["finished_at"] = _now()
        run["duration_seconds"] = round(perf_counter() - start_clock, 3)
        run["api_cost_report"] = (claim_child or {}).get("api_cost_report") or CostTracker().report()
        budget = PaperBudget(paper.paper_id, question_id=run_id, limit_usd=limit_usd)
        try:
            run["budget_report"] = budget.report()
        finally:
            budget.close()
        output = {
            "paper_id": paper.paper_id,
            "paper_revision": paper.revision,
            "run_id": run_id,
            "status": run["status"],
            "cache_hit": not missing,
            "selected_segments": segments,
            "claim_ids": [c.claim_id for c in claims],
            "claims": [
                c.model_dump(mode="json")
                for c in claims
            ],
            "answer": answer,
            "analysis_path": str(path.resolve()),
        }

        store.save(data)

        if export_result:
            # The central store writes a small current index plus response.json.
            # The Python return value above remains backward-compatible.
            out_path = path.parent / "selected_claims.json"
            run["result_file"] = str(out_path.resolve())
            output["result_file"] = str(out_path.resolve())
            store.save(data)

        return output
    except BaseException as exc:
        data = store.load(paper)
        run = next(r for r in data.runs if r.get("run_id") == run_id)
        run["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        run["issues"].append(f"{type(exc).__name__}: {exc}")
        run["finished_at"] = _now()
        store.save(data)
        raise


def main():
    parser = argparse.ArgumentParser(description="Extract or reuse all Claims in a user-highlighted text span")
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--selection-file", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--question-limit-usd", type=float, default=0.5)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--export-result", action="store_true")
    args = parser.parse_args()
    if not args.paper.is_file() or not args.selection_file.is_file():
        parser.error("The paper and selection JSON files must exist.")
    if not 0 < args.question_limit_usd < float("inf"):
        parser.error("The question budget must be finite and positive.")
    payload = json.loads(args.selection_file.read_text(encoding="utf-8-sig"))
    os.chdir(PROJECT_ROOT)
    result = run_selection(paper_path=args.paper, selection=payload,
                           model=args.model, limit_usd=args.question_limit_usd,
                           preview=args.preview,
                           export_result=args.export_result)
    print("\n=== SELECTED CLAIMS ===", flush=True)
    print("Status:", result.get("status", "preview"), flush=True)
    print("Cache hit:", result["cache_hit"], flush=True)
    print("Claim IDs:", ", ".join(result.get("claim_ids", [])), flush=True)
    print("Analysis JSON:", result.get("analysis_path", "preview"), flush=True)
    if result.get("result_file"):
        print("Export JSON:", result["result_file"], flush=True)
    if result.get("answer"):
        print("\n" + result["answer"])


if __name__ == "__main__":
    main()
