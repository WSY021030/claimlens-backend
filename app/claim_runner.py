
import argparse
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone
from time import perf_counter

from app.core.token_report import task_token_report, print_task_tokens
from app.agents.claim_agent import ClaimAgent
from app.budget import PaperBudget
from app.llm import StructuredOutputParseError
from app.core.analysis_store import (
    AnalysisStore,
    default_analysis_path,
)
from app.core.source_locator import locate_claim
from app.models.document_v2 import (
    PaperDocument,
    Paragraph,
)
from app.models.reader_v3 import ClaimWindow
from app.usage_cost import CostTracker


def make_windows(
    paper: PaperDocument,
    selected_sections: set[str] | None,
    window_chars: int = 2800,
    overlap: int = 180,
    selected_elements: set[str] | None = None,
) -> list[ClaimWindow]:

    windows = []

    for section in paper.sections:
        # Sections and individual paragraphs form a union of requested ranges.
        if selected_sections is not None and (
            section.section_id not in selected_sections
            and not (set(section.block_ids) & (selected_elements or set()))
        ):
            continue

        for element_id in section.block_ids:
            if selected_sections is not None and (
                section.section_id not in selected_sections
                and element_id not in (selected_elements or set())
            ):
                continue
            element = paper.elements[element_id]

            if not isinstance(element, Paragraph):
                continue

            text = element.text

            if not text.strip():
                continue

            start = 0

            while start < len(text):
                end = min(
                    start + window_chars,
                    len(text),
                )

                windows.append(
                    ClaimWindow(
                        window_id=f"{element_id}:{start}:{end}",
                        element_id=element_id,
                        section_id=section.section_id,
                        section_title=section.title,
                        start=start,
                        end=end,
                        text=text[start:end],
                    )
                )

                if end == len(text):
                    break

                start = max(
                    start + 1,
                    end - overlap,
                )

    return windows


def make_batches(
    windows: list[ClaimWindow],
    max_chars: int = 6000,
    max_windows: int = 3,
):
    batch = []
    size = 0

    for window in windows:
        length = len(window.text)

        if batch and (
            size + length > max_chars
            or len(batch) >= max_windows
        ):
            yield batch
            batch = []
            size = 0

        batch.append(window)
        size += length

    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--paper",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--section",
        action="append",
        help="Section ID; may be supplied more than once",
    )

    parser.add_argument(
        "--element-id",
        action="append",
        help="Specific paragraph element ID; repeatable and combinable with --section",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract claims from the whole paper",
    )

    parser.add_argument(
        "--span", nargs=3, action="append", metavar=("ELEMENT_ID", "START", "END"),
        help="Extract an exact [start,end) span of Paragraph.text; repeatable",
    )

    parser.add_argument(
        "--model",
        default="gpt-5.4-mini",
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show selection without API calls",
    )

    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Re-extract selected windows and append new claims",
    )

    args = parser.parse_args()

    if args.span and (args.all or args.section or args.element_id):
        parser.error("--span cannot be combined with --all, --section or --element-id.")
    if args.all and (args.section or args.element_id):
        parser.error("--all cannot be combined with --section or --element-id.")
    if not args.all and not (args.section or args.element_id or args.span):
        parser.error("Choose --all, --section, --element-id, or --span.")

    paper = PaperDocument.model_validate_json(
        args.paper.read_text(encoding="utf-8")
    )

    selected = None if args.all else set(args.section or [])
    selected_elements = None if args.all else set(args.element_id or [])

    if selected is not None:
        known = {section.section_id for section in paper.sections}
        unknown = selected - known
        if unknown:
            parser.error(f"Unknown sections: {sorted(unknown)}")
        bad_elements = {
            element_id for element_id in selected_elements
            if not isinstance(paper.elements.get(element_id), Paragraph)
        }
        if bad_elements:
            parser.error(f"Unknown/non-paragraph elements: {sorted(bad_elements)}")

    if args.span:
        # A highlighted span is NOT a whole paragraph/section coverage receipt.
        # Exact offsets refer to PaperDocument Paragraph.text, not publisher HTML.
        windows = []
        seen = set()
        for element_id, start_text, end_text in args.span:
            element = paper.elements.get(element_id)
            if not isinstance(element, Paragraph):
                parser.error(f"--span requires a real paragraph element: {element_id}")
            try:
                first, last = int(start_text), int(end_text)
            except ValueError:
                parser.error(f"Invalid --span offsets: {start_text}, {end_text}")
            if not 0 <= first < last <= len(element.text):
                parser.error(f"Span outside paragraph {element_id}: {first}, {last}")
            selected_elements.add(element_id)
            # Keep selected content in full passage windows, without marking any
            # unselected neighboring text as extracted.
            cursor = first
            while cursor < last:
                right = min(cursor + 2800, last)
                key = (element_id, cursor, right)
                if key not in seen:
                    windows.append(ClaimWindow(
                        window_id=f"{element_id}:{cursor}:{right}",
                        element_id=element_id, section_id=element.section_id,
                        section_title=next((sec.title for sec in paper.sections
                                            if sec.section_id == element.section_id), ""),
                        start=cursor, end=right, text=element.text[cursor:right],
                    ))
                    seen.add(key)
                if right == last:
                    break
                cursor = right - 180
    else:
        windows = make_windows(paper, selected, selected_elements=selected_elements)

    store = AnalysisStore(
        default_analysis_path(paper.paper_id)
    )

    data = store.load(paper)

    pending = [
        window for window in windows
        if args.rerun
        or data.completed_claim_windows.get(window.window_id) != "completed"
    ]

    print("Paper:", paper.paper_id)
    print("Selected windows:", len(windows))
    print("Pending windows:", len(pending))
    print("Existing claims:", len(data.claim_pool))

    if args.preview:
        for window in pending:
            print(
                window.window_id,
                window.section_title,
            )
        return

    if not pending:
        print("No new windows to process.")
        return

    tracker = CostTracker()
    budget = PaperBudget(paper.paper_id)

    agent = ClaimAgent(
        model=args.model,
        tracker=tracker,
        budget=budget,
    )

    run_started_at = datetime.now(timezone.utc).isoformat()
    run_start_clock = perf_counter()
    run_id = uuid4().hex[:12]

    run = {
        "run_id": run_id,
        "stage": "claim",
        "model": args.model,
        "sections": (["ALL"] if args.all else sorted(selected)),
        "element_ids": sorted(selected_elements or []),
        "status": "running",
        "new_claims": 0,
        "accepted_claim_ids": [],
        "processed_windows": 0,
        "started_at": run_started_at,
        "finished_at": None,
        "duration_seconds": None,
        "llm_wall_time_seconds": 0.0,
    }

    data.runs.append(run)
    store.save(data)

    def process_batch(batch, depth=0):
        if all(data.completed_claim_windows.get(w.window_id) == "completed" for w in batch) and not args.rerun:
            return True
        call_start = perf_counter()
        parse_error = None
        try:
            extraction = agent.extract(batch)
        except StructuredOutputParseError as exc:
            parse_error = exc
        finally:
            run["llm_wall_time_seconds"] += perf_counter() - call_start
        if parse_error is not None:
            run.setdefault("issues", []).append({
                "window_ids": [w.window_id for w in batch],
                "status": "parse_failed", "message": str(parse_error),
            })
            store.save(data)
            if depth >= 4:
                return False
            if len(batch) > 1:
                middle = len(batch) // 2
                first = process_batch(batch[:middle], depth + 1)
                second = process_batch(batch[middle:], depth + 1)
                return first and second
            window = batch[0]
            if len(window.text) < 800:
                return False
            middle = len(window.text) // 2
            ranges = [(0, min(len(window.text), middle + 80)), (max(0, middle - 80), len(window.text))]
            fragments = [window.model_copy(update={
                "window_id": f"{window.element_id}:{window.start + a}:{window.start + b}",
                "start": window.start + a, "end": window.start + b,
                "text": window.text[a:b],
            }) for a, b in ranges]
            first = process_batch([fragments[0]], depth + 1)
            second = process_batch([fragments[1]], depth + 1)
            if first and second:
                data.completed_claim_windows[window.window_id] = "completed"
                store.save(data)
            return first and second
        for raw in extraction.claims:
            if not raw.source_quote.strip():
                run["empty_quote_candidates"] = run.get("empty_quote_candidates", 0) + 1
                continue
            candidate = locate_claim(paper=paper, raw=raw, windows=batch, run_id=run_id)
            candidate.scope = raw.scope
            claim_id, created = store.add_claim(data, candidate)
            # Deduplicate identical source-grounded Claims and track this run.
            if claim_id not in run["accepted_claim_ids"]:
                run["accepted_claim_ids"].append(claim_id)
            if created:
                run["new_claims"] += 1
        for window in batch:
            data.completed_claim_windows[window.window_id] = "completed"
        store.save(data)
        return True

    try:
        complete = True
        for batch in make_batches(pending):
            batch_complete = process_batch(batch)
            complete = complete and batch_complete
            run["processed_windows"] = sum(
                data.completed_claim_windows.get(w.window_id) == "completed" for w in pending
            )
            store.save(data)
            print("Processed:", run["processed_windows"], "/", len(pending),
                  "| Total claims:", len(data.claim_pool), flush=True)
        run["status"] = "completed" if complete else "partial"

    except BaseException:
        run["status"] = "interrupted"
        raise

    finally:
        run["finished_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        run["duration_seconds"] = round(
            perf_counter() - run_start_clock, 3
        )

        run["llm_wall_time_seconds"] = round(
            run["llm_wall_time_seconds"], 3
        )
        
        run["api_cost_report"] = tracker.report()
        run["budget_report"] = budget.report()
        run["token_usage_report"] = task_token_report(run)
        print_task_tokens(run["token_usage_report"], run["budget_report"])
        store.save(data)

    print("\n=== CLAIM EXTRACTION FINISHED ===")
    print("Status:", run["status"])
    print("Claims:", len(data.claim_pool))
    print("New claims:", run["new_claims"])
    print("Analysis JSON:", store.path)
    print(
        "API cost estimate:",
        tracker.report()["estimated_total_usd"],
        "USD",
    )

    budget.close()


if __name__ == "__main__":
    main()
