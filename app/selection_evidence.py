
"""Evidence-only extraction from an explicit reader selection.

This module reuses the existing Evidence Agent and AnalysisStore. It never
extracts Claims, creates Claim-Evidence links, or expands the user's selection.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from openai import APIConnectionError, APIStatusError, APITimeoutError

from app.agents.evidence_agent import EvidenceAgent
from app.budget import BudgetStop, PaperBudget
from app.core.analysis_store import AnalysisStore, default_analysis_path
from app.core.evidence_locator import locate_evidence
from app.evidence_runner import digest, make_chunks, save_findings
from app.llm import StructuredOutputParseError
from app.models.document_v2 import PaperDocument, Paragraph
from app.models.reader_v3 import EvidenceExtraction
from app.selection_claims import _validated_segments
from app.tools.image_inputs import ImageInputs
from app.usage_cost import CostTracker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY = "selected_evidence_v1"

OPERATION = (
    "Extract distinct, meaningful, source-grounded evidence from ONLY the "
    "provided selected passage or explicitly selected source element. "
    "Include concrete reported findings, numbers, experimental comparisons, "
    "method details, table cells, figures, or equations when present. "
    "Use exact short source quotations for text. Do not introduce material "
    "outside the selection, invent observations, extract Claims, create "
    "Claim links, or judge whether any scientific proposition is correct."
)

MODEL_ERRORS = (
    BudgetStop, StructuredOutputParseError,
    APIStatusError, APIConnectionError, APITimeoutError,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_selection(
    paper: PaperDocument,
    payload: dict,
) -> tuple[list[dict], list[str]]:
    """Reuse Claim's text-offset contract and allow selected media IDs."""
    if not isinstance(payload, dict):
        raise ValueError("Selection must be a JSON object.")

    if payload.get("paper_id") != paper.paper_id:
        raise ValueError("Selection paper_id does not match the open paper.")

    if payload.get("paper_revision") != paper.revision:
        raise ValueError(
            "Selection revision is stale. Refresh the paper reader."
        )

    ids = payload.get("element_ids", [])
    segments_input = payload.get("segments", [])

    if not isinstance(ids, list) or not isinstance(segments_input, list):
        raise ValueError("element_ids and segments must both be arrays.")

    paragraphs, media = [], []

    for eid in ids:
        element = paper.elements.get(eid) if isinstance(eid, str) else None

        if isinstance(element, Paragraph):
            paragraphs.append(eid)
        elif element is not None and element.kind in {
            "figure", "table", "equation"
        }:
            media.append(eid)
        else:
            raise ValueError(
                f"Unknown or unsupported selected element: {eid!r}"
            )

    if not paragraphs and not segments_input:
        segments = []
    else:
        segments = _validated_segments(
            paper,
            {
                "paper_id": paper.paper_id,
                "paper_revision": paper.revision,
                "element_ids": paragraphs,
                "segments": segments_input,
            },
        )

    for seg in segments:
        text = paper.elements[seg["element_id"]].text[
            seg["start"]:seg["end"]
        ]
        if not text.strip():
            raise ValueError("Whitespace-only selection is not extractable.")

    if not segments and not media:
        raise ValueError(
            "Select at least one paragraph span or source element."
        )

    order = {
        eid: index
        for index, eid in enumerate(
            block
            for section in paper.sections
            for block in section.block_ids
        )
    }

    media = sorted(
        set(media),
        key=lambda eid: order.get(eid, len(order)),
    )

    return segments, media


def selected_sources(
    paper: PaperDocument,
    segments: list[dict],
    media: list[str],
) -> list[dict]:
    """Send only selected characters or explicitly selected media."""
    titles = {
        section.section_id: section.title
        for section in paper.sections
    }

    items = []

    for seg in segments:
        element = paper.elements[seg["element_id"]]
        cursor, stop = seg["start"], seg["end"]

        while cursor < stop:
            right = min(cursor + 2600, stop)

            items.append({
                "element_id": seg["element_id"],
                "kind": "paragraph",
                "section_id": element.section_id,
                "section_title": titles.get(element.section_id, ""),
                "text": element.text[cursor:right],
                "source_start": cursor,
                "source_end": right,
            })

            if right == stop:
                break

            cursor = max(cursor + 1, right - 120)

    if media:
        requested = set(media)

        for chunk in make_chunks(paper):
            for item in chunk["elements"]:
                if item["element_id"] in requested:
                    items.append(item)

    return items


def record_findings(
    *,
    extraction,
    source_item,
    paper,
    store,
    data,
    run,
    run_id,
    image_receipt=None,
) -> list[str]:
    """Reject findings outside the selected source before persistence."""
    valid = []

    for raw in extraction.evidence:
        if (
            raw.source_element_id != source_item["element_id"]
            or raw.kind != source_item["kind"]
        ):
            run["rejected_findings"] += 1
            continue

        record = locate_evidence(
            paper=paper,
            raw=raw,
            scanned_elements=[source_item],
            run_id=run_id,
            image_receipt=image_receipt,
        )

        if record.source_status != "located":
            run["rejected_findings"] += 1
            continue

        if source_item["kind"] == "paragraph":
            selector = record.source_anchor.selector

            if (
                not record.source_quote
                or not hasattr(selector, "start")
                or not (
                    source_item["source_start"]
                    <= selector.start
                    < selector.end
                    <= source_item["source_end"]
                )
            ):
                run["rejected_findings"] += 1
                continue

        # Evidence-only extraction never creates Claim links.
        valid.append(raw.model_copy(update={"links": []}))

    if not valid:
        return []

    filtered = EvidenceExtraction(
        evidence=valid,
        image_candidates=[],
    )

    found = save_findings(
        extraction=filtered,
        source_elements=[source_item],
        claim_batch=[],
        paper=paper,
        store=store,
        data=data,
        run=run,
        run_id=run_id,
        image_receipt=image_receipt,
    )

    if source_item["kind"] == "paragraph":
        for eid in found:
            record = data.evidence_pool[eid]

            # Preserve existing shared Evidence records unchanged.
            if record.created_run_id != run_id:
                continue

            span = record.source_anchor.selector
            text = paper.elements[record.source_element_id].text

            left = max(
                source_item["source_start"],
                span.start - 200,
            )
            right = min(
                source_item["source_end"],
                span.end + 200,
            )

            record.source_content["context"] = text[left:right]

    return found


def render_result(
    paper,
    segments,
    media,
    data,
    run,
    key,
) -> dict:
    receipt = data.direct_evidence_searches.get(key) or {}

    evidence_ids = [
        eid
        for eid in dict.fromkeys(receipt.get("evidence_ids", []))
        if eid in data.evidence_pool
    ]

    records = []

    for eid in evidence_ids:
        record = data.evidence_pool[eid].model_dump(mode="json")

        if record["kind"] == "paragraph":
            anchor = record.get("source_anchor") or {}
            selector = anchor.get("selector") or {}

            for seg in segments:
                if (
                    record["source_element_id"] == seg["element_id"]
                    and seg["start"] <= selector.get("start", -1)
                    and selector.get("end", -1) <= seg["end"]
                ):
                    text = paper.elements[seg["element_id"]].text

                    left = max(
                        seg["start"],
                        selector["start"] - 200,
                    )
                    right = min(
                        seg["end"],
                        selector["end"] + 200,
                    )

                    record["source_content"]["context"] = text[left:right]

                    selector["prefix"] = text[
                        max(seg["start"], selector["start"] - 40):
                        selector["start"]
                    ]
                    selector["suffix"] = text[
                        selector["end"]:
                        min(seg["end"], selector["end"] + 40)
                    ]
                    break

        records.append(record)

    return {
        "paper_id": paper.paper_id,
        "paper_revision": paper.revision,
        "mode": "selected_evidence",
        "status": receipt.get("status", "partial"),
        "run_id": run["run_id"],
        "selection": {
            "segments": segments,
            "element_ids": media,
        },
        "evidence_ids": evidence_ids,
        "evidence": records,
        "claim_ids_used": [],
        "claim_links_created": 0,
        "issues": run.get("issues", []),
    }


def run_selection_evidence(
    *,
    paper_path: Path,
    selection: dict,
    model="gpt-5.4-mini",
    image_mode="relevant",
    limit_usd=0.50,
    preview=False, export_result=False,
) -> dict:
    paper_path = Path(paper_path).resolve()

    paper = PaperDocument.model_validate_json(
        paper_path.read_text(encoding="utf-8-sig")
    )

    segments, media = validate_selection(paper, selection)
    items = selected_sources(paper, segments, media)

    key = digest({
        "policy": POLICY,
        "paper_revision": paper.revision,
        "segments": segments,
        "element_ids": media,
        "model": model,
        "image_mode": image_mode,
    })

    store = AnalysisStore(
        PROJECT_ROOT / default_analysis_path(paper.paper_id)
    )
    data = store.load(paper)

    receipt = data.direct_evidence_searches.get(key) or {}

    old_run = next(
        (
            run
            for run in reversed(data.runs)
            if run.get("run_id") == receipt.get("run_id")
        ),
        None,
    )

    if preview:
        print(
            f"Selected text/media chunks: {len(items)}; "
            "no model calls or writes."
        )
        return {
            "status": "preview",
            "selection": {
                "segments": segments,
                "element_ids": media,
            },
            "chunks": len(items),
        }

    if (
        receipt.get("status") == "completed"
        and all(
            eid in data.evidence_pool
            for eid in receipt.get("evidence_ids", [])
        )
    ):
        print("Reusing completed selected-Evidence extraction.")

        output = render_result(
            paper, segments, media, data,
            old_run or {
                "run_id": receipt.get("run_id"),
                "issues": [],
            },
            key,
        )

        if export_result:
            export_path = (
                store.path.parent
                / f"selected_evidence_{output['run_id']}.json"
            )
            export_path.write_text(
                json.dumps(
                    output,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            output["export_file"] = str(export_path.resolve())

            if old_run is not None:
                old_run["result_file"] = str(
                    export_path.resolve()
                )
                store.save(data)

        print("Analysis JSON:", store.path.resolve())

        if export_result:
            print("Export JSON:", output["export_file"])

        return output

    if old_run is not None and old_run.get("status") in {
        "partial", "interrupted", "failed", "running"
    }:
        run = old_run

        if float(run.get("question_limit_usd", limit_usd)) != float(
            limit_usd
        ):
            raise ValueError(
                "A retry must retain its original budget limit."
            )

        run["attempt_count"] = run.get("attempt_count", 1) + 1
        run["unavailable_images"] = []
        run["status"] = "running"

    else:
        run = {
            "run_id": uuid4().hex[:12],
            "stage": "evidence",
            "mode": "selected_evidence",
            "question_budget_id": None,
            "question_limit_usd": limit_usd,
            "budget_policy": "per_question_v1",
            "source_question": None,
            "selection": {
                "segments": segments,
                "element_ids": media,
            },
            "model": model,
            "image_mode": image_mode,
            "status": "running",
            "started_at": now(),
            "attempt_count": 1,
            "new_evidence": 0,
            "new_links": 0,
            "rejected_findings": 0,
            "unresolved_links": [],
            "issues": [],
            "unavailable_images": [],
            "text_scans_completed": 0,
            "image_scans_completed": 0,
        }
        run["question_budget_id"] = run["run_id"]
        data.runs.append(run)

    run_id = run["run_id"]
    found = list(receipt.get("evidence_ids", []))

    data.direct_evidence_searches[key] = {
        "status": "partial",
        "mode": "selected_evidence",
        "run_id": run_id,
        "paper_revision": paper.revision,
        "selection": run["selection"],
        "evidence_ids": found,
    }
    store.save(data)

    tracker = CostTracker()
    budget = PaperBudget(
        paper.paper_id,
        question_id=run_id,
        limit_usd=limit_usd,
    )
    agent = EvidenceAgent(
        model=model,
        tracker=tracker,
        budget=budget,
    )
    images = ImageInputs(
        paper,
        document_dir=paper_path.parent,
        enabled=image_mode != "none",
        max_images_per_call=1,
    )

    start = perf_counter()
    pending_image_scans = []
    stopped = False

    try:
        for source_item in items:
            scan = digest({
                "search_key": key,
                "kind": "text",
                "item": source_item,
            })

            if data.completed_evidence_scans.get(scan) == "completed":
                continue

            try:
                extraction, _ = agent.extract(
                    [],
                    [source_item],
                    source_focus=OPERATION,
                )
            except MODEL_ERRORS as exc:
                run["issues"].append(
                    f"Source scan pending: "
                    f"{type(exc).__name__}: {exc}"
                )
                stopped = True
                break

            found.extend(
                record_findings(
                    extraction=extraction,
                    source_item=source_item,
                    paper=paper,
                    store=store,
                    data=data,
                    run=run,
                    run_id=run_id,
                )
            )

            data.completed_evidence_scans[scan] = "completed"
            run["text_scans_completed"] += 1

            data.direct_evidence_searches[key]["evidence_ids"] = list(
                dict.fromkeys(found)
            )
            store.save(data)

        if not stopped and image_mode != "none":
            for element_id in media:
                element = paper.elements[element_id]

                for asset_id, cell_id in images.asset_refs(element_id):
                    try:
                        prepared = images.prepare(
                            element_id,
                            asset_id,
                            cell_id,
                        )
                    except (OSError, ValueError, ImportError) as exc:
                        run["unavailable_images"].append({
                            "element_id": element_id,
                            "asset_id": asset_id,
                            "message": str(exc),
                        })
                        continue

                    scan = digest({
                        "search_key": key,
                        "kind": "image",
                        "element_id": element_id,
                        "asset_id": asset_id,
                        "cell_id": cell_id,
                        "source_sha256": prepared.source_sha256,
                    })

                    pending_image_scans.append(scan)

                    if (
                        data.completed_evidence_scans.get(scan)
                        == "completed"
                    ):
                        continue

                    item = {
                        "element_id": element_id,
                        "kind": element.kind,
                        "asset_id": asset_id,
                        "cell_id": cell_id,
                        "label": getattr(element, "label", None),
                        "caption": getattr(element, "caption", ""),
                    }

                    try:
                        extraction, response = agent.extract(
                            [],
                            [item],
                            image=prepared,
                            source_focus=OPERATION,
                        )
                    except MODEL_ERRORS as exc:
                        run["issues"].append(
                            f"Image scan pending at {element_id}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        stopped = True
                        break

                    images.record_submission(
                        [prepared],
                        model,
                        response,
                    )

                    receipt_data = {
                        **prepared.receipt(),
                        "model": getattr(response, "model", model),
                        "response_id": getattr(response, "id", None),
                    }

                    found.extend(
                        record_findings(
                            extraction=extraction,
                            source_item=item,
                            paper=paper,
                            store=store,
                            data=data,
                            run=run,
                            run_id=run_id,
                            image_receipt=receipt_data,
                        )
                    )

                    data.completed_evidence_scans[scan] = "completed"
                    run["image_scans_completed"] += 1

                    data.direct_evidence_searches[key][
                        "evidence_ids"
                    ] = list(dict.fromkeys(found))

                    store.save(data)

                if stopped:
                    break

        text_scans = [
            digest({
                "search_key": key,
                "kind": "text",
                "item": item,
            })
            for item in items
        ]

        text_complete = all(
            data.completed_evidence_scans.get(scan) == "completed"
            for scan in text_scans
        )

        image_complete = (
            not run["unavailable_images"]
            and all(
                data.completed_evidence_scans.get(scan) == "completed"
                for scan in pending_image_scans
            )
        )

        run["status"] = (
            "completed"
            if text_complete and image_complete and not stopped
            else "partial"
        )

    except BaseException:
        run["status"] = "interrupted"
        raise

    finally:
        run["finished_at"] = now()
        run["duration_seconds_last_attempt"] = round(
            perf_counter() - start, 3
        )
        run["api_cost_report"] = tracker.report()
        run["budget_report"] = budget.report()
        run["image_report"] = images.report()

        data.direct_evidence_searches[key] = {
            **data.direct_evidence_searches[key],
            "status": run["status"],
            "evidence_ids": list(dict.fromkeys(found)),
            "text_scan_complete": all(
                data.completed_evidence_scans.get(
                    digest({
                        "search_key": key,
                        "kind": "text",
                        "item": item,
                    })
                ) == "completed"
                for item in items
            ),
            "image_scan_complete": (
                not run["unavailable_images"]
                and all(
                    data.completed_evidence_scans.get(scan)
                    == "completed"
                    for scan in pending_image_scans
                )
            ),
        }

        output = render_result(
            paper, segments, media, data, run, key
        )

        store.save(data)
        budget.close()

        if export_result:
            export_path = (
                store.path.parent
                / f"selected_evidence_{run_id}.json"
            )
            export_path.write_text(
                json.dumps(
                    output,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            run["result_file"] = str(export_path.resolve())
            output["export_file"] = str(export_path.resolve())

            store.save(data)

    print(f"Selected Evidence status: {run['status']}")
    print(
        f"Evidence IDs: {len(output['evidence_ids'])}; "
        f"new Evidence: {run['new_evidence']}"
    )
    print(f"Claim links created: {run['new_links']}")
    print(f"Analysis JSON: {store.path.resolve()}")
    if export_result:
        print("Export JSON:", output["export_file"])

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Evidence-only extraction from a reader selection"
    )

    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--selection-file", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument(
        "--image-mode",
        choices=["none", "relevant", "all"],
        default="relevant",
    )
    parser.add_argument(
        "--question-limit-usd",
        type=float,
        default=0.50,
    )
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--export-result", action="store_true")

    args = parser.parse_args()

    if (
        not args.paper.is_file()
        or not args.selection_file.is_file()
    ):
        parser.error("Paper or selection JSON does not exist.")

    if not 0 < args.question_limit_usd < float("inf"):
        parser.error("Question budget must be finite and positive.")

    selection = json.loads(
        args.selection_file.read_text(encoding="utf-8-sig")
    )

    result = run_selection_evidence(
        paper_path=args.paper,
        selection=selection,
        model=args.model,
        image_mode=args.image_mode,
        limit_usd=args.question_limit_usd,
        preview=args.preview,
        export_result=args.export_result,
    )

    print("Status:", result["status"])


if __name__ == "__main__":
    main()
