"""Direct Evidence discovery without requiring a pre-existing Claim Pool.

This is an alternate entry point for the existing Evidence Agent, NOT an
additional agent. It saves ordinary EvidenceRecord items in the same pool.
Scanning is resumable at source-chunk and image level. No arbitrary maximum
silently removes source candidates; the existing per-question budget guards calls.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from app.core.token_report import task_token_report, print_task_tokens
from app.agents.evidence_agent import EvidenceAgent
from app.budget import PaperBudget, BudgetStop
from app.core.analysis_store import AnalysisStore, default_analysis_path
from app.evidence_runner import digest, make_chunks, save_findings
from app.llm import StructuredOutputParseError
from app.models.document_v2 import PaperDocument
from app.tools.image_inputs import ImageInputs
from app.usage_cost import CostTracker
from openai import APIStatusError, APIConnectionError, APITimeoutError


MODEL_ERRORS = (BudgetStop, StructuredOutputParseError, APIStatusError,
                APIConnectionError, APITimeoutError)


def direct_search_key(paper, *, question, model, image_mode, section_ids, element_ids):
    """Stable identity of the requested source discovery scope."""
    identity = {
        "direct_evidence": True, "paper_revision": paper.revision,
        "question": question.strip(), "model": model,
        "image_mode": image_mode,
        "section_ids": sorted(section_ids),
        "element_ids": sorted(element_ids),
    }
    # Invalidate old discovery receipts ONLY when the requested scope can
    # contain a table. Text/figure-only searches keep their prior cache keys.
    includes_table = any(
        element.kind == "table" and (
            not section_ids and not element_ids
            or element.element_id in element_ids
            or element.section_id in section_ids
        )
        for element in paper.elements.values()
    )
    if includes_table:
        identity["source_pack_version"] = "table_row_context_v2"
    return digest(identity)


def select_chunks(paper, section_ids, element_ids):
    """Keep exact requested source elements, not unrelated chunk neighbors."""
    source_chunks = make_chunks(paper)
    if not section_ids and not element_ids:
        return source_chunks
    selected = []
    for chunk in source_chunks:
        records = [
            item for item in chunk["elements"]
            if item["section_id"] in section_ids
            or item["element_id"] in element_ids
        ]
        if records:
            selected.append({
                "chunk_id": digest({"section": chunk["section_id"], "elements": records}),
                "section_id": chunk["section_id"],
                "elements": records,
            })
    return selected


def run_direct_discovery(*, paper_path, question, section_ids=(), element_ids=(),
                         model="gpt-5.4-mini", image_mode="relevant", preview=False):
    """Discover requested source material and persist it in analysis.json.

    Completion means only that planned source scans ran, NOT that all relevant
    evidence was necessarily recalled or independently verified.
    """
    paper_path = Path(paper_path)
    paper = PaperDocument.model_validate_json(paper_path.read_text(encoding="utf-8"))
    section_ids = set(section_ids)
    element_ids = set(element_ids)
    unknown_sections = section_ids - {section.section_id for section in paper.sections}
    unknown_elements = element_ids - set(paper.elements)
    if unknown_sections or unknown_elements:
        raise ValueError(f"Unknown source scope: sections={sorted(unknown_sections)} elements={sorted(unknown_elements)}")

    chunks = select_chunks(paper, section_ids, element_ids)
    store = AnalysisStore(default_analysis_path(paper.paper_id))
    data = store.load(paper)
    key = direct_search_key(
        paper, question=question, model=model, image_mode=image_mode,
        section_ids=section_ids, element_ids=element_ids,
    )
    old = data.direct_evidence_searches.get(key) or {}
    if old.get("status") == "completed":
        print(f"Reusing direct Evidence search: {key} ({len(old.get('evidence_ids', []))} records)")
        return {"status": "reused", "search_key": key, "evidence_ids": list(old.get("evidence_ids", []))}

    print("Direct Evidence source chunks:", len(chunks))
    print("Image mode:", image_mode)
    if preview:
        print("Preview only; no API calls or file writes.")
        return {"status": "preview", "search_key": key, "evidence_ids": list(old.get("evidence_ids", []))}

    tracker = CostTracker()
    budget = PaperBudget(paper.paper_id)
    agent = EvidenceAgent(model=model, tracker=tracker, budget=budget)
    images = ImageInputs(
        paper, document_dir=paper_path.parent,
        enabled=image_mode != "none", max_images_per_call=1,
    )
    run_id = uuid4().hex[:12]
    start = perf_counter()
    run = {
        "run_id": run_id, "stage": "evidence", "mode": "direct_discovery",
        "model": model, "source_question": question,
        "section_ids": sorted(section_ids), "element_ids": sorted(element_ids),
        "image_mode": image_mode, "status": "running", "search_key": key,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "text_scans_completed": 0, "image_scans_completed": 0,
        "new_evidence": 0,
        "issues": [], "unavailable_images": [],
    }
    data.runs.append(run)
    found = list(old.get("evidence_ids") or [])
    data.direct_evidence_searches[key] = {
        **old,
        "status": "partial", "question": question,
        "paper_revision": paper.revision, "model": model,
        "image_mode": image_mode, "section_ids": sorted(section_ids),
        "element_ids": sorted(element_ids), "evidence_ids": list(dict.fromkeys(found)),
        "run_id": run_id,
    }
    scan_keys = []
    image_scan_keys = []
    stopped = False
    store.save(data)

    try:
        for chunk in chunks:
            scan = digest({"search_key": key, "kind": "text", "chunk": chunk["chunk_id"]})
            scan_keys.append(scan)
            if data.completed_evidence_scans.get(scan) == "completed":
                # Preserve results found in earlier interrupted runs.
                continue
            try:
                extraction, _ = agent.extract(
                    [], chunk["elements"], source_focus=question,
                )
            except StructuredOutputParseError as exc:
                run["issues"].append(f"Text scan pending at {chunk['chunk_id']}: {type(exc).__name__}")
                store.save(data)
                continue
            except MODEL_ERRORS as exc:
                run["issues"].append(f"Text search interrupted at {chunk['chunk_id']}: {type(exc).__name__}")
                stopped = True
                break
            found.extend(save_findings(
                extraction=extraction, source_elements=chunk["elements"],
                claim_batch=[], paper=paper, store=store, data=data,
                run=run, run_id=run_id,
            ))
            data.evidence_image_candidates[scan] = list(dict.fromkeys(
                extraction.image_candidates
            ))
            data.completed_evidence_scans[scan] = "completed"
            run["text_scans_completed"] += 1
            data.direct_evidence_searches[key] = {
                "status": "partial", "question": question,
                "paper_revision": paper.revision, "model": model,
                "image_mode": image_mode, "section_ids": sorted(section_ids),
                "element_ids": sorted(element_ids), "evidence_ids": list(dict.fromkeys(found)),
                "run_id": run_id, "text_scan_complete": False,
                "image_scan_complete": False,
            }
            store.save(data)

        image_candidates = set()
        if image_mode == "all":
            image_candidates.update(
                element_id for element_id in paper.elements
                if images.asset_refs(element_id)
                and (not section_ids and not element_ids
                     or paper.elements[element_id].section_id in section_ids
                     or element_id in element_ids)
            )
        elif image_mode == "relevant":
            for scan in scan_keys:
                image_candidates.update(data.evidence_image_candidates.get(scan, []))
            # Explicitly named tables/figures in the question should not be
            # silently missed merely because the caption-only scan skipped them.
            normalized_question = question.casefold()
            for element_id, element in paper.elements.items():
                if not images.asset_refs(element_id):
                    continue
                if (section_ids or element_ids) and (
                    element.section_id not in section_ids and element_id not in element_ids
                ):
                    continue
                label = str(getattr(element, "label", "") or "").strip()
                if label and label.casefold().rstrip(".") in normalized_question:
                    image_candidates.add(element_id)
        available = {
            element_id for element_id in image_candidates
            if element_id in paper.elements
            and images.asset_refs(element_id)
            and (
                not section_ids and not element_ids
                or paper.elements[element_id].section_id in section_ids
                or element_id in element_ids
            )
        }
        for element_id in sorted(available):
            if stopped:
                break
            element = paper.elements[element_id]
            for asset_id, cell_id in images.asset_refs(element_id):
                try:
                    prepared = images.prepare(element_id, asset_id, cell_id)
                except (OSError, ImportError, ValueError) as exc:
                    run["unavailable_images"].append({
                        "element_id": element_id, "asset_id": asset_id,
                        "message": str(exc),
                    })
                    continue
                scan = digest({
                    "search_key": key, "kind": "image",
                    "element_id": element_id, "asset_id": asset_id,
                    "cell_id": cell_id, "sha256": prepared.source_sha256,
                })
                image_scan_keys.append(scan)
                if data.completed_evidence_scans.get(scan) == "completed":
                    continue
                source_item = {
                    "element_id": element_id, "kind": element.kind,
                    "asset_id": asset_id, "cell_id": cell_id,
                    "caption": element.caption, "label": element.label,
                }
                try:
                    extraction, response = agent.extract(
                        [], [source_item], image=prepared, source_focus=question,
                    )
                except StructuredOutputParseError as exc:
                    run["issues"].append(f"Image scan pending at {element_id}: {type(exc).__name__}")
                    store.save(data)
                    continue
                except MODEL_ERRORS as exc:
                    run["issues"].append(
                        f"Image search interrupted at {element_id}: {type(exc).__name__}"
                    )
                    stopped = True
                    break
                images.record_submission([prepared], model, response)
                receipt = {
                    **prepared.receipt(),
                    "model": getattr(response, "model", model),
                    "response_id": getattr(response, "id", None),
                }
                found.extend(save_findings(
                    extraction=extraction, source_elements=[source_item],
                    claim_batch=[], paper=paper, store=store, data=data,
                    run=run, run_id=run_id, image_receipt=receipt,
                ))
                data.completed_evidence_scans[scan] = "completed"
                run["image_scans_completed"] += 1
                data.direct_evidence_searches[key]["evidence_ids"] = list(dict.fromkeys(found))
                store.save(data)

        text_complete = all(
            data.completed_evidence_scans.get(scan) == "completed" for scan in scan_keys
        ) and not stopped
        image_complete = all(
            data.completed_evidence_scans.get(scan) == "completed" for scan in image_scan_keys
        ) and not stopped and not run["unavailable_images"]
        status = "completed" if text_complete and image_complete else "partial"
        run["status"] = status
        data.direct_evidence_searches[key] = {
            "status": status, "question": question,
            "paper_revision": paper.revision, "model": model,
            "image_mode": image_mode,
            "section_ids": sorted(section_ids),
            "element_ids": sorted(element_ids),
            "evidence_ids": list(dict.fromkeys(found)),
            "run_id": run_id, "text_scan_complete": text_complete,
            "image_scan_complete": image_complete,
        }
    except BaseException:
        run["status"] = "interrupted"
        raise
    finally:
        run["finished_at"] = datetime.now(timezone.utc).isoformat()
        run["duration_seconds"] = round(perf_counter() - start, 3)
        run["api_cost_report"] = tracker.report()
        run["budget_report"] = budget.report()
        run["token_usage_report"] = task_token_report(run)
        print_task_tokens(run["token_usage_report"], run["budget_report"])
        run["image_report"] = images.report()
        store.save(data)
        budget.close()

    print("=== DIRECT EVIDENCE DISCOVERY ===")
    print("Status:", run["status"])
    print("Evidence records for query:", len(data.direct_evidence_searches[key]["evidence_ids"]))
    print("Estimated USD:", run["api_cost_report"].get("estimated_total_usd"))
    return {"status": run["status"], "search_key": key,
            "evidence_ids": data.direct_evidence_searches[key]["evidence_ids"]}


def main():
    parser = argparse.ArgumentParser(description="Direct Evidence discovery")
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--section", action="append", default=[])
    parser.add_argument("--element-id", action="append", default=[])
    parser.add_argument("--image-mode", choices=["none", "relevant", "all"], default="relevant")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    run_direct_discovery(
        paper_path=args.paper, question=args.question,
        section_ids=args.section, element_ids=args.element_id,
        model=args.model, image_mode=args.image_mode, preview=args.preview,
    )


if __name__ == "__main__":
    main()
