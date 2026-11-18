import argparse
import hashlib
import json

from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from app.core.token_report import task_token_report, print_task_tokens
from app.agents.evidence_agent import EvidenceAgent
from app.budget import PaperBudget, BudgetStop
from app.llm import StructuredOutputParseError
from app.core.analysis_store import (
    AnalysisStore,
    default_analysis_path,
)
from app.core.evidence_locator import locate_evidence
from app.core.evidence_integrity import normalize_table_descriptions, _table_facts
from app.core.retired_associations import (
    SOURCE_SCAN_POLICY, scan_policy, scan_hash_compatibility_fields,
)

from app.models.document_v2 import (
    PaperDocument,
    Paragraph,
    Table,
    Figure,
    Equation,
)

from app.tools.image_inputs import ImageInputs
from app.usage_cost import CostTracker

DISCOVERY_POLICY = "claim_topic_element_coverage_v7"
SOURCE_SELECTION_POLICY = "claim_source_catalog_v2_context"


def digest(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()[:24]


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def table_row_groups(cells, body_rows):
    """Best-effort dataset/method row groups; never exclude a source row.

    Some HTML tables split one dataset name across two consecutive rows.
    Only infer grouping when column headers and a repeated method cycle make
    the boundaries clear. Unrecognized tables retain ordinary row chunking.
    """
    by_position = {(cell.row, cell.col): cell for cell in cells}
    headers = {
        cell.col: " ".join(cell.text.casefold().split())
        for cell in cells
        if cell.row == 0 or cell.role == "column_header"
    }

    def header_col(names):
        return next((col for col, label in sorted(headers.items())
                     if label in names), None)

    dataset_col = header_col({
        "dataset", "data set", "data source", "corpus", "benchmark",
        "scenario", "group", "category", "cohort",
    })
    method_col = header_col({
        "method", "model", "algorithm", "approach", "system",
        "technique", "baseline",
    })
    rows = sorted(set(body_rows))
    if not rows:
        return [], []

    label_fragments = [
        {"row": row, "text": by_position[(row, dataset_col)].text.strip()}
        for row in rows
        if dataset_col is not None and (row, dataset_col) in by_position
        and by_position[(row, dataset_col)].text.strip()
    ]

    if dataset_col is not None and method_col is not None:
        first = by_position.get((rows[0], method_col))
        first_method = first.text.strip().casefold() if first else ""
        starts = [
            row for row in rows
            if first_method
            and (row, method_col) in by_position
            and by_position[(row, method_col)].text.strip().casefold() == first_method
            and (row, dataset_col) in by_position
            and by_position[(row, dataset_col)].text.strip()
        ]
        if len(starts) >= 2 and starts[0] == rows[0]:
            grouped = []
            for index, start in enumerate(starts):
                stop = starts[index + 1] if index + 1 < len(starts) else None
                group_rows = [r for r in rows if r >= start and (stop is None or r < stop)]
                fragments = [entry for entry in label_fragments if entry["row"] in group_rows]
                # Values come verbatim from the dataset column, not model text.
                # Joining fragments is a soft reading aid, not a correctness gate.
                distinct = list(dict.fromkeys(entry["text"] for entry in fragments))
                grouped.append({
                    "rows": group_rows,
                    "dataset_label": " ".join(distinct) if distinct else None,
                    "label_fragments": fragments,
                    "basis": "repeated_method_cycle",
                })
            if sum(len(g["rows"]) for g in grouped) == len(rows):
                return grouped, label_fragments

    # No reliable pattern: leave every row accessible, without guessed labels.
    return [
        {"rows": group, "dataset_label": None,
         "label_fragments": [entry for entry in label_fragments if entry["row"] in group],
         "basis": "sequential_fallback"}
        for group in batches(rows, 10)
    ], label_fragments


def table_row_context(table, requested_cell_ids):
    """Describe the source row groups of referenced cells without rejecting any."""
    if not table.grid or not table.grid.cells:
        return []
    cells = table.grid.cells
    ids = {cell.cell_id: cell for cell in cells}
    headers = {cell.cell_id for cell in cells if cell.row == 0 or cell.role == "column_header"}
    body_rows = sorted({cell.row for cell in cells if cell.cell_id not in headers})
    groups, _ = table_row_groups(cells, body_rows)
    chosen_rows = {ids[cid].row for cid in requested_cell_ids
                   if cid in ids and ids[cid].cell_id not in headers}
    return [
        {"dataset_label": group["dataset_label"],
         "label_fragments": group["label_fragments"],
         "row_start": group["rows"][0], "row_end": group["rows"][-1],
         "selected_rows": sorted(chosen_rows & set(group["rows"])),
         "basis": group["basis"]}
        for group in groups
        if group["rows"] and chosen_rows.intersection(group["rows"])
    ]


def make_chunks(paper, chunk_chars=6000):
    """Build source-aware text/table/caption/equation chunks."""
    chunks = []

    mentions = {}

    for mention in paper.mentions:
        element_id = mention.source_anchor.element_id

        mentions.setdefault(element_id, []).append({
            "target_kind": mention.target_kind,
            "target_id": mention.target_id,
            "mention_text": mention.mention_text,
        })

    for section in paper.sections:
        current = []
        current_size = 0

        def flush():
            nonlocal current, current_size

            if current:
                chunk_id = digest({
                    "section_id": section.section_id,
                    "elements": current,
                })

                chunks.append({
                    "chunk_id": chunk_id,
                    "section_id": section.section_id,
                    "elements": current,
                })

                current = []
                current_size = 0

        def add(item):
            nonlocal current_size

            size = len(json.dumps(
                item, ensure_ascii=False
            ))

            if current and current_size + size > chunk_chars:
                flush()

            current.append(item)
            current_size += size

        for element_id in section.block_ids:
            element = paper.elements[element_id]

            base = {
                "element_id": element_id,
                "kind": element.kind,
                "section_id": section.section_id,
                "section_title": section.title,
            }

            if isinstance(element, Paragraph):
                text = element.text
                start = 0

                while start < len(text):
                    end = min(start + 2600, len(text))

                    add({
                        **base,
                        "text": text[start:end],
                        "source_start": start,
                        "source_end": end,
                        "references": mentions.get(
                            element_id, []
                        ),
                    })

                    if end == len(text):
                        break

                    start = end - 120

            elif isinstance(element, Table):
                common = {
                    **base,
                    "label": element.label,
                    "caption": element.caption,
                    "representation": element.representation,
                    "recognition_status": element.recognition_status,
                }
                # Keep table source groups apart from adjacent paragraphs and
                # other dataset groups. This prevents cross-dataset blending.
                flush()
                if element.grid and element.grid.cells:
                    cells = element.grid.cells
                    header_cells = [c for c in cells if c.role == "column_header" or c.row == 0]
                    header_ids = {c.cell_id for c in header_cells}
                    body_rows = sorted({c.row for c in cells if c.cell_id not in header_ids})
                    if not body_rows:
                        body_rows = sorted({c.row for c in cells})
                    groups, label_index = table_row_groups(cells, body_rows)
                    for group in groups:
                        # Large dataset groups are split only for request size;
                        # their dataset/column context stays with EVERY part.
                        for rows_part in batches(group["rows"], 10):
                            row_set = set(rows_part)
                            selected = [c for c in cells if c.row in row_set or c.cell_id in header_ids]
                            context = {
                                "dataset_label": group["dataset_label"],
                                "label_fragments": group["label_fragments"],
                                "row_start": group["rows"][0],
                                "row_end": group["rows"][-1],
                                "basis": group["basis"],
                            }
                            add({
                                **common,
                                "row_group_context": context,
                                # Retain raw labels for fallback when a table
                                # does not have an inferable dataset grouping.
                                "row_label_index": label_index if group["basis"] == "sequential_fallback" else None,
                                "cells": [
                                    {"cell_id": c.cell_id, "row": c.row,
                                     "col": c.col, "role": c.role, "text": c.text[:800]}
                                    for c in selected
                                ],
                            })
                            flush()
                else:
                    add({**common, "cells": []})
                    flush()

            elif isinstance(element, Figure):
                add({
                    **base,
                    "label": element.label,
                    "caption": element.caption,
                    "alt_text": element.alt_text,
                    "image_asset_id": element.image_asset_id,
                })

            elif isinstance(element, Equation):
                add({
                    **base,
                    "label": element.label,
                    "latex": element.latex,
                    "plain_text": element.plain_text,
                    "mathml": (
                        element.mathml
                        if not element.latex else None
                    ),
                })

        flush()

    return chunks



def _source_preview(item):
    """Index source content as hints; extraction still reads canonical originals."""
    kind = item.get("kind")
    if kind == "paragraph":
        value = str(item.get("text") or "")
        if len(value) > 1100:
            value = value[:700] + " ... " + value[-350:]
    elif kind == "table":
        labels = [str(c.get("text") or "")[:80] for c in item.get("cells", [])
                  if c.get("role") == "column_header" or c.get("row") == 0]
        rows = [str(c.get("text") or "")[:60] for c in item.get("cells", [])
                if c.get("col") == 0 and c.get("row") != 0]
        value = " ".join([str(item.get("caption") or "")[:600],
                          "Headers: " + " | ".join(labels[:16]),
                          "Rows: " + " | ".join(rows[:14])])
    elif kind == "figure":
        value = " ".join([str(item.get("label") or ""),
                          str(item.get("caption") or ""),
                          str(item.get("alt_text") or "")])
    else:
        value = " ".join(str(item.get(k) or "") for k in
                         ("label", "latex", "plain_text"))
    return value[:1400]


def source_catalog(chunks):
    """One catalog entry per canonical paper element (not per chunk split)."""
    catalog = {}
    for chunk in chunks:
        for item in chunk["elements"]:
            element_id = item["element_id"]
            entry = {
                "element_id": element_id,
                "kind": item["kind"],
                "section_title": item.get("section_title") or "",
                "section_id": item["section_id"],
                "preview": _source_preview(item),
            }
            existing = catalog.get(element_id)
            if existing is None:
                catalog[element_id] = entry
            elif len(existing["preview"]) < 900:
                # Split long paragraphs and tables can expose new content.
                previous = existing["preview"]
                extra = entry["preview"]
                if extra and extra not in previous:
                    existing["preview"] = (previous + " ... " + extra)[:1500]
    return list(catalog.values())


def catalog_batches(catalog, max_chars=17000):
    """Every source enters exactly one selection request, even on long papers."""
    group = []
    size = 0
    for entry in catalog:
        entry_size = len(json.dumps(entry, ensure_ascii=False))
        if group and size + entry_size > max_chars:
            yield group
            group, size = [], 0
        group.append(entry)
        size += entry_size
    if group:
        yield group


def claim_focus_context(paper, claims):
    """Give both Evidence operations the Claim's real scientific context.

    Context is drawn only from canonical paper elements. It does not add
    new Claims or determine support; it disambiguates task-specific wording.
    """
    section_by_element = {
        element_id: section
        for section in paper.sections
        for element_id in section.block_ids
    }
    context = []
    for element_id in dict.fromkeys(c.source_element_id for c in claims):
        element = paper.elements.get(element_id)
        section = section_by_element.get(element_id)
        origin_text = element.text if isinstance(element, Paragraph) else ""
        context.append({
            "claim_source_element_id": element_id,
            "source_section_id": section.section_id if section else None,
            "source_section_title": section.title if section else "",
            "source_paragraph": origin_text[:3200],
        })
    return context


def mandatory_context_elements(paper, claims):
    """Protect context around each Claim to resolve anaphora and nearby figures."""
    origin_ids = {claim.source_element_id for claim in claims}
    keep = set(origin_ids)
    for section in paper.sections:
        blocks = list(section.block_ids)
        for index, element_id in enumerate(blocks):
            if element_id in origin_ids:
                # Same-section neighbors often omit the subject in the sentence.
                keep.update(blocks[max(0, index-2):index+3])
                if len(blocks) <= 8:
                    keep.update(blocks)
    for mention in paper.mentions:
        if (mention.source_anchor.element_id in origin_ids
                and mention.target_id in paper.elements):
            keep.add(mention.target_id)
    return keep


def focused_chunks(chunks, selected_element_ids):
    """Retain chosen canonical source elements and stable scan identities."""
    selected = set(selected_element_ids)
    narrowed = []
    for chunk in chunks:
        elements = [item for item in chunk["elements"]
                    if item["element_id"] in selected]
        if elements:
            narrowed.append({
                "section_id": chunk["section_id"],
                "chunk_id": (chunk["chunk_id"] if len(elements) == len(chunk["elements"])
                             else digest({"section": chunk["section_id"],
                                          "elements": elements})),
                "elements": elements,
            })
    return narrowed


def isolated_source_chunks(chunks):
    """One canonical source element/window per extraction call.

    A selected paragraph must not disappear merely because another paragraph
    in its section dominates a multi-element extraction response. Preserve
    separate 2,600-character windows and existing table row groups; never
    truncate their source text or merge different experimental elements.
    """
    result = []
    for chunk in chunks:
        for item in chunk["elements"]:
            parts = [item]
            result.append({
                "section_id": chunk["section_id"],
                "chunk_id": digest({
                    "section_id": chunk["section_id"],
                    "source_elements": parts,
                }),
                "elements": parts,
            })
    return result


def source_plan(agent, paper, claims, chunks, previous_runs=(), *, rerun=False):
    """Search full source index; reuse completed per-Claim source plans on resume."""
    catalog = source_catalog(chunks)
    valid = {entry["element_id"] for entry in catalog}
    index_fingerprint = digest(catalog)
    focus_ids = sorted({claim.claim_id for claim in claims})
    protected = set(mandatory_context_elements(paper, claims)) & valid
    if not rerun:
        for old in reversed(previous_runs):
            receipt = old.get("source_selection") or {}
            if (receipt.get("policy") == SOURCE_SELECTION_POLICY
                    and receipt.get("index_fingerprint") == index_fingerprint
                    and old.get("paper_revision") == paper.revision
                    and old.get("model") == agent.model
                    and sorted(old.get("claim_ids") or []) == focus_ids):
                reused = set(receipt.get("selected_element_ids") or [])
                if reused and reused <= valid and protected <= reused:
                    return reused, {**receipt, "reused_from_run_id": old.get("run_id")}
    selected = set(protected)
    proposals = set()
    batches_scanned = 0
    for group in catalog_batches(catalog):
        response = agent.select_sources(
            claims, group, target_context=claim_focus_context(paper, claims)
        )
        allowed = {entry["element_id"] for entry in group}
        returned = set(response.element_ids)
        unknown = returned - allowed
        if unknown:
            raise ValueError("Source selection returned unknown element IDs: "
                             + ", ".join(sorted(unknown)[:12]))
        proposals.update(returned)
        batches_scanned += 1
    selected.update(proposals)
    return selected, {
        "policy": SOURCE_SELECTION_POLICY,
        "index_fingerprint": index_fingerprint,
        "catalog_elements": len(catalog),
        "catalog_batches_scanned": batches_scanned,
        "model_selected_elements": len(proposals),
        "protected_context_elements": sorted(mandatory_context_elements(paper, claims) & valid),
        "selected_element_ids": sorted(selected),
        "excluded_catalog_elements": len(valid - selected),
        "note": "All catalog elements were screened. Extraction used selected originals only; this is not a guarantee of scientific completeness.",
    }


def scan_key(paper, claim_ids, chunk_id, model):
    return digest({
        "stage": "text",
        "discovery_policy": DISCOVERY_POLICY,
        **scan_hash_compatibility_fields(),
        "paper_revision": paper.revision,
        "claims": sorted(claim_ids),
        "chunk_id": chunk_id,
        "model": model,
    })


def image_key(
    paper, claim_ids, image, model,
):
    return digest({
        "stage": "image",
        "discovery_policy": DISCOVERY_POLICY,
        **scan_hash_compatibility_fields(),
        "paper_revision": paper.revision,
        "claims": sorted(claim_ids),
        "element_id": image.element_id,
        "asset_id": image.asset_id,
        "cell_id": image.cell_id,
        "source_sha256": image.source_sha256,
        "model": model,
    })


def historical_scan_index(paper, data, model, text_chunks, chunk_chars=6000):
    """Reconstruct completed per-Claim source coverage from saved batch receipts.

    A prior run may be partial, but an individual completed scan is reusable.
    Cached hash matching is mandatory; a run-level `completed` flag alone
    does not prove that a particular chunk was scanned. Earlier versions used
    claim batches of 15 and chunks of 6000 chars; mismatches simply miss cache.
    Nothing in analysis.json is rewritten by this reconstruction.
    """
    coverage = defaultdict(set)
    image_candidates = defaultdict(set)
    old_batches = []
    for run in [*data.evidence_scan_batches, *data.runs]:
        if (run.get("stage") != "evidence" or run.get("model") != model
                or scan_policy(run) != SOURCE_SCAN_POLICY
                or run.get("discovery_policy") != DISCOVERY_POLICY):
            continue
        if run.get("paper_revision") not in (None, paper.revision):
            continue
        run_chunk_chars = run.get("chunk_chars", 6000)
        if run_chunk_chars != chunk_chars:
            continue
        size = run.get("claim_batch_size", 15)
        if not isinstance(size, int) or size < 1:
            continue
        ids = [cid for cid in run.get("claim_ids", []) if cid in data.claim_pool]
        for group in batches(ids, size):
            group_ids = [claim.claim_id if hasattr(claim, "claim_id") else claim for claim in group]
            old_batches.append(group_ids)
            for chunk in text_chunks:
                key = scan_key(paper, group_ids, chunk["chunk_id"], model)
                if data.completed_evidence_scans.get(key) != "completed":
                    continue
                coverage[chunk["chunk_id"]].update(group_ids)
                image_candidates[chunk["chunk_id"]].update(
                    data.evidence_image_candidates.get(key, [])
                )
    # Newer scans also record per-Claim receipts for arbitrary future batches.
    for chunk in text_chunks:
        for cid in data.claim_pool:
            key = scan_key(paper, [cid], chunk["chunk_id"], model)
            if data.completed_evidence_scans.get(key) == "completed":
                coverage[chunk["chunk_id"]].add(cid)
                image_candidates[chunk["chunk_id"]].update(
                    data.evidence_image_candidates.get(key, [])
                )
    return coverage, image_candidates, old_batches


def historical_image_claims(paper, data, model, image, old_batches, claim_ids):
    """Find previously submitted image requests for these exact image bytes."""
    covered = set()
    for group in old_batches:
        if not (set(group) & set(claim_ids)):
            continue
        key = image_key(paper, group, image, model)
        if data.completed_evidence_scans.get(key) == "completed":
            covered.update(group)
    for cid in claim_ids:
        key = image_key(paper, [cid], image, model)
        if data.completed_evidence_scans.get(key) == "completed":
            covered.add(cid)
    return covered


def mark_text_covered(data, paper, model, chunk_id, claim_ids, candidates):
    """Record verified per-Claim coverage, without deleting older receipts."""
    for cid in claim_ids:
        key = scan_key(paper, [cid], chunk_id, model)
        data.completed_evidence_scans[key] = "completed"
        data.evidence_image_candidates[key] = list(dict.fromkeys(candidates))


def mark_image_covered(data, paper, model, image, claim_ids):
    for cid in claim_ids:
        data.completed_evidence_scans[
            image_key(paper, [cid], image, model)
        ] = "completed"


def image_mode_covers(old_mode, required):
    levels = {
        "none": 0,
        "relevant": 1,
        "all": 2,
    }

    return levels.get(old_mode, -1) >= levels[required]


def saved_search_covers(record, source_sections, source_elements):
    """Check whether an earlier scan covers this requested source scope."""
    mode = record.get("search_scope")
    if mode == "whole_paper":
        return True
    if mode != "selected_sources":
        return False
    # Scoped scans can be reused for narrower queries, but not broader ones.
    return (
        set(record.get("source_section_ids") or []) >= set(source_sections)
        and set(record.get("source_element_ids") or []) >= set(source_elements)
    )


def source_scoped_chunks(paper, chunks, source_sections, source_elements):
    """Preserve only explicitly selected source records per chunk."""
    if not source_sections and not source_elements:
        return chunks
    result = []
    for chunk in chunks:
        chosen = [
            item for item in chunk["elements"]
            if item["section_id"] in source_sections
            or item["element_id"] in source_elements
        ]
        if chosen:
            result.append({
                "section_id": chunk["section_id"],
                "chunk_id": (chunk["chunk_id"] if len(chosen) == len(chunk["elements"])
                             else digest({"section": chunk["section_id"], "elements": chosen})),
                "elements": chosen,
            })
    return result


def select_claims(args, data, paper):
    if args.claim_id:
        ids = list(dict.fromkeys(args.claim_id))

    elif args.section:
        ids = [
            cid for cid, claim in data.claim_pool.items()
            if (
                claim.source_element_id in paper.elements
                and paper.elements[
                    claim.source_element_id
                ].section_id == args.section
            )
        ]

    else:
        ids = list(data.claim_pool)

    unknown = [
        cid for cid in ids
        if cid not in data.claim_pool
    ]

    if unknown:
        raise ValueError(
            f"Unknown Claim IDs: {unknown}"
        )

    result = []

    for cid in ids:
        claim = data.claim_pool[cid]
        old = data.evidence_search_by_claim.get(cid, {})

        reusable = (
            old.get("status") == "completed"
            and old.get("discovery_policy") == DISCOVERY_POLICY
            and old.get("paper_revision") == paper.revision
            and old.get("model") == args.model
            and saved_search_covers(
                old, args.source_sections, args.source_elements,
            )
            and image_mode_covers(
                old.get("image_mode"),
                args.image_mode,
            )
        )

        if args.rerun or not reusable:
            result.append(claim)

    return result


def save_findings(
    extraction,
    source_elements,
    claim_batch,
    paper,
    store,
    data,
    run,
    run_id,
    image_receipt=None,
    *,
    scan_id,
    scan_stage,
):
    """Persist admissible findings and expose genuine structural failures.

    A rejected candidate is a retriable scan-integrity failure, not proof that
    no Evidence exists. A deliberate Claim-source omission is NOT a failure.
    Saved records remain available during a retry and deduplicate normally.
    """
    result = {
        "evidence_ids": [],
        "created": 0,
        "reused": 0,
        "claim_origin_omitted": 0,
        "invalid": 0,
    }

    if not claim_batch:
        raise ValueError("Evidence findings require at least one target Claim.")

    def reject(raw, index, reason):
        result["invalid"] += 1
        run["rejected_findings"] += 1
        run.setdefault("finding_rejections", []).append({
            "stage": scan_stage,
            "scan_id": scan_id,
            "finding_index": index,
            "source_element_id": raw.source_element_id,
            "kind": raw.kind,
            "reason": reason,
            "source_quote_preview": str(raw.source_quote or "")[:240],
            "description_preview": str(raw.description or "")[:240],
        })

    for index, raw in enumerate(extraction.evidence):
        scanned = next((item for item in source_elements
                        if item.get("element_id") == raw.source_element_id), None)
        if scanned is None:
            reject(raw, index, "source_element_not_in_scanned_input")
            continue
        if scanned.get("kind") != raw.kind:
            reject(raw, index, "source_kind_mismatch")
            continue
        if raw.kind == "table" and scanned.get("cells") and raw.cell_ids:
            visible = {cell.get("cell_id") for cell in scanned["cells"]}
            if any(cid not in visible for cid in raw.cell_ids):
                reject(raw, index, "table_cell_not_in_scanned_input")
                continue
        record = locate_evidence(
            paper=paper,
            raw=raw,
            scanned_elements=source_elements,
            run_id=run_id,
            image_receipt=image_receipt,
        )

        if record.source_status != "located":
            reject(raw, index, "source_location_unverified")
            continue

        # A paragraph quotation supplied by the model must really align to the
        # original paper. The locator's fallback passage is not verification of
        # that quote; do not persist it as a successfully checked finding.
        if raw.kind == "paragraph" and raw.source_quote and not record.source_quote:
            reject(raw, index, "paragraph_quote_not_aligned")
            continue

        # Keep per-table row provenance; never widen the source to unrelated
        # dataset rows when the model selected a narrower row group.
        if record.kind == "table" and record.source_element_id in paper.elements:
            table = paper.elements[record.source_element_id]
            if isinstance(table, Table):
                ref_ids = list(raw.cell_ids or [])
                if not ref_ids:
                    ref_ids = [c.get("cell_id") for c in scanned.get("cells", [])
                               if c.get("cell_id")]
                provenance = table_row_context(table, ref_ids)
                if provenance:
                    record.source_content["row_group_context"] = provenance
                    record.source_content["row_group_context_note"] = (
                        "Best-effort grouping from original table headers and row labels; "
                        "original cell values remain authoritative."
                    )
                group = scanned.get("row_group_context") or {}
                if group and table.grid:
                    by_id = {cell.cell_id: cell for cell in table.grid.cells}
                    outside = sorted({
                        by_id[cid].row for cid in ref_ids
                        if cid in by_id and by_id[cid].row != 0
                        and not (group["row_start"] <= by_id[cid].row <= group["row_end"])
                    })
                    if outside:
                        record.source_content["referenced_rows_outside_scanned_group"] = outside

        # A Claim quotation is not independent Evidence about itself.
        if record.kind == "paragraph" and record.source_anchor is not None:
            from app.core.evidence_retrieval import same_claim_source
            if any(same_claim_source(claim, record) for claim in claim_batch):
                result["claim_origin_omitted"] += 1
                run["claim_origin_omitted"] += 1
                continue

        if record.kind == "table" and record.observed_via == "structured_table":
            factual_description = _table_facts(paper, record)
            if factual_description:
                record.description = factual_description

        if record.kind == "paragraph" and record.source_quote:
            selector = record.source_anchor.selector if record.source_anchor else None
            if (getattr(selector, "start", -1) < int(scanned.get("source_start", 0))
                    or getattr(selector, "end", 10**15) > int(
                        scanned.get("source_end", 10**15))):
                reject(raw, index, "paragraph_quote_outside_scanned_window")
                continue

        evidence_id, created = store.add_evidence(data, record)
        if not created and record.kind == "table":
            existing = data.evidence_pool[evidence_id]
            for field in ("row_group_context", "row_group_context_note",
                          "referenced_rows_outside_scanned_group"):
                if field in record.source_content:
                    existing.source_content.setdefault(field, record.source_content[field])

        result["evidence_ids"].append(evidence_id)
        if created:
            run["new_evidence"] += 1
            result["created"] += 1
        else:
            result["reused"] += 1

    result["evidence_ids"] = list(dict.fromkeys(result["evidence_ids"]))
    return result



def suggested_images(
    paper,
    data,
    batch,
    text_keys,
    image_mode,
    images,
    selected_source_ids=None,
):
    """Pixels need an admitted visual candidate or a direct Claim-source cite.

    Source selection is intentionally high recall; it is not permission to
    submit every selected figure to the visual model.
    """
    if image_mode == "none":
        return []

    available = {
        element_id
        for element_id in paper.elements
        if images.asset_refs(element_id)
    }
    candidates = set()
    for key in text_keys:
        candidates.update(data.evidence_image_candidates.get(key, []))

    origin_ids = {claim.source_element_id for claim in batch}
    for mention in paper.mentions:
        if (mention.source_anchor.element_id in origin_ids
                and mention.target_kind in {"figure", "table"}
                and mention.target_id):
            candidates.add(mention.target_id)

    return sorted(candidates & available & set(selected_source_ids or ()))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--paper",
        type=Path,
        required=True,
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--section",
        help="Process Claims from this section ID",
    )

    group.add_argument(
        "--claim-id",
        action="append",
        help="Specific Claim ID; repeatable",
    )

    group.add_argument(
        "--all-claims",
        action="store_true",
    )

    parser.add_argument(
        "--model",
        default="gpt-5.4-mini",
    )
    parser.add_argument(
        "--source-section", action="append", default=[],
        help="Limit paper Evidence scanning to specified source sections",
    )
    parser.add_argument(
        "--source-element-id", action="append", default=[],
        help="Limit paper Evidence scanning to specified source elements",
    )

    parser.add_argument(
        "--image-mode",
        choices=["none", "relevant", "all"],
        default="relevant",
    )

    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=6000,
    )

    parser.add_argument(
        "--claim-batch-size",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--preview",
        action="store_true",
    )

    parser.add_argument(
        "--rerun",
        action="store_true",
    )

    args = parser.parse_args()

    if args.chunk_chars < 1000:
        parser.error("--chunk-chars must be >= 1000")

    if args.claim_batch_size < 1:
        parser.error("--claim-batch-size must be >= 1")

    paper = PaperDocument.model_validate_json(
        args.paper.read_text(encoding="utf-8")
    )
    args.source_sections = set(args.source_section)
    args.source_elements = set(args.source_element_id)
    unknown_sections = args.source_sections - {part.section_id for part in paper.sections}
    unknown_elements = args.source_elements - set(paper.elements)
    if unknown_sections or unknown_elements:
        parser.error(
            f"Unknown source scope: sections={sorted(unknown_sections)} "
            f"elements={sorted(unknown_elements)}"
        )

    store = AnalysisStore(
        default_analysis_path(paper.paper_id)
    )

    data = store.load(paper)
    selected = select_claims(
        args, data, paper
    )

    chunks = source_scoped_chunks(
        paper,
        make_chunks(paper, chunk_chars=args.chunk_chars),
        args.source_sections,
        args.source_elements,
    )

    claim_batches = list(batches(
        selected,
        args.claim_batch_size,
    ))
    text_coverage, cached_candidates, image_batches = historical_scan_index(
        paper, data, args.model, chunks, args.chunk_chars,
    )
    new_text_calls = sum(
        bool({claim.claim_id for claim in batch} - text_coverage[chunk["chunk_id"]])
        for batch in claim_batches for chunk in chunks
    )

    images = ImageInputs(
        paper,
        document_dir=args.paper.parent,
        enabled=args.image_mode != "none",
        max_images_per_call=1,
    )

    print("Paper:", paper.paper_id)
    print("Selected Claims:", len(selected))
    print("Text chunks:", len(chunks))
    print("Claim batches:", len(claim_batches))
    print("Image mode:", args.image_mode)
    print("Source scope:", "selected_sources" if (args.source_sections or args.source_elements) else "whole_paper")
    print("Source sections:", sorted(args.source_sections))
    print("Source elements:", sorted(args.source_elements))
    print(
        "Maximum text matches before source selection:",
        len(chunks) * len(claim_batches),
    )
    print("Potential calls before source selection:", new_text_calls)
    print("Historical cache savings:", len(chunks) * len(claim_batches) - new_text_calls)

    if args.preview:
        for chunk in chunks:
            print(
                chunk["chunk_id"],
                chunk["section_id"],
                [
                    item["element_id"]
                    for item in chunk["elements"]
                ],
            )

        print(
            "Relevant image candidates are selected "
            "after text discovery."
        )
        return

    if not selected:
        print("No new Claims require this scan.")
        return

    # Normalize descriptions from original table cells before matching.
    # This does not touch Evidence IDs, anchors, or visual observations.
    corrected_descriptions = normalize_table_descriptions(paper, data)

    tracker = CostTracker()
    budget = PaperBudget(paper.paper_id)

    agent = EvidenceAgent(
        model=args.model,
        tracker=tracker,
        budget=budget,
    )

    run_id = uuid4().hex[:12]
    started = perf_counter()

    run = {
        "run_id": run_id,
        "stage": "evidence",
        "model": args.model,
        "discovery_policy": DISCOVERY_POLICY,
        "scan_cache_policy": SOURCE_SCAN_POLICY,
        "corrected_table_descriptions": corrected_descriptions,
        "reused_source_chunks": 0,
        "reused_image_assets": 0,
        "rejected_findings": 0,
        "finding_rejections": [],
        "source_processing": [],
        "claim_origin_omitted": 0,
        "topic_review": {
            "calls": 0,
            "candidate_findings": 0,
            "accepted_findings": 0,
            "excluded_findings": 0,
            "excluded_source_element_ids": [],
            "reviewed_image_candidates": 0,
        },
        "paper_revision": paper.revision,
        "claim_batch_size": args.claim_batch_size,
        "chunk_chars": args.chunk_chars,
        "claim_ids": [
            c.claim_id for c in selected
        ],
        "search_scope": "selected_sources" if (args.source_sections or args.source_elements) else "whole_paper",
        "source_section_ids": sorted(args.source_sections),
        "source_element_ids": sorted(args.source_elements),
        "image_mode": args.image_mode,
        "status": "running",
        "started_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "finished_at": None,
        "duration_seconds": None,
        "llm_wall_time_seconds": 0.0,
        "call_timings": [],
        "text_scans_completed": 0,
        "image_scans_completed": 0,
        "new_evidence": 0,
        "unavailable_images": [],
        "issues": [],
    }

    data.runs.append(run)
    # Save ordinary source Evidence only; Mapping owns all argument relations.
    store.save(data)

    stopped = False
    newly_seen_evidence = set()
    planned_text_keys = []
    planned_targeted_text_keys = []
    planned_targeted_text_windows = []
    planned_image_keys = []

    def call_agent(batch, elements, stage, image=None):
        start = perf_counter()

        try:
            return agent.extract(
                claims=batch,
                source_elements=elements,
                image=image,
                target_context=claim_focus_context(paper, batch),
            )
        finally:
            elapsed = round(
                perf_counter() - start, 3
            )

            run["llm_wall_time_seconds"] += elapsed

            run["call_timings"].append({
                "stage": stage,
                "duration_seconds": elapsed,
                "claim_count": len(batch),
            })

    def review_extraction(batch, elements, extraction, stage):
        start = perf_counter()
        try:
            approved, receipt = agent.review_topic(
                claims=batch,
                source_elements=elements,
                extraction=extraction,
                target_context=claim_focus_context(paper, batch),
            )
        finally:
            elapsed = round(perf_counter() - start, 3)
            run["llm_wall_time_seconds"] += elapsed
            run["call_timings"].append({
                "stage": f"topic_review_{stage}",
                "duration_seconds": elapsed,
                "claim_count": len(batch),
            })
        summary = run["topic_review"]
        summary["calls"] += int(receipt["model_call"])
        summary["candidate_findings"] += receipt["candidate_findings"]
        summary["accepted_findings"] += receipt["accepted_findings"]
        summary["excluded_findings"] += (
            receipt["candidate_findings"] - receipt["accepted_findings"]
        )
        summary["reviewed_image_candidates"] += receipt["reviewed_image_candidates"]
        summary["excluded_source_element_ids"] = sorted(set(
            summary["excluded_source_element_ids"]
        ) | set(receipt["excluded_source_elements"]))
        return approved

    try:
        # First screen the full canonical source catalog against the Claims.
        # Only the SAME Evidence Agent is used; there is no second Agent and
        # no persistent Claim/Evidence pairing or relevance-score record.
        selection_started = perf_counter()
        try:
            focused_ids, selection_receipt = source_plan(
                agent, paper, selected, chunks,
                previous_runs=data.runs[:-1], rerun=args.rerun,
            )
            run["source_selection"] = selection_receipt
            chunks = isolated_source_chunks(focused_chunks(chunks, focused_ids))
            text_coverage, cached_candidates, image_batches = historical_scan_index(
                paper, data, args.model, chunks, args.chunk_chars,
            )
            planned_targeted_text_windows = [
                {
                    "scan_id": scan_key(paper, [c.claim_id for c in batch],
                                        chunk["chunk_id"], args.model),
                    "element_id": chunk["elements"][0]["element_id"],
                    "source_start": chunk["elements"][0].get("source_start"),
                    "source_end": chunk["elements"][0].get("source_end"),
                    "claim_ids": [c.claim_id for c in batch],
                }
                for batch in claim_batches for chunk in chunks
            ]
            planned_targeted_text_keys = [x["scan_id"] for x in planned_targeted_text_windows]
            run["planned_targeted_text_scans"] = len(planned_targeted_text_keys)
            print("Targeted source elements:", len(focused_ids), flush=True)
            print("Targeted text chunks:", len(chunks), flush=True)
            store.save(data)
        except (StructuredOutputParseError, BudgetStop, ValueError, RuntimeError) as exc:
            run["issues"].append({
                "stage": "claim_source_selection", "error": str(exc),
                "status": "requires_resume",
            })
            stopped = True
            # Never write source-coverage receipts when source selection failed.
        finally:
            selection_elapsed = round(perf_counter() - selection_started, 3)
            run["llm_wall_time_seconds"] += selection_elapsed
            run["call_timings"].append({
                "stage": "source_selection",
                "duration_seconds": selection_elapsed,
                "claim_count": len(selected),
            })

        for claim_batch in ([] if stopped else claim_batches):
            if stopped:
                break

            batch_ids = [
                c.claim_id for c in claim_batch
            ]

            text_keys = []

            # PASS 1: Scan only source elements selected for these Claims.
            for chunk in chunks:
                key = scan_key(
                    paper,
                    batch_ids,
                    chunk["chunk_id"],
                    args.model,
                )

                planned_text_keys.append(key)
                text_keys.append(key)

                chunk_id = chunk["chunk_id"]
                if args.rerun:
                    data.completed_evidence_scans.pop(key, None)
                    data.evidence_image_candidates.pop(key, None)
                    for cid in batch_ids:
                        old_key = scan_key(paper, [cid], chunk_id, args.model)
                        data.completed_evidence_scans.pop(old_key, None)
                        data.evidence_image_candidates.pop(old_key, None)
                    cached_candidates[chunk_id].clear()
                batch_missing = (
                    list(claim_batch) if args.rerun else [
                        claim for claim in claim_batch
                        if claim.claim_id not in text_coverage[chunk_id]
                    ]
                )
                if not batch_missing:
                    # The matching source chunks were actually scanned before.
                    # Copy image candidate receipts into the current batch key.
                    candidates = sorted(cached_candidates[chunk_id])
                    data.completed_evidence_scans[key] = "completed"
                    data.evidence_image_candidates[key] = candidates
                    mark_text_covered(
                        data, paper, args.model, chunk_id, batch_ids, candidates,
                    )
                    run["source_processing"].append({
                        "stage": "text_scan", "scan_id": chunk_id,
                        "element_id": chunk["elements"][0]["element_id"],
                        "status": "reused_completed_scan",
                    })
                    continue

                try:
                    extraction, response = call_agent(
                        batch_missing,
                        chunk["elements"],
                        "text_scan",
                    )
                    raw_count = len(extraction.evidence)
                    extraction = review_extraction(
                        batch_missing, chunk["elements"], extraction, "text_scan"
                    )

                except StructuredOutputParseError as exc:
                    run["source_processing"].append({
                        "stage": "text_scan", "scan_id": chunk_id,
                        "element_id": chunk["elements"][0]["element_id"],
                        "status": "pending_parse_retry",
                    })
                    run["issues"].append({
                        "stage": "text_scan", "chunk_id": chunk["chunk_id"],
                        "error": str(exc), "status": "pending_parse_retry",
                    })
                    store.save(data)
                    continue
                except BudgetStop as exc:
                    run["source_processing"].append({
                        "stage": "text_scan", "scan_id": chunk_id,
                        "element_id": chunk["elements"][0]["element_id"],
                        "status": "pending_budget_retry",
                    })
                    run["issues"].append({
                        "stage": "text_scan", "chunk_id": chunk["chunk_id"],
                        "error": str(exc),
                    })
                    stopped = True
                    break
                except (ValueError, RuntimeError) as exc:
                    run["source_processing"].append({
                        "stage": "text_scan", "scan_id": chunk_id,
                        "element_id": chunk["elements"][0]["element_id"],
                        "status": "pending_review_retry",
                    })
                    run["issues"].append({
                        "stage": "text_topic_review", "chunk_id": chunk["chunk_id"],
                        "error": str(exc), "status": "pending_review_retry",
                    })
                    store.save(data)
                    continue

                saved = save_findings(
                    extraction=extraction,
                    source_elements=chunk["elements"],
                    claim_batch=batch_missing,
                    paper=paper,
                    store=store,
                    data=data,
                    run=run,
                    run_id=run_id,
                    scan_id=chunk_id,
                    scan_stage="text_scan",
                )
                newly_seen_evidence.update(saved["evidence_ids"])
                element = chunk["elements"][0]
                source_status = (
                    "pending_source_retry" if saved["invalid"] else
                    "evidence_saved_or_reused" if saved["evidence_ids"] else
                    "claim_origin_only" if saved["claim_origin_omitted"] else
                    "all_candidates_excluded_by_topic" if raw_count else
                    "no_related_findings_returned"
                )
                run["source_processing"].append({
                    "stage": "text_scan", "scan_id": chunk_id,
                    "element_id": element["element_id"],
                    "source_start": element.get("source_start"),
                    "source_end": element.get("source_end"),
                    "raw_candidates": raw_count,
                    "topic_admitted": len(extraction.evidence),
                    "evidence_ids": saved["evidence_ids"],
                    "new_evidence": saved["created"],
                    "reused_evidence": saved["reused"],
                    "claim_origin_omitted": saved["claim_origin_omitted"],
                    "structural_rejections": saved["invalid"],
                    "status": source_status,
                })
                if saved["invalid"]:
                    run["issues"].append({
                        "stage": "text_source_validation",
                        "chunk_id": chunk_id,
                        "element_id": element["element_id"],
                        "invalid_findings": saved["invalid"],
                        "status": "pending_source_retry",
                    })
                    # A failed candidate validation cannot certify source
                    # coverage. Valid findings are retained and deduplicated
                    # on the next attempt; this chunk remains retriable.
                    store.save(data)
                    continue

                # Preserve old candidates plus the new source findings.
                candidates = sorted(set(extraction.image_candidates)
                                    | cached_candidates[chunk_id])
                data.evidence_image_candidates[key] = candidates
                data.completed_evidence_scans[key] = "completed"
                text_coverage[chunk_id].update(c.claim_id for c in batch_missing)
                cached_candidates[chunk_id].update(candidates)
                mark_text_covered(
                    data, paper, args.model, chunk_id, batch_ids, candidates,
                )

                run["text_scans_completed"] += 1
                store.save(data)

            if stopped:
                break

            # PASS 2: Visually inspect relevant images.
            image_elements = suggested_images(
                paper,
                data,
                claim_batch,
                text_keys,
                args.image_mode,
                images,
                selected_source_ids=focused_ids,
            )
            if args.source_sections or args.source_elements:
                image_elements = [
                    element_id for element_id in image_elements
                    if paper.elements[element_id].section_id in args.source_sections
                    or element_id in args.source_elements
                ]

            for element_id in image_elements:
                if stopped:
                    break

                element = paper.elements[element_id]

                for asset_id, cell_id in images.asset_refs(
                    element_id
                ):
                    if stopped:
                        break

                    try:
                        prepared = images.prepare(
                            element_id,
                            asset_id,
                            cell_id,
                        )

                    except (
                        ValueError,
                        OSError,
                        ImportError,
                    ) as exc:
                        run["source_processing"].append({
                            "stage": "image_prepare", "element_id": element_id,
                            "asset_id": asset_id, "cell_id": cell_id,
                            "status": "pending_image_preparation_retry",
                        })
                        run["unavailable_images"].append({
                            "element_id": element_id,
                            "asset_id": asset_id,
                            "message": str(exc),
                        })
                        continue

                    key = image_key(
                        paper,
                        batch_ids,
                        prepared,
                        args.model,
                    )

                    planned_image_keys.append(key)
                    if args.rerun:
                        data.completed_evidence_scans.pop(key, None)
                        for cid in batch_ids:
                            data.completed_evidence_scans.pop(
                                image_key(paper, [cid], prepared, args.model), None
                            )

                    image_missing = (
                        list(claim_batch) if args.rerun else [
                            claim for claim in claim_batch
                            if claim.claim_id not in historical_image_claims(
                                paper, data, args.model, prepared,
                                image_batches, batch_ids,
                            )
                        ]
                    )
                    # Only exact historical source/Claim scan receipts authorize
                    # skipping this focus-specific image scan. A generic earlier
                    # observation is not proof that all requested values were read.
                    if not image_missing:
                        data.completed_evidence_scans[key] = "completed"
                        mark_image_covered(
                            data, paper, args.model, prepared, batch_ids,
                        )
                        run["source_processing"].append({
                            "stage": "image_scan", "scan_id": key,
                            "element_id": element_id, "asset_id": asset_id,
                            "status": "reused_completed_scan",
                        })
                        continue

                    source_item = {
                        "element_id": element_id,
                        "kind": element.kind,
                        "asset_id": asset_id,
                        "cell_id": cell_id,
                        "caption": element.caption,
                        "label": element.label,
                        "section_id": element.section_id,
                        "section_title": next(
                            (section.title for section in paper.sections
                             if section.section_id == element.section_id), ""
                        ),
                    }

                    try:
                        extraction, response = call_agent(
                            image_missing,
                            [source_item],
                            "image_scan",
                            image=prepared,
                        )
                        raw_image_count = len(extraction.evidence)
                        # Record a submitted image even if the later topic
                        # admission check fails and this scan needs a retry.
                        images.record_submission([prepared], args.model, response)
                        extraction = review_extraction(
                            image_missing, [source_item], extraction, "image_scan"
                        )

                    except StructuredOutputParseError as exc:
                        run["source_processing"].append({
                            "stage": "image_scan", "scan_id": key,
                            "element_id": element_id, "asset_id": asset_id,
                            "status": "pending_parse_retry",
                        })
                        run["issues"].append({
                            "stage": "image_scan", "element_id": element_id,
                            "asset_id": asset_id, "error": str(exc),
                            "status": "pending_parse_retry",
                        })
                        store.save(data)
                        continue
                    except BudgetStop as exc:
                        run["source_processing"].append({
                            "stage": "image_scan", "scan_id": key,
                            "element_id": element_id, "asset_id": asset_id,
                            "status": "pending_budget_retry",
                        })
                        run["issues"].append({
                            "stage": "image_scan", "element_id": element_id,
                            "asset_id": asset_id, "error": str(exc),
                        })
                        stopped = True
                        break
                    except (ValueError, RuntimeError) as exc:
                        run["source_processing"].append({
                            "stage": "image_scan", "scan_id": key,
                            "element_id": element_id, "asset_id": asset_id,
                            "status": "pending_review_retry",
                        })
                        run["issues"].append({
                            "stage": "image_topic_review", "element_id": element_id,
                            "asset_id": asset_id, "error": str(exc),
                            "status": "pending_review_retry",
                        })
                        store.save(data)
                        continue

                    receipt = {
                        **prepared.receipt(),
                        "model": getattr(
                            response, "model", args.model
                        ),
                        "response_id": getattr(
                            response, "id", None
                        ),
                    }

                    saved = save_findings(
                        extraction=extraction,
                        source_elements=[source_item],
                        claim_batch=image_missing,
                        paper=paper,
                        store=store,
                        data=data,
                        run=run,
                        run_id=run_id,
                        image_receipt=receipt,
                        scan_id=key,
                        scan_stage="image_scan",
                    )
                    newly_seen_evidence.update(saved["evidence_ids"])
                    run["source_processing"].append({
                        "stage": "image_scan", "scan_id": key,
                        "element_id": element_id, "asset_id": asset_id,
                        "cell_id": cell_id,
                        "raw_candidates": raw_image_count,
                        "topic_admitted": len(extraction.evidence),
                        "evidence_ids": saved["evidence_ids"],
                        "new_evidence": saved["created"],
                        "reused_evidence": saved["reused"],
                        "structural_rejections": saved["invalid"],
                        "status": (
                            "pending_source_retry" if saved["invalid"] else
                            "evidence_saved_or_reused" if saved["evidence_ids"] else
                            "no_related_findings_returned"
                        ),
                    })
                    if saved["invalid"]:
                        run["issues"].append({
                            "stage": "image_source_validation",
                            "element_id": element_id, "asset_id": asset_id,
                            "invalid_findings": saved["invalid"],
                            "status": "pending_source_retry",
                        })
                        store.save(data)
                        continue

                    data.completed_evidence_scans[key] = "completed"
                    mark_image_covered(
                        data, paper, args.model, prepared, batch_ids,
                    )

                    run["image_scans_completed"] += 1
                    store.save(data)

        # Only topic-admitted discoveries remain in the Evidence Pool.
        # Evidence admission checks subject alignment, not Claim support;
        # Mapping independently evaluates scientific relations.

        text_complete = (
            not stopped
            and all(
                data.completed_evidence_scans.get(key) == "completed"
                for key in planned_targeted_text_keys
            )
        )

        image_complete = (
            not stopped
            and not run["unavailable_images"]
            and all(
                data.completed_evidence_scans.get(key) == "completed"
                for key in planned_image_keys
            )
        )

        completed = text_complete and image_complete
        run["pending_text_source_windows"] = [
            item for item in planned_targeted_text_windows
            if data.completed_evidence_scans.get(item["scan_id"]) != "completed"
        ]
        run["pending_text_scan_keys"] = [
            item["scan_id"] for item in run["pending_text_source_windows"]
        ]
        run["pending_image_scan_keys"] = sorted(
            {key for key in planned_image_keys
             if data.completed_evidence_scans.get(key) != "completed"}
        )
        run["source_processing_summary"] = {
            "attempted_or_reused_source_windows": len(run["source_processing"]),
            "pending_text_scans": len(run["pending_text_scan_keys"]),
            "pending_image_scans": len(run["pending_image_scan_keys"]),
            "structural_rejections": run["rejected_findings"],
            "source_selection_completed": bool(run.get("source_selection")),
        }

        run["status"] = (
            "completed" if completed else "partial"
        )

        run["coverage"] = {
            "text_scan_complete": text_complete,
            "image_scan_complete": image_complete,
            "image_scope": {
                "none": "pixels_not_requested",
                "relevant": "selected_candidates_only",
                "all": "all_claim_relevant_image_assets",
            }[args.image_mode],
            "planned_text_scans": len(planned_targeted_text_keys),
            "source_index_screened": bool(run.get("source_selection")),
            "all_paper_chunks_extracted": False,
            "source_selection_complete": not any(
                issue.get("stage") == "claim_source_selection"
                for issue in run["issues"]
            ),
            "topic_review_policy": "candidate_subject_admission_v1",
            "planned_image_scans": len(planned_image_keys),
        }

        for claim in selected:
            data.evidence_search_by_claim[
                claim.claim_id
            ] = {
                "status": run["status"],
                "scan_cache_policy": SOURCE_SCAN_POLICY,
                "discovery_policy": DISCOVERY_POLICY,
                "paper_revision": paper.revision,
                "model": args.model,
                "search_scope": "selected_sources" if (args.source_sections or args.source_elements) else "whole_paper",
                "source_section_ids": sorted(args.source_sections),
                "source_element_ids": sorted(args.source_elements),
                "image_mode": args.image_mode,
                "text_scan_complete": text_complete,
                "image_scan_complete": image_complete,
                "run_id": run_id,
            }

    except BaseException:
        run["status"] = "interrupted"
        raise

    finally:
        run["finished_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        run["duration_seconds"] = round(
            perf_counter() - started, 3
        )

        run["llm_wall_time_seconds"] = round(
            run["llm_wall_time_seconds"], 3
        )

        run["api_cost_report"] = tracker.report()
        run["budget_report"] = budget.report()
        run["token_usage_report"] = task_token_report(run)
        print_task_tokens(run["token_usage_report"], run["budget_report"])
        run["image_report"] = images.report()

        store.save(data)
        budget.close()

    print("\n=== EVIDENCE ANALYSIS ===")
    print("Status:", run["status"])
    print("Evidence:", len(data.evidence_pool))
    print("Duration:", run["duration_seconds"], "seconds")
    print(
        "LLM time:",
        run["llm_wall_time_seconds"],
        "seconds",
    )
    print(
        "Estimated cost:",
        run["api_cost_report"].get(
            "estimated_total_usd"
        ),
        "USD",
    )
    print("Saved:", store.path)


if __name__ == "__main__":
    main()
