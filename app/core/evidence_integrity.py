"""Shared Evidence provenance, reuse, and relevance checks.

This module owns deterministic source checks, not scientific judgment.
It is used by the existing Evidence Agent and its linked runner.
"""
from __future__ import annotations

import re
from app.models.document_v2 import Paragraph, Table

LINKING_POLICY = "linked_source_reuse_v1"


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _table_facts(paper, record):
    """Generate a numeric description from the original table grid only."""
    if record.kind != "table" or record.observed_via != "structured_table":
        return None
    table = paper.elements.get(record.source_element_id)
    if not isinstance(table, Table) or not table.grid or not table.grid.cells:
        return None
    cells = table.grid.cells
    by_id = {c.cell_id: c for c in cells}
    at = {(c.row, c.col): c for c in cells}
    selector = record.source_anchor.selector if record.source_anchor else None
    ids = getattr(selector, "cell_ids", []) or []
    selected = [by_id[eid] for eid in ids if eid in by_id]
    if not selected:
        return None
    cols = sorted({c.col for c in selected if c.row > 0})
    rows = sorted({c.row for c in selected if c.row > 0})
    headers = {
        c.col: c.text.strip()
        for c in cells
        if c.row == 0
    }
    label = table.label or record.source_element_id
    if not rows:
        return f"{label} columns: " + ", ".join(
            f"{headers.get(c.col, f'Column {c.col}')}: {c.text}"
            for c in sorted(selected, key=lambda x: (x.row, x.col))
        )
    # Include the row label and method even when only numeric cells were cited.
    identifying = [col for col, name in headers.items()
                   if name.casefold().strip() in {"dataset", "method", "model", "algorithm"}]
    wanted = sorted(set(cols) | set(identifying))
    groups = record.source_content.get("row_group_context") or []
    statements = []
    for row in rows:
        group = next((g.get("dataset_label") for g in groups
                      if g.get("row_start", 10**9) <= row <= g.get("row_end", -1)), None)
        values = []
        for col in wanted:
            cell = at.get((row, col))
            if cell is not None and cell.text.strip():
                values.append(f"{headers.get(col, f'Column {col}')}={cell.text.strip()}")
        prefix = f"{label}, row {row}"
        if group:
            prefix += f", dataset={group}"
        if values:
            statements.append(prefix + ": " + "; ".join(values))
    return " | ".join(statements) or None


def normalize_table_descriptions(paper, data):
    """Correct old and new table descriptions without changing source anchors."""
    changed = 0
    for record in data.evidence_pool.values():
        fact_text = _table_facts(paper, record)
        if fact_text and record.description != fact_text:
            record.description = fact_text
            changed += 1
    return changed


def _within_scope(paper, evidence, sections, elements):
    if not sections and not elements:
        return True
    source = paper.elements.get(evidence.source_element_id)
    return source is not None and (
        evidence.source_element_id in elements or source.section_id in sections
    )


def candidates_for_scope(paper, data, sections, elements):
    """Return grounded candidates; never filter by a guessed scientific result."""
    return [record for record in data.evidence_pool.values()
            if record.source_status == "located"
            and _within_scope(paper, record, sections, elements)
            and not (record.kind == "paragraph" and not record.source_quote)]


def _selection_covers_paragraph(paper, segments, element_id):
    source = paper.elements.get(element_id)
    if not isinstance(source, Paragraph):
        return False
    intervals = sorted((s.get("start", -1), s.get("end", -1))
                       for s in segments if s.get("element_id") == element_id)
    stop = 0
    for begin, end in intervals:
        if not isinstance(begin, int) or not isinstance(end, int) or begin > stop:
            return False
        stop = max(stop, end)
    return stop >= len(source.text)


def completed_direct_sources(paper, data):
    """Source-level discovery coverage is distinct from Claim-link coverage."""
    found = set()
    for receipt in data.direct_evidence_searches.values():
        if (receipt.get("status") != "completed" or
                receipt.get("paper_revision") != paper.revision or
                receipt.get("text_scan_complete") is not True):
            continue
        if receipt.get("mode") == "selected_evidence":
            selection = receipt.get("selection") or {}
            found.update(eid for eid in selection.get("element_ids", [])
                         if eid in paper.elements)
            for eid in {s.get("element_id") for s in selection.get("segments", [])}:
                if eid and _selection_covers_paragraph(
                        paper, selection.get("segments", []), eid):
                    found.add(eid)
        else:
            sections = set(receipt.get("section_ids") or [])
            elements = set(receipt.get("element_ids") or [])
            if not sections and not elements:
                found.update(paper.elements)
            found.update(eid for eid in elements if eid in paper.elements)
            found.update(eid for eid, item in paper.elements.items()
                         if item.section_id in sections)
    return found


def _dataset_names(data):
    names = set()
    for record in data.evidence_pool.values():
        for group in record.source_content.get("row_group_context") or []:
            name = group.get("dataset_label")
            if isinstance(name, str) and name.strip():
                names.add(name.strip())
    return names


def _dataset_targets(claim_text, dataset_names):
    text = _compact(claim_text)
    matches = set()
    for name in dataset_names:
        aliases = {_compact(name)}
        words = name.split()
        if len(words) > 1:
            aliases.add(_compact(" ".join(words[1:])))
        if any(len(alias) >= 5 and alias in text for alias in aliases):
            matches.add(_compact(name))
    return matches


def validated_relation(paper, data, claim, evidence):
    """Reject clear source-scope mismatches, not weak scientific support."""
    if evidence.source_status != "located":
        return False
    claim_text = claim.source_quote or claim.model_quote
    targets = _dataset_targets(claim_text, _dataset_names(data))
    if targets:
        if evidence.kind == "table":
            groups = {_compact(x["dataset_label"])
                      for x in evidence.source_content.get("row_group_context") or []
                      if x.get("dataset_label")}
            if groups and not (targets & groups):
                return False
        elif evidence.kind == "paragraph":
            quoted = _compact(evidence.source_quote or "")
            if not any(t in quoted for t in targets):
                return False

    # A chart cannot directly document a named table-only metric unless that
    # metric was actually read from the chart or stated in its caption.
    if evidence.kind == "figure":
        known_metrics = set()
        for item in paper.elements.values():
            if isinstance(item, Table) and item.grid:
                known_metrics.update(
                    c.text.strip() for c in item.grid.cells
                    if c.row == 0 and re.fullmatch(r"[A-Z][A-Za-z0-9]{1,7}", c.text.strip())
                )
        named_metrics = [metric for metric in known_metrics
                         if re.search(rf"\b{re.escape(metric)}\b", claim_text)]
        if named_metrics:
            visible_text = " ".join(str(evidence.source_content.get(key, ""))
                                    for key in ("caption", "visual_observation"))
            if not any(re.search(rf"\b{re.escape(metric)}\b", visible_text)
                       for metric in named_metrics):
                return False
    return True


def candidate_view(record):
    """Compact, source-grounded model view of an already persisted Evidence."""
    source = record.source_content or {}
    out = {
        "evidence_id": record.evidence_id,
        "kind": record.kind,
        "source_element_id": record.source_element_id,
        "observed_via": record.observed_via,
        "description": record.description[:1400],
    }
    if record.source_quote:
        out["source_quote"] = record.source_quote[:1100]
    if record.kind == "table":
        out["dataset_groups"] = [x.get("dataset_label") for x in
                                 source.get("row_group_context") or []]
        out["cell_ids"] = (getattr(record.source_anchor.selector, "cell_ids", [])
                           if record.source_anchor else [])[:100]
    if record.observed_via == "image_input":
        out["visual_observation"] = str(source.get("visual_observation", ""))[:1300]
        out["visual_reading_verified"] = False
    return out
