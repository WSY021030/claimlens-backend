"""Deterministic table context for Reader-v3.

Original paper/table cells and Evidence anchors remain authoritative. This
module never creates Evidence, accepts model-generated values as ground truth,
or assigns a 'best' direction to an unfamiliar metric.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


# Metrics with a well-defined comparison direction in this project's tables.
# Unknown metrics are reported with their values but never ranked automatically.
KNOWN_DIRECTIONS = {"md": "min", "mse": "min", "aurc": "max"}


def _mentioned(question: str, label: str) -> bool:
    return bool(label) and bool(re.search(
        r"(?<!\w)" + re.escape(label.strip()) + r"(?!\w)", question,
        flags=re.IGNORECASE,
    ))


def _value(text):
    try:
        raw = str(text).strip().replace("−", "-")
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", raw):
            return None
        return Decimal(raw)
    except (InvalidOperation, TypeError):
        return None


def source_table_audit(paper, data, table_id: str, question: str):
    """Compare all located Evidence cells with the original standardized grid.

    Results are emitted only if the whole standardized grid matches the
    Evidence pool exactly. Row-group labels come from the original grid's
    deterministic repeated-method grouping, not from model prose.
    """
    table = getattr(paper, "elements", {}).get(table_id) if paper is not None else None
    grid = getattr(table, "grid", None)
    if (getattr(table, "kind", None) != "table"
            or getattr(table, "representation", None) != "structured"
            or grid is None or not grid.cells):
        return None

    original = {cell.cell_id: cell for cell in grid.cells}
    if len(original) != len(grid.cells):
        return {"table_id": table_id, "status": "unverified", "issues": ["Duplicate original Cell IDs"]}
    located = {}
    conflicts = []
    cell_sources = {}
    for eid, evidence in data.evidence_pool.items():
        if (evidence.kind != "table" or evidence.source_element_id != table_id
                or evidence.source_status != "located"):
            continue
        content = evidence.source_content or {}
        if not isinstance(content, dict):
            continue
        for cell in content.get("cells") or []:
            if not isinstance(cell, dict) or not cell.get("cell_id"):
                continue
            key = cell["cell_id"]
            observed = (cell.get("row"), cell.get("col"), str(cell.get("text", "")))
            if key in located and located[key] != observed:
                conflicts.append(key)
            located.setdefault(key, observed)
        anchor = evidence.source_anchor
        if anchor is None:
            continue
        selector = anchor.model_dump(mode="json")["selector"]
        if selector.get("type") == "table_cells":
            for cid in selector.get("cell_ids") or []:
                cell_sources.setdefault(cid, []).append(eid)

    missing = sorted(set(original) - set(located))
    extra = sorted(set(located) - set(original))
    mismatched = sorted(
        cid for cid in set(original) & set(located)
        if located[cid] != (original[cid].row, original[cid].col, original[cid].text)
    )
    if missing or extra or mismatched or conflicts:
        return {"table_id": table_id, "status": "unverified", "issues": [
            "Stored Evidence does not exactly cover the standardized source grid.",
        ], "missing_cell_count": len(missing), "extra_cell_count": len(extra),
            "mismatched_cell_count": len(mismatched),
            "conflicting_cell_count": len(set(conflicts))}

    cells = {(c.row, c.col): c for c in grid.cells}
    column_names = {c.col: c.text.strip() for c in grid.cells
                    if c.row == 0 and c.role == "column_header" and c.text.strip()}
    method_cols = [col for col, label in column_names.items()
                   if label.casefold() in {"method", "model", "algorithm", "approach"}]
    if len(method_cols) != 1:
        return {"table_id": table_id, "status": "verified_cells_only", "issues": [
            "Method column is not unambiguous; numeric ranking was skipped.",
        ], "cell_count": len(original)}
    method_col = method_cols[0]
    # Only numeric data columns can be ranked. Words such as "dataset" may
    # occur in the question but identify source labels, not numeric metrics.
    non_metric_labels = {"dataset", "data set", "dataset name", "group", "group name"}
    numeric_columns = {c.col for c in grid.cells
                       if c.row > 0 and _value(c.text) is not None}
    metrics = [(col, name, KNOWN_DIRECTIONS.get(name.casefold()))
               for col, name in sorted(column_names.items())
               if col != method_col and col in numeric_columns
               and name.casefold() not in non_metric_labels
               and _mentioned(question, name)]
    if not metrics:
        return {"table_id": table_id, "status": "verified_cells_only",
                "cell_count": len(original), "metric_checks": []}

    # The same table-row grouping logic is used by the existing Evidence Agent.
    from app.evidence_runner import table_row_groups
    body_rows = sorted({c.row for c in grid.cells if c.row > 0})
    row_groups, _ = table_row_groups(grid.cells, body_rows)
    groups = [g for g in row_groups if g.get("basis") == "repeated_method_cycle"
              and g.get("dataset_label") and g.get("rows")]
    if not groups or set(row for g in groups for row in g["rows"]) != set(body_rows):
        return {"table_id": table_id, "status": "verified_cells_only", "cell_count": len(original),
                "issues": ["Dataset grouping is not fully source-grounded; ranking skipped."],
                "metric_checks": []}

    checks = []
    for col, metric, direction in metrics:
        dataset_checks = []
        valid = True
        for group in groups:
            rows = []
            for row in group["rows"]:
                method = cells.get((row, method_col))
                value_cell = cells.get((row, col))
                if (method is None or value_cell is None
                        or not method.text.strip() or _value(value_cell.text) is None):
                    valid = False
                    break
                ids = cell_sources.get(value_cell.cell_id) or []
                rows.append({"method": method.text.strip(), "value": value_cell.text,
                             "cell_id": value_cell.cell_id, "row": row,
                             "evidence_ids": ids[:3]})
            if not valid:
                break
            synthetic = [row for row in rows if row["method"].casefold() not in {
                "original", "original data", "real", "real data"}]
            if not synthetic:
                valid = False
                break
            best = None
            if direction:
                extreme = (min if direction == "min" else max)(
                    _value(row["value"]) for row in synthetic)
                best = [row["method"] for row in synthetic if _value(row["value"]) == extreme]
            dataset_checks.append({
                "dataset": group["dataset_label"],
                "rows": rows,
                "best_synthetic_methods": best,
                "best_synthetic_value": next(
                    (r["value"] for r in synthetic if best and r["method"] in best), None),
                "tie_for_best": bool(best and len(best) > 1),
            })
        checks.append({"metric": metric, "direction": direction or "unranked",
                       "status": "verified" if valid else "unranked_incomplete",
                       "datasets": dataset_checks if valid else []})
    return {"table_id": table_id, "table_label": table.label, "status": "verified_complete",
            "cell_count": len(original), "source_revision": data.paper_revision,
            "metric_checks": checks,
            "comparison_scope": "Best excludes Original/Real rows; their values remain visible in rows.",
            "source_note": "These are deterministic calculations over exactly matching original standardized cells."
            " A comparison direction is used only for known metrics."}


def compact_audits_for_model(audits):
    """A small, exact audit view for prompts; source cells remain available.

    Keep the full audit only for deterministic rendering. Repeating every
    original metric value, Cell ID and Evidence ID in an audit alongside the
    already-shared source table wastes context and can obscure counterexamples.
    """
    result = []
    for audit in audits:
        view = {key: audit[key] for key in (
            "table_id", "table_label", "status", "cell_count", "source_revision",
            "comparison_scope", "issues", "source_note",
        ) if key in audit}
        checks = []
        for metric in audit.get("metric_checks") or []:
            item = {key: metric[key] for key in (
                "metric", "direction", "status",
            ) if key in metric}
            item["datasets"] = []
            for group in metric.get("datasets") or []:
                rows = group.get("rows") or []
                best = set(group.get("best_synthetic_methods") or [])
                selected = [row for row in rows if row.get("method") in best]
                ties = {}
                for row in rows:
                    if str(row.get("method", "")).casefold() in {
                        "original", "original data", "real", "real data",
                    }:
                        continue
                    value = _value(row.get("value"))
                    if value is not None:
                        ties.setdefault(value, []).append(row)
                all_ties = [
                    {"value": members[0]["value"],
                     "methods": [row["method"] for row in members],
                     "source_cells": [row["cell_id"] for row in members]}
                    for members in ties.values() if len(members) > 1
                ]
                item["datasets"].append({
                    "dataset": group.get("dataset"),
                    "best_synthetic_methods": group.get("best_synthetic_methods"),
                    "best_synthetic_value": group.get("best_synthetic_value"),
                    "tie_for_best": group.get("tie_for_best"),
                    "best_source_cells": [row["cell_id"] for row in selected],
                    "all_synthetic_ties": all_ties,
                })
            checks.append(item)
        view["metric_checks"] = checks
        view["audit_note"] = (
            "Best methods and ties are calculated from all source-matched cells. "
            "Full row values and original Cell IDs remain in shared_source_content. "
            "This is not a replacement for reading the original table grid."
        )
        result.append(view)
    return result


def audits_for_packet(paper, data, question: str, evidence_ids):
    """Return audits for tables actually represented among selected Evidence."""
    ids = sorted({data.evidence_pool[eid].source_element_id for eid in evidence_ids
                  if eid in data.evidence_pool and data.evidence_pool[eid].kind == "table"})
    return [audit for table_id in ids
            if (audit := source_table_audit(paper, data, table_id, question)) is not None]


def render_verified_numeric_checks(audits, *, language="en"):
    """Exact comparisons; always include every synthetic method for context."""
    lines = []
    for audit in audits or []:
        if audit.get("status") != "verified_complete":
            continue
        for metric in audit.get("metric_checks") or []:
            if metric.get("status") != "verified" or metric.get("direction") == "unranked":
                continue
            if not lines:
                lines.append("\n### 原表数值核对（最优排名只比较合成方法）" if language == "zh"
                             else "\n### Verified numerical comparisons (synthetic methods)")
            lines.append("\n**" + str(audit.get("table_label") or audit["table_id"]) +
                         " — " + metric["metric"] + "**")
            lines.append("| 数据集 | 最优合成方法 | 数值 | 全部合成方法的原始数值 | 最优值来源单元格 |" if language == "zh"
                         else "| Dataset | Best synthetic method(s) | Best value | All synthetic method values | Best-value source cells |")
            lines.append("|---|---|---:|---|---|")
            for item in metric["datasets"]:
                methods = ", ".join(item.get("best_synthetic_methods") or [])
                values = "; ".join(
                    row["method"] + " " + str(row["value"])
                    for row in item["rows"]
                    if row["method"].casefold() not in {
                        "original", "original data", "real", "real data"})
                sources = ", ".join(
                    row["cell_id"] for row in item["rows"]
                    if row["method"] in set(item.get("best_synthetic_methods") or []))
                safe = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {safe(item['dataset'])} | {safe(methods)} | "
                             f"{safe(item.get('best_synthetic_value') or '?')} | "
                             f"{safe(values)} | {safe(sources)} |")
    if lines:
        lines.append("\n" + (
            "以上数值与标准化原表单元格逐项匹配。Original/Real 行未参与合成方法排名。"
            if language == "zh" else
            "Values were matched to the standardized original cells. Original/Real "
            "rows are excluded from synthetic-method rankings."))
    return "\n".join(lines)
