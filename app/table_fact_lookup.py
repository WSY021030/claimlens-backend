"""Ground exact table-value lookups in original cells before answer generation.

This optional, read-only path complements the existing Evidence Agent. It is used
only when dataset, method, and metric are all unambiguously identified in a
structured table and already-located Evidence covers every requested cell.
If any precondition fails, the ordinary Coordinator/Evidence flow is unchanged.
"""

from __future__ import annotations

import re
import unicodedata

from app.evidence_runner import table_row_groups


_DATASET_HEADERS = {"dataset", "data set", "data source", "corpus", "benchmark",
                    "scenario", "group", "category", "cohort"}
_METHOD_HEADERS = {"method", "model", "algorithm", "approach", "system",
                   "technique", "baseline"}
_NOT_PLAIN_FACT = re.compile(
    r"\b(why|how|explain|interpret|reason|significance|implication|"
    r"prove|justify|compare|comparison|best|optimal|highest|lowest|"
    r"trend|causality|causal)\b", re.IGNORECASE,
)
_NUMBER = re.compile(
    r"^\s*[+\-−]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?\s*%?\s*$"
)


def _normalize(value):
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.findall(r"[^\W_]+", value, flags=re.UNICODE))


def _mentions(text, phrase):
    phrase = _normalize(phrase)
    return bool(phrase) and f" {phrase} " in f" {_normalize(text)} "


def _headers(cells):
    result = {}
    for cell in cells:
        if cell.row == 0 or cell.role == "column_header":
            label = str(cell.text or "").strip()
            if label:
                result[cell.col] = label
    return result


def _column(headers, names):
    return next((col for col, label in sorted(headers.items())
                 if _normalize(label) in names), None)


def _metric_aliases(caption):
    """Read long-form metric names from a table caption, never from assumptions."""
    aliases = {}
    for match in re.finditer(
        r"([A-Za-z][A-Za-z’'\-]*(?:\s+[A-Za-z][A-Za-z’'\-]*){0,5})"
        r"\s*\(([A-Z][A-Z0-9]{1,5})\)", str(caption or ""),
    ):
        words = match.group(1).split()
        acronym = match.group(2)
        aliases.setdefault(acronym, set()).add(acronym)
        for width in range(2, len(words) + 1):
            aliases[acronym].add(" ".join(words[-width:]))
    return aliases


def _table_lookup(table, question):
    grid = getattr(table, "grid", None)
    cells = list(grid.cells) if grid is not None and grid.cells else []
    if not cells:
        return None
    by_pos = {(cell.row, cell.col): cell for cell in cells}
    headers = _headers(cells)
    dataset_col = _column(headers, _DATASET_HEADERS)
    method_col = _column(headers, _METHOD_HEADERS)
    if dataset_col is None or method_col is None:
        return None
    header_ids = {cell.cell_id for cell in cells
                  if cell.row == 0 or cell.role == "column_header"}
    body_rows = sorted({cell.row for cell in cells if cell.cell_id not in header_ids})
    groups, _ = table_row_groups(cells, body_rows)
    # Sequential fallback has no verified row-to-dataset association.
    groups = [group for group in groups
              if group["basis"] == "repeated_method_cycle" and group["dataset_label"]]
    if not groups:
        return None

    all_names = sorted({_normalize(group["dataset_label"]) for group in groups},
                       key=len, reverse=True)
    matches = [group for group in groups if _mentions(question, group["dataset_label"])]
    # A shorter label mentioned only as part of a longer one is not a second request.
    matches = [group for group in matches if not any(
        _normalize(group["dataset_label"]) != name
        and f" {_normalize(group['dataset_label'])} " in f" {name} "
        and _mentions(question, name)
        for name in all_names
    )]
    if not matches or len({_normalize(g["dataset_label"]) for g in matches}) != len(matches):
        return None

    metric_aliases = _metric_aliases(getattr(table, "caption", ""))
    metrics = [(col, label) for col, label in sorted(headers.items())
               if col not in (dataset_col, method_col)
               and (_mentions(question, label) or any(
                   _mentions(question, alias)
                   for alias in metric_aliases.get(label.strip(), ())
               ))]
    if not metrics:
        return None

    # Identify method names only from the dataset groups, not from unrestricted
    # model output or unrelated rows of another table.
    names = list(dict.fromkeys(
        str(by_pos[(row, method_col)].text).strip()
        for group in matches for row in group["rows"]
        if (row, method_col) in by_pos and by_pos[(row, method_col)].text.strip()
    ))
    methods = [name for name in names if _mentions(question, name)]
    if not methods:
        return None

    results = []
    required_ids = set()
    for group in matches:
        row_by_method = {}
        for row in group["rows"]:
            cell = by_pos.get((row, method_col))
            if cell is None or not str(cell.text).strip():
                continue
            row_by_method.setdefault(_normalize(cell.text), []).append(row)
        entries = []
        for name in methods:
            rows = row_by_method.get(_normalize(name), [])
            # Duplicate method rows need scientific context; do not guess.
            if len(rows) != 1:
                return None
            row = rows[0]
            values = []
            for col, metric in metrics:
                cell = by_pos.get((row, col))
                if cell is None or not _NUMBER.fullmatch(str(cell.text or "")):
                    return None
                values.append({"metric": metric, "value": str(cell.text).strip(),
                               "cell_id": cell.cell_id, "row": row, "col": col})
                required_ids.add(cell.cell_id)
            entries.append({"method": name, "row": row, "values": values})
        results.append({"dataset": group["dataset_label"], "row_start": group["rows"][0],
                        "row_end": group["rows"][-1], "rows": group["rows"],
                        "entries": entries})

    return {"table_id": table.element_id, "table_label": str(table.label or "Table"),
            "metrics": [name for _, name in metrics], "methods": methods,
            "datasets": results, "required_cell_ids": sorted(required_ids)}


def resolve_exact_table_lookup(paper, question, table_ids):
    """Return exact source cells or None; unsupported questions use normal agents."""
    if _NOT_PLAIN_FACT.search(question):
        return None
    tables = [paper.elements[eid] for eid in table_ids
              if eid in paper.elements and paper.elements[eid].kind == "table"]
    if not tables:
        return None
    # Avoid silently choosing between multiple unrelated requested tables.
    named = [table for table in tables if _mentions(question, table.label)]
    if len(named) == 1:
        tables = named
    if len(tables) != 1:
        return None
    return _table_lookup(tables[0], question)


def covering_evidence_ids(data, paper, lookup):
    """Choose records that actually cover all requested source cells.

    Stale model-written descriptions are intentionally ignored. Evidence IDs
    that mix cells from another dataset group are not used for this precise
    answer; those records remain available to the ordinary workflow.
    """
    table = paper.elements[lookup["table_id"]]
    by_id = {cell.cell_id: cell for cell in table.grid.cells}
    header_ids = {cell.cell_id for cell in table.grid.cells
                  if cell.row == 0 or cell.role == "column_header"}
    allowed_rows = {row for group in lookup["datasets"] for row in group["rows"]}
    required = set(lookup["required_cell_ids"])
    candidates = []
    for eid, record in data.evidence_pool.items():
        if record.kind != "table" or record.source_element_id != lookup["table_id"]:
            continue
        if record.source_status != "located" or record.source_anchor is None:
            continue
        anchor = record.source_anchor.model_dump(mode="json")
        selector = anchor.get("selector") or {}
        if selector.get("type") != "table_cells":
            continue
        cell_ids = set(selector.get("cell_ids") or [])
        relevant = cell_ids & required
        if not relevant:
            continue
        # A label in narrative description does NOT establish table membership.
        # Source row and cell IDs, plus full original-table row grouping, do.
        if any(by_id[cid].row not in allowed_rows for cid in cell_ids
               if cid in by_id and cid not in header_ids):
            continue
        candidates.append((eid, relevant, len(cell_ids)))

    chosen = []
    uncovered = set(required)
    while uncovered:
        options = [(eid, ids & uncovered, count) for eid, ids, count in candidates
                   if ids & uncovered and eid not in chosen]
        if not options:
            return []
        eid, newly, _ = max(options, key=lambda part: (len(part[1]), -part[2]))
        chosen.append(eid)
        uncovered -= newly
    return chosen


def render_exact_table_answer(lookup, evidence_ids, language="en"):
    """Render only validated values, preserving the cell coordinates verbatim."""
    output = []
    for group in lookup["datasets"]:
        if language == "zh":
            output.append(f"{lookup['table_label']} 中 {group['dataset']} 数据集的数值：")
        else:
            output.append(f"{group['dataset']} dataset values from {lookup['table_label']}:")
        metrics = lookup["metrics"]
        output += ["| Method | " + " | ".join(metrics) + " |",
                   "| --- | " + " | ".join("---:" for _ in metrics) + " |"]
        for entry in group["entries"]:
            output.append("| " + entry["method"] + " | " +
                          " | ".join(item["value"] for item in entry["values"]) + " |")
        output.append("")
    ids = ", ".join(f"`{cid}`" for cid in lookup["required_cell_ids"])
    sources = ", ".join(f"`{eid}`" for eid in evidence_ids)
    if language == "zh":
        output.append(f"来源：{lookup['table_label']} (`{lookup['table_id']}`)；原始单元格：{ids}；Evidence：{sources}。")
    else:
        output.append(f"Source: {lookup['table_label']} (`{lookup['table_id']}`); original cells: {ids}; Evidence: {sources}.")
    return "\n".join(output)


# TOKEN_SCOPE_OPT_20261110: conservative, model-free numeric comparison.
_COMPARE = re.compile(
    r"\b(compare|comparison|vs|versus|lowest|highest|best|"
    r"outperform|better|worse|ranking)\b|"
    r"比较|对比|相比|最低|最高|最优|优于|低于|高于|并列",
    re.I,
)

_REASON = re.compile(
    r"\b(why|causal|causality|mechanism|hypothesis|assumption|"
    r"privacy|ablation|theoretical|prove|justify|scientific support)\b|"
    r"为什么|因果|机制|假设|隐私|消融|理论依据",
    re.I,
)


def verified_numeric_comparison(
    paper, data, question, allowed_table_ids=None
):
    """Use one completely source-verified table group or return None."""
    if not _COMPARE.search(question) or _REASON.search(question):
        return None

    from app.core.table_context import source_table_audit

    results = []

    for table in paper.elements.values():
        if (
            table.kind != "table"
            or table.representation != "structured"
            or not table.grid
            or not table.grid.cells
            or (
                allowed_table_ids is not None
                and table.element_id not in allowed_table_ids
            )
        ):
            continue

        selection = _table_lookup(table, question)

        if (
            selection is None
            or len(selection["datasets"]) != 1
            or len(selection["metrics"]) != 1
            or len(selection["methods"]) != 2
        ):
            continue

        audit = source_table_audit(
            paper, data, table.element_id, question
        )

        if not audit or audit["status"] != "verified_complete":
            continue

        metric = next(
            (
                item
                for item in audit["metric_checks"]
                if (
                    _normalize(item["metric"])
                    == _normalize(selection["metrics"][0])
                    and item["status"] == "verified"
                    and item["direction"] in {"min", "max"}
                )
            ),
            None,
        )

        if metric is None:
            continue

        dataset = selection["datasets"][0]

        group = next(
            (
                item
                for item in metric["datasets"]
                if (
                    _normalize(item["dataset"])
                    == _normalize(dataset["dataset"])
                )
            ),
            None,
        )

        if group is None:
            continue

        methods = {
            _normalize(name)
            for name in selection["methods"]
        }

        selected_rows = [
            row
            for row in group["rows"]
            if _normalize(row["method"]) in methods
        ]

        if len(selected_rows) != 2:
            continue

        found = {
            "kind": "numeric_comparison",
            "table_id": table.element_id,
            "table_label": table.label or table.element_id,
            "dataset": group["dataset"],
            "metric": metric["metric"],
            "direction": metric["direction"],
            "rows": group["rows"],
            "requested_methods": selection["methods"],
            "best_synthetic_methods": group["best_synthetic_methods"],
            "best_synthetic_value": group["best_synthetic_value"],
            "required_cell_ids": [
                row["cell_id"]
                for row in group["rows"]
            ],
            "datasets": [{
                "dataset": dataset["dataset"],
                "rows": dataset["rows"],
            }],
        }

        refs = covering_evidence_ids(data, paper, found)

        if refs:
            found["evidence_ids"] = refs
            results.append(found)

    return results[0] if len(results) == 1 else None


def render_numeric_comparison(lookup, language="zh"):
    """Distinguish pairwise and global source-table rankings."""
    from decimal import Decimal

    rows = lookup["rows"]
    requested = {
        _normalize(value)
        for value in lookup["requested_methods"]
    }

    pair = [
        row
        for row in rows
        if _normalize(row["method"]) in requested
    ]

    if len(pair) != 2:
        raise ValueError(
            "Two distinct compared methods are required."
        )

    a, b = pair
    va, vb = Decimal(a["value"]), Decimal(b["value"])

    first_wins = (
        va < vb
        if lookup["direction"] == "min"
        else va > vb
    )

    if va == vb:
        pairwise = (
            f"{a['method']} 与 {b['method']} 在该指标上并列。"
            if language == "zh"
            else (
                f"{a['method']} and {b['method']} "
                "are tied on this metric."
            )
        )
    else:
        winner, loser = (
            (a, b)
            if first_wins
            else (b, a)
        )

        pairwise = (
            f"两者比较：{winner['method']} "
            f"({winner['value']}) 优于 "
            f"{loser['method']} ({loser['value']})。"
            if language == "zh"
            else (
                f"Pairwise: {winner['method']} "
                f"({winner['value']}) outperforms "
                f"{loser['method']} ({loser['value']})."
            )
        )

    intro = (
        f"{lookup['table_label']}，{lookup['dataset']} "
        f"数据集，{lookup['metric']} 原表数值："
        if language == "zh"
        else (
            f"Original {lookup['metric']} values for "
            f"{lookup['dataset']} in {lookup['table_label']}:"
        )
    )

    lines = [
        intro,
        "",
        f"| Method | {lookup['metric']} | Cell ID |",
        "|---|---:|---|",
    ]

    lines.extend(
        f"| {row['method']} | {row['value']} | "
        f"`{row['cell_id']}` |"
        for row in rows
    )

    leaders = ", ".join(
        lookup["best_synthetic_methods"]
    )

    values = [
        Decimal(row["value"])
        for row in rows
    ]

    best_overall = (
        min(values)
        if lookup["direction"] == "min"
        else max(values)
    )

    overall_names = ", ".join(
        row["method"]
        for row in rows
        if Decimal(row["value"]) == best_overall
    )

    if language == "zh":
        lines.extend([
            "",
            pairwise,
            (
                "包括 Original/Real 参考行在内的全部表格行，"
                f"最优的是 {overall_names}（{best_overall}）。"
            ),
            (
                f"仅比较合成方法时，最优的是 {leaders}"
                f"（{lookup['best_synthetic_value']}）。"
            ),
            (
                "两种方法的相对比较成立，不代表其中一方"
                "优于表中所有方法。"
            ),
        ])
    else:
        lines.extend([
            "",
            pairwise,
            (
                "Best including Original/Real reference rows: "
                f"{overall_names} ({best_overall})."
            ),
            (
                "Best among synthetic methods only: "
                f"{leaders} ({lookup['best_synthetic_value']})."
            ),
            (
                "A pairwise advantage does not establish "
                "the best result among all methods."
            ),
        ])

    lines.append(
        ("来源" if language == "zh" else "Sources")
        + ": "
        + lookup["table_id"]
        + "; "
        + ", ".join(lookup["evidence_ids"])
    )

    return "\n".join(lines)
