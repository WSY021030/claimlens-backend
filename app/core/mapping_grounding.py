"""Source preparation and numeric grounding for per-Claim Mapping.

No network calls. Source completion copies only already relevant, same-revision
structured tables. Numeric validation checks provenance and arithmetic, not the
truth of every scientific interpretation.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re

POLICY = "mapping_grounding_v1_1_group_labels"
TOKEN = re.compile(r"\{\{(cell|check):([^{}]+)\}\}")
DECIMAL_LITERAL = re.compile(r"(?<![\w.])[+-]?(?:\d+\.\d+|\.\d+|\d+[eE][+-]?\d+)(?![\w.])")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def number(value):
    raw = str(value).strip().replace("\u2212", "-")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", raw):
        return None
    try:
        result = Decimal(raw)
        return result if result.is_finite() else None
    except InvalidOperation:
        return None


def source_groups(cells):
    """Infer groups only from explicit headers AND identical method cycles.

    Joined dataset fragments remain a documented structural interpretation.
    Complex/merged layouts are retained but do not receive inferred identities.
    """
    at = {(c["row"], c["col"]): c for c in cells}
    headers = {c["col"]: c["text"].strip() for c in cells if c["row"] == 0}
    body = sorted({c["row"] for c in cells if c["row"] > 0})
    method_cols = [col for col, label in headers.items()
                   if label.casefold() in {"method", "model", "algorithm", "approach", "system"}]
    group_cols = [col for col, label in headers.items()
                  if label.casefold() in {"dataset", "data set", "benchmark", "corpus", "scenario", "cohort", "group"}]
    if (not body or len(method_cols) != 1 or len(group_cols) != 1
            or any(c.get("rowspan", 1) != 1 or c.get("colspan", 1) != 1 for c in cells)):
        return []
    mc, gc = method_cols[0], group_cols[0]
    def method(row):
        raw = at.get((row, mc), {}).get("text", "").strip()
        # Comparison key only. Preserve every original label, value and cell ID.
        # Keep digits, plus signs and other meaningful symbols.
        return re.sub(r"[\s\u2010-\u2015-]+", "", raw.casefold())
    label = lambda row: at.get((row, gc), {}).get("text", "").strip()
    first = method(body[0])
    starts = [r for r in body if first and method(r) == first and label(r)]
    if not starts or starts[0] != body[0]:
        return []
    groups, cycles = [], []
    for i, start in enumerate(starts):
        stop = starts[i + 1] if i + 1 < len(starts) else body[-1] + 1
        rows = [r for r in body if start <= r < stop]
        cycle = [method(r) for r in rows]
        if not all(cycle) or len(set(cycle)) != len(cycle):
            return []
        fragments = [at[(r, gc)] for r in rows if label(r)]
        # Do not concatenate repeated complete dataset names.
        labels = list(dict.fromkeys(c["text"].strip() for c in fragments))
        groups.append({"dataset_label": " ".join(labels), "row_start": rows[0],
                       "row_end": rows[-1], "rows": rows, "method_col": mc,
                       "label_cell_ids": [c["cell_id"] for c in fragments],
                       "basis": "validated_repeated_method_cycle"})
        cycles.append(cycle)
    if len(cycles) > 1 and any(cycle != cycles[0] for cycle in cycles[1:]):
        return []
    return groups


def prepare_tables(data, paper, claim_ids, run_id, allowed_element_ids=None):
    """Register complete comparison groups before Mapping, never inside an LLM.

    Existing records and scan receipts are preserved. Preview callers use their
    own in-memory document and never save it. IDs never refill deleted slots.
    """
    if paper is None:
        return []
    if (paper.paper_id, paper.revision) != (data.paper_id, data.paper_revision):
        raise ValueError("Paper revision mismatch during table preparation.")
    from app.core.evidence_retrieval import EvidenceDataManager
    from app.models.document_v2 import SourceAnchor, TableCellsSelector
    from app.models.reader_v3 import EvidenceRecord
    manager = EvidenceDataManager(data, paper)
    table_ids = set()
    for cid in claim_ids:
        for source in manager.references.get(data.claim_pool[cid].source_element_id, ()):
            if getattr(paper.elements.get(source), "kind", None) == "table":
                table_ids.add(source)
        # Reuse the established local relevance rules; do not scan arbitrary tables.
        ranked = manager._rank(cid, allowed_element_ids)
        peak = ranked[0]["score"] if ranked else 0
        for item in ranked:
            record = manager.records[item["id"]]
            if record.kind == "table" and (item["direct_reference"] or item["score"] >= peak * .35):
                table_ids.add(record.source_element_id)
    if allowed_element_ids is not None:
        table_ids.intersection_update(allowed_element_ids)
    created = []
    for table_id in sorted(table_ids):
        table = paper.elements.get(table_id)
        if (getattr(table, "kind", None) != "table" or table.representation != "structured"
                or not table.grid or not table.grid.cells):
            continue
        cells = [c.model_dump(mode="json") for c in table.grid.cells]
        if len({c["cell_id"] for c in cells}) != len(cells):
            raise ValueError(f"Duplicate source cell IDs: {table_id}")
        groups = source_groups(cells)
        # Irregular tables retain exact cells without fabricated row identities.
        scopes = groups or [{"rows": sorted({c["row"] for c in cells}),
                             "basis": "raw_table_without_inferred_groups"}]
        requested = []
        for group in scopes:
            label = re.sub(r"\W", "", group.get("dataset_label", "").casefold())
            for cid in claim_ids:
                text = data.claim_pool[cid].source_quote or data.claim_pool[cid].model_quote
                compact = re.sub(r"\W", "", text.casefold())
                named = [g for g in groups if len(re.sub(r"\W", "", g.get("dataset_label", ""))) >= 4
                         and re.sub(r"\W", "", g["dataset_label"].casefold()) in compact]
                broad = bool(re.search(r"\b(all|every|across|datasets|overall)\b|全部|所有|各数据集", text, re.I))
                if broad or not named or (label and label in compact):
                    requested.append(group)
                    break
        for group in requested:
            rows = set(group["rows"])
            chosen = [c for c in cells if c["row"] in rows or c["row"] == 0
                      or c.get("role") == "column_header"]
            key = digest({"policy": POLICY, "revision": data.paper_revision,
                          "table": table_id, "cells": chosen, "group": group})
            if any(e.source_content.get("mapping_scope_key") == key for e in data.evidence_pool.values()):
                continue
            used = [int(e.rsplit("_", 1)[-1]) for e in data.evidence_pool
                    if re.fullmatch(r"evidence_\d+", e)]
            index = max([data.id_counters.get("evidence", 0), *used], default=0) + 1
            eid = f"evidence_{index:06d}"
            data.id_counters["evidence"] = index
            content = {"label": table.label, "caption": table.caption,
                       "representation": table.representation,
                       "recognition_status": table.recognition_status,
                       "cells": chosen, "row_group_context": [group] if groups else [],
                       "mapping_scope_key": key, "mapping_scope_policy": POLICY,
                       "source_revision": data.paper_revision,
                       "scope_note": "All methods in these original rows; completeness is not support."}
            data.evidence_pool[eid] = EvidenceRecord(
                evidence_id=eid, kind="table", source_element_id=table_id,
                source_anchor=SourceAnchor(element_id=table_id,
                    selector=TableCellsSelector(cell_ids=[c["cell_id"] for c in chosen])),
                source_status="located", source_note="mapping_complete_table_scope",
                source_content=content, description="Deterministic copy of a relevant source table scope.",
                observed_via="structured_table", created_run_id=run_id)
            created.append(eid)
    return created


def redundant_table_ids(records):
    coverage = {}
    for eid, rec in records.items():
        if rec.source_content.get("mapping_scope_policy") == POLICY:
            coverage.setdefault(rec.source_element_id, set()).update(
                c["cell_id"] for c in rec.source_content.get("cells", []))
    return {eid for eid, rec in records.items()
            if rec.kind == "table" and rec.observed_via == "structured_table"
            and rec.source_content.get("mapping_scope_policy") != POLICY
            and rec.source_content.get("cells")
            and {c["cell_id"] for c in rec.source_content["cells"]} <= coverage.get(rec.source_element_id, set())}


def compact_packet(payload):
    """Lossless table-cell sharing; only derived prompt representation changes."""
    result = deepcopy(payload)
    shared = {}
    for item in result["evidence_items"]:
        if item.get("kind") != "table" or not item.get("cells"):
            continue
        table_id = item["source_element_id"]
        fields = item["cell_fields"]
        cells = [dict(zip(fields, c)) for c in item["cells"]]
        table = shared.setdefault(table_id, {"label": item.get("label"),
            "caption": item.get("caption", ""), "by_id": {}, "groups": {}})
        for cell in cells:
            old = table["by_id"].get(cell["cell_id"])
            if old is not None and old != cell:
                raise ValueError(f"Conflicting source cell: {cell['cell_id']}")
            table["by_id"][cell["cell_id"]] = cell
        for group in item.get("row_group_context", []):
            table["groups"][digest(group)] = group
        item["table_ref"] = table_id
        item["cell_ids"] = [c["cell_id"] for c in cells]
        item.pop("cells", None)
        item.pop("cell_fields", None)
        item.pop("row_group_context", None)
        if item.get("caption"):
            item["caption_ref"] = table_id
        item.pop("caption", None)
        item.pop("source_anchor", None)
    for table_id, table in shared.items():
        cells = sorted(table.pop("by_id").values(), key=lambda c: (c["row"], c["col"]))
        regular = all(c["cell_id"] == f"{table_id}.R{c['row']}C{c['col']}"
                      and c.get("rowspan", 1) == c.get("colspan", 1) == 1 for c in cells)
        table["groups"] = list(table["groups"].values())
        if regular:
            table["cell_id_rule"] = table_id + ".R{row}C{col}"
            table["cell_fields"] = ["row", "col", "text", "role"]
            table["cells"] = [[c["row"], c["col"], c["text"], c.get("role")] for c in cells]
            lookup = {c["cell_id"]: (c["row"], c["col"]) for c in cells}
            for item in result["evidence_items"]:
                if item.get("table_ref") == table_id:
                    item["cell_positions"] = [list(lookup[c]) for c in item.pop("cell_ids")]
        else:
            table["cell_fields"] = ["cell_id", "row", "col", "text", "role", "rowspan", "colspan"]
            table["cells"] = [[c.get(k) for k in table["cell_fields"]] for c in cells]
    result["shared_tables"] = shared
    result["grounding_policy"] = POLICY
    return result


def table_catalog(packet):
    """Recover exact visible cell identities; never read outside the packet."""
    catalog, ownership = {}, {}
    for table_id, table in packet.get("shared_tables", {}).items():
        records = [dict(zip(table["cell_fields"], cell)) for cell in table["cells"]]
        for cell in records:
            cell.setdefault("cell_id", table.get("cell_id_rule", "").format(row=cell["row"], col=cell["col"]))
        at = {(c["row"], c["col"]): c for c in records}
        headers = {c["col"]: c["text"].strip() for c in records if c["row"] == 0}
        methods = [col for col, text in headers.items() if text.casefold() in {
            "method", "model", "algorithm", "approach", "system"}]
        for cell in records:
            group = next((g for g in table["groups"] if g.get("basis") == "validated_repeated_method_cycle"
                          and cell["row"] in g.get("rows", [])), None)
            mc = methods[0] if len(methods) == 1 else None
            cell = {**cell, "table_id": table_id, "metric": headers.get(cell["col"], ""),
                    "method": at.get((cell["row"], mc), {}).get("text", ""),
                    "group": group, "number": number(cell["text"])}
            catalog[cell["cell_id"]] = cell
    for item in packet["evidence_items"]:
        tid = item.get("table_ref")
        if not tid:
            continue
        table = packet["shared_tables"][tid]
        ids = item.get("cell_ids")
        if ids is None:
            ids = [table["cell_id_rule"].format(row=r, col=c) for r, c in item.get("cell_positions", [])]
        ownership[item["evidence_id"]] = set(ids)
    return catalog, ownership


def numeric_check(check, catalog, ownership):
    """Check one complete group, with an explicitly declared comparison set."""
    eid = check["evidence_id"]
    pool = [catalog[c] for c in ownership.get(eid, ()) if c in catalog]
    metric = check["metric"].strip().casefold()
    rows = [c for c in pool if c["row"] > 0 and c["metric"].casefold() == metric]
    groups = {digest(c["group"]): c["group"] for c in rows if c["group"]}
    if len(groups) != 1 or not rows:
        raise ValueError(f"{eid}: numeric check requires one source-grounded group and an exact metric.")
    group = next(iter(groups.values()))
    if {c["row"] for c in rows} != set(group["rows"]) or any(c["number"] is None for c in rows):
        raise ValueError(f"{eid}: comparison rows are incomplete or not plain numbers.")
    excluded = set(check.get("exclude_methods", []))
    # Exclusions must be transparent; arbitrary competitor removal is forbidden.
    if any(name.casefold() not in {"original", "original data", "real", "real data"} for name in excluded):
        raise ValueError(f"{eid}: a compared method cannot be removed from a ranking.")
    if not excluded <= {c["method"] for c in rows}:
        raise ValueError(f"{eid}: excluded reference label is not present in this source group.")
    selected = [c for c in rows if c["method"] not in excluded]
    target = [c for c in selected if c["method"] == check["method"]]
    if len(target) != 1 or not selected:
        raise ValueError(f"{eid}: target method does not match exactly one source row.")
    extreme = (min if check["operation"] == "min" else max)(c["number"] for c in selected)
    winners = [c for c in selected if c["number"] == extreme]
    actual = ("not_extreme" if target[0]["number"] != extreme else
              "tied_extreme" if len(winners) > 1 else "unique_extreme")
    if check["expected_result"] != actual:
        raise ValueError(f"{eid}/{group['dataset_label']}/{check['metric']}/{check['method']}: "
                         f"expected_result must be {actual}, not {check['expected_result']}.")
    labels = "; ".join(f"{c['method']}={c['text']}" for c in sorted(selected, key=lambda c: c["row"]))
    text = (f"{group['dataset_label']} / {check['metric']}: {labels}. "
            f"{check['operation']}={next(c['text'] for c in winners)} "
            f"({', '.join(c['method'] for c in winners)}); {eid}")
    if excluded:
        text += "; excluded reference rows: " + ", ".join(sorted(excluded))
    return text, {c["cell_id"] for c in rows}


ANALYSIS_POLICY = "mapping_readable_v2"


def readable_packet(packet):
    """Lossless labeled view for simple tables; never increase input bytes."""
    view = deepcopy(packet)
    catalog, ownership = table_catalog(packet)
    for item in view.get("evidence_items", []):
        tid = item.get("table_ref")
        table = packet.get("shared_tables", {}).get(tid)
        if not table or not table.get("cell_id_rule"):
            continue
        cells = [catalog[c] for c in ownership.get(item["evidence_id"], ())]
        # Leave merged/multiheader/ambiguous layouts in their existing format.
        if any(c.get("role") not in (None, "data", "column_header")
               or (c.get("role") == "column_header" and c["row"] != 0)
               for c in cells):
            continue
        headers = {c["col"]: c["text"] for c in cells if c["row"] == 0}
        if not headers or any(c["col"] not in headers for c in cells):
            continue
        rows = {}
        for c in sorted(cells, key=lambda c: (c["row"], c["col"])):
            if c["row"] == 0:
                continue
            row = rows.setdefault(c["row"], {"row": c["row"], "values": {}})
            group = c.get("group") or {}
            if group.get("dataset_label"):
                row["source_group_label"] = group["dataset_label"]
            row["values"][f"C{c['col']} {headers[c['col']]}"] = c["text"]
        item["table_reading"] = {
            "label": table.get("label"), "caption": table.get("caption", ""),
            "cell_id_rule": table["cell_id_rule"],
            "header_row": 0,
            "columns": {f"C{k}": v for k, v in sorted(headers.items())},
            "rows": list(rows.values()),
        }
        for key in ("table_ref", "cell_positions", "cell_ids", "caption_ref"):
            item.pop(key, None)
    remaining = {
        i["table_ref"] for i in view.get("evidence_items", [])
        if i.get("table_ref")
    }
    view["shared_tables"] = {
        k: v for k, v in view.get("shared_tables", {}).items()
        if k in remaining
    }
    size = lambda x: len(json.dumps(
        x, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8"))
    # Retain the complete original representation when the alternative is larger.
    return view if size(view) <= size(packet) else deepcopy(packet)


def validate_and_render(result, packet):
    """Check actual source bindings, not arbitrary prose or inferred truth."""
    raw = (
        result.model_dump(mode="json")
        if hasattr(result, "model_dump")
        else deepcopy(result)
    )
    rows = raw.get("claim_analyses", [])
    if [r.get("claim_id") for r in rows] != packet["target_claim_ids"]:
        return [], ["Return exactly one analysis for the requested Claim."]
    catalog, ownership = table_catalog(packet)
    visible = {x["evidence_id"]: x for x in packet["evidence_items"]}
    relations = {
        "supports", "contradicts", "qualifies",
        "context", "insufficient_support"
    }
    errors, output = [], []
    for source in rows:
        row = deepcopy(source)
        edges = row.get("evidence_relations") or []
        ids = [edge.get("evidence_id") for edge in edges]
        if len(ids) != len(set(ids)):
            errors.append("Use one relation per adopted Evidence ID.")
        if set(ids) - set(visible):
            errors.append(
                "A relation cites Evidence outside this Claim's supplied packet."
            )
        if not str(row.get("explanation") or "").strip():
            errors.append("A nonempty Claim explanation is required.")
        texts = [
            row.get("explanation", ""),
            *(row.get("qualifications") or []),
        ]
        for edge in edges:
            eid = edge.get("evidence_id")
            item = visible.get(eid, {})
            parts = list(dict.fromkeys(edge.get("source_parts_used") or []))
            if not parts or set(parts) - set(
                item.get("available_source_parts", [])
            ):
                errors.append(f"Source components were not supplied for {eid}.")
            if (
                edge.get("relation") not in relations
                or not str(edge.get("explanation") or "").strip()
            ):
                errors.append(f"Invalid or unexplained relation for {eid}.")
            edge["source_parts_used"] = parts
            cells = list(dict.fromkeys(edge.get("cell_ids_used") or []))
            if cells and (
                "table_cells" not in parts
                or set(cells) - ownership.get(eid, set())
            ):
                errors.append(
                    f"Selected cells are not in the adopted table Evidence: {eid}."
                )
            edge["cell_ids_used"] = cells
            texts.append(edge.get("explanation", ""))
        mentioned = set(re.findall(
            r"\bevidence_\d{6,}\b", "\n".join(texts)
        ))
        if mentioned - set(ids):
            errors.append(
                "Prose cites Evidence without an adopted relation: "
                + ", ".join(sorted(mentioned - set(ids)))
            )
        if any("{{cell:" in t or "{{check:" in t for t in texts):
            errors.append(
                "Remove numeric template tokens; write source-grounded ordinary prose."
            )
        # Derive the repeated ID index from explicit relations, never infer labels.
        row["evidence_ids_used"] = ids
        row.pop("numeric_checks", None)
        if not ids:
            row["material_status"] = "limited"
        row["grounding_validation"] = {
            "policy": "mapping_readable_v2",
            "status": "failed" if errors else "passed",
            "scope": (
                "Supplied Evidence IDs, source components "
                "and any explicit cell bindings."
            ),
            "numeric_semantics_verified": False,
        }
        output.append(row)
    return output, list(dict.fromkeys(errors))
