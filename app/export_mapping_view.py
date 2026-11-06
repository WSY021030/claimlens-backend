"""Export a read-only, per-Claim view of existing Mapping records.

No API client, Agent, budget, extraction or Mapping execution is used here.
The source analysis is never saved or modified. Default display.json is replaced;
explicit custom exports do not overwrite existing files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.mapping_views import display_view, select_saved_mapping
from app.core.current_storage import atomic_json
from app.models.reader_v3 import AnalysisDocument


def main():
    parser = argparse.ArgumentParser(description="Export existing Mapping display data without model calls")
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--claim-id", action="append", required=True)
    parser.add_argument("--mapping-id", help="Optional exact historical Mapping; requires one Claim")
    parser.add_argument("--output", type=Path, help="Optional explicit export; default replaces current display.json")
    args = parser.parse_args()
    claims = list(dict.fromkeys(args.claim_id))
    if args.mapping_id and len(claims) != 1:
        parser.error("--mapping-id requires exactly one --claim-id.")
    source = args.analysis.resolve()
    output = args.output.resolve() if args.output else source.parent / "display.json"
    if not source.is_file():
        parser.error("Analysis JSON was not found.")
    if source == output or output.name.casefold() in {"analysis.json", "paper.json"}:
        parser.error("Output must be a separate display export, never a source JSON.")
    if output.exists() and args.output and output != source.parent / "display.json":
        parser.error("Explicit export already exists. Default display.json is the only replaceable output.")
    if output.suffix.casefold() != ".json":
        parser.error("The display output must use a .json extension.")
    data = AnalysisDocument.model_validate_json(source.read_text(encoding="utf-8-sig"))
    if any(cid not in data.claim_pool for cid in claims):
        parser.error("An explicitly selected Claim does not exist.")
    views = {}
    for cid in claims:
        mid, check = select_saved_mapping(data, cid, pinned_mapping_id=args.mapping_id)
        views[cid] = (display_view(data, cid, mid) if mid else
            {"claim_id": cid, "mapping_id": args.mapping_id, "status": "unavailable",
             "evidence_ids": [], "edges": [], "issues": check["errors"],
             "rejected_versions": check.get("rejected_versions", [])})
    payload = {"paper_id": data.paper_id, "paper_revision": data.paper_revision,
               "snapshot_id": data.snapshot_id, "workflow_id": None, "display_policy": "per_claim_pinned_mapping_origin_guard_v2",
               "export_mode": "saved_records_only", "not_a_reanalysis": True,
               "display_by_claim": views, "explicit_evidence_only_ids": [],
               "note": "Use each bound Claim view. Null relations are not support labels. Historical raw results remain unchanged; no new scientific judgments were generated."}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, payload)
    for cid, view in views.items():
        print(json.dumps({"claim_id": cid, "mapping_id": view.get("mapping_id"),
                          "status": view.get("status"),
                          "relation_review_required": view.get("relation_review_required"),
                          "evidence_ids": view.get("evidence_ids", [])}, ensure_ascii=False))
    print("Display export:", output)
    print("No model calls. Original analysis, sources and budget ledger were not changed.")


if __name__ == "__main__":
    main()
