"""One-time, API-free migration of historic batch Mappings to per-Claim records.

Use --preview first; --apply creates a timestamped backup before writing.
No Evidence, Claim, Link, or budget ledger data is modified by this command.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2

from app.core.analysis_store import AnalysisStore
from app.core.mapping_records import migrate_grouped_mappings
from app.models.reader_v3 import AnalysisDocument


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preview", action="store_true")
    action.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if not args.analysis.is_file():
        parser.error(f"Analysis file not found: {args.analysis}")
    data = AnalysisDocument.model_validate_json(
        args.analysis.read_text(encoding="utf-8")
    )
    moved = migrate_grouped_mappings(data)

    print("Paper:", data.paper_id)
    print("Grouped Mapping batches to archive:", len(moved))
    for entry in moved:
        print("Archived:", entry["batch_mapping_id"])
        for cid, mid in entry["mapping_ids_by_claim"].items():
            record = data.mappings[mid]
            print("  Claim:", cid, "->", mid,
                  "| Cacheable:", bool(record.get("input_signature")))
        print("  Cross-Claim relation:", entry["relation_id"])
        if entry["issues"]:
            print("  Migration notes:", entry["issues"])
    print("Current per-Claim Mappings:", len(data.mapping_by_claim))
    print("Cross-Claim relations:", len(data.mapping_relations))

    if args.preview:
        print("PREVIEW ONLY. No files changed and no API calls.")
        return
    if not moved:
        print("No grouped Mapping records to migrate. No file changed.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = args.analysis.with_name(f"{args.analysis.stem}.before_mapping_split_{stamp}.json")
    if backup.exists():
        raise FileExistsError(f"Backup already exists: {backup}")
    copy2(args.analysis, backup)
    AnalysisStore(args.analysis).save(data)
    print("Backup:", backup)
    print("MIGRATION COMPLETED:", args.analysis)


if __name__ == "__main__":
    main()
