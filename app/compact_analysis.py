"""Migrate Reader-v3 JSON to current-state storage without any model calls."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from uuid import uuid4

from app.core.analysis_store import AnalysisStore
from app.core.current_storage import (
    STORAGE_POLICY, atomic_json, current_document, describe_change,
)
from app.models.reader_v3 import AnalysisDocument

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXED_EXPORTS = ("display.json", "response.json", "answer.md", "selected_claims.json")


def digest(content):
    return hashlib.sha256(content).hexdigest()


def reject_linked(path):
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError(f"Linked paths are not supported for migration: {part}")


def generated_history(base, payload):
    """Recognize program-generated artifacts; never glob-delete user uploads."""
    run_ids = {str(r.get("run_id")) for r in payload.get("runs") or []}
    result = []
    for path in base.iterdir():
        if not path.is_file() or path.is_symlink() or path.name in FIXED_EXPORTS:
            continue
        if path.suffix == ".json" and path.name.startswith(("display_", "selected_claims_", "processing_report_")):
            try:
                item = json.loads(path.read_text(encoding="utf-8-sig"))
            except (ValueError, OSError):
                continue
            if not isinstance(item, dict):
                continue
            is_display = (path.name.startswith("display_")
                and item.get("paper_id") == payload["paper_id"]
                and isinstance(item.get("display_by_claim"), dict)
                and bool(item.get("display_policy")))
            is_selection = (path.name.startswith("selected_claims_")
                and item.get("paper_id") == payload["paper_id"]
                and str(item.get("run_id")) in run_ids and "claim_ids" in item)
            is_report = (path.name.startswith("processing_report_")
                and str(item.get("workflow_id")) in run_ids and item.get("not_a_final_answer") is True)
            if is_display or is_selection or is_report:
                result.append(path)
        elif path.suffix == ".md" and path.name.startswith("answer_"):
            try:
                text = path.read_text(encoding="utf-8-sig")
            except OSError:
                continue
            match = re.search(r"^Workflow:\s*(\S+)", text, re.M)
            if text.startswith("# Paper question response") and match and match.group(1) in run_ids:
                result.append(path)
    return sorted(result)


def inspect(path, prune):
    reject_linked(path)
    old_bytes = path.read_bytes()
    raw = json.loads(old_bytes.decode("utf-8-sig"))
    data = AnalysisDocument.model_validate(raw)
    if any(r.get("status") == "running" for r in data.runs):
        raise ValueError(f"A running checkpoint exists in {path}. Stop/finish that workflow before migration.")
    hydrated = data.model_dump(mode="json")
    projected = current_document(hydrated)
    if (hydrated["claim_pool"] != projected["claim_pool"]
            or hydrated["evidence_pool"] != projected["evidence_pool"]):
        raise ValueError("Source-pool preservation check failed; nothing will be written.")
    legacy = generated_history(path.parent, raw) if prune else []
    report = {"analysis": str(path), "policy": STORAGE_POLICY,
              **describe_change(hydrated, projected),
              "actual_existing_bytes": len(old_bytes),
              "generated_history_to_remove": [p.name for p in legacy],
              "budget_ledger": "unchanged", "model_calls": 0}
    return data, raw, old_bytes, projected, legacy, report


def apply_one(path, prepared, backup_root):
    data, raw, before, projected, legacy, report = prepared
    lock = path.parent / ".storage-maintenance.lock"
    token = uuid4().hex
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token)
        if path.read_bytes() != before:
            raise ValueError("Analysis changed after preview; no migration applied to this paper.")
        serialized = (json.dumps(projected, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        changed = serialized != before
        backup = None
        saved = {}
        if changed or legacy:
            backup = backup_root / data.paper_id
            backup.mkdir(parents=True, exist_ok=False)
            for old in [path, *[path.parent / name for name in FIXED_EXPORTS if (path.parent / name).is_file()], *legacy]:
                reject_linked(old)
                content = old.read_bytes()
                saved[old.name] = content
                (backup / old.name).write_bytes(content)
            atomic_json(backup / "restore_manifest.json", {
                "paper_id": data.paper_id, "original_directory": str(path.parent),
                "files": {name: digest(content) for name, content in saved.items()},
                "note": "One-time migration backup, not an ongoing runtime archive. Budget and standardized sources were not touched.",
            })
        store = AnalysisStore(path)
        store._maintenance_token = token
        store.save(data)
        # Source pools and current mappings are authoritative even when a derived
        # output fails. Do not delete older displays unless the new ones exist
        # with the committed snapshot and a non-error display payload.
        check = json.loads((path.parent / "display.json").read_text(encoding="utf-8"))
        if check.get("snapshot_id") != projected["snapshot_id"] or check.get("status") == "unavailable":
            raise RuntimeError("Analysis was compacted but display export failed. Legacy exports were preserved; inspect the backup and output.")
        for old in legacy:
            if old.read_bytes() != saved[old.name]:
                raise ValueError(f"Generated artifact changed during migration; preserved: {old.name}")
            old.unlink()
        print("Compacted:", path, flush=True)
        print("Current outputs: analysis.json, display.json, response.json; answer.md when available.", flush=True)
        if backup is not None:
            print("One-time data backup:", backup, flush=True)
        else:
            print("Already current; derived outputs refreshed without an extra backup.", flush=True)
        print("Original Claims/Evidence, IDs, numerical values, anchors and budget ledger were preserved.", flush=True)
    finally:
        try:
            if lock.read_text(encoding="utf-8") == token:
                lock.unlink()
        except FileNotFoundError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Current-state JSON migration; preview by default, no model calls")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--analysis", type=Path, action="append", help="Repeat for exact canonical paper analysis files")
    scope.add_argument("--all-papers", action="store_true", help="Direct paper folders under data/analysis; never _archives")
    parser.add_argument("--apply", action="store_true", help="Back up once and atomically write the compact state")
    parser.add_argument("--prune-generated", action="store_true", help="Back up and remove recognized old per-run outputs, not arbitrary files")
    args = parser.parse_args()
    root = (PROJECT_ROOT / "data" / "analysis").resolve()
    paths = (sorted(p / "analysis.json" for p in root.iterdir()
                    if p.is_dir() and not p.name.startswith(("_", ".")) and (p / "analysis.json").is_file())
             if args.all_papers and root.is_dir() else list(args.analysis or []))
    paths = list(dict.fromkeys(path.resolve() for path in paths))
    if not paths:
        parser.error("No analysis files were selected.")
    prepared = []
    for path in paths:
        if path.name != "analysis.json" or path.parent.parent != root or not path.is_file():
            parser.error("Use a canonical data/analysis/<paper_id>/analysis.json path.")
        item = inspect(path, args.prune_generated)
        if item[0].paper_id != path.parent.name:
            parser.error("Analysis paper_id does not match its canonical directory.")
        print(json.dumps(item[-1], ensure_ascii=False, indent=2), flush=True)
        prepared.append((path, item))
    if not args.apply:
        print("Preview only: no files, APIs, budget or checkpoints were changed.", flush=True)
        return
    backup_root = PROJECT_ROOT / ".reader_storage_backups" / datetime.now().strftime("current_state_%Y%m%d_%H%M%S_%f")
    for path, item in prepared:
        apply_one(path, item, backup_root)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print("Storage migration stopped: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
