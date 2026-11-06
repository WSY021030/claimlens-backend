"""Remove retired candidate metadata locally; preview unless --apply is used.

No models, agents, AnalysisStore, source compaction, or budget code are imported.
Existing scientific records are compared exactly before any write. Only active
paper folders are visited; archives, uploads, sources and databases are untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4

from app.core.retired_associations import clean_analysis_payload, clean_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXED_OUTPUTS = ("display.json", "response.json", "selected_claims.json")
PROTECTED = ("claim_pool", "evidence_pool", "mappings", "mapping_by_claim",
             "mapping_relations", "mapping_batch_archive", "completed_claim_windows",
             "completed_evidence_scans", "evidence_image_candidates", "id_counters")


def safe_path(path: Path, root: Path) -> None:
    absolute = path.absolute()
    if not absolute.is_relative_to(root):
        raise ValueError("Only canonical paper analysis folders may be edited.")
    current = root
    for part in absolute.relative_to(root).parts:
        current /= part
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            raise ValueError(f"Refusing a filesystem alias: {current}")


def encode(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def snapshot(value: dict) -> str:
    body = {key: item for key, item in value.items() if key != "snapshot_id"}
    return sha256(json.dumps(body, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def atomic_write(path: Path, content: bytes) -> None:
    fd, temp = tempfile.mkstemp(prefix=".reader-schema-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def prepare(path: Path, analysis_root: Path):
    safe_path(path, analysis_root)
    if path.name != "analysis.json" or path.parent.parent != analysis_root or not path.is_file():
        raise ValueError("Use data/analysis/<paper_id>/analysis.json, not a manual experiment copy.")
    if (path.parent / ".storage-maintenance.lock").exists():
        raise ValueError(f"Storage maintenance is already active: {path.parent}")
    before = path.read_bytes()
    raw = json.loads(before.decode("utf-8-sig"))
    if not isinstance(raw, dict) or raw.get("schema_version") != "reader.v3":
        raise ValueError(f"Unrecognized analysis schema: {path}")
    if raw.get("paper_id") != path.parent.name:
        raise ValueError(f"Paper ID does not match its canonical folder: {path}")
    if any(run.get("status") == "running" for run in raw.get("runs", []) if isinstance(run, dict)):
        raise ValueError(f"A running checkpoint is recorded: {path}. Stop or finish it before migration.")
    removed = []
    result = clean_analysis_payload(raw, removed=removed)
    for key in PROTECTED:
        if raw.get(key) != result.get(key):
            raise ValueError(f"Protected data would change: {key}; nothing will be written.")
    old_snapshot = raw.get("snapshot_id")
    if removed and old_snapshot:
        result["snapshot_id"] = snapshot(result)
    changed = [(path, before, encode(result))] if removed else []
    removed_by_file = {path.name: removed} if removed else {}
    for name in FIXED_OUTPUTS:
        output = path.parent / name
        if not output.exists():
            continue
        safe_path(output, analysis_root)
        previous = output.read_bytes()
        value = json.loads(previous.decode("utf-8-sig"))
        if not isinstance(value, dict) or value.get("paper_id") != raw["paper_id"]:
            raise ValueError(f"Unrecognized fixed output; no migration applied: {output}")
        output_removed = []
        newer = clean_checkpoint(value, removed=output_removed)
        # Never relabel a stale derived view as current. A matching prior view
        # can share the new identity because scientific content is unchanged.
        matched = bool(removed and old_snapshot and value.get("snapshot_id") == old_snapshot
                       and value.get("paper_revision") == raw.get("paper_revision"))
        if matched:
            newer["snapshot_id"] = result["snapshot_id"]
        if output_removed or matched:
            changed.append((output, previous, encode(newer)))
            removed_by_file[name] = output_removed
    report = {"paper_id": raw["paper_id"], "analysis": str(path),
              "candidate_metadata_fields_to_remove": sum(len(x) for x in removed_by_file.values()),
              "changed_files": [item[0].name for item in changed],
              "source_records_unchanged": True, "mapping_results_unchanged": True,
              "completed_scan_hashes_unchanged": True, "budget_ledger": "untouched",
              "model_calls": 0, "preview_only": True}
    return path, changed, report


def apply_one(prepared, backup_root: Path, analysis_root: Path) -> None:
    path, changes, _ = prepared
    if not changes:
        print("Already current:", path.parent.name, flush=True)
        return
    lock = path.parent / ".storage-maintenance.lock"
    token = uuid4().hex
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token)
        for target, original, _ in changes:
            safe_path(target, analysis_root)
            if target.read_bytes() != original:
                raise ValueError(f"A file changed after preview; no edits applied: {target}")
        backup = backup_root / path.parent.name
        backup.mkdir(parents=True, exist_ok=False)
        for target, original, _ in changes:
            with (backup / target.name).open("xb") as handle:
                handle.write(original)
        written = []
        try:
            for target, original, replacement in changes:
                if target.read_bytes() != original:
                    raise ValueError(f"Concurrent writer detected: {target}; stop all Reader processes.")
                atomic_write(target, replacement)
                written.append((target, original, replacement))
        except BaseException:
            for target, original, replacement in reversed(written):
                if target.read_bytes() == replacement:
                    atomic_write(target, original)
            raise
        print("Updated:", path.parent.name, flush=True)
        print("One-time rollback backup:", backup, flush=True)
        print("Claims, Evidence, Mapping, scan hashes and budget spending were preserved.", flush=True)
    finally:
        if lock.exists() and lock.read_text(encoding="utf-8") == token:
            lock.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Retire candidate associations; local-only, preview by default")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--all-papers", action="store_true")
    choice.add_argument("--analysis", type=Path, action="append")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = (PROJECT_ROOT / "data" / "analysis").absolute()
    if not root.is_dir() or root.is_symlink():
        parser.error("Canonical data/analysis directory was not found or is a filesystem alias.")
    paths = (sorted(p / "analysis.json" for p in root.iterdir()
                    if p.is_dir() and not p.name.startswith((".", "_")) and (p / "analysis.json").is_file())
             if args.all_papers else [p.absolute() for p in args.analysis or []])
    if not paths:
        parser.error("No canonical paper analyses were found.")
    try:
        prepared = [prepare(path, root) for path in dict.fromkeys(paths)]
        for _, _, report in prepared:
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if args.apply:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            backup = PROJECT_ROOT / ".reader_storage_backups" / ("retired_candidates_" + stamp)
            if backup.parent.is_symlink() or (hasattr(backup.parent, "is_junction") and backup.parent.is_junction()):
                raise ValueError("Rollback backup directory must not be a filesystem alias.")
            for item in prepared:
                apply_one(item, backup, root)
        else:
            print("Preview only. No file writes or model calls. Add --apply for the local migration.")
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
