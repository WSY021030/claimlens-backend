"""Current-state JSON persistence for Reader-v3.

This module never calls a model or changes scientific content. Source pools keep
all records, including independent Evidence not yet adopted by a Mapping. Only
superseded results and diagnostic histories are discarded. Active checkpoints,
scan identities, source fingerprints and accumulated cost totals are retained.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
from app.core.retired_associations import clean_analysis_payload, scan_policy

STORAGE_POLICY = "current_state_v2"
REPORT_COUNTS = (
    "llm_calls_recorded", "input_tokens", "cached_input_tokens", "output_tokens",
    "reasoning_tokens", "failed_attempts", "input_images",
)
BUDGET_FIELDS = (
    "paper_id", "question_budget_id", "budget_policy", "limit_usd",
    "operating_limit_usd", "safety_margin_usd", "committed_usd", "remaining_usd",
    "uncertain_entries", "blocked", "stop_reason", "estimate_only",
)
# These are diagnostics, not the checkpoints used to avoid repeating paid work.
RUN_DROP = {
    "orchestration_history", "answer_history", "answer_history_signature",
    "historical_processing_reports", "mapping_input_receipts", "mapping_attempts",
    "call_timings", "image_report", "source_reading", "answer_reference_check",
    "answer_material_ids", "display_by_claim", "display_evidence_ids",
    "display_file", "answer_file", "processing_report_file", "result_file",
    "llm_wall_time_seconds", "duration_seconds_last_attempt", "duration_seconds",
    "available_source_ids", "claim_selection_page_decisions",
}
MAPPING_DROP = {
    "available_evidence_ids", "cache_dependency_evidence_ids",
    "cache_dependency_claim_ids", "migrated_from_mapping_id",
}
FINISHED = {"completed", "cancelled", "abandoned"}
STAGE_NAMES = {"claim", "evidence", "mapping", "direct_evidence"}


def unique(values):
    return list(dict.fromkeys(value for value in values if value is not None))


def encoded(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def atomic_text(path: Path, text: str) -> None:
    """Atomic replacement on the destination filesystem; no rolling backups."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Refusing to replace a linked output: {path}")
    fd, temp = tempfile.mkstemp(prefix=".reader-current-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def atomic_json(path: Path, payload: dict) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def compact_cost_report(report: dict | None) -> dict:
    """Keep aggregate reported usage, not API-response or per-call history."""
    report = report or {}
    result = {key: deepcopy(report[key]) for key in (*REPORT_COUNTS, "complete",
        "estimated_total_usd", "known_cost_usd", "currency", "estimate_only") if key in report}
    result["by_agent"] = []
    for agent in report.get("by_agent") or []:
        if isinstance(agent, dict):
            result["by_agent"].append({key: deepcopy(agent[key]) for key in (
                "agent", "llm_calls", "input_tokens", "cached_input_tokens",
                "output_tokens", "estimated_cost_usd", "known_cost_usd") if key in agent})
    return result


def combined_cost_reports(reports: list[dict]) -> dict:
    """Collapse attempts into one additive receipt without resetting any total."""
    rows = [r for r in reports if isinstance(r, dict)]
    totals = {key: sum(int(r.get(key) or 0) for r in rows) for key in REPORT_COUNTS}
    totals.update({"currency": "USD", "estimate_only": True,
        "complete": all(r.get("complete") is True for r in rows),
        "estimated_total_usd": (round(sum(float(r["estimated_total_usd"]) for r in rows), 9)
            if all(r.get("estimated_total_usd") is not None for r in rows) else None),
        "known_cost_usd": round(sum(float(r.get("known_cost_usd") or 0) for r in rows), 9)})
    buckets: dict[str, dict] = {}
    for report in rows:
        for item in report.get("by_agent") or []:
            name = item.get("agent") or "Unknown"
            bucket = buckets.setdefault(name, {"agent": name, "llm_calls": 0,
                "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0})
            for key in ("llm_calls", "input_tokens", "cached_input_tokens", "output_tokens"):
                bucket[key] += int(item.get(key) or 0)
    totals["by_agent"] = [buckets[name] for name in sorted(buckets)]
    return totals


def compact_budget(report):
    return {key: deepcopy(report[key]) for key in BUDGET_FIELDS if key in (report or {})}


def compact_usage(report):
    keys = (*REPORT_COUNTS, "total_tokens", "uncached_input_tokens",
            "estimated_total_usd", "token_usage_completeness", "by_agent")
    return {key: deepcopy(report[key]) for key in keys if key in (report or {})}


def _budget_id(run):
    return run.get("question_budget_id") or (run.get("budget_report") or {}).get("question_budget_id")


def _retained_run_ids(runs):
    """Keep live jobs, the latest question and the latest worker handshake.

    Children of retained questions remain small cost receipts so the existing
    Token reporter can add them exactly once. Closed older questions are not a
    persistent chat history. Their budget ledger is deliberately untouched.
    """
    if not runs:
        return set()
    parents = [r for r in runs if r.get("stage") == "coordinator"]
    activity = lambda r: str(r.get("finished_at") or r.get("started_at") or "")
    ids = {runs[-1].get("run_id"), max(runs, key=activity).get("run_id")}
    if parents:
        ids.add(max(parents, key=activity).get("run_id"))
    # A resumed job can be older in insertion order. Keep its final save even
    # when it has just changed from running to completed.
    active_id = os.environ.get("READER_V3_QUESTION_BUDGET_ID")
    if any(r.get("run_id") == active_id for r in runs):
        ids.add(active_id)
    for run in runs:
        if run.get("status") == "running" or (
            run.get("stage") == "coordinator" and run.get("status") not in FINISHED):
            ids.add(run.get("run_id"))
    changed = True
    while changed:
        before = set(ids)
        parents_kept = {r.get("run_id") for r in parents if r.get("run_id") in ids}
        for run in runs:
            if run.get("run_id") in ids:
                for step in run.get("stage_results") or []:
                    ids.update(step.get("child_run_ids") or [])
                    if step.get("run_id"):
                        ids.add(step["run_id"])
            if run.get("parent_workflow_id") in parents_kept or _budget_id(run) in parents_kept:
                ids.add(run.get("run_id"))
        changed = ids != before
    return ids - {None}


def _compact_state(state, finished):
    out = deepcopy(state)
    for key in ("collection_reviews", "source_packet", "source_packet_bytes",
                "mapping_read_checks", "collection_review_reference_check"):
        out.pop(key, None)
    selection = out.get("claim_selection")
    if isinstance(selection, dict):
        selection.pop("page_decisions", None)
        selection.pop("not_selected_claim_ids", None)
    if finished:
        for key in ("numeric_table_audit", "exact_table_lookup", "source_notes"):
            out.pop(key, None)
    return out


def _compact_run(run):
    out = {key: deepcopy(value) for key, value in run.items() if key not in RUN_DROP}
    if "api_cost_report" in run:
        out["api_cost_report"] = compact_cost_report(run["api_cost_report"])
    if run.get("attempt_cost_reports"):
        out["attempt_cost_reports"] = [combined_cost_reports(run["attempt_cost_reports"])]
    if "budget_report" in run:
        out["budget_report"] = compact_budget(run["budget_report"])
    if "token_usage_report" in run:
        out["token_usage_report"] = compact_usage(run["token_usage_report"])
    finished = run.get("status") in FINISHED
    if finished:
        # Page notes are a retry checkpoint, not another completed answer copy.
        out.pop("source_page_cache", None)
    if isinstance(out.get("task_states"), dict):
        out["task_states"] = {key: _compact_state(state, finished)
                              for key, state in out["task_states"].items()}
    # A current error must remain visible. Lists here are already bounded by
    # the work of the retained task, not by the paper's lifetime.
    out.setdefault("issues", [])
    out.setdefault("stage_results", [])
    return out


def _mapping_references(run):
    """Protect results actually bound to a current/resumable request."""
    ids = set(run.get("used_mapping_ids") or [])
    for state in (run.get("task_states") or {}).values():
        ids.update(state.get("mapping_ids") or [])
        ids.update((state.get("mapping_ids_by_claim") or {}).values())
    request = (run.get("request_contract") or {}).get("request") or {}
    ids.update((request.get("mapping_ids_by_claim") or {}).values())
    for source in (run.get("answer") or {}).get("used_source_ids") or []:
        if isinstance(source, str) and source.startswith("mapping_"):
            ids.add(source)
    # Incomplete worker results are retained for diagnosis/recovery, never
    # promoted to a successful Mapping by the retention policy.
    if run.get("status") not in FINISHED:
        ids.update(run.get("created_mapping_ids") or [])
    return ids


def _compact_mapping(record):
    out = {key: deepcopy(value) for key, value in record.items() if key not in MAPPING_DROP}
    receipt = record.get("retrieval_receipt")
    if isinstance(receipt, dict):
        keep = ("policy", "claim_id", "expanded", "packet_bytes", "packet_byte_limit",
                "pool_records_considered_locally", "complete_recall_guaranteed",
                "scope_element_ids", "missing_direct_source_ids")
        minimal = {key: deepcopy(receipt[key]) for key in keep if key in receipt}
        for source, target in (("selected_evidence_ids", "candidate_count"),
            ("priority_deferred", "priority_deferred_count"),
            ("deferred_for_size", "deferred_for_size_count"),
            ("claim_origin_only_evidence_ids", "claim_origin_only_count")):
            minimal[target] = len(receipt[source]) if source in receipt else int(receipt.get(target) or 0)
        out["retrieval_receipt"] = minimal
    return out


def _scan_batches(payload):
    """Retain hash reconstruction inputs, not historical Evidence run logs."""
    records = list(payload.get("evidence_scan_batches") or [])
    records.extend(run for run in payload.get("runs") or []
                   if run.get("stage") == "evidence" and scan_policy(run))
    result = {}
    for record in records:
        if record.get("paper_revision") not in (None, payload["paper_revision"]):
            continue
        claims = [cid for cid in record.get("claim_ids") or [] if cid in payload["claim_pool"]]
        if not claims:
            continue
        item = {"stage": "evidence", "model": record.get("model"),
                "paper_revision": record.get("paper_revision"),
                "scan_cache_policy": scan_policy(record), "claim_ids": claims,
                "chunk_chars": record.get("chunk_chars", 6000),
                "claim_batch_size": record.get("claim_batch_size", 15)}
        result[encoded(item)] = item
    return list(result.values())


def _highwater(keys, prefix):
    pattern = re.compile(re.escape(prefix) + r"_(\d+)$")
    numbers = [int(match.group(1)) for key in keys if (match := pattern.fullmatch(str(key)))]
    return max(numbers, default=0)


def current_document(payload: dict) -> dict:
    """Project an in-memory document to current state without mutating it.

    Keep source records byte-for-byte as JSON values. Do not infer correctness,
    manufacture edge labels, merge observations, or prune unused source Evidence.
    """
    if payload.get("schema_version") != "reader.v3":
        raise ValueError("Only reader.v3 analysis files support this retention policy.")
    payload = clean_analysis_payload(payload)
    out = deepcopy(payload)
    runs = payload.get("runs") or []
    retained_ids = _retained_run_ids(runs)
    retained = [_compact_run(r) for r in runs if r.get("run_id") in retained_ids]
    mappings = payload.get("mappings") or {}
    archives = payload.get("mapping_batch_archive") or {}
    counters = dict(payload.get("id_counters") or {})
    counters["mapping"] = max(int(counters.get("mapping") or 0),
        _highwater([*mappings, *archives], "mapping"))
    counters["relation"] = max(int(counters.get("relation") or 0),
        _highwater(payload.get("mapping_relations") or {}, "relation"))
    pointers = dict(payload.get("mapping_by_claim") or {})
    for cid, mid in pointers.items():
        if cid not in payload["claim_pool"] or mid not in mappings:
            raise ValueError(f"Broken current Mapping pointer; no compaction performed: {cid}/{mid}")
    required = set(pointers.values())
    # Do not discard a usable per-Claim result solely because an older writer
    # failed to create its pointer. Retain one such version without promoting it.
    unbound_latest = {}
    for mid, record in mappings.items():
        cid = record.get("claim_id")
        if (cid in payload["claim_pool"] and cid not in pointers
                and record.get("status") == "completed"
                and record.get("paper_revision") == payload["paper_revision"]):
            unbound_latest[cid] = mid
    required.update(unbound_latest.values())
    for run in retained:
        required.update(_mapping_references(run))
    out["mappings"] = {mid: _compact_mapping(row) for mid, row in mappings.items() if mid in required}
    out["mapping_batch_archive"] = {mid: deepcopy(row) for mid, row in archives.items() if mid in required}
    existing = set(out["mappings"]) | set(out["mapping_batch_archive"])
    out["mapping_relations"] = {rid: deepcopy(row)
        for rid, row in (payload.get("mapping_relations") or {}).items()
        if set(row.get("mapping_ids") or []) <= existing
        and row.get("paper_revision") == payload["paper_revision"]}
    for run in retained:
        for key in ("created_mapping_ids", "reused_mapping_ids", "used_mapping_ids"):
            if key in run:
                run[key] = [mid for mid in run[key] if mid in existing]
    out["runs"] = retained
    out["evidence_scan_batches"] = _scan_batches(payload)
    # Missing image-candidate keys already mean an empty list in all runners.
    out["evidence_image_candidates"] = {key: value for key, value in
        (payload.get("evidence_image_candidates") or {}).items() if value}
    out["id_counters"] = counters
    out["storage_policy"] = STORAGE_POLICY
    out.pop("snapshot_id", None)
    out["snapshot_id"] = sha256(encoded(out).encode("utf-8")).hexdigest()
    return out


def current_root_run(payload):
    runs = payload.get("runs") or []
    if not runs:
        return None
    # A resumed older checkpoint need not be last in the list. The workflow
    # owns this environment ID and restores it after completion. Prefer that
    # exact question over insertion order, including its final successful save.
    active_id = os.environ.get("READER_V3_QUESTION_BUDGET_ID")
    active = next((r for r in reversed(runs) if r.get("stage") == "coordinator"
                   and r.get("run_id") == active_id), None)
    if active is not None:
        return active
    # On offline migration use the latest recorded activity, not whichever
    # child happened to have been appended after its parent.
    latest = max(enumerate(runs), key=lambda pair: (
        str(pair[1].get("finished_at") or pair[1].get("started_at") or ""), pair[0]))[1]
    parent_id = latest.get("parent_workflow_id") or _budget_id(latest)
    parent = next((r for r in reversed(runs) if r.get("stage") == "coordinator"
                   and r.get("run_id") == parent_id), None)
    return parent or latest


def response_payload(payload):
    run = current_root_run(payload) or {}
    states = run.get("task_states") or {}
    claims = unique(cid for state in states.values() for cid in state.get("selected_claim_ids") or [])
    claims = claims or list(run.get("target_claim_ids") or run.get("claim_ids") or [])
    result = {"schema_version": "reader.response.current.v1", "paper_id": payload["paper_id"],
        "paper_revision": payload["paper_revision"], "snapshot_id": payload["snapshot_id"],
        "workflow_id": run.get("run_id"), "status": run.get("status", "ready"),
        "stage": run.get("stage"), "mode": run.get("mode"),
        "question": run.get("user_question") or run.get("source_question"),
        "target_claim_ids": claims, "answer_status": run.get("answer_status", "unavailable"),
        "answer": deepcopy(run.get("answer")),
        "unfinished_tasks": deepcopy(run.get("unfinished_tasks") or []),
        "issues": deepcopy(run.get("issues") or []),
        "token_usage": compact_usage(run.get("token_usage_report")),
        "budget": compact_budget(run.get("budget_report")),
        "analysis_file": "analysis.json", "display_file": "display.json"}
    if run.get("selection"):
        result["selection"] = deepcopy(run["selection"])
    return result


def write_current_outputs(data, payload, base: Path) -> None:
    """Replace a fixed set of derived files; no run-ID output history.

    analysis.json is the authoritative commit. Clients must require matching
    snapshot_id values before pairing it with derived display/response JSON.
    """
    from app.core.mapping_views import display_view
    base = Path(base)
    bindings = dict(payload.get("mapping_by_claim") or {})
    root = current_root_run(payload) or {}
    requested = {}
    for state in (root.get("task_states") or {}).values():
        requested.update(state.get("mapping_ids_by_claim") or {})
    # An explicitly pinned active request keeps its own version. Other Claims
    # show current pointers. Superseded unbound versions are not published.
    bindings.update({cid: mid for cid, mid in requested.items() if mid in payload["mappings"]})
    display = {"paper_id": payload["paper_id"], "paper_revision": payload["paper_revision"],
        "snapshot_id": payload["snapshot_id"], "workflow_id": root.get("run_id"),
        "display_policy": "current_per_claim_origin_guard_v1",
        "display_by_claim": {cid: display_view(data, cid, mid) for cid, mid in bindings.items()},
        "explicit_evidence_only_ids": list(root.get("explicit_evidence_only_ids") or []),
        "note": "Use per-Claim bindings. Null relations are not support. Require snapshot_id to match analysis.json."}
    atomic_json(base / "display.json", display)
    response = response_payload(payload)
    atomic_json(base / "response.json", response)
    answer = response.get("answer") or {}
    text = answer.get("answer") if isinstance(answer, dict) else None
    answer_path = base / "answer.md"
    if text:
        content = (f"# Paper question response\n\nWorkflow: {response['workflow_id']}\n\n"
            f"Status: {response['status']}\n\n## Question\n\n{response.get('question') or ''}\n\n"
            f"## Response\n\n{text}\n")
        if answer.get("limitations"):
            content += "\n## Limitations\n\n" + "\n".join("- " + item for item in answer["limitations"]) + "\n"
        atomic_text(answer_path, content)
    else:
        # Never leave an earlier successful answer looking like this operation.
        answer_path.unlink(missing_ok=True)
    # Claim-only result is a small index, not another copy of the source pool.
    selected_path = base / "selected_claims.json"
    if root.get("mode") == "selected_claims":
        atomic_json(selected_path, {"paper_id": payload["paper_id"],
            "paper_revision": payload["paper_revision"], "snapshot_id": payload["snapshot_id"],
            "run_id": root.get("run_id"), "status": root.get("status"),
            "selected_segments": (root.get("selection") or {}).get("segments", []),
            "claim_ids": (root.get("selection") or {}).get("claim_ids", []),
            "analysis_file": "analysis.json", "response_file": "response.json"})
    else:
        selected_path.unlink(missing_ok=True)


def describe_change(before, after):
    counts = {}
    for name in ("claim_pool", "evidence_pool", "mappings", "runs", "mapping_batch_archive"):
        counts[name] = {"before": len(before.get(name) or []), "after": len(after.get(name) or [])}
    counts["old_format_bytes"] = len(json.dumps(before, ensure_ascii=False, indent=2).encode("utf-8"))
    counts["current_format_bytes"] = len(json.dumps(after, ensure_ascii=False, indent=2).encode("utf-8"))
    counts["source_records_unchanged"] = (before.get("claim_pool") == after.get("claim_pool")
                                           and before.get("evidence_pool") == after.get("evidence_pool"))
    counts["id_counters"] = after["id_counters"]
    return counts
