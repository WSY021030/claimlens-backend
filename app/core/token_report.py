"""User-facing Token reporting for one independent Reader-v3 task.

The USD budget remains keyed to this task/Workflow ID. Child agents and
resumed attempts count toward that same task, never toward other workflows.
"""
from __future__ import annotations


def task_token_report(run, all_runs=None):
    """Count recorded API usage once per direct or child invocation."""
    reports = list(run.get("attempt_cost_reports") or [])
    direct = run.get("api_cost_report")
    if direct is not None:
        reports.append(direct)
    run_id = run.get("run_id")
    child_ids = set()
    for stage in run.get("stage_results") or []:
        child_ids.update(stage.get("child_run_ids") or [])
    for candidate in all_runs or []:
        if candidate.get("run_id") == run_id:
            continue
        if candidate.get("parent_workflow_id") == run_id:
            child_ids.add(candidate.get("run_id"))
    for candidate in all_runs or []:
        if candidate.get("run_id") in child_ids and candidate.get("stage") in {
                "claim", "evidence", "mapping", "direct_evidence"}:
            report = candidate.get("api_cost_report")
            if report is not None:
                reports.append(report)
    counts = {name: 0 for name in (
        "llm_calls_recorded", "input_tokens", "cached_input_tokens", "output_tokens",
        "reasoning_tokens", "failed_attempts",
    )}
    agents = {}
    costs = []
    for report in reports:
        for name in counts:
            counts[name] += int(report.get(name) or 0)
        if report.get("estimated_total_usd") is not None:
            costs.append(report["estimated_total_usd"])
        for item in report.get("by_agent") or []:
            name = item.get("agent") or "Unknown"
            result = agents.setdefault(name, {"calls": 0, "input_tokens": 0,
                                             "cached_input_tokens": 0, "output_tokens": 0})
            for source_key, target_key in (
                    ("llm_calls", "calls"), ("input_tokens", "input_tokens"),
                    ("cached_input_tokens", "cached_input_tokens"),
                    ("output_tokens", "output_tokens")):
                result[target_key] += int(item.get(source_key) or 0)
    return {
        **counts,
        "total_tokens": counts["input_tokens"] + counts["output_tokens"],
        "uncached_input_tokens": max(0, counts["input_tokens"] - counts["cached_input_tokens"]),
        "estimated_total_usd": round(sum(costs), 9) if len(costs) == len(reports) else None,
        "token_usage_completeness": "complete" if all(r.get("complete") for r in reports) else "partial",
        "by_agent": {key: agents[key] for key in sorted(agents)},
        "counting_scope": "this task, including recorded child agents and resume attempts",
        "note": "Based on API usage receipts. Missing receipts/hidden SDK retries may not be counted.",
    }


def print_task_tokens(usage, budget=None):
    """Compact terminal report; all quantities are actual reported Tokens."""
    print("\n=== TASK TOKEN USAGE ===", flush=True)
    print("Recorded LLM calls:", usage["llm_calls_recorded"], flush=True)
    print("Input tokens:", usage["input_tokens"], flush=True)
    print("Cached input tokens (included above):", usage["cached_input_tokens"], flush=True)
    print("Uncached input tokens:", usage["uncached_input_tokens"], flush=True)
    print("Output tokens:", usage["output_tokens"], flush=True)
    print("Reasoning tokens (included in output):", usage["reasoning_tokens"], flush=True)
    print("Input + output tokens:", usage["total_tokens"], flush=True)
    for agent, record in usage["by_agent"].items():
        print("Agent:", agent, "calls=", record["calls"],
              "input=", record["input_tokens"],
              "output=", record["output_tokens"], flush=True)
    print("Estimated task USD:", usage["estimated_total_usd"], flush=True)
    if budget:
        print("Independent task budget ID:", budget.get("question_budget_id"), flush=True)
        print("Task USD limit:", budget.get("limit_usd"), flush=True)
        print("Task USD operating limit:", budget.get("operating_limit_usd"), flush=True)
        print("Task USD committed:", budget.get("committed_usd"), flush=True)
        print("Task USD remaining (operating):", budget.get("remaining_usd"), flush=True)
    if usage["token_usage_completeness"] != "complete":
        print("Warning: some API usage receipts are unavailable.", flush=True)
