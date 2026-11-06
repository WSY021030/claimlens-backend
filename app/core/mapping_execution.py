"""Shared bounded Mapping execution with per-target failure isolation."""
from __future__ import annotations

from time import perf_counter
from openai import APIConnectionError, APIStatusError, APITimeoutError
from app.agents.mapping_agent import MappingAgent
from app.budget import BudgetStop
from app.llm import StructuredOutputParseError
from app.core.evidence_retrieval import EvidenceDataManager
from app.core.mapping_records import claim_cache_key, find_cached_claim, save_mapping_analyses

MODEL_ERRORS = (StructuredOutputParseError, APIConnectionError, APIStatusError, APITimeoutError)


def run_claim_mappings(*, data, store, run, claim_ids, model, language, tracker, budget,
                       paper=None, analysis_focus="general", allowed_element_ids=None,
                       rerun=False, on_saved=None):
    """At most two calls per target in this attempt; never expand Claim targets.

    An optional expansion failure does not erase a valid first-pass analysis.
    Required failures remain explicit. Retries/resume retain the caller's budget.
    """
    manager = EvidenceDataManager(data, paper)
    agent = MappingAgent(model=model, tracker=tracker, budget=budget)
    selected_claims = list(dict.fromkeys(claim_ids))
    successful, failures, diagnostics = {}, [], []
    stop_for_budget = False
    run.pop("interruption_reason", None)
    for cid in selected_claims:
        try:
            initial, first_receipt = manager.packet(
                cid, allowed_element_ids=allowed_element_ids, analysis_focus=analysis_focus)
            expanded, expanded_receipt = manager.packet(
                cid, expanded=True, allowed_element_ids=allowed_element_ids,
                analysis_focus=analysis_focus, required_evidence_ids=first_receipt["selected_evidence_ids"])
            packet, receipt = ((expanded, expanded_receipt) if first_receipt["priority_deferred"]
                               else (initial, first_receipt))
        except (ValueError, KeyError) as exc:
            failures.append({"claim_id": cid, "type": "input_packet_error", "message": str(exc)[:500]})
            store.save(data)
            continue
        for view in (expanded, packet):
            key = claim_cache_key(data, cid, model, language, analysis_focus, input_packet=view)
            cached = find_cached_claim(data, cid, key)
            # An explicit refresh is paid once, not again on same-question resume.
            if cached and (not rerun or data.mappings[cached].get("created_run_id") == run["run_id"]):
                successful[cid] = cached
                reused = run.setdefault("reused_mapping_ids", [])
                if cached not in reused:
                    reused.append(cached)
                if on_saved:
                    on_saved(cid, cached)
                print("Reusing Claim Mapping:", cid, cached, flush=True)
                break
        if cid in successful:
            store.save(data)
            continue
        for attempt_number in (1, 2):
            print("Mapping:", cid, "Evidence:", len(packet["evidence_items"]),
                  "Input bytes:", receipt["packet_bytes"], "Expanded:", receipt["expanded"], flush=True)
            attempt = {"claim_id": cid, "attempt": attempt_number, "retrieval": receipt,
                       "purpose": "optional_expansion" if attempt_number == 2 else "required_mapping"}
            run.setdefault("mapping_input_receipts", []).append(attempt)
            store.save(data)
            call_start = perf_counter()
            try:
                result = agent.analyze(mapping_input=packet, language=language)
            except (MODEL_ERRORS + (BudgetStop,)) as exc:
                issue = {"claim_id": cid, "type": "budget_stop" if isinstance(exc, BudgetStop) else type(exc).__name__,
                         "message": str(exc)[:500], "optional_expansion": cid in successful}
                attempt.update({"status": "budget_stopped" if isinstance(exc, BudgetStop) else "model_error", "message": str(exc)[:500]})
                (diagnostics if cid in successful else failures).append(issue)
                if isinstance(exc, BudgetStop):
                    run["interruption_reason"] = "BudgetStop"
                    stop_for_budget = True
                store.save(data)
                break
            finally:
                run["llm_wall_time_seconds"] = run.get("llm_wall_time_seconds", 0.0) + perf_counter() - call_start
            attempt["status"] = "returned"
            saved = save_mapping_analyses(
                data, analyses=result.claim_analyses, target_ids=[cid],
                available_evidence_ids=[item["evidence_id"] for item in packet["evidence_items"]],
                model=model, language=language, run_id=run["run_id"], analysis_focus=analysis_focus,
                context_claim_ids=[cid], input_packets={cid: packet}, retrieval_receipts={cid: receipt})
            mid = saved["mapping_ids_by_claim"].get(cid)
            had_valid_result = cid in successful
            if mid:
                run.setdefault("created_mapping_ids", []).append(mid)
                if data.mappings[mid]["status"] == "completed":
                    successful[cid] = mid
                print("Saved Claim Mapping:", mid, cid, data.mappings[mid]["status"], flush=True)
            if saved["issues"]:
                (diagnostics if had_valid_result else failures).extend(saved["issues"])
            store.save(data)
            rows = [row for row in result.claim_analyses if row.claim_id == cid]
            current_ids = {item["evidence_id"] for item in packet["evidence_items"]}
            expanded_ids = {item["evidence_id"] for item in expanded["evidence_items"]}
            can_expand = (attempt_number == 1 and not receipt["expanded"] and mid is not None
                          and data.mappings[mid]["status"] == "completed" and len(rows) == 1
                          and rows[0].material_status == "limited" and current_ids < expanded_ids)
            if not can_expand:
                break
            packet, receipt = expanded, expanded_receipt
        if cid in successful and on_saved:
            on_saved(cid, successful[cid])
        store.save(data)
        if stop_for_budget:
            break
    run.setdefault("mapping_diagnostics", []).extend(diagnostics)
    return {"mapping_ids_by_claim": successful, "issues": failures, "diagnostics": diagnostics,
            "pending_claim_ids": [cid for cid in selected_claims if cid not in successful]}
