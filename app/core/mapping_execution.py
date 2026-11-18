# MAPPING_MODEL_INTEGRATION_V1
"""Per-Claim Mapping with bounded repair and source-grounded numeric rendering."""
from __future__ import annotations

from copy import deepcopy
from time import perf_counter
from openai import APIConnectionError, APIStatusError, APITimeoutError
from app.agents.mapping_agent import (
    MappingAgent, resolve_mapping_model, MAPPING_REASONING_EFFORT,
)
from app.budget import BudgetStop
from app.llm import StructuredOutputParseError
from app.core.evidence_retrieval import EvidenceDataManager
from app.core.mapping_records import (
    claim_cache_key, find_cached_claim, save_mapping_analysis,
    mapping_policy_signature,
)
from app.core.mapping_grounding import ANALYSIS_POLICY as POLICY, digest, prepare_tables, validate_and_render

MODEL_ERRORS = (StructuredOutputParseError, APIConnectionError, APIStatusError, APITimeoutError)


def run_claim_mappings(*, data, store, run, claim_ids, model, language, tracker, budget,
                       paper=None, analysis_focus="general", allowed_element_ids=None,
                       rerun=False, on_saved=None):
    model = resolve_mapping_model(model)
    saved_model = run.get("mapping_model")
    if saved_model is not None and resolve_mapping_model(saved_model) != model:
        raise ValueError("Mapping execution model differs from the saved run.")
    run["mapping_model"] = model
    run["mapping_reasoning_effort"] = MAPPING_REASONING_EFFORT
    targets = list(dict.fromkeys(claim_ids))
    successful, failures, diagnostics = {}, [], []
    agent = MappingAgent(model=model, tracker=tracker, budget=budget)
    run.pop("interruption_reason", None)
    added = prepare_tables(data, paper, targets, run["run_id"], allowed_element_ids)
    if added:
        print("Registered original comparison-table scopes:", ", ".join(added), flush=True)
        store.save(data)
    # Reuse saved visual Evidence; no automatic second image-reading call.
    manager = EvidenceDataManager(data, paper)
    counts = run.setdefault("mapping_grounding_attempts", {})
    for cid in targets:
        try:
            initial, first_receipt = manager.packet(cid, analysis_focus=analysis_focus,
                                                   allowed_element_ids=allowed_element_ids)
            expanded, expanded_receipt = manager.packet(cid, expanded=True, analysis_focus=analysis_focus,
                allowed_element_ids=allowed_element_ids, required_evidence_ids=first_receipt["selected_evidence_ids"])
            packet, receipt = ((expanded, expanded_receipt) if first_receipt["priority_deferred"]
                               else (initial, first_receipt))
        except (ValueError, KeyError) as exc:
            failures.append({"claim_id": cid, "type": "input_packet_error", "message": str(exc)[:500]})
            continue
        for view in (expanded, packet):
            key = claim_cache_key(data, cid, model, language, analysis_focus, input_packet=view)
            mid = find_cached_claim(data, cid, key)
            if mid and (not rerun or data.mappings[mid].get("created_run_id") == run["run_id"]):
                successful[cid] = mid
                run.setdefault("reused_mapping_ids", []).append(mid)
                print("Reusing grounded Mapping:", cid, mid, flush=True)
                break
        if cid in successful:
            if on_saved:
                on_saved(cid, successful[cid])
            store.save(data)
            continue
        attempt_key = digest({"policy": POLICY, "claim": cid, "model": model,
                              "mapping_policy_signature": mapping_policy_signature(),
                              "language": language, "focus": analysis_focus,
                              "initial": packet, "expanded": expanded})
        checkpoint = counts.setdefault(attempt_key, {"calls": 0, "errors": [], "claim_id": cid})
        accepted = None
        while checkpoint["calls"] < 2 and not checkpoint.get("terminal_error"):
            sent = deepcopy(packet)
            if checkpoint["errors"]:
                sent["correction_feedback"] = checkpoint["errors"][:8]
            checkpoint["calls"] += 1
            store.save(data)
            print("Mapping:", cid, "attempt:", checkpoint["calls"],
                  "Evidence:", len(packet["evidence_items"]),
                  "Input bytes:", receipt["packet_bytes"], flush=True)
            start = perf_counter()
            try:
                result = agent.analyze(sent, language=language)
            except BudgetStop as exc:
                checkpoint["calls"] -= 1
                run["interruption_reason"] = "BudgetStop"
                checkpoint["errors"] = [str(exc)[:500]]
                break
            except (MODEL_ERRORS + (RuntimeError,)) as exc:
                if getattr(exc, "request_not_sent", False):
                    checkpoint["calls"] -= 1
                checkpoint["errors"] = [type(exc).__name__ + ": " + str(exc)[:400]]
                receipt = getattr(exc, "response_receipt", None) or {}
                if (isinstance(exc, StructuredOutputParseError)
                        and receipt.get("status") == "incomplete"):
                    checkpoint["terminal_error"] = "MappingOutputIncomplete"
                    checkpoint["response_id"] = receipt.get("id")
                    run["interruption_reason"] = "MappingOutputIncomplete"
                    break
                # Only a completed response with a reconciled usage receipt may
                # use the existing one-repair allowance. Never retry a timeout,
                # connection failure, incomplete output, or unknown charge.
                if (isinstance(exc, StructuredOutputParseError)
                        and receipt.get("status") == "completed"
                        and receipt.get("usage") is not None
                        and not budget.report()["blocked"]):
                    continue
                run["interruption_reason"] = "MappingRequestError"
                break
            finally:
                run["llm_wall_time_seconds"] = run.get("llm_wall_time_seconds", 0.0) + perf_counter() - start
            rows, errors = validate_and_render(result, packet)

            if len(rows) != 1 or rows[0].get("claim_id") != cid:
                errors.append(
                    "Mapping must return exactly one analysis for its target Claim."
                )

            if errors:
                checkpoint["errors"] = [e[:350] for e in errors]
                store.save(data)
                continue
            accepted = (rows, packet, receipt)
            checkpoint["errors"] = []
            new_ids = set(expanded_receipt["selected_evidence_ids"]) - set(receipt["selected_evidence_ids"])
            break
        if accepted is None:
            if checkpoint.get("terminal_error"):
                run["interruption_reason"] = checkpoint["terminal_error"]
            failures.append({"claim_id": cid, "type": "mapping_not_validated",
                             "message": "; ".join(checkpoint["errors"])[:1200] or
                             "This task used its two Mapping attempts. No unvalidated Mapping was saved."})
        else:
            rows, selected_packet, selected_receipt = accepted
            saved = save_mapping_analysis(
                data,
                analysis=rows[0],
                target_claim_id=cid,
                available_evidence_ids=selected_receipt["selected_evidence_ids"],
                model=model,
                language=language,
                run_id=run["run_id"],
                analysis_focus=analysis_focus,
                input_packet=selected_packet,
                retrieval_receipt=selected_receipt,
            )
            mid = saved["mapping_ids_by_claim"].get(cid)
            if mid and data.mappings[mid]["status"] == "completed":
                successful[cid] = mid
                if mid not in run.setdefault("created_mapping_ids", []):
                    run["created_mapping_ids"].append(mid)
                if on_saved:
                    on_saved(cid, mid)
                print("Saved grounded Mapping:", cid, mid, flush=True)
                if checkpoint["errors"]:
                    diagnostics.append({"claim_id": cid, "type": "optional_refinement_failed",
                                        "message": "; ".join(checkpoint["errors"])[:800]})
            else:
                failures.extend(saved["issues"] or [{"claim_id": cid, "type": "mapping_save_failed"}])
        store.save(data)
        if run.get("interruption_reason"):
            break
    run.setdefault("mapping_diagnostics", []).extend(diagnostics)
    return {"mapping_ids_by_claim": successful, "issues": failures, "diagnostics": diagnostics,
            "pending_claim_ids": [cid for cid in targets if cid not in successful]}
