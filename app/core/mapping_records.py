"""Per-Claim persistence and actual-input caching for link-free Mapping.

Historical records remain auditable. Only a matching current policy AND the
actual bounded input packet can authorize reuse. No inference is made from
legacy Links, Evidence ID counts, or a previous run's completed flag.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache

from app.agents.mapping_agent import MAPPING_INSTRUCTIONS
from app.models.reader_v3 import MappingExtraction
from app.core.mapping_views import record_fingerprint
from app.core.evidence_retrieval import (
    EvidenceDataManager, POLICY, available_source_parts, fingerprint, claim_origin_only,
)

MAPPING_RECORD_POLICY = "per_claim_no_links_relations_v4_origin_guard"
RELATIONS = {"supports", "contradicts", "qualifies", "context", "insufficient_support"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _unique(values):
    return list(dict.fromkeys(v for v in values if v))


@lru_cache(maxsize=1)
def mapping_policy_signature():
    return fingerprint({"record_policy": MAPPING_RECORD_POLICY,
                        "retrieval_policy": POLICY,
                        "instructions": MAPPING_INSTRUCTIONS,
                        "schema": MappingExtraction.model_json_schema()})


def claim_cache_key(data, claim_id, model, language, focus="general",
                    dependency_evidence_ids=None, context_claim_ids=None,
                    *, input_packet=None):
    if input_packet is None:
        input_packet, _ = EvidenceDataManager(data).packet(
            claim_id, analysis_focus=focus, context_claim_ids=context_claim_ids or ())
    # Hash what the model actually receives, including coverage limitations.
    # No full Evidence Pool and no Links are embedded or sent for this cache.
    return fingerprint({"policy": mapping_policy_signature(),
                        "paper_revision": data.paper_revision,
                        "model": model, "language": language,
                        "analysis_focus": focus, "input": input_packet})


def claim_source_ids(data, claim_id):
    packet, _ = EvidenceDataManager(data).packet(claim_id)
    return {e["evidence_id"] for e in packet["evidence_items"]}


def legacy_batch_input_is_current(data, record):
    # Historical link-filtered inputs cannot certify the new retrieval contract.
    return False


def next_mapping_id(data):
    occupied = set(data.mappings) | set(data.mapping_batch_archive)
    numbers = [int(mid.rsplit("_", 1)[1]) for mid in occupied
               if mid.startswith("mapping_") and mid.rsplit("_", 1)[1].isdigit()]
    number = max([int(data.id_counters.get("mapping", 0)), *numbers], default=0) + 1
    data.id_counters["mapping"] = number
    return f"mapping_{number:06d}"


def next_relation_id(data):
    numbers = [int(rid.rsplit("_", 1)[1]) for rid in data.mapping_relations
               if rid.startswith("relation_") and rid.rsplit("_", 1)[1].isdigit()]
    number = max([int(data.id_counters.get("relation", 0)), *numbers], default=0) + 1
    data.id_counters["relation"] = number
    return f"relation_{number:06d}"


def find_cached_claim(data, claim_id, cache_key):
    active = data.mapping_by_claim.get(claim_id)
    ordered = _unique([active, *reversed(list(data.mappings))])
    for mid in ordered:
        item = data.mappings.get(mid)
        if not isinstance(item, dict):
            continue
        if (item.get("claim_id") == claim_id and item.get("status") == "completed"
                and item.get("paper_revision") == data.paper_revision
                and item.get("record_policy") == MAPPING_RECORD_POLICY
                and item.get("mapping_policy_signature") == mapping_policy_signature()
                and item.get("input_signature") == cache_key
                and len(item.get("claim_analyses") or []) == 1):
            return mid
    return None


def find_cached_claim_with_context(data, claim_id, model, language, focus,
                                   allowed_evidence_ids, allowed_claim_ids):
    """Compatibility entry point; new execution supplies exact packet keys."""
    packet, _ = EvidenceDataManager(data).packet(claim_id, analysis_focus=focus)
    ids = {item["evidence_id"] for item in packet["evidence_items"]}
    if not ids <= set(allowed_evidence_ids):
        return None
    return find_cached_claim(data, claim_id, claim_cache_key(
        data, claim_id, model, language, focus, input_packet=packet))


def add_relation(data, *, claim_ids, mapping_ids, explanation, run_id,
                 source_mapping_id=None):
    text = str(explanation or "").strip()
    claims, mappings = _unique(claim_ids), _unique(mapping_ids)
    if len(claims) < 2 or not text:
        return None
    key = fingerprint({"claims": claims, "mappings": mappings, "text": text,
                       "revision": data.paper_revision, "legacy": source_mapping_id})
    for rid, item in data.mapping_relations.items():
        if item.get("relation_signature") == key:
            return rid
    rid = next_relation_id(data)
    data.mapping_relations[rid] = {
        "relation_id": rid, "claim_ids": claims, "mapping_ids": mappings,
        "explanation": text, "relation_signature": key,
        "paper_revision": data.paper_revision, "created_run_id": run_id,
        "source_mapping_id": source_mapping_id, "created_at": _now(),
    }
    return rid


def save_mapping_analyses(data, *, analyses, target_ids, available_evidence_ids,
                          model, language, run_id, analysis_focus="general",
                          cross_claim_explanation="", context_claim_ids=None,
                          inherited_record=None, allow_cache=True,
                          cache_context_evidence_ids=None, input_packets=None,
                          retrieval_receipts=None):
    targets = _unique(target_ids)
    available = set(available_evidence_ids) & set(data.evidence_pool)
    rows = [r.model_dump(mode="json") if hasattr(r, "model_dump") else deepcopy(r)
            for r in analyses]
    by_claim, duplicates, unexpected = {}, set(), []
    for row in rows:
        cid = row.get("claim_id")
        if cid not in targets or cid not in data.claim_pool:
            unexpected.append(cid)
        elif cid in by_claim:
            duplicates.add(cid)
        else:
            by_claim[cid] = row
    saved, issues = {}, []
    for cid in targets:
        row = by_claim.get(cid)
        if row is None:
            issues.append({"claim_id": cid, "type": "missing_mapping_analysis"})
            continue
        used = _unique(row.get("evidence_ids_used") or [])
        row["evidence_ids_used"] = used
        relations = row.get("evidence_relations") or []
        relation_ids = [r.get("evidence_id") for r in relations if isinstance(r, dict)]
        errors = []
        invalid = sorted(set(used) - available)
        if invalid:
            errors.append("Evidence IDs outside the actual supplied packet.")
        if cid in duplicates:
            errors.append("Duplicate Claim output.")
        if not str(row.get("explanation") or "").strip():
            errors.append("Empty Mapping explanation.")
        legacy = inherited_record is not None
        if not legacy:
            if len(relation_ids) != len(set(relation_ids)) or set(relation_ids) != set(used):
                errors.append("Every used Evidence must have exactly one explicit relation.")
            if any(not isinstance(r, dict) or r.get("relation") not in RELATIONS
                   or not str(r.get("explanation") or "").strip() for r in relations):
                errors.append("Invalid or unexplained Evidence relation.")
            if row.get("material_status") not in {"sufficient_for_scoped_analysis", "limited"}:
                errors.append("Missing material sufficiency status.")
        packet = (input_packets or {}).get(cid)
        receipt = deepcopy((retrieval_receipts or {}).get(cid) or {})
        if not legacy:
            # Validate provenance only. Never infer a relation from keywords,
            # scientific topic, retrieval score, Evidence ID, or legacy Links.
            visible = {item["evidence_id"]: item
                       for item in (packet or {}).get("evidence_items", [])}
            for edge in relations:
                if not isinstance(edge, dict):
                    continue
                eid = edge.get("evidence_id")
                parts = edge.get("source_parts_used")
                if (not isinstance(parts, list) or not parts
                        or any(not isinstance(part, str) for part in parts)):
                    errors.append(f"Missing or invalid source_parts_used for {eid}.")
                    continue
                if len(parts) != len(set(parts)):
                    errors.append(f"Repeated source component for {eid}.")
                item = visible.get(eid)
                if item is None:
                    errors.append(f"Source components cannot be grounded in the supplied packet for {eid}.")
                    continue
                absent = sorted(set(parts) - set(available_source_parts(item)))
                if absent:
                    errors.append(f"Source components not supplied for {eid}: {', '.join(absent)}.")
                source = data.evidence_pool.get(eid)
                if (not absent and source is not None
                        and edge.get("relation") in {"supports", "contradicts", "qualifies"}
                        and claim_origin_only(data.claim_pool[cid], source,
                                              source_parts=parts, visible_view=item)):
                    # Preserve invalid output for audit. Never silently relabel
                    # a model assertion or rewrite its explanation as a repair.
                    errors.append(f"Exact Claim-origin text is not an additional argument: {eid}.")
        if packet is not None:
            packet_ids = {item["evidence_id"] for item in packet["evidence_items"]}
            if not set(used) <= packet_ids:
                errors.append("Used Evidence was not visible to this Claim.")
            available_for_claim = packet_ids
        else:
            available_for_claim = available
        coverage_limited = bool(receipt.get("priority_deferred") or receipt.get("missing_direct_source_ids"))
        if coverage_limited and not legacy:
            row["material_status"] = "limited"
            notes = list(row.get("qualifications") or [])
            note = "Some priority sources were not supplied; this analysis is limited to the displayed material."
            if note not in notes:
                notes.append(note)
            row["qualifications"] = notes
        status = "completed" if not errors else "partial"
        cacheable = allow_cache and not errors and not legacy and packet is not None
        mid = next_mapping_id(data)
        data.mappings[mid] = {
            "mapping_id": mid, "claim_id": cid, "status": status,
            "record_policy": "legacy_preserved" if legacy else MAPPING_RECORD_POLICY,
            "paper_revision": data.paper_revision, "model": model, "language": language,
            "analysis_focus": analysis_focus or "general",
            "input_signature": claim_cache_key(data, cid, model, language, analysis_focus,
                                                input_packet=packet) if cacheable else None,
            "mapping_policy_signature": mapping_policy_signature() if not legacy else None,
            "target_claim_ids": [cid], "context_claim_ids": _unique(context_claim_ids or [cid]),
            "available_evidence_ids": sorted(available_for_claim), "claim_analyses": [row],
            "missing_claim_ids": [], "unexpected_claim_ids": [],
            "unresolved_evidence_ids": invalid, "validation_issues": errors,
            "created_run_id": run_id, "created_at": (inherited_record or {}).get("created_at") or _now(),
            "migrated_from_mapping_id": (inherited_record or {}).get("mapping_id"),
            "cache_scope": "exact_bounded_packet" if cacheable else "not_reusable",
            "cache_dependency_evidence_ids": sorted(available_for_claim),
            "cache_dependency_claim_ids": [x for x in (context_claim_ids or []) if x != cid],
            "retrieval_receipt": receipt,
            "claim_snapshot_fingerprint": record_fingerprint(data.claim_pool[cid]),
            "used_evidence_fingerprints": {
                eid: record_fingerprint(data.evidence_pool[eid])
                for eid in used if eid in data.evidence_pool
            },
        }
        if status == "completed":
            data.mapping_by_claim[cid] = mid
        saved[cid] = mid
        issues.extend({"claim_id": cid, "type": "mapping_contract", "message": error} for error in errors)
    relation_id = add_relation(data, claim_ids=[cid for cid in targets if cid in saved],
                              mapping_ids=list(saved.values()), explanation=cross_claim_explanation,
                              run_id=run_id, source_mapping_id=(inherited_record or {}).get("mapping_id"))
    if unexpected:
        issues.append({"type": "unexpected_claims", "claim_ids": unexpected})
    return {"mapping_ids_by_claim": saved, "relation_id": relation_id, "issues": issues,
            "missing_claim_ids": [cid for cid in targets if cid not in saved]}


def migrate_grouped_mappings(data):
    """Preserve legacy batches losslessly; never promote them to new caches."""
    moved = []
    for old_id, item in list(data.mappings.items()):
        if not isinstance(item, dict) or len(item.get("target_claim_ids") or []) < 2:
            continue
        data.mapping_batch_archive[old_id] = deepcopy(item)
        del data.mappings[old_id]
        result = save_mapping_analyses(
            data, analyses=item.get("claim_analyses") or [],
            target_ids=[cid for cid in item.get("target_claim_ids", []) if cid in data.claim_pool],
            available_evidence_ids=item.get("available_evidence_ids") or list(data.evidence_pool),
            model=item.get("model", "unknown"), language=item.get("language", "en"),
            run_id=item.get("created_run_id", "legacy"), inherited_record=item, allow_cache=False,
            cross_claim_explanation=item.get("cross_claim_explanation", ""))
        moved.append({"batch_mapping_id": old_id, **result})
    return moved


def completed_mapping_evidence(data, mapping_ids):
    """Authoritative final-display IDs: never fall back to Links or candidates."""
    ids = []
    for mid in _unique(mapping_ids):
        record = data.mappings.get(mid) or {}
        if record.get("status") != "completed" or record.get("paper_revision") != data.paper_revision:
            continue
        for analysis in record.get("claim_analyses") or []:
            ids.extend(eid for eid in analysis.get("evidence_ids_used") or [] if eid in data.evidence_pool)
    return _unique(ids)
