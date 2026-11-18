"""Discard retired candidate associations at Reader-v3 data boundaries.

This module is deliberately separate from Mapping relations and document
references. It has no I/O and never rewrites scientific source records, Mapping
judgments, budget IDs, or completed scan hashes.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

SOURCE_SCAN_POLICY = "source_scan_v1"
_OLD_SCAN_POLICY = "linked_source_reuse_v1"
_RETIRED_FIELDS = frozenset({
    "links", "claim_evidence_links", "new_links", "unresolved_links",
    "superseded_links", "link_generation_enabled", "existing_pool_matches",
    "link_count", "links_count", "link_ids",
    "mapping_relations", "mapping_batch_archive", "cross_claim_explanation",
})
# These are scientific source/results, not candidate-association checkpoints.
_PRESERVED_POOLS = frozenset({
    "claim_pool", "evidence_pool", "mappings",
})


def scan_policy(record: dict[str, Any]) -> Any:
    """Read an old cache identity without reviving its retired relation data."""
    value = record.get("scan_cache_policy", record.get("linking_policy"))
    return SOURCE_SCAN_POLICY if value == _OLD_SCAN_POLICY else value


def scan_hash_compatibility_fields() -> dict[str, str]:
    """Only used INSIDE a SHA input; never persist or expose this dictionary.

    Existing text/image scan hashes used this literal key/value. Keeping the
    hash preimage avoids invalidating already-paid source scans. It does not
    enable, reconstruct, or select any Claim/Evidence candidate associations.
    """
    return {"linking_policy": _OLD_SCAN_POLICY}


def clean_checkpoint(value: Any, *, removed: list[str] | None = None,
                     path: str = "$") -> Any:
    """Clean application metadata only, preserving text and unrelated fields."""
    if isinstance(value, list):
        return [clean_checkpoint(item, removed=removed, path=f"{path}[{i}]")
                for i, item in enumerate(value)]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        location = f"{path}.{key}"
        if key in _PRESERVED_POOLS:
            result[key] = item
        elif key in _RETIRED_FIELDS:
            if removed is not None:
                removed.append(location)
        elif key == "linking_policy":
            if removed is not None:
                removed.append(location)
            if "scan_cache_policy" not in value:
                result["scan_cache_policy"] = scan_policy(value)
        else:
            result[key] = clean_checkpoint(item, removed=removed, path=location)
    return result


def clean_analysis_payload(payload: dict[str, Any], *,
                           removed: list[str] | None = None) -> dict[str, Any]:
    """Return an association-free analysis without modifying the caller's data.

    This is not retention, scientific reanalysis, or a budget reset. Claim and
    Evidence values, Mapping relations/edges and all accumulated cost numbers
    are retained. Old format fields are accepted solely to be discarded.
    """
    if not isinstance(payload, dict):
        raise TypeError("Reader analysis must be a JSON object.")
    # Never discard scientific results or historical batches implicitly.
    for field in ("mapping_relations", "mapping_batch_archive"):
        if payload.get(field):
            raise ValueError(
                f"Nonempty retired Mapping pool: {field}. "
                "Migrate this file separately before saving."
            )

    result = clean_checkpoint(payload, removed=removed)

    mappings = result.get("mappings")

    if mappings is None:
        return result

    if not isinstance(mappings, dict):
        raise ValueError("Mapping storage must be an object.")

    normalized = {}

    for mid, original in mappings.items():
        if not isinstance(original, dict):
            raise ValueError(f"Invalid Mapping record: {mid}")

        record = deepcopy(original)
        cid = record.get("claim_id")
        targets = record.get("target_claim_ids") or []

        if len(targets) > 1 or (targets and targets != [cid]):
            raise ValueError(
                f"Grouped Mapping requires migration: {mid}"
            )

        historical = record.pop("claim_analyses", None)
        singular = record.get("claim_analysis")

        if historical is not None:
            if (
                not isinstance(historical, list)
                or len(historical) != 1
                or not isinstance(historical[0], dict)
                or historical[0].get("claim_id") != cid
                or (
                    singular is not None
                    and historical[0] != singular
                )
            ):
                raise ValueError(
                    f"Unconvertible historical Mapping: {mid}"
                )

            record["claim_analysis"] = deepcopy(
                historical[0]
            )

        analysis = record.get("claim_analysis")

        if analysis is not None and (
            not isinstance(analysis, dict)
            or analysis.get("claim_id") != cid
        ):
            raise ValueError(
                f"Invalid single-Claim Mapping: {mid}"
            )

        if (
            analysis is None
            and record.get("status") == "completed"
        ):
            raise ValueError(
                f"Completed Mapping has no analysis: {mid}"
            )

        record.pop("cross_claim_explanation", None)
        record.pop("context_claim_ids", None)

        normalized[mid] = record

    result["mappings"] = normalized

    return result
