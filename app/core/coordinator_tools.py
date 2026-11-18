"""Read-only Coordinator utilities for independent Claim Mappings."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

from app.core.mapping_views import mapping_row

SOURCE_ID_PATTERN = re.compile('\\b(?:claim|evidence|mapping)_\\d{6}\\b', re.I)


def _unique(values):
    return list(dict.fromkeys(
        value for value in values if value
    ))


def load_analysis(analysis_path: str | Path) -> dict[str, Any]:
    """Read an analysis without modifying it or calling a model."""
    with Path(analysis_path).open(
        "r", encoding="utf-8-sig"
    ) as handle:
        data = json.load(handle)

    if not isinstance(data, dict) or not data.get("paper_id"):
        raise ValueError(
            "The analysis must be an object with paper_id."
        )

    return data


def inspect_analysis(analysis_path: str | Path) -> dict[str, Any]:
    data = load_analysis(analysis_path)

    claims = data.get("claim_pool") or {}
    evidence = data.get("evidence_pool") or {}
    mappings = data.get("mappings") or {}

    latest_budget = next(
        (
            run["budget_report"]
            for run in reversed(data.get("runs") or [])
            if isinstance(run, dict)
            and run.get("budget_report")
        ),
        None,
    )

    return {
        "paper_id": data["paper_id"],
        "paper_revision": data.get("paper_revision"),
        "claim_count": len(claims),
        "evidence_count": len(evidence),
        "mapping_count": len(mappings),
        "active_mapping_by_claim": deepcopy(
            data.get("mapping_by_claim") or {}
        ),
        "claim_element_ids": sorted({
            item.get("source_element_id")
            for item in claims.values()
            if isinstance(item, dict)
            and item.get("source_element_id")
        }),
        "completed_claim_windows": [
            key
            for key, value in (
                data.get("completed_claim_windows") or {}
            ).items()
            if value == "completed"
        ],
        "evidence_coverage": {
            cid: {
                key: deepcopy(row.get(key))
                for key in (
                    "status",
                    "search_scope",
                    "text_scan_complete",
                    "image_scan_complete",
                    "paper_revision",
                )
            }
            for cid, row in (
                data.get("evidence_search_by_claim") or {}
            ).items()
            if isinstance(row, dict)
        },
        "mapping_index": {
            mid: {
                "status": rec.get("status"),
                "claim_id": rec.get("claim_id"),
                "analysis_focus": rec.get("analysis_focus"),
                "paper_revision": rec.get("paper_revision"),
            }
            for mid, rec in mappings.items()
            if isinstance(rec, dict)
        },
        "latest_budget": deepcopy(latest_budget),
        "section_coverage_requires_paper_index": True,
    }


def build_answer_pack(
    analysis_path: str | Path,
    *,
    question: str,
    claim_ids: list[str] | None = None,
    evidence_ids: list[str] | None = None,
    mapping_ids: list[str] | None = None,
    include_linked_evidence: bool = False,
) -> dict[str, Any]:
    """Select saved sources without using retired Links."""
    data = load_analysis(analysis_path)

    claims = data.get("claim_pool") or {}
    evidence = data.get("evidence_pool") or {}
    mappings = data.get("mappings") or {}

    wanted_claims = _unique(claim_ids or [])
    wanted_evidence = _unique(evidence_ids or [])

    selected_mappings = {}
    warnings = []

    for mid in _unique(mapping_ids or []):
        record = mappings.get(mid)

        if not isinstance(record, dict):
            warnings.append(
                f"Mapping not found: {mid}"
            )
            continue

        cid = record.get("claim_id")
        row = mapping_row(record, cid)

        if (
            record.get("status") != "completed"
            or record.get("paper_revision")
               != data.get("paper_revision")
            or row is None
        ):
            warnings.append(
                f"Mapping is not a usable single-Claim result: {mid}"
            )
            continue

        selected_mappings[mid] = {
            "mapping_id": mid,
            "claim_id": cid,
            "status": "completed",
            "claim_analysis": deepcopy(row),
        }

        wanted_claims.append(cid)

        wanted_evidence.extend(
            row.get("evidence_ids_used") or []
        )

    selected_claims = {}

    for cid in _unique(wanted_claims):
        if cid in claims:
            selected_claims[cid] = deepcopy(claims[cid])
        else:
            warnings.append(
                f"Claim not found: {cid}"
            )

    selected_evidence = {}

    for eid in _unique(wanted_evidence):
        if eid not in evidence:
            warnings.append(
                f"Evidence not found: {eid}"
            )
            continue

        rec = deepcopy(evidence[eid])
        selected_evidence[eid] = rec

        if rec.get("source_status") == "unlocated":
            warnings.append(
                f"Unlocated evidence retained: {eid}"
            )

        if (
            rec.get("observed_via") == "image_input"
            and (
                rec.get("source_content") or {}
            ).get("visual_reading_verified") is False
        ):
            warnings.append(
                f"Unverified visual reading: {eid}"
            )

    index = {
        **{
            cid: {
                "kind": "claim",
                "source_element_id": row.get(
                    "source_element_id"
                ),
            }
            for cid, row in selected_claims.items()
        },
        **{
            eid: {
                "kind": "evidence",
                "source_element_id": row.get(
                    "source_element_id"
                ),
            }
            for eid, row in selected_evidence.items()
        },
        **{
            mid: {
                "kind": "mapping",
                "source_status": "analysis",
            }
            for mid in selected_mappings
        },
    }

    return {
        "question": question,
        "paper_id": data["paper_id"],
        "paper_revision": data.get("paper_revision"),
        "claims": selected_claims,
        "evidence": selected_evidence,
        "mappings": selected_mappings,
        "source_index": index,
        "warnings": _unique(warnings),
        "answer_rules": {
            "use_provided_sources_only": True,
            "preserve_numbers_and_units": True,
            "preserve_uncertainty": True,
            "no_new_scientific_reasoning": True,
        },
    }


def check_answer_references(
    answer: str,
    answer_pack: dict[str, Any],
) -> dict[str, Any]:
    mentioned = {
        match.lower()
        for match in SOURCE_ID_PATTERN.findall(
            str(answer or "")
        )
    }

    known = {
        source_id.lower()
        for source_id in (
            answer_pack.get("source_index") or {}
        )
    }

    return {
        "mentioned_ids": sorted(mentioned),
        "unknown_ids": sorted(mentioned - known),
        "references_recognized": bool(mentioned),
    }


def render_source_fallback(
    answer_pack: dict[str, Any],
) -> str:
    """Render saved Mapping explanations without new judgments."""
    lines = []

    for mid, record in (
        answer_pack.get("mappings") or {}
    ).items():
        cid = record.get("claim_id")
        row = mapping_row(record, cid)

        if row is None:
            continue

        lines.extend([
            f"### Mapping {mid}",
            f"\\n**{cid}**",
            row.get("explanation")
            or "No analysis text is available.",
        ])

        for edge in row.get("evidence_relations") or []:
            if isinstance(edge, dict):
                lines.append(
                    f"- {edge.get('evidence_id')}: "
                    f"{edge.get('relation')} - "
                    f"{edge.get('explanation') or ''}"
                )

        for note in row.get("qualifications") or []:
            lines.append(
                f"- Qualification: {note}"
            )

        for note in row.get("evidence_gaps") or []:
            lines.append(
                f"- Historical analysis note: {note}"
            )

    for cid, record in (
        answer_pack.get("claims") or {}
    ).items():
        quote = (
            record.get("source_quote")
            or record.get("model_quote")
            or ""
        )

        lines.append(
            f"- Claim {cid}: {quote}"
        )

    for eid, record in (
        answer_pack.get("evidence") or {}
    ).items():
        content = record.get("source_content") or {}

        quote = (
            record.get("source_quote")
            or content.get("passage")
            or content.get("visual_observation")
            or content.get("caption")
            or ""
        )

        uncertainty = (
            " (visual reading not independently verified)"
            if (
                record.get("observed_via") == "image_input"
                and content.get(
                    "visual_reading_verified"
                ) is False
            )
            else ""
        )

        lines.append(
            f"- Evidence {eid}{uncertainty}: {quote}"
        )

    return "\\n".join(lines) if lines else (
        "No selected analysis records are available "
        "for a source-grounded answer."
    )


def build_compact_answer_pack(
    full_pack: dict[str, Any],
    *,
    max_direct_evidence: int = 12,
) -> dict[str, Any]:
    """Retain all adopted Evidence; bound direct-only selection."""
    compact = deepcopy(full_pack)

    mappings = compact.get("mappings") or {}
    evidence = compact.get("evidence") or {}

    if mappings:
        adopted = {
            eid
            for record in mappings.values()
            for eid in (
                record.get("claim_analysis") or {}
            ).get("evidence_ids_used", [])
        }

        compact["evidence"] = {
            eid: item
            for eid, item in evidence.items()
            if eid in adopted
        }

    else:
        compact["evidence"] = dict(
            list(evidence.items())[
                :max(0, max_direct_evidence)
            ]
        )

    retained = (
        set(compact.get("claims") or {})
        | set(compact.get("evidence") or {})
        | set(mappings)
    )

    compact["source_index"] = {
        key: value
        for key, value in (
            compact.get("source_index") or {}
        ).items()
        if key in retained
    }

    return compact
