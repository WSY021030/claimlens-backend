import json
import re

from pathlib import Path
from typing import Any

from app.core.evidence_integrity import LINKING_POLICY


SOURCE_ID_PATTERN = re.compile(
    r"\b(?:claim|evidence|mapping|relation)_\d{6}\b",
    re.IGNORECASE,
)


def _unique(values: list[Any]) -> list[Any]:
    """Remove duplicates while preserving input order."""
    return list(dict.fromkeys(
        value for value in values if value
    ))


def load_analysis(
    analysis_path: str | Path,
) -> dict[str, Any]:
    """
    Load an existing analysis document.

    This function is read-only and does not call an LLM.
    """
    path = Path(analysis_path)

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "The analysis file must contain a JSON object."
        )

    if not data.get("paper_id"):
        raise ValueError(
            "The analysis file is missing paper_id."
        )

    return data


def inspect_analysis(
    analysis_path: str | Path,
) -> dict[str, Any]:
    """
    Build a lightweight catalog of existing analysis data.

    Data existence, search coverage, and answerability
    must be evaluated separately.
    """
    data = load_analysis(analysis_path)

    claims = data.get("claim_pool") or {}
    evidence = data.get("evidence_pool") or {}
    mappings = data.get("mappings") or {}
    mapping_relations = data.get("mapping_relations") or {}
    runs = data.get("runs") or []

    mapping_index = {}

    for mapping_id, record in mappings.items():
        mapping_index[mapping_id] = {
            "status": record.get("status"),
            "target_claim_ids": record.get(
                "target_claim_ids", []
            ),
            "analysis_focus": record.get(
                "analysis_focus"
            ),
            "paper_revision": record.get(
                "paper_revision"
            ),
        }

    evidence_coverage = {}

    for claim_id, record in (
        data.get("evidence_search_by_claim") or {}
    ).items():
        evidence_coverage[claim_id] = {
            "status": (record.get("status")
                       if record.get("linking_policy") == LINKING_POLICY
                       else "needs_relinking"),
            "search_scope": record.get("search_scope"),
            "text_scan_complete": record.get(
                "text_scan_complete", False
            ),
            "image_scan_complete": record.get(
                "image_scan_complete", False
            ),
            "paper_revision": record.get(
                "paper_revision"
            ),
        }

    latest_budget = None

    for run in reversed(runs):
        if run.get("budget_report"):
            latest_budget = run["budget_report"]
            break

    completed_windows = [
        window
        for window, status in (
            data.get("completed_claim_windows") or {}
        ).items()
        if status == "completed"
    ]

    return {
        "paper_id": data["paper_id"],
        "paper_revision": data.get("paper_revision"),

        "claim_count": len(claims),
        "evidence_count": len(evidence),
        "mapping_count": len(mappings),
        "active_mapping_by_claim": data.get("mapping_by_claim") or {},
        "mapping_relation_count": len(mapping_relations),
        "archived_mapping_batch_count": len(data.get("mapping_batch_archive") or {}),

        "claim_element_ids": sorted({
            record.get("source_element_id")
            for record in claims.values()
            if record.get("source_element_id")
        }),

        "completed_claim_windows": completed_windows,

        "evidence_coverage": evidence_coverage,
        "mapping_index": mapping_index,

        "latest_budget": latest_budget,

        # Section-level completeness requires the document index.
        "section_coverage_requires_paper_index": True,
    }


def build_answer_pack(
    analysis_path: str | Path,
    *,
    question: str,
    claim_ids: list[str] | None = None,
    evidence_ids: list[str] | None = None,
    mapping_ids: list[str] | None = None,
    include_linked_evidence: bool = True,
) -> dict[str, Any]:
    """
    Assemble existing records for a Coordinator response.

    This function does not retrieve new document content,
    generate evidence, or perform scientific reasoning.

    Missing records produce warnings without discarding
    the remaining valid material.
    """
    data = load_analysis(analysis_path)

    claim_pool = data.get("claim_pool") or {}
    evidence_pool = data.get("evidence_pool") or {}
    mapping_pool = data.get("mappings") or {}
    archive = data.get("mapping_batch_archive") or {}
    relation_pool = data.get("mapping_relations") or {}
    all_links = [link for link in (data.get("claim_evidence_links") or [])
                 if (data.get("evidence_search_by_claim") or {}).get(
                     link.get("claim_id"), {}).get("linking_policy")
                     == LINKING_POLICY]

    warnings = []

    wanted_claims = list(claim_ids or [])
    wanted_evidence = list(evidence_ids or [])
    wanted_mappings = _unique(mapping_ids or [])

    selected_mappings = {}

    for mapping_id in wanted_mappings:
        record = mapping_pool.get(mapping_id) or archive.get(mapping_id)

        if record is None:
            warnings.append(
                f"Mapping not found: {mapping_id}"
            )
            continue

        selected_mappings[mapping_id] = record

        wanted_claims.extend(
            record.get("target_claim_ids") or []
        )

        for analysis in (
            record.get("claim_analyses") or []
        ):
            wanted_evidence.extend(
                analysis.get("evidence_ids_used") or []
            )

        mapping_revision = record.get(
            "paper_revision"
        )

        if (
            mapping_revision is not None
            and mapping_revision
            != data.get("paper_revision")
        ):
            warnings.append(
                f"Mapping revision mismatch: {mapping_id}"
            )

    wanted_claims = _unique(wanted_claims)

    selected_claims = {}

    for claim_id in wanted_claims:
        record = claim_pool.get(claim_id)

        if record is None:
            warnings.append(
                f"Claim not found: {claim_id}"
            )
            continue

        selected_claims[claim_id] = record

    # Linked evidence is reused, never searched again here.
    if include_linked_evidence:
        for link in all_links:
            if (
                link.get("claim_id")
                in selected_claims
            ):
                wanted_evidence.append(
                    link.get("evidence_id")
                )

    wanted_evidence = _unique(wanted_evidence)

    selected_evidence = {}

    for evidence_id in wanted_evidence:
        record = evidence_pool.get(evidence_id)

        if record is None:
            warnings.append(
                f"Evidence not found: {evidence_id}"
            )
            continue

        selected_evidence[evidence_id] = record

        if (
            record.get("source_status")
            == "unlocated"
        ):
            warnings.append(
                f"Unlocated evidence retained: {evidence_id}"
            )

        content = record.get("source_content") or {}

        if (
            record.get("observed_via") == "image_input"
            and content.get(
                "visual_reading_verified"
            ) is False
        ):
            warnings.append(
                f"Unverified visual reading: {evidence_id}"
            )

    selected_links = [
        link
        for link in all_links
        if (
            link.get("claim_id")
            in selected_claims
            and link.get("evidence_id")
            in selected_evidence
        )
    ]

    source_index = {}

    for claim_id, record in selected_claims.items():
        source_index[claim_id] = {
            "kind": "claim",
            "source_element_id": record.get(
                "source_element_id"
            ),
            "source_status": record.get(
                "source_status"
            ),
        }

    for evidence_id, record in selected_evidence.items():
        content = record.get("source_content") or {}

        source_index[evidence_id] = {
            "kind": "evidence",
            "source_element_id": record.get(
                "source_element_id"
            ),
            "source_status": record.get(
                "source_status"
            ),
            "observed_via": record.get(
                "observed_via"
            ),
            "visual_reading_verified": content.get(
                "visual_reading_verified"
            ),
        }

    selected_relations = {
        rid: record for rid, record in relation_pool.items()
        if (set(record.get("claim_ids") or []) <= set(selected_claims)
            and set(record.get("mapping_ids") or []) <= set(selected_mappings)
            and record.get("paper_revision") == data.get("paper_revision"))
    }
    for mapping_id in selected_mappings:
        source_index[mapping_id] = {
            "kind": "mapping", "source_status": "analysis",
        }
    for relation_id in selected_relations:
        source_index[relation_id] = {
            "kind": "mapping_relation", "source_status": "analysis",
        }

    return {
        "question": question,
        "paper_id": data["paper_id"],
        "paper_revision": data.get(
            "paper_revision"
        ),

        "claims": selected_claims,
        "evidence": selected_evidence,
        "links": selected_links,
        "mappings": selected_mappings,
        "mapping_relations": selected_relations,

        "source_index": source_index,
        "warnings": _unique(warnings),

        "answer_rules": {
            "use_provided_sources_only": True,
            "preserve_meaning": True,
            "preserve_numbers_and_units": True,
            "preserve_uncertainty": True,
            "no_new_scientific_reasoning": True,
        },
    }


def check_answer_references(
    answer: str,
    answer_pack: dict[str, Any],
) -> dict[str, Any]:
    """
    Perform a non-blocking source ID check.

    The check accepts different punctuation and
    surrounding formatting.

    It does not claim to verify semantic faithfulness
    or numerical correctness.
    """
    mentioned = {
        match.lower()
        for match in SOURCE_ID_PATTERN.findall(answer)
    }

    known = {
        source_id.lower()
        for source_id in (
            answer_pack.get("source_index") or {}
        )
    }

    return {
        "mentioned_ids": sorted(mentioned),
        "unknown_ids": sorted(
            mentioned - known
        ),
        "references_recognized": bool(mentioned),
    }


def render_source_fallback(
    answer_pack: dict[str, Any],
) -> str:
    """
    Render existing analysis content without an LLM.

    Preserve available results even when some records
    are missing, unlocated, or not independently verified.
    """
    lines = []

    mappings = answer_pack.get("mappings") or {}
    claims = answer_pack.get("claims") or {}
    evidence = answer_pack.get("evidence") or {}
    relations = answer_pack.get("mapping_relations") or {}

    for mapping_id, record in mappings.items():
        lines.append(f"### Mapping {mapping_id}")

        for analysis in (
            record.get("claim_analyses") or []
        ):
            claim_id = analysis.get(
                "claim_id", "unknown"
            )

            lines.append(
                f"\n**{claim_id}**"
            )

            explanation = (
                analysis.get("explanation")
                or "No analysis text is available."
            )

            lines.append(explanation)

            for note in (
                analysis.get("qualifications") or []
            ):
                lines.append(
                    f"- Qualification: {note}"
                )

            # Read legacy data without treating gaps
            # as requests for additional retrieval.
            for note in (
                analysis.get("evidence_gaps") or []
            ):
                lines.append(
                    f"- Historical analysis note: {note}"
                )

        cross_claim = record.get(
            "cross_claim_explanation"
        )

        if cross_claim:
            lines.append(
                "\n**Cross-claim analysis**"
            )
            lines.append(cross_claim)

    if relations:
        lines.append("\n### Cross-Claim Relationships")
        for relation_id, relation in relations.items():
            lines.append(
                f"- **{relation_id}** ({', '.join(relation.get('claim_ids') or [])}): "
                f"{relation.get('explanation') or ''}"
            )

    if claims:
        lines.append("\n### Existing Claims")

        for claim_id, record in claims.items():
            quote = (
                record.get("source_quote")
                or record.get("model_quote")
                or "No excerpt is available."
            )

            status = record.get(
                "source_status", "unknown"
            )

            lines.append(
                f"- **{claim_id}** [{status}]: {quote}"
            )

    if evidence:
        lines.append("\n### Existing Evidence")

        for evidence_id, record in evidence.items():
            content = (
                record.get("source_content") or {}
            )

            excerpt = (
                record.get("source_quote")
                or content.get("passage")
                or content.get("visual_observation")
                or content.get("caption")
                or "No verified excerpt is available."
            )

            status = record.get(
                "source_status", "unknown"
            )

            observed_via = record.get(
                "observed_via"
            )

            notes = []

            if status == "unlocated":
                notes.append(
                    "source location not confirmed"
                )

            if (
                observed_via == "image_input"
                and content.get(
                    "visual_reading_verified"
                ) is False
            ):
                notes.append(
                    "visual reading not independently verified"
                )

            note_text = (
                f" ({'; '.join(notes)})"
                if notes else ""
            )

            lines.append(
                f"- **{evidence_id}** "
                f"[{status}]{note_text}: {excerpt}"
            )

    if not mappings and not claims and not evidence:
        return (
            "No selected analysis records are available. "
            "A source-grounded answer cannot be generated "
            "from the current input."
        )

    return "\n".join(lines)

def build_compact_answer_pack(
    full_pack: dict[str, Any],
    *,
    max_direct_evidence: int = 12,
) -> dict[str, Any]:
    """
    Build a compact, source-grounded response package.

    Preserve Mapping explanations and important source
    content without copying full analysis records.

    This function is read-only and performs no reasoning.
    """
    raw_claims = full_pack.get("claims") or {}
    raw_evidence = full_pack.get("evidence") or {}
    raw_mappings = full_pack.get("mappings") or {}
    raw_relations = full_pack.get("mapping_relations") or {}

    compact_mappings = {}
    used_evidence_ids = set()

    for mapping_id, record in raw_mappings.items():
        analyses = []

        for item in record.get("claim_analyses") or []:
            evidence_ids = list(dict.fromkeys(
                item.get("evidence_ids_used") or []
            ))

            used_evidence_ids.update(evidence_ids)

            analyses.append({
                "claim_id": item.get("claim_id"),
                "explanation": item.get("explanation"),
                "evidence_ids_used": evidence_ids,
                "qualifications": (
                    item.get("qualifications") or []
                ),
            })

        compact_mappings[mapping_id] = {
            "mapping_id": mapping_id,
            "status": record.get("status"),
            "language": record.get("language"),
            "analysis_focus": record.get(
                "analysis_focus", "general"
            ),
            "target_claim_ids": record.get(
                "target_claim_ids", []
            ),
            "claim_analyses": analyses,
            "cross_claim_explanation": record.get(
                "cross_claim_explanation"
            ),
        }

    compact_claims = {}

    for claim_id, record in raw_claims.items():
        compact_claims[claim_id] = {
            "claim_id": claim_id,
            "type": record.get("type"),
            "scope": record.get("scope"),
            "source_quote": (
                record.get("source_quote")
                or record.get("model_quote")
            ),
            "source_status": record.get("source_status"),
            "source_element_id": record.get(
                "source_element_id"
            ),
        }

    warnings = []

    if compact_mappings:
        # Include evidence actually used by the Mappings.
        # Do not expand to every linked Evidence record.
        chosen_evidence_ids = [
            evidence_id
            for evidence_id in raw_evidence
            if evidence_id in used_evidence_ids
        ]
    else:
        # Use lightweight lexical relevance for direct
        # questions that do not require Mapping.
        question_words = {
            word
            for word in re.findall(
                r"[a-z][a-z0-9_]+",
                full_pack.get("question", "").lower(),
            )
            if len(word) > 2
        }

        scored = []

        for position, (evidence_id, record) in enumerate(
            raw_evidence.items()
        ):
            content = record.get("source_content") or {}

            searchable = " ".join(str(value) for value in (
                record.get("description") or "",
                record.get("source_quote") or "",
                content.get("passage") or "",
                content.get("visual_observation") or "",
                content.get("caption") or "",
            ))

            words = set(re.findall(
                r"[a-z][a-z0-9_]+",
                searchable.lower(),
            ))

            score = len(question_words & words)

            scored.append((
                -score,
                position,
                evidence_id,
            ))

        scored.sort()

        chosen_evidence_ids = [
            item[2]
            for item in scored[:max_direct_evidence]
        ]

        if len(scored) > max_direct_evidence:
            warnings.append(
                "Only the most relevant existing Evidence "
                "records were included in this response pack. "
                "Other records remain in the Evidence Pool."
            )

    compact_evidence = {}

    for evidence_id in chosen_evidence_ids:
        record = raw_evidence.get(evidence_id)

        if record is None:
            continue

        content = record.get("source_content") or {}

        source_content = {}

        for key in (
            "passage",
            "visual_observation",
            "caption",
            "label",
            "unverified_model_quote",
            "visual_reading_verified",
        ):
            if content.get(key) is not None:
                source_content[key] = content[key]

        compact_evidence[evidence_id] = {
            "evidence_id": evidence_id,
            "kind": record.get("kind"),
            "source_element_id": record.get(
                "source_element_id"
            ),
            "source_status": record.get("source_status"),
            "source_note": record.get("source_note"),
            "source_quote": record.get("source_quote"),
            "source_content": source_content,
            "description": record.get("description"),
            "observed_via": record.get("observed_via"),
        }

        if record.get("source_status") == "unlocated":
            warnings.append(
                f"Unlocated evidence: {evidence_id}"
            )

        if (
            record.get("observed_via") == "image_input"
            and content.get("visual_reading_verified") is False
        ):
            warnings.append(
                f"Visual reading not independently verified: "
                f"{evidence_id}"
            )

    compact_links = [
        link
        for link in full_pack.get("links") or []
        if (
            link.get("claim_id") in compact_claims
            and link.get("evidence_id") in compact_evidence
        )
    ]

    selected_ids = (
        set(compact_claims)
        | set(compact_evidence)
        | set(compact_mappings)
        | set(raw_relations)
    )

    original_index = full_pack.get("source_index") or {}

    compact_index = {
        source_id: original_index[source_id]
        for source_id in selected_ids
        if source_id in original_index
    }

    for warning in full_pack.get("warnings") or []:
        if warning.startswith("Mapping revision mismatch:"):
            warnings.append(warning)

    return {
        "question": full_pack.get("question"),
        "paper_id": full_pack.get("paper_id"),
        "paper_revision": full_pack.get("paper_revision"),
        "claims": compact_claims,
        "evidence": compact_evidence,
        "links": compact_links,
        "mappings": compact_mappings,
        "mapping_relations": raw_relations,
        "source_index": compact_index,
        "warnings": list(dict.fromkeys(warnings)),
        "answer_rules": full_pack.get("answer_rules") or {},
    }    

    return "\n".join(lines)