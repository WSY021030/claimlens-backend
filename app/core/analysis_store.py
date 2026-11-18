

import os
import re
import tempfile

from pathlib import Path

from app.models.document_v2 import PaperDocument
from app.core.current_storage import current_document, atomic_json, write_current_outputs

from app.models.reader_v3 import (
    AnalysisDocument,
    ClaimRecord,
    EvidenceRecord,
)


class AnalysisStore:

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(
        self,
        paper: PaperDocument,
    ) -> AnalysisDocument:

        if not self.path.exists():
            return AnalysisDocument(
                paper_id=paper.paper_id,
                paper_revision=paper.revision,
            )

        data = AnalysisDocument.model_validate_json(
            self.path.read_text(encoding="utf-8-sig")
        )

        if data.paper_id != paper.paper_id:
            raise ValueError("Paper ID mismatch")

        if data.paper_revision != paper.revision:
            raise ValueError(
                "Paper revision changed. Existing source "
                "anchors need review before reuse."
            )

        return data

    @staticmethod
    def _next_id(pool: dict, prefix: str) -> str:
        number = 1

        while f"{prefix}_{number:06d}" in pool:
            number += 1

        return f"{prefix}_{number:06d}"

    def add_claim(
        self,
        data: AnalysisDocument,
        candidate: ClaimRecord,
    ) -> tuple[str, bool]:

        for old_id, old in data.claim_pool.items():

            if old.source_element_id != (
                candidate.source_element_id
            ):
                continue

            if old.type != candidate.type:
                continue

            same_span = (
                old.source_anchor is not None
                and candidate.source_anchor is not None
                and old.source_anchor == candidate.source_anchor
            )

            same_unlocated = (
                old.source_anchor is None
                and candidate.source_anchor is None
                and old.model_quote == candidate.model_quote
            )

            if same_span or same_unlocated:
                # Keep the original Claim and ID.
                # A new scope may fill an older null field.
                if old.scope is None and candidate.scope:
                    old.scope = candidate.scope

                return old_id, False

        claim_id = self._next_id(
            data.claim_pool, "claim"
        )

        data.claim_pool[claim_id] = candidate.model_copy(
            update={"claim_id": claim_id}
        )

        return claim_id, True

    def add_evidence(
        self,
        data: AnalysisDocument,
        candidate: EvidenceRecord,
    ) -> tuple[str, bool]:
        """Reuse a source record without suppressing independent observations.

        Text is deduplicated only when verified source spans are identical or
        nested AND their source quotations are identical or nested.
        Image observations are grouped only for identical submitted image
        bytes and the same crop/cell, retaining non-identical observations.
        """

        def normalize(value):
            return " ".join(str(value or "").split()).casefold()

        def track_merge(existing, incoming):
            if existing.created_run_id != incoming.created_run_id:
                origin_ids = existing.source_content.setdefault(
                    "merged_from_run_ids", []
                )
                if incoming.created_run_id not in origin_ids:
                    origin_ids.append(incoming.created_run_id)

        def same_image_input(existing, incoming):
            if existing.observed_via != "image_input":
                return False
            a = existing.image_receipt or {}
            b = incoming.image_receipt or {}
            return bool(
                a.get("sent_sha256")
                and a.get("sent_sha256") == b.get("sent_sha256")
                and a.get("cell_id") == b.get("cell_id")
            )

        def add_nonduplicate_sentences(existing_text, incoming_text, separator=" "):
            import re

            def parts(value):
                return [
                    sentence.strip()
                    for sentence in re.split(
                        r"(?<=[.!?。！？])\s+|[\r\n]+",
                        str(value or "").strip(),
                    )
                    if sentence.strip()
                ]

            retained = parts(existing_text)
            for sentence in parts(incoming_text):
                key = normalize(sentence)
                if not key:
                    continue
                if any(key == normalize(old) for old in retained):
                    continue
                # Prefer the more informative enclosing sentence instead of
                # keeping the same sentence twice with an added clause.
                covering = [index for index, old in enumerate(retained)
                            if normalize(old) in key]
                if covering:
                    retained[covering[0]] = sentence
                    for index in reversed(covering[1:]):
                        retained.pop(index)
                else:
                    retained.append(sentence)
            return separator.join(retained)

        def merge_image(existing, incoming):
            old_content = existing.source_content
            new_content = incoming.source_content
            old_content["visual_observation"] = add_nonduplicate_sentences(
                old_content.get("visual_observation"),
                new_content.get("visual_observation"),
                separator="\n\n",
            )
            existing.description = add_nonduplicate_sentences(
                existing.description, incoming.description,
            )
            # Keep the original image receipt and preserve any further
            # different receipt metadata without changing the primary ID.
            if incoming.image_receipt and incoming.image_receipt != existing.image_receipt:
                receipts = old_content.setdefault("additional_image_receipts", [])
                if incoming.image_receipt not in receipts:
                    receipts.append(incoming.image_receipt)
            for field in ("caption", "label", "alt_text", "image_asset_id"):
                if not old_content.get(field) and new_content.get(field):
                    old_content[field] = new_content[field]
            track_merge(existing, incoming)

        def nested_text_quotes(existing, incoming):
            if (existing.kind != "paragraph"
                    or existing.observed_via != "text"
                    or existing.source_status != "located"
                    or incoming.source_status != "located"
                    or not existing.source_anchor
                    or not incoming.source_anchor):
                return False
            a = existing.source_anchor.selector
            b = incoming.source_anchor.selector
            if (getattr(a, "type", None) != "text_span"
                    or getattr(b, "type", None) != "text_span"):
                return False
            if not all(isinstance(getattr(s, k, None), int)
                       for s in (a, b) for k in ("start", "end")):
                return False
            nested = (
                (a.start <= b.start and a.end >= b.end)
                or (b.start <= a.start and b.end >= a.end)
            )
            older = normalize(existing.source_quote)
            newer = normalize(incoming.source_quote)
            return bool(nested and older and newer
                        and (older in newer or newer in older))

        def merge_text(existing, incoming):
            a = existing.source_anchor.selector
            b = incoming.source_anchor.selector
            if b.end - b.start > a.end - a.start:
                # Keep the established ID but use the fuller verified span.
                existing.source_anchor = incoming.source_anchor
                existing.source_quote = incoming.source_quote
                for field in ("passage", "context", "start", "end"):
                    if field in incoming.source_content:
                        existing.source_content[field] = incoming.source_content[field]
            # The source quotation, rather than competing generated
            # paraphrases, remains the authoritative substantive content.
            # Preserve an established description to avoid redundant prose.
            track_merge(existing, incoming)

        for old_id, old in data.evidence_pool.items():
            if (
                old.kind != candidate.kind
                or old.source_element_id != candidate.source_element_id
                or old.source_asset_id != candidate.source_asset_id
                or old.observed_via != candidate.observed_via
            ):
                continue

            if old.observed_via == "image_input":
                if same_image_input(old, candidate):
                    merge_image(old, candidate)
                    return old_id, False
                # Different submitted image bytes/crops remain independent.
                continue

            same_anchor = (
                old.source_anchor is not None
                and candidate.source_anchor is not None
                and old.source_anchor == candidate.source_anchor
            )
            same_unlocated = (
                old.source_anchor is None
                and candidate.source_anchor is None
                and old.source_quote == candidate.source_quote
                and old.description == candidate.description
            )

            if nested_text_quotes(old, candidate):
                merge_text(old, candidate)
                return old_id, False
            if same_anchor or same_unlocated:
                track_merge(old, candidate)
                return old_id, False

        high_water = max(
            int(data.id_counters.get("evidence", 0)),
            max(
                (int(eid.split("_", 1)[1]) for eid in data.evidence_pool
                 if re.fullmatch(r"evidence_\d{6}", eid)),
                default=0,
            ),
        )
        next_number = high_water + 1
        data.id_counters["evidence"] = next_number
        evidence_id = f"evidence_{next_number:06d}"
        data.evidence_pool[evidence_id] = candidate.model_copy(
            update={"evidence_id": evidence_id}
        )
        return evidence_id, True

    def save(self, data: AnalysisDocument) -> None:
        """Persist current state, keeping live objects unchanged for the caller.

        Retention is applied to a serialized copy, so Session.run and agent-local
        dictionaries never become detached during a save or subprocess reload.
        Source pools, cache identities and the SQLite budget ledger are unchanged.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        maintenance = self.path.parent / ".storage-maintenance.lock"
        if maintenance.exists() and maintenance.read_text(encoding="utf-8") != getattr(self, "_maintenance_token", None):
            raise RuntimeError("Storage migration is active; do not run agents concurrently.")
        payload = current_document(data.model_dump(mode="json"))
        atomic_json(self.path, payload)
        data.id_counters.update(payload["id_counters"])
        data.storage_policy = payload["storage_policy"]
        data.snapshot_id = payload["snapshot_id"]
        try:
            write_current_outputs(data, payload, self.path.parent)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # The authoritative analysis is already safe. Surface a derived
            # export failure without discarding successful, paid agent results.
            print("Current analysis saved; derived export needs attention: "
                  + type(exc).__name__ + ": " + str(exc)[:300], flush=True)
            try:
                atomic_json(self.path.parent / "display.json", {
                    "paper_id": data.paper_id, "paper_revision": data.paper_revision,
                    "snapshot_id": payload["snapshot_id"], "status": "unavailable",
                    "display_by_claim": {}, "issues": ["Derived export failed; regenerate from analysis.json."],
                })
                atomic_json(self.path.parent / "response.json", {
                    "paper_id": data.paper_id, "paper_revision": data.paper_revision,
                    "snapshot_id": payload["snapshot_id"], "status": "unavailable",
                    "answer": None, "issues": ["Derived export failed; analysis.json remains authoritative."],
                })
                (self.path.parent / "answer.md").unlink(missing_ok=True)
            except OSError:
                pass


def default_analysis_path(paper_id: str) -> Path:

    if not re.fullmatch(
        r"[A-Za-z0-9_-]+", paper_id
    ):
        raise ValueError("Unsafe paper ID")

    return (
        Path("data")
        / "analysis"
        / paper_id
        / "analysis.json"
    )

