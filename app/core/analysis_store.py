

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

        for old_id, old in data.evidence_pool.items():

            if (
                old.kind != candidate.kind
                or old.source_element_id != candidate.source_element_id
                or old.source_asset_id != candidate.source_asset_id
                or old.observed_via != candidate.observed_via
            ):
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

            if same_anchor or same_unlocated:

                # Different observations from one image
                # are separate evidence records.
                if old.observed_via == "image_input":
                    old_observation = old.source_content.get(
                        "visual_observation"
                    )

                    new_observation = (
                        candidate.source_content.get(
                            "visual_observation"
                        )
                    )

                    if old_observation != new_observation:
                        continue

                return old_id, False

        evidence_id = self._next_id(
            data.evidence_pool, "evidence"
        )

        data.evidence_pool[evidence_id] = (
            candidate.model_copy(
                update={"evidence_id": evidence_id}
            )
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

