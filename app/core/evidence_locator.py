
from typing import Any

from app.models.document_v2 import (
    PaperDocument,
    Paragraph,
    Table,
    Figure,
    Equation,
    SourceAnchor,
    ElementSelector,
    TextSpanSelector,
    TableCellsSelector,
)

from app.models.reader_v3 import (
    RawEvidence,
    EvidenceRecord,
)

from app.tools.quote_alignment import (
    align_quote,
    recover_quote_boundary,
)


def _span(element_id, text, start, end):
    return SourceAnchor(
        element_id=element_id,
        selector=TextSpanSelector(
            start=start,
            end=end,
            exact=text[start:end],
            prefix=text[max(0, start - 40):start],
            suffix=text[end:end + 40],
        ),
    )


def _cell_data(cell):
    return {
        "cell_id": cell.cell_id,
        "row": cell.row,
        "col": cell.col,
        "role": cell.role,
        "rowspan": cell.rowspan,
        "colspan": cell.colspan,
        "text": cell.text,
        "image_asset_id": cell.image_asset_id,
    }


def locate_evidence(
    paper: PaperDocument,
    raw: RawEvidence,
    scanned_elements: list[dict],
    run_id: str,
    image_receipt: dict[str, Any] | None = None,
) -> EvidenceRecord:

    element = paper.elements.get(raw.source_element_id)

    scanned = next(
        (
            item for item in scanned_elements
            if item.get("element_id") == raw.source_element_id
        ),
        None,
    )

    # A model-proposed location outside the supplied chunk
    # is retained but never presented as verified.
    if element is None or scanned is None:
        return EvidenceRecord(
            evidence_id="",
            kind=raw.kind,
            source_element_id=raw.source_element_id,
            source_status="unlocated",
            source_note="source_not_in_scanned_material",
            source_quote=None,
            source_content={
                "unverified_model_quote": raw.source_quote,
            },
            description=raw.description,
            observed_via="text",
            created_run_id=run_id,
        )

    kind = element.kind
    note = None
    anchor = None
    verified_quote = None
    content = {}

    actual_image = (
        image_receipt is not None
        and image_receipt.get("element_id") == element.element_id
    )

    source_asset_id = (
        image_receipt.get("asset_id")
        if actual_image else None
    )

    observed_via = {
        "paragraph": "text",
        "table": "structured_table",
        "figure": "caption_only",
        "equation": "equation",
    }[kind]

    if isinstance(element, Paragraph):
        text = element.text

        match = None

        if raw.source_quote:
            match, _ = align_quote(
                text, raw.source_quote
            )

            if match is None:
                match, _ = recover_quote_boundary(
                    text, raw.source_quote
                )

        if match is not None:
            start, end = match.start, match.end
            note = "exact_source_quote_located"

        else:
            # Preserve a real source passage from the
            # chunk, rather than inventing an anchor.
            start = int(scanned.get("source_start", 0))
            end = int(scanned.get(
                "source_end", len(text)
            ))

            start = max(0, min(start, len(text)))
            end = max(start, min(end, len(text)))

            if start == end:
                note = "empty_source_passage"
            else:
                note = "source_passage_located_quote_unconfirmed"

        if end > start:
            anchor = _span(
                element.element_id, text, start, end
            )

            context_start = max(0, start - 250)
            context_end = min(len(text), end + 250)

            verified_quote = (
                text[start:end]
                if match is not None else None
            )

            content = {
                "passage": text[start:end],
                "context": text[context_start:context_end],
                "start": start,
                "end": end,
            }

        if match is None and raw.source_quote:
            content["unverified_model_quote"] = (
                raw.source_quote
            )

    elif isinstance(element, Table):
        observed_via = "structured_table"

        content = {
            "label": element.label,
            "caption": element.caption,
            "representation": element.representation,
            "recognition_status": element.recognition_status,
        }

        if element.grid and element.grid.cells:
            cells = element.grid.cells

            by_id = {
                cell.cell_id: cell
                for cell in cells
            }

            # Prefer cells selected by the model.
            requested_ids = list(dict.fromkeys(
                raw.cell_ids
            ))

            # If no cells were selected, use the portion
            # of the table that the model actually saw.
            if not requested_ids:
                requested_ids = [
                    item["cell_id"]
                    for item in scanned.get("cells", [])
                ]

            valid_ids = [
                cid for cid in requested_ids
                if cid in by_id
            ]

            missing_ids = [
                cid for cid in requested_ids
                if cid not in by_id
            ]

            if valid_ids:
                relevant_rows = {
                    by_id[cid].row
                    for cid in valid_ids
                }

                # Keep entire selected rows and header
                # cells, so column meanings are not lost.
                chosen = [
                    cell for cell in cells
                    if (
                        cell.row in relevant_rows
                        or cell.role == "column_header"
                        or cell.row == 0
                    )
                ]

                anchor = SourceAnchor(
                    element_id=element.element_id,
                    selector=TableCellsSelector(
                        cell_ids=sorted(set(valid_ids))
                    ),
                )

                content["cells"] = [
                    _cell_data(cell)
                    for cell in chosen
                ]

                note = "table_cells_located"

            else:
                anchor = SourceAnchor(
                    element_id=element.element_id,
                    selector=ElementSelector(),
                )

                # Retain real table data even if no
                # precise cells were selected.
                content["cells"] = [
                    _cell_data(cell)
                    for cell in cells
                ]

                note = "table_element_located"

            if missing_ids:
                content["unresolved_cell_ids"] = missing_ids
                note += "_some_cell_ids_unresolved"

        else:
            anchor = SourceAnchor(
                element_id=element.element_id,
                selector=ElementSelector(),
            )

            content["cells"] = []
            note = "table_has_no_structured_cells"

    elif isinstance(element, Figure):
        anchor = SourceAnchor(
            element_id=element.element_id,
            selector=ElementSelector(),
        )

        content = {
            "label": element.label,
            "caption": element.caption,
            "alt_text": element.alt_text,
            "image_asset_id": element.image_asset_id,
        }

        note = "figure_element_located"

        if actual_image:
            observed_via = "image_input"

            if raw.visual_observation:
                content["visual_observation"] = (
                    raw.visual_observation
                )

                content["visual_observation_source"] = (
                    "model_reading_of_attached_image"
                )

                content["visual_reading_verified"] = False

        elif raw.visual_observation:
            content["unverified_visual_observation"] = (
                raw.visual_observation
            )

            note = "image_pixels_not_supplied"

    elif isinstance(element, Equation):
        anchor = SourceAnchor(
            element_id=element.element_id,
            selector=ElementSelector(),
        )

        content = {
            "label": element.label,
            "latex": element.latex,
            "mathml": element.mathml,
            "plain_text": element.plain_text,
        }

        note = "equation_element_located"

    if actual_image and isinstance(element, Table):
        observed_via = "image_input"

        if raw.visual_observation:
            content["visual_observation"] = (
                raw.visual_observation
            )
            content["visual_observation_source"] = (
                "model_reading_of_attached_image"
            )
            content["visual_reading_verified"] = False

    if raw.source_quote and verified_quote is None:
        content.setdefault(
            "unverified_model_quote",
            raw.source_quote,
        )

    return EvidenceRecord(
        evidence_id="",
        kind=kind,
        source_element_id=element.element_id,
        source_anchor=anchor,
        source_status=(
            "located" if anchor else "unlocated"
        ),
        source_note=note,
        source_quote=verified_quote,
        source_content=content,
        description=raw.description,
        observed_via=observed_via,
        source_asset_id=source_asset_id,
        image_receipt=(
            image_receipt if actual_image else None
        ),
        created_run_id=run_id,
    )
