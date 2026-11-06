
from app.models.document_v2 import (
    PaperDocument,
    Paragraph,
    SourceAnchor,
    TextSpanSelector,
)
from app.models.reader_v3 import (
    RawClaim,
    ClaimRecord,
    ClaimWindow,
)
from app.tools.quote_alignment import (
    align_quote,
    recover_quote_boundary,
)


def locate_claim(
    paper: PaperDocument,
    raw: RawClaim,
    windows: list[ClaimWindow],
    run_id: str,
) -> ClaimRecord:

    def unlocated(reason: str) -> ClaimRecord:
        return ClaimRecord(
            claim_id="",
            type=raw.type,
            source_element_id=raw.source_element_id,
            model_quote=raw.source_quote,
            source_quote=raw.source_quote,
            source_anchor=None,
            source_status="unlocated",
            alignment_note=reason,
            created_run_id=run_id,
            scope=raw.scope,
        )

    element = paper.elements.get(raw.source_element_id)

    if not isinstance(element, Paragraph):
        return unlocated("source_paragraph_not_found")

    source_windows = [
        w for w in windows
        if w.element_id == raw.source_element_id
    ]

    if not source_windows:
        return unlocated("source_not_in_input_batch")

    matches = {}

    for window in source_windows:
        match, _ = align_quote(
            window.text,
            raw.source_quote,
        )

        method = "exact"

        if match is None:
            match, _ = recover_quote_boundary(
                window.text,
                raw.source_quote,
            )
            method = "quote_boundary_recovered"

        if match is None:
            continue

        start = window.start + match.start
        end = window.start + match.end

        # 再与完整原始段落核对
        if element.text[start:end] != match.exact:
            continue

        if method == "exact" and match.formatting_repaired:
            method = "formatting_recovered"

        matches[(start, end)] = method

    if not matches:
        return unlocated("quote_not_found_in_source")

    if len(matches) != 1:
        return unlocated("ambiguous_quote_location")

    (start, end), method = next(iter(matches.items()))

    exact_text = element.text[start:end]

    anchor = SourceAnchor(
        element_id=element.element_id,
        selector=TextSpanSelector(
            start=start,
            end=end,
            exact=exact_text,
            prefix=element.text[max(0, start - 40):start],
            suffix=element.text[end:end + 40],
        ),
    )

    return ClaimRecord(
        claim_id="",
        type=raw.type,
        source_element_id=element.element_id,
        model_quote=raw.source_quote,
        source_quote=exact_text,
        source_anchor=anchor,
        source_status="located",
        alignment_note=method,
        created_run_id=run_id,
    )
