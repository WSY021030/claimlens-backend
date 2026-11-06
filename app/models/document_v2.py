"""Unified source document schema for HTML-first paper ingestion (Reader-v3 foundation)."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


SourcePlatform = Literal["arxiv", "acm", "ieee", "springer"]
Completeness = Literal["complete", "partial", "missing", "unknown"]


class ElementSelector(BaseModel):
    type: Literal["element"] = "element"


class TextSpanSelector(BaseModel):
    type: Literal["text_span"] = "text_span"
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    exact: str = Field(min_length=1)
    prefix: str = ""
    suffix: str = ""

    @model_validator(mode="after")
    def check_range(self):
        if self.end <= self.start:
            raise ValueError("Text span end must be greater than start")
        return self


class TableCellsSelector(BaseModel):
    type: Literal["table_cells"] = "table_cells"
    cell_ids: list[str] = Field(min_length=1)


class ImageRegionSelector(BaseModel):
    type: Literal["image_region"] = "image_region"
    # Normalized coordinates: [left, top, right, bottom], each from 0 to 1.
    bbox: tuple[float, float, float, float]

    @model_validator(mode="after")
    def check_bbox(self):
        x1, y1, x2, y2 = self.bbox
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("Invalid normalized image bounding box")
        return self


Selector = Annotated[
    ElementSelector | TextSpanSelector | TableCellsSelector | ImageRegionSelector,
    Field(discriminator="type"),
]


class SourceAnchor(BaseModel):
    element_id: str
    selector: Selector


class PaperMetadata(BaseModel):
    title: str
    source_platform: SourcePlatform
    original_url: str | None = None
    authors: list[str] = Field(default_factory=list)
    doi: str | None = None
    year: int | None = None
    publication: str | None = None


class Section(BaseModel):
    section_id: str
    title: str
    level: int = Field(ge=1)
    parent_section_id: str | None = None
    block_ids: list[str] = Field(default_factory=list)


class InlineFormatSpan(BaseModel):
    """A presentation-only inline mark over Paragraph.text character offsets.

    Stored as ranges instead of publisher HTML, so rich display remains safe,
    model-neutral, and compatible with exact SourceAnchor offsets.
    """
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    mark: Literal["bold", "italic", "code", "sup", "sub"]

    @model_validator(mode="after")
    def valid_span(self):
        if self.end <= self.start:
            raise ValueError("Format span end must exceed start")
        return self


class InlineMathSpan(BaseModel):
    """Inline math in ordinary paragraph text, using exact character offsets.

    ``Paragraph.text`` remains an indexable plain-text fallback; renderers use
    the source-grounded MathML after sanitization. Optional for older files.
    """
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    mathml: str = Field(min_length=1)
    plain_text: str

    @model_validator(mode="after")
    def valid_span(self):
        if self.end <= self.start:
            raise ValueError("Invalid inline math span")
        return self


class Paragraph(BaseModel):
    kind: Literal["paragraph"] = "paragraph"
    element_id: str
    section_id: str
    text: str
    format_spans: list[InlineFormatSpan] = Field(default_factory=list)
    inline_math_spans: list[InlineMathSpan] = Field(default_factory=list)
    original_html_id: str | None = None

    @model_validator(mode="after")
    def check_format_spans(self):
        if any(mark.end > len(self.text) for mark in self.format_spans):
            raise ValueError("Format spans must lie inside paragraph text")
        maths = sorted(self.inline_math_spans, key=lambda s: s.start)
        last = 0
        for span in maths:
            if span.end > len(self.text) or span.start < last:
                raise ValueError("Inline math ranges must be inside text and not overlap")
            if self.text[span.start:span.end] != span.plain_text:
                raise ValueError("Inline math fallback text mismatch")
            last = span.end
        return self


class InlineAssetSpan(BaseModel):
    """Source-grounded inline asset in a figure caption, measured in caption text."""
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    asset_id: str
    alt_text: str = ""

    @model_validator(mode="after")
    def check_range(self):
        if self.end <= self.start:
            raise ValueError("Invalid inline asset span")
        return self


class Figure(BaseModel):
    kind: Literal["figure"] = "figure"
    element_id: str
    section_id: str
    label: str | None = None
    caption: str = ""
    caption_inline_assets: list[InlineAssetSpan] = Field(default_factory=list)
    caption_format_spans: list[InlineFormatSpan] = Field(default_factory=list)
    image_asset_id: str | None = None
    alt_text: str | None = None
    original_html_id: str | None = None


class TableCell(BaseModel):
    cell_id: str
    original_html_id: str | None = None
    image_asset_id: str | None = None
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    text: str
    role: Literal["data", "column_header", "row_header"] = "data"
    rowspan: int = Field(default=1, ge=1)
    colspan: int = Field(default=1, ge=1)


class TableGrid(BaseModel):
    cells: list[TableCell] = Field(default_factory=list)


class Table(BaseModel):
    kind: Literal["table"] = "table"
    element_id: str
    section_id: str
    label: str | None = None
    caption: str = ""
    representation: Literal["structured", "image_only", "hybrid"]
    grid: TableGrid | None = None
    image_asset_id: str | None = None
    recognition_status: Literal[
        "not_needed", "not_attempted", "partial", "verified", "failed"
    ] = "not_attempted"
    original_html_id: str | None = None

    @model_validator(mode="after")
    def check_representation(self):
        if self.representation in ("structured", "hybrid") and self.grid is None:
            raise ValueError("Structured/hybrid table requires a grid")
        if self.representation in ("image_only", "hybrid") and not self.image_asset_id:
            raise ValueError("Image-only/hybrid table requires an image_asset_id")
        return self


class Equation(BaseModel):
    kind: Literal["equation"] = "equation"
    element_id: str
    section_id: str
    label: str | None = None
    latex: str | None = None
    mathml: str | None = None
    plain_text: str | None = None
    original_html_id: str | None = None


PaperElement = Annotated[
    Paragraph | Figure | Table | Equation, Field(discriminator="kind")
]


class Reference(BaseModel):
    reference_id: str
    raw_text: str
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    original_html_id: str | None = None


class Mention(BaseModel):
    mention_id: str
    source_anchor: SourceAnchor
    mention_text: str
    target_kind: Literal["figure", "table", "equation", "reference"]
    target_id: str | None = None
    resolution_status: Literal["resolved", "unresolved"] = "unresolved"

    @model_validator(mode="after")
    def check_resolved(self):
        if self.resolution_status == "resolved" and self.target_id is None:
            raise ValueError("Resolved mention requires target_id")
        return self


class Asset(BaseModel):
    asset_id: str
    original_relative_path: str
    normalized_relative_path: str | None = None
    status: Literal["available", "missing"] = "missing"
    media_type: str | None = None

    @field_validator("original_relative_path", "normalized_relative_path")
    @classmethod
    def check_relative_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.replace("\\", "/")
        parts = PurePosixPath(value).parts
        if not value or value.startswith("/") or ".." in parts or ":" in parts[0]:
            raise ValueError("Asset paths must be safe relative paths")
        return value


class PaperQuality(BaseModel):
    body: Completeness = "unknown"
    figures: Completeness = "unknown"
    tables: Completeness = "unknown"
    references: Completeness = "unknown"
    warnings: list[str] = Field(default_factory=list)


class PaperDocument(BaseModel):
    schema_version: Literal["paper.v2"] = "paper.v2"
    paper_id: str
    revision: str  # Filled by the future normalizer from normalized content.
    metadata: PaperMetadata
    sections: list[Section]
    elements: dict[str, PaperElement]
    references: dict[str, Reference] = Field(default_factory=dict)
    mentions: list[Mention] = Field(default_factory=list)
    assets: dict[str, Asset] = Field(default_factory=dict)
    quality: PaperQuality = Field(default_factory=PaperQuality)

    @model_validator(mode="after")
    def verify_links(self):
        sections = {section.section_id: section for section in self.sections}
        if len(sections) != len(self.sections):
            raise ValueError("Duplicate section_id")

        used_blocks: set[str] = set()
        for section in self.sections:
            if section.parent_section_id is not None and section.parent_section_id not in sections:
                raise ValueError(f"Unknown parent section: {section.parent_section_id}")
            for block_id in section.block_ids:
                if block_id in used_blocks:
                    raise ValueError(f"Repeated block ID: {block_id}")
                used_blocks.add(block_id)
                if block_id not in self.elements:
                    raise ValueError(f"Unknown block: {block_id}")
                if self.elements[block_id].section_id != section.section_id:
                    raise ValueError(f"Wrong section for block: {block_id}")

        if used_blocks != set(self.elements):
            raise ValueError("Every element must appear exactly once in section block_ids")

        for element_id, element in self.elements.items():
            if element_id != element.element_id:
                raise ValueError(f"Element dictionary key mismatch: {element_id}")
            if isinstance(element, (Figure, Table)) and element.image_asset_id is not None:
                if element.image_asset_id not in self.assets:
                    raise ValueError(f"Unknown image asset: {element.image_asset_id}")
            if isinstance(element, Figure):
                if any(x.end > len(element.caption) for x in element.caption_format_spans):
                    raise ValueError(f"Invalid caption formatting offsets: {element.element_id}")
                for inline in element.caption_inline_assets:
                    if inline.end > len(element.caption) or inline.asset_id not in self.assets:
                        raise ValueError(f"Invalid caption inline asset: {inline.asset_id}")
            if isinstance(element, Table) and element.grid:
                for cell in element.grid.cells:
                    if cell.image_asset_id and cell.image_asset_id not in self.assets:
                        raise ValueError(f"Unknown table cell image asset: {cell.image_asset_id}")

        for ref_id, ref in self.references.items():
            if ref_id != ref.reference_id:
                raise ValueError(f"Reference dictionary key mismatch: {ref_id}")

        for asset_id, asset in self.assets.items():
            if asset_id != asset.asset_id:
                raise ValueError(f"Asset dictionary key mismatch: {asset_id}")

        for mention in self.mentions:
            if mention.source_anchor.element_id not in self.elements:
                raise ValueError(f"Unknown mention source: {mention.source_anchor.element_id}")
            if mention.resolution_status == "resolved":
                if mention.target_kind == "reference":
                    if mention.target_id not in self.references:
                        raise ValueError(f"Unknown reference target: {mention.target_id}")
                elif mention.target_id not in self.elements or self.elements[mention.target_id].kind != mention.target_kind:
                    raise ValueError(f"Invalid element target: {mention.target_id}")

        return self
