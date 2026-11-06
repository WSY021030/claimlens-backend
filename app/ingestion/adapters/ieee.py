"""Offline IEEE Xplore saved-HTML adapter (Document Schema v2).

Consumes HTMLBundle from app.ingestion.loader. It never runs saved scripts,
fetches remote URLs, or invents cells for image-only tables.

This first adapter targets IEEE Xplore *HTML full-text* pages containing
#article > div.section (e.g. document/9633431). Some IEEE pages only
provide an abstract; these are rejected rather than treated as full text.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup, Tag

from app.ingestion.loader import HTMLBundle, ImageReference
from app.models.document_v2 import (
    Asset, ElementSelector, Equation, Figure, Mention, PaperDocument,
    PaperMetadata, PaperQuality, Paragraph, Reference, Section, SourceAnchor,
    Table, TextSpanSelector,
)


class IEEEFullTextUnavailableError(ValueError):
    """The saved IEEE page does not contain HTML full text in #article."""


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _has_class(tag: Tag, name: str) -> bool:
    return name in tag.get("class", [])


def _is_section(tag: Tag) -> bool:
    return tag.name in ("div", "section") and any(
        name == "section" or re.fullmatch(r"section_\d+", name)
        for name in tag.get("class", [])
    )


def _section_title(tag: Tag) -> str:
    # IEEE full-text section: div.header > h2; subsection: h3/h4.
    header = next((x for x in tag.find_all(recursive=False)
                   if isinstance(x, Tag) and _has_class(x, "header")), None)
    heading = (header.find(["h2", "h3", "h4"]) if header else None)
    if heading is None:
        heading = next((x for x in tag.find_all(recursive=False)
                        if isinstance(x, Tag) and x.name in ("h2", "h3", "h4", "h5")), None)
    return _clean(heading.get_text(" ", strip=True)) if heading else "Untitled section"


def _latex_from_formula(formula: Tag) -> str | None:
    tex = formula.select_one("span.tex")
    if tex is not None:
        result = tex.get_text("", strip=True)
        return result or None
    annotation = formula.select_one('annotation[encoding="application/x-tex"]')
    if annotation:
        return annotation.get_text("", strip=True) or None
    return None


def _paragraph_text(tag: Tag) -> str:
    # Saved pages contain huge MathJax-rendered markup. Extract the display
    # equations into separate Equation blocks, rather than duplicate them in
    # Paragraph.text and then in the standalone equation block.
    clone = BeautifulSoup(str(tag), "lxml")
    for formula in clone.select("disp-formula, span.formula"):
        if formula.find_parent("disp-formula") and formula.name == "span":
            continue
        latex = _latex_from_formula(formula)
        formula.replace_with(" ")
    for junk in clone.select("script, style, .MathJax_Preview"):
        junk.decompose()
    return _clean(clone.get_text(" ", strip=True))


def _paragraph_formulas(tag: Tag) -> list[str]:
    result = []
    for formula in tag.select("disp-formula"):
        latex = _latex_from_formula(formula)
        if latex:
            result.append(latex)
    for formula in tag.select("span.formula"):
        if formula.find_parent("disp-formula"):
            continue
        latex = _latex_from_formula(formula)
        if latex:
            result.append(latex)
    return result


def _resource_image_ref(bundle: HTMLBundle, img: Tag | None) -> ImageReference | None:
    if img is None:
        return None
    src = img.get("src") or img.get("data-src") or ""
    return next((ref for ref in bundle.images if ref.src == src), None)


def _safe_image_relative_path(ref: ImageReference) -> str | None:
    src = urlsplit(ref.src)
    if src.scheme or src.netloc or not src.path:
        return None
    path = unquote(src.path).replace("\\", "/")
    parts = [part for part in path.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts) or path.startswith("/"):
        return None
    return "/".join(parts)


def _metadata(soup: BeautifulSoup, bundle: HTMLBundle) -> PaperMetadata:
    def meta_value(key: str, value: str) -> str | None:
        el = soup.find("meta", attrs={key: value})
        if el:
            return el.get("content") or None
        return None

    title_tag = soup.select_one("h1.document-title")
    title = (_clean(title_tag.get_text(" ", strip=True)) if title_tag
             else (meta_value("property", "og:title") or "Untitled IEEE paper"))
    authors = []
    for el in soup.select(".authors-info"):
        name = _clean(el.get_text(" ", strip=True)).rstrip(" ;,")
        if name and name not in authors:
            authors.append(name)
    # IEEE's HTML page may omit publisher metadata in <meta>. Do not infer
    # DOI or year from unrelated in-page mentions.
    doi = meta_value("name", "citation_doi")
    date = (meta_value("name", "citation_publication_date")
            or meta_value("name", "citation_date"))
    m = re.search(r"\b(?:19|20)\d{2}\b", date or "")
    year = int(m.group(0)) if m else None
    return PaperMetadata(title=title, source_platform="ieee",
                         original_url=bundle.source_url, authors=authors,
                         doi=doi, year=year)



def _extract_ieee_references(soup: BeautifulSoup) -> tuple[dict[str, Reference], dict[str, str]]:
    """Read IEEE's expanded bibliography, ignoring closed placeholder panels.

    Modern IEEE saved pages may contain multiple #references-section-container
    nodes: one with 'not available' and another with the actual list. Therefore
    parse specifically the numbered bibliography items, not the first panel.
    """
    references: dict[str, Reference] = {}
    anchor_to_reference_id: dict[str, str] = {}

    for item in soup.select("#references-section-container xpl-reference-item-migr"):
        number_node = item.select_one('div.number[id$="-all-ref"]')
        if number_node is None:
            continue
        raw_id = number_node.get("id", "")
        match = re.fullmatch(r"ref(\d+)-all-ref", raw_id)
        if not match:
            continue

        number = int(match.group(1))
        # Bibliography text lives here; exclude 'Show in Context' / Scholar links.
        citation_text = item.select_one('div.col[xplmathjax]')
        if citation_text is None:
            citation_text = item.select_one('[xplmathjax]')
        if citation_text is None:
            continue
        raw_text = _clean(citation_text.get_text(" ", strip=True))
        if not raw_text:
            continue

        reference_id = f"ref_{number:03d}"
        if reference_id in references:
            continue  # duplicate list/view in the same saved page
        references[reference_id] = Reference(
            reference_id=reference_id,
            raw_text=raw_text,
            original_html_id=raw_id,
        )
        anchor_to_reference_id[f"ref{number}"] = reference_id

    return references, anchor_to_reference_id


def parse_ieee_bundle(bundle: HTMLBundle) -> PaperDocument:
    """Turn an IEEE HTMLBundle into a validated PaperDocument v2."""
    soup = BeautifulSoup(bundle.html, "lxml")
    article = soup.select_one("#article")
    main_sections = (
        [child for child in article.find_all(recursive=False)
         if isinstance(child, Tag) and _is_section(child)]
        if article else []
    )
    if not main_sections:
        raise IEEEFullTextUnavailableError(
            "No IEEE full-text #article sections were found. "
            "Please save an IEEE page with the HTML Full Text visible."
        )

    sections: list[Section] = []
    elements = {}
    assets: dict[str, Asset] = {}
    raw_to_element_id: dict[str, str] = {}
    pending_mentions: list[tuple[str, Tag, str]] = []
    warnings: list[str] = []
    counters = {"para": 0, "fig": 0, "table": 0, "equation": 0,
                "asset": 0, "mention": 0}

    def next_id(kind: str) -> str:
        counters[kind] += 1
        return f"{kind}_{counters[kind]:04d}"

    def attach_image(img: Tag | None) -> str | None:
        ref = _resource_image_ref(bundle, img)
        if ref is None:
            warnings.append("A paper image is not present in the HTML image inventory")
            return None
        rel = _safe_image_relative_path(ref)
        if rel is None:
            warnings.append("Paper image uses an unresolved or nonlocal resource URL")
            return None
        asset_id = next_id("asset")
        assets[asset_id] = Asset(
            asset_id=asset_id, original_relative_path=rel,
            status="available" if ref.status == "available" else "missing",
            media_type=ref.media_type,
        )
        if ref.status != "available":
            warnings.append(f"Image resource not available: {rel}")
        return asset_id

    def add_paragraph(p: Tag, section: Section) -> None:
        text = _paragraph_text(p)
        if not text:
            return
        eid = next_id("para")
        elements[eid] = Paragraph(element_id=eid, section_id=section.section_id,
                                  text=text, original_html_id=p.get("id"))
        section.block_ids.append(eid)
        for anchor in p.select("a[ref-type]"):
            pending_mentions.append((eid, anchor, text))
        for latex in _paragraph_formulas(p):
            eqid = next_id("equation")
            elements[eqid] = Equation(element_id=eqid,
                                     section_id=section.section_id, latex=latex,
                                     plain_text=latex)
            section.block_ids.append(eqid)

    def add_figure(node: Tag, section: Section) -> None:
        is_table = _has_class(node, "table")
        label_tag = node.select_one(".figcaption .title")
        label = _clean(label_tag.get_text(" ", strip=True)) if label_tag else None
        caption_tag = node.select_one(".figcaption")
        caption = _clean(caption_tag.get_text(" ", strip=True)) if caption_tag else ""
        if label and caption.lower().startswith(label.lower()):
            caption = caption[len(label):].strip(" :-.")
        img = node.select_one(".img-wrap img") or node.find("img")
        asset_id = attach_image(img)
        if is_table:
            eid = next_id("table")
            if asset_id is None:
                # An image-only table needs a valid reference to an asset.
                # Missing local files still have an Asset(status='missing');
                # unsupported URLs are treated as an extraction failure.
                warnings.append(f"Cannot represent image table: {label or eid}")
                return
            elements[eid] = Table(
                element_id=eid, section_id=section.section_id,
                label=label, caption=caption, representation="image_only",
                image_asset_id=asset_id, recognition_status="not_attempted",
                original_html_id=node.get("id"),
            )
        else:
            eid = next_id("fig")
            elements[eid] = Figure(
                element_id=eid, section_id=section.section_id,
                label=label, caption=caption, image_asset_id=asset_id,
                alt_text=img.get("alt") if img else None,
                original_html_id=node.get("id"),
            )
        section.block_ids.append(eid)
        if node.get("id"):
            raw_to_element_id[node["id"]] = eid

    def walk_block(block: Tag, section: Section) -> None:
        if _is_section(block):
            add_section(block, section.section_id, section.level + 1)
        elif block.name == "p":
            add_paragraph(block, section)
        elif _has_class(block, "figure"):
            add_figure(block, section)
        elif block.name in ("ol", "ul"):
            for li in block.find_all("li", recursive=False):
                # Lists are paragraph-like source text, including formulas.
                own_p = li.find("p", recursive=False)
                add_paragraph(own_p if own_p else li, section)
        elif block.name in ("div", "section"):
            # Some IEEE versions use extra wrappers around prose.
            if _has_class(block, "header"):
                return
            for child in block.find_all(recursive=False):
                if isinstance(child, Tag):
                    walk_block(child, section)

    def add_section(tag: Tag, parent_id: str | None, level: int) -> Section:
        sid = f"sec_{len(sections)+1:03d}"
        section = Section(section_id=sid, title=_section_title(tag), level=level,
                          parent_section_id=parent_id)
        sections.append(section)
        for child in tag.find_all(recursive=False):
            if isinstance(child, Tag):
                walk_block(child, section)
        return section

    abstracts = soup.select(".abstract-text-content")
    # IEEE sometimes has both a truncated mobile snippet and a full abstract.
    abstract = max(abstracts, key=lambda el: len(el.get_text(" ", strip=True)),
                   default=None)
    if abstract:
        text = _clean(abstract.get_text(" ", strip=True))
        if text:
            sid = "sec_001"
            sec = Section(section_id=sid, title="Abstract", level=1)
            sections.append(sec)
            pid = next_id("para")
            elements[pid] = Paragraph(element_id=pid, section_id=sid, text=text)
            sec.block_ids.append(pid)

    for tag in main_sections:
        add_section(tag, parent_id=None, level=1)

    references, ref_anchor_lookup = _extract_ieee_references(soup)

    mentions = []
    search_positions: dict[tuple[str, str], int] = {}
    for para_id, anchor, para_text in pending_mentions:
        ref_type = anchor.get("ref-type", "")
        kind = {"fig": "figure", "table": "table", "bibr": "reference",
                "disp-formula": "equation"}.get(ref_type)
        if kind is None:
            continue
        target_id = raw_to_element_id.get(anchor.get("anchor", ""))
        resolved = target_id in elements and elements[target_id].kind == kind
        # Document citations link to real IEEE bibliography items, never
        # to invented entries. Absent items remain explicitly unresolved.
        if kind == "reference":
            target_id = ref_anchor_lookup.get(anchor.get("anchor", ""))
            resolved = target_id in references
        mention_text = _clean(anchor.get_text(" ", strip=True))
        position_key = (para_id, mention_text)
        from_position = search_positions.get(position_key, 0)
        start = para_text.find(mention_text, from_position) if mention_text else -1
        if start < 0 and mention_text:
            start = para_text.find(mention_text)
        if start >= 0:
            search_positions[position_key] = start + len(mention_text)
        selector = (TextSpanSelector(start=start, end=start + len(mention_text),
                                     exact=mention_text)
                    if start >= 0 else ElementSelector())
        counters["mention"] += 1
        mentions.append(Mention(
            mention_id=f"mention_{counters['mention']:04d}",
            source_anchor=SourceAnchor(element_id=para_id, selector=selector),
            mention_text=mention_text, target_kind=kind,
            target_id=target_id if resolved else None,
            resolution_status="resolved" if resolved else "unresolved",
        ))

    reference_mentions = [m for m in mentions if m.target_kind == "reference"]
    unresolved_references = [m for m in reference_mentions if m.resolution_status != "resolved"]
    if not references:
        references_quality = "missing"
        if reference_mentions:
            warnings.append(
                "IEEE HTML does not contain a usable bibliography; citation mentions are unresolved"
            )
    elif unresolved_references:
        references_quality = "partial"
        warnings.append(
            f"{len(unresolved_references)} bibliography citation mention(s) could not be matched"
        )
    else:
        references_quality = "complete"
    fig_ids = [e for e in elements.values() if isinstance(e, Figure)]
    table_ids = [e for e in elements.values() if isinstance(e, Table)]
    images_missing = any(a.status != "available" for a in assets.values())
    main_titles = [s.title.lower() for s in sections if s.level == 1]
    has_conclusion = any("conclusion" in x for x in main_titles)
    if not has_conclusion:
        warnings.append("No Conclusion section was found; paper body may be incomplete")

    metadata = _metadata(soup, bundle)
    revision_content = "\n".join(metadata.title + "\n" + s.title + ":" + "|".join(
        (elements[block].text if isinstance(elements[block], Paragraph)
         else getattr(elements[block], "caption", "") or getattr(elements[block], "latex", "") or "")
        for block in s.block_ids) for s in sections)
    # Reference data is part of the paper revision: replacing a partial
    # HTML export with one containing references must invalidate old anchors.
    revision_content += "\n" + "\n".join(
        f"{key}: {reference.raw_text}" for key, reference in references.items()
    )
    revision = "sha256:" + hashlib.sha256(revision_content.encode("utf-8")).hexdigest()
    doc_id = re.search(r"/document/(\d+)", bundle.source_url or "")
    paper_id = "ieee_" + (doc_id.group(1) if doc_id else revision[7:19])
    return PaperDocument(
        paper_id=paper_id, revision=revision, metadata=metadata,
        sections=sections, elements=elements, references=references,
        assets=assets, mentions=mentions,
        quality=PaperQuality(
            body="complete" if len(main_sections) >= 2 and has_conclusion else "partial",
            figures="partial" if images_missing else ("complete" if fig_ids else "unknown"),
            tables="partial" if table_ids else "unknown",
            references=references_quality,
            warnings=warnings,
        ),
    )