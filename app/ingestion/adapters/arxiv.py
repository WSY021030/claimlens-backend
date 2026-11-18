"""Offline arXiv HTML (LaTeXML) adapter for PaperDocument v2.

Supports locally saved https://arxiv.org/html/<id> pages, alongside their
browser-saved image directory. No remote requests or scripts are executed.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath
from urllib.parse import unquote, urlparse

from bs4 import BeautifulSoup, Tag, NavigableString

from app.ingestion.loader import HTMLBundle
from app.models.document_v2 import (
    Asset, ElementSelector, Equation, Figure, Mention, PaperDocument,
    PaperMetadata, PaperQuality, Paragraph, Reference, Section, SourceAnchor,
    Table, TableCell, TableGrid, TextSpanSelector, InlineFormatSpan, InlineMathSpan,
)


class ArxivFullTextUnavailableError(ValueError):
    """No usable LaTeXML paper content was found in the downloaded HTML."""


def _has(node: Tag, name: str) -> bool:
    return name in node.get("class", [])


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _meta(soup: BeautifulSoup, name: str) -> str | None:
    el = soup.find("meta", attrs={"name": name})
    return _clean(el.get("content", "")) or None if el else None


def _math_text(math: Tag) -> str:
    # Prefer source LaTeX preserved by LaTeXML. Avoid MathML's noisy text nodes.
    if math.get("alttext"):
        return math["alttext"]
    ann = math.select_one('annotation[encoding="application/x-tex"]')
    if ann:
        return _clean(ann.get_text("", strip=True))
    return _clean(math.get_text(" ", strip=True))


def _prose(tag: Tag) -> str:
    """Plain text, retaining inline math as LaTeX and dropping formula graphics."""
    clone = BeautifulSoup(str(tag), "lxml")
    for math in clone.select("math"):
        math.replace_with(" " + _math_text(math) + " ")
    for junk in clone.select("script,style,.ltx_tag_section,.ltx_tag_subsection,.ltx_tag_subsubsection,.ltx_tag_paragraph"):
        junk.decompose()
    return _clean(clone.get_text(" ", strip=True))



def _formatted_prose(tag: Tag) -> tuple[str, list[InlineFormatSpan], list[InlineMathSpan], list[str], list[tuple[int, int, str]]]:
    """ARXIV_INLINE_MATH_READABILITY_V1: preserve MathML with exact offsets."""
    cloned = BeautifulSoup(str(tag), "lxml")
    root = cloned.find(tag.name)

    if root is None:
        raise ValueError("Unable to reconstruct the source paragraph")

    # ARXIV_BROKEN_CROSSREF_METADATA_V2
    # Preserve broken parenthesized LaTeXML references as metadata.
    # Do not invent a figure number or target element.
    broken_refs: list[str] = []

    for error in list(root.select(".ltx_ERROR")):
        command = _clean(error.get_text("", strip=True))

        if command not in {
            r"\cref", r"\Cref", r"\ref", r"\autoref"
        }:
            continue

        previous = error.previous_sibling
        following = error.next_sibling

        if (
            not isinstance(previous, NavigableString)
            or not isinstance(following, NavigableString)
        ):
            continue

        before = str(previous)
        after = str(following)

        if re.search(r"\(\s*$", before) is None:
            continue

        match = re.match(
            r"(?P<labels>[A-Za-z]+:[A-Za-z0-9_./:-]+"
            r"(?:\s*,\s*[A-Za-z]+:[A-Za-z0-9_./:-]+)*)",
            after,
        )

        if match is None or not after[match.end():].startswith(")"):
            continue

        labels = [
            label.strip()
            for label in match.group("labels").split(",")
        ]

        if not labels or any(not label for label in labels):
            continue

        broken_refs.extend(labels)

        retained_before = re.sub(
            r"\(\s*$", "", before
        )
        retained_after = after[match.end() + 1:]

        # A source may contain ". (broken-reference).".
        # Preserve one original period rather than producing "..".
        if (
            retained_before.rstrip().endswith(".")
            and retained_after.startswith(".")
        ):
            retained_after = retained_after[1:]

        previous.replace_with(
            NavigableString(retained_before)
        )
        following.replace_with(
            NavigableString(retained_after)
        )

        error.decompose()

    tokens: list[
        tuple[str, frozenset[str], int | None, int | None]
    ] = []

    mathml_by_key: dict[int, str] = {}
    # ARXIV_PRECISE_INLINE_LINKS_V1
    link_href_by_key: dict[int, str] = {}

    block_tags = {
        "p", "ul", "ol", "li", "blockquote", "pre"
    }

    boundary_classes = {
        "ltx_listing", "ltx_listingline",
        "ltx_quote", "ltx_item",
    }

    discarded_classes = {
        "ltx_tag_section",
        "ltx_tag_subsection",
        "ltx_tag_subsubsection",
        "ltx_tag_paragraph",
        "ltx_tag_subparagraph",
    }

    def add(value, marks, math_key=None, link_key=None):
        for char in value.replace("\xa0", " "):
            tokens.append((
                " " if char.isspace() else char,
                marks,
                math_key,
                link_key,
            ))

    def walk(node, marks, link_key=None):
        if isinstance(node, NavigableString):
            add(str(node), marks, link_key=link_key)
            return

        if not isinstance(node, Tag):
            return

        classes = set(node.get("class", []))

        # Assign a distinct identity to each actual source hyperlink.
        # The final text offsets are derived from its own characters.
        if node.name == "a" and node.get("href"):
            link_key = len(link_href_by_key) + 1
            link_href_by_key[link_key] = str(node["href"])

        if (
            node.name in {"script", "style"}
            or classes & discarded_classes
        ):
            return

        if node.name == "br":
            tokens.append(("\n", marks, None, link_key))
            return

        if node.name == "math":
            latex = _math_text(node)

            if latex:
                key = len(mathml_by_key) + 1
                mathml_by_key[key] = str(node)

                add(" ", marks, link_key=link_key)
                add(latex, marks, key, link_key)
                add(" ", marks, link_key=link_key)

            return

        active = set(marks)

        if (
            node.name in {"strong", "b"}
            or "ltx_font_bold" in classes
        ):
            active.add("bold")

        if (
            node.name in {"em", "i"}
            or "ltx_font_italic" in classes
        ):
            active.add("italic")

        if (
            node.name in {"code", "tt", "pre"}
            or "ltx_font_typewriter" in classes
        ):
            active.add("code")

        css = node.get("style", "")

        if re.search(
            r"(?:^|;)\s*font-weight\s*:\s*"
            r"(?:bold|[6-9]00)\b",
            css,
            re.I,
        ):
            active.add("bold")

        if re.search(
            r"(?:^|;)\s*font-style\s*:\s*italic\b",
            css,
            re.I,
        ):
            active.add("italic")

        active_marks = frozenset(active)

        boundary = (
            node.name in block_tags
            or bool(classes & boundary_classes)
        )

        if boundary:
            tokens.append(("\n", marks, None, link_key))

        for child in node.children:
            walk(child, active_marks, link_key)

        if boundary:
            tokens.append(("\n", marks, None, link_key))

    walk(root, frozenset())

    normalized = []

    for char, marks, math_key, link_key in tokens:
        if char == "\n":
            while normalized and normalized[-1][0] == " ":
                normalized.pop()

            if sum(
                1
                for item in normalized[-2:]
                if item[0] == "\n"
            ) < 2:
                normalized.append((char, marks, None, link_key))

        elif char == " ":
            if (
                normalized
                and normalized[-1][0] not in {" ", "\n"}
            ):
                normalized.append((
                    char, marks, math_key, link_key
                ))

        else:
            normalized.append((
                char, marks, math_key, link_key
            ))

    while normalized and normalized[0][0] in {
        " ", "\n"
    }:
        normalized.pop(0)

    while normalized and normalized[-1][0] in {
        " ", "\n"
    }:
        normalized.pop()

    text = "".join(
        item[0] for item in normalized
    )

    formatting = []

    for mark in ("bold", "italic", "code"):
        beginning = None

        for index in range(len(normalized) + 1):
            active = (
                index < len(normalized)
                and mark in normalized[index][1]
            )

            if active and beginning is None:
                beginning = index

            elif not active and beginning is not None:
                formatting.append(
                    InlineFormatSpan(
                        start=beginning,
                        end=index,
                        mark=mark,
                    )
                )
                beginning = None

    inline_math = []
    current_key = None
    beginning = 0

    for index in range(len(normalized) + 1):
        key = (
            normalized[index][2]
            if index < len(normalized)
            else None
        )

        if key == current_key:
            continue

        if (
            current_key is not None
            and index > beginning
        ):
            inline_math.append(
                InlineMathSpan(
                    start=beginning,
                    end=index,
                    mathml=mathml_by_key[current_key],
                    plain_text=text[beginning:index],
                )
            )

        if key is not None:
            beginning = index

        current_key = key

    # Recover exact source positions after whitespace normalization.
    # No substring search or guessed citation position is permitted.
    link_spans: list[tuple[int, int, str]] = []
    active_link = None
    link_start = 0

    for index in range(len(normalized) + 1):
        current = (
            normalized[index][3]
            if index < len(normalized)
            else None
        )

        if current == active_link:
            continue

        if active_link is not None:
            start, end = link_start, index

            while start < end and text[start].isspace():
                start += 1

            while end > start and text[end - 1].isspace():
                end -= 1

            if start < end:
                link_spans.append((
                    start,
                    end,
                    link_href_by_key[active_link],
                ))

        if current is not None:
            link_start = index

        active_link = current

    return text, formatting, inline_math, broken_refs, link_spans


def _heading(tag: Tag) -> str:
    header = next((child for child in tag.children if isinstance(child, Tag) and _has(child, "ltx_title")), None)
    if header is None:
        header = tag.find(["h2", "h3", "h4", "h5"], recursive=False)
    return _prose(header) if header else "Untitled section"


def _metadata(soup: BeautifulSoup, bundle: HTMLBundle) -> PaperMetadata:
    title_el = soup.select_one("h1.ltx_title_document")
    title = (_prose(title_el) if title_el else None) or _meta(soup, "citation_title")
    if not title:
        raise ArxivFullTextUnavailableError("arXiv HTML has no document title")
    # Citation metadata is more consistent than individual LaTeXML creator blocks.
    authors = [_clean(m.get("content", "")) for m in soup.select('meta[name="citation_author"]')]
    authors = list(dict.fromkeys(a for a in authors if a))
    if not authors:
        authors = list(dict.fromkeys(
            _clean(a.get_text(" ", strip=True))
            for a in soup.select(".ltx_authors .ltx_personname")
            if _clean(a.get_text(" ", strip=True))
        ))
    url = bundle.source_url
    if not url:
        link = soup.find("link", rel=lambda v: v and "canonical" in v)
        url = link.get("href") if link else None
    date = _meta(soup, "citation_date") or _meta(soup, "citation_online_date") or ""
    match = re.search(r"\b(?:19|20)\d{2}\b", date)
    arxiv_id = _meta(soup, "citation_arxiv_id")
    if not arxiv_id and url:
        idmatch = re.search(r"/(?:html|abs)/(\d{4}\.\d{4,5})(?:v\d+)?", urlparse(url).path)
        arxiv_id = idmatch.group(1) if idmatch else None
    return PaperMetadata(
        title=title, source_platform="arxiv", original_url=url,
        authors=authors, doi=_meta(soup, "citation_doi"),
        year=int(match.group()) if match else None,
        publication=f"arXiv:{arxiv_id}" if arxiv_id else "arXiv",
    )


def _body(soup: BeautifulSoup) -> Tag:
    # On current arXiv LaTeXML pages, ltx_document is the converted paper;
    # site chrome (TOC, author navbar, footer) sits outside that element.
    candidates = soup.select("article.ltx_document, div.ltx_document, main.ltx_document")
    body = next((b for b in candidates if b.select_one(".ltx_section,.ltx_bibliography")), None)
    if body is None:
        body = soup.select_one(".ltx_document .ltx_body")
    if body is None or not body.select_one(".ltx_section,.ltx_subsection,.ltx_para"):
        raise ArxivFullTextUnavailableError(
            "No arXiv LaTeXML full text found. Save the arxiv.org/html/<id> "
            "page, not the /abs/<id> abstract page."
        )
    return body


def _raw_to_safe_relative(src: str) -> str | None:
    url = urlparse(src)
    if url.scheme or url.netloc or not url.path or url.path.startswith("/"):
        return None
    path = unquote(url.path).replace("\\", "/")
    components = PurePosixPath(path).parts
    if ".." in components or not components or ":" in components[0]:
        return None
    return str(PurePosixPath(path))


def parse_arxiv_bundle(bundle: HTMLBundle) -> PaperDocument:
    """Parse an arXiv LaTeXML HTMLBundle to validated PaperDocument v2."""
    soup = BeautifulSoup(bundle.html, "lxml")
    body = _body(soup)
    metadata = _metadata(soup, bundle)

    origin = metadata.original_url or ""
    if origin and "arxiv.org" not in (urlparse(origin).hostname or ""):
        raise ValueError("This adapter expects a saved arxiv.org HTML paper")

    idmatch = re.search(r"/(?:html|abs)/(\d{4}\.\d{4,5})(?:v\d+)?", urlparse(origin).path)
    stable_id = idmatch.group(1).replace(".", "_") if idmatch else hashlib.sha256(bundle.html.encode()).hexdigest()[:12]
    paper_id = f"arxiv_{stable_id}"

    sections: list[Section] = []
    elements: dict = {}
    assets: dict[str, Asset] = {}
    references: dict[str, Reference] = {}
    raw_link_map: dict[str, tuple[str, str]] = {}
    pending_mentions: list[tuple[str, str, int, int, str]] = []
    warnings: list[str] = []
    counts = {"sec": 0, "para": 0, "fig": 0, "table": 0, "equation": 0, "asset": 0, "mention": 0}
    image_inventory = {r.src: r for r in bundle.images}

    def new_id(prefix: str) -> str:
        counts[prefix] += 1
        return f"{prefix}_{counts[prefix]:04d}"

    def add_image(img: Tag | None) -> str | None:
        if not img:
            return None
        src = img.get("src") or img.get("data-src") or ""
        rel = _raw_to_safe_relative(src)
        if not rel:
            warnings.append("Image is not a local safe HTML resource: " + src[:160])
            return None
        match = image_inventory.get(src)
        aid = new_id("asset")
        status = "available" if match and match.status == "available" else "missing"
        if status != "available":
            warnings.append("Image file missing: " + rel)
        assets[aid] = Asset(asset_id=aid, original_relative_path=rel,
                            media_type=match.media_type if match else None,
                            status=status)
        return aid

    def register_html_id(tag: Tag, eid: str, kind: str) -> None:
        if tag.get("id"):
            raw_link_map[tag["id"]] = (eid, kind)

    def remember_links(
        owner_id: str,
        source_text: str,
        spans: list[tuple[int, int, str]],
    ) -> None:
        """Register links from exact normalized DOM character offsets."""

        for start, end, href in spans:
            if not (0 <= start < end <= len(source_text)):
                continue

            snippet = source_text[start:end]

            if not snippet.strip() or len(snippet) > 120:
                continue

            parsed = urlparse(href.strip())
            fragment = unquote(parsed.fragment)

            if not fragment:
                continue

            if parsed.scheme or parsed.netloc:
                if (
                    parsed.scheme != "https"
                    or parsed.hostname not in (
                        "arxiv.org",
                        "www.arxiv.org",
                    )
                ):
                    continue

                match = re.search(
                    r"/(?:html|abs)/(\d{4}\.\d{4,5})"
                    r"(?:v\d+)?/?$",
                    parsed.path,
                )

                if (
                    match is None
                    or match.group(1).replace(".", "_") != stable_id
                ):
                    continue

            pending_mentions.append((
                owner_id,
                fragment,
                start,
                end,
                snippet,
            ))

    def remember_caption_links(
        owner_id: str,
        caption_tag: Tag,
    ) -> None:
        """Preserve only caption links whose offsets can be verified.

        Captions may lose a leading Figure/Table label during extraction.
        An unmatched layout is left unlinked rather than guessed.
        """

        caption = elements[owner_id].caption

        if not caption:
            return

        full_text, _, _, _, spans = _formatted_prose(caption_tag)

        if not full_text.endswith(caption):
            return

        offset = len(full_text) - len(caption)
        adjusted = [
            (start - offset, end - offset, href)
            for start, end, href in spans
            if start >= offset and end <= len(full_text)
        ]

        remember_links(owner_id, caption, adjusted)

    def add_paragraph(tag: Tag, section: Section) -> None:
        # LaTeXML ltx_para may contain a <p>, a pseudo-code listing, lists,
        # quotes, or multiple prose siblings. Preserve each in source order.
        if _has(tag, "ltx_para"):
            parts = [ch for ch in tag.children if isinstance(ch, Tag) and (
                ch.name in ("p", "ul", "ol", "blockquote") or
                _has(ch, "ltx_listing") or _has(ch, "ltx_quote")
            )]
            if not parts:
                parts = [tag]
        else:
            parts = [tag]
        for part in parts:
            text, fmt, maths, broken_refs, link_spans = _formatted_prose(part)
            if not text:
                continue
            eid = new_id("para")
            elements[eid] = Paragraph(element_id=eid, section_id=section.section_id,
                                      text=text, format_spans=fmt, inline_math_spans=maths,
                                      unresolved_crossrefs=broken_refs,
                                      original_html_id=part.get("id") or tag.get("id"))
            section.block_ids.append(eid)
            remember_links(eid, text, link_spans)

    def add_equation(node: Tag, section: Section) -> None:
        # LaTeXML numbered equations may use a TABLE.ltx_equation, not a DIV.
        # Never split the layout <tr>/<td> into separate formula elements.
        math = node.find("math", attrs={"display": "block"}) or node.find("math")
        latex = _math_text(math) if math else None
        # Some LaTeXML equations contain multiple MathML snippets.
        if latex is None and node.get("alttext"):
            latex = node["alttext"]
        if not latex:
            warnings.append("Equation has no machine-readable LaTeX: " + str(node.get("id")))
        eid = new_id("equation")
        elements[eid] = Equation(element_id=eid, section_id=section.section_id,
                                 latex=latex, mathml=str(math) if math else None,
                                 plain_text=latex,
                                 label=_clean(node.select_one(".ltx_tag_equation").get_text(" ", strip=True)) if node.select_one(".ltx_tag_equation") else (_clean(node.select_one(".ltx_tag").get_text(" ", strip=True)) if node.select_one(".ltx_tag") else None),
                                 original_html_id=node.get("id"))
        section.block_ids.append(eid)
        register_html_id(node, eid, "equation")

    def add_figure_or_table(node: Tag, section: Section, *, force_table: bool = False) -> str | None:
        # LaTeXML can use nested <table class="ltx_tabular"> inside an
        # actual Figure, and sometimes adds ltx_table to its outer wrapper.
        # Figure IDs (A2.F4) and caption tags are more specific than the
        # presence of a layout table.
        node_id = node.get("id", "")
        # Two or more side-by-side numbered TABLES sometimes sit inside a
        # single ltx_table wrapper whose children are ltx_figure_panel.
        # Emit the independently captioned tabular panels rather than a phantom
        # Figure or a single merged table. A reference to the outer wrapper
        # links to the matching numbered table if it exists.
        panels = [part for part in node.select("figure.ltx_figure_panel")
                  if part.select_one("table.ltx_tabular") and part.select_one(".ltx_caption")]
        if _has(node, "ltx_table") and panels and not node.select_one(":scope > table.ltx_tabular"):
            candidates = []
            for panel in panels:
                added = add_figure_or_table(panel, section, force_table=True)
                if added:
                    candidates.append((panel, added))
            if candidates and node_id:
                num = re.search(r"\.T(\d+)(?:$|\.)", node_id)
                wanted = num.group(1) if num else None
                chosen = next((eid for panel, eid in candidates
                               if wanted and re.search(r"\bTable\s*" + wanted + r"\b",
                                   _prose(panel.select_one(".ltx_caption")), re.I)),
                              candidates[-1][1])
                raw_link_map[node_id] = (chosen, "table")
            return None
        caption_tag = node.select_one(".ltx_caption")
        caption_class = set(caption_tag.select_one(".ltx_tag").get("class", [])) if caption_tag and caption_tag.select_one(".ltx_tag") else set()
        is_table = force_table or _has(node, "ltx_table") or "ltx_tag_table" in caption_class or (
            bool(re.search(r"\.T\d+(?:\.|$)", node_id)) and "ltx_tag_figure" not in caption_class)
        if "ltx_tag_figure" in caption_class and not force_table:
            is_table = False
        caption = _prose(caption_tag) if caption_tag else ""
        label_m = re.match(r"\s*((?:Fig(?:ure)?|Table)\s*\d+[A-Za-z]?)\s*[.:]?\s*", caption, re.I)
        label = label_m.group(1) if label_m else None
        if label_m:
            caption = caption[label_m.end():]
        # A Figure can be wholly inline SVG (GraphRAG has several) or have
        # multiple visual subpanels. Those are rasterized/composited by the
        # arXiv-only exporter into one safe PNG. Do not drop SVG-only figures.
        media_svgs = node.find_all("svg")
        media_imgs = node.find_all("img")
        media_objects = [o for o in node.find_all("object")
                         if o.get("type", "").lower().strip() == "image/svg+xml"]
        # arXiv may link .svg externally rather than embedding <svg>.
        # Such assets require sanitization, not a raw SVG copy.
        svg_file = any(
            image_inventory.get(img.get("src", "")) is not None and
            image_inventory[img.get("src", "")].media_type == "image/svg+xml"
            for img in media_imgs
        )
        composite = (not is_table and (bool(media_svgs) or bool(media_objects) or len(media_imgs) > 1 or svg_file))
        img = media_imgs[0] if media_imgs else None
        aid = None if composite else add_image(img)
        if composite:
            # Virtual source marker: exporter renders paper-only SVG/image panels.
            # Exporter resolves it only via the matching Figure source ID.
            virtual_aid = new_id("asset")
            assets[virtual_aid] = Asset(
                asset_id=virtual_aid,
                original_relative_path=f"__arxiv_render__/{virtual_aid}.svg",
                status="missing", media_type="image/svg+xml",
            )
            aid = virtual_aid
        if is_table:
            grid_tag = node.find("table")
            if grid_tag:
                cells = []
                occupied: set[tuple[int, int]] = set()
                for ridx, tr in enumerate(grid_tag.find_all("tr")):
                    colidx = 0
                    for col in tr.find_all(["td", "th"], recursive=False):
                        while (ridx, colidx) in occupied:
                            colidx += 1
                        try:
                            colspan = max(1, int(col.get("colspan", "1")))
                            rowspan = max(1, int(col.get("rowspan", "1")))
                        except ValueError:
                            colspan = rowspan = 1
                        # LaTeXML may use <td class=ltx_th> instead of <th>.
                        role = "column_header" if col.name == "th" or _has(col, "ltx_th") else "data"
                        image_aid = None
                        if col.find(["img", "object", "svg"]):
                            # Virtual, source-local visual for an image-containing
                            # cell. The renderer only imports a sanitized SVG or
                            # supported raster; source HTML/JS is never embedded.
                            if not col.get("id"):
                                warnings.append("Table cell media lacks source ID: " + node_id)
                            else:
                                image_aid = new_id("asset")
                                assets[image_aid] = Asset(
                                    asset_id=image_aid,
                                    original_relative_path=f"__arxiv_render__/{image_aid}.svg",
                                    status="missing", media_type="image/svg+xml")
                        cells.append(TableCell(cell_id=f"cell_{counts['table']+1:04d}_{len(cells)+1:04d}",
                                               original_html_id=col.get("id"), image_asset_id=image_aid,
                                               row=ridx, col=colidx, text=_prose(col),
                                               role=role, colspan=colspan, rowspan=rowspan))
                        for r in range(ridx, ridx + rowspan):
                            for c in range(colidx, colidx + colspan):
                                occupied.add((r, c))
                        colidx += colspan
                grid = TableGrid(cells=cells) if cells else None
            else:
                grid = None
            if grid is None and aid is None:
                warnings.append("Cannot preserve table without cells or image: " + str(node.get("id")))
                return
            rep = "hybrid" if grid and aid else "structured" if grid else "image_only"
            eid = new_id("table")
            elements[eid] = Table(element_id=eid, section_id=section.section_id,
                                  label=label, caption=caption, representation=rep,
                                  grid=grid, image_asset_id=aid,
                                  recognition_status="not_needed" if grid else "not_attempted",
                                  original_html_id=node.get("id"))
            section.block_ids.append(eid)
            register_html_id(node, eid, "table")
            if caption_tag is not None:
                remember_caption_links(eid, caption_tag)
            return eid
        else:
            eid = new_id("fig")
            elements[eid] = Figure(element_id=eid, section_id=section.section_id,
                                   label=label, caption=caption, image_asset_id=aid,
                                   alt_text=img.get("alt") if img else None,
                                   original_html_id=node.get("id"))
            section.block_ids.append(eid)
            register_html_id(node, eid, "figure")
            if caption_tag is not None:
                remember_caption_links(eid, caption_tag)
            if not aid:
                warnings.append("Figure has no local image: " + str(node.get("id")))
            return eid

    section_roles = {"ltx_section": 1, "ltx_subsection": 2,
                     "ltx_subsubsection": 3, "ltx_paragraph": 4,
                     "ltx_subparagraph": 5, "ltx_appendix": 1,
                     "ltx_chapter": 1, "ltx_part": 1}

    def read_child(tag: Tag, section: Section) -> None:
        cls = set(tag.get("class", []))
        if any(k in cls for k in section_roles):
            read_section(tag, parent=section.section_id)
        elif "ltx_listing" in cls or "ltx_quote" in cls:
            add_paragraph(tag, section)
        elif "ltx_para" in cls:
            # A LaTeXML paragraph wrapper can contain P, numbered equation
            # TABLEs, lists and arbitrary block siblings. Handle each in
            # source order; unlike a broad .get_text() this preserves formula
            # IDs, numbers and precise citation positions.
            children = [c for c in tag.children if isinstance(c, Tag)]
            if not children:
                add_paragraph(tag, section)
            else:
                for child in children:
                    if child.name == "p" and "ltx_p" in child.get("class", []):
                        add_paragraph(child, section)
                    else:
                        read_child(child, section)
        elif tag.name == "p" and "ltx_p" in cls:
            add_paragraph(tag, section)
        elif "ltx_equationgroup" in cls:
            subequations = tag.select(".ltx_equation")
            if subequations:
                for item in subequations:
                    if item.find_parent(class_="ltx_equation") is None:
                        add_equation(item, section)
            else:
                add_equation(tag, section)
        elif "ltx_equation" in cls:
            add_equation(tag, section)
        elif "ltx_figure" in cls or ("ltx_table" in cls and tag.name not in ("table",)):
            add_figure_or_table(tag, section)
        elif "ltx_bibliography" in cls or "ltx_abstract" in cls:
            return
        elif tag.name in ("ul", "ol") and "ltx_biblist" not in cls:
            for li in tag.find_all("li", recursive=False):
                add_paragraph(li, section)
        elif tag.name in ("div", "section", "article") or "ltx_theorem" in cls:
            for child in tag.find_all(recursive=False):
                if isinstance(child, Tag):
                    read_child(child, section)

    def read_section(tag: Tag, parent: str | None) -> Section:
        count = len(sections)+1
        role_level = next((num for name, num in section_roles.items() if _has(tag, name)), 1)
        level = role_level
        # Hierarchical level is determined by LaTeXML's section role, not DOM depth.
        if parent:
            parent_sec = next((s for s in sections if s.section_id == parent), None)
            if parent_sec:
                level = max(parent_sec.level+1, role_level)
        sid = f"sec_{count:03d}"
        sec = Section(section_id=sid, title=_heading(tag), level=level,
                      parent_section_id=parent)
        sections.append(sec)
        for child in tag.find_all(recursive=False):
            if isinstance(child, Tag) and not _has(child, "ltx_title"):
                read_child(child, sec)
        return sec

    abstract = body.select_one(".ltx_abstract")
    if abstract:
        sec = Section(section_id="sec_001", title="Abstract", level=1)
        sections.append(sec)
        for para in abstract.select(".ltx_para"):
            if not para.find_parent(".ltx_para"):
                add_paragraph(para, sec)
        if not sec.block_ids:
            for p in abstract.select("p.ltx_p"):
                add_paragraph(p, sec)
    # Walk top-level wrappers, preserving reading order; ignore front matter and
    # TOC. Do not convert bibliography into ordinary prose.
    def read_root(tag: Tag) -> None:
        cls = set(tag.get("class", []))
        if any(c in cls for c in section_roles):
            read_section(tag, parent=None)
        elif cls & {"ltx_abstract", "ltx_bibliography", "ltx_front", "ltx_toc", "ltx_toclist"}:
            return
        elif tag.name in ("div", "main", "article", "section"):
            for child in tag.find_all(recursive=False):
                if isinstance(child, Tag):
                    read_root(child)

    for top in body.find_all(recursive=False):
        if isinstance(top, Tag):
            read_root(top)

    if not sections or not any(s.block_ids for s in sections):
        raise ArxivFullTextUnavailableError("No nonempty arXiv sections were extracted")

    for item in body.select(".ltx_bibitem"):
        rawid = item.get("id")
        if not rawid:
            continue
        ref_id = f"ref_{len(references)+1:03d}"
        clone = BeautifulSoup(str(item), "lxml")
        for tag in clone.select(".ltx_tag_bibitem"):
            tag.decompose()
        raw_text = _prose(clone)
        if not raw_text or rawid in raw_link_map:
            continue
        references[ref_id] = Reference(reference_id=ref_id, raw_text=raw_text,
                                       original_html_id=rawid)
        raw_link_map[rawid] = (ref_id, "reference")

    mentions: list[Mention] = []

    for owner_id, raw_target, start, end, snippet in pending_mentions:
        owner = elements[owner_id]

        if isinstance(owner, Paragraph):
            source_text = owner.text
        elif isinstance(owner, (Figure, Table)):
            source_text = owner.caption
        else:
            continue

        # Never accept an anchor computed from a different text string.
        if (
            not 0 <= start < end <= len(source_text)
            or source_text[start:end] != snippet
        ):
            continue

        target = raw_link_map.get(unquote(raw_target))

        kind = (
            target[1]
            if target
            else (
                "reference"
                if "bib" in raw_target.lower()
                else None
            )
        )

        if kind not in {
            "figure", "table", "equation", "reference"
        }:
            continue

        selector = TextSpanSelector(
            start=start,
            end=end,
            exact=snippet,
        )

        mid = new_id("mention")

        mentions.append(
            Mention(
                mention_id=mid,
                source_anchor=SourceAnchor(
                    element_id=owner_id,
                    selector=selector,
                ),
                mention_text=snippet,
                target_kind=kind,
                target_id=target[0] if target else None,
                resolution_status=(
                    "resolved" if target else "unresolved"
                ),
            )
        )

    # A present bibliography with 0 matched citations must never be labeled
    # complete when the source contains bibliographic links.
    has_citation_links = any("bib." in raw_target.lower() for _, raw_target, *_ in pending_mentions)
    linked_refs = [m for m in mentions if m.target_kind == "reference"]
    reference_quality = ("missing" if not references else
                         "partial" if (has_citation_links and not linked_refs) or any(
                             m.resolution_status != "resolved" for m in linked_refs
                         ) else "complete")
    if reference_quality == "missing":
        warnings.append("No structured arXiv bibliography found")
    elif reference_quality == "partial":
        warnings.append("Some in-text references could not be linked to bibliography entries")

    figures = [x for x in elements.values() if isinstance(x, Figure)]
    tables = [x for x in elements.values() if isinstance(x, Table)]
    fig_quality = ("complete" if all(f.image_asset_id and assets[f.image_asset_id].status == "available" for f in figures)
                   else "partial")
    tbl_quality = ("complete" if all(t.grid is not None and all(
        not cell.image_asset_id or assets[cell.image_asset_id].status == "available"
        for cell in t.grid.cells) for t in tables) else "partial")
    unresolved_count = sum(
        len(element.unresolved_crossrefs)
        for element in elements.values()
        if isinstance(element, Paragraph)
    )
    if unresolved_count:
        warnings.append(
            f"{unresolved_count} broken LaTeXML reference labels "
            "were preserved as paragraph metadata."
        )

    # Include normalized document content in the paper revision.
    revision = "sha256:"+hashlib.sha256(bundle.html.encode("utf-8")).hexdigest()
    paper = PaperDocument(
        paper_id=paper_id, revision=revision, metadata=metadata,
        sections=sections, elements=elements, references=references,
        mentions=mentions, assets=assets,
        quality=PaperQuality(body="complete", figures=fig_quality,
                             tables=tbl_quality, references=reference_quality,
                             warnings=warnings),
    )

    digest = hashlib.sha256()
    digest.update(b"arxiv-source-normalization-v2")
    digest.update(bundle.html.encode("utf-8"))
    digest.update(
        paper.model_dump_json(
            exclude={"revision"}
        ).encode("utf-8")
    )
    paper.revision = "sha256:" + digest.hexdigest()
    return paper
