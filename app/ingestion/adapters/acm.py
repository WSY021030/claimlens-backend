"""ACM Digital Library HTML full-text adapter (PaperDocument v2).

Designed around ACM's semantic bodymatter/backmatter document structure, not
specific DOIs, titles, numbered figures, or journal/conference layout.

Offline only. Neither publisher scripts nor CSS are executed/reused. The
browser's duplicate figure viewer / bibliography sidebars are excluded.
"""
from __future__ import annotations

from collections import Counter
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit
import hashlib
import re
import copy

from bs4 import BeautifulSoup, NavigableString, Tag

from app.ingestion.loader import HTMLBundle
from app.models.document_v2 import (
    Asset, ElementSelector, Equation, Figure, InlineFormatSpan, Mention,
    PaperDocument, PaperMetadata, PaperQuality, Paragraph, Reference, Section, InlineAssetSpan, InlineMathSpan,
    SourceAnchor, Table, TableCell, TableGrid, TableCellsSelector, TextSpanSelector,
)


class ACMFullTextUnavailableError(ValueError):
    """Saved ACM page does not have actual full-text HTML bodymatter."""


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _meta(soup: BeautifulSoup, *, name: str | None = None, prop: str | None = None) -> str | None:
    tag = soup.find("meta", attrs=({"name": name} if name else {"property": prop}))
    return _clean(tag.get("content", "")) or None if tag else None


def _valid_rel_path(src: str) -> str | None:
    if not src or src.startswith(("data:", "blob:")):
        return None
    u = urlsplit(src)
    if u.scheme or u.netloc or u.path.startswith(("/", "\\")):
        return None
    value = unquote(u.path).replace("\\", "/")
    parts = [p for p in PurePosixPath(value).parts if p not in (".", "")]
    if not parts or ".." in parts or any(":" in x for x in parts):
        return None
    return "/".join(parts)


def _ref_id(anchor: Tag) -> str | None:
    """Prefer ACM's stable reference target, not its presentation URL."""
    rid = anchor.get("data-xml-rid", "")
    if re.fullmatch(r"Bib\d+", rid, re.I):
        return rid
    # Fallback for snapshots that omit data-xml-rid.
    frag = unquote(urlsplit(anchor.get("href", "")).fragment)
    if anchor.get("role") == "doc-biblioref":
        m = re.search(r"(?:^|-)\b(Bib\d+)\b", frag, re.I)
        if m:
            return m.group(1)
    # Figure/table/equation links use ordinary fragment hrefs. Section and
    # footnote navigational links are deliberately not treated as evidence.
    if re.fullmatch(r"(?:fig(?:ure)?|uid|tab(?:le)?|tbl|eq(?:uation)?|eqn)[A-Za-z0-9_.-]*", frag, re.I):
        return "XREF:" + frag
    return None




def _math_plain(node: Tag) -> str:
    """Readable, deterministic MathML fallback (not an invented LaTeX string).

    Browser-saved ACM pages frequently contain MathJax's assistive MathML.
    The visual CHTML glyphs have no text; this representation preserves
    variables and numerical values for indexing and table-cell evidence.
    """
    if not isinstance(node, Tag):
        return ""
    name = node.name
    kids = [x for x in node.children if isinstance(x, Tag)]
    if name in {"mn", "mi", "mo", "mtext"}:
        return node.get_text("", strip=True)
    if name == "msub" and len(kids) >= 2:
        return _math_plain(kids[0]) + "_(" + _math_plain(kids[1]) + ")"
    if name == "msup" and len(kids) >= 2:
        return _math_plain(kids[0]) + "^(" + _math_plain(kids[1]) + ")"
    if name == "msubsup" and len(kids) >= 3:
        return (_math_plain(kids[0]) + "_(" + _math_plain(kids[1]) + ")^("
                + _math_plain(kids[2]) + ")")
    if name == "mfrac" and len(kids) >= 2:
        return "(" + _math_plain(kids[0]) + ")/(" + _math_plain(kids[1]) + ")"
    if name == "msqrt":
        return "sqrt(" + "".join(_math_plain(x) for x in kids) + ")"
    if name == "mroot" and len(kids) >= 2:
        return "root(" + _math_plain(kids[0]) + "," + _math_plain(kids[1]) + ")"
    if name == "mover" and len(kids) >= 2:
        base, over = _math_plain(kids[0]), _math_plain(kids[1]).strip()
        # Human/agent-readable *fallback*: the reader renders source MathML.
        if over in {"―", "¯", "‾", "_", "̄", "−", "-"}:
            return "bar(" + base + ")"
        if over in {"^", "ˆ", "̂", "∧"}:
            return "hat(" + base + ")"
        return "overset(" + over + "," + base + ")"
    if name == "munder" and len(kids) >= 2:
        return "underset(" + _math_plain(kids[1]) + "," + _math_plain(kids[0]) + ")"
    if name == "munderover" and len(kids) >= 3:
        return "underoverset(" + ",".join(_math_plain(x) for x in (kids[1], kids[2], kids[0])) + ")"
    return "".join(_math_plain(x) for x in kids)

def _contents(node: Tag, *, multiline: bool = False, inline_image_placeholder: bool = False,
              inline_math: list[InlineMathSpan] | None = None):
    """Text, typography spans, and exact bibliography-anchor spans.

    All offset ranges are calculated *after* whitespace normalization so they
    always align with Paragraph.text. Publisher/extension HTML is never stored
    in the format spans.
    """
    raw: list[tuple[str, frozenset[str], int | None, str | None, int | None]] = []
    anchor_serial = 0
    math_serial = 0
    math_markup: dict[int, str] = {}
    block = {"p", "li", "blockquote", "pre"}

    def char_add(text: str, marks: frozenset[str], ix: int | None,
                 rid: str | None, mid: int | None = None):
        for c in text.replace("\xa0", " "):
            raw.append((" " if c.isspace() else c, marks, ix, rid, mid))

    def rec(n, marks: frozenset[str], aidx: int | None = None, rid: str | None = None):
        nonlocal anchor_serial, math_serial
        if isinstance(n, NavigableString):
            char_add(str(n), marks, aidx, rid)
            return
        if not isinstance(n, Tag):
            return
        name = n.name or ""
        classes = set(n.get("class", []))
        # Extract only the *assistive* MathML under MathJax, never its
        # generated visual glyph layout or scripts. This preserves inline
        # expressions in prose and numbers/variables inside table cells.
        if name == "mjx-container" or name == "math":
            math = n.select_one("mjx-assistive-mml math") if name == "mjx-container" else n
            if math is not None:
                mid = None
                if inline_math is not None:
                    math_serial += 1
                    mid = math_serial
                    # Only the original/assistive MathML tree is retained,
                    # never MathJax's generated visual glyph HTML.
                    math_markup[mid] = str(math)
                char_add(_math_plain(math), marks, aidx, rid, mid)
            return
        if name in {"script", "style", "noscript", "template", "button", "svg", "mjx-assistive-mml"} or (
            "MathJax" in classes or "MathJax_Preview" in classes or
            n.get("aria-hidden") == "true"
        ):
            return
        if name == "br":
            raw.append(("\n", marks, None, None, None))
            return
        # Inline icon in a caption: retain a visible indication rather than
        # quietly deleting it or fabricating a meaning.
        if name == "img":
            label = _clean(n.get("alt") or n.get("title") or "")
            char_add(" [inline graphic] " if inline_image_placeholder or not label else label, marks, None, None)
            return
        new_marks = set(marks)
        if name in {"b", "strong"} or re.search(r"(?:^|;)\s*font-weight\s*:\s*(?:bold|[6-9]00)\b", n.get("style", ""), re.I):
            new_marks.add("bold")
        if name in {"i", "em"} or re.search(r"(?:^|;)\s*font-style\s*:\s*italic\b", n.get("style", ""), re.I):
            new_marks.add("italic")
        if name in {"pre", "code", "tt", "kbd"}:
            new_marks.add("code")
        if name == "sup":
            new_marks.add("sup")
        if name == "sub":
            new_marks.add("sub")
        new_marks = frozenset(new_marks)
        if name == "a":
            new_rid = _ref_id(n)
            if new_rid:
                anchor_serial += 1
                aidx, rid = anchor_serial, new_rid
        new_block = multiline and (name in block or n.get("role") in {"listitem", "paragraph"})
        if new_block:
            raw.append(("\n", marks, None, None, None))
        for c in n.children:
            rec(c, new_marks, aidx, rid)
        if new_block:
            raw.append(("\n", marks, None, None, None))

    rec(node, frozenset())
    compact: list[tuple[str, frozenset[str], int | None, str | None, int | None]] = []
    for tok in raw:
        c = tok[0]
        if c == " ":
            if compact and compact[-1][0] not in {" ", "\n"}:
                compact.append(tok)
        elif c == "\n":
            while compact and compact[-1][0] == " ":
                compact.pop()
            if compact and compact[-1][0] != "\n":
                compact.append(tok)
        else:
            compact.append(tok)
    while compact and compact[0][0] in {" ", "\n"}:
        compact.pop(0)
    while compact and compact[-1][0] in {" ", "\n"}:
        compact.pop()
    text = "".join(x[0] for x in compact)
    spans: list[InlineFormatSpan] = []
    for mark in ("bold", "italic", "code", "sup", "sub"):
        start = None
        for i in range(len(compact) + 1):
            has = i < len(compact) and mark in compact[i][1]
            if has and start is None:
                start = i
            if not has and start is not None:
                spans.append(InlineFormatSpan(start=start, end=i, mark=mark))
                start = None
    cites: list[tuple[int, int, str]] = []
    start = None
    previous = None
    for i in range(len(compact) + 1):
        value = (compact[i][2], compact[i][3]) if i < len(compact) else (None, None)
        if value != previous:
            if previous is not None and previous[0] is not None and start is not None:
                cites.append((start, i, previous[1]))
            start = i if value[0] is not None else None
            previous = value
    if inline_math is not None:
        current = None
        start = 0
        for idx in range(len(compact) + 1):
            mid = compact[idx][4] if idx < len(compact) else None
            if mid != current:
                if current is not None and idx > start:
                    fallback = text[start:idx]
                    inline_math.append(InlineMathSpan(
                        start=start, end=idx, mathml=math_markup[current],
                        plain_text=fallback))
                current = mid
                start = idx
    return text, spans, cites


def _find_metadata(soup: BeautifulSoup, bundle: HTMLBundle) -> PaperMetadata:
    title = (_meta(soup, name="dc.Title") or _meta(soup, name="citation_title") or
             (_clean(soup.h1.get_text(" ", strip=True)) if soup.h1 else None) or
             (_meta(soup, prop="og:title") or "Untitled ACM article"))
    doi = (_meta(soup, name="publication_doi") or
           next((_clean(x.get("content", "")) for x in soup.find_all("meta", attrs={"name": "dc.Identifier"})
                 if x.get("scheme", "").lower() == "doi"), None) or
           _meta(soup, name="citation_doi"))
    # ACM's dc.Creator often merges given/family names without a space.
    # Prefer the visible canonical author links and use metadata only as a
    # fallback, never include hidden profile-dialog duplicate authors.
    authors = []
    for author in soup.select('[property="author"][role="listitem"] > .dropBlock > a[title]'):
        value = _clean(author.get("title", ""))
        if value and value not in authors:
            authors.append(value)
    if not authors:
        for name in ("citation_author", "dc.Creator"):
            for meta in soup.find_all("meta", attrs={"name": name}):
                value = _clean(meta.get("content", ""))
                if value and value not in authors:
                    authors.append(value)
            if authors:
                break
    date = (_meta(soup, name="dc.Date") or _meta(soup, name="citation_date") or
            _meta(soup, name="citation_publication_date"))
    year_match = re.search(r"\b(?:19|20)\d{2}\b", date or "")
    publication = (_meta(soup, name="citation_journal_title") or
                   _meta(soup, name="citation_conference_title"))
    if publication is None:
        pub = soup.select_one("meta[property='og:site_name']")
        publication = pub.get("content") if pub else None
    return PaperMetadata(title=title, source_platform="acm", original_url=bundle.source_url,
                         doi=doi, year=int(year_match.group()) if year_match else None,
                         authors=authors, publication=publication)


def parse_acm_bundle(bundle: HTMLBundle) -> PaperDocument:
    """Parse a browser-saved ACM HTML Full Text page into PaperDocument v2."""
    soup = BeautifulSoup(bundle.html, "lxml")
    body = soup.select_one("#bodymatter[data-extent='bodymatter']")
    if body is None or not body.select("section") or len(body.select("[role='paragraph']")) < 2:
        raise ACMFullTextUnavailableError(
            "ACM HTML Full Text bodymatter was not found. Save a full article, "
            "not an abstract-only DOI page or PDF landing page."
        )
    metadata = _find_metadata(soup, bundle)
    if not metadata.doi:
        raise ACMFullTextUnavailableError("ACM HTML has no DOI metadata")
    safe_doi = re.sub(r"[^a-zA-Z0-9]+", "_", metadata.doi).strip("_")
    paper_id = f"acm_{safe_doi}"
    revision = "sha256:" + hashlib.sha256(bundle.html.encode("utf-8")).hexdigest()

    # Only the canonical bibliography in backmatter; the right-hand
    # collateral panel contains *duplicates* and navigation links.
    references: dict[str, Reference] = {}
    for node in soup.select("#backmatter #bibliography [id^='Bib']"):
        rid = node.get("id", "")
        if not re.fullmatch(r"Bib\d+", rid, re.I) or rid in references:
            continue
        content = node.select_one(".citation-content") or node
        raw = _clean(content.get_text(" ", strip=True))
        if not raw:
            continue
        year_text = re.search(r"\b(?:19|20)\d{2}\b", raw)
        references[rid] = Reference(reference_id=rid, raw_text=raw,
                                    year=int(year_text.group()) if year_text else None,
                                    original_html_id=rid)

    sections: list[Section] = []
    section_of: dict[int, str] = {}
    element_by_id = {}
    assets: dict[str, Asset] = {}
    mentions: list[Mention] = []
    html_to_element = {}
    element_counts: Counter = Counter()
    warnings: list[str] = []
    pending_xrefs: list[tuple[int, str]] = []

    def create_section(key: str, title: str, level: int, parent: str | None = None) -> str:
        section_id = "S-" + re.sub(r"[^A-Za-z0-9_-]", "_", key)
        if any(x.section_id == section_id for x in sections):
            raise ValueError(f"Duplicate ACM section ID {section_id}")
        sections.append(Section(section_id=section_id, title=title, level=level,
                                parent_section_id=parent))
        return section_id

    abstract = soup.select_one("#abstracts #abstract")
    if abstract:
        atext = abstract.select_one("[role='paragraph']")
        if atext:
            abs_id = create_section("abstract", "Abstract", 1)
            math_spans: list[InlineMathSpan] = []
            text, fmt, cite = _contents(atext, inline_math=math_spans)
            if text:
                pid = f"{abs_id}.P1"
                element_by_id[pid] = Paragraph(element_id=pid, section_id=abs_id,
                                               text=text, format_spans=fmt,
                                               inline_math_spans=math_spans)
                sections[-1].block_ids.append(pid)
                # Abstract references are rare but still traceable.
                for start, end, rid in cite:
                    mentions.append(Mention(mention_id=f"M{len(mentions)+1}",
                        source_anchor=SourceAnchor(element_id=pid,
                          selector=TextSpanSelector(start=start, end=end, exact=text[start:end])),
                        mention_text=text[start:end], target_kind="reference",
                        target_id=rid if rid in references else None,
                        resolution_status="resolved" if rid in references else "unresolved"))

    # Some ACM papers put a lead Figure before Section 1. Pre-create a real
    # front-content section IN DOCUMENT ORDER, rather than appending it after
    # all normal sections on first encounter.
    front_section_id = None
    leading = [x for x in body.select("figure.graphic, div[role='paragraph'], table")
               if x.find_parent("section") is body]
    if leading:
        front_section_id = create_section("front-content", "Front matter", 1)

    # Allocate sections in source order; nesting reflects ACM's actual section DOM.
    for n, item in enumerate(body.select("section"), start=1):
        if item is body:
            continue
        heading = item.find(re.compile(r"h[1-6]"), recursive=False)
        if heading is None:
            continue
        parent_node = item.find_parent("section", id=True)
        parent = section_of.get(id(parent_node)) if parent_node and parent_node is not body else None
        level = (next((x.level for x in sections if x.section_id == parent), 0) + 1) if parent else 1
        sid = create_section(item.get("id") or f"section_{n}",
                             _clean(heading.get_text(" ", strip=True)), level, parent)
        section_of[id(item)] = sid
    if not section_of:
        raise ACMFullTextUnavailableError("ACM bodymatter has no valid section headings")

    s_lookup = {x.section_id: x for x in sections}
    def active_section(tag: Tag) -> str:
        nonlocal front_section_id
        for parent in (tag, *tag.parents):
            if isinstance(parent, Tag) and id(parent) in section_of:
                return section_of[id(parent)]
            if parent is body:
                break
        if front_section_id is None:
            front_section_id = create_section("front-content", "Front matter", 1)
            s_lookup[front_section_id] = sections[-1]
        return front_section_id

    def new_element_id(sid: str, code: str) -> str:
        element_counts[(sid, code)] += 1
        return f"{sid}.{code}{element_counts[(sid, code)]}"

    def add_mentions(source_id: str, text: str, cites, *, caption: bool = False,
                     cell_id: str | None = None):
        for st, en, rid in cites:
            if st >= en or not text[st:en].strip():
                continue
            is_xref = rid.startswith("XREF:")
            resolved = not is_xref and rid in references
            selector = (TableCellsSelector(cell_ids=[cell_id]) if cell_id else
                        ElementSelector() if caption else
                        TextSpanSelector(start=st, end=en, exact=text[st:en]))
            mentions.append(Mention(mention_id=f"M{len(mentions)+1}",
                source_anchor=SourceAnchor(element_id=source_id, selector=selector),
                mention_text=text[st:en], target_kind="figure" if is_xref else "reference",
                target_id=rid if resolved else None,
                resolution_status="resolved" if resolved else "unresolved"))
            if is_xref:
                pending_xrefs.append((len(mentions)-1, rid[5:]))

    def make_asset(src: str) -> str | None:
        rel = _valid_rel_path(src)
        if not rel:
            return None
        aid = f"A{len(assets)+1}"
        matched = next((x for x in bundle.images if x.src == src), None)
        assets[aid] = Asset(asset_id=aid, original_relative_path=rel,
                            media_type=matched.media_type if matched else None,
                            status="available" if matched and matched.status == "available" else "missing")
        return aid

    def add_paragraph(tag: Tag, *, multiline=False, sid_override: str | None = None):
        sid = sid_override or active_section(tag)
        math_spans: list[InlineMathSpan] = []
        value, fmt, cites = _contents(tag, multiline=multiline, inline_math=math_spans)
        if not value:
            return
        # An ACM semantic list item is not the same as a navigation list.
        if not multiline and tag.find_parent(attrs={"role": "listitem"}):
            value = "• " + value
            fmt = [f.model_copy(update={"start": f.start + 2, "end": f.end + 2}) for f in fmt]
            cites = [(s+2, e+2, rid) for s, e, rid in cites]
            math_spans = [s.model_copy(update={"start":s.start+2,"end":s.end+2}) for s in math_spans]
        pid = new_element_id(sid, "P")
        element_by_id[pid] = Paragraph(element_id=pid, section_id=sid, text=value,
                                       format_spans=fmt, inline_math_spans=math_spans,
                                       original_html_id=tag.get("id"))
        s_lookup[sid].block_ids.append(pid)
        add_mentions(pid, value, cites)

    def add_figure(tag: Tag):
        sid = active_section(tag)
        fid = new_element_id(sid, "F")
        header = tag.find_parent(class_="figure-wrap")
        lbl = header.select_one("header .core-label") if header else None
        label = _clean(lbl.get_text(" ", strip=True)) if lbl else _clean(tag.get("aria-label", ""))
        if tag.get("data-type") == "algorithm":
            ident = tag.get("id", "")
            num = re.search(r"(?:algorithm|algo)[-_]?(\d+)", ident, re.I)
            label = f"Algorithm {num.group(1)}" if num else "Algorithm"
        caption_node = tag.find("figcaption")
        caption, caption_format, cites = _contents(caption_node, multiline=True, inline_image_placeholder=True) if caption_node else ("", [], [])
        primary = (tag.find("img", recursive=False) or tag.find("object", recursive=False) or
                   (tag.find("img") if tag.get("data-type") == "algorithm" else None))
        src = primary.get("src") if primary and primary.name == "img" else None
        if primary and primary.name == "object":
            src = primary.get("data")
        asset_id = make_asset(src) if src else None
        if not asset_id:
            warnings.append(f"Figure lacks a safe local image: {tag.get('id') or fid}")
        # Inline caption graphics are actual evidence (Y/N symbols, etc.).
        # Span offsets point to neutral text placeholders, while source image
        # bytes are exported as ordinary vetted assets.
        cap_icons = []
        cursor = 0
        if caption_node:
            for img in caption_node.select("img"):
                needle = "[inline graphic]"
                start = caption.find(needle, cursor)
                if start < 0:
                    warnings.append(f"Could not locate caption graphic text: {tag.get('id') or fid}")
                    break
                aid = make_asset(img.get("src", ""))
                if aid:
                    cap_icons.append(InlineAssetSpan(start=start, end=start + len(needle),
                         asset_id=aid, alt_text=_clean(img.get("alt") or img.get("title") or "")))
                else:
                    warnings.append(f"Caption has unsafe/unavailable inline image: {tag.get('id') or fid}")
                cursor = start + len(needle)
        element_by_id[fid] = Figure(element_id=fid, section_id=sid,
                                    label=label or None, caption=caption,
                                    caption_inline_assets=cap_icons,
                                    caption_format_spans=caption_format,
                                    image_asset_id=asset_id,
                                    alt_text=(_clean(primary.get("alt", "")) if primary else "") or (label or None),
                                    original_html_id=tag.get("id"))
        s_lookup[sid].block_ids.append(fid)
        if tag.get("id"):
            html_to_element[tag["id"]] = fid
        add_mentions(fid, caption, cites, caption=True)

    def add_table(tag: Tag):
        sid = active_section(tag)
        tid = new_element_id(sid, "T")
        wrapper = tag.find_parent(class_="table-wrap")
        container = tag.find_parent("figure", class_="table")
        lbl = wrapper.select_one(".core-label") if wrapper else None
        label = _clean(lbl.get_text(" ", strip=True)) if lbl else None
        if not label and container:
            m = re.search(r"(?:tab|table)[-_]?(\d+)", container.get("id", ""), re.I)
            label = f"Table {m.group(1)}" if m else "Table"
        cap = (tag.find("caption") or (container.select_one("figcaption") if container else None) or
               (wrapper.select_one(".caption, figcaption") if wrapper else None))
        caption, _, caption_cites = _contents(cap, multiline=True) if cap else ("", [], [])
        add_mentions(tid, caption, caption_cites, caption=True)
        cells = []
        row_nodes = tag.select("tr")
        occupied = set()
        for rowidx, tr in enumerate(row_nodes):
            col = 0
            for c in tr.find_all(["th", "td"], recursive=False):
                while (rowidx, col) in occupied:
                    col += 1
                try:
                    rowspan = max(1, int(c.get("rowspan") or 1))
                    colspan = max(1, int(c.get("colspan") or 1))
                except ValueError:
                    rowspan = colspan = 1
                for ri in range(rowidx, rowidx+rowspan):
                    for ci in range(col, col+colspan):
                        occupied.add((ri, ci))
                value, _, cites = _contents(c)
                assetid = make_asset(c.find("img").get("src", "")) if c.find("img") else None
                cellid = f"{tid}.R{rowidx}C{col}"
                add_mentions(tid, value, cites, cell_id=cellid)
                cells.append(TableCell(cell_id=cellid, row=rowidx, col=col, text=value,
                                       role="column_header" if c.name == "th" else "data",
                                       rowspan=rowspan, colspan=colspan,
                                       image_asset_id=assetid, original_html_id=c.get("id")))
                col += colspan
        if not cells:
            warnings.append(f"Table has no readable cells: {tag.get('id') or tid}")
        element_by_id[tid] = Table(element_id=tid, section_id=sid, label=label,
                                   caption=caption, representation="structured",
                                   grid=TableGrid(cells=cells), recognition_status="not_needed",
                                   original_html_id=tag.get("id"))
        s_lookup[sid].block_ids.append(tid)
        for htmlid in (tag.get("id"), container.get("id") if container else None):
            if htmlid:
                html_to_element[htmlid] = tid

    def add_equation(tag: Tag):
        sid = active_section(tag)
        eid = new_element_id(sid, "E")
        math = tag if tag.name == "math" else tag.find("math")
        ann = math.select_one('annotation[encoding="application/x-tex"]') if math else None
        latex = (ann.get_text("", strip=True) if ann else
                 tag.get("data-tex") or tag.get("data-latex"))
        # It is preferable to preserve original MathML rather than run
        # javascript generated MathJax layout. The reader sanitizes MathML.
        mathml = str(math) if math else None
        label_node = tag.select_one(".eqno, .equation-label, .label")
        label = (tag.get("data-label") or
                 (_clean(label_node.get_text(" ", strip=True)) if label_node else None))
        if not label and tag.get("id"):
            m = re.fullmatch(r"(?:eq|equation|eqn)[-_]?(\d+)", tag["id"], re.I)
            if m:
                label = f"({m.group(1)})"
        if not latex and not mathml:
            warnings.append(f"Equation lacks machine-readable representation: {tag.get('id') or eid}")
        element_by_id[eid] = Equation(element_id=eid, section_id=sid,
                                      label=label, latex=latex or None, mathml=mathml,
                                      plain_text=_math_plain(math)[:800] if math else _clean(tag.get_text(" ", strip=True))[:800],
                                      original_html_id=tag.get("id"))
        s_lookup[sid].block_ids.append(eid)
        if tag.get("id"):
            html_to_element[tag["id"]] = eid

    def walk(tag):
        if not isinstance(tag, Tag):
            return
        if tag.name in {"script", "style", "noscript", "nav", "aside", "template"}:
            return
        if tag.name in {"mjx-container", "mjx-assistive-mml"} or "MathJax" in tag.get("class", []):
            return
        if tag.name == "section" and tag is not body and id(tag) not in section_of:
            return
        if tag.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            return
        if tag.name == "figure":
            if "table" in tag.get("class", []):
                for actual in tag.select(".table-wrap > table"):
                    add_table(actual)
            elif ("graphic" in tag.get("class", []) or
                  (tag.get("data-type") == "algorithm" and tag.find("img")) or
                  (tag.find("img", recursive=False) and "boxed-text" not in tag.get("class", []))):
                add_figure(tag)
            else:
                add_paragraph(tag, multiline=True)
            return
        if tag.name == "table":
            # Only genuine scholarly tables, not browser/CSS layout tables.
            add_table(tag)
            return
        if (tag.get("data-type") in {"disp-formula", "equation"} or
            "display-formula" in tag.get("class", []) or
            ("equation" in tag.get("class", []) and tag.find("math")) or
            (tag.name == "math" and tag.get("display") == "block")):
            add_equation(tag)
            return
        if tag.get("role") == "paragraph" or (tag.name == "p" and not tag.find("div", attrs={"role":"paragraph"})):
            # ACM frequently embeds a whole .figure-wrap *inside* a semantic
            # paragraph. Process surrounding prose as separate fragments,
            # maintaining DOM order, rather than consuming the figure as text.
            nested = [c for c in tag.children if isinstance(c, Tag) and
                      (c.select_one("figure.graphic, figure.table, figure[data-type='algorithm'], table, [data-type='disp-formula'], .display-formula") or
                       c.name in {"figure", "table"} or
                       "display-formula" in c.get("class", []) or
                       c.get("data-type") in {"disp-formula", "equation"})]
            if nested:
                sid = active_section(tag)
                group = []
                def flush():
                    if not group:
                        return
                    wrapper = BeautifulSoup("<div></div>", "lxml").div
                    for child in group:
                        wrapper.append(copy.copy(child))
                    add_paragraph(wrapper, sid_override=sid)
                    group.clear()
                for child in tag.children:
                    if child in nested:
                        flush()
                        walk(child)
                    else:
                        group.append(child)
                flush()
            else:
                add_paragraph(tag)
            return
        for child in tag.children:
            walk(child)

    walk(body)
    # Non-reference backmatter carries scientifically relevant footnotes.
    for block in soup.select("#backmatter #footnotes"):
        sid = create_section("footnotes", "Footnotes", 1)
        s_lookup[sid] = sections[-1]
        for item in block.select("[role='paragraph'], li, p"):
            if item.find_parent(attrs={"role": "paragraph"}) and item.get("role") != "paragraph":
                continue
            math_spans: list[InlineMathSpan] = []
            text, fmt, cites = _contents(item, inline_math=math_spans)
            if not text:
                continue
            pid = new_element_id(sid, "P")
            element_by_id[pid] = Paragraph(element_id=pid, section_id=sid,
                                           text=text, format_spans=fmt,
                                           inline_math_spans=math_spans,
                                           original_html_id=item.get("id"))
            s_lookup[sid].block_ids.append(pid)
            add_mentions(pid, text, cites)

    if not element_by_id:
        raise ACMFullTextUnavailableError("ACM full text has no readable document blocks")

    for index, html_id in pending_xrefs:
        elem_id = html_to_element.get(html_id)
        if elem_id is not None:
            mentions[index] = mentions[index].model_copy(update={
                "target_id": elem_id,
                "target_kind": element_by_id[elem_id].kind,
                "resolution_status": "resolved",
            })
    missing_citations = sum(1 for x in mentions if x.target_kind == "reference" and
                            x.resolution_status == "unresolved")
    missing_xrefs = sum(1 for x in mentions if x.target_kind != "reference" and
                        x.resolution_status == "unresolved")
    if missing_xrefs:
        warnings.append(f"Unresolved figure/table/equation links: {missing_xrefs}")
    if missing_citations:
        warnings.append(f"Unresolved bibliography links: {missing_citations}")
    if not references:
        warnings.append("No bibliography entries in saved HTML; try opening References before saving")
    source_tables = len(body.select(".table-wrap > table"))
    source_equations = len(body.select(".display-formula, [data-type='disp-formula']"))
    source_figures = len(body.select('figure.graphic, figure[data-type="algorithm"]:has(img)'))
    parsed_figures = sum(x.kind == "figure" for x in element_by_id.values())
    parsed_tables = sum(x.kind == "table" for x in element_by_id.values())
    parsed_equations = sum(x.kind == "equation" for x in element_by_id.values())
    if parsed_figures < source_figures:
        warnings.append(f"ACM figure/algorithm extraction incomplete: {parsed_figures}/{source_figures}")
    if parsed_tables < source_tables:
        warnings.append(f"ACM table extraction incomplete: {parsed_tables}/{source_tables}")
    if parsed_equations < source_equations:
        warnings.append(f"ACM equation extraction incomplete: {parsed_equations}/{source_equations}")
    quality = PaperQuality(
        body="partial" if parsed_equations < source_equations else "complete" if body.select("section") and any(x.kind == "paragraph" for x in element_by_id.values()) else "partial",
        figures="partial" if parsed_figures < source_figures or any(
                    x.kind == "figure" and (not x.image_asset_id or
                    assets[x.image_asset_id].status != "available" or
                    any(assets[ic.asset_id].status != "available" for ic in x.caption_inline_assets))
                    for x in element_by_id.values()) or
                any("Caption has unsafe" in w or "Could not locate caption" in w for w in warnings) else "complete",
        tables="partial" if parsed_tables < source_tables or any("Table has no readable cells" in w for w in warnings) else "complete",
        references="complete" if references and missing_citations == 0 else "partial",
        warnings=list(dict.fromkeys(warnings)),
    )
    doc = PaperDocument(paper_id=paper_id, revision=revision, metadata=metadata,
                        sections=sections, elements=element_by_id,
                        references=references, mentions=mentions,
                        assets=assets, quality=quality)
    # Distinguish original DOM link counts from extracted mention counts.
    original_citations = len(body.select('a[data-xml-rid^="Bib"], a[role="doc-biblioref"]'))
    matched = len([x for x in doc.mentions if x.target_kind == "reference"])
    if matched < original_citations:
        doc.quality.references = "partial"
        doc.quality.warnings.append(
            f"Some in-body bibliography links not captured: {matched}/{original_citations}")
    return doc