"""Render PaperDocument v2 as safe, standalone, read-only HTML.

The source is structured PaperDocument data, not untrusted publisher HTML.
Original website JS/CSS is never included or executed.
"""
from __future__ import annotations

from collections import defaultdict
from html import escape
from typing import Iterable, Mapping

from app.models.document_v2 import (
    Equation, Figure, Mention, PaperDocument, Paragraph, Table,
    TableCellsSelector, TextSpanSelector, InlineFormatSpan, InlineMathSpan,
)


STYLE = """
:root{color-scheme:light;--text:#17212f;--muted:#56657a;--line:#dbe1ea;--accent:#2067ae}
*{box-sizing:border-box}body{background:#f5f6f8;color:var(--text);margin:0;
font:16px/1.72 Georgia,'Times New Roman',serif}
main{max-width:880px;margin:36px auto;padding:46px 58px;background:white;
box-shadow:0 2px 24px #25344712;border-radius:8px}
header{border-bottom:1px solid var(--line);padding-bottom:25px;margin-bottom:35px}
header h1{font:700 31px/1.28 Georgia,serif;margin:0 0 12px}
header .metadata{font:14px/1.6 system-ui,sans-serif;color:var(--muted)}
section{margin:22px 0}h2,h3,h4,h5,h6{line-height:1.35;color:#17212f}
h2{font-size:24px;margin:32px 0 14px}h3{font-size:20px;margin:27px 0 12px}
h4{font-size:18px;margin:23px 0 10px}
p.paper-paragraph{margin:12px 0;white-space:pre-wrap;overflow-wrap:anywhere}
p.paper-paragraph .paper-code{font:0.91em/1.7 Consolas,'Courier New',monospace;
white-space:inherit;overflow-wrap:anywhere}
p.paper-paragraph strong{font-weight:700}
p.paper-paragraph em{font-style:italic}
.native-inline-math{display:inline;white-space:nowrap;vertical-align:baseline}
.native-inline-math math{font-size:1em;vertical-align:middle}
figure{margin:24px 0;padding:14px 18px;border:1px solid var(--line);border-radius:7px}
figure img{display:block;max-width:100%;height:auto;margin:0 auto 12px}
figcaption{font:14px/1.6 system-ui,sans-serif;color:var(--muted);margin-top:10px}
figcaption img.inline-caption-image{display:inline-block;vertical-align:middle;max-width:32px;height:auto;margin:0 2px}
.figure-label{font-weight:700;color:var(--text)}
.figure-open-original{display:block;margin:7px 0 5px;font:13px system-ui,sans-serif;text-align:center}
.asset-missing{font:14px system-ui,sans-serif;color:#9b402b;background:#fff6f0;padding:12px}
.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font:14px/1.5 system-ui,sans-serif}
th,td{border:1px solid #cfd9e3;padding:7px 10px;text-align:left}
th{background:#edf3f8;font-weight:600}
.equation{margin:20px 0;overflow-x:auto;background:#f7f8fa;border-radius:6px;padding:13px 16px}
.native-mathml{overflow-x:auto;text-align:center;padding:8px 0}.native-mathml math{font-size:1.15em}
.equation pre{white-space:pre-wrap;word-break:break-word;font-size:14px;margin:0}
.equation img.equation-svg{display:block;max-width:100%;width:auto;height:auto;margin:0 auto}
.equation .fallback-note{font:12px system-ui,sans-serif;color:var(--muted);margin-bottom:6px}
.equation .label{float:right;font:13px system-ui,sans-serif;color:var(--muted)}
.source-mention{border-bottom:1px dotted #6f85a4;text-decoration:none;color:inherit}
a.source-mention{color:var(--accent);cursor:pointer}
.quality-notice{border-left:3px solid #c19048;background:#fffaf1;padding:12px 16px;
font:13px/1.6 system-ui,sans-serif;color:#66502b}
.paper-references{padding-left:24px;overflow-wrap:anywhere}
.paper-references li{margin:6px 0}footer{font:12px/1.6 system-ui,sans-serif;
border-top:1px solid var(--line);color:var(--muted);margin-top:40px;padding-top:15px}
@media(max-width:700px){main{margin:0;padding:26px 20px;border-radius:0}header h1{font-size:25px}}
"""


def e(text: object | None) -> str:
    return escape('' if text is None else str(text), quote=True)


def _mention_markup(text: str, mentions: Iterable[Mention], paper: PaperDocument) -> str:
    """Highlight only exact, non-overlapping spans measured in Paragraph.text."""
    candidates = []
    for mention in mentions:
        selector = mention.source_anchor.selector
        if not isinstance(selector, TextSpanSelector):
            continue
        start, end = selector.start, selector.end
        if end > len(text) or text[start:end] != selector.exact:
            continue
        candidates.append((start, end, mention))
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    result = []
    cursor = 0
    for start, end, mention in candidates:
        if start < cursor:
            continue
        result.append(e(text[cursor:start]))
        escaped = e(text[start:end])
        if mention.resolution_status == 'resolved' and mention.target_id:
            # Both target IDs and anchor selectors were validated by PaperDocument.
            result.append(
                f'<a class="source-mention" href="#{e(mention.target_id)}" '
                f'data-mention-id="{e(mention.mention_id)}" '
                f'title="{e(mention.target_kind)} reference">{escaped}</a>'
            )
        else:
            result.append(
                f'<span class="source-mention" data-mention-id="{e(mention.mention_id)}" '
                f'data-resolution="unresolved">{escaped}</span>'
            )
        cursor = end
    result.append(e(text[cursor:]))
    return ''.join(result)



def _paragraph_markup(text: str, mentions: Iterable[Mention],
                      formatting: Iterable[InlineFormatSpan], paper: PaperDocument,
                      inline_math: Iterable[InlineMathSpan] = ()) -> str:
    """Compose exact citation highlights and safe inline typography by offsets.

    No publisher HTML is inserted in the reader; text is always escaped.
    Offsets stay aligned with Paragraph.text, including preserved line breaks.
    """
    candidates = []
    for mention in mentions:
        selector = mention.source_anchor.selector
        if isinstance(selector, TextSpanSelector) and selector.end <= len(text) and text[selector.start:selector.end] == selector.exact:
            candidates.append((selector.start, selector.end, mention))
    candidates.sort(key=lambda r: (r[0], -(r[1] - r[0])))
    accepted = []
    current_end = 0
    for start, end, mention in candidates:
        if start >= current_end:
            accepted.append((start, end, mention))
            current_end = end
    styles = [s for s in formatting if 0 <= s.start < s.end <= len(text)]
    maths = sorted((s for s in inline_math if 0 <= s.start < s.end <= len(text) and
                    text[s.start:s.end] == s.plain_text), key=lambda s: s.start)
    cuts = {0, len(text)}
    for start, end, _ in accepted:
        cuts.update((start, end))
    for span in styles:
        cuts.update((span.start, span.end))
    for span in maths:
        cuts.update((span.start, span.end))
    boundaries = sorted(cuts)
    out = []
    tags = {"bold": ("<strong>", "</strong>"), "italic": ("<em>", "</em>"),
            "code": ('<code class="paper-code">', "</code>"),
            "sup": ("<sup>", "</sup>"), "sub": ("<sub>", "</sub>")}
    for i, begin in enumerate(boundaries[:-1]):
        end = boundaries[i+1]
        if begin == end:
            continue
        active = {span.mark for span in styles if span.start <= begin < span.end}
        math_span = next((m for m in maths if m.start <= begin < m.end), None)
        if math_span is not None and begin != math_span.start:
            # A typography boundary may fall inside MathML's fallback text.
            # The math is emitted once, whole, at the first segment.
            continue
        if math_span is not None:
            safe = _safe_mathml(math_span.mathml, display=False)
            chunk = ('<span class="native-inline-math">' + safe + '</span>') if safe else e(math_span.plain_text)
        else:
            chunk = e(text[begin:end])
        for mark in ("code", "italic", "bold", "sup", "sub"):
            if mark in active:
                a, b = tags[mark]
                chunk = a + chunk + b
        mention = next((m for st, en, m in accepted if st <= begin < en), None)
        if mention is not None:
            if mention.resolution_status == "resolved" and mention.target_id:
                chunk = (f'<a class="source-mention" href="#{e(mention.target_id)}" '
                         f'data-mention-id="{e(mention.mention_id)}">{chunk}</a>')
            else:
                chunk = (f'<span class="source-mention" data-mention-id="{e(mention.mention_id)}" '
                         f'data-resolution="unresolved">{chunk}</span>')
        out.append(chunk)
    return "".join(out)


def _table_markup(table: Table, paper: PaperDocument) -> str:
    if not table.grid or not table.grid.cells:
        return '<div class="asset-missing">原始表格只有标题；独立表格页面数据尚未获取（partial）。</div>'
    rows = defaultdict(list)
    for cell in table.grid.cells:
        rows[cell.row].append(cell)
    content = ['<div class="table-scroll"><table>']
    for row_idx in sorted(rows):
        content.append('<tr>')
        for cell in sorted(rows[row_idx], key=lambda item: item.col):
            is_header = cell.role != 'data'
            tag = 'th' if is_header else 'td'
            scope = (' scope="col"' if cell.role == 'column_header' else
                     ' scope="row"' if cell.role == 'row_header' else '')
            image = (_image_markup(cell.image_asset_id, paper, cell.text or 'Table cell image')
                     if cell.image_asset_id else '')
            content.append(
                f'<{tag} id="{e(cell.cell_id)}" data-row="{cell.row}" '
                f'data-col="{cell.col}" rowspan="{cell.rowspan}" '
                f'colspan="{cell.colspan}"{scope}>{image}{e(cell.text)}</{tag}>'
            )
        content.append('</tr>')
    content.append('</table></div>')
    return '\n'.join(content)


def _safe_mathml(raw: str | None, *, display: bool = True) -> str | None:
    """Render a tiny inert MathML whitelist; never reuse publisher tags/JS."""
    if not raw or len(raw) > 150_000:
        return None
    from bs4 import BeautifulSoup, NavigableString, Tag
    import re
    allowed = {'math', 'semantics', 'mrow', 'mi', 'mo', 'mn', 'mtext',
               'msub', 'msup', 'msubsup', 'mfrac', 'msqrt', 'mroot', 'mtable',
               'mtr', 'mtd', 'munder', 'mover', 'munderover', 'mphantom',
               'mpadded', 'mspace', 'menclose', 'mmultiscripts', 'none'}
    attrs = {'display', 'stretchy', 'accent', 'accentunder', 'columnalign',
             'columnspacing', 'rowspacing', 'displaystyle', 'scriptlevel',
             'movablelimits', 'mathvariant', 'lspace', 'rspace', 'mathsize',
             'linethickness', 'minsize', 'maxsize'}
    soup = BeautifulSoup(raw, 'lxml')
    root = soup.find('math')
    if root is None:
        return None
    def render(node):
        if isinstance(node, NavigableString):
            return e(str(node))
        if not isinstance(node, Tag) or node.name not in allowed:
            return ''
        av = ''.join(f' {name}="{e(str(value))}"' for name, value in node.attrs.items()
                     if name in attrs and not (node.name == 'math' and name == 'display')
                     and isinstance(value, str) and
                     re.fullmatch(r'[A-Za-z0-9, .+\-_]*(?:em|px|pt|%)?', value))
        if node.name == 'math':
            av += ' xmlns="http://www.w3.org/1998/Math/MathML"'
            if display:
                av += ' display="block"'
        return '<' + node.name + av + '>' + ''.join(render(n) for n in node.children) + '</' + node.name + '>'
    return render(root)


def _image_markup(asset_id: str | None, paper: PaperDocument, alt: str) -> str:
    if not asset_id or asset_id not in paper.assets:
        return '<div class="asset-missing">原始论文图片未提供。</div>'
    asset = paper.assets[asset_id]
    if asset.status != 'available' or not asset.normalized_relative_path:
        return '<div class="asset-missing">该图片资源未成功复制，已保留原始引用信息。</div>'
    return f'<img loading="lazy" src="{e(asset.normalized_relative_path)}" alt="{e(alt)}">'




def _caption_markup(figure: Figure, paper: PaperDocument) -> str:
    """Safe caption typography and tiny inline assets, offset-aligned to caption."""
    content = figure.caption
    icons = sorted(figure.caption_inline_assets, key=lambda s: s.start)
    fmt = [s for s in figure.caption_format_spans if s.end <= len(content)]
    cuts = {0, len(content)}
    for span in [*icons, *fmt]:
        cuts.update((span.start, span.end))
    out = []
    tags = {'bold': ('<strong>', '</strong>'), 'italic': ('<em>', '</em>'),
            'code': ('<code>', '</code>'), 'sup': ('<sup>', '</sup>'),
            'sub': ('<sub>', '</sub>')}
    points = sorted(cuts)
    for i, start in enumerate(points[:-1]):
        end = points[i+1]
        icon = next((a for a in icons if a.start <= start < a.end), None)
        if icon:
            asset = paper.assets.get(icon.asset_id)
            if asset and asset.status == 'available' and asset.normalized_relative_path:
                # One complete img element per inline icon; it should not be split.
                if start == icon.start:
                    out.append('<img class="inline-caption-image" src="' +
                               e(asset.normalized_relative_path) + '" alt="' +
                               e(icon.alt_text or 'caption symbol') + '">')
            else:
                out.append(e(content[start:end]))
            continue
        segment = e(content[start:end])
        styles = {f.mark for f in fmt if f.start <= start < f.end}
        for mark in ('code', 'italic', 'bold', 'sup', 'sub'):
            if mark in styles:
                open_tag, close_tag = tags[mark]
                segment = open_tag + segment + close_tag
        out.append(segment)
    return ''.join(out)


def render_normalized_html(
    paper: PaperDocument,
    equation_assets: Mapping[str, str] | None = None,
) -> str:
    """Generate a static document; must be safe to open without publisher scripts."""
    equation_assets = equation_assets or {}
    mention_index: dict[str, list[Mention]] = defaultdict(list)
    for mention in paper.mentions:
        mention_index[mention.source_anchor.element_id].append(mention)

    output = [
        '<!doctype html>', '<html lang="en"><head>', '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f'<title>{e(paper.metadata.title)}</title>', f'<style>{STYLE}</style>',
        '</head><body><main>', '<header>', f'<h1>{e(paper.metadata.title)}</h1>',
    ]
    info = []
    if paper.metadata.authors:
        info.append(e(', '.join(paper.metadata.authors)))
    if paper.metadata.publication:
        info.append(e(paper.metadata.publication))
    if paper.metadata.year:
        info.append(e(paper.metadata.year))
    if paper.metadata.doi:
        info.append('DOI: ' + e(paper.metadata.doi))
    info.append('Source: ' + e(paper.metadata.source_platform.upper()))
    output.append('<div class="metadata">' + ' · '.join(info) + '</div></header>')

    if paper.quality.body != 'complete' or paper.quality.references == 'missing':
        notes = []
        if paper.quality.body != 'complete':
            notes.append('正文完整性尚未确认。')
        if paper.quality.references == 'missing':
            notes.append('此 HTML 没有完整参考文献列表；文中引用保留，但无法链接到文献条目。')
        output.append('<aside class="quality-notice">' + ' '.join(notes) + '</aside>')

    section_stack: list[str] = []
    for section in paper.sections:
        while section_stack and section_stack[-1] != section.parent_section_id:
            output.append('</section>')
            section_stack.pop()
        if section.parent_section_id is not None and (
            not section_stack or section_stack[-1] != section.parent_section_id
        ):
            raise ValueError(f'Section order is not hierarchical: {section.section_id}')
        heading = f'h{min(section.level + 1, 6)}'
        output.append(f'<section id="{e(section.section_id)}" data-level="{section.level}">')
        output.append(f'<{heading}>{e(section.title)}</{heading}>')
        section_stack.append(section.section_id)
        for block_id in section.block_ids:
            block = paper.elements[block_id]
            if isinstance(block, Paragraph):
                body = _paragraph_markup(block.text, mention_index.get(block_id, []),
                                         block.format_spans, paper, block.inline_math_spans)
                output.append(f'<p class="paper-paragraph" id="{e(block_id)}" '
                              f'data-source-id="{e(block_id)}">{body}</p>')
            elif isinstance(block, Figure):
                image = _image_markup(block.image_asset_id, paper,
                                      block.alt_text or block.caption or block.label or 'Figure')
                if paper.metadata.source_platform == 'springer':
                    asset = paper.assets.get(block.image_asset_id) if block.image_asset_id else None
                    if asset and asset.status == 'available' and asset.normalized_relative_path:
                        original = e(asset.normalized_relative_path)
                        is_preview = any(
                            w.startswith('Springer figure preview-only: ' + str(block.original_html_id) + ';')
                            for w in paper.quality.warnings
                        )
                        display_label = ('查看已保存预览图（清晰度有限）' if is_preview
                                         else '查看已保存高清图（单独打开，可放大）')
                        image += (f'<a class="figure-open-original" href="{original}" '
                                  f'target="_blank" rel="noopener">{display_label}</a>')
                output.append(f'<figure id="{e(block_id)}" data-kind="figure">{image}'
                              f'<figcaption><span class="figure-label">{e(block.label)}</span> '
                              f'{_caption_markup(block, paper)}</figcaption></figure>')
            elif isinstance(block, Table):
                parts = [f'<figure id="{e(block_id)}" data-kind="table" '
                         f'data-representation="{e(block.representation)}">']
                if block.representation in ('structured', 'hybrid'):
                    parts.append(_table_markup(block, paper))
                if block.representation in ('image_only', 'hybrid'):
                    if block.representation == 'hybrid':
                        parts.append('<details><summary>查看原始表格图片</summary>')
                    parts.append(_image_markup(block.image_asset_id, paper,
                                               block.caption or block.label or 'Table'))
                    if block.representation == 'hybrid':
                        parts.append('</details>')
                parts.append(f'<figcaption><span class="figure-label">{e(block.label)}</span> '
                             f'{e(block.caption)}</figcaption></figure>')
                output.append(''.join(parts))
            elif isinstance(block, Equation):
                content = block.latex or block.plain_text or block.mathml or ''
                path = equation_assets.get(block_id)
                mathml = _safe_mathml(block.mathml) if paper.metadata.source_platform in ('arxiv', 'acm', 'springer') else None
                if mathml:
                    # Chrome/Edge have native MathML support. This is a
                    # sanitized subset, not active publisher HTML.
                    visible = '<div class="native-mathml">' + mathml + '</div>'
                elif path:
                    visible = (f'<img class="equation-svg" src="{e(path)}" '
                               f'alt="{e(content)}">')
                else:
                    visible = ('<div class="fallback-note">公式未能自动排版，显示原始表达式：</div>'
                               f'<pre>{e(content)}</pre>')
                output.append(f'<div class="equation" id="{e(block_id)}" data-kind="equation">'
                              f'<span class="label">{e(block.label)}</span>'
                              f'{visible}</div>')
    output.extend('</section>' for _ in section_stack)

    if paper.references:
        output.append('<section id="references"><h2>References</h2><ol class="paper-references">')
        for reference in paper.references.values():
            output.append(f'<li id="{e(reference.reference_id)}">{e(reference.raw_text)}</li>')
        output.append('</ol></section>')
    output.extend([
        f'<footer>Document ID: {e(paper.paper_id)} · Revision: {e(paper.revision)}',
        ' · Generated from user-provided HTML. Scripts from the original site are excluded.</footer>',
        '</main></body></html>',
    ])
    return '\n'.join(output)