"""SpringerLink saved full-text HTML to PaperDocument v2.

Target: Springer Nature Link (link.springer.com) journal article HTML, with
satellite table pages. No scripts or publisher CSS are copied/executed.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit
from bs4 import BeautifulSoup, NavigableString, Tag

from app.ingestion.loader import HTMLBundle
from app.ingestion.springer_tables import collect_satellite_tables, fetch_satellite_table, fetch_tables_in_edge
from app.ingestion.springer_figures import FigureTarget, recover_springer_figures
from app.models.document_v2 import (
    Asset, Equation, Figure, Mention, PaperDocument, PaperMetadata,
    PaperQuality, Paragraph, Reference, Section, SourceAnchor,
    Table, TableGrid, TextSpanSelector, InlineMathSpan, InlineFormatSpan
)


def _norm(s: str) -> str:
    return re.sub(r'\s+', ' ', s.replace('\xa0',' ')).strip()


def _meta(soup, name):
    t = soup.find('meta', attrs={'name':name})
    return _norm(t.get('content','')) if t else ''


def _safe_src(src):
    if not src: return None
    p = urlsplit(src)
    if p.scheme or p.netloc or p.path.startswith(('/','\\')):return None
    parts=PurePosixPath(unquote(p.path).replace('\\','/')).parts
    if not parts or any(x=='..' or ':' in x for x in parts):return None
    return '/'.join(x for x in parts if x not in ('','.'))


def _ref_for_anchor(a: Tag) -> tuple[str,str] | None:
    p=urlsplit(a.get('href',''))
    fragment=unquote(p.fragment)
    if not fragment:return None
    if a.get('data-test')=='citation-ref' and re.fullmatch(r'ref-CR\d+',fragment,re.I):
        return ('reference',fragment)
    if re.fullmatch(r'Fig\d+',fragment,re.I):return ('figure',fragment)
    if re.fullmatch(r'Tab\d+',fragment,re.I):return ('table',fragment)
    if re.fullmatch(r'Equ\d+',fragment,re.I):return ('equation',fragment)
    return None


def _math_data(tag: Tag):
    """Prefer original TeX script and MathJax's inert assistive MathML."""
    script=tag.select_one('script[type^="math/tex"]')
    latex=script.string.strip() if script and script.string else None
    math=tag.select_one('.MJX_Assistive_MathML math')
    if math is not None:
        mathml=str(math)
    else:
        svg=tag.select_one('[data-mathml]')
        mathml=svg.get('data-mathml') if svg else None
    if mathml and '<math' not in mathml:
        mathml=None
    return latex,mathml


def _paragraph_tokens(node:Tag):
    """Return canonical paragraph text, rich spans and source refs by offsets."""
    chars=[]
    maths={}
    math_n=0
    ref_n=0
    def put(value,styles,anchor,math_id=None):
        for c in value.replace('\xa0',' '):
            chars.append(((' ' if c.isspace() else c),styles,anchor,math_id))
    def rec(n,styles=frozenset(), anchor=None):
        nonlocal math_n,ref_n
        if isinstance(n,NavigableString):
            put(str(n),styles,anchor)
            return
        if not isinstance(n,Tag):return
        name=n.name or ''
        classes=set(n.get('class',[]))
        if name in ('script','style','noscript','template','svg','button'):
            return
        if name=='br':put(' ',styles,anchor);return
        if 'mathjax-tex' in classes:
            latex,mathml=_math_data(n)
            if latex or mathml:
                plain=latex or _norm(BeautifulSoup(mathml,'lxml').get_text(' ',strip=True))
                math_n+=1
                maths[math_n]=(plain,mathml)
                put(plain,styles,anchor,math_n)
            return
        if name=='math':
            math_n+=1
            plain=_norm(n.get_text(' ',strip=True))
            maths[math_n]=(plain,str(n))
            put(plain,styles,anchor,math_n)
            return
        if 'MathJax' in ''.join(classes) or 'MJX_' in ''.join(classes) or name=='mjx-container':
            return
        if name=='img':put(_norm(n.get('alt','')) or '[inline graphic]',styles,anchor);return
        new=set(styles)
        if name in ('b','strong'):new.add('bold')
        if name in ('i','em'):new.add('italic')
        if name=='sup':new.add('sup')
        if name=='sub':new.add('sub')
        if name in ('pre','code','tt'):new.add('code')
        if name=='a':
            m=_ref_for_anchor(n)
            if m:
                ref_n+=1
                anchor=(ref_n,m[0],m[1])
        for c in n.children:
            rec(c,frozenset(new),anchor)
    rec(node)
    cleaned=[]
    for tok in chars:
        if tok[0]==' ' and (not cleaned or cleaned[-1][0]==' '):continue
        cleaned.append(tok)
    while cleaned and cleaned[-1][0]==' ':cleaned.pop()
    text=''.join(x[0]for x in cleaned)
    formats=[]
    for mark in ('bold','italic','code','sup','sub'):
        start=None
        for i in range(len(cleaned)+1):
            flag=i<len(cleaned) and mark in cleaned[i][1]
            if flag and start is None:start=i
            if not flag and start is not None:
                formats.append(InlineFormatSpan(start=start,end=i,mark=mark));start=None
    mathspans=[]
    ids=sorted({x[3] for x in cleaned if x[3] is not None})
    for mid in ids:
        subset=[i for i,x in enumerate(cleaned) if x[3]==mid]
        if not subset:continue
        begin,end=min(subset),max(subset)+1
        if end-begin!=len(subset):continue
        plain,mathml=maths[mid]
        if mathml and text[begin:end]==plain:
            mathspans.append(InlineMathSpan(start=begin,end=end,plain_text=plain,mathml=mathml))
    anchors=[]
    for aid,kind,key in {x[2]for x in cleaned if x[2] is not None}:
        idx=[i for i,x in enumerate(cleaned) if x[2] is not None and x[2][0]==aid]
        if idx:
            begin,end=min(idx),max(idx)+1
            if text[begin:end].strip():anchors.append((begin,end,kind,key))
    anchors.sort()
    return text,formats,mathspans,anchors


def parse_springer_bundle(bundle:HTMLBundle, *, fetch_tables:bool=False,
                          table_files:list[str]|None=None, browser_tables:bool=False,
                          browser_figures:bool=False) -> PaperDocument:
    soup=BeautifulSoup(bundle.html,'lxml')
    main=soup.select_one('main .c-article-body .main-content')
    if main is None or main.select_one('.c-article-section__content') is None:
        raise ValueError('Not a saved SpringerLink full-text article')
    doi=_meta(soup,'citation_doi').lower()
    if not re.fullmatch(r'10\.\d{4,9}/[\w.()/-]+',doi):
        raise ValueError('Invalid Springer article DOI')
    title=_meta(soup,'citation_title') or (soup.title.get_text(' ',strip=True) if soup.title else '')
    if not title:raise ValueError('Missing Springer article title')
    url=f'https://link.springer.com/article/{doi}'
    if bundle.source_url and not bundle.source_url.rstrip('/').lower().endswith('/'+doi):
        raise ValueError('HTML canonical DOI mismatch')
    authors=[x.get('content','')for x in soup.select('meta[name="citation_author"]') if x.get('content')]
    if not authors:authors=[x.get('content','')for x in soup.select('meta[name="dc.creator"]')if x.get('content')]
    date=_meta(soup,'citation_online_date') or _meta(soup,'dc.date')
    year=int(date[:4]) if re.match(r'^\d{4}',date) else None
    paper_id='springer_'+re.sub(r'[^a-z0-9]+','_',doi.lower()).strip('_')
    metadata=PaperMetadata(title=title,source_platform='springer',original_url=url,
                           doi=doi,authors=authors,year=year,publication=_meta(soup,'citation_journal_title') or None)
    refs={}
    for x in soup.select('.c-article-references .c-article-references__text[id^="ref-CR"]'):
        rid=x.get('id')
        if rid and rid not in refs:
            refs[rid]=Reference(reference_id=rid,raw_text=_norm(x.get_text(' ',strip=True)),
                                original_html_id=rid)
    local_tables, warns=collect_satellite_tables(bundle.html_path,doi,explicit_files=table_files)
    if browser_tables:
        # Includes tables in appendices, which sit outside .main-content.
        source_tables = {int(m.group(1)) for t in soup.select('.c-article-table[id]')
                         if (m := re.fullmatch(r'table-(\d+)', t.get('id', '')))}
        pending = sorted(source_tables - set(local_tables))
        cache = bundle.html_path.parent / '_springer_tables_cache'
        recovered, browser_warnings = fetch_tables_in_edge(doi, pending, cache_dir=cache)
        local_tables.update(recovered)
        warns.extend(browser_warnings)
    figure_targets=[]
    for node in soup.select('.c-article-section__figure'):
        sid=node.get('id') or 'figure-abstract'
        im=node.select_one('img')
        if not im:continue
        original=urlsplit(im.get('src','')).path.rsplit('/',1)[-1]
        if not original:continue
        m=re.fullmatch(r'figure-(\d+)',sid)
        figure_targets.append(FigureTarget(source_id=sid,figure_num=int(m.group(1)) if m else None,
                                           filename=unquote(original),preview_src=im.get('src','')))
    highres,figure_warnings=recover_springer_figures(bundle,doi,figure_targets,browser=browser_figures)
    warns.extend(figure_warnings)
    sections=[];els={};mentions=[];assets={}; target_by_source={};target_by_alias={}

    seq=0; refseq=0; seen_tables={}; seen_figs={};seen_equations={}
    current=None;stack=[]
    def new_section(title,level,htmlid=None):
        nonlocal current
        while stack and stack[-1].level >= level:stack.pop()
        parent=stack[-1].section_id if stack else None
        sid=f'section-{len(sections)+1}'
        section=Section(section_id=sid,title=title or 'Untitled',level=level,
                        parent_section_id=parent)
        sections.append(section);stack.append(section);current=section
        return section
    def add(e):
        els[e.element_id]=e
        current.block_ids.append(e.element_id)
    def add_paragraph(tag,multiline=False):
        nonlocal seq,refseq
        text,fmt,math,anchors=_paragraph_tokens(tag)
        if not text:return
        seq+=1;eid=f'paragraph-{seq}'
        add(Paragraph(element_id=eid,section_id=current.section_id,text=text,
                      format_spans=fmt,inline_math_spans=math,original_html_id=tag.get('id')))
        for b,e,k,key in anchors:
            refseq+=1
            mentions.append(Mention(mention_id=f'mention-{refseq}',
               source_anchor=SourceAnchor(element_id=eid,selector=TextSpanSelector(
                   start=b,end=e,exact=text[b:e],prefix=text[max(0,b-30):b],suffix=text[e:e+30])),
               mention_text=text[b:e],target_kind=k,target_id=key,resolution_status='unresolved'))
    def add_img(node):
        nonlocal seq
        is_graphical='c-article-section__figure--no-border' in node.get('class',[]) or not node.get('id')
        source_id=node.get('id') or ('figure-abstract' if is_graphical else f'unnamed-{len(seen_figs)+1}')
        if source_id in seen_figs:return
        seen_figs[source_id]=True
        seq+=1;eid=f'figure-{seq}'
        im=node.select_one('img')
        src=im.get('src','') if im else ''
        rel=highres.get(source_id) or _safe_src(src)
        if source_id not in highres:
            warns.append(f'Springer figure preview-only: {source_id}; full-resolution image not verified')
        aid=None
        if rel:
            aid=f'springer-image-{len(assets)+1}'
            assets[aid]=Asset(asset_id=aid,original_relative_path=rel)
        capnode=node.select_one('[data-test="bottom-caption"]')
        caption=_norm(capnode.get_text(' ',strip=True)) if capnode else ''
        labelnode=node.select_one('[data-test="figure-caption-text"]')
        label=_norm(labelnode.get_text(' ',strip=True)) if labelnode else ('Graphical abstract' if is_graphical else source_id)
        add(Figure(element_id=eid,section_id=current.section_id,label=label,caption=caption,
                   alt_text=im.get('alt','') if im else None,image_asset_id=aid,original_html_id=source_id))
        if labelnode and labelnode.get('id'):
            target_by_source[labelnode.get('id')]=eid
        target_by_source[source_id]=eid
    def add_table(node):
        nonlocal seq
        raw_id=node.get('id','')
        m=re.fullmatch(r'table-(\d+)',raw_id)
        if not m or raw_id in seen_tables:return
        n=int(m.group(1));seen_tables[raw_id]=True
        seq+=1;eid=f'table-{seq}'
        captionnode=node.select_one('figcaption')
        caption=_norm(captionnode.get_text(' ',strip=True)) if captionnode else f'Table {n}'
        grid=local_tables.get(n)
        if grid is None and fetch_tables and not browser_tables:
            try:grid=fetch_satellite_table(doi,n)
            except Exception as exc:
                warns.append(f'Table {n} network fetch failed: {str(exc)[:160]}')
        if not grid:warns.append(f'Table {n} data missing: {doi}/tables/{n}')
        add(Table(element_id=eid,section_id=current.section_id,label=f'Table {n}',caption=caption,
                  representation='structured',grid=grid or TableGrid(cells=[]),
                  recognition_status='verified' if grid else 'not_attempted',original_html_id=raw_id))
        target_by_source[f'Tab{n}']=eid;target_by_source[raw_id]=eid
    def add_equation(node):
        nonlocal seq
        eid0=node.get('id','')
        if eid0 in seen_equations:return
        seen_equations[eid0]=True
        seq+=1;eid=f'equation-{seq}'
        latex,mathml=_math_data(node)
        if not latex and not mathml:
            warns.append(f'Equation lacks TeX/MathML: {eid0}')
        add(Equation(element_id=eid,section_id=current.section_id,
                     label=(eid0 if re.match(r'Equ\d+',eid0) else None),
                     latex=latex,mathml=mathml,plain_text=latex or '',original_html_id=eid0 or None))
        if eid0:target_by_source[eid0]=eid
    def walk(tag):
        if not isinstance(tag,Tag):return
        cls=set(tag.get('class',[]))
        if tag.name in ('script','style','nav','aside'):return
        if 'c-article-table' in cls:
            add_table(tag);return
        if 'c-article-section__figure' in cls:
            add_img(tag);return
        if 'c-article-equation' in cls:
            add_equation(tag);return
        if tag.name in ('h2','h3','h4','h5','h6'):
            lev=int(tag.name[1])-1
            if lev>=1:new_section(_norm(tag.get_text(' ',strip=True)),lev,tag.get('id'))
            return
        if tag.name == 'li' and tag.select_one('.c-article-equation,.c-article-table,.c-article-section__figure'):
            # Springer occasionally puts a full display equation inside a
            # list item. Preserve it as its own block, not inside prose.
            add_paragraph(tag)
            for embedded in tag.select('.c-article-equation,.c-article-table,.c-article-section__figure'):
                walk(embedded)
            return
        if tag.name in ('p','li','pre','blockquote'):
            if tag.find_parent('.c-article-references'):
                return
            add_paragraph(tag);return
        for ch in tag.children:
            if isinstance(ch,Tag):walk(ch)
    # The graphical abstract belongs to the abstract, not a separate body
    # section. Body sections are exclusively .main-content direct sections.
    abstract=soup.select_one('#Abs1-section')
    if abstract:
        new_section('Abstract',1)
        for child in abstract.select_one('.c-article-section__content').children:
            if isinstance(child,Tag):walk(child)
    for s in main.find_all('section',recursive=False):
        # Each direct <section> holds one top-level article block.
        for t in s.children:
            if isinstance(t,Tag):walk(t)
    # Appendices are placed outside div.main-content by Springer. They can
    # contain most of a technical paper's tables; never silently skip them.
    for appendix in soup.select('.c-article-body #appendices-section'):
        for child in appendix.children:
            if isinstance(child,Tag):walk(child)
    # Match only genuine anchors; links in menus, sidebar, bibliography and
    # duplicate reading companion panels never enter main-content paragraphs.
    for m in mentions:
        t=target_by_source.get(m.target_id) if m.target_id else None
        if m.target_kind=='reference' and m.target_id in refs:
            m.resolution_status='resolved'
        elif t and els[t].kind==m.target_kind:
            m.target_id=t;m.resolution_status='resolved'
        else:
            m.target_id=None
    tables=[x for x in els.values() if isinstance(x,Table)]
    figures=[x for x in els.values() if isinstance(x,Figure)]
    warnings=list(dict.fromkeys(warns))
    quality=PaperQuality(body='complete' if sections and len(els)>=5 else 'partial',
        references='complete' if refs else 'missing',
        tables='complete' if all(t.grid and t.grid.cells for t in tables) else 'partial',
        figures='unknown' if figures else 'complete',warnings=warnings)
    doc=PaperDocument(paper_id=paper_id,revision='',metadata=metadata,sections=sections,
       elements=els,references=refs,mentions=mentions,assets=assets,quality=quality)
    doc.revision=hashlib.sha256(doc.model_dump_json().encode()).hexdigest()[:16]
    return doc