"""Deterministic, offline static visual packaging for LaTeXML figures.

Handles single/multiple inline SVGs and saved image panels. It creates one
standalone sanitized SVG per Figure, referenced through an <img> tag. No HTML
scripts, stylesheets, or network links from the publisher are embedded.
"""
from __future__ import annotations

import base64
import copy
import io
import re
import xml.etree.ElementTree as ET
from urllib.parse import urlsplit

from bs4 import BeautifulSoup, Tag
from app.ingestion.loader import HTMLBundle, _resolve_img

SVG_NS = 'http://www.w3.org/2000/svg'
HTML_NS = 'http://www.w3.org/1999/xhtml'
XLINK_NS = 'http://www.w3.org/1999/xlink'
ET.register_namespace('', SVG_NS)
ET.register_namespace('xlink', XLINK_NS)

SVG_TAGS = {
    'svg', 'g', 'defs', 'symbol', 'use', 'rect', 'circle', 'ellipse', 'line',
    'polyline', 'polygon', 'path', 'text', 'tspan', 'textpath', 'title', 'desc',
    'clippath', 'mask', 'lineargradient', 'radialgradient', 'stop', 'pattern',
    'marker', 'filter', 'fegaussianblur', 'feoffset', 'fecolormatrix',
    'feblend', 'fecomposite', 'femerge', 'femergenode', 'feflood',
    'femorphology', 'feturbulence', 'fedisplacementmap', 'fetile',
    'fecomponenttransfer', 'fefuncr', 'fefuncg', 'fefuncb', 'fefunca',
    'fedropshadow', 'foreignobject',
}
CANON = {
    'textpath':'textPath', 'clippath':'clipPath', 'lineargradient':'linearGradient',
    'radialgradient':'radialGradient', 'foreignobject':'foreignObject',
    'fegaussianblur':'feGaussianBlur', 'feoffset':'feOffset',
    'fecolormatrix':'feColorMatrix', 'feblend':'feBlend',
    'fecomposite':'feComposite', 'femerge':'feMerge',
    'femergenode':'feMergeNode', 'feflood':'feFlood',
    'femorphology':'feMorphology', 'feturbulence':'feTurbulence',
    'fedisplacementmap':'feDisplacementMap', 'fetile':'feTile',
    'fecomponenttransfer':'feComponentTransfer', 'fefuncr':'feFuncR',
    'fefuncg':'feFuncG', 'fefuncb':'feFuncB', 'fefunca':'feFuncA',
    'fedropshadow':'feDropShadow',
}
HTML_TAGS = {'span','div','p','br','strong','b','em','i'}
SVG_ATTRS = {
    'id','class','width','height','viewbox','version','x','y','x1','y1','x2','y2',
    'cx','cy','r','rx','ry','d','points','transform','fill','fill-opacity',
    'stroke','stroke-opacity','stroke-width','stroke-linecap','stroke-linejoin',
    'stroke-miterlimit','stroke-dasharray','stroke-dashoffset','opacity',
    'clip-path','clip-rule','fill-rule','mask','filter','gradientunits',
    'gradienttransform','offset','stop-color','stop-opacity','preserveaspectratio',
    'overflow','font-size','font-family','font-weight','font-style','text-anchor',
    'dominant-baseline','letter-spacing','word-spacing','style','href',
}
CSS_PROPS = {'fill','stroke','stroke-width','opacity','fill-opacity',
             'stroke-opacity','font-size','font-weight','font-style',
             'font-family','color','background-color','text-align','line-height'}
URL_PATTERN = re.compile(r'url\(\s*["\']?([^\)"\']+)["\']?\s*\)', re.I)


def _lname(value: str) -> str:
    return value.rsplit('}', 1)[-1].split(':')[-1]


def _safe_value(value: str, *, allow_url: bool = True) -> bool:
    val = value.strip()
    if any(c in val for c in '<>') or len(val) > 15000:
        return False
    low = val.lower()
    if any(s in low for s in ('javascript:', 'data:', 'expression(', '@import', '@font-face', 'behavior:', 'var(--')):
        return False
    refs = URL_PATTERN.findall(val)
    if refs and (not allow_url or not all(x.strip().startswith('#') for x in refs)):
        return False
    return True


def _clean_style(value: str, html: bool = False) -> str:
    props = {}
    for decl in value.split(';'):
        if ':' not in decl: continue
        key, val = (x.strip() for x in decl.split(':',1))
        lk = key.lower()
        if lk == '--ltx-fg-color' and html and re.fullmatch(r'#[A-Fa-f0-9]{3,8}', val):
            props['color'] = val
        if lk in CSS_PROPS and _safe_value(val):
            props[lk] = val
    return ';'.join(f'{k}:{v}' for k,v in props.items())


def _sanitize_node(raw: ET.Element, *, in_html: bool = False) -> ET.Element | None:
    local = _lname(raw.tag)
    low = local.lower()
    # Inside foreignObject permit inert XHTML text formatting only.
    if in_html:
        if low not in HTML_TAGS:
            return None
        out = ET.Element(f'{{{HTML_NS}}}{low}')
        if raw.get('style'):
            style = _clean_style(raw.get('style',''), html=True)
            if style: out.set('style', style)
        if low in ('div','p','span'):
            out.set('style', (out.get('style','') + ';font-family:Arial,sans-serif').strip(';'))
    else:
        if low not in SVG_TAGS:
            return None
        out = ET.Element(f'{{{SVG_NS}}}{CANON.get(low,low)}')
        for k,v in raw.attrib.items():
            name = _lname(k)
            nk = name.lower()
            if nk not in SVG_ATTRS or nk.startswith('on') or not _safe_value(v):
                continue
            if nk == 'href' and not v.strip().startswith('#'):
                continue
            if nk == 'style':
                v = _clean_style(v)
                if not v: continue
            dest = {'viewbox':'viewBox', 'preserveaspectratio':'preserveAspectRatio',
                    'gradientunits':'gradientUnits','gradienttransform':'gradientTransform'}.get(nk,name)
            out.set(dest, v)
    out.text = raw.text
    child_html = in_html or low == 'foreignobject'
    for child in raw:
        new = _sanitize_node(child, in_html=child_html)
        if new is not None:
            out.append(new)
            new.tail = child.tail
        elif child.tail:
            if len(out): out[-1].tail = (out[-1].tail or '') + child.tail
            else: out.text = (out.text or '') + child.tail
    return out


def _clean_svg(svg: Tag) -> ET.Element:
    markup = str(svg).encode('utf8')
    if len(markup) > 4_000_000 or b'<!DOCTYPE' in markup.upper() or b'<!ENTITY' in markup.upper():
        raise ValueError('SVG size or XML declarations unsafe')
    src = ET.fromstring(markup)
    dst = _sanitize_node(src)
    if dst is None or _lname(dst.tag) != 'svg':
        raise ValueError('Not an SVG root')
    return dst


def _size(node: ET.Element) -> tuple[float,float]:
    def scalar(key):
        val = node.get(key,'')
        match = re.match(r'^\s*(\d+(?:\.\d+)?)(?:pt|px)?\s*$',val)
        return float(match.group(1)) if match else None
    w,h = scalar('width'),scalar('height')
    vb = re.split(r'[\s,]+',node.get('viewBox','').strip())
    if len(vb)==4:
        try: w=w or float(vb[2]); h=h or float(vb[3])
        except ValueError: pass
    w,h = w or 640, h or 360
    if not 1 <= w <= 5000 or not 1 <= h <= 5000:
        raise ValueError('Invalid SVG dimensions')
    return w,h


def _raster(img: Tag, bundle: HTMLBundle):
    # LaTeXML can serve figures as <object type=image/svg+xml data=...>.
    # Resolve with the same strict, offline path validator as <img>.
    if img.name == 'object':
        if (img.get('type') or '').lower().strip() != 'image/svg+xml':
            return None
        src = img.get('data') or ''
        status, local_path, media_type = _resolve_img(src, bundle.html_path, bundle.resource_dir)
        class ObjectRef:
            pass
        ref = ObjectRef()
        ref.status, ref.local_path, ref.media_type = status, local_path, media_type
        if media_type != 'image/svg+xml':
            return None
    else:
        src = img.get('src') or img.get('data-src') or ''
        ref = next((r for r in bundle.images if r.src == src), None)
        if ref is None: return None
    raw = None
    if ref.status == 'available' and ref.local_path:
        local = ref.local_path.resolve()
        root = (bundle.resource_dir or bundle.html_path.parent).resolve()
        if not local.is_relative_to(root) or not local.is_file() or local.stat().st_size > 10_000_000:
            raise ValueError('Unsafe raster file')
        raw = local.read_bytes()
    elif ref.status == 'embedded' and src.lower().startswith('data:image/') and len(src) < 13_500_000:
        if re.match(r'^data:image/(png|jpeg|gif|webp);base64,',src,re.I):
            raw = base64.b64decode(src.split(',',1)[1],validate=True)
    if not raw: return None
    if ref.media_type == 'image/svg+xml':
        xml = BeautifulSoup(raw, 'xml')
        svg = xml.find('svg')
        if svg is None: return None
        cleaned = _clean_svg(svg)
        w,h = _size(cleaned)
        return cleaned,w,h
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as pic:
        w,h=pic.size
        if not w or not h or w*h > 15_000_000: raise ValueError('Unsafe raster dimensions')
        if pic.format not in ('PNG','JPEG','GIF','WEBP'): return None
        fmt = pic.format
        # Uniformly encode to PNG to avoid unsupported browser formats.
        rgb = pic.convert('RGBA')
        stream=io.BytesIO()
        rgb.save(stream, format='PNG')
        raw=stream.getvalue()
    image = ET.Element(f'{{{SVG_NS}}}image',{
        'width':str(w),'height':str(h),
        'href':'data:image/png;base64,'+base64.b64encode(raw).decode('ascii'),
    })
    return image,float(w),float(h)


def render_figure_svg(bundle: HTMLBundle, original_html_id: str, *, allow_cell: bool = False):
    """Returns (sanitized SVG bytes, preserved visual panel count).

    Caller must treat None as incomplete figure, NOT as an empty success.
    """
    if not original_html_id: return None,0
    soup = BeautifulSoup(bundle.html,'lxml')
    body = soup.select_one('article.ltx_document,div.ltx_document,main.ltx_document')
    node = body.find(id=original_html_id) if body else None
    if node is None or (not allow_cell and 'ltx_figure' not in node.get('class',[])): return None,0
    panels=[]
    failures=0
    for media in node.find_all(['svg','img','object']):
        # SVG nested inside other SVG is part of the same scene.
        if media.find_parent('svg') is not None: continue
        try:
            if media.name=='svg':
                el=_clean_svg(media)
                w,h=_size(el)
                panels.append((el,w,h))
            else:
                item=_raster(media,bundle)
                if item: panels.append(item)
                else: failures+=1
        except (ValueError,OSError,ET.ParseError,base64.binascii.Error):
            failures+=1
    if not panels or len(panels)>24 or failures:
        return None,len(panels)
    columns=1 if len(panels)==1 else 2
    gap=18
    cell_width=min(1100,max(max(p[1] for p in panels),240))
    sizes=[]
    for _,w,h in panels:
        scale=min(1.,cell_width/w,1600/h)
        sizes.append((w*scale,h*scale,scale))
    rows=(len(panels)+columns-1)//columns
    row_height=[max(sizes[i][1] for i in range(r*columns,min(len(sizes),(r+1)*columns))) for r in range(rows)]
    cw=columns*cell_width+(columns-1)*gap
    ch=sum(row_height)+gap*(rows-1)
    if cw*ch>14_000_000: raise ValueError('Composite figure too large')
    root=ET.Element(f'{{{SVG_NS}}}svg',{'viewBox':f'0 0 {cw:.2f} {ch:.2f}',
                'width':f'{cw:.2f}', 'height':f'{ch:.2f}'})
    y=0
    for r in range(rows):
        for c in range(columns):
            i=r*columns+c
            if i>=len(panels): break
            el,w,h=panels[i]
            dw,dh,sc=sizes[i]
            x=c*(cell_width+gap)+(cell_width-dw)/2
            group=ET.SubElement(root,f'{{{SVG_NS}}}g',{'transform':f'translate({x:.2f},{y:.2f}) scale({sc:.6f})'})
            group.append(copy.deepcopy(el))
        y+=row_height[r]+gap
    result=ET.tostring(root,encoding='utf8',xml_declaration=True)
    if any(x in result.lower() for x in (b'<script',b'javascript:',b'<iframe')):
        raise ValueError('Active or external content in sanitized SVG')
    return result,len(panels)