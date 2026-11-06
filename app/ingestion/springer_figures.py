"""Offline-first Springer figure resource enrichment.

Uses only DOI-bound URLs advertised in the user-provided article/figure HTML.
Network use is explicit (--browser-figures) and restricted to SpringerLink and
Springer Nature's image CDN. No website scripts are copied to the output.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, unquote
from bs4 import BeautifulSoup

from app.ingestion.loader import HTMLBundle, ImageReference, _sniff_media_type

IMAGE_MIME = {'image/png': '.png', 'image/jpeg': '.jpg', 'image/gif': '.gif', 'image/webp': '.webp'}
MAX_BYTES = 24_000_000


@dataclass(frozen=True)
class FigureTarget:
    source_id: str
    figure_num: int | None
    filename: str
    preview_src: str


def _image_url(url: str, doi: str, filename: str) -> bool:
    """Only official CDN image for exactly this DOI and figure filename."""
    try:
        p = urlsplit(url)
        decoded = unquote(p.path)
    except ValueError:
        return False
    if p.scheme != 'https' or p.hostname != 'media.springernature.com' or p.port is not None or p.username or p.password or p.query or p.fragment:
        return False
    segments = decoded.split('/', 2)
    if len(segments) != 3 or not segments[1] or segments[1] in ('.','..'):
        return False
    expected = f'springer-static/image/art:{doi}/MediaObjects/{filename}'
    return segments[2].lower() == expected.lower()


def _ranking(url: str) -> int:
    segment = urlsplit(url).path.strip('/').split('/', 1)[0].lower()
    return {'full': 400, 'original': 400, 'lw2000': 350, 'lw1500': 310, 'lw1200': 280, 'm685': 100}.get(segment, 150)


def _candidate_links(soup: BeautifulSoup, doi: str, filename: str, *, fig_page=False) -> list[str]:
    ranked = set()
    # The article's JSON-LD often lists lw1200 versions absent in ordinary <img src>.
    for tag in soup.select('script[type="application/ld+json"]'):
        try:
            item = json.loads(tag.string or tag.get_text())
            entity = item.get('mainEntity', {}) if isinstance(item, dict) else {}
            items = entity.get('image', []) if isinstance(entity, dict) else []
            if isinstance(items, str): items = [items]
            for url in items:
                if isinstance(url, str) and _image_url(url, doi, filename): ranked.add(url)
        except (ValueError, AttributeError, TypeError):
            continue
    # Satellite figure page can advertise an even larger /full/ media image.
    if fig_page:
        for tag in soup.select('main img, main a[href], main source'):
            for attr in ('href','src','data-src','srcset'):
                value = tag.get(attr)
                if not value or not isinstance(value,str): continue
                if attr == 'srcset':
                    values = [part.strip().split(' ')[0] for part in value.split(',')]
                else:values = [value]
                for url in values:
                    if _image_url(url,doi,filename): ranked.add(url)
    return sorted(ranked, key=_ranking, reverse=True)


def _satellite_url(doi: str, n: int) -> str:
    return f'https://link.springer.com/article/{doi}/figures/{n}'


def _valid_satellite_url(url: str, doi: str, n: int) -> bool:
    p=urlsplit(url)
    return p.scheme == 'https' and p.hostname == 'link.springer.com' and p.port is None and not p.query and not p.fragment and p.path.rstrip('/') == f'/article/{doi}/figures/{n}'


def _image_dimensions(raw: bytes, kind: str) -> tuple[int,int] | None:
    """Check downloaded image isn't a 1x1 tracking pixel or tiny thumbnail."""
    if kind=='image/png' and raw.startswith(b'\x89PNG\r\n\x1a\n') and len(raw)>=24:
        return int.from_bytes(raw[16:20],'big'),int.from_bytes(raw[20:24],'big')
    if kind=='image/gif' and raw.startswith((b'GIF87a',b'GIF89a')) and len(raw)>=10:
        return int.from_bytes(raw[6:8],'little'),int.from_bytes(raw[8:10],'little')
    if kind=='image/jpeg' and raw.startswith(b'\xff\xd8'):
        i=2
        while i+9 < len(raw):
            if raw[i] != 0xff: i+=1; continue
            marker=raw[i+1];i+=2
            if marker in (0xd8,0xd9,0x01) or 0xd0<=marker<=0xd7:continue
            if i+2>len(raw):break
            length=int.from_bytes(raw[i:i+2],'big')
            if length<2 or i+length>len(raw):break
            if marker in (0xc0,0xc1,0xc2,0xc3,0xc5,0xc6,0xc7,0xc9,0xca,0xcb,0xcd,0xce,0xcf):
                return int.from_bytes(raw[i+5:i+7],'big'),int.from_bytes(raw[i+3:i+5],'big')
            i+=length
    if kind=='image/webp' and raw.startswith(b'RIFF') and raw[8:12]==b'WEBP':
        try:
            from PIL import Image
            import io
            with Image.open(io.BytesIO(raw)) as im:return im.size
        except ImportError:pass
    return None


def _image_media(raw:bytes) -> str|None:
    if raw.startswith(b'\x89PNG\r\n\x1a\n'):return 'image/png'
    if raw.startswith(b'\xff\xd8\xff'):return 'image/jpeg'
    if raw.startswith((b'GIF87a',b'GIF89a')):return 'image/gif'
    if raw.startswith(b'RIFF') and raw[8:12]==b'WEBP':return 'image/webp'
    return None


def _cache_folder(bundle: HTMLBundle, doi: str) -> Path:
    slug = re.sub(r'[^a-z0-9]+', '_', doi.lower()).strip('_')
    return (bundle.resource_dir or bundle.html_path.parent) / '_springer_highres_cache' / slug


def _cached_figures(bundle:HTMLBundle, doi:str, targets:list[FigureTarget]) -> dict[str, str]:
    root=bundle.html_path.parent
    folder=_cache_folder(bundle,doi)
    mapping={}
    for target in targets:
        stem = re.sub(r'[^a-zA-Z0-9_-]+','_',target.source_id)
        for ext in IMAGE_MIME.values():
            path = folder / (stem+ext)
            if path.is_file() and path.stat().st_size <= MAX_BYTES:
                media = _sniff_media_type(path)
                if media not in IMAGE_MIME:continue
                with path.open('rb') as fh:
                    dimensions = _image_dimensions(fh.read(100_000),media)
                if not dimensions or dimensions[0]<=685:continue
                relative = path.relative_to(root).as_posix()
                mapping[target.source_id] = relative
                if not any(ref.src==relative for ref in bundle.images):
                    bundle.images.append(ImageReference(index=len(bundle.images)+1,src=relative,alt=target.source_id,
                                                        status='available',local_path=path.resolve(),media_type=media))
                break
    return mapping


def recover_springer_figures(bundle:HTMLBundle, doi:str, targets:list[FigureTarget],
                             *, browser:bool=False) -> tuple[dict[str,str],list[str]]:
    """Find existing high-res cache, then (optionally) fetch DOI-verified CDN image.

    Results are bundle-local ImageReferences so the existing safe exporter works.
    Unknown/blocked images stay as their original preview; never fabricate data.
    """
    recovered=_cached_figures(bundle,doi,targets)
    if not browser:return recovered,[]
    pending=[t for t in targets if t.source_id not in recovered]
    if not pending:return recovered,[]
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return recovered, ['Edge figure fetch unavailable: Playwright is not installed. Run: uv add playwright']
    warns=[]
    base=bundle.html_path.parent
    cache=_cache_folder(bundle,doi)
    article_soup=BeautifulSoup(bundle.html,'lxml')
    try:
        with sync_playwright() as p:
            browser_instance=p.chromium.launch(channel='msedge',headless=False)
            try:
                context=browser_instance.new_context(accept_downloads=False)
                page=context.new_page()
                for target in pending:
                    candidates=[]
                    if target.figure_num is not None:
                        pageurl=_satellite_url(doi,target.figure_num)
                        try:
                            page.goto(pageurl,wait_until='domcontentloaded',timeout=20000)
                            if _valid_satellite_url(page.url,doi,target.figure_num):
                                candidates.extend(_candidate_links(BeautifulSoup(page.content(),'lxml'),doi,target.filename,fig_page=True))
                            else:warns.append(f'Figure {target.source_id} Edge redirected off its verified page: {page.url[:110]}')
                        except Exception as exc:
                            warns.append(f'Figure {target.source_id} satellite page unavailable: {str(exc)[:130]}')
                    candidates.extend(_candidate_links(article_soup,doi,target.filename))
                    candidates=sorted(set(candidates),key=_ranking,reverse=True)
                    if not candidates:
                        warns.append(f'Figure {target.source_id} no DOI-matched high-res image URL in source HTML/figure page')
                        continue
                    failure=None
                    for url in candidates:
                        try:
                            response=context.request.get(url,timeout=20000,max_redirects=0)
                            if not response.ok:
                                failure=f'HTTP {response.status}'
                                continue
                            body=response.body()
                            if len(body)>MAX_BYTES:
                                failure='image exceeds 24MB limit';continue
                            media=_image_media(body)
                            if media not in IMAGE_MIME:
                                failure='not a supported raster image';continue
                            dims=_image_dimensions(body,media)
                            if not dims or min(dims)<20 or max(dims)>30000:
                                failure=f'unsupported or suspicious image dimensions {dims}';continue
                            if dims[0]<=685:
                                failure=f'image is still preview-sized: {dims[0]}px wide';continue
                            ext=IMAGE_MIME[media]
                            stem=re.sub(r'[^a-zA-Z0-9_-]+','_',target.source_id)
                            cache.mkdir(parents=True,exist_ok=True)
                            path=cache/(stem+ext)
                            # File is staged only after validating response bytes.
                            path.write_bytes(body)
                            relative=path.relative_to(base).as_posix()
                            recovered[target.source_id]=relative
                            bundle.images.append(ImageReference(index=len(bundle.images)+1,src=relative,
                               alt=target.source_id,status='available',local_path=path.resolve(),media_type=media))
                            print(f'[Springer] Edge recovered Figure {target.source_id}: {len(body)} bytes'
                                  + (f' ({dims[0]}x{dims[1]})' if dims else ''),flush=True)
                            failure=None
                            break
                        except Exception as exc:
                            failure=f'{type(exc).__name__}: {str(exc)[:130]}'
                    if target.source_id not in recovered:
                        warns.append(f'Figure {target.source_id} high-resolution fetch failed: {failure or "unknown"}')
            finally:
                browser_instance.close()
    except Exception as exc:
        warns.append(f'Edge figure fetch cannot start: {str(exc)[:220]}')
    return recovered,warns