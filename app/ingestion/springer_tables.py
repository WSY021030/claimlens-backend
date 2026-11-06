"""Springer satellite-table discovery and safe offline-first enrichment.

Only tables from the SAME article DOI and exact /tables/<n> URLs are accepted.
Network retrieval is opt-in; redirects off Springer are refused.
"""
from __future__ import annotations

import re
from itertools import islice
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler
from bs4 import BeautifulSoup, Tag, NavigableString


def _doc_doi(soup: BeautifulSoup) -> str | None:
    tag = soup.select_one('meta[name="citation_doi"], meta[name="DOI"]')
    return tag.get('content', '').lower().strip() if tag else None


def _satellite_id(url: str | None, doi: str) -> int | None:
    if not url:
        return None
    p = urlsplit(url)
    if p.scheme != 'https' or p.hostname != 'link.springer.com' or p.username or p.password or p.port:
        return None
    m = re.fullmatch(r'/article/(10\.\d{4,9}/[^/\s?#]+)/tables/([1-9]\d*)/?', p.path, re.I)
    if m and m.group(1).lower() == doi.lower() and not p.query and not p.fragment:
        return int(m.group(2))
    return None


def _saved_url(text: str) -> str | None:
    m = re.search(r'<!--\s*saved from url=\(\d+\)(https://link\.springer\.com/[^\s]+)\s*-->', text[:1500])
    return m.group(1) if m else None


def _table_cell_text(node: Tag) -> str:
    """Strip MathJax visual clones, retaining original inline TeX as fallback."""
    parts = []
    def walk(x):
        if isinstance(x, NavigableString):
            parts.append(str(x))
            return
        if not isinstance(x,Tag): return
        name = x.name or ''
        cls=set(x.get('class',[]))
        if name in ('script','style','button','noscript','svg'):
            return
        if 'mathjax-tex' in cls:
            tex=x.select_one('script[type^="math/tex"]')
            if tex and tex.string:
                parts.append(' ' + tex.string.strip() + ' ')
            else:
                math=x.select_one('.MJX_Assistive_MathML math')
                if math:parts.append(math.get_text(' ',strip=True))
            return
        if name == 'math':
            parts.append(x.get_text(' ',strip=True));return
        if 'MathJax' in ''.join(cls) or 'MJX_' in ''.join(cls):return
        if name=='img':
            parts.append(x.get('alt','') or '[inline graphic]')
            return
        for child in x.children:walk(child)
    walk(node)
    return ' '.join(''.join(parts).replace('\xa0',' ').split())


def _satellite_grid(soup: BeautifulSoup, doi: str, n: int, url: str | None):
    from app.models.document_v2 import TableGrid, TableCell
    if _doc_doi(soup) != doi.lower() or _satellite_id(url, doi) != n:
        return None
    heading = soup.select_one('main h1.c-article-satellite-title')
    if not heading or not re.match(fr'^Table\s+{n}\b', heading.get_text(' ', strip=True), re.I):
        return None
    table = soup.select_one('main .c-article-table-container table')
    if table is None:
        return None
    cells = []
    occupied = set()
    rows = table.select('tr')
    for row_index, tr in enumerate(rows):
        col_index = 0
        for node in tr.find_all(['th', 'td'], recursive=False):
            while (row_index, col_index) in occupied:
                col_index += 1
            try:
                rowspan = max(1, min(200, int(node.get('rowspan', 1))))
                colspan = max(1, min(200, int(node.get('colspan', 1))))
            except ValueError:
                rowspan = colspan = 1
            role = 'column_header' if node.name == 'th' else 'data'
            if node.get('scope') == 'row':
                role = 'row_header'
            text = _table_cell_text(node)
            cells.append(TableCell(cell_id=f'springer-table-{n}-r{row_index}-c{col_index}',
                                   row=row_index,col=col_index,text=text,role=role,
                                   rowspan=rowspan,colspan=colspan,original_html_id=node.get('id')))
            for r in range(row_index, row_index+rowspan):
                for c in range(col_index, col_index+colspan):
                    occupied.add((r,c))
            col_index += colspan
    return TableGrid(cells=cells) if cells else None


class _NoCrossDomainRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        p = urlsplit(newurl)
        if p.scheme != 'https' or p.hostname != 'link.springer.com':
            raise ValueError('Cross-domain table redirect refused')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _read_online(url: str, doi: str, n: int, timeout: float = 9.0) -> str:
    if _satellite_id(url, doi) != n:
        raise ValueError('Table URL does not belong to article DOI')
    opener = build_opener(_NoCrossDomainRedirect)
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0 (compatible; PaperUnderstandingAssistant/1.0)',
                                'Accept': 'text/html'})
    with opener.open(req, timeout=timeout) as res:
        data = res.read(4_000_001)
    if len(data) > 4_000_000:
        raise ValueError('Table page too large')
    return data.decode('utf-8', errors='replace')


def _discover_candidate_pages(html_path: Path, explicit_files: list[str | Path] | None):
    """Find saved satellite pages in nearby folders, not only data/ root.

    Ignore generated readers/resource folders. DOI + saved URL must still match,
    so a similarly named page from another paper can never be imported.
    """
    candidates: list[Path] = []
    for item in explicit_files or []:
        path = Path(item)
        if path.is_dir():
            # Allow --table-dir or --table-file pointing to a folder.
            candidates.extend(islice(path.rglob('*.html'), 200))
        else:
            candidates.append(path)

    base = html_path.parent
    if base.is_dir():
        for pattern in ('Table *.html', 'table *.html', 'Table*.htm', 'table*.htm', 'table_*.html'):
            candidates.extend(islice(base.rglob(pattern), 200))
    out = []
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if any(p.lower().endswith('_files') or p.lower() in ('normalized', '.git')
               for p in path.parts):
            continue
        out.append(path)
    return out[:500]


def collect_satellite_tables(html_path: Path, doi: str, *, fetch_online: bool = False,
                             explicit_files: list[str | Path] | None = None) -> tuple[dict[int, object],list[str]]:
    """Load nearby real Springer table pages (DOI + URL + grid verified)."""
    found = {}
    warns = []
    for path in _discover_candidate_pages(html_path, explicit_files):
        if not path.is_file():
            if explicit_files and path in map(Path, explicit_files):
                warns.append(f'Explicit Springer table file not found: {path}')
            continue
        try:
            if path.stat().st_size > 8_000_000:
                continue
            raw = path.read_text(encoding='utf-8', errors='replace')
            url = _saved_url(raw)
            n = _satellite_id(url, doi)
            if n is None:
                # Handle clean browser-export caches with explicit DOI/id markers.
                if path.parent.name == '_springer_tables_cache':
                    m = re.fullmatch(r'table_(\d+)\.html', path.name)
                    n = int(m.group(1)) if m else None
                    if n is not None:
                        url = f'https://link.springer.com/article/{doi}/tables/{n}'
            if n is None:
                continue
            grid = _satellite_grid(BeautifulSoup(raw, 'lxml'), doi, n, url)
            if grid and n not in found:
                found[n] = grid
            elif not grid:
                warns.append(f'Saved Springer Table {n} found but no valid grid: {path.name}')
        except (OSError, ValueError) as exc:
            warns.append(f'Saved Springer table skipped: {path.name}: {exc}')
    return found, warns


def fetch_tables_in_edge(doi: str, table_ids: list[int], *, cache_dir: Path):
    """Explicitly requested *visible* Edge session for normal article table pages.

    Optional dependency: playwright; uses installed Microsoft Edge. A fresh
    session is used, not the user's private browser profile or cookies.
    Only DOI-matched real grids can enter PaperDocument. Successful pages are
    saved for later offline reuse. Does not automate sign-in/challenges.
    """
    grids, warnings = {}, []
    if not table_ids:
        return grids, warnings
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return grids, ['Edge table fetch unavailable: Playwright is not installed. Run: uv add playwright']

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel='msedge', headless=False)
            try:
                context = browser.new_context(accept_downloads=False)
                page = context.new_page()
                for n in sorted(set(table_ids)):
                    url = f'https://link.springer.com/article/{doi}/tables/{n}'
                    try:
                        # We wait for the table DOM itself, not a publisher loading
                        # spinner. This is also a source-content completeness check.
                        page.goto(url, wait_until='domcontentloaded', timeout=25000)
                        page.locator('main .c-article-table-container table').wait_for(timeout=12000)
                        # Do not accept content from login/redirect/challenge hosts.
                        if _satellite_id(page.url, doi) != n:
                            warnings.append(f'Table {n} Edge redirected away from verified table URL: {page.url[:120]}')
                            continue
                        raw = page.content()
                        grid = _satellite_grid(BeautifulSoup(raw, 'lxml'), doi, n, url)
                        if grid is None:
                            warnings.append(f'Table {n} Edge opened page, but no verified grid found')
                            continue
                        grids[n] = grid
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        # Our own cache is only written after real-grid validation.
                        cache_target = cache_dir / f'table_{n}.html'
                        # Include the original URL in normal browser Ctrl+S form.
                        prefix = f'<!-- saved from url=({len(url):04d}){url} -->\n'
                        cache_target.write_text(prefix + raw, encoding='utf-8')
                        print(f'[Springer] Edge recovered Table {n}: {len(grid.cells)} cells', flush=True)
                    except Exception as exc:
                        warnings.append(f'Table {n} Edge fetch failed: {str(exc)[:180]}')
            finally:
                browser.close()
    except Exception as exc:
        warnings.append(f'Edge table fetch cannot start: {str(exc)[:240]}')
    return grids, warnings


def fetch_satellite_table(doi: str, n: int):
    url = f'https://link.springer.com/article/{doi}/tables/{n}'
    raw = _read_online(url, doi, n)
    return _satellite_grid(BeautifulSoup(raw,'lxml'),doi,n,url)
