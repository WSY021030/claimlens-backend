"""Export normalized PaperDocument v2 and paper-only image assets.

Inputs are from a user-provided HTMLBundle and a trusted Publisher Adapter.
No website JavaScript is executed, downloaded, or copied.
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

from app.ingestion.loader import HTMLBundle
from app.ingestion.normalizer import render_normalized_html
from app.ingestion.math_renderer import render_math_svg
from app.models.document_v2 import PaperDocument, PaperQuality, Equation


SAFE_IMAGE_SUFFIXES = {
    'image/png': '.png',
    'image/jpeg': '.jpg',
    'image/gif': '.gif',
    'image/webp': '.webp',
    'image/bmp': '.bmp',
    'image/tiff': '.tiff',
}


def _local_key(path_or_url: str) -> str | None:
    parts = urlsplit(path_or_url)
    if parts.scheme or parts.netloc or parts.path.startswith('/'):
        return None
    path = unquote(parts.path).replace('\\', '/')
    components = [p for p in path.split('/') if p not in ('', '.')]
    if not components or '..' in components:
        return None
    return '/'.join(components)


def _sha256(raw: bytes) -> str:
    return 'sha256:' + hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ExportReport:
    output_dir: Path
    copied_assets: int
    missing_assets: int
    skipped_assets: int
    paper_json: Path
    normalized_html: Path
    manifest_json: Path


def export_paper(
    paper: PaperDocument,
    bundle: HTMLBundle,
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> ExportReport:
    """Generate data/normalized/<paper_id>/{paper.json, normalized.html, manifest.json, assets/}.

    Source files stay untouched. All output is staged before final directory
    rename. To replace an existing export, pass overwrite=True explicitly.
    """
    # Do not accept a paper schema from another HTML page silently.
    if paper.metadata.original_url and bundle.source_url and (
        paper.metadata.original_url.rstrip('/') != bundle.source_url.rstrip('/')
    ):
        raise ValueError('Paper metadata URL differs from supplied HTML bundle URL')
    if not paper.paper_id or paper.paper_id in ('.', '..') or any(
        char not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-'
        for char in paper.paper_id
    ):
        raise ValueError('Unsafe paper_id for output directory')

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    output_dir = root / paper.paper_id
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f'Export exists: {output_dir}. Use --overwrite to replace it.')

    ref_by_key = {}
    for ref in bundle.images:
        key = _local_key(ref.src)
        if key is not None and ref.status == 'available' and ref.local_path:
            ref_by_key.setdefault(key, ref)

    stage = Path(tempfile.mkdtemp(prefix='.paper-export-', dir=root))
    copied = missing = skipped = 0
    try:
        (stage / 'assets').mkdir()
        updated = paper.model_copy(deep=True)
        problems = []
        missing_ids = []
        for asset_id, asset in updated.assets.items():
            # Only arXiv's adapter may declare virtual assets used to
            # preserve inline SVG / multipanel figures. They are converted
            # to sanitized static SVG image files, never active publisher HTML.
            if (updated.metadata.source_platform == 'arxiv' and
                asset.original_relative_path == f'__arxiv_render__/{asset_id}.svg'):
                from app.ingestion.arxiv_vector import render_figure_svg
                matching = [el for el in updated.elements.values()
                            if el.kind == 'figure' and el.image_asset_id == asset_id]
                cells = [cell for el in updated.elements.values()
                         if el.kind == 'table' and el.grid
                         for cell in el.grid.cells if cell.image_asset_id == asset_id]
                try:
                    if len(matching) + len(cells) != 1:
                        raise ValueError('No unique arXiv visual source')
                    source = (matching[0].original_html_id if matching else cells[0].original_html_id)
                    if not source:
                        raise ValueError('Visual source ID missing')
                    raw_svg, panels = render_figure_svg(bundle, source, allow_cell=bool(cells))
                    if not raw_svg:
                        raise ValueError('No safe source visual panels found')
                    target_name = f'{asset_id}.svg'
                    (stage / 'assets' / target_name).write_bytes(raw_svg)
                    updated.assets[asset_id] = asset.model_copy(update={
                        'status': 'available', 'media_type': 'image/svg+xml',
                        'normalized_relative_path': f'assets/{target_name}',
                    })
                    copied += 1
                except (ValueError, OSError) as exc:
                    missing += 1
                    missing_ids.append(asset_id)
                    problems.append(f'arXiv visual conversion failed: {asset_id}: {exc}')
                    updated.assets[asset_id] = asset.model_copy(update={
                        'status': 'missing', 'normalized_relative_path': None,
                    })
                continue
            key = _local_key(asset.original_relative_path)
            ref = ref_by_key.get(key) if key else None
            if not ref:
                missing += 1
                missing_ids.append(asset_id)
                problems.append(f'Asset not found: {asset_id}: {asset.original_relative_path}')
                updated.assets[asset_id] = asset.model_copy(update={
                    'status': 'missing', 'normalized_relative_path': None,
                })
                continue
            # The loader inspected the actual resource bytes, not just the suffix.
            # SVG deliberately excluded: saved SVG can embed active content.
            suffix = SAFE_IMAGE_SUFFIXES.get(ref.media_type or '')
            if suffix is None:
                skipped += 1
                missing_ids.append(asset_id)
                problems.append(f'Unsafe or unsupported image type: {asset_id}: {ref.media_type}')
                updated.assets[asset_id] = asset.model_copy(update={
                    'status': 'missing', 'normalized_relative_path': None,
                })
                continue
            local_path = ref.local_path.resolve()
            # The loader guarantees this is within its upload root, but check
            # again to avoid following changed symlinks between load and copy.
            upload_root = (bundle.resource_dir or bundle.html_path.parent).resolve()
            if not local_path.is_relative_to(upload_root) or not local_path.is_file():
                raise ValueError(f'Unsafe or disappeared resource: {local_path}')
            target_name = f'{asset_id}{suffix}'
            destination = stage / 'assets' / target_name
            shutil.copyfile(local_path, destination)
            updated.assets[asset_id] = asset.model_copy(update={
                'status': 'available',
                'media_type': ref.media_type,
                'normalized_relative_path': f'assets/{target_name}',
            })
            copied += 1

        # Recalculate figure completeness AFTER converting generated visual assets.
        figure_ids = [e.image_asset_id for e in updated.elements.values() if e.kind == 'figure']
        if figure_ids:
            updated.quality.figures = ('complete' if all(
                aid and aid in updated.assets and updated.assets[aid].status == 'available'
                for aid in figure_ids) else 'partial')
        # A copied Springer 685px preview is not proof that a full-resolution
        # illustration was recovered, even though the bytes are available.
        if updated.metadata.source_platform == 'springer' and any(
            str(w).startswith('Springer figure preview-only:')
            for w in updated.quality.warnings
        ):
            updated.quality.figures = 'partial'
        # A caption may contain small inline graphics not representable in v2.
        # A copied primary figure image alone does not make that complete.
        if any('Caption has inline graphics not separately exported' in x
               for x in updated.quality.warnings):
            updated.quality.figures = 'partial'
        if any(elem.kind == 'figure' and any(
            a.asset_id not in updated.assets or updated.assets[a.asset_id].status != 'available'
            for a in elem.caption_inline_assets) for elem in updated.elements.values()):
            updated.quality.figures = 'partial'
        # A table is complete only when its structured cells exist and any
        # original figures embedded *inside those cells* were exported too.
        updated.quality.tables = ('complete' if all(
            el.grid is not None and bool(el.grid.cells) and all(
                not cell.image_asset_id or updated.assets[cell.image_asset_id].status == 'available'
                for cell in el.grid.cells)
            for el in updated.elements.values() if el.kind == 'table') else 'partial')
        if missing_ids:
            has_figure_missing = any(
                elem.kind == 'figure' and elem.image_asset_id in missing_ids
                for elem in updated.elements.values()
            )
            if has_figure_missing:
                updated.quality.figures = 'partial'
        # Remove adapter-time missing-media warnings for figures that have
        # successfully been materialized by the offline exporter.
        updated.quality.warnings = list(dict.fromkeys(
            [*updated.quality.warnings, *problems]
        ))
        # Render displayed equations locally. Preserve the LaTeX in paper.json.
        # The original publisher's JS is never copied/executed.
        equation_assets = {}
        equation_render_errors = []
        equations = [elem for elem in updated.elements.values() if isinstance(elem, Equation)]
        native_mathml_count = 0
        for equation in equations:
            # LaTeXML already includes presentation MathML, which modern
            # browsers display natively. Skip unsupported matplotlib mathtext
            # parsing rather than reporting false warnings for cases/splits.
            if updated.metadata.source_platform in ('arxiv', 'acm', 'springer') and equation.mathml:
                native_mathml_count += 1
                continue
            if not equation.latex:
                continue
            rel = f'assets/{equation.element_id}.svg'
            try:
                render_math_svg(equation.latex, stage / rel)
            except (ValueError, ImportError, RuntimeError) as exc:
                equation_render_errors.append(f'{equation.element_id}: {str(exc)[:180]}')
                continue
            equation_assets[equation.element_id] = rel
        updated.quality.warnings = list(dict.fromkeys([
            *updated.quality.warnings,
            *(f'Equation render fallback: {error}' for error in equation_render_errors),
        ]))
        # Validate every cross-reference and the normalized relative asset paths.
        updated = PaperDocument.model_validate(updated.model_dump())

        paper_bytes = updated.model_dump_json(indent=2).encode('utf-8')
        html_bytes = render_normalized_html(updated, equation_assets).encode('utf-8')
        (stage / 'paper.json').write_bytes(paper_bytes)
        (stage / 'standardized.html').write_bytes(html_bytes)
        manifest = {
            'export_schema': 'paper-export.v1',
            'paper_id': updated.paper_id,
            'paper_revision': updated.revision,
            'source_url': updated.metadata.original_url,
            'source_html_sha256': _sha256(bundle.html_path.read_bytes()),
            'paper_json_sha256': _sha256(paper_bytes),
            'normalized_html_sha256': _sha256(html_bytes),
            'asset_counts': {
                'total': len(updated.assets),
                'copied': copied,
                'missing': missing,
                'skipped_unsafe_or_unsupported': skipped,
            },
            'equation_rendering': {
                'total': len(equations),
                'rendered': len(equation_assets) + native_mathml_count,
                'native_mathml': native_mathml_count,
                'fallback': len(equation_render_errors),
            },
            'quality': updated.quality.model_dump(),
        }
        (stage / 'manifest.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
        )

        if output_dir.exists():
            # Stage has already succeeded: only now swap the old export out.
            backup = Path(tempfile.mkdtemp(prefix='.paper-backup-', dir=root))
            backup.rmdir()
            output_dir.rename(backup)
            try:
                stage.rename(output_dir)
            except Exception:
                backup.rename(output_dir)
                raise
            shutil.rmtree(backup)
        else:
            stage.rename(output_dir)
        return ExportReport(
            output_dir=output_dir,
            copied_assets=copied,
            missing_assets=missing,
            skipped_assets=skipped,
            paper_json=output_dir / 'paper.json',
            normalized_html=output_dir / 'standardized.html',
            manifest_json=output_dir / 'manifest.json',
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
