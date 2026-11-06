"""Normalize a user-downloaded IEEE HTML paper and its _files resource folder.

Example:
uv run python export_ieee_paper.py "data/Flight delay prediction based on LightGBM _ IEEE Conference Publication _ IEEE Xplore.html"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from app.ingestion.adapters.ieee import parse_ieee_bundle
from app.ingestion.exporter import export_paper
from app.ingestion.loader import load_html_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description='IEEE HTML -> safe normalized paper export')
    parser.add_argument('html_path', type=Path)
    parser.add_argument('--resources', type=Path, help='Optional explicit _files directory')
    parser.add_argument('--output-root', type=Path, default=Path('data/standardized'), help='Output root folder for exported paper')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    bundle = load_html_bundle(args.html_path, args.resources)
    paper = parse_ieee_bundle(bundle)
    result = export_paper(paper, bundle, args.output_root, overwrite=args.overwrite)

    print('=== NORMALIZED EXPORT ===')
    print('Paper:', paper.paper_id)
    print('Output:', result.output_dir)
    print('Assets copied:', result.copied_assets)
    print('Assets missing:', result.missing_assets)
    print('Assets skipped (unsafe/unsupported):', result.skipped_assets)
    print('Paper JSON:', result.paper_json)
    print('Reader HTML:', result.normalized_html)
    print('Manifest:', result.manifest_json)
    print('Quality:', paper.quality.model_dump(exclude={'warnings'}))


if __name__ == '__main__':
    main()