"""Offline ACM full-text HTML to PaperDocument v2 + reader export."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

from app.ingestion.loader import load_html_bundle
from app.ingestion.adapters.acm import parse_acm_bundle
from app.ingestion.exporter import export_paper


def main():
    ap = argparse.ArgumentParser(description="Export a saved ACM full-text HTML page")
    ap.add_argument("html_path", help="Browser-saved ACM article HTML file")
    ap.add_argument("--resources", default=None, help="Optional matching _files folder")
    ap.add_argument("--output-root", default="data/standardized", help="Output root folder for exported paper")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    bundle = load_html_bundle(args.html_path, args.resources)
    paper = parse_acm_bundle(bundle)
    report = export_paper(paper, bundle, args.output_root, overwrite=args.overwrite)
    counts = Counter(x.kind for x in paper.elements.values())
    print("=== ACM DOCUMENT V2 ===")
    print("Paper:", paper.paper_id)
    print("Title:", paper.metadata.title)
    print("Sections:", len(paper.sections), "Elements:", dict(counts))
    print("References:", len(paper.references), "Mentions:", len(paper.mentions))
    print("Resolved:", sum(x.resolution_status == "resolved" for x in paper.mentions),
          "/", len(paper.mentions))
    print("Assets copied:", report.copied_assets, "Missing:", report.missing_assets,
          "Skipped:", report.skipped_assets)
    exported = json.loads(report.paper_json.read_text(encoding="utf-8"))
    print("Quality (exported):", {k:v for k,v in exported["quality"].items() if k != "warnings"})
    print("Reader:", report.normalized_html)
    print("JSON:", report.paper_json)
    for line in exported["quality"]["warnings"][:15]:
        print("Warning:", line)


if __name__ == "__main__":
    main()