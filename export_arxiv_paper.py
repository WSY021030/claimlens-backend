"""Export user-saved arXiv HTML to the same paper.v2 normalized format as IEEE."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.ingestion.adapters.arxiv import parse_arxiv_bundle
from app.ingestion.exporter import export_paper
from app.ingestion.loader import load_html_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Saved arXiv HTML -> paper.v2 + normalized.html")
    parser.add_argument("html_path", type=Path)
    parser.add_argument("--resources", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("data/standardized"), help="Output root folder for exported paper")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    bundle = load_html_bundle(args.html_path, args.resources)
    paper = parse_arxiv_bundle(bundle)
    result = export_paper(paper, bundle, args.output_root, overwrite=args.overwrite)
    counts = {name: sum(v.kind == name for v in paper.elements.values())
              for name in ("paragraph", "figure", "table", "equation")}
    print("=== ARXIV DOCUMENT V2 ===")
    print("Paper:", paper.paper_id)
    print("Title:", paper.metadata.title)
    print("Sections:", len(paper.sections), "Elements:", counts)
    print("References:", len(paper.references), "Mentions:", len(paper.mentions))
    print("Assets copied:", result.copied_assets, "Missing:", result.missing_assets,
          "Skipped:", result.skipped_assets)
    exported = json.loads(result.paper_json.read_text(encoding="utf-8"))
    final_quality = exported["quality"]
    print("Quality (exported):", {k: v for k, v in final_quality.items() if k != "warnings"})
    matched = sum(m["resolution_status"] == "resolved" for m in exported["mentions"])
    print("Resolved mentions:", matched, "/", len(exported["mentions"]))
    for warning in final_quality["warnings"][:12]:
        print("Warning:", warning)
    print("Reader:", result.normalized_html)
    print("JSON:", result.paper_json)


if __name__ == "__main__":
    main()