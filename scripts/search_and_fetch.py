#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_KEYWORDS = [
    # Oncology / Immunology
    "cancer", "tumor microenvironment", "immunotherapy", "checkpoint inhibitor",
    "PD-1", "CTLA-4", "CAR-T", "tumour biomarker", "precision oncology",
    # Genomics / Multi-omics
    "genomics", "transcriptomics", "single-cell", "scRNA-seq", "epigenomics",
    "multi-omics", "CRISPR", "gene editing",
    # AI / Methods
    "machine learning", "deep learning", "foundation model", "graph neural network",
    # Metabolism / Microbiome
    "metabolism", "metabolomics", "microbiome", "gut microbiota",
    # Neuroscience
    "neuroscience", "brain", "neurodegeneration", "Alzheimer",
    # Materials / Physics / Chemistry
    "materials", "quantum", "superconductivity", "perovskite", "catalysis",
    # Climate / Earth / Environment
    "climate change", "carbon", "ocean", "biodiversity", "ecosystem",
    # Bioinformatics / Systems biology
    "bioinformatics", "systems biology", "network biology",
    # Medicine / Public health
    "COVID-19", "vaccination", "rare disease", "drug discovery",
    # Chinese keywords
    "肿瘤免疫", "免疫检查点", "单细胞", "基因编辑", "代谢组学", "微生物组",
]


def run(cmd: list[str]) -> int:
    print("[exec] ", " ".join(cmd))
    return subprocess.call(cmd)


def main():
    p = argparse.ArgumentParser(description="Search multiple keywords then fetch ALL content (figures + source data)")
    p.add_argument("--keywords-file", default=None, help="Optional keywords file (one per line). If omitted, use built-in list")
    p.add_argument("--max-per-keyword", type=int, default=50, help="Max search results per keyword")
    p.add_argument("--search-out", default="outputs/search_auto", help="Directory for search outputs (merged)")
    p.add_argument("--content-out", default="outputs/nature_content", help="Directory for fetched content")
    p.add_argument("--mailto", default=None, help="Optional Crossref mailto identifier")
    p.add_argument("--sleep", type=float, default=1.0, help="Sleep seconds between requests")
    p.add_argument("--timeout", type=float, default=30, help="Timeout seconds per request")
    p.add_argument("--max-retries", type=int, default=3, help="Max retries per request")
    p.add_argument("--max-articles", type=int, default=0, help="Limit number of articles in post-fetch (0=all)")
    p.add_argument("--max-figs", type=int, default=12, help="Max figures per article in post-fetch")
    p.add_argument("--sort", choices=["year_desc", "year_asc", "input"], default="year_desc", help="Post-fetch article order")
    args = p.parse_args()

    # load keywords
    keywords = []
    if args.keywords_file:
        for line in Path(args.keywords_file).read_text(encoding="utf-8").splitlines():
            kw = line.strip()
            if kw:
                keywords.append(kw)
    else:
        keywords = DEFAULT_KEYWORDS

    print(f"[info] Keywords: {len(keywords)} items")

    # run search for each keyword, append + dedup into one JSONL
    for kw in keywords:
        run([
            sys.executable, "scripts/nature_cli.py",
            "--query", kw,
            "--max", str(args.max_per_keyword),
            "--out", args.search_out,
            "--sleep", str(args.sleep),
            "--timeout", str(args.timeout),
            "--max-retries", str(args.max_retries),
            "--append",
        ] + (["--mailto", args.mailto] if args.mailto else []))

    jsonl = str(Path(args.search_out) / "articles.jsonl")
    # post-fetch ALL for Nature.com articles
    run([
        sys.executable, "scripts/nature_post_fetch.py",
        "--jsonl", jsonl,
        "--out", args.content_out,
        "--max-figs", str(args.max_figs),
        "--max-articles", str(args.max_articles),
        "--sort", args.sort,
        "--sleep", str(args.sleep),
        "--timeout", str(args.timeout),
        "--max-retries", str(args.max_retries),
    ])

    print("[done] Search and full-content fetch completed.")


if __name__ == "__main__":
    main()

