#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def ensure_package(module_name: str, pip_name: str | None = None):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        to_install = pip_name or module_name
        print(f"[setup] Installing missing dependency: {to_install} ...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", to_install])
        return importlib.import_module(module_name)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def norm_article_url(url: str | None, doi: str | None) -> str | None:
    if url:
        m = re.search(r"https?://(?:www\.)?nature\.com/articles/([^/?#]+)", url)
        if m:
            aid = m.group(1)
            return f"https://www.nature.com/articles/{aid}"
    if doi and doi.lower().startswith("10.1038/"):
        suffix = doi.split("/", 1)[1]
        return f"https://www.nature.com/articles/{suffix}"
    return None


def article_id_from_url(article_url: str) -> str:
    m = re.search(r"/articles/([^/?#]+)", article_url)
    return m.group(1) if m else "unknown"


def run(cmd: list[str]) -> int:
    try:
        print("[exec] ", " ".join(cmd))
        return subprocess.call(cmd)
    except Exception as e:
        print(f"[warn] failed to run: {e}")
        return 1


def main():
    p = argparse.ArgumentParser(description="Post-fetch Nature.com figures and source data for search results (authorized use)")
    p.add_argument("--jsonl", required=True, help="Path to articles.jsonl produced by nature_cli.py")
    p.add_argument("--out", default="outputs/nature_content", help="Base output directory for fetched content")
    p.add_argument("--authorized-figs", action="store_true", help="Fetch figure images+captions from nature.com")
    p.add_argument("--authorized-source", action="store_true", help="Fetch 'Source data' attachments from nature.com")
    p.add_argument("--max-figs", type=int, default=8, help="Max figures to probe per article")
    p.add_argument("--max-articles", type=int, default=0, help="Limit number of articles (0 means all)")
    p.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between sub-requests")
    p.add_argument("--timeout", type=float, default=30, help="Timeout seconds for sub-requests")
    p.add_argument("--max-retries", type=int, default=3, help="Retries for sub-requests")
    args = p.parse_args()

    rows = load_jsonl(Path(args.jsonl))
    print(f"[info] Loaded {len(rows)} records from {args.jsonl}")

    taken = 0
    for idx, r in enumerate(rows, 1):
        art_url = norm_article_url(r.get("url"), r.get("doi"))
        if not art_url:
            continue
        aid = article_id_from_url(art_url)
        print(f"[{idx}] Nature article detected: {aid}")

        if args.authorized_figs:
            for i in range(1, args.max_figs + 1):
                fig_url = f"{art_url}/figures/{i}"
                code = run([
                    sys.executable, "scripts/nature_fig_fetch.py",
                    "--url", fig_url,
                    "--out", args.out,
                    "--sleep", str(args.sleep),
                    "--timeout", str(args.timeout),
                    "--max-retries", str(args.max_retries),
                ])
                # best-effort; do not hard-stop on non-zero exit
                time.sleep(args.sleep)

        if args.authorized_source:
            code = run([
                sys.executable, "scripts/nature_source_data_fetch.py",
                "--url", art_url,
                "--out", args.out,
                "--sleep", str(args.sleep),
                "--timeout", str(args.timeout),
                "--max-retries", str(args.max_retries),
            ])
            time.sleep(args.sleep)

        taken += 1
        if args.max_articles and taken >= args.max_articles:
            break

    print("[done] Post-fetch complete.")


if __name__ == "__main__":
    main()

