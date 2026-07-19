#!/usr/bin/env python3
"""Finalize the C2 dataset: union all crawl outputs by DOI and, once the unique
article count reaches a threshold (default 4000), freeze a locked copy under
``data/c2_p5_4k/``.

Design
------
* The heavy per-article content (figures PNGs, source-data XLSX/CSV/ZIP) is
  physically copied to ``data/c2_p5_4k/<article_id>/`` so the C2 pipeline can
  consume real files, but is **git-ignored** (a repo of 4000 articles would be
  tens of GB).
* Git tracks only the lightweight, *definitive* dataset definition:
  ``_manifest.jsonl`` (one row per article: doi, journal, year, license, figure
  URLs, source-data files + bytes + sha256), ``_frozen.json`` (freeze record:
  count, per-journal/year breakdown, git commit, manifest sha256), the small
  ``_provenance/<id>.json`` copies, and ``README.md``. These fully pin the
  dataset and make the heavy content reproducible from the recorded URLs.

Usage
-----
  # just report the current union (no copy):
  python3 finalize_c2_4k.py --dry-run

  # freeze once >= 4000 unique eligible articles exist (cap dataset at 4000):
  python3 finalize_c2_4k.py --threshold 4000

  # force a freeze now regardless of count (e.g. for testing):
  python3 finalize_c2_4k.py --force --cap 20 --dest data/_c2_4k_smoketest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
DEFAULT_ROOTS = [
    REPO / "nature_download" / "outputs" / "extreme_content",
    REPO / "nature_download" / "outputs" / "nature_content",
    REPO / "nature_download" / "outputs" / "elife_2020_2026",
    REPO / "nature_download" / "outputs" / "extreme_merged",
]
DEFAULT_DEST = REPO / "data" / "c2_p5_4k"

GITIGNORE_BODY = """\
# C2 4k dataset — heavy per-article content (figures/source_data/meta) is kept on
# local disk for the pipeline but NOT committed. Git tracks only the dataset
# definition below; the content is reproducible from _manifest.jsonl URLs+sha256.
*
!.gitignore
!README.md
!_manifest.jsonl
!_frozen.json
!_provenance/
!_provenance/**
"""


def sha256_of(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=str(REPO), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return ""


def load_json(path: Path):
    try:
        with path.open() as fh:
            return json.load(fh)
    except Exception:
        return None


JOURNAL_CANON = {
    "The EMBO Reports": "EMBO Reports",
    "The EMBO Journal": "The EMBO Journal",
}


def canon_journal(name: str | None) -> str:
    n = (name or "?").strip()
    return JOURNAL_CANON.get(n, n)


def discover(roots: list[Path]) -> dict[str, dict]:
    """Return {lower_doi: candidate} deduped across roots.

    A *usable* C2 article is a content directory that has BOTH rendered figures
    (``figures/*.png``) AND a non-empty ``source_data/`` folder -- data fidelity
    against the source table is the whole point, so articles without source data
    are excluded. Provenance (doi/journal/year) is read from the in-dir
    ``meta/provenance.json`` (eLife/EMBO) or, failing that, from the root-level
    ``_provenance/<article_id>.json`` ledger (Nature layout). On DOI collision
    keep the more complete candidate, breaking ties by root order.
    """
    best: dict[str, dict] = {}
    for order, root in enumerate(roots):
        if not root.is_dir():
            continue
        prov_dir = root / "_provenance"
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.name.startswith("_"):
                continue
            fig_dir = entry / "figures"
            sd_dir = entry / "source_data"
            figs = list(fig_dir.glob("*.png")) if fig_dir.is_dir() else []
            sds = [p for p in sd_dir.iterdir() if p.is_file()] if sd_dir.is_dir() else []
            if not figs or not sds:  # require both figures and source data
                continue
            prov = load_json(entry / "meta" / "provenance.json")
            if not prov and prov_dir.is_dir():
                prov = load_json(prov_dir / f"{entry.name}.json")
            if not prov:
                continue
            doi = (prov.get("doi") or "").strip().lower()
            if not doi:
                continue
            n_fig, n_sd = len(figs), len(sds)
            cand = {
                "article_id": entry.name,
                "dir": entry,
                "prov": prov,
                "n_fig": n_fig,
                "n_sd": n_sd,
                "root_order": order,
                # completeness score: more source-data, then more figures, earlier root
                "score": (n_sd, n_fig, -order),
            }
            prev = best.get(doi)
            if prev is None or cand["score"] > prev["score"]:
                best[doi] = cand
    return best


def manifest_row(doi: str, cand: dict, with_sha: bool) -> dict:
    prov = cand["prov"]
    d = cand["dir"]
    figs = load_json(d / "meta" / "figures.json") or []
    sds = load_json(d / "meta" / "source_data.json") or []
    fig_rows = []
    for f in figs if isinstance(figs, list) else []:
        fig_rows.append({
            "tag": f.get("figure_tag"),
            "no": f.get("figure_no"),
            "image_url": f.get("image_url"),
            "source_url": f.get("source_url"),
        })
    if not fig_rows:  # fall back to on-disk figures
        for p in sorted((d / "figures").glob("*.png")):
            fig_rows.append({"tag": p.stem, "no": None, "image_url": None, "source_url": None})
    sd_dir = d / "source_data"
    sd_rows = []
    seen_names = set()
    for s in sds if isinstance(sds, list) else []:
        name = s.get("saved_name") or (Path(s.get("saved_as", "")).name or None)
        if not name:
            continue
        seen_names.add(name)
        row = {"saved_name": name, "url": s.get("url"), "label": s.get("label")}
        p = sd_dir / name
        if p.is_file():
            row["bytes"] = p.stat().st_size
            if with_sha:
                row["sha256"] = sha256_of(p)
        sd_rows.append(row)
    if sd_dir.is_dir():  # add any on-disk source-data files missing from meta
        for p in sorted(sd_dir.iterdir()):
            if p.is_file() and p.name not in seen_names:
                row = {"saved_name": p.name, "url": None, "bytes": p.stat().st_size}
                if with_sha:
                    row["sha256"] = sha256_of(p)
                sd_rows.append(row)
    lic = prov.get("license") or {}
    return {
        "article_id": cand["article_id"],
        "doi": doi,
        "journal": canon_journal(prov.get("journal")),
        "year": prov.get("year"),
        "published_date": prov.get("published_date"),
        "article_url": prov.get("article_url"),
        "license": lic.get("license_id"),
        "n_figures": len(fig_rows),
        "n_source_data": len(sd_rows),
        "figures": fig_rows,
        "source_data": sd_rows,
    }


def summarize(cands: dict[str, dict]) -> tuple[dict, dict]:
    by_journal, by_year = {}, {}
    for c in cands.values():
        j = canon_journal(c["prov"].get("journal"))
        y = c["prov"].get("year") or "?"
        by_journal[j] = by_journal.get(j, 0) + 1
        by_year[str(y)] = by_year.get(str(y), 0) + 1
    return dict(sorted(by_journal.items(), key=lambda kv: -kv[1])), dict(sorted(by_year.items()))


def copy_article(cand: dict, dest_dir: Path, overwrite: bool) -> None:
    src = cand["dir"]
    if dest_dir.exists():
        if not overwrite:
            return
        shutil.rmtree(dest_dir)
    shutil.copytree(src, dest_dir)


def freeze(cands: dict[str, dict], dest: Path, cap: int | None, overwrite: bool,
           with_sha: bool) -> dict:
    # deterministic order: by DOI so the frozen set is reproducible
    items = sorted(cands.items(), key=lambda kv: kv[0])
    if cap is not None:
        items = items[:cap]
    chosen = dict(items)

    dest.mkdir(parents=True, exist_ok=True)
    prov_out = dest / "_provenance"
    prov_out.mkdir(exist_ok=True)

    manifest_rows = []
    for i, (doi, cand) in enumerate(items, 1):
        aid = cand["article_id"]
        copy_article(cand, dest / aid, overwrite)
        # small provenance copy for git
        with (prov_out / f"{aid}.json").open("w") as fh:
            json.dump(cand["prov"], fh, ensure_ascii=False, indent=1)
        manifest_rows.append(manifest_row(doi, cand, with_sha))
        if i % 200 == 0:
            print(f"  ... froze {i}/{len(items)}", flush=True)

    # manifest.jsonl
    man_path = dest / "_manifest.jsonl"
    with man_path.open("w") as fh:
        for r in manifest_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    man_sha = sha256_of(man_path)

    by_journal, by_year = summarize(chosen)
    frozen = {
        "dataset": "c2_p5_4k",
        "description": "Frozen C2 evaluation dataset: real published multi-panel "
                       "(>=5 panel) CC-BY figures with per-figure Source Data.",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": git("rev-parse", "HEAD"),
        "n_articles": len(items),
        "cap": cap,
        "by_journal": by_journal,
        "by_year": by_year,
        "manifest_sha256": man_sha,
        "source_roots": [str(r.relative_to(REPO)) for r in DEFAULT_ROOTS],
        "eligibility": "CC-BY, >=5 panels, per-figure source data present",
    }
    with (dest / "_frozen.json").open("w") as fh:
        json.dump(frozen, fh, ensure_ascii=False, indent=2)
    with (dest / ".gitignore").open("w") as fh:
        fh.write(GITIGNORE_BODY)
    with (dest / "_processed.txt").open("w") as fh:
        fh.write("\n".join(sorted(c["article_id"] for c in chosen.values())) + "\n")
    _write_readme(dest, frozen)
    return frozen


def _write_readme(dest: Path, frozen: dict) -> None:
    bj = "\n".join(f"| {k} | {v} |" for k, v in frozen["by_journal"].items())
    readme = f"""# C2 dataset (`c2_p5_4k`) — FROZEN

Frozen: `{frozen['frozen_at']}`  ·  commit `{frozen['git_commit'][:12]}`  ·  **{frozen['n_articles']} articles**

Real published multi-panel scientific figures with per-figure **Source Data**
(CC-BY, >=5 panels), harvested from Nature Communications, eLife, and the EMBO
family. Each article dir holds `figures/` (PNG + caption `.txt`), `source_data/`
(XLSX/CSV/ZIP), and `meta/` (provenance + figure/source-data manifests).

## What is version-controlled
To keep the git repo small, **only the dataset definition is committed**:
`_manifest.jsonl`, `_frozen.json`, `_provenance/*.json`, this `README.md`. The
heavy per-article content is git-ignored (see `.gitignore`) and lives on local
disk. Every source-data file is pinned by `bytes` + `sha256` and every figure /
source-data file has its origin URL in `_manifest.jsonl`, so the content is fully
reproducible.

## Per-journal breakdown
| Journal | Articles |
|---|---|
{bj}

## Rebuilding the heavy content
Re-run the harvesters (`nature_download/`), or re-download each file from the
`article_url` / `source_data[].url` recorded in `_manifest.jsonl` and verify the
`sha256`.

Regenerate this frozen snapshot with:
```
python3 finalize_c2_4k.py --threshold {frozen['n_articles']} --dest data/c2_p5_4k
```
"""
    with (dest / "README.md").open("w") as fh:
        fh.write(readme)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    ap.add_argument("--threshold", type=int, default=4000,
                    help="freeze only when unique article count >= this")
    ap.add_argument("--cap", type=int, default=None,
                    help="cap frozen dataset to N articles (default: = threshold)")
    ap.add_argument("--no-cap", action="store_true", help="include ALL unique articles")
    ap.add_argument("--force", action="store_true", help="freeze regardless of threshold")
    ap.add_argument("--dry-run", action="store_true", help="report union only, no copy")
    ap.add_argument("--overwrite", action="store_true", help="recopy existing article dirs")
    ap.add_argument("--no-sha", action="store_true", help="skip sha256 (faster, for testing)")
    ap.add_argument("--roots", nargs="*", default=None)
    args = ap.parse_args()

    roots = [Path(r) for r in args.roots] if args.roots else DEFAULT_ROOTS
    dest = Path(args.dest)
    cap = None if args.no_cap else (args.cap if args.cap is not None else args.threshold)

    print("Scanning crawl outputs for eligible articles ...", flush=True)
    cands = discover(roots)
    n = len(cands)
    by_journal, by_year = summarize(cands)
    print(f"\nUNIQUE eligible articles (by DOI): {n}")
    print("by journal:", json.dumps(by_journal, ensure_ascii=False))
    print("by year:   ", json.dumps(by_year, ensure_ascii=False))

    if args.dry_run:
        print(f"\n[dry-run] threshold={args.threshold} -> "
              f"{'READY to freeze' if n >= args.threshold else f'need {args.threshold - n} more'}")
        return 0

    if not args.force and n < args.threshold:
        print(f"\nNot freezing: {n} < threshold {args.threshold} "
              f"(need {args.threshold - n} more). Re-run later or use --force.")
        return 3

    eff_cap = None if cap is None else min(cap, n)
    print(f"\nFreezing -> {dest}  (cap={eff_cap}, sha256={'off' if args.no_sha else 'on'})",
          flush=True)
    frozen = freeze(cands, dest, eff_cap, args.overwrite, not args.no_sha)
    print(f"\nFROZEN {frozen['n_articles']} articles -> {dest}")
    print("manifest sha256:", frozen["manifest_sha256"])
    print("by journal:", json.dumps(frozen["by_journal"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
