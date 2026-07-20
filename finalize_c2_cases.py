#!/usr/bin/env python3
"""Freeze the C2 *case-level* dataset (panel-level, machine-composable).

Unlike ``finalize_c2_4k.py`` (which freezes 4000 *articles*), this tool freezes
individual **panel cases**: one row per (doi, figure, panel) whose source table
is honestly composable under the pipeline's own simple-2D contract
(``corpus.c2_p5plus_doi_clusters._panel_composable``).

Two provenance tiers are recorded per case (HONEST, no overclaim):

* ``A_raw``            -- the published source table is already a clean 2D sheet;
                         zero reshaping. Directly sealed-eligible.
* ``B_normalizer``    -- the published sheet was ragged/multi-header/merged and
                         was reshaped by the deterministic
                         ``source_table_normalizer.normalize_source_sheet``.
                         Fully traceable (raw xlsx sha256 + recipe + output
                         sha256) but flagged EXPLORATORY: using tier B requires
                         approving the normalization step as part of the
                         benchmark contract.

Every case pins the byte-identical *published original* (raw xlsx/csv sha256)
so the freeze is reproducible against the source of record.

Run from repo root:
    python finalize_c2_cases.py \
        --proposed nature_download/outputs/c2_full_casecount_exploratory_20260720/corrected_sheet_binding/proposals/proposed.jsonl \
        --dest data/c2_cases_v1
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
ND = REPO / "nature_download"
sys.path.insert(0, str(ND))

from corpus.c2_p5plus_doi_clusters import _panel_composable, _panel_sheet  # noqa: E402

DEFAULT_PROPOSED = (
    ND
    / "outputs/c2_full_casecount_exploratory_20260720"
    / "corrected_sheet_binding/proposals/proposed.jsonl"
)
DEFAULT_DEST = REPO / "data/c2_cases_v1"
FROZEN_4K = REPO / "data/c2_p5_4k/_manifest.jsonl"

JOURNAL_BY_PREFIX = [
    ("10.7554/elife", "eLife"),
    ("10.1038/s41467", "Nature Communications"),
    ("10.1038/s44318", "The EMBO Journal"),
    ("10.1038/s44319", "EMBO Reports"),
    ("10.1038/s44320", "Molecular Systems Biology"),
    ("10.1038/s44321", "EMBO Molecular Medicine"),
    ("10.26508/lsa", "Life Science Alliance"),
]


def sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def infer_journal(doi: str) -> str:
    d = doi.strip().lower()
    for pre, name in JOURNAL_BY_PREFIX:
        if d.startswith(pre):
            return name
    return "?"


def load_journal_year_map() -> dict[str, tuple[str | None, int | None]]:
    m: dict[str, tuple[str | None, int | None]] = {}
    if FROZEN_4K.is_file():
        for line in open(FROZEN_4K):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            m[o["doi"].strip().lower()] = (o.get("journal"), o.get("year"))
    return m


def rel(path_str: str | None) -> str | None:
    if not path_str:
        return None
    p = Path(path_str)
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return path_str


def build(proposed: Path, dest: Path) -> dict:
    rows = [json.loads(l) for l in open(proposed) if l.strip()]
    jymap = load_journal_year_map()

    # composability cache keyed by (path, sheet)
    cache: dict[tuple[str, str | None], bool] = {}

    def composable(path: str, sheet: str | None) -> bool:
        key = (path, sheet)
        if key not in cache:
            cache[key] = _panel_composable(path, sheet)
        return cache[key]

    # dedupe by (doi, figure_no, panel_id): keep first composable occurrence
    seen: set[tuple] = set()
    cases: list[dict] = []
    fig_comp: dict[tuple, int] = collections.Counter()

    for r in rows:
        pids = r.get("panel_ids") or []
        if len(pids) != 1:
            continue
        ec = r.get("experiment_case") or {}
        dp = ec.get("data_path")
        if not dp:
            continue
        pid = str(pids[0])
        sheet = _panel_sheet(ec, pid)
        if not composable(str(dp), sheet):
            continue
        doi = (r.get("doi") or "").strip()
        fig = r.get("figure_no")
        ukey = (doi.lower(), fig, pid)
        if ukey in seen:
            continue
        seen.add(ukey)
        fig_comp[(doi.lower(), fig)] += 1

        norm = r.get("source_normalization")
        st = r.get("source_table") or {}
        raw = r.get("raw_source_table") or {}
        tier = "B_normalizer" if norm else "A_raw"

        # the published ORIGINAL of record: raw_source_table for tier B, else source_table
        original = raw if (norm and raw.get("path")) else st
        orig_path = original.get("path")
        orig_on_disk = bool(orig_path and os.path.isfile(orig_path))

        journal, year = jymap.get(doi.lower(), (None, None))
        if not journal:
            journal = infer_journal(doi)

        case = {
            "case_id": ec.get("case_id"),
            "doi": doi,
            "journal": journal,
            "year": year,
            "figure_no": fig,
            "panel_id": pid,
            "chart_family": ec.get("chart_family"),
            "provenance_tier": tier,
            # published original (byte-pinned to source of record)
            "original_source_table": {
                "format": original.get("format"),
                "relative_path": rel(orig_path),
                "sha256": original.get("sha256"),
                "size_bytes": original.get("size_bytes"),
                "sheet_name": original.get("sheet_name"),
                "source_url": original.get("source_url"),
                "on_disk": orig_on_disk,
            },
            # working table the model actually reads (tier B: normalized csv)
            "working_source_table": {
                "format": st.get("format"),
                "relative_path": rel(st.get("path")),
                "sha256": st.get("sha256"),
                "size_bytes": st.get("size_bytes"),
                "path_root": st.get("path_root"),
            },
            # deterministic reshaping recipe (tier B only)
            "source_normalization": (
                {
                    "implementation": norm.get("implementation"),
                    "header_row_index": norm.get("header_row_index"),
                    "orientation": norm.get("orientation"),
                    "dropped_side_blocks": norm.get("dropped_side_blocks"),
                    "aggregated_replicates": norm.get("aggregated_replicates"),
                    "normalizer_output_sha256": norm.get("normalizer_output_sha256"),
                    "notes": norm.get("notes"),
                }
                if norm
                else None
            ),
            "intent": ec.get("intent"),
            "evaluation_expectation": ec.get("evaluation_expectation"),
        }
        cases.append(case)

    # attach figure-level P5 stratum flags
    for c in cases:
        n = fig_comp[(c["doi"].lower(), c["figure_no"])]
        c["figure_composable_panels"] = n
        c["is_p5plus"] = n >= 5

    # deterministic order
    cases.sort(key=lambda c: (c["doi"].lower(), str(c["figure_no"]), c["panel_id"]))

    dest.mkdir(parents=True, exist_ok=True)
    man_path = dest / "_manifest.jsonl"
    with open(man_path, "w") as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_sha = sha256_file(man_path)

    # ---- summary / stratification ----
    tiers = collections.Counter(c["provenance_tier"] for c in cases)
    by_journal = collections.Counter(c["journal"] for c in cases)
    by_year = collections.Counter(str(c["year"]) for c in cases)
    orig_present = sum(1 for c in cases if c["original_source_table"]["on_disk"])
    dois = {c["doi"].lower() for c in cases}
    figs = {(c["doi"].lower(), c["figure_no"]) for c in cases}

    # panel_count strata by figure composable panels
    fig_bucket = collections.Counter()
    for key, n in fig_comp.items():
        b = "5+" if n >= 5 else str(n)
        fig_bucket[b] += 1

    p5_cases = [c for c in cases if c["is_p5plus"]]
    p5_figs = {(c["doi"].lower(), c["figure_no"]) for c in p5_cases}
    p5_dois = {c["doi"].lower() for c in p5_cases}
    p5_tier = collections.Counter(c["provenance_tier"] for c in p5_cases)

    frozen = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "unit": "panel_case",
        "definition": (
            "One row per (doi, figure, panel) whose source table is honestly "
            "composable under corpus.c2_p5plus_doi_clusters._panel_composable "
            "(simple-2D contract). Panel-level is the natural benchmark unit; "
            "the >=5-composable-panel figures form the P5+ compositional stratum."
        ),
        "source_proposals": rel(str(proposed)),
        "source_proposals_sha256": sha256_file(proposed),
        "manifest": rel(str(man_path)),
        "manifest_sha256": manifest_sha,
        "content_materialized": False,
        "total_cases": len(cases),
        "unique_dois": len(dois),
        "unique_figures": len(figs),
        "provenance_tiers": dict(tiers),
        "original_source_on_disk": {
            "present": orig_present,
            "missing": len(cases) - orig_present,
        },
        "panel_stratum_by_figure_composable_panels": dict(sorted(fig_bucket.items())),
        "p5plus": {
            "cases": len(p5_cases),
            "figures": len(p5_figs),
            "dois": len(p5_dois),
            "by_tier": dict(p5_tier),
        },
        "by_journal": dict(by_journal.most_common()),
        "by_year": dict(sorted(by_year.items())),
        "eligibility": (
            "HONEST SCOPE: tier A_raw cases use the published sheet verbatim "
            "(sealed-eligible). tier B_normalizer cases apply the deterministic, "
            "fully-traceable source_table_normalizer to a ragged published sheet "
            "(raw sha256 + recipe + output sha256 all pinned); tier B is "
            "EXPLORATORY and requires approving the normalization step as part of "
            "the benchmark contract. No case is human-review-verified yet "
            "(eligible_for_experiment=0 upstream)."
        ),
    }
    (dest / "_frozen.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n"
    )
    return frozen


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposed", default=str(DEFAULT_PROPOSED))
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    args = ap.parse_args(argv)
    frozen = build(Path(args.proposed), Path(args.dest))
    print(json.dumps(frozen, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
