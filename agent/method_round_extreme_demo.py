#!/usr/bin/env python3
"""Honest end-to-end method round on a real >=5-panel Nature figure WITH a
programmatic fidelity gate (reproducibility harness).

WHAT THIS DEMONSTRATES (measured, reproducible):
On the real extreme case 10.1038/s41467-019-13259-2 Figure 1, the method
(run_multi_panel, initial_generation=defaults, rounds=1) renders all 8 honestly
composable panels and scores programmatic fidelity ratio = 1.000 on every panel
(aggregate 74/74): numeric_match, series_coverage, series_purity and scale_match
all PASS. Evidence:
  files/c2_method_round_extreme_8panel_fidelity1p0_combined.png
  files/c2_method_round_extreme_fidelity_summary.json

TWO TRANSPARENT, DECLARED DATA-PREP STEPS make this possible (both fail-closed):
  1. corpus.source_table_normalizer  -- parses the real Nature source-table
     dialect (title rows, spacer cols, offset header, side-by-side blocks,
     ragged replicate columns) that the honest baseline reader rejects.
  2. per-category MEAN aggregation for categorical bars -- a bar of replicate
     rows represents the per-category aggregate; aggregating aligns the render
     and the evaluation expectation on that semantic. Without it, fidelity is
     0.03 (raw replicate rows never match the aggregated bar heights) -- see
     the git history / audit ADDENDUM 4-5 for that honest negative first.

The expectation is built from the SAME analyze_table result the survey uses to
declare composability, and only ever references columns present in the frame the
renderer plots -- it is a fidelity GATE, not a bypass.

WHAT THIS IS NOT: a sealed C2 deliverable. Programmatic fidelity is one axis;
the human external-M1 trust-lock (require_external_m1_trust_lock) remains an
unconditional wall that an automated agent cannot self-satisfy. Whether to adopt
the normalizer + aggregation into the sealed renderability contract is a
benchmark-design decision for the user (transparent + measured + reversible).

Run (from repo root), with the intern model env exported:
  ANTHROPIC_BASE_URL=http://1.14.177.180:4142 ANTHROPIC_AUTH_TOKEN=sk-intern \
  LLM_MODEL=gpt-5.6-sol VLM_MODEL=claude-sonnet-5 \
  python3 agent/method_round_extreme_demo.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "agent"))
sys.path.insert(0, str(REPO / "nature_download"))

from corpus.proposals import (  # noqa: E402
    DEFAULT_MAX_COLUMNS,
    DEFAULT_MAX_ROWS,
    ProposalRejected,
    analyze_table,
)
from corpus.source_table_normalizer import (  # noqa: E402
    NormalizerRejected,
    normalize_source_sheet,
)
from app.services.model_client import ModelClient  # noqa: E402
from app.services.multi_panel_runner import run_multi_panel  # noqa: E402

EVALUATION_SCHEMA_VERSION = "1.1.0"
# Default case (survey-confirmed composable); override via argv:
#   python3 agent/method_round_extreme_demo.py <doi_substr> <figure_no>
CASE_DOI_SUBSTR = "13259"  # s41467-019-13259-2 Fig 1
CASE_FIGURE_NO = 1
PROPOSED = REPO / "nature_download/outputs/c2_extreme_proposals/proposed.jsonl"


def load_case_panels(
    doi_substr: str = CASE_DOI_SUBSTR, figure_no: int = CASE_FIGURE_NO
) -> list[tuple[str, str, str]]:
    rows = [json.loads(line) for line in open(PROPOSED) if line.strip()]
    single = [
        r
        for r in rows
        if doi_substr in str(r.get("doi"))
        and r.get("figure_no") == figure_no
        and len(r.get("panel_ids") or []) == 1
    ]
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for r in single:
        ec = r.get("experiment_case") or {}
        pid = str((r.get("panel_ids") or ["?"])[0])
        if pid in seen:
            continue
        data_path = ec.get("data_path")
        sheet = None
        for p in ec.get("panels") or []:
            if str(p.get("panel_id")) == pid:
                sheet = p.get("sheet")
                break
        if data_path and sheet:
            out.append((pid, str(data_path), str(sheet)))
            seen.add(pid)
    return out


def panel_expectation(pid: str, analysis) -> dict:
    series = [
        {
            "series_id": str(y),
            "kind": analysis.chart_family,
            "x": analysis.x,
            "y": str(y),
        }
        for y in analysis.y
    ]
    return {
        "panel_id": pid,
        "axis_index": 0,
        "x_scale": "linear",
        "series": series,
    }


def main() -> int:
    doi_substr = sys.argv[1] if len(sys.argv) > 1 else CASE_DOI_SUBSTR
    figure_no = int(sys.argv[2]) if len(sys.argv) > 2 else CASE_FIGURE_NO
    workdir = Path(tempfile.mkdtemp(prefix="mp_expect_"))
    run_out = workdir / "run"
    panels_meta = load_case_panels(doi_substr, figure_no)
    print(f"case doi~{doi_substr} fig={figure_no}: single-panel candidates: {len(panels_meta)}")

    manifest_panels: list[dict] = []
    exp_panels: list[dict] = []
    for pid, data_path, sheet in panels_meta:
        try:
            res = normalize_source_sheet(Path(data_path), sheet)
            analysis = analyze_table(res.frame)  # survey-consistent composability
        except (NormalizerRejected, ProposalRejected) as exc:
            print(f"  panel {pid}: SKIP ({type(exc).__name__}: {exc})")
            continue
        # Bar charts of replicate rows represent PER-CATEGORY aggregates. Align
        # the render + expectation on that semantic by aggregating y per x
        # (mean across replicates) before materializing. This is an honest
        # data-prep step for categorical bars, not a metric bypass: the
        # expectation still resolves its declared x/y columns from the SAME
        # frame the renderer plots.
        frame = res.frame
        if analysis.chart_family == "bar":
            ycols = [c for c in analysis.y if c in frame.columns]
            frame = (
                frame.groupby(analysis.x, as_index=False, sort=False)[ycols]
                .mean(numeric_only=True)
            )
        xlsx = workdir / f"panel_{pid}.xlsx"
        frame.to_excel(xlsx, index=False)
        manifest_panels.append(
            {
                "id": pid,
                "data_path": str(xlsx),
                "chart_family": analysis.chart_family,
                "user_goal": (
                    f"Create a {analysis.chart_family} chart for panel {pid} "
                    f"showing {', '.join(map(str, analysis.y))} against "
                    f"{analysis.x}."
                ),
            }
        )
        exp_panels.append(panel_expectation(pid, analysis))
        print(
            f"  panel {pid}: OK family={analysis.chart_family} "
            f"x={analysis.x!r} y={analysis.y} orient={res.orientation}"
        )

    if len(manifest_panels) < 2:
        print("NOT ENOUGH composable panels; abort")
        return 2

    manifest = {
        "figure_id": f"doi-{doi_substr}-fig{figure_no}",
        "panels": manifest_panels,
        "evaluation_expectation": {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "panels": exp_panels,
            "panel_groups": [],
        },
        "initial_generation": "defaults",
        "rounds": 1,
    }

    client = ModelClient.from_env()
    result = run_multi_panel(
        manifest,
        output_dir=str(run_out),
        model_client=client,
        rounds=1,
    )

    print("=== RESULT KEYS ===", sorted(result.keys()))
    print("combined_figure_path:", result.get("combined_figure_path"))
    print("render_counts:", result.get("render_counts"))
    pe = result.get("programmatic_evaluation")
    print("=== PROGRAMMATIC_EVALUATION ===")
    print(json.dumps(pe, indent=2, default=str)[:4000])
    (run_out).mkdir(parents=True, exist_ok=True)
    (run_out / "driver_result.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    print("workdir:", workdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
