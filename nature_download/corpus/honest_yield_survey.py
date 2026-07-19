"""Honest yield survey over extreme (>=5-panel) multi-panel proposals.

Reads each panel's REAL (data_path, sheet) from a proposals jsonl and runs the
pipeline's OWN honest reader+analyzer (_read_xlsx + analyze_table). Reports, at
panel granularity, how many extreme panels are genuinely simple-2D and how many
>=5-panel cases have >=5 honest panels (i.e. are honestly composable).

This is the measurement the builder-side fabrication existed to hide: real
Nature >=5-panel source sheets are ragged / multi-header / merged, so almost none
pass the simple-2D contract. Run:

    python -m corpus.honest_yield_survey [proposed.jsonl]
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from corpus.proposals import (
    _read_xlsx,
    analyze_table,
    ProposalRejected,
    DEFAULT_MAX_ROWS,
    DEFAULT_MAX_COLUMNS,
)

DEFAULT_PROPOSED = (
    Path(__file__).resolve().parents[1]
    / "outputs/c2_extreme_proposals/proposed.jsonl"
)


def survey(proposed: Path, min_panels: int = 5) -> dict:
    reject = collections.Counter()
    panels_total = 0
    panels_ok = 0
    cases = []
    for line in proposed.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("proposal_type") != "multi_panel":
            continue
        ec = obj.get("experiment_case") or {}
        pans = ec.get("panels") or []
        if len(pans) < min_panels:
            continue
        n_ok = 0
        for p in pans:
            dp = p.get("data_path")
            sheet = p.get("sheet")
            panels_total += 1
            if not dp or not sheet:
                reject["missing-data_path/sheet"] += 1
                continue
            try:
                frame = _read_xlsx(
                    Path(dp),
                    sheet_name=sheet,
                    max_rows=DEFAULT_MAX_ROWS,
                    max_columns=DEFAULT_MAX_COLUMNS,
                )
                analyze_table(frame)
                n_ok += 1
                panels_ok += 1
            except ProposalRejected as exc:
                reject[str(exc).split(":")[0]] += 1
            except Exception as exc:  # noqa: BLE001
                reject["EXC:" + type(exc).__name__] += 1
        cases.append((obj.get("case_id"), len(pans), n_ok))
    viable = [c for c in cases if c[2] >= min_panels]
    return {
        "min_panels": min_panels,
        "cases": len(cases),
        "panels_total": panels_total,
        "panels_honest_simple2d": panels_ok,
        "panel_yield_pct": round(100 * panels_ok / max(1, panels_total), 2),
        "composable_cases": len(viable),
        "viable": viable,
        "reject_histogram": dict(reject.most_common()),
    }


def main(argv: list[str]) -> int:
    proposed = Path(argv[1]) if len(argv) > 1 else DEFAULT_PROPOSED
    result = survey(proposed)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
