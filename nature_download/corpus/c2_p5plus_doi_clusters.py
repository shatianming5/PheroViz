"""Count INDEPENDENT DOI clusters that have a composable P5+ (>=5-panel) figure.

The C2 experiment terminated at BLOCKED_INSUFFICIENT_INDEPENDENT_P5PLUS: it had
only 1 independent DOI cluster in the P=5+ stratum and required >=2. This module
measures, reproducibly, how many DISTINCT DOIs have at least one >=5-panel figure
whose panels are honestly composable (i.e. the honest fail-closed normalizer +
analyze_table accept >=5 of them). It is the direct data-sufficiency question the
C2 blocker turns on.

Run from nature_download/:
    python -m corpus.c2_p5plus_doi_clusters [--min-composable 5]
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path
from typing import Any

from corpus.proposals import ProposalRejected, analyze_table
from corpus.source_table_normalizer import NormalizerRejected, normalize_source_sheet

PROPOSED = Path("outputs/c2_extreme_proposals/proposed.jsonl")


def _panel_sheet(experiment_case: dict[str, Any], pid: str) -> str | None:
    for panel in experiment_case.get("panels") or []:
        if str(panel.get("panel_id")) == pid:
            return panel.get("sheet")
    return None


def analyze(proposed: Path, min_panels: int = 5, min_composable: int = 5) -> dict:
    rows = [json.loads(line) for line in open(proposed) if line.strip()]
    groups: dict[tuple, dict[str, tuple[str, str]]] = collections.defaultdict(dict)
    for r in rows:
        pids = r.get("panel_ids") or []
        if len(pids) != 1:
            continue
        ec = r.get("experiment_case") or {}
        data_path = ec.get("data_path")
        pid = str(pids[0])
        sheet = _panel_sheet(ec, pid)
        if data_path and sheet:
            groups[(r.get("doi"), r.get("figure_no"))][pid] = (str(data_path), str(sheet))

    by_doi: dict[str, list] = collections.defaultdict(list)
    for (doi, fig), panels in groups.items():
        if len(panels) < min_panels:
            continue
        n_comp = 0
        for _pid, (data_path, sheet) in panels.items():
            try:
                res = normalize_source_sheet(Path(data_path), sheet)
                analyze_table(res.frame)
                n_comp += 1
            except (NormalizerRejected, ProposalRejected):
                pass
        if n_comp >= min_composable:
            by_doi[str(doi)].append({"figure_no": fig, "panels": len(panels), "composable": n_comp})

    return {
        "min_panels": min_panels,
        "min_composable": min_composable,
        "independent_doi_clusters": len(by_doi),
        "c2_requirement": 2,
        "satisfied": len(by_doi) >= 2,
        "clusters": {doi: figs for doi, figs in sorted(by_doi.items())},
    }


def main(argv: list[str]) -> int:
    min_comp = 5
    if "--min-composable" in argv:
        min_comp = int(argv[argv.index("--min-composable") + 1])
    result = analyze(PROPOSED, min_composable=min_comp)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
