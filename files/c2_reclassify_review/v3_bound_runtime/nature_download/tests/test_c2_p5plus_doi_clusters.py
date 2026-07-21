"""Regression tests for the C2 P5+ independent-DOI-cluster counter.

The counter re-validates each proposed panel's source table before crediting a
figure. It previously (a) dropped every panel whose ``sheet`` was ``None`` while
grouping and (b) fed CSV files to the xlsx-only normalizer, so any figure whose
per-panel Source Data ships as CSV was silently counted as non-composable. That
undercount could hide legitimate >=5-composable-panel DOIs and wrongly report the
C2 P5+ stratum as insufficient. These tests pin the honest, format-agnostic
behaviour.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from openpyxl import Workbook

from nature_download.corpus.c2_p5plus_doi_clusters import analyze


def _write_composable_csv(path: Path) -> None:
    # Monotonic numeric x + a non-monotonic numeric y => analyze_table accepts it
    # as a linear line chart (exactly one monotonic x column).
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["X", "Y"])
        for x, y in ((1, 2), (2, 5), (3, 3), (4, 8), (5, 4)):
            writer.writerow([x, y])


def _write_composable_xlsx(path: Path, sheet_name: str) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = sheet_name
    worksheet.append(["X", "Y"])
    for x, y in ((1, 2), (2, 5), (3, 3), (4, 8), (5, 4)):
        worksheet.append([x, y])
    workbook.save(path)


def _single_proposal(doi: str, figure_no: int, panel_id: str, data_path: Path, sheet):
    return {
        "doi": doi,
        "figure_no": figure_no,
        "panel_ids": [panel_id],
        "experiment_case": {
            "data_path": str(data_path),
            "panels": [{"panel_id": panel_id, "sheet": sheet}],
        },
    }


def _write_proposed(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_csv_backed_p5plus_figure_is_counted(workdir: Path) -> None:
    doi = "10.1038/s44319-025-00503-8"
    records = []
    for idx, panel in enumerate("cdefg"):
        csv_path = workdir / f"panel_{panel}.csv"
        _write_composable_csv(csv_path)
        records.append(_single_proposal(doi, 4, panel, csv_path, None))
    proposed = workdir / "proposed.jsonl"
    _write_proposed(proposed, records)

    result = analyze(proposed, min_panels=5, min_composable=5)

    assert result["satisfied"] is False  # one cluster is not enough on its own
    assert result["independent_doi_clusters"] == 1
    assert result["clusters"][doi] == [{"figure_no": 4, "panels": 5, "composable": 5}]


def test_two_independent_csv_clusters_satisfy_c2(workdir: Path) -> None:
    records = []
    for doi, figure_no in (("10.1038/a", 4), ("10.1038/b", 3)):
        for panel in "cdefg":
            csv_path = workdir / f"{doi.replace('/', '_')}_{panel}.csv"
            _write_composable_csv(csv_path)
            records.append(_single_proposal(doi, figure_no, panel, csv_path, None))
    proposed = workdir / "proposed.jsonl"
    _write_proposed(proposed, records)

    result = analyze(proposed, min_panels=5, min_composable=5)

    assert result["independent_doi_clusters"] == 2
    assert result["satisfied"] is True


def test_mixed_csv_and_xlsx_panels_both_validate(workdir: Path) -> None:
    doi = "10.1038/mixed"
    records = []
    for panel in "cde":
        csv_path = workdir / f"panel_{panel}.csv"
        _write_composable_csv(csv_path)
        records.append(_single_proposal(doi, 2, panel, csv_path, None))
    for panel in "fg":
        xlsx_path = workdir / f"panel_{panel}.xlsx"
        _write_composable_xlsx(xlsx_path, "Fig2")
        records.append(_single_proposal(doi, 2, panel, xlsx_path, "Fig2"))
    proposed = workdir / "proposed.jsonl"
    _write_proposed(proposed, records)

    result = analyze(proposed, min_panels=5, min_composable=5)

    assert result["clusters"][doi] == [{"figure_no": 2, "panels": 5, "composable": 5}]


def test_below_min_panels_not_counted(workdir: Path) -> None:
    doi = "10.1038/small"
    records = []
    for panel in "cdef":  # only four panels
        csv_path = workdir / f"panel_{panel}.csv"
        _write_composable_csv(csv_path)
        records.append(_single_proposal(doi, 1, panel, csv_path, None))
    proposed = workdir / "proposed.jsonl"
    _write_proposed(proposed, records)

    result = analyze(proposed, min_panels=5, min_composable=5)

    assert result["independent_doi_clusters"] == 0
    assert doi not in result["clusters"]
