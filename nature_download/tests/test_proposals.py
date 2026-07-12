from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
from openpyxl import Workbook
import pytest

from nature_download.corpus.proposals import (
    DEFAULT_MAX_COLUMNS,
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_ROWS,
    PROPOSAL_RULE_V1,
    propose_cases,
    propose_single_candidate,
    write_proposal_outputs,
)
from nature_download.corpus.provenance import sha256_file


DOI = "10.1038/s41467-024-13579-1"
EXPECTATION_SCHEMA = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "agent/app/evaluation/schemas/expectation.schema.json"
    ).read_text(encoding="utf-8")
)


def candidate_for(
    path: Path,
    *,
    candidate_id: str,
    panel_id: str = "a",
    figure_no: int = 1,
    sheet_name: str | None = None,
    status: str = "unverified",
    experiment_case: dict | None = None,
) -> dict:
    return {
        "schema_version": "1.0",
        "candidate_id": candidate_id,
        "doi": DOI,
        "figure_no": figure_no,
        "panel_ids": [panel_id],
        "source_table": {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "format": path.suffix.lstrip("."),
            "sheet_name": sheet_name,
        },
        "curation_status": status,
        "eligible_for_experiment": status == "verified",
        "experiment_case": experiment_case,
    }


def write_candidates(path: Path, candidates: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(candidate, sort_keys=True) + "\n"
            for candidate in candidates
        ),
        encoding="utf-8",
    )


def write_xlsx(path: Path, sheet_name: str, rows: list[list]) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = sheet_name
    for row in rows:
        worksheet.append(row)
    workbook.save(path)
    workbook.close()


def run_one(workdir: Path, candidate: dict) -> tuple[list[dict], list[dict], dict]:
    source = workdir / "candidates.jsonl"
    write_candidates(source, [candidate])
    result = propose_cases(candidates_path=source, code_commit="test-commit")
    for proposal in result[0]:
        Draft202012Validator(EXPECTATION_SCHEMA).validate(
            proposal["experiment_case"]["evaluation_expectation"]
        )
    return result


def test_explicit_time_generates_line_with_real_columns_and_units(
    workdir: Path,
) -> None:
    table = workdir / "line.csv"
    table.write_text(
        "Time (s),Signal (mV),Baseline\n0,1.0,2.0\n1,1.5,2.1\n2,1.2,2.2\n",
        encoding="utf-8",
    )
    proposed, rejected, summary = run_one(
        workdir,
        candidate_for(table, candidate_id="line-a"),
    )
    assert rejected == []
    assert summary["single_proposals"] == 1
    proposal = proposed[0]
    case = proposal["experiment_case"]
    assert case["chart_family"] == "line"
    assert case["intent"]["x"] == "Time (s)"
    assert case["intent"]["y"] == "Signal (mV)"
    assert case["intent"]["series"] == ["Signal (mV)", "Baseline"]
    assert case["intent"]["units"] == {"Time (s)": "s", "Signal (mV)": "mV"}
    assert case["evaluation_expectation"]["schema_version"] == "1.1.0"
    series = case["evaluation_expectation"]["panels"][0]["series"]
    assert {(item["x"], item["y"]) for item in series} == {
        ("Time (s)", "Signal (mV)"),
        ("Time (s)", "Baseline"),
    }
    assert proposal["curation_status"] == "proposed"
    assert proposal["eligible_for_experiment"] is False
    assert proposal["eligibility_reasons"] == ["external-validation-required"]
    assert proposal["code_commit"] == "test-commit"


def test_run_order_generates_versioned_scatter_proposal(
    workdir: Path,
) -> None:
    table = workdir / "scatter.csv"
    table.write_text(
        "Run order,Value\n1,4.0\n2,3.5\n3,4.2\n",
        encoding="utf-8",
    )

    proposed, rejected, summary = run_one(
        workdir,
        candidate_for(table, candidate_id="scatter-a"),
    )

    assert rejected == []
    assert summary["single_proposals"] == 1
    proposal = proposed[0]
    assert proposal["proposal_rule_version"] == "simple-2d-v2"
    assert proposal["experiment_case"]["chart_family"] == "scatter"
    assert proposal["experiment_case"]["intent"]["x"] == "Run order"
    assert proposal["experiment_case"]["evaluation_expectation"]["panels"][0][
        "series"
    ][0]["kind"] == "scatter"


@pytest.mark.parametrize(
    "header",
    ["Run order", "run_order", "RUN-ORDER", "Run.order", "RunOrder"],
)
def test_explicit_run_order_wins_over_other_monotonic_columns(
    workdir: Path,
    header: str,
) -> None:
    table = workdir / "scatter-monotonic.csv"
    table.write_text(
        f"{header},Value\n1,10\n2,20\n3,30\n",
        encoding="utf-8",
    )

    proposed, rejected, _ = run_one(
        workdir,
        candidate_for(table, candidate_id="scatter-monotonic"),
    )

    assert rejected == []
    assert proposed[0]["experiment_case"]["chart_family"] == "scatter"


def test_run_order_ambiguity_and_invalid_values_fail_closed(
    workdir: Path,
) -> None:
    ambiguous = workdir / "ambiguous-run-order.csv"
    ambiguous.write_text(
        "Run order,Run_order,Value\n1,1,3\n2,2,1\n3,3,2\n",
        encoding="utf-8",
    )
    invalid = workdir / "invalid-run-order.csv"
    invalid.write_text(
        "Run order,Value\n1,3\n3,1\n2,2\n",
        encoding="utf-8",
    )
    source = workdir / "candidates.jsonl"
    write_candidates(
        source,
        [
            candidate_for(ambiguous, candidate_id="ambiguous-run-order"),
            candidate_for(invalid, candidate_id="invalid-run-order"),
        ],
    )

    proposed, rejected, _ = propose_cases(
        candidates_path=source,
        code_commit="test-commit",
    )

    assert proposed == []
    reasons = {
        item["candidate_id"]: item["proposal_rejection_reasons"]
        for item in rejected
    }
    assert reasons["ambiguous-run-order"] == [
        "x-column-ambiguous-run-order"
    ]
    assert reasons["invalid-run-order"] == [
        "explicit-run-order-column-invalid"
    ]


def test_explicit_v1_rule_preserves_run_order_line_behavior(
    workdir: Path,
) -> None:
    table = workdir / "legacy-run-order.csv"
    table.write_text(
        "Run order,Value\n1,3\n2,1\n3,2\n",
        encoding="utf-8",
    )
    proposal = propose_single_candidate(
        candidate_for(table, candidate_id="legacy-run-order"),
        input_candidates_sha256="f" * 64,
        code_commit="a" * 40,
        max_file_bytes=DEFAULT_MAX_FILE_BYTES,
        max_rows=DEFAULT_MAX_ROWS,
        max_columns=DEFAULT_MAX_COLUMNS,
        rule_version=PROPOSAL_RULE_V1,
    )

    assert proposal["proposal_rule_version"] == PROPOSAL_RULE_V1
    assert proposal["experiment_case"]["chart_family"] == "line"


def test_unique_categorical_generates_bar_and_does_not_guess_units(
    workdir: Path,
) -> None:
    table = workdir / "bar.csv"
    table.write_text(
        "Category,Temperature\nA,20\nB,21\nC,22\n",
        encoding="utf-8",
    )
    proposed, rejected, _ = run_one(
        workdir,
        candidate_for(table, candidate_id="bar-a"),
    )
    assert rejected == []
    case = proposed[0]["experiment_case"]
    assert case["chart_family"] == "bar"
    assert case["intent"]["x"] == "Category"
    assert case["intent"]["y"] == "Temperature"
    assert case["intent"]["series"] == ["Temperature"]
    assert case["intent"]["units"] == {}


def test_unique_monotonic_numeric_generates_line(workdir: Path) -> None:
    table = workdir / "monotonic.csv"
    table.write_text(
        "Distance,Response\n0,3\n1,1\n2,2\n",
        encoding="utf-8",
    )
    proposed, rejected, _ = run_one(
        workdir,
        candidate_for(table, candidate_id="numeric-a"),
    )
    assert rejected == []
    assert proposed[0]["experiment_case"]["chart_family"] == "line"
    assert proposed[0]["experiment_case"]["intent"]["x"] == "Distance"


def test_ambiguous_x_and_more_than_four_y_are_rejected(workdir: Path) -> None:
    ambiguous = workdir / "ambiguous.csv"
    ambiguous.write_text("a,b,c\n1,2,3\n2,3,2\n3,4,1\n", encoding="utf-8")
    too_many = workdir / "too-many.csv"
    too_many.write_text(
        "Time,y1,y2,y3,y4,y5\n0,1,2,3,4,5\n1,2,3,4,5,6\n",
        encoding="utf-8",
    )
    source = workdir / "candidates.jsonl"
    write_candidates(
        source,
        [
            candidate_for(ambiguous, candidate_id="ambiguous-a"),
            candidate_for(too_many, candidate_id="too-many-a", panel_id="b"),
        ],
    )
    proposed, rejected, _ = propose_cases(
        candidates_path=source,
        code_commit="test-commit",
    )
    assert proposed == []
    reasons = {
        item["candidate_id"]: item["proposal_rejection_reasons"]
        for item in rejected
    }
    assert reasons["ambiguous-a"] == ["x-column-ambiguous-monotonic-numeric"]
    assert reasons["too-many-a"] == ["too-many-y-columns"]


def test_partial_nan_is_allowed_but_empty_column_is_rejected(workdir: Path) -> None:
    partial = workdir / "partial.csv"
    partial.write_text(
        "Category,Value\nA,1\nB,\nC,3\n",
        encoding="utf-8",
    )
    empty = workdir / "empty.csv"
    empty.write_text(
        "Category,Value\nA,\nB,\n",
        encoding="utf-8",
    )
    source = workdir / "candidates.jsonl"
    write_candidates(
        source,
        [
            candidate_for(partial, candidate_id="partial-a"),
            candidate_for(empty, candidate_id="empty-a", panel_id="b"),
        ],
    )
    proposed, rejected, _ = propose_cases(
        candidates_path=source,
        code_commit="test-commit",
    )
    assert [item["candidate_id"] for item in proposed] == ["partial-a"]
    assert proposed[0]["proposal_analysis"]["non_null_y"]["Value"] == 2
    assert rejected[0]["proposal_rejection_reasons"] == ["table-empty-column"]


def test_xlsx_reads_only_specified_sheet_and_records_it(workdir: Path) -> None:
    table = workdir / "generic.xlsx"
    write_xlsx(
        table,
        "Figure 3C",
        [
            ["Date", "Value"],
            ["2024-01-01", 1.0],
            ["2024-01-02", 2.0],
        ],
    )
    proposed, rejected, _ = run_one(
        workdir,
        candidate_for(
            table,
            candidate_id="sheet-c",
            panel_id="c",
            figure_no=3,
            sheet_name="Figure 3C",
        ),
    )
    assert rejected == []
    case = proposed[0]["experiment_case"]
    assert case["data_path"] == str(table.resolve())
    assert case["sheet"] == "Figure 3C"
    assert case["panels"][0]["sheet"] == "Figure 3C"
    assert proposed[0]["source_table"]["sha256"] == sha256_file(table)


def test_multi_panel_proposal_uses_only_schema_provable_cohesion(
    workdir: Path,
) -> None:
    candidates = []
    for panel_id in ("a", "b"):
        table = workdir / f"panel-{panel_id}.csv"
        table.write_text(
            "Category,Value (kg)\nA,1\nB,2\n",
            encoding="utf-8",
        )
        candidates.append(
            candidate_for(
                table,
                candidate_id=f"panel-{panel_id}",
                panel_id=panel_id,
                figure_no=2,
            )
        )
    source = workdir / "candidates.jsonl"
    write_candidates(source, candidates)
    proposed, rejected, summary = propose_cases(
        candidates_path=source,
        code_commit="test-commit",
    )
    assert rejected == []
    assert summary["single_proposals"] == 2
    assert summary["multi_panel_proposals"] == 1
    assert summary["proposal_rule_version"] == "simple-2d-v2"
    assert all(
        proposal["proposal_rule_version"] == "simple-2d-v2"
        for proposal in proposed
    )
    for proposal in proposed:
        Draft202012Validator(EXPECTATION_SCHEMA).validate(
            proposal["experiment_case"]["evaluation_expectation"]
        )
    multi = next(item for item in proposed if item["proposal_type"] == "multi_panel")
    case = multi["experiment_case"]
    assert case["panel_count"] == 2
    assert [panel["id"] for panel in case["panels"]] == ["a", "b"]
    group = case["evaluation_expectation"]["panel_groups"][0]
    assert group["checks"]["shared_x_scale"] is True
    assert group["checks"]["palette_consistent"] is True
    assert group["series"] == ["Value (kg)"]
    assert "shared_x_unit" not in group["checks"]
    assert "legend" not in json.dumps(group).casefold()
    assert "layout" not in json.dumps(group).casefold()
    assert multi["eligible_for_experiment"] is False


def test_verified_candidate_is_preserved_and_not_overwritten(workdir: Path) -> None:
    table = workdir / "verified.csv"
    table.write_text("Category,Value\nA,1\nB,2\n", encoding="utf-8")
    existing_case = {"case_id": "verified", "user_goal": "human authored"}
    original = candidate_for(
        table,
        candidate_id="verified-a",
        status="verified",
        experiment_case=existing_case,
    )
    source = workdir / "candidates.jsonl"
    write_candidates(source, [original])
    proposed, rejected, summary = propose_cases(
        candidates_path=source,
        code_commit="test-commit",
    )
    assert proposed == []
    assert summary["verified_preserved"] == 1
    assert rejected[0]["curation_status"] == "verified"
    assert rejected[0]["eligible_for_experiment"] is True
    assert rejected[0]["experiment_case"] == existing_case
    assert rejected[0]["proposal_rejection_reasons"] == [
        "verified-candidate-preserved"
    ]


def test_file_and_row_limits_reject_before_unbounded_loading(workdir: Path) -> None:
    table = workdir / "limited.csv"
    table.write_text(
        "Category,Value\nA,1\nB,2\n",
        encoding="utf-8",
    )
    source = workdir / "candidates.jsonl"
    write_candidates(
        source,
        [candidate_for(table, candidate_id="limited-a")],
    )
    proposed, rejected, _ = propose_cases(
        candidates_path=source,
        code_commit="test-commit",
        max_file_bytes=1,
    )
    assert proposed == []
    assert rejected[0]["proposal_rejection_reasons"] == [
        "source-table-file-size-limit"
    ]
    proposed, rejected, _ = propose_cases(
        candidates_path=source,
        code_commit="test-commit",
        max_rows=1,
    )
    assert proposed == []
    assert rejected[0]["proposal_rejection_reasons"] == ["table-row-limit"]


def test_proposal_outputs_are_deterministic_and_include_input_hash(
    workdir: Path,
) -> None:
    table = workdir / "bar.csv"
    table.write_text("Category,Value\nA,1\nB,2\n", encoding="utf-8")
    source = workdir / "candidates.jsonl"
    write_candidates(
        source,
        [candidate_for(table, candidate_id="deterministic-a")],
    )
    output = workdir / "proposals"
    first = propose_cases(candidates_path=source, code_commit="test-commit")
    write_proposal_outputs(output, *first)
    first_bytes = {
        name: (output / name).read_bytes()
        for name in ("proposed.jsonl", "rejected.jsonl", "summary.json")
    }
    second = propose_cases(candidates_path=source, code_commit="test-commit")
    write_proposal_outputs(output, *second)
    assert first == second
    assert first_bytes == {
        name: (output / name).read_bytes()
        for name in ("proposed.jsonl", "rejected.jsonl", "summary.json")
    }
    summary = first[2]
    assert summary["input_candidates_sha256"] == sha256_file(source)
    assert summary["code_commit"] == "test-commit"
    assert summary["eligible_for_experiment"] == 0
