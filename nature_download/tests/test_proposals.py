from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from openpyxl import Workbook
import pandas as pd
import pytest

from nature_download.corpus.proposals import (
    BAR_CARDINALITY_REJECTION,
    DEFAULT_MAX_COLUMNS,
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_ROWS,
    MAX_BAR_CATEGORICAL_X,
    PROPOSAL_RULE_V1,
    PROPOSAL_RULE_V2,
    PROPOSAL_RULE_V3,
    PROPOSAL_RULE_V4,
    RENDERABILITY_POLICY_HASH,
    RENDERABILITY_POLICY_ID,
    ProposalRejected,
    analyze_table,
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


def write_bar_csv(
    path: Path,
    categories: int,
    *,
    two_series: bool = False,
) -> None:
    if two_series:
        rows = ["Category,Value A,Value B"]
        split = categories // 2
        rows.extend(
            f"C{index},{index if index < split else ''},"
            f"{index if index >= split else ''}"
            for index in range(categories)
        )
    else:
        rows = ["Category,Value"]
        rows.extend(f"C{index},{index}" for index in range(categories))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


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


@pytest.mark.parametrize("rule_version", [PROPOSAL_RULE_V1, PROPOSAL_RULE_V2])
def test_bar_renderability_boundary_applies_to_every_rule_version(
    workdir: Path,
    rule_version: str,
) -> None:
    accepted_table = workdir / f"bar-200-{rule_version}.csv"
    rejected_table = workdir / f"bar-201-{rule_version}.csv"
    write_bar_csv(accepted_table, MAX_BAR_CATEGORICAL_X)
    write_bar_csv(rejected_table, MAX_BAR_CATEGORICAL_X + 1)

    accepted = propose_single_candidate(
        candidate_for(accepted_table, candidate_id="accepted-boundary"),
        input_candidates_sha256="f" * 64,
        code_commit="a" * 40,
        max_file_bytes=DEFAULT_MAX_FILE_BYTES,
        max_rows=DEFAULT_MAX_ROWS,
        max_columns=DEFAULT_MAX_COLUMNS,
        rule_version=rule_version,
    )

    audit = accepted["renderability_audit"]
    assert accepted["renderability_policy_id"] == RENDERABILITY_POLICY_ID
    assert accepted["renderability_policy_hash"] == RENDERABILITY_POLICY_HASH
    assert audit["decision"] == "accepted"
    assert audit["reason"] == "bar_categorical_x_at_or_below_200"
    assert audit["unique_categorical_x"] == MAX_BAR_CATEGORICAL_X
    assert audit["max_unique_categorical_x_per_panel"] == (
        MAX_BAR_CATEGORICAL_X
    )

    with pytest.raises(ProposalRejected) as error:
        propose_single_candidate(
            candidate_for(rejected_table, candidate_id="rejected-boundary"),
            input_candidates_sha256="f" * 64,
            code_commit="a" * 40,
            max_file_bytes=DEFAULT_MAX_FILE_BYTES,
            max_rows=DEFAULT_MAX_ROWS,
            max_columns=DEFAULT_MAX_COLUMNS,
            rule_version=rule_version,
        )

    assert error.value.reasons == (BAR_CARDINALITY_REJECTION,)
    assert error.value.audit is not None
    assert error.value.audit["decision"] == "rejected"
    assert error.value.audit["unique_categorical_x"] == 201
    assert error.value.audit["policy_hash"] == RENDERABILITY_POLICY_HASH


def test_two_series_bar_counts_union_of_x_categories_not_series_sum(
    workdir: Path,
) -> None:
    table = workdir / "bar-two-series.csv"
    write_bar_csv(table, MAX_BAR_CATEGORICAL_X, two_series=True)

    proposed, rejected, _ = run_one(
        workdir,
        candidate_for(table, candidate_id="bar-two-series"),
    )

    assert rejected == []
    audit = proposed[0]["renderability_audit"]
    assert audit["bar_series_count"] == 2
    assert audit["unique_categorical_x"] == 200
    assert audit["decision"] == "accepted"


def test_line_cardinality_is_uncapped_at_ten_thousand_points(
    workdir: Path,
) -> None:
    table = workdir / "line-10000.csv"
    table.write_text(
        "Time,Signal\n"
        + "".join(f"{index},{index % 17}\n" for index in range(10_000)),
        encoding="utf-8",
    )

    proposed, rejected, _ = run_one(
        workdir,
        candidate_for(table, candidate_id="line-10000"),
    )

    assert rejected == []
    assert proposed[0]["experiment_case"]["chart_family"] == "line"
    audit = proposed[0]["renderability_audit"]
    assert audit["decision"] == "accepted"
    assert audit["reason"] == "non_bar_or_column_no_cardinality_cap"
    assert audit["unique_categorical_x"] is None


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


def test_v3_wide_groups_use_explicit_in_memory_melt_contract(
    workdir: Path,
) -> None:
    table = workdir / "wide-groups.csv"
    table.write_text(
        "WT,KO\n1.0,2.0\n1.1,2.1\n",
        encoding="utf-8",
    )
    proposal = propose_single_candidate(
        candidate_for(table, candidate_id="wide-groups"),
        input_candidates_sha256="f" * 64,
        code_commit="a" * 40,
        max_file_bytes=DEFAULT_MAX_FILE_BYTES,
        max_rows=DEFAULT_MAX_ROWS,
        max_columns=DEFAULT_MAX_COLUMNS,
        rule_version=PROPOSAL_RULE_V3,
    )
    case = proposal["experiment_case"]
    assert case["chart_family"] == "scatter"
    assert case["intent"]["x"] == "__wide_group__"
    assert case["intent"]["series"] == ["__wide_value__"]
    assert case["intent"]["binding_mode"] == "wide_melt"
    assert case["intent"]["wide_melt"]["source_value_columns"] == ["WT", "KO"]
    assert proposal["source_table"]["sha256"] == sha256_file(table)
    Draft202012Validator(EXPECTATION_SCHEMA).validate(
        case["evaluation_expectation"]
    )


def test_propose_cases_accepts_explicit_v3_rule_version(workdir: Path) -> None:
    table = workdir / "wide-v3.csv"
    table.write_text("WT,KO\n1.0,2.0\n1.1,2.1\n", encoding="utf-8")
    candidates = workdir / "candidates-v3.jsonl"
    write_candidates(
        candidates,
        [candidate_for(table, candidate_id="wide-v3")],
    )
    proposed, rejected, summary = propose_cases(
        candidates_path=candidates,
        code_commit="test-commit",
        rule_version=PROPOSAL_RULE_V3,
    )
    assert rejected == []
    assert summary["proposal_rule_version"] == PROPOSAL_RULE_V3
    assert proposed[0]["proposal_rule_version"] == PROPOSAL_RULE_V3


def test_v3_two_measurements_use_correlation_scatter(workdir: Path) -> None:
    table = workdir / "correlation.csv"
    table.write_text(
        "Branchpoint distance,Full length\n1,4\n2,3\n3,5\n",
        encoding="utf-8",
    )
    analysis = analyze_table(
        pd.read_csv(table),
        rule_version=PROPOSAL_RULE_V3,
    )
    assert analysis.chart_family == "scatter"
    assert analysis.x == "Branchpoint distance"
    assert analysis.y == ("Full length",)


def test_v3_excludes_replicate_counter_from_categorical_y(workdir: Path) -> None:
    table = workdir / "replicate-counter.csv"
    table.write_text(
        "Treatment,Replicate,Value\nA,2,3\nA,4,4\nB,1,5\nB,3,6\n",
        encoding="utf-8",
    )
    proposal = propose_single_candidate(
        candidate_for(table, candidate_id="replicate-counter"),
        input_candidates_sha256="f" * 64,
        code_commit="a" * 40,
        max_file_bytes=DEFAULT_MAX_FILE_BYTES,
        max_rows=DEFAULT_MAX_ROWS,
        max_columns=DEFAULT_MAX_COLUMNS,
        rule_version=PROPOSAL_RULE_V3,
    )
    assert proposal["experiment_case"]["intent"]["series"] == ["Value"]
    assert proposal["proposal_analysis"]["dropped_index_columns"] == ["Replicate"]


def test_v3_recognizes_msec_as_a_controlled_x(workdir: Path) -> None:
    table = workdir / "msec.csv"
    table.write_text(
        "msec,Vehicle,Treatment\n0,1,2\n1,2,3\n2,3,4\n",
        encoding="utf-8",
    )
    proposal = propose_single_candidate(
        candidate_for(table, candidate_id="msec"),
        input_candidates_sha256="f" * 64,
        code_commit="a" * 40,
        max_file_bytes=DEFAULT_MAX_FILE_BYTES,
        max_rows=DEFAULT_MAX_ROWS,
        max_columns=DEFAULT_MAX_COLUMNS,
        rule_version=PROPOSAL_RULE_V3,
    )
    assert proposal["experiment_case"]["chart_family"] == "line"
    assert proposal["experiment_case"]["intent"]["x"] == "msec"


def test_v4_large_categorical_groups_become_dot_plot_scatter(
    workdir: Path,
) -> None:
    table = workdir / "large-groups.csv"
    table.write_text(
        "Line,Value\n"
        + "".join(f"A,{value}\n" for value in range(5))
        + "".join(f"B,{value}\n" for value in range(5, 10)),
        encoding="utf-8",
    )
    analysis = analyze_table(pd.read_csv(table), rule_version=PROPOSAL_RULE_V4)
    assert analysis.chart_family == "scatter"
    assert analysis.x == "Line"


def test_v4_mixed_excel_group_labels_do_not_crash_dot_plot_detection() -> None:
    frame = pd.DataFrame(
        {
            "Group": [1] * 5 + [datetime(2024, 1, 1)] * 5,
            "Value": list(range(10)),
        }
    )
    analysis = analyze_table(frame, rule_version=PROPOSAL_RULE_V4)
    assert analysis.chart_family == "scatter"
    assert analysis.x == "Group"


def test_v4_small_categorical_groups_remain_aggregate_bar(
    workdir: Path,
) -> None:
    table = workdir / "small-groups.csv"
    table.write_text(
        "Line,Value\n"
        + "".join(f"A,{value}\n" for value in range(4))
        + "".join(f"B,{value}\n" for value in range(4, 8)),
        encoding="utf-8",
    )
    analysis = analyze_table(pd.read_csv(table), rule_version=PROPOSAL_RULE_V4)
    assert analysis.chart_family == "bar"
    assert analysis.x == "Line"


def test_v4_excludes_noncontiguous_named_integer_identifier(
    workdir: Path,
) -> None:
    table = workdir / "sample-identifier.csv"
    table.write_text(
        "Treatment,Sample ID,Value\n"
        "A,10,3\nA,21,4\nB,31,5\nB,42,6\n",
        encoding="utf-8",
    )
    analysis = analyze_table(pd.read_csv(table), rule_version=PROPOSAL_RULE_V4)
    assert analysis.chart_family == "bar"
    assert analysis.x == "Treatment"
    assert analysis.y == ("Value",)
    assert analysis.dropped_index_columns == ("Sample ID",)


def test_v4_multiseries_group_labels_remain_bar(workdir: Path) -> None:
    table = workdir / "multi-series-groups.csv"
    table.write_text(
        "Genotype,Cell type A,Cell type B,Cell type C\n"
        + "".join(f"WT,{value},{value + 1},{value + 2}\n" for value in range(6))
        + "".join(
            f"Mutant,{value},{value + 1},{value + 2}\n" for value in range(6, 12)
        ),
        encoding="utf-8",
    )
    analysis = analyze_table(pd.read_csv(table), rule_version=PROPOSAL_RULE_V4)
    assert analysis.chart_family == "bar"
    assert analysis.x == "Genotype"


def test_v4_timing_column_precedes_categorical_grouping(workdir: Path) -> None:
    table = workdir / "timing.csv"
    table.write_text(
        "Subject,Treatment,Timing [min],Value\n"
        "1,A,10,3\n2,A,10,4\n1,A,60,5\n2,A,60,6\n"
        "1,B,10,7\n2,B,10,8\n1,B,60,9\n2,B,60,10\n",
        encoding="utf-8",
    )
    analysis = analyze_table(pd.read_csv(table), rule_version=PROPOSAL_RULE_V4)
    assert analysis.chart_family == "line"
    assert analysis.x == "Timing [min]"
    assert analysis.y == ("Value",)
    assert analysis.dropped_index_columns == ("Subject",)


def test_v4_repeated_measured_grid_x_becomes_scatter(workdir: Path) -> None:
    table = workdir / "grid.csv"
    table.write_text(
        "Nutrient,Dispersal,Fraction\n"
        "1,10,0.1\n2,10,0.9\n3,10,0.2\n"
        "1,20,0.6\n2,20,0.3\n3,20,0.8\n"
        "1,30,0.4\n2,30,0.7\n3,30,0.5\n",
        encoding="utf-8",
    )
    analysis = analyze_table(pd.read_csv(table), rule_version=PROPOSAL_RULE_V4)
    assert analysis.chart_family == "scatter"
    assert analysis.x == "Dispersal"


def test_v4_scattering_vector_is_a_controlled_line_x(workdir: Path) -> None:
    table = workdir / "scattering.csv"
    table.write_text(
        "q (Å-1),Observed,Fitted\n"
        "0.00,1.0,1.1\n0.02,0.9,1.0\n0.02,0.8,0.9\n0.04,0.7,0.8\n",
        encoding="utf-8",
    )
    analysis = analyze_table(pd.read_csv(table), rule_version=PROPOSAL_RULE_V4)
    assert analysis.chart_family == "line"
    assert analysis.x == "q (Å-1)"


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
    assert multi["renderability_policy_id"] == RENDERABILITY_POLICY_ID
    assert multi["renderability_policy_hash"] == RENDERABILITY_POLICY_HASH
    assert multi["renderability_audit"]["decision"] == "accepted"
    assert len(multi["renderability_audit"]["panels"]) == 2


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


def test_renderability_rejection_is_deterministic_and_audited(
    workdir: Path,
) -> None:
    table = workdir / "bar-deterministic-201.csv"
    write_bar_csv(table, 201)
    source = workdir / "candidates.jsonl"
    write_candidates(
        source,
        [candidate_for(table, candidate_id="bar-deterministic-201")],
    )

    first = propose_cases(candidates_path=source, code_commit="test-commit")
    second = propose_cases(candidates_path=source, code_commit="test-commit")

    assert first == second
    assert first[0] == []
    rejected = first[1][0]
    assert rejected["proposal_rejection_reasons"] == [
        BAR_CARDINALITY_REJECTION
    ]
    assert rejected["proposal_rejection_detail"] == (
        "unique_categorical_x=201;max=200;"
        f"policy={RENDERABILITY_POLICY_ID}"
    )
    assert rejected["renderability_audit"]["policy_hash"] == (
        RENDERABILITY_POLICY_HASH
    )
    first_hash = hashlib.sha256(
        json.dumps(
            first,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    second_hash = hashlib.sha256(
        json.dumps(
            second,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert first_hash == second_hash
    assert first[2]["renderability_rejected"] == 1
    assert first[2]["renderability_policy_hash"] == RENDERABILITY_POLICY_HASH


def test_multi_is_blocked_when_any_constituent_crosses_bar_cap(
    workdir: Path,
) -> None:
    candidates = []
    for panel_id, categories in (("a", 200), ("b", 200), ("c", 201)):
        table = workdir / f"panel-{panel_id}.csv"
        write_bar_csv(table, categories)
        candidates.append(
            candidate_for(
                table,
                candidate_id=f"panel-{panel_id}",
                panel_id=panel_id,
                figure_no=8,
            )
        )
    source = workdir / "candidates.jsonl"
    write_candidates(source, candidates)

    proposed, rejected, summary = propose_cases(
        candidates_path=source,
        code_commit="test-commit",
    )

    assert [item["candidate_id"] for item in proposed] == [
        "panel-a",
        "panel-b",
    ]
    assert all(item["proposal_type"] == "single_panel" for item in proposed)
    assert all(item["eligible_for_experiment"] is False for item in proposed)
    assert all(
        item["renderability_audit"]["unique_categorical_x"] <= 200
        for item in proposed
    )
    assert rejected[0]["candidate_id"] == "panel-c"
    assert rejected[0]["proposal_rejection_reasons"] == [
        BAR_CARDINALITY_REJECTION
    ]
    assert summary["multi_panel_proposals"] == 0
    assert summary["multi_groups_blocked_by_renderability"] == 1
    assert summary["eligible_for_experiment"] == 0


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
