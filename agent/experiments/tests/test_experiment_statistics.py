from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import pytest

from experiments.cli import main
from experiments.models import SCHEMA_VERSION, sha256_json, write_json_atomic
from experiments.production_statistics import (
    StatisticsError,
    analyze_summary,
    kendall_tau_b,
    load_provenance_summary,
    paired_bootstrap,
    paired_sign_flip_permutation,
)


@contextmanager
def _workspace(label: str) -> Iterator[Path]:
    parent = Path(__file__).resolve().parent / ".statistics_test_work"
    path = parent / f"{label}-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        try:
            parent.rmdir()
        except OSError:
            pass


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _row(
    method: str,
    case_id: str,
    value: float,
    *,
    panel_count: int = 1,
    second_value: float | None = None,
    manifest_hash: str = "a" * 64,
    metric_config_hash: str = "b" * 64,
    seed: int = 7,
    doi: str | None = None,
    status: str = "completed",
    failure_attribution: str = "",
    git_commit: str = "c" * 40,
) -> dict[str, Any]:
    run_name = f"{method}-{case_id}-{seed}"
    execution_success = 1.0 if status == "completed" else 0.0
    return {
        "run_name": run_name,
        "method": method,
        "backbone": "model-a",
        "case_id": case_id,
        "doi": doi or f"10.1234/{case_id}",
        "panel_count": panel_count,
        "split": "test",
        "seed": seed,
        "budget_type": "renders",
        "budget_value": 3.0,
        "dataset_manifest_hash": manifest_hash,
        "git_commit": git_commit,
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:01:00Z",
        "status": status,
        "execution_success": execution_success,
        "failure_attribution": failure_attribution,
        "failure_type": "MethodFailure" if status == "failed" else "",
        "test_only": False,
        "render_count": 3,
        "wall_clock_seconds": 1.0,
        "metric_version": "metrics-v1",
        "metric_config_hash": metric_config_hash,
        "provider": "provider",
        "provider_name": "provider",
        "best_candidate_id": "candidate_0001",
        "spec_hash": _digest(f"spec-{run_name}"),
        "record_hash": _digest(f"record-{run_name}"),
        "metric.data_fidelity": value,
        "metric.execution_success": execution_success,
        **(
            {"metric.second_data_fidelity": second_value}
            if second_value is not None
            else {}
        ),
    }


def _write_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    materialized = [dict(row) for row in rows]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": "2026-01-01T00:02:00Z",
        "source_root": "production-runs",
        "run_count": len(materialized),
        "columns": sorted({key for row in materialized for key in row}),
        "runs": materialized,
        "input_record_hashes": {
            row["run_name"]: row["record_hash"] for row in materialized
        },
    }
    payload["summary_hash"] = sha256_json(payload)
    write_json_atomic(path, payload)
    return path


def test_paired_bootstrap_is_known_and_deterministic() -> None:
    constant = paired_bootstrap([2.0, 2.0, 2.0], seed=41, resamples=250)
    first = paired_bootstrap([1.0, -1.0, 2.0], seed=41, resamples=250)
    second = paired_bootstrap([1.0, -1.0, 2.0], seed=41, resamples=250)

    assert constant["mean_gap"] == 2.0
    assert constant["ci95"] == [2.0, 2.0]
    assert constant["sampling_unit"] == "case_id"
    assert first == second


def test_exact_sign_flip_permutation() -> None:
    result = paired_sign_flip_permutation(
        [1.0, 1.0],
        seed=3,
        exact_max_n=10,
    )

    assert result["mode"] == "exact"
    assert result["permutations"] == 4
    assert result["p_value"] == 0.5
    assert result["alternative"] == "two-sided"


def test_monte_carlo_sign_flip_is_seeded() -> None:
    gaps = [float(index - 8) for index in range(17)]
    first = paired_sign_flip_permutation(
        gaps,
        seed=19,
        monte_carlo_permutations=250,
        exact_max_n=16,
    )
    second = paired_sign_flip_permutation(
        gaps,
        seed=19,
        monte_carlo_permutations=250,
        exact_max_n=16,
    )

    assert first == second
    assert first["mode"] == "monte_carlo"
    assert first["permutations"] == 250


def test_case_mismatch_is_rejected() -> None:
    with _workspace("case-mismatch") as workspace:
        summary_path = _write_summary(
            workspace / "summary.json",
            [
                _row("reference", "case-1", 1.0),
                _row("reference", "case-2", 2.0),
                _row("method", "case-1", 1.5),
            ],
        )
        summary = load_provenance_summary(summary_path)

        with pytest.raises(StatisticsError, match="Paired case mismatch"):
            analyze_summary(
                summary,
                reference="reference",
                methods=["method"],
                metric="metric.data_fidelity",
                bootstrap_resamples=20,
                monte_carlo_permutations=20,
            )


def test_analysis_averages_seeds_and_tasks_before_doi_resampling() -> None:
    with _workspace("doi-clusters") as workspace:
        method_values = {
            ("case-1", 7): 2.0,
            ("case-1", 9): 4.0,
            ("case-2", 7): 6.0,
            ("case-2", 9): 8.0,
            ("case-3", 7): 10.0,
            ("case-3", 9): 14.0,
        }
        dois = {
            "case-1": "10.1234/article-a",
            "case-2": "10.1234/article-a",
            "case-3": "10.1234/article-b",
        }
        rows = []
        for case_id in dois:
            for run_seed in (7, 9):
                rows.extend(
                    [
                        _row(
                            "reference",
                            case_id,
                            0.0,
                            seed=run_seed,
                            doi=dois[case_id],
                        ),
                        _row(
                            "method",
                            case_id,
                            method_values[(case_id, run_seed)],
                            seed=run_seed,
                            doi=dois[case_id],
                        ),
                    ]
                )
        summary = load_provenance_summary(
            _write_summary(workspace / "summary.json", rows)
        )

        analysis = analyze_summary(
            summary,
            reference="reference",
            methods=["method"],
            metric="metric.data_fidelity",
            bootstrap_resamples=30,
            monte_carlo_permutations=30,
        )
        result = analysis["slices"][0]
        overall = result["comparisons"][0]["overall"]

        assert result["seeds"] == [7, 9]
        assert result["case_count"] == 3
        assert result["doi_count"] == 2
        assert overall["n"] == 2
        assert overall["mean_gap"] == 8.5
        assert overall["sampling_unit"] == "doi"
        assert analysis["analysis_config"]["sampling_unit"] == "doi"
        assert result["experiment_git_commit"] == "c" * 40


def test_sparse_optional_json_metrics_are_accepted() -> None:
    with _workspace("sparse-metrics") as workspace:
        rows = []
        for case_id in ("case-1", "case-2"):
            rows.extend(
                [
                    _row("reference", case_id, 1.0),
                    _row("method", case_id, 2.0),
                ]
            )
        rows[0]["metric.optional_judge"] = 0.75
        summary = load_provenance_summary(
            _write_summary(workspace / "summary.json", rows)
        )

        assert "metric.optional_judge" in summary.rows[0]
        assert all(
            "metric.optional_judge" not in row for row in summary.rows[1:]
        )
        analysis = analyze_summary(
            summary,
            reference="reference",
            methods=["method"],
            metric="metric.data_fidelity",
            bootstrap_resamples=20,
            monte_carlo_permutations=20,
        )
        assert analysis["slices"][0]["comparisons"][0]["overall"][
            "mean_gap"
        ] == 1.0


def test_seed_merged_slice_rejects_mixed_experiment_commits() -> None:
    with _workspace("mixed-seed-commits") as workspace:
        rows = []
        for run_seed, commit in ((7, "c" * 40), (9, "d" * 40)):
            for case_id in ("case-1", "case-2"):
                rows.extend(
                    [
                        _row(
                            "reference",
                            case_id,
                            1.0,
                            seed=run_seed,
                            git_commit=commit,
                        ),
                        _row(
                            "method",
                            case_id,
                            2.0,
                            seed=run_seed,
                            git_commit=commit,
                        ),
                    ]
                )
        summary = load_provenance_summary(
            _write_summary(workspace / "summary.json", rows)
        )

        with pytest.raises(StatisticsError, match="mixes code"):
            analyze_summary(
                summary,
                reference="reference",
                methods=["method"],
                metric="metric.data_fidelity",
                bootstrap_resamples=20,
                monte_carlo_permutations=20,
            )


def test_explicit_method_failures_are_zero_outcomes_not_survivors() -> None:
    with _workspace("method-failures") as workspace:
        rows = []
        for case_id in ("case-1", "case-2"):
            rows.extend(
                [
                    _row("reference", case_id, 1.0),
                    _row(
                        "method",
                        case_id,
                        0.0,
                        status="failed",
                        failure_attribution="method",
                    ),
                ]
            )
        summary = load_provenance_summary(
            _write_summary(workspace / "summary.json", rows)
        )

        analysis = analyze_summary(
            summary,
            reference="reference",
            methods=["method"],
            metric="metric.data_fidelity",
            bootstrap_resamples=20,
            monte_carlo_permutations=20,
        )

        overall = analysis["slices"][0]["comparisons"][0]["overall"]
        assert overall["n"] == 2
        assert overall["mean_gap"] == -1.0


def test_infrastructure_failure_and_missing_doi_fail_closed() -> None:
    with _workspace("unsafe-failures") as workspace:
        infrastructure = _row(
            "method",
            "case-1",
            0.0,
            status="failed",
            failure_attribution="infrastructure",
        )
        with pytest.raises(StatisticsError, match="explicit method attribution"):
            load_provenance_summary(
                _write_summary(
                    workspace / "infrastructure.json",
                    [infrastructure],
                )
            )

        missing_rows = []
        for case_id in ("case-1", "case-2"):
            missing_rows.extend(
                [
                    _row("reference", case_id, 1.0),
                    _row("method", case_id, 2.0),
                ]
            )
        missing_rows[0]["doi"] = ""
        missing_summary = load_provenance_summary(
            _write_summary(
                workspace / "missing-doi.json",
                missing_rows,
            )
        )
        with pytest.raises(StatisticsError, match="DOI cluster"):
            analyze_summary(
                missing_summary,
                reference="reference",
                methods=["method"],
                metric="metric.data_fidelity",
                bootstrap_resamples=20,
                monte_carlo_permutations=20,
            )


def test_task_sets_must_match_across_seeds() -> None:
    with _workspace("seed-task-mismatch") as workspace:
        rows = []
        for run_seed, case_ids in ((7, ("case-1", "case-2")), (9, ("case-1",))):
            for case_id in case_ids:
                rows.extend(
                    [
                        _row("reference", case_id, 1.0, seed=run_seed),
                        _row("method", case_id, 2.0, seed=run_seed),
                    ]
                )
        summary = load_provenance_summary(
            _write_summary(workspace / "summary.json", rows)
        )

        with pytest.raises(StatisticsError, match="task sets across seeds"):
            analyze_summary(
                summary,
                reference="reference",
                methods=["method"],
                metric="metric.data_fidelity",
                bootstrap_resamples=20,
                monte_carlo_permutations=20,
            )


def test_panel_strata_use_na_for_empty_stratum() -> None:
    with _workspace("strata") as workspace:
        rows = []
        for case_id, panel_count in (
            ("case-1", 1),
            ("case-3", 3),
            ("case-5", 5),
        ):
            rows.extend(
                [
                    _row(
                        "reference",
                        case_id,
                        1.0,
                        panel_count=panel_count,
                    ),
                    _row(
                        "method",
                        case_id,
                        2.0,
                        panel_count=panel_count,
                    ),
                ]
            )
        summary = load_provenance_summary(
            _write_summary(workspace / "summary.json", rows)
        )

        analysis = analyze_summary(
            summary,
            reference="reference",
            methods=["method"],
            metric="metric.data_fidelity",
            bootstrap_resamples=20,
            monte_carlo_permutations=20,
        )
        strata = analysis["slices"][0]["comparisons"][0]["panel_strata"]

        assert strata["P=1"]["mean_gap"] == 1.0
        assert strata["P=3-4"]["mean_gap"] == 1.0
        assert strata["P=5+"]["mean_gap"] == 1.0
        assert strata["P=2"]["status"] == "NA"
        assert strata["P=2"]["n"] == 0
        assert strata["P=2"]["mean_gap"] is None
        assert strata["P=2"]["ci95"] is None


def test_kendall_tau_b_handles_ties() -> None:
    assert kendall_tau_b([1.0, 1.0, 2.0], [1.0, 2.0, 2.0]) == 0.5


@pytest.mark.parametrize(
    "mutation",
    ["test_only", "missing_case", "duplicate_case"],
)
def test_summary_provenance_fail_closed(mutation: str) -> None:
    with _workspace(f"provenance-{mutation}") as workspace:
        first = _row("reference", "case-1", 1.0)
        second = _row("method", "case-1", 2.0)
        rows = [first, second]
        if mutation == "test_only":
            rows[0]["test_only"] = True
        elif mutation == "missing_case":
            rows[0]["case_id"] = ""
        else:
            duplicate = dict(second)
            duplicate["run_name"] = "method-case-1-duplicate"
            duplicate["record_hash"] = _digest("duplicate-record")
            duplicate["spec_hash"] = _digest("duplicate-spec")
            rows.append(duplicate)
        path = _write_summary(workspace / "summary.json", rows)

        with pytest.raises(StatisticsError):
            load_provenance_summary(path)


def test_nan_and_missing_metric_are_rejected() -> None:
    with _workspace("nan") as workspace:
        nan_path = workspace / "nan.json"
        nan_path.write_text(
            '{"schema_version":"2.0","summary_hash":"'
            + ("a" * 64)
            + '","runs":[NaN]}',
            encoding="utf-8",
        )
        with pytest.raises(StatisticsError, match="Non-finite"):
            load_provenance_summary(nan_path)

        rows = [
            _row("reference", "case-1", 1.0),
            _row("reference", "case-2", 1.0),
            _row("method", "case-1", 2.0),
            _row("method", "case-2", 2.0),
        ]
        for row in rows:
            row.pop("metric.data_fidelity")
        summary = load_provenance_summary(
            _write_summary(workspace / "missing.json", rows)
        )
        with pytest.raises(StatisticsError, match="Metric"):
            analyze_summary(
                summary,
                reference="reference",
                methods=["method"],
                metric="metric.data_fidelity",
                bootstrap_resamples=20,
                monte_carlo_permutations=20,
            )


@pytest.mark.parametrize("provenance_field", ["manifest", "metric_config"])
def test_mixed_manifest_or_metric_config_is_rejected(
    provenance_field: str,
) -> None:
    with _workspace(f"mixed-{provenance_field}") as workspace:
        rows = []
        for case_id in ("case-1", "case-2"):
            rows.append(_row("reference", case_id, 1.0))
            kwargs = (
                {"manifest_hash": "d" * 64}
                if provenance_field == "manifest"
                else {"metric_config_hash": "e" * 64}
            )
            rows.append(_row("method", case_id, 2.0, **kwargs))
        summary = load_provenance_summary(
            _write_summary(workspace / "summary.json", rows)
        )

        with pytest.raises(StatisticsError):
            analyze_summary(
                summary,
                reference="reference",
                methods=["method"],
                metric="metric.data_fidelity",
                bootstrap_resamples=20,
                monte_carlo_permutations=20,
            )


def test_fewer_than_two_pairs_is_rejected() -> None:
    with _workspace("few-pairs") as workspace:
        summary = load_provenance_summary(
            _write_summary(
                workspace / "summary.json",
                [
                    _row("reference", "case-1", 1.0),
                    _row("method", "case-1", 2.0),
                ],
            )
        )

        with pytest.raises(StatisticsError, match="fewer than two"):
            analyze_summary(
                summary,
                reference="reference",
                methods=["method"],
                metric="metric.data_fidelity",
                bootstrap_resamples=20,
                monte_carlo_permutations=20,
            )


def test_cli_writes_rankings_tau_and_provenance(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with _workspace("cli") as workspace:
        rows = []
        scores = {
            "reference": (1.0, 1.0),
            "method-y": (1.0, 2.0),
            "method-z": (2.0, 2.0),
        }
        for method, (primary, second) in scores.items():
            for case_id in ("case-1", "case-2"):
                rows.append(
                    _row(
                        method,
                        case_id,
                        primary,
                        second_value=second,
                    )
                )
        summary_path = _write_summary(workspace / "summary.json", rows)
        output_dir = workspace / "analysis"

        exit_code = main(
            [
                "analyze",
                str(summary_path),
                "--reference",
                "reference",
                "--methods",
                "method-y",
                "method-z",
                "--metric",
                "metric.data_fidelity",
                "--second-judge-metric",
                "metric.second_data_fidelity",
                "--bootstrap-resamples",
                "25",
                "--permutations",
                "25",
                "--out",
                str(output_dir),
            ]
        )

        assert exit_code == 0
        cli_output = json.loads(capsys.readouterr().out)
        analysis_path = Path(cli_output["analysis_json"])
        csv_path = Path(cli_output["analysis_csv"])
        assert analysis_path.is_file()
        assert csv_path.is_file()
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        ranking = analysis["slices"][0]["rankings"]
        assert ranking["kendall_tau_b"] == 0.5
        assert ranking["primary_ranking"]
        assert ranking["second_judge_ranking"]
        assert analysis["input_summary_hash"]
        assert analysis["analysis_config_hash"]
        assert analysis["code_git_commit"]
