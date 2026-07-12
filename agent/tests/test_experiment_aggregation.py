from __future__ import annotations

import json

import pytest

from experiments.aggregate import (
    AggregationError,
    _summary_row,
    aggregate_runs,
    assert_paired_ready,
)
from experiments.harness import execute_experiment
from tests.test_experiment_support import (
    TestOnlySequenceProvider,
    experiment_workspace,
    make_spec,
    write_manifest,
)


def test_aggregation_rejects_legacy_run_without_provenance() -> None:
    with experiment_workspace("legacy") as workspace:
        run_root = workspace / "runs"
        legacy = run_root / "old-timestamp"
        legacy.mkdir(parents=True)
        (legacy / "iterations.json").write_text(
            json.dumps([{"score": 99}]),
            encoding="utf-8",
        )

        with pytest.raises(AggregationError, match="Legacy or untracked"):
            aggregate_runs(run_root)


def test_aggregation_rejects_test_only_run() -> None:
    with experiment_workspace("test-only") as workspace:
        spec = make_spec(
            workspace,
            run_name="test-only-run",
            budget_value=1,
        )
        provider = TestOnlySequenceProvider([0.9])
        outcome = execute_experiment(
            spec,
            provider_loader=lambda import_path, options: provider,
        )
        assert outcome.record.status == "completed"
        assert outcome.record.test_only is True

        with pytest.raises(AggregationError, match="Test-only"):
            aggregate_runs(workspace / "runs")


def test_summary_row_includes_case_pairing_fields() -> None:
    with experiment_workspace("summary-case") as workspace:
        spec = make_spec(
            workspace,
            run_name="summary-case",
            case_id="paper-42",
            panel_count=4,
            split="test",
            budget_value=1,
        )
        provider = TestOnlySequenceProvider([0.6])
        outcome = execute_experiment(
            spec,
            provider_loader=lambda import_path, options: provider,
        )

        row = _summary_row(outcome.record)

        assert row["case_id"] == "paper-42"
        assert row["panel_count"] == 4
        assert row["split"] == "test"


def test_paired_ready_rejects_case_set_mismatch() -> None:
    with experiment_workspace("paired-mismatch") as workspace:
        write_manifest(
            workspace,
            [
                {"case_id": "case-1", "panel_count": 1, "split": "test"},
                {"case_id": "case-2", "panel_count": 2, "split": "test"},
            ],
        )
        records = []
        for method, case_id, panel_count in (
            ("method-a", "case-1", 1),
            ("method-a", "case-2", 2),
            ("method-b", "case-1", 1),
        ):
            spec = make_spec(
                workspace,
                run_name=f"{method}-{case_id}",
                method=method,
                schedule="iterative",
                case_id=case_id,
                panel_count=panel_count,
                split="test",
                budget_value=1,
            )
            provider = TestOnlySequenceProvider([0.5])
            records.append(
                execute_experiment(
                    spec,
                    provider_loader=lambda import_path, options, p=provider: p,
                ).record
            )

        with pytest.raises(AggregationError, match="Paired case mismatch"):
            assert_paired_ready(
                records,
                methods=["method-a", "method-b"],
                backbone="test-backbone",
                seed=7,
                budget_type="renders",
                budget_value=1,
            )
