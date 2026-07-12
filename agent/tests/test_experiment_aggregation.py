from __future__ import annotations

import json

import pytest

from experiments.aggregate import AggregationError, aggregate_runs
from experiments.harness import execute_experiment
from tests.test_experiment_support import (
    TestOnlySequenceProvider,
    experiment_workspace,
    make_spec,
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
