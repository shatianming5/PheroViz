from __future__ import annotations

from dataclasses import replace
import json

import pytest

from experiments.aggregate import (
    AggregationError,
    _summary_row,
    aggregate_runs,
    assert_paired_ready,
)
from experiments.harness import execute_experiment
from experiments.providers import (
    GenerationRequest,
    ProviderExecutionError,
    ProviderUnavailableError,
)
from tests.test_experiment_support import (
    TestOnlySequenceProvider,
    experiment_workspace,
    make_spec,
    write_manifest,
)


class _ProductionSequenceProvider(TestOnlySequenceProvider):
    test_only = False

    def generate(self, request: GenerationRequest):
        return replace(
            super().generate(request),
            metadata={},
            test_only=False,
        )


class _MethodFailure(ProviderExecutionError):
    failure_attribution = "method"


class _MethodFailureProvider:
    name = "method_failure"
    test_only = False

    def check_available(self) -> None:
        return None

    def generate(self, request: GenerationRequest):
        raise _MethodFailure("generated method could not produce an outcome")


class _InfrastructureFailureProvider:
    name = "infrastructure_failure"
    test_only = False

    def check_available(self) -> None:
        raise ProviderUnavailableError("required service is unavailable")

    def generate(self, request: GenerationRequest):
        raise AssertionError("Unavailable provider must not generate")


class _UnclassifiedFailureProvider:
    name = "unclassified_failure"
    test_only = False

    def check_available(self) -> None:
        return None

    def generate(self, request: GenerationRequest):
        raise ProviderExecutionError("failure has no safe attribution")


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


def test_aggregation_carries_doi_and_scores_explicit_method_failure_zero() -> None:
    with experiment_workspace("method-failure-row") as workspace:
        write_manifest(
            workspace,
            [
                {
                    "case_id": "case-1",
                    "doi": "https://doi.org/10.1234/Article",
                    "panel_count": 1,
                    "split": "test",
                }
            ],
        )
        reference_spec = make_spec(
            workspace,
            run_name="reference",
            method="reference",
            case_id="case-1",
            budget_value=1,
        )
        failed_spec = make_spec(
            workspace,
            run_name="failed-method",
            method="method",
            case_id="case-1",
            budget_value=1,
        )
        execute_experiment(
            reference_spec,
            provider_loader=lambda import_path, options: (
                _ProductionSequenceProvider([0.8])
            ),
        )
        failed = execute_experiment(
            failed_spec,
            provider_loader=lambda import_path, options: _MethodFailureProvider(),
        )
        assert failed.record.error is not None
        assert failed.record.error["attribution"] == "method"

        _, summary_path = aggregate_runs(workspace / "runs")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = {row["method"]: row for row in summary["runs"]}

        assert rows["reference"]["doi"] == "10.1234/article"
        assert rows["reference"]["execution_success"] == 1.0
        assert rows["method"]["status"] == "failed"
        assert rows["method"]["failure_attribution"] == "method"
        assert rows["method"]["execution_success"] == 0.0
        assert rows["method"]["metric.execution_success"] == 0.0
        assert rows["method"]["metric.score"] == 0.0
        assert summary["method_failed_run_count"] == 1


@pytest.mark.parametrize(
    ("provider", "attribution"),
    [
        (_InfrastructureFailureProvider(), "infrastructure"),
        (_UnclassifiedFailureProvider(), "unclassified"),
    ],
)
def test_aggregation_blocks_non_method_failure(
    provider: object,
    attribution: str,
) -> None:
    with experiment_workspace(f"{attribution}-blocker") as workspace:
        write_manifest(
            workspace,
            [
                {
                    "case_id": "case-1",
                    "doi": "10.1234/article",
                    "panel_count": 1,
                    "split": "test",
                }
            ],
        )
        spec = make_spec(
            workspace,
            run_name="infrastructure-failure",
            method="method",
            case_id="case-1",
            budget_value=1,
        )
        outcome = execute_experiment(
            spec,
            provider_loader=lambda import_path, options: provider,
        )
        assert outcome.record.error is not None
        assert outcome.record.error["attribution"] == attribution

        with pytest.raises(AggregationError, match=f"attribution='{attribution}'"):
            aggregate_runs(workspace / "runs")
