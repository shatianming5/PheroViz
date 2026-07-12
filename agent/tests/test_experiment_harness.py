from __future__ import annotations

from pathlib import Path

from experiments.harness import execute_experiment
from experiments.models import RunRecord
from tests.test_experiment_support import (
    ClockedTestProvider,
    ManualClock,
    TestOnlySequenceProvider,
    UnavailableTestProvider,
    experiment_workspace,
    make_spec,
)


def _loader(provider: object):
    return lambda import_path, options: provider


def test_best_so_far_archive_keeps_highest_measured_candidate() -> None:
    with experiment_workspace("best") as workspace:
        spec = make_spec(workspace, run_name="best-of-three", budget_value=3)
        provider = TestOnlySequenceProvider([0.2, 0.9, 0.4])

        outcome = execute_experiment(
            spec,
            provider_loader=_loader(provider),
        )

        assert outcome.record.status == "completed"
        assert outcome.record.best_candidate_id == "candidate_0002"
        assert outcome.record.metrics == {"score": 0.9}
        assert [
            item["best_candidate_id"] for item in outcome.record.best_history
        ] == ["candidate_0001", "candidate_0002", "candidate_0002"]
        assert (
            Path(spec.artifact_root)
            / spec.run_name
            / "best_so_far.json"
        ).is_file()


def test_resume_skips_completed_run_without_provider_call() -> None:
    with experiment_workspace("resume") as workspace:
        spec = make_spec(workspace, run_name="resume-run", budget_value=2)
        first_provider = TestOnlySequenceProvider([0.3, 0.4])
        first = execute_experiment(
            spec,
            provider_loader=_loader(first_provider),
        )
        assert first.record.status == "completed"

        second_provider = TestOnlySequenceProvider([0.8, 0.9])
        resumed = execute_experiment(
            spec,
            resume=True,
            provider_loader=_loader(second_provider),
        )

        assert resumed.skipped is True
        assert resumed.record.record_hash == first.record.record_hash
        assert second_provider.requests == []


def test_resume_retries_failure_and_preserves_previous_record() -> None:
    with experiment_workspace("resume-failure") as workspace:
        spec = make_spec(
            workspace,
            run_name="resume-failure",
            budget_value=1,
        )
        failed = execute_experiment(
            spec,
            provider_loader=_loader(UnavailableTestProvider()),
        )
        assert failed.record.status == "failed"
        assert failed.record.attempt == 1

        provider = TestOnlySequenceProvider([0.7])
        retried = execute_experiment(
            spec,
            resume=True,
            provider_loader=_loader(provider),
        )

        assert retried.record.status == "completed"
        assert retried.record.attempt == 2
        preserved = (
            Path(spec.artifact_root)
            / spec.run_name
            / "attempt_records"
            / "attempt_001.json"
        )
        assert preserved.is_file()
        assert "previous_attempt_001" in retried.record.artifact_paths


def test_best_of_n_and_iterative_obey_same_render_budget() -> None:
    with experiment_workspace("fair-renders") as workspace:
        best_spec = make_spec(
            workspace,
            run_name="fair-best",
            schedule="best_of_n",
            budget_value=3,
        )
        iterative_spec = make_spec(
            workspace,
            run_name="fair-iterative",
            schedule="iterative",
            budget_value=3,
        )
        best_provider = TestOnlySequenceProvider([0.1, 0.2, 0.3])
        iterative_provider = TestOnlySequenceProvider([0.1, 0.2, 0.3])

        best = execute_experiment(
            best_spec,
            provider_loader=_loader(best_provider),
        )
        iterative = execute_experiment(
            iterative_spec,
            provider_loader=_loader(iterative_provider),
        )

        assert best.record.render_count == iterative.record.render_count == 3
        assert len(best_provider.requests) == len(iterative_provider.requests) == 3
        assert all(request.previous_candidate is None for request in best_provider.requests)
        assert iterative_provider.requests[0].previous_candidate is None
        assert iterative_provider.requests[1].previous_candidate is not None
        assert [len(request.history) for request in iterative_provider.requests] == [
            0,
            1,
            2,
        ]


def test_render_budget_overrun_is_failed_not_accepted() -> None:
    with experiment_workspace("render-overrun") as workspace:
        spec = make_spec(
            workspace,
            run_name="render-overrun",
            schedule="iterative",
            budget_value=1,
        )
        provider = TestOnlySequenceProvider([0.5], render_count=2)

        outcome = execute_experiment(
            spec,
            provider_loader=_loader(provider),
        )

        assert outcome.record.status == "failed"
        assert outcome.record.render_count == 2
        assert outcome.record.error is not None
        assert outcome.record.error["type"] == "BudgetError"


def test_wall_clock_provider_must_stop_within_budget() -> None:
    with experiment_workspace("fair-clock") as workspace:
        clock = ManualClock()
        spec = make_spec(
            workspace,
            run_name="clock-run",
            schedule="iterative",
            budget_type="wall_clock_seconds",
            budget_value=1.0,
        )
        provider = ClockedTestProvider(clock)

        outcome = execute_experiment(
            spec,
            provider_loader=_loader(provider),
            monotonic=clock,
        )

        assert outcome.record.status == "completed"
        assert outcome.record.wall_clock_seconds <= spec.budget_value


def test_wall_clock_overrun_is_failed_not_accepted() -> None:
    with experiment_workspace("clock-overrun") as workspace:
        clock = ManualClock()
        spec = make_spec(
            workspace,
            run_name="clock-overrun",
            schedule="iterative",
            budget_type="wall_clock_seconds",
            budget_value=1.0,
        )
        provider = ClockedTestProvider(clock, overrun=True)

        outcome = execute_experiment(
            spec,
            provider_loader=_loader(provider),
            monotonic=clock,
        )

        assert outcome.record.status == "failed"
        assert outcome.record.error is not None
        assert outcome.record.error["type"] == "BudgetError"


def test_provider_failure_is_written_explicitly() -> None:
    with experiment_workspace("failure") as workspace:
        spec = make_spec(
            workspace,
            run_name="provider-failure",
            budget_value=1,
        )
        outcome = execute_experiment(
            spec,
            provider_loader=_loader(UnavailableTestProvider()),
        )

        record_path = (
            Path(spec.artifact_root)
            / spec.run_name
            / "run_record.json"
        )
        persisted = RunRecord.read(record_path)
        persisted.validate_provenance()
        assert outcome.record.status == persisted.status == "failed"
        assert persisted.error == {
            "type": "ProviderUnavailableError",
            "message": "intentional provider outage",
        }
        assert persisted.test_only is True
        assert persisted.render_count == 0


def test_missing_provider_never_falls_back_to_success() -> None:
    with experiment_workspace("missing-provider") as workspace:
        spec = make_spec(
            workspace,
            run_name="missing-provider",
            budget_value=1,
            provider="",
        )

        outcome = execute_experiment(spec)

        assert outcome.record.status == "failed"
        assert outcome.record.render_count == 0
        assert outcome.record.error is not None
        assert outcome.record.error["type"] == "ProviderUnavailableError"
