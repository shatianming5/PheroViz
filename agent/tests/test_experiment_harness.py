from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from experiments.aggregate import verify_frozen_manifest
from experiments.harness import execute_experiment
from experiments.models import RunRecord
from experiments.providers import (
    GenerationRequest,
    MultiPanelProvider,
    ProviderBatch,
    ProviderExecutionError,
    SingleChainProvider,
)
from tests.test_experiment_support import (
    ClockedTestProvider,
    ManualClock,
    TestOnlySequenceProvider,
    UnavailableTestProvider,
    experiment_workspace,
    make_spec,
    write_manifest,
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


def test_run_freezes_manifest_for_future_aggregation() -> None:
    with experiment_workspace("frozen-manifest") as workspace:
        spec = make_spec(
            workspace,
            run_name="frozen-manifest",
            budget_value=1,
        )
        provider = TestOnlySequenceProvider([0.5])
        outcome = execute_experiment(
            spec,
            provider_loader=_loader(provider),
        )
        run_dir = Path(spec.artifact_root) / spec.run_name
        frozen = run_dir / outcome.record.artifact_paths["dataset_manifest"]

        assert frozen.is_file()
        assert (
            outcome.record.artifact_hashes["dataset_manifest"]
            == outcome.record.dataset_manifest_hash
        )
        Path(spec.dataset_manifest_path).unlink()
        assert verify_frozen_manifest(outcome.record, run_dir) == frozen


def test_single_chain_selects_case_and_rejects_multi_panel() -> None:
    with experiment_workspace("single-chain-multi") as workspace:
        manifest = write_manifest(
            workspace,
            [
                {"case_id": "single", "panel_count": 1, "split": "test"},
                {"case_id": "multi", "panel_count": 3, "split": "test"},
            ],
        )
        spec = make_spec(
            workspace,
            run_name="single-chain-multi",
            schedule="iterative",
            case_id="multi",
            panel_count=3,
            split="test",
            budget_value=1,
        )
        output_dir = workspace / "provider-output"
        output_dir.mkdir()
        request = GenerationRequest(
            spec=spec,
            dataset_manifest_path=manifest,
            output_dir=output_dir,
            call_index=1,
            remaining_renders=1,
            remaining_seconds=None,
            deadline_monotonic=None,
            history=(),
            previous_candidate=None,
        )

        with pytest.raises(
            ProviderExecutionError,
            match="use a multi-panel provider",
        ):
            SingleChainProvider().generate(request)


def test_single_chain_rejects_source_data_changed_after_manifest() -> None:
    with experiment_workspace("single-chain-data-hash") as workspace:
        data = workspace / "case.csv"
        data.write_text("x,y\n0,1\n", encoding="utf-8")
        manifest = write_manifest(
            workspace,
            [
                {
                    "case_id": "case-001",
                    "panel_count": 1,
                    "split": "test",
                    "data_path": str(data),
                    "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                    "eligible_for_experiment": True,
                }
            ],
        )
        spec = make_spec(
            workspace,
            run_name="single-chain-data-hash",
            budget_value=1,
        )
        output_dir = workspace / "provider-output"
        output_dir.mkdir()
        request = GenerationRequest(
            spec=spec,
            dataset_manifest_path=manifest,
            output_dir=output_dir,
            call_index=1,
            remaining_renders=1,
            remaining_seconds=None,
            deadline_monotonic=None,
            history=(),
            previous_candidate=None,
        )
        data.write_text("x,y\n0,999\n", encoding="utf-8")

        with pytest.raises(ProviderExecutionError, match="SHA-256 changed"):
            SingleChainProvider().generate(request)


def test_eligible_multi_panel_case_rejects_external_manifest_bypass() -> None:
    with experiment_workspace("multi-inline-only") as workspace:
        panels = []
        for panel_id in ("a", "b"):
            data = workspace / f"{panel_id}.csv"
            data.write_text("x,y\n0,1\n", encoding="utf-8")
            panels.append(
                {
                    "id": panel_id,
                    "data_path": str(data),
                    "data_sha256": hashlib.sha256(
                        data.read_bytes()
                    ).hexdigest(),
                    "user_goal": f"Panel {panel_id}",
                    "chart_family": "line",
                    "intent": {"x": "x", "y": "y"},
                }
            )
        external = workspace / "external.json"
        external.write_text(
            json.dumps({"panels": panels}),
            encoding="utf-8",
        )
        manifest = write_manifest(
            workspace,
            [
                {
                    "case_id": "multi-case",
                    "panel_count": 2,
                    "split": "test",
                    "panels": panels,
                    "multi_panel_manifest": str(external),
                    "eligible_for_experiment": True,
                }
            ],
        )
        spec = make_spec(
            workspace,
            run_name="multi-inline-only",
            case_id="multi-case",
            panel_count=2,
            budget_value=2,
        )
        output_dir = workspace / "provider-output"
        output_dir.mkdir()
        request = GenerationRequest(
            spec=spec,
            dataset_manifest_path=manifest,
            output_dir=output_dir,
            call_index=1,
            remaining_renders=2,
            remaining_seconds=None,
            deadline_monotonic=None,
            history=(),
            previous_candidate=None,
        )
        with pytest.raises(
            ProviderExecutionError,
            match="verified inline panels",
        ):
            MultiPanelProvider().generate(request)


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


def test_best_of_n_accounts_for_multi_render_candidates() -> None:
    with experiment_workspace("best-multi-render") as workspace:
        spec = make_spec(
            workspace,
            run_name="best-multi-render",
            schedule="best_of_n",
            budget_value=4,
        )
        provider = TestOnlySequenceProvider([0.2, 0.4], render_count=2)

        outcome = execute_experiment(
            spec,
            provider_loader=_loader(provider),
        )

        assert outcome.record.status == "completed"
        assert outcome.record.render_count == 4
        assert len(outcome.record.candidates) == 2


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


def test_multi_panel_provider_emits_programmatic_candidate(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text("category,value\nA,1\nB,2\n", encoding="utf-8")
    right.write_text("category,value\nA,3\nB,4\n", encoding="utf-8")
    panels = [
        {
            "id": panel_id,
            "data_path": path.name,
            "user_goal": panel_id,
            "chart_family": "bar",
            "intent": {"x": "category", "y": "value"},
        }
        for panel_id, path in (("left", left), ("right", right))
    ]
    expectation_panels = [
        {
            "panel_id": panel_id,
            "axis_index": 0,
            "series": [
                {
                    "series_id": "value",
                    "kind": "bar",
                    "x": "category",
                    "value": "value",
                }
            ],
        }
        for panel_id in ("left", "right")
    ]
    manifest = write_manifest(
        tmp_path,
        [
            {
                "case_id": "multi-case",
                "panel_count": 2,
                "split": "test",
                "panels": panels,
                "evaluation_expectation": {
                    "schema_version": "1.1.0",
                    "panels": expectation_panels,
                    "panel_groups": [
                        {
                            "group_id": "shared",
                            "panels": ["left", "right"],
                            "checks": {"shared_y_scale": True},
                        }
                    ],
                },
            }
        ],
    )
    spec = make_spec(
        tmp_path,
        run_name="multi-provider",
        schedule="iterative",
        case_id="multi-case",
        panel_count=2,
        split="test",
        budget_value=2,
        selection_metric="data_fidelity",
    )
    spec = replace(
        spec,
        method_config={
            "initial_generation": "defaults",
            "memory_mode": "full",
        },
    )
    output_dir = tmp_path / "provider"
    output_dir.mkdir()
    request = GenerationRequest(
        spec=spec,
        dataset_manifest_path=manifest,
        output_dir=output_dir,
        call_index=1,
        remaining_renders=2,
        remaining_seconds=None,
        deadline_monotonic=None,
        history=(),
        previous_candidate=None,
    )

    batch = MultiPanelProvider().generate(request)

    assert isinstance(batch, ProviderBatch)
    candidate = batch.candidates[0]
    assert candidate.render_count == 2
    assert candidate.metrics["data_fidelity"] == 1.0
    assert candidate.metrics["series_cohesion"] == 1.0
    assert Path(candidate.artifacts["render"]).is_file()
    assert Path(candidate.artifacts["programmatic_evaluation"]).is_file()
