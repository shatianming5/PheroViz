from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image
import pytest

import app.services.multi_panel_runner as multi_runner
from app.services.pheromones import ConstraintRecord, Scope
from experiments.harness import execute_experiment
from experiments.models import sha256_path
from experiments.providers import MultiPanelProvider
from tests.test_experiment_support import make_spec, write_manifest


PANEL_IDS = ("left", "right")


def _expectation() -> dict[str, Any]:
    return {
        "schema_version": "1.1.0",
        "panels": [
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
            for panel_id in PANEL_IDS
        ],
        "panel_groups": [
            {
                "group_id": "shared",
                "panels": list(PANEL_IDS),
                "checks": {"shared_y_scale": True},
            }
        ],
    }


def _panel_case(tmp_path: Path) -> dict[str, Any]:
    panels = []
    for offset, panel_id in enumerate(PANEL_IDS):
        data_path = tmp_path / f"{panel_id}.csv"
        data_path.write_text(
            f"category,value\nA,{offset + 1}\nB,{offset + 2}\n",
            encoding="utf-8",
        )
        panels.append(
            {
                "id": panel_id,
                "data_path": str(data_path.resolve()),
                "user_goal": panel_id,
                "chart_family": "bar",
                "intent": {"x": "category", "y": "value"},
            }
        )
    return {
        "case_id": "multi-case",
        "panel_count": len(PANEL_IDS),
        "split": "test",
        "panels": panels,
        "evaluation_expectation": _expectation(),
    }


def _panel_case_with_count(tmp_path: Path, panel_count: int) -> dict[str, Any]:
    panel_ids = [f"panel-{index}" for index in range(1, panel_count + 1)]
    panels = []
    for index, panel_id in enumerate(panel_ids, 1):
        data_path = tmp_path / f"{panel_id}.csv"
        data_path.write_text(
            f"category,value\nA,{index}\nB,{index + 1}\n",
            encoding="utf-8",
        )
        panels.append(
            {
                "id": panel_id,
                "data_path": str(data_path.resolve()),
                "user_goal": panel_id,
                "chart_family": "bar",
                "intent": {"x": "category", "y": "value"},
            }
        )
    return {
        "case_id": f"multi-case-{panel_count}",
        "panel_count": panel_count,
        "split": "test",
        "panels": panels,
        "evaluation_expectation": {
            "schema_version": "1.1.0",
            "panels": [
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
                for panel_id in panel_ids
            ],
            "panel_groups": [
                {
                    "group_id": "shared",
                    "panels": panel_ids,
                    "checks": {"shared_y_scale": True},
                }
            ],
        },
    }


class _CombinedManifest:
    def __init__(self, manifests: Mapping[str, Mapping[str, Any]]) -> None:
        self.manifests = dict(manifests)

    def to_dict(self) -> dict[str, Any]:
        return {
            "panel_rounds": {
                panel_id: manifest["round"]
                for panel_id, manifest in self.manifests.items()
            },
            "panel_scores": {
                panel_id: manifest["score"]
                for panel_id, manifest in self.manifests.items()
            },
        }


class _Cohesion:
    def __init__(self, ratio: float) -> None:
        self.ratio = ratio

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": "cohesion",
            "applicable": True,
            "status": "pass" if self.ratio == 1.0 else "fail",
            "numerator": int(self.ratio * 100),
            "denominator": 100,
            "ratio": self.ratio,
            "checks": {},
            "mismatches": [],
        }


def _install_fake_panel_core(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_iter_chain(*args: Any, **kwargs: Any):
        del args
        run_dir = Path(kwargs["run_dir"]).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        panel_id = str(kwargs["panel_id"])
        rounds = int(kwargs["rounds"])
        memory_mode = str(kwargs["memory_mode"])
        memory = kwargs["memory"]
        untyped_memory = kwargs["untyped_memory"]
        panel_offset = (
            PANEL_IDS.index(panel_id) if panel_id in PANEL_IDS else 0
        )

        for round_number in range(1, rounds + 1):
            if memory_mode in {"constraints", "full", "ephemeral"}:
                memory.add_constraint(
                    ConstraintRecord(
                        id=f"{panel_id}-round-{round_number}",
                        scope=Scope.PANEL,
                        scope_key=panel_id,
                        level=1,
                        slot=f"checkpoint.{panel_id}.{round_number}",
                        value=round_number,
                        hard=False,
                        provenance={"grounding": "executable"},
                        validated_executable=True,
                    )
                )
            elif memory_mode == "untyped":
                untyped_memory.append(
                    {
                        "panel_id": panel_id,
                        "round": round_number,
                    }
                )

            render_path = run_dir / f"figure_round_{round_number}.png"
            Image.new(
                "RGB",
                (12, 8),
                (
                    30 + round_number * 50,
                    40 + panel_offset * 60,
                    90,
                ),
            ).save(render_path)
            (run_dir / f"code_round_{round_number}.py").write_text(
                f"ROUND = {round_number}\n",
                encoding="utf-8",
            )
            (run_dir / f"slots_round_{round_number}.json").write_text(
                json.dumps({"round": round_number}),
                encoding="utf-8",
            )

            score = round_number / rounds
            evaluation = {
                "metric_config": {"metric_version": "test"},
                "figure_manifest": {
                    "panel_id": panel_id,
                    "round": round_number,
                    "score": score,
                },
                "fidelity": {
                    "name": "fidelity",
                    "applicable": True,
                    "status": "pass" if score == 1.0 else "fail",
                    "numerator": round_number,
                    "denominator": rounds,
                    "ratio": score,
                    "checks": {},
                    "mismatches": [],
                },
                "cohesion": {
                    "name": "cohesion",
                    "applicable": False,
                    "status": "na",
                    "numerator": 0,
                    "denominator": 0,
                    "ratio": None,
                    "checks": {},
                    "mismatches": [],
                },
            }
            evaluation_path = (
                run_dir
                / f"programmatic_evaluation_round_{round_number}.json"
            )
            evaluation_path.write_text(
                json.dumps(evaluation, sort_keys=True),
                encoding="utf-8",
            )
            iteration_path = run_dir / f"iteration_{round_number}.json"
            result = {
                "round": round_number,
                "png_path": str(render_path),
                "artifact_path": str(iteration_path),
                "scores": {
                    "visual_form": score,
                    "data_fidelity": score,
                },
                "programmatic_evaluation": evaluation,
                "programmatic_evaluation_path": str(evaluation_path),
                "stages": {
                    "L1": {
                        "response": {"source": "fake-core"},
                        "notes": "",
                        "model_metadata": {
                            "model": f"served-round-{round_number}"
                        },
                    }
                },
                "judge_model_metadata": {
                    "model": f"judge-round-{round_number}"
                },
            }
            iteration_path.write_text(
                json.dumps(result, sort_keys=True),
                encoding="utf-8",
            )
            yield result

    def fake_combine(
        manifests: Mapping[str, Mapping[str, Any]],
    ) -> _CombinedManifest:
        return _CombinedManifest(manifests)

    def fake_cohesion(
        manifest: _CombinedManifest,
        expectation: Mapping[str, Any],
        config: Any,
    ) -> _Cohesion:
        del expectation, config
        return _Cohesion(
            min(
                float(item["score"])
                for item in manifest.manifests.values()
            )
        )

    monkeypatch.setattr(multi_runner, "iter_chain", fake_iter_chain)
    monkeypatch.setattr(
        multi_runner,
        "combine_figure_manifests",
        fake_combine,
    )
    monkeypatch.setattr(
        multi_runner,
        "evaluate_cohesion",
        fake_cohesion,
    )


def _iterative_spec(tmp_path: Path, *, budget_value: int = 6):
    manifest = write_manifest(tmp_path, [_panel_case(tmp_path)])
    spec = make_spec(
        tmp_path,
        run_name="multi-checkpoints",
        schedule="iterative",
        case_id="multi-case",
        panel_count=len(PANEL_IDS),
        split="test",
        budget_value=budget_value,
        selection_metric="data_fidelity",
    )
    return manifest, replace(
        spec,
        method_config={
            "initial_generation": "defaults",
            "memory_mode": "full",
        },
    )


def _assert_candidate_artifacts_are_frozen(
    outcome: Any,
    *,
    run_dir: Path,
) -> None:
    required = {
        "output",
        "render",
        "result",
        "programmatic_evaluation",
        "memory_snapshot",
        "memory_trace",
        "memory_compatibility",
        "schedule_trace",
        "timing",
    }
    paths_by_label = {label: set() for label in required}
    for candidate in outcome.record.candidates:
        candidate_id = candidate["candidate_id"]
        assert required <= set(candidate["artifact_paths"])
        metadata_path = (
            run_dir
            / outcome.record.artifact_paths[f"{candidate_id}.metadata"]
        )
        assert json.loads(metadata_path.read_text(encoding="utf-8")) == candidate
        assert (
            outcome.record.artifact_hashes[f"{candidate_id}.metadata"]
            == sha256_path(metadata_path)
        )
        for label, relative_path in candidate["artifact_paths"].items():
            artifact = (run_dir / relative_path).resolve(strict=True)
            artifact.relative_to(run_dir.resolve())
            expected_hash = sha256_path(artifact)
            assert candidate["artifact_hashes"][label] == expected_hash
            assert (
                outcome.record.artifact_hashes[
                    f"{candidate_id}.{label}"
                ]
                == expected_hash
            )
            if label in paths_by_label:
                paths_by_label[label].add(relative_path)
    candidate_count = len(outcome.record.candidates)
    assert all(
        len(paths) == candidate_count for paths in paths_by_label.values()
    )


def test_runner_writes_one_immutable_checkpoint_per_global_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_panel_core(monkeypatch)
    case = _panel_case(tmp_path)
    output_dir = tmp_path / "multi-output"

    result = multi_runner.run_multi_panel(
        case,
        output_dir=output_dir,
        rounds=3,
        initial_generation="defaults",
        memory_mode="full",
    )

    assert result["render_counts"] == {"left": 3, "right": 3}
    assert len(result["checkpoints"]) == 3
    render_hashes = []
    previous_cumulative_wall_clock = 0.0
    for round_number, checkpoint in enumerate(result["checkpoints"], 1):
        assert checkpoint["global_round"] == round_number
        assert checkpoint["render_count"] == len(PANEL_IDS)
        assert checkpoint["cumulative_render_count"] == (
            round_number * len(PANEL_IDS)
        )
        assert checkpoint["schedule_event_count"] == (
            round_number * len(PANEL_IDS)
        )
        assert (
            checkpoint["cumulative_wall_clock_seconds"]
            > previous_cumulative_wall_clock
        )
        previous_cumulative_wall_clock = checkpoint[
            "cumulative_wall_clock_seconds"
        ]
        schedule = json.loads(
            Path(checkpoint["artifacts"]["schedule_trace"]).read_text(
                encoding="utf-8"
            )
        )
        assert len(schedule) == round_number * len(PANEL_IDS)
        assert {
            item["round"] for item in checkpoint["panels"].values()
        } == {round_number}
        assert (
            checkpoint["programmatic_evaluation"]["panel_fidelity"][
                "left"
            ]["ratio"]
            == pytest.approx(round_number / 3)
        )
        for artifact_path in checkpoint["artifacts"].values():
            artifact = Path(artifact_path).resolve(strict=True)
            artifact.relative_to(output_dir.resolve())
        render_hashes.append(
            hashlib.sha256(
                Path(checkpoint["artifacts"]["render"]).read_bytes()
            ).hexdigest()
        )

    assert len(set(render_hashes)) == 3
    final_checkpoint = result["checkpoints"][-1]
    assert (
        result["programmatic_evaluation"]
        == final_checkpoint["programmatic_evaluation"]
    )
    assert (
        Path(result["combined_figure_path"]).read_bytes()
        == Path(final_checkpoint["artifacts"]["render"]).read_bytes()
    )
    assert (
        Path(result["schedule_trace_path"]).read_bytes()
        == Path(final_checkpoint["artifacts"]["schedule_trace"]).read_bytes()
    )
    assert (
        Path(result["shared_memory_snapshot_path"]).read_bytes()
        == Path(final_checkpoint["artifacts"]["memory_snapshot"]).read_bytes()
    )


@pytest.mark.parametrize("panel_count", [2, 3, 6])
def test_global_round_timing_is_deterministic_for_each_panel_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    panel_count: int,
) -> None:
    _install_fake_panel_core(monkeypatch)
    clock_values = iter((10.0, 11.0, 13.0))
    result = multi_runner.run_multi_panel(
        _panel_case_with_count(tmp_path, panel_count),
        output_dir=tmp_path / f"timed-{panel_count}",
        rounds=2,
        initial_generation="defaults",
        memory_mode="none",
        monotonic=lambda: next(clock_values),
    )

    assert [
        checkpoint["cumulative_render_count"]
        for checkpoint in result["checkpoints"]
    ] == [panel_count, panel_count * 2]
    assert [
        checkpoint["cumulative_wall_clock_seconds"]
        for checkpoint in result["checkpoints"]
    ] == [1.0, 3.0]
    for checkpoint in result["checkpoints"]:
        persisted = json.loads(
            Path(checkpoint["result_path"]).read_text(encoding="utf-8")
        )
        timing = json.loads(
            Path(checkpoint["timing_path"]).read_text(encoding="utf-8")
        )
        assert "cumulative_wall_clock_seconds" not in persisted
        assert (
            timing["cumulative_wall_clock_seconds"]
            == checkpoint["cumulative_wall_clock_seconds"]
        )
        assert timing["checkpoint_sha256"] == hashlib.sha256(
            Path(checkpoint["result_path"]).read_bytes()
        ).hexdigest()
        assert timing["archive_boundary"] == (
            "after_checkpoint_json_archive_before_timing_sidecar"
        )


def test_nonmonotonic_global_clock_fails_after_preserving_prior_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_panel_core(monkeypatch)
    clock_values = iter((10.0, 11.0, 11.0))
    output_dir = tmp_path / "nonmonotonic-time"

    with pytest.raises(RuntimeError, match="strictly increasing"):
        multi_runner.run_multi_panel(
            _panel_case(tmp_path),
            output_dir=output_dir,
            rounds=2,
            initial_generation="defaults",
            memory_mode="none",
            monotonic=lambda: next(clock_values),
        )

    first_timing = json.loads(
        (
            output_dir / "checkpoints/round_0001/checkpoint_timing.json"
        ).read_text(encoding="utf-8")
    )
    assert first_timing["cumulative_wall_clock_seconds"] == 1.0
    assert (
        output_dir / "checkpoints/round_0002/checkpoint.json"
    ).is_file()
    assert not (
        output_dir / "checkpoints/round_0002/checkpoint_timing.json"
    ).exists()


def test_untyped_memory_is_checkpointed_at_each_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_panel_core(monkeypatch)
    result = multi_runner.run_multi_panel(
        _panel_case(tmp_path),
        output_dir=tmp_path / "untyped-output",
        rounds=2,
        initial_generation="defaults",
        memory_mode="untyped",
    )

    for round_number, checkpoint in enumerate(result["checkpoints"], 1):
        untyped = json.loads(
            Path(checkpoint["artifacts"]["untyped_memory"]).read_text(
                encoding="utf-8"
            )
        )
        assert len(untyped) == round_number * len(PANEL_IDS)
        assert checkpoint["memory_mode"] == "untyped"


@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_iterative_provider_batch_exact_fills_scheduler_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rounds: int,
) -> None:
    _install_fake_panel_core(monkeypatch)
    _, spec = _iterative_spec(
        tmp_path,
        budget_value=len(PANEL_IDS) * rounds,
    )
    provider = MultiPanelProvider()
    monkeypatch.setattr(provider, "check_available", lambda: None)

    outcome = execute_experiment(
        spec,
        provider_loader=lambda import_path, options: provider,
    )

    assert outcome.record.status == "completed"
    assert outcome.record.render_count == len(PANEL_IDS) * rounds
    assert len(outcome.record.candidates) == rounds
    assert {
        item["render_count"] for item in outcome.record.candidates
    } == {len(PANEL_IDS)}
    assert [
        item["call_index"] for item in outcome.record.candidates
    ] == [1] * rounds
    assert [
        item["provider_metadata"]["global_round"]
        for item in outcome.record.candidates
    ] == list(range(1, rounds + 1))
    cumulative_times = [
        item["provider_metadata"]["cumulative_wall_clock_seconds"]
        for item in outcome.record.candidates
    ]
    assert all(
        current > previous
        for previous, current in zip([0.0, *cumulative_times], cumulative_times)
    )
    for candidate, cumulative_time in zip(
        outcome.record.candidates,
        cumulative_times,
    ):
        timing = json.loads(
            (
                Path(spec.artifact_root)
                / spec.run_name
                / candidate["artifact_paths"]["timing"]
            ).read_text(encoding="utf-8")
        )
        assert timing["cumulative_wall_clock_seconds"] == cumulative_time
    assert [
        item["metrics"]["data_fidelity"]
        for item in outcome.record.candidates
    ] == pytest.approx(
        [round_number / rounds for round_number in range(1, rounds + 1)]
    )
    assert outcome.record.best_candidate_id == f"candidate_{rounds:04d}"

    run_dir = Path(spec.artifact_root) / spec.run_name
    _assert_candidate_artifacts_are_frozen(outcome, run_dir=run_dir)
    for label in (
        "render",
        "programmatic_evaluation",
        "memory_snapshot",
        "schedule_trace",
    ):
        assert len(
            {
                candidate["artifact_hashes"][label]
                for candidate in outcome.record.candidates
            }
        ) == rounds

    final = outcome.record.candidates[-1]
    final_programmatic = json.loads(
        (
            run_dir
            / final["artifact_paths"]["programmatic_evaluation"]
        ).read_text(encoding="utf-8")
    )
    assert final_programmatic["panel_fidelity"]["right"]["ratio"] == 1.0


@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_best_of_n_exact_fills_with_one_complete_candidate_per_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rounds: int,
) -> None:
    _install_fake_panel_core(monkeypatch)
    write_manifest(tmp_path, [_panel_case(tmp_path)])
    spec = make_spec(
        tmp_path,
        run_name="multi-best-of-n",
        schedule="best_of_n",
        case_id="multi-case",
        panel_count=len(PANEL_IDS),
        split="test",
        budget_value=len(PANEL_IDS) * rounds,
        selection_metric="data_fidelity",
    )
    spec = replace(
        spec,
        method_config={"initial_generation": "defaults"},
    )
    provider = MultiPanelProvider()
    monkeypatch.setattr(provider, "check_available", lambda: None)

    outcome = execute_experiment(
        spec,
        provider_loader=lambda import_path, options: provider,
    )

    assert outcome.record.status == "completed"
    assert outcome.record.render_count == len(PANEL_IDS) * rounds
    assert len(outcome.record.candidates) == rounds
    assert {
        item["render_count"] for item in outcome.record.candidates
    } == {len(PANEL_IDS)}
    assert [
        item["call_index"] for item in outcome.record.candidates
    ] == list(range(1, rounds + 1))
    assert {
        item["provider_metadata"]["global_round"]
        for item in outcome.record.candidates
    } == {1}
    assert {
        item["provider_metadata"]["memory_mode"]
        for item in outcome.record.candidates
    } == {"none"}
    assert len(
        {
            item["provider_metadata"]["seed"]
            for item in outcome.record.candidates
        }
    ) == rounds
    assert {
        item["metrics"]["data_fidelity"]
        for item in outcome.record.candidates
    } == {1.0}

    run_dir = Path(spec.artifact_root) / spec.run_name
    _assert_candidate_artifacts_are_frozen(outcome, run_dir=run_dir)
