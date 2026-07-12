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
from experiments.providers import (
    CandidateResult,
    GenerationRequest,
    MultiPanelProvider,
    ProviderBatch,
)
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
        panel_offset = PANEL_IDS.index(panel_id)

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
    for round_number, checkpoint in enumerate(result["checkpoints"], 1):
        assert checkpoint["global_round"] == round_number
        assert checkpoint["render_count"] == len(PANEL_IDS)
        assert checkpoint["cumulative_render_count"] == (
            round_number * len(PANEL_IDS)
        )
        assert checkpoint["schedule_event_count"] == (
            round_number * len(PANEL_IDS)
        )
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


def test_iterative_provider_batch_archives_exact_round_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_panel_core(monkeypatch)
    _, spec = _iterative_spec(tmp_path)
    provider = MultiPanelProvider()
    monkeypatch.setattr(provider, "check_available", lambda: None)

    outcome = execute_experiment(
        spec,
        provider_loader=lambda import_path, options: provider,
    )

    assert outcome.record.status == "completed"
    assert outcome.record.render_count == 6
    assert [item["render_count"] for item in outcome.record.candidates] == [
        2,
        2,
        2,
    ]
    assert [
        item["call_index"] for item in outcome.record.candidates
    ] == [1, 1, 1]
    assert [
        item["provider_metadata"]["global_round"]
        for item in outcome.record.candidates
    ] == [1, 2, 3]
    assert [
        item["metrics"]["data_fidelity"]
        for item in outcome.record.candidates
    ] == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert outcome.record.best_candidate_id == "candidate_0003"

    run_dir = Path(spec.artifact_root) / spec.run_name
    output_paths = set()
    for candidate in outcome.record.candidates:
        candidate_id = candidate["candidate_id"]
        output_paths.add(candidate["artifact_paths"]["output"])
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
    assert len(output_paths) == 3
    assert len(
        {
            candidate["artifact_hashes"]["schedule_trace"]
            for candidate in outcome.record.candidates
        }
    ) == 3
    assert len(
        {
            candidate["artifact_hashes"]["memory_snapshot"]
            for candidate in outcome.record.candidates
        }
    ) == 3

    final = outcome.record.candidates[-1]
    final_programmatic = json.loads(
        (
            run_dir
            / final["artifact_paths"]["programmatic_evaluation"]
        ).read_text(encoding="utf-8")
    )
    assert final_programmatic["panel_fidelity"]["right"]["ratio"] == 1.0


def test_best_of_n_returns_one_independent_complete_candidate_per_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_panel_core(monkeypatch)
    manifest = write_manifest(tmp_path, [_panel_case(tmp_path)])
    spec = make_spec(
        tmp_path,
        run_name="multi-best-of-n",
        schedule="best_of_n",
        case_id="multi-case",
        panel_count=len(PANEL_IDS),
        split="test",
        budget_value=4,
        selection_metric="data_fidelity",
    )
    provider = MultiPanelProvider()
    candidates = []
    for call_index in (1, 2):
        call_dir = tmp_path / f"call-{call_index}"
        call_dir.mkdir()
        result = provider.generate(
            GenerationRequest(
                spec=spec,
                dataset_manifest_path=manifest,
                output_dir=call_dir,
                call_index=call_index,
                remaining_renders=4 - (call_index - 1) * 2,
                remaining_seconds=None,
                deadline_monotonic=None,
                history=(),
                previous_candidate=None,
            )
        )
        assert isinstance(result, CandidateResult)
        assert not isinstance(result, ProviderBatch)
        assert result.render_count == len(PANEL_IDS)
        assert result.metadata["global_round"] == 1
        assert result.metadata["memory_mode"] == "none"
        candidates.append(result)

    assert candidates[0].metadata["seed"] != candidates[1].metadata["seed"]
    assert candidates[0].artifacts["output"] != candidates[1].artifacts["output"]
