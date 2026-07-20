from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import validate

from experiments.harness import execute_experiment
from experiments.models import sha256_path
from experiments.providers import (
    CandidateResult,
    GenerationRequest,
    PHEROVIZ_PROVIDER_IMPORT_PATH,
    PheroVizProvider,
    ProviderBatch,
    ProviderExecutionError,
    UnifiedBenchmarkProvider,
    UnifiedPheroVizProvider,
    load_provider,
)
from tests.test_experiment_support import make_spec, write_manifest


PANEL_COUNTS = (1, 2, 3, 6)
TIMING_SCHEMA = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "experiments/schemas/trajectory_candidate_metadata.schema.json"
    ).read_text(encoding="utf-8")
)


def _write_mixed_manifest(tmp_path: Path) -> Path:
    cases = []
    for panel_count in PANEL_COUNTS:
        panels = []
        for panel_index in range(panel_count):
            data_path = (
                tmp_path
                / f"case-p{panel_count}-panel-{panel_index + 1}.csv"
            )
            data_path.write_text(
                f"x,y\n0,{panel_index + 1}\n",
                encoding="utf-8",
            )
            panels.append(
                {
                    "id": f"panel-{panel_index + 1}",
                    "data_path": str(data_path.resolve()),
                }
            )
        case = {
            "case_id": f"case-p{panel_count}",
            "panel_count": panel_count,
            "split": "test",
        }
        if panel_count == 1:
            case["data_path"] = panels[0]["data_path"]
        else:
            case["panels"] = panels
        cases.append(case)
    return write_manifest(tmp_path, cases)


def _spec(
    tmp_path: Path,
    *,
    panel_count: int,
    schedule: str,
    budget_value: int,
):
    manifest = _write_mixed_manifest(tmp_path)
    spec = make_spec(
        tmp_path,
        run_name=f"unified-{schedule}-p{panel_count}",
        schedule=schedule,
        case_id=f"case-p{panel_count}",
        panel_count=panel_count,
        split="test",
        budget_value=budget_value,
        selection_metric="score",
        provider=PHEROVIZ_PROVIDER_IMPORT_PATH,
    )
    return manifest, replace(
        spec,
        method_config={
            "initial_generation": "defaults",
            "memory_mode": "full",
        },
    )


class _RecordingDelegate:
    test_only = False

    def __init__(
        self,
        name: str,
        *,
        fail_message: str | None = None,
    ) -> None:
        self.name = name
        self.fail_message = fail_message
        self.requests: list[GenerationRequest] = []
        self.availability_checks = 0

    def check_available(self) -> None:
        self.availability_checks += 1

    def _candidate(
        self,
        request: GenerationRequest,
        *,
        candidate_index: int,
    ) -> CandidateResult:
        panel_count = int(request.spec.panel_count or 0)
        candidate_dir = (
            request.output_dir
            / f"{self.name}_candidate_{candidate_index:04d}"
        )
        candidate_dir.mkdir(parents=True, exist_ok=False)
        payload = {
            "delegate": self.name,
            "call_index": request.call_index,
            "candidate_index": candidate_index,
            "panel_count": panel_count,
        }
        artifacts = {"output": str(candidate_dir)}
        for label, suffix in (
            ("render", "png"),
            ("programmatic_evaluation", "json"),
            ("memory_snapshot", "json"),
            ("schedule_trace", "json"),
        ):
            path = candidate_dir / f"{label}.{suffix}"
            if suffix == "png":
                path.write_bytes(
                    json.dumps(payload, sort_keys=True).encode("utf-8")
                )
            else:
                path.write_text(
                    json.dumps(
                        {**payload, "artifact": label},
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
            artifacts[label] = str(path)
        return CandidateResult(
            metrics={
                "score": float(
                    request.call_index * 100 + candidate_index
                )
            },
            render_count=panel_count,
            artifacts=artifacts,
            metadata={
                "delegate_marker": self.name,
                "delegate_candidate_index": candidate_index,
                "cumulative_render_count": (
                    panel_count * candidate_index
                    if request.spec.schedule == "iterative"
                    else panel_count
                ),
                "cumulative_wall_clock_seconds": float(candidate_index),
            },
        )

    def generate(
        self,
        request: GenerationRequest,
    ) -> CandidateResult | ProviderBatch:
        self.requests.append(request)
        if self.fail_message is not None:
            raise ProviderExecutionError(self.fail_message)
        panel_count = int(request.spec.panel_count or 0)
        remaining = request.remaining_renders
        if remaining is None:
            raise AssertionError("Tests require a render budget")
        if remaining < panel_count or remaining % panel_count:
            raise ProviderExecutionError(
                "render budget is not divisible by routed panel_count"
            )
        if request.spec.schedule == "best_of_n":
            candidate = self._candidate(
                request,
                candidate_index=1,
            )
            if panel_count == 1:
                return ProviderBatch(candidates=(candidate,))
            return candidate

        rounds = remaining // panel_count
        return ProviderBatch(
            candidates=tuple(
                self._candidate(
                    request,
                    candidate_index=round_number,
                )
                for round_number in range(1, rounds + 1)
            ),
            stop=True,
        )


def _provider(
    *,
    single_failure: str | None = None,
    multi_failure: str | None = None,
) -> tuple[PheroVizProvider, _RecordingDelegate, _RecordingDelegate]:
    clock_value = 100.0

    def monotonic() -> float:
        nonlocal clock_value
        clock_value += 1.0
        return clock_value

    provider = PheroVizProvider(monotonic=monotonic)
    single = _RecordingDelegate(
        "fake_single",
        fail_message=single_failure,
    )
    multi = _RecordingDelegate(
        "fake_multi",
        fail_message=multi_failure,
    )
    provider.single_provider = single
    provider.multi_provider = multi
    return provider, single, multi


def _assert_archived_artifacts(outcome: Any, run_dir: Path) -> None:
    required = {
        "output",
        "render",
        "programmatic_evaluation",
        "memory_snapshot",
        "schedule_trace",
    }
    for candidate in outcome.record.candidates:
        candidate_id = candidate["candidate_id"]
        metadata_path = (
            run_dir
            / outcome.record.artifact_paths[f"{candidate_id}.metadata"]
        )
        assert json.loads(metadata_path.read_text(encoding="utf-8")) == candidate
        assert (
            outcome.record.artifact_hashes[f"{candidate_id}.metadata"]
            == sha256_path(metadata_path)
        )
        assert required <= set(candidate["artifact_paths"])
        for label, relative_path in candidate["artifact_paths"].items():
            artifact = (run_dir / relative_path).resolve(strict=True)
            artifact.relative_to(run_dir.resolve())
            assert (
                candidate["artifact_hashes"][label]
                == sha256_path(artifact)
            )


def test_canonical_unified_provider_import_path() -> None:
    provider = load_provider(PHEROVIZ_PROVIDER_IMPORT_PATH, {})

    assert PHEROVIZ_PROVIDER_IMPORT_PATH == (
        "experiments.providers:UnifiedBenchmarkProvider"
    )
    assert isinstance(provider, UnifiedBenchmarkProvider)
    assert isinstance(provider, PheroVizProvider)
    assert UnifiedPheroVizProvider is PheroVizProvider


def test_offline_defaults_provider_skips_model_credential_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "MODEL_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    provider = UnifiedBenchmarkProvider(offline_defaults=True)

    provider.check_available()

    assert provider.test_only is True
    assert provider.single_provider.test_only is True
    assert provider.multi_provider.test_only is True


def test_provider_preflight_uses_matrix_backbone_without_llm_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_API_BASE", "https://models.example.test")
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    PheroVizProvider().check_available()


@pytest.mark.parametrize("panel_count", PANEL_COUNTS)
@pytest.mark.parametrize("schedule", ["iterative", "best_of_n"])
def test_unified_provider_routes_mixed_cases_with_exact_render_accounting(
    tmp_path: Path,
    panel_count: int,
    schedule: str,
) -> None:
    rounds = 2
    _, spec = _spec(
        tmp_path,
        panel_count=panel_count,
        schedule=schedule,
        budget_value=panel_count * rounds,
    )
    provider, single, multi = _provider()

    outcome = execute_experiment(
        spec,
        provider_loader=lambda import_path, options: provider,
    )

    assert outcome.record.status == "completed"
    assert outcome.record.provider_name == "phero_viz"
    assert outcome.record.render_count == panel_count * rounds
    assert len(outcome.record.candidates) == rounds
    assert {
        candidate["render_count"]
        for candidate in outcome.record.candidates
    } == {panel_count}
    routed = single if panel_count == 1 else multi
    unused = multi if panel_count == 1 else single
    assert len(routed.requests) == (1 if schedule == "iterative" else rounds)
    assert unused.requests == []
    assert [
        candidate["call_index"]
        for candidate in outcome.record.candidates
    ] == (
        [1] * rounds
        if schedule == "iterative"
        else list(range(1, rounds + 1))
    )
    assert [
        candidate["provider_metadata"]["cumulative_render_count"]
        for candidate in outcome.record.candidates
    ] == [panel_count, panel_count * 2]
    assert [
        candidate["provider_metadata"]["cumulative_wall_clock_seconds"]
        for candidate in outcome.record.candidates
    ] == [1.0, 2.0]
    for candidate in outcome.record.candidates:
        metadata = candidate["provider_metadata"]
        validate(instance=metadata, schema=TIMING_SCHEMA)
        assert metadata["delegate_marker"] == routed.name
        assert metadata["phero_viz_provider"] == {
            "router": "phero_viz",
            "delegate": routed.name,
            "panel_count": panel_count,
        }
        if schedule == "iterative":
            assert "provider_call_cumulative_render_count" not in metadata
            assert "provider_call_cumulative_wall_clock_seconds" not in metadata
        else:
            assert (
                metadata["provider_call_cumulative_render_count"]
                == panel_count
            )
            assert (
                metadata["provider_call_cumulative_wall_clock_seconds"]
                == 1.0
            )

    run_dir = Path(spec.artifact_root) / spec.run_name
    _assert_archived_artifacts(outcome, run_dir)


@pytest.mark.parametrize("mutation", ["missing", "nonmonotonic"])
def test_unified_provider_rejects_invalid_iterative_timing(
    tmp_path: Path,
    mutation: str,
) -> None:
    class BadTimingDelegate(_RecordingDelegate):
        def _candidate(
            self,
            request: GenerationRequest,
            *,
            candidate_index: int,
        ) -> CandidateResult:
            candidate = super()._candidate(
                request,
                candidate_index=candidate_index,
            )
            metadata = dict(candidate.metadata)
            if mutation == "missing":
                metadata.pop("cumulative_wall_clock_seconds")
            else:
                metadata["cumulative_wall_clock_seconds"] = 1.0
            return CandidateResult(
                metrics=dict(candidate.metrics),
                render_count=candidate.render_count,
                artifacts=dict(candidate.artifacts),
                metadata=metadata,
            )

    _, spec = _spec(
        tmp_path,
        panel_count=2,
        schedule="iterative",
        budget_value=4,
    )
    provider, single, _ = _provider()
    provider.multi_provider = BadTimingDelegate("bad_multi")

    outcome = execute_experiment(
        spec,
        provider_loader=lambda import_path, options: provider,
    )

    assert outcome.record.status == "failed"
    assert outcome.record.error is not None
    assert outcome.record.error["type"] == "ProviderExecutionError"
    assert "trajectory metadata" in outcome.record.error["message"]
    assert single.requests == []


def test_schedule_timing_resets_for_call_one_retry_on_same_provider(
    tmp_path: Path,
) -> None:
    _, spec = _spec(
        tmp_path,
        panel_count=1,
        schedule="best_of_n",
        budget_value=2,
    )
    provider, single, _ = _provider(single_failure="transient setup failure")

    first = execute_experiment(
        spec,
        provider_loader=lambda import_path, options: provider,
    )
    assert first.record.status == "failed"
    assert provider._schedule_trajectory == {}

    single.fail_message = None
    second = execute_experiment(
        spec,
        resume=True,
        provider_loader=lambda import_path, options: provider,
    )

    assert second.record.status == "completed"
    assert second.record.attempt == 2
    assert [request.call_index for request in single.requests] == [1, 1, 2]
    assert [
        candidate["provider_metadata"]["cumulative_wall_clock_seconds"]
        for candidate in second.record.candidates
    ] == [1.0, 2.0]
    assert provider._schedule_trajectory == {}
    assert "previous_attempt_001" in second.record.artifact_paths


def test_best_of_n_schedule_clock_includes_setup_archive_and_between_call_time(
    tmp_path: Path,
) -> None:
    _, spec = _spec(
        tmp_path,
        panel_count=1,
        schedule="best_of_n",
        budget_value=2,
    )
    clock_values = iter((100.0, 102.5, 106.0))
    provider = PheroVizProvider(monotonic=lambda: next(clock_values))
    single = _RecordingDelegate("fake_single")
    multi = _RecordingDelegate("fake_multi")
    provider.single_provider = single
    provider.multi_provider = multi

    outcome = execute_experiment(
        spec,
        provider_loader=lambda import_path, options: provider,
    )

    assert outcome.record.status == "completed"
    assert [
        candidate["provider_metadata"][
            "provider_call_cumulative_wall_clock_seconds"
        ]
        for candidate in outcome.record.candidates
    ] == [1.0, 1.0]
    assert [
        candidate["provider_metadata"]["cumulative_wall_clock_seconds"]
        for candidate in outcome.record.candidates
    ] == [2.5, 6.0]


@pytest.mark.parametrize("panel_count", [2, 3, 6])
@pytest.mark.parametrize("schedule", ["iterative", "best_of_n"])
def test_unified_multi_panel_budget_must_be_divisible(
    tmp_path: Path,
    panel_count: int,
    schedule: str,
) -> None:
    _, spec = _spec(
        tmp_path,
        panel_count=panel_count,
        schedule=schedule,
        budget_value=panel_count * 2 + 1,
    )
    provider, single, multi = _provider()

    outcome = execute_experiment(
        spec,
        provider_loader=lambda import_path, options: provider,
    )

    assert outcome.record.status == "failed"
    assert outcome.record.render_count == 0
    assert outcome.record.candidates == []
    assert outcome.record.error is not None
    assert outcome.record.error["type"] == "ProviderExecutionError"
    assert "not divisible" in outcome.record.error["message"]
    assert len(multi.requests) == 1
    assert single.requests == []


@pytest.mark.parametrize(
    ("panel_count", "single_failure", "multi_failure", "message"),
    [
        (1, "single route failed", None, "single route failed"),
        (3, None, "multi route failed", "multi route failed"),
    ],
)
def test_unified_provider_never_falls_back_to_other_route(
    tmp_path: Path,
    panel_count: int,
    single_failure: str | None,
    multi_failure: str | None,
    message: str,
) -> None:
    _, spec = _spec(
        tmp_path,
        panel_count=panel_count,
        schedule="iterative",
        budget_value=panel_count,
    )
    provider, single, multi = _provider(
        single_failure=single_failure,
        multi_failure=multi_failure,
    )

    outcome = execute_experiment(
        spec,
        provider_loader=lambda import_path, options: provider,
    )

    assert outcome.record.status == "failed"
    assert outcome.record.render_count == 0
    assert outcome.record.candidates == []
    assert outcome.record.error is not None
    assert outcome.record.error["message"] == message
    if panel_count == 1:
        assert len(single.requests) == 1
        assert multi.requests == []
    else:
        assert len(multi.requests) == 1
        assert single.requests == []
