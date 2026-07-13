from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest

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
    provider = PheroVizProvider()
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
    for candidate in outcome.record.candidates:
        metadata = candidate["provider_metadata"]
        assert metadata["delegate_marker"] == routed.name
        assert metadata["phero_viz_provider"] == {
            "router": "phero_viz",
            "delegate": routed.name,
            "panel_count": panel_count,
        }

    run_dir = Path(spec.artifact_root) / spec.run_name
    _assert_archived_artifacts(outcome, run_dir)


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
