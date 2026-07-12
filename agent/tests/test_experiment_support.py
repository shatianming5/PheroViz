from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

from experiments.models import ExperimentSpec, sha256_file, sha256_json
from experiments.providers import (
    CandidateResult,
    GenerationRequest,
    ProviderBatch,
    ProviderUnavailableError,
)


@contextmanager
def experiment_workspace(label: str) -> Iterator[Path]:
    parent = Path(__file__).resolve().parent / ".experiment_test_work"
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


def make_spec(
    workspace: Path,
    *,
    run_name: str,
    schedule: str = "best_of_n",
    budget_type: str = "renders",
    budget_value: float = 3,
    provider: str = "tests.test_experiment_support:TestOnlySequenceProvider",
) -> ExperimentSpec:
    manifest = workspace / "dataset_manifest.json"
    if not manifest.exists():
        manifest.write_text(
            json.dumps({"cases": []}, sort_keys=True),
            encoding="utf-8",
        )
    repo_root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    metric_config = {
        "selection": {
            "metric": "score",
            "direction": "maximize",
        }
    }
    return ExperimentSpec(
        run_name=run_name,
        method=schedule,
        schedule=schedule,
        backbone="test-backbone",
        seed=7,
        budget_type=budget_type,
        budget_value=budget_value,
        dataset_manifest_path=str(manifest),
        dataset_manifest_hash=sha256_file(manifest),
        git_commit=commit,
        git_dirty=False,
        provider=provider,
        artifact_root=str(workspace / "runs"),
        repo_root=str(repo_root),
        metric_config=metric_config,
        metric_config_hash=sha256_json(metric_config),
        metric_version="test-metric-v1",
    )


class TestOnlySequenceProvider:
    __test__ = False
    name = "test_only_sequence"
    test_only = True

    def __init__(
        self,
        scores: Sequence[float],
        *,
        render_count: int = 1,
    ) -> None:
        self.scores = list(scores)
        self.render_count = render_count
        self.requests: list[GenerationRequest] = []

    def check_available(self) -> None:
        return None

    def generate(self, request: GenerationRequest) -> CandidateResult:
        self.requests.append(request)
        if len(self.requests) > len(self.scores):
            raise AssertionError("Test provider received an unexpected extra call")
        score = self.scores[len(self.requests) - 1]
        artifact = request.output_dir / "test_only_output.json"
        artifact.write_text(
            json.dumps(
                {
                    "score": score,
                    "test_only": True,
                    "call_index": request.call_index,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return CandidateResult(
            metrics={"score": score},
            render_count=self.render_count,
            artifacts={"output": str(artifact)},
            metadata={"test_only": True},
            test_only=True,
        )


class UnavailableTestProvider:
    __test__ = False
    name = "unavailable_test_provider"
    test_only = True

    def check_available(self) -> None:
        raise ProviderUnavailableError("intentional provider outage")

    def generate(self, request: GenerationRequest) -> CandidateResult:
        raise AssertionError("Unavailable provider must never generate")


class ManualClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class ClockedTestProvider(TestOnlySequenceProvider):
    __test__ = False

    def __init__(
        self,
        clock: ManualClock,
        *,
        overrun: bool = False,
    ) -> None:
        super().__init__([0.1, 0.2, 0.3, 0.4])
        self.clock = clock
        self.overrun = overrun

    def generate(
        self,
        request: GenerationRequest,
    ) -> CandidateResult | ProviderBatch:
        if self.overrun:
            self.clock.advance(2.0)
        elif request.remaining_seconds is not None:
            self.clock.advance(min(0.4, request.remaining_seconds / 2.0))
        result = super().generate(request)
        stop = bool(
            request.remaining_seconds is not None
            and request.remaining_seconds <= 0.5
        )
        return ProviderBatch(
            candidates=(result,),
            stop=stop,
            test_only=True,
        )
