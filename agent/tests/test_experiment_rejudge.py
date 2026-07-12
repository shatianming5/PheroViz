from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest
from PIL import Image

import experiments.rejudge as rejudge_module
from app.services.model_client import ModelClientError, ModelResponse
from experiments.aggregate import aggregate_runs
from experiments.cli import main
from experiments.harness import execute_experiment
from experiments.models import RunRecord, sha256_file, sha256_json
from experiments.production_statistics import load_provenance_summary
from experiments.providers import CandidateResult, GenerationRequest
from experiments.rejudge import (
    SIDECAR_BATCH_FILENAME,
    VISUAL_FORM_RUBRIC_HASH,
    merge_rejudged_summary,
    rejudge_batch,
)
from tests.test_experiment_support import (
    experiment_workspace,
    make_spec,
)


class ImageProvider:
    name = "sealed_image_provider"
    test_only = False

    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = list(scores)
        self.calls = 0

    def check_available(self) -> None:
        return None

    def generate(self, request: GenerationRequest) -> CandidateResult:
        score = self.scores[self.calls]
        self.calls += 1
        render = request.output_dir / "render.png"
        Image.new(
            "RGB",
            (8, 8),
            color=(int(255 * score), 20, 40),
        ).save(render)
        return CandidateResult(
            metrics={"score": score},
            render_count=1,
            artifacts={"render": str(render)},
            metadata={"call_index": request.call_index},
            test_only=False,
        )


class FakeModelClient:
    def __init__(
        self,
        *,
        score: float = 0.75,
        stop_reason: str = "end_turn",
        error: Exception | None = None,
    ) -> None:
        self.score = score
        self.stop_reason = stop_reason
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def evaluate_image_json(
        self,
        prompt: str,
        image_path: str | Path,
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "image_path": Path(image_path),
                "kwargs": dict(kwargs),
            }
        )
        if self.error is not None:
            raise self.error
        return ModelResponse(
            value={
                "visual_form": self.score,
                "diagnostics": ["Text and layout are legible."],
            },
            model="served-judge-v2",
            request_id="request-123",
            usage={"input_tokens": 10, "output_tokens": 5},
            stop_reason=self.stop_reason,
            latency_seconds=0.01,
        )


def _create_run(
    workspace: Path,
    *,
    run_name: str = "rejudge-run",
    scores: Sequence[float] = (0.2, 0.9),
):
    spec = make_spec(
        workspace,
        run_name=run_name,
        budget_value=len(scores),
    )
    provider = ImageProvider(scores)
    outcome = execute_experiment(
        spec,
        provider_loader=lambda import_path, options: provider,
    )
    assert outcome.record.status == "completed"
    assert outcome.record.test_only is False
    return spec, outcome.record


def _run_snapshot(run_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(run_dir).as_posix(): sha256_file(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
    }


def _sidecar_path(sidecar_dir: Path) -> Path:
    paths = [
        path
        for path in sidecar_dir.glob("*.json")
        if path.name != SIDECAR_BATCH_FILENAME
    ]
    assert len(paths) == 1
    return paths[0]


def test_rejudge_binds_best_render_is_read_only_and_resumes() -> None:
    with experiment_workspace("rejudge-sealed") as workspace:
        spec, record = _create_run(workspace)
        run_root = Path(spec.artifact_root)
        run_dir = run_root / record.run_name
        before = _run_snapshot(run_dir)
        client = FakeModelClient()
        sidecar_dir = workspace / "sidecars"

        first = rejudge_batch(
            run_root,
            judge_model="judge/model-v1",
            output_dir=sidecar_dir,
            model_client=client,
        )

        assert first.exit_code == 0
        assert len(client.calls) == 1
        assert "call_0002" in client.calls[0]["image_path"].as_posix()
        assert client.calls[0]["kwargs"]["model"] == "judge/model-v1"
        payload = json.loads(_sidecar_path(sidecar_dir).read_text(encoding="utf-8"))
        assert payload["run_name"] == record.run_name
        assert payload["record_hash"] == record.record_hash
        assert payload["render_sha256"] == sha256_file(
            client.calls[0]["image_path"]
        )
        assert payload["judge_request_model"] == "judge/model-v1"
        assert payload["judge_served_model"] == "served-judge-v2"
        assert payload["rubric_hash"] == VISUAL_FORM_RUBRIC_HASH
        assert payload["score"] == 0.75
        assert payload["usage"] == {"input_tokens": 10, "output_tokens": 5}
        unhashed = dict(payload)
        sidecar_hash = unhashed.pop("sidecar_hash")
        assert sidecar_hash == sha256_json(unhashed)
        assert _run_snapshot(run_dir) == before

        resumed = rejudge_batch(
            run_root,
            judge_model="judge/model-v1",
            output_dir=sidecar_dir,
            resume=True,
            model_client=client,
        )

        assert resumed.exit_code == 0
        assert resumed.resumed == (record.run_name,)
        assert len(client.calls) == 1
        assert _run_snapshot(run_dir) == before


def test_resume_rejects_tampered_sidecar() -> None:
    with experiment_workspace("rejudge-sidecar-tamper") as workspace:
        spec, _ = _create_run(workspace, scores=(0.8,))
        client = FakeModelClient()
        sidecar_dir = workspace / "sidecars"
        first = rejudge_batch(
            Path(spec.artifact_root),
            judge_model="judge-a",
            output_dir=sidecar_dir,
            model_client=client,
        )
        assert first.exit_code == 0
        sidecar = _sidecar_path(sidecar_dir)
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        payload["score"] = 0.01
        sidecar.write_text(json.dumps(payload), encoding="utf-8")

        resumed = rejudge_batch(
            Path(spec.artifact_root),
            judge_model="judge-a",
            output_dir=sidecar_dir,
            resume=True,
            model_client=client,
        )

        assert resumed.exit_code == 1
        assert "hash mismatch" in resumed.failures[0]["message"]
        assert len(client.calls) == 1


def test_render_tamper_fails_before_model_call() -> None:
    with experiment_workspace("rejudge-render-tamper") as workspace:
        spec, record = _create_run(workspace, scores=(0.8,))
        run_dir = Path(spec.artifact_root) / record.run_name
        best = next(
            item
            for item in record.candidates
            if item["candidate_id"] == record.best_candidate_id
        )
        render = run_dir / best["artifact_paths"]["render"]
        render.write_bytes(render.read_bytes() + b"tamper")
        client = FakeModelClient()

        result = rejudge_batch(
            Path(spec.artifact_root),
            judge_model="judge-a",
            output_dir=workspace / "sidecars",
            model_client=client,
        )

        assert result.exit_code == 1
        assert "integrity check" in result.failures[0]["message"]
        assert client.calls == []
        batch = json.loads(result.batch_path.read_text(encoding="utf-8"))
        assert batch["status"] == "failed"
        assert batch["failures"]


@pytest.mark.parametrize(
    ("client", "message"),
    [
        (
            FakeModelClient(error=ModelClientError("empty vision response")),
            "empty vision response",
        ),
        (
            FakeModelClient(stop_reason="max_tokens"),
            "truncated",
        ),
    ],
)
def test_model_failure_or_truncation_never_falls_back(
    client: FakeModelClient,
    message: str,
) -> None:
    with experiment_workspace("rejudge-model-failure") as workspace:
        spec, _ = _create_run(workspace, scores=(0.8,))
        sidecar_dir = workspace / "sidecars"

        result = rejudge_batch(
            Path(spec.artifact_root),
            judge_model="judge-a",
            output_dir=sidecar_dir,
            model_client=client,
        )

        assert result.exit_code == 1
        assert message in result.failures[0]["message"]
        sidecar = json.loads(_sidecar_path(sidecar_dir).read_text(encoding="utf-8"))
        assert sidecar["status"] == "failed"
        assert sidecar["score"] is None
        assert sidecar["diagnostics"] == []


def test_merge_creates_new_hashed_summary_and_preserves_original() -> None:
    with experiment_workspace("rejudge-merge") as workspace:
        spec, record = _create_run(workspace, scores=(0.8,))
        _, summary_path = aggregate_runs(Path(spec.artifact_root))
        original_bytes = summary_path.read_bytes()
        original = json.loads(original_bytes)
        sidecar_dir = workspace / "sidecars"
        result = rejudge_batch(
            summary_path,
            judge_model="judge/model-v1",
            output_dir=sidecar_dir,
            model_client=FakeModelClient(score=0.61),
        )
        assert result.exit_code == 0

        merged_path, metric = merge_rejudged_summary(
            summary_path,
            sidecar_dir,
            output_path=workspace / "rejudged_summary.json",
        )

        assert summary_path.read_bytes() == original_bytes
        assert metric == "metric.visual_form.judge-model-v1"
        merged = json.loads(merged_path.read_text(encoding="utf-8"))
        assert merged["original_summary_hash"] == original["summary_hash"]
        assert merged["summary_hash"] != original["summary_hash"]
        assert merged["runs"][0][metric] == 0.61
        assert merged["runs"][0]["record_hash"] == record.record_hash
        assert metric in merged["columns"]
        validated = load_provenance_summary(merged_path)
        assert validated.summary_hash == merged["summary_hash"]


def test_cli_rejudge_and_merge_commands(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with experiment_workspace("rejudge-cli") as workspace:
        spec, _ = _create_run(workspace, scores=(0.8,))
        _, summary_path = aggregate_runs(Path(spec.artifact_root))
        sidecar_dir = workspace / "sidecars"
        fake = FakeModelClient()
        monkeypatch.setattr(
            rejudge_module.ModelClient,
            "from_env",
            classmethod(lambda cls, **kwargs: fake),
        )

        rejudge_exit = main(
            [
                "rejudge",
                str(summary_path),
                "--judge-model",
                "judge-cli",
                "--out",
                str(sidecar_dir),
            ]
        )
        rejudge_output = json.loads(capsys.readouterr().out)
        assert rejudge_exit == 0
        assert rejudge_output["exit_code"] == 0

        merged_path = workspace / "cli_rejudged_summary.json"
        merge_exit = main(
            [
                "merge-rejudge",
                str(summary_path),
                str(sidecar_dir),
                "--out",
                str(merged_path),
            ]
        )
        merge_output = json.loads(capsys.readouterr().out)
        assert merge_exit == 0
        assert merge_output["rejudged_summary"] == str(merged_path)
        assert (
            merge_output["second_judge_metric"]
            == "metric.visual_form.judge-cli"
        )


def test_cli_rejudge_returns_nonzero_and_records_batch_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with experiment_workspace("rejudge-cli-failure") as workspace:
        spec, _ = _create_run(workspace, scores=(0.8,))
        sidecar_dir = workspace / "sidecars"
        fake = FakeModelClient(error=ModelClientError("intentional outage"))
        monkeypatch.setattr(
            rejudge_module.ModelClient,
            "from_env",
            classmethod(lambda cls, **kwargs: fake),
        )

        exit_code = main(
            [
                "rejudge",
                str(Path(spec.artifact_root)),
                "--judge-model",
                "judge-cli",
                "--out",
                str(sidecar_dir),
            ]
        )
        output = json.loads(capsys.readouterr().out)

        assert exit_code == 1
        assert output["exit_code"] == 1
        batch = json.loads(
            (sidecar_dir / SIDECAR_BATCH_FILENAME).read_text(encoding="utf-8")
        )
        assert batch["status"] == "failed"
        assert batch["failures"][0]["message"] == "intentional outage"
