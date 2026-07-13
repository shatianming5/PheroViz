from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest
from jsonschema import validate
from PIL import Image

import experiments.rejudge as rejudge_module
from app.services.model_client import ModelClientError, ModelResponse
from experiments.aggregate import aggregate_runs
from experiments.cli import main
from experiments.harness import execute_experiment
from experiments.models import RunRecord, sha256_file, sha256_json
from experiments.production_statistics import load_provenance_summary
from experiments.providers import (
    CandidateResult,
    GenerationRequest,
    ProviderExecutionError,
)
from experiments.rejudge import (
    CodeGitState,
    RejudgeError,
    SIDECAR_BATCH_FILENAME,
    VISUAL_FORM_PROMPT_HASH,
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


class MethodFailure(ProviderExecutionError):
    failure_attribution = "method"


class MethodFailureProvider:
    name = "method_failure_provider"
    test_only = False

    def check_available(self) -> None:
        return None

    def generate(self, request: GenerationRequest) -> CandidateResult:
        raise MethodFailure("method produced no render")


class FakeModelClient:
    def __init__(
        self,
        *,
        score: float = 0.75,
        stop_reason: str = "end_turn",
        error: Exception | None = None,
        served_model: str | None = None,
        on_call: Any = None,
    ) -> None:
        self.score = score
        self.stop_reason = stop_reason
        self.error = error
        self.served_model = served_model
        self.on_call = on_call
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
        if self.on_call is not None:
            self.on_call()
        return ModelResponse(
            value={
                "visual_form": self.score,
                "diagnostics": ["Text and layout are legible."],
            },
            model=self.served_model or str(kwargs["model"]),
            request_id="request-123",
            usage={"input_tokens": 10, "output_tokens": 5},
            stop_reason=self.stop_reason,
            latency_seconds=0.01,
        )


CLEAN_CODE_COMMIT = "d" * 40


@pytest.fixture(autouse=True)
def _clean_rejudge_git_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        rejudge_module,
        "_current_git_state",
        lambda: CodeGitState(commit=CLEAN_CODE_COMMIT, dirty=False),
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


def _create_failed_run(
    workspace: Path,
    *,
    run_name: str = "failed-rejudge-run",
):
    spec = make_spec(
        workspace,
        run_name=run_name,
        method="failed-method",
        budget_value=1,
    )
    outcome = execute_experiment(
        spec,
        provider_loader=lambda import_path, options: MethodFailureProvider(),
    )
    assert outcome.record.status == "failed"
    assert outcome.record.test_only is False
    assert outcome.record.error is not None
    assert outcome.record.error["attribution"] == "method"
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
        _, summary_path = aggregate_runs(
            run_root,
            output_dir=workspace / "summary",
        )
        before = _run_snapshot(run_dir)
        client = FakeModelClient()
        sidecar_dir = workspace / "sidecars"

        first = rejudge_batch(
            summary_path,
            judge_model="claude-sonnet-4.6",
            output_dir=sidecar_dir,
            model_client=client,
        )

        assert first.exit_code == 0
        assert len(client.calls) == 1
        assert "call_0002" in client.calls[0]["image_path"].as_posix()
        assert client.calls[0]["kwargs"]["model"] == "claude-sonnet-4.6"
        assert client.calls[0]["kwargs"]["max_tokens"] == 1024
        payload = json.loads(_sidecar_path(sidecar_dir).read_text(encoding="utf-8"))
        assert payload["run_name"] == record.run_name
        assert payload["record_hash"] == record.record_hash
        assert payload["render_sha256"] == sha256_file(
            client.calls[0]["image_path"]
        )
        assert payload["judge_request_model"] == "claude-sonnet-4.6"
        assert payload["judge_served_model"] == "claude-sonnet-4.6"
        assert payload["rubric_hash"] == VISUAL_FORM_RUBRIC_HASH
        assert payload["code_git_commit"] == CLEAN_CODE_COMMIT
        assert payload["code_git_dirty"] is False
        assert payload["score"] == 0.75
        assert payload["usage"] == {"input_tokens": 10, "output_tokens": 5}
        unhashed = dict(payload)
        sidecar_hash = unhashed.pop("sidecar_hash")
        assert sidecar_hash == sha256_json(unhashed)
        assert _run_snapshot(run_dir) == before
        batch = json.loads(first.batch_path.read_text(encoding="utf-8"))
        assert batch["code_git_commit"] == CLEAN_CODE_COMMIT
        assert batch["code_git_dirty"] is False

        resumed = rejudge_batch(
            summary_path,
            judge_model="claude-sonnet-4.6",
            output_dir=sidecar_dir,
            resume=True,
            model_client=client,
        )

        assert resumed.exit_code == 0
        assert resumed.resumed == (record.run_name,)
        assert len(client.calls) == 1
        assert _run_snapshot(run_dir) == before


def test_resume_rejects_rejudge_code_commit_mismatch() -> None:
    with experiment_workspace("rejudge-commit-mismatch") as workspace:
        spec, _ = _create_run(workspace, scores=(0.8,))
        _, summary_path = aggregate_runs(
            Path(spec.artifact_root),
            output_dir=workspace / "summary",
        )
        sidecar_dir = workspace / "sidecars"
        client = FakeModelClient()
        first = rejudge_batch(
            summary_path,
            judge_model="claude-sonnet-4.6",
            output_dir=sidecar_dir,
            model_client=client,
            git_state=CodeGitState(commit="a" * 40, dirty=False),
        )
        assert first.exit_code == 0

        resumed = rejudge_batch(
            summary_path,
            judge_model="claude-sonnet-4.6",
            output_dir=sidecar_dir,
            resume=True,
            model_client=client,
            git_state=CodeGitState(commit="b" * 40, dirty=False),
        )

        assert resumed.exit_code == 1
        assert "code_git_commit" in resumed.failures[0]["message"]
        assert len(client.calls) == 1


def test_dirty_rejudge_code_is_rejected_by_default() -> None:
    with experiment_workspace("rejudge-dirty") as workspace:
        spec, _ = _create_run(workspace, scores=(0.8,))
        _, summary_path = aggregate_runs(
            Path(spec.artifact_root),
            output_dir=workspace / "summary",
        )
        client = FakeModelClient()

        with pytest.raises(RejudgeError, match="worktree is dirty"):
            rejudge_batch(
                summary_path,
                judge_model="claude-sonnet-4.6",
                output_dir=workspace / "sidecars",
                model_client=client,
                git_state=CodeGitState(commit="a" * 40, dirty=True),
            )

        assert client.calls == []
        assert not (workspace / "sidecars").exists()


def test_resume_rejects_tampered_sidecar() -> None:
    with experiment_workspace("rejudge-sidecar-tamper") as workspace:
        spec, _ = _create_run(workspace, scores=(0.8,))
        _, summary_path = aggregate_runs(
            Path(spec.artifact_root),
            output_dir=workspace / "summary",
        )
        client = FakeModelClient()
        sidecar_dir = workspace / "sidecars"
        first = rejudge_batch(
            summary_path,
            judge_model="claude-sonnet-4.6",
            output_dir=sidecar_dir,
            model_client=client,
        )
        assert first.exit_code == 0
        sidecar = _sidecar_path(sidecar_dir)
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        payload["score"] = 0.01
        sidecar.write_text(json.dumps(payload), encoding="utf-8")

        resumed = rejudge_batch(
            summary_path,
            judge_model="claude-sonnet-4.6",
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
        _, summary_path = aggregate_runs(
            Path(spec.artifact_root),
            output_dir=workspace / "summary",
        )
        run_dir = Path(spec.artifact_root) / record.run_name
        best = next(
            item
            for item in record.candidates
            if item["candidate_id"] == record.best_candidate_id
        )
        render = run_dir / best["artifact_paths"]["render"]
        render.write_bytes(render.read_bytes() + b"tamper")
        client = FakeModelClient()

        with pytest.raises(RejudgeError, match="integrity check"):
            rejudge_batch(
                summary_path,
                judge_model="claude-sonnet-4.6",
                output_dir=workspace / "sidecars",
                model_client=client,
            )
        assert client.calls == []


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
        _, summary_path = aggregate_runs(
            Path(spec.artifact_root),
            output_dir=workspace / "summary",
        )
        sidecar_dir = workspace / "sidecars"

        result = rejudge_batch(
            summary_path,
            judge_model="claude-sonnet-4.6",
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
            judge_model="claude-sonnet-4.6",
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
        assert metric == "metric.visual_form.claude-sonnet-4.6"
        merged = json.loads(merged_path.read_text(encoding="utf-8"))
        assert merged["original_summary_hash"] == original["summary_hash"]
        assert merged["summary_hash"] != original["summary_hash"]
        assert merged["runs"][0][metric] == 0.61
        assert merged["runs"][0]["record_hash"] == record.record_hash
        assert metric in merged["columns"]
        primary = merged["c5_rejudge"]["judges"]["visual-form-primary-v1"]
        assert primary["code_git_commit"] == CLEAN_CODE_COMMIT
        assert primary["code_git_dirty"] is False
        assert primary["batch_hash"] == json.loads(
            result.batch_path.read_text(encoding="utf-8")
        )["batch_hash"]
        validated = load_provenance_summary(merged_path)
        assert validated.summary_hash == merged["summary_hash"]


def test_rejudge_zeros_failed_methods_without_sidecars() -> None:
    with experiment_workspace("rejudge-failed-method") as workspace:
        completed_spec, _ = _create_run(
            workspace,
            run_name="completed-rejudge-run",
            scores=(0.8,),
        )
        _, failed_record = _create_failed_run(workspace)
        _, summary_path = aggregate_runs(Path(completed_spec.artifact_root))
        sidecar_dir = workspace / "sidecars"
        client = FakeModelClient(score=0.61)

        result = rejudge_batch(
            summary_path,
            judge_model="claude-sonnet-4.6",
            output_dir=sidecar_dir,
            model_client=client,
        )

        assert result.exit_code == 0
        assert len(client.calls) == 1
        assert len(
            [
                path
                for path in sidecar_dir.glob("*.json")
                if path.name != SIDECAR_BATCH_FILENAME
            ]
        ) == 1
        merged_path, metric = merge_rejudged_summary(
            summary_path,
            sidecar_dir,
            output_path=workspace / "mixed-rejudged-summary.json",
        )
        merged = json.loads(merged_path.read_text(encoding="utf-8"))
        rows = {row["run_name"]: row for row in merged["runs"]}
        assert rows[failed_record.run_name][metric] == 0.0
        assert merged["c5_rejudge"]["judges"]["visual-form-primary-v1"][
            "failed_zero_runs"
        ] == [
            failed_record.run_name
        ]


def test_rejudge_all_failed_summary_needs_no_render_sidecars() -> None:
    with experiment_workspace("rejudge-all-failed") as workspace:
        spec, failed_record = _create_failed_run(workspace)
        _, summary_path = aggregate_runs(Path(spec.artifact_root))
        sidecar_dir = workspace / "sidecars"
        client = FakeModelClient()

        result = rejudge_batch(
            summary_path,
            judge_model="claude-sonnet-4.6",
            output_dir=sidecar_dir,
            model_client=client,
        )

        assert result.exit_code == 0
        assert result.completed == ()
        assert client.calls == []
        assert sorted(path.name for path in sidecar_dir.glob("*.json")) == [
            SIDECAR_BATCH_FILENAME
        ]
        merged_path, metric = merge_rejudged_summary(
            summary_path,
            sidecar_dir,
            output_path=workspace / "failed-only-rejudged-summary.json",
        )
        merged = json.loads(merged_path.read_text(encoding="utf-8"))
        assert merged["runs"][0]["run_name"] == failed_record.run_name
        assert merged["runs"][0][metric] == 0.0
        assert merged["c5_rejudge"]["judges"]["visual-form-primary-v1"][
            "sidecar_hashes"
        ] == {}


def test_merge_requires_exact_completed_sidecar_coverage() -> None:
    with experiment_workspace("rejudge-sidecar-coverage") as workspace:
        spec, _ = _create_run(workspace, scores=(0.8,))
        _, summary_path = aggregate_runs(Path(spec.artifact_root))
        sidecar_dir = workspace / "sidecars"
        result = rejudge_batch(
            summary_path,
            judge_model="claude-sonnet-4.6",
            output_dir=sidecar_dir,
            model_client=FakeModelClient(),
        )
        assert result.exit_code == 0
        _sidecar_path(sidecar_dir).unlink()
        batch_path = sidecar_dir / SIDECAR_BATCH_FILENAME
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        batch["sidecar_hashes"] = {}
        batch.pop("batch_hash")
        batch["batch_hash"] = sha256_json(batch)
        batch_path.write_text(json.dumps(batch), encoding="utf-8")

        with pytest.raises(RejudgeError, match="input manifest"):
            merge_rejudged_summary(
                summary_path,
                sidecar_dir,
                output_path=workspace / "missing-sidecar-summary.json",
            )


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
            rejudge_module,
            "_model_client_from_config",
            lambda config: fake,
        )

        rejudge_exit = main(
            [
                "rejudge",
                str(summary_path),
                "--judge-model",
                "claude-sonnet-4.6",
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
            merge_output["merged_judge_metric"]
            == "metric.visual_form.claude-sonnet-4.6"
        )


def test_cli_rejudge_returns_nonzero_and_records_batch_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with experiment_workspace("rejudge-cli-failure") as workspace:
        spec, _ = _create_run(workspace, scores=(0.8,))
        _, summary_path = aggregate_runs(
            Path(spec.artifact_root),
            output_dir=workspace / "summary",
        )
        sidecar_dir = workspace / "sidecars"
        fake = FakeModelClient(error=ModelClientError("intentional outage"))
        monkeypatch.setattr(
            rejudge_module,
            "_model_client_from_config",
            lambda config: fake,
        )

        exit_code = main(
            [
                "rejudge",
                str(summary_path),
                "--judge-model",
                "claude-sonnet-4.6",
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


def test_rejudge_binds_frozen_manifest_and_excludes_request_leakage() -> None:
    with experiment_workspace("rejudge-manifest") as workspace:
        spec, record = _create_run(workspace, scores=(0.8,))
        _, summary_path = aggregate_runs(Path(spec.artifact_root))
        client = FakeModelClient()

        result = rejudge_batch(
            summary_path,
            judge_model="claude-sonnet-4.6",
            output_dir=workspace / "sidecars",
            model_client=client,
        )

        assert result.exit_code == 0
        batch = json.loads(result.batch_path.read_text(encoding="utf-8"))
        sidecar = json.loads(
            _sidecar_path(result.output_dir).read_text(encoding="utf-8")
        )
        manifest = batch["input_manifest"]
        assert batch["prompt_hash"] == VISUAL_FORM_PROMPT_HASH
        assert batch["judge_max_tokens"] == 1024
        assert batch["judge_protocol"] == "anthropic_messages"
        assert batch["judge_endpoint_class"] == "anthropic_compatibility_gateway"
        assert manifest["source_summary_hash"] == load_provenance_summary(
            summary_path
        ).summary_hash
        assert manifest["completed_runs"][0]["record_hash"] == record.record_hash
        assert (
            manifest["completed_runs"][0]["render_sha256"]
            == sidecar["render_sha256"]
        )
        assert batch["selected_render_hashes"][record.run_name] == sidecar[
            "render_sha256"
        ]
        assert sidecar["input_manifest_hash"] == batch["input_manifest_hash"]
        prompt = client.calls[0]["prompt"]
        assert record.method not in prompt
        assert str(summary_path) not in prompt
        assert "prior_scores" not in prompt


def test_rejudge_rejects_unregistered_or_mismatched_identity() -> None:
    with experiment_workspace("rejudge-identities") as workspace:
        spec, _ = _create_run(workspace, scores=(0.8,))
        _, summary_path = aggregate_runs(Path(spec.artifact_root))
        client = FakeModelClient()

        with pytest.raises(RejudgeError, match="exactly registered"):
            rejudge_batch(
                summary_path,
                judge_model="judge-not-frozen",
                output_dir=workspace / "unregistered",
                model_client=client,
            )
        assert client.calls == []

        mismatch = FakeModelClient(served_model="claude-sonnet-4.6-latest")
        result = rejudge_batch(
            summary_path,
            judge_model="claude-sonnet-4.6",
            output_dir=workspace / "mismatch",
            model_client=mismatch,
        )
        assert result.exit_code == 1
        assert "Served judge identity mismatch" in result.failures[0]["message"]


def test_judge_client_uses_only_exact_registry_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://frozen-gateway.example")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("MODEL_API_BASE", "https://wrong-precedence.example")
    monkeypatch.setenv("LLM_MAX_TOKENS", "4096")
    config = rejudge_module._load_judge_config("claude-sonnet-4.6")

    client = rejudge_module._model_client_from_config(config)

    assert client.config.base_url == "https://frozen-gateway.example"
    assert client.config.model == "claude-sonnet-4.6"
    assert client.config.max_tokens == 1024
    assert client.config.retries == 2


def test_rejudge_marks_batch_failed_if_source_changes_during_calls() -> None:
    with experiment_workspace("rejudge-stale-source") as workspace:
        spec, _ = _create_run(workspace, scores=(0.8,))
        _, summary_path = aggregate_runs(Path(spec.artifact_root))

        def mutate_summary() -> None:
            summary_path.write_bytes(summary_path.read_bytes() + b"\n")

        result = rejudge_batch(
            summary_path,
            judge_model="claude-sonnet-4.6",
            output_dir=workspace / "sidecars",
            model_client=FakeModelClient(on_call=mutate_summary),
        )

        assert result.exit_code == 1
        assert result.failures[-1]["run_name"] == "__source_summary__"
        batch = json.loads(result.batch_path.read_text(encoding="utf-8"))
        assert batch["status"] == "failed"


def test_merge_rejects_duplicate_sidecar_run_name() -> None:
    with experiment_workspace("rejudge-duplicate-sidecar") as workspace:
        spec, _ = _create_run(workspace, scores=(0.8,))
        _, summary_path = aggregate_runs(Path(spec.artifact_root))
        result = rejudge_batch(
            summary_path,
            judge_model="claude-sonnet-4.6",
            output_dir=workspace / "sidecars",
            model_client=FakeModelClient(),
        )
        duplicate = result.output_dir / "duplicate.json"
        duplicate.write_bytes(_sidecar_path(result.output_dir).read_bytes())

        with pytest.raises(RejudgeError, match="Duplicate sidecar"):
            merge_rejudged_summary(
                summary_path,
                result.output_dir,
                output_path=workspace / "merged.json",
            )


def test_two_judge_merge_preserves_independent_provenance() -> None:
    with experiment_workspace("rejudge-dual-merge") as workspace:
        spec, _ = _create_run(workspace, scores=(0.8,))
        _, summary_path = aggregate_runs(Path(spec.artifact_root))
        source = json.loads(summary_path.read_text(encoding="utf-8"))
        primary = rejudge_batch(
            summary_path,
            judge_model="claude-sonnet-4.6",
            output_dir=workspace / "primary",
            model_client=FakeModelClient(score=0.7),
        )
        secondary = rejudge_batch(
            summary_path,
            judge_model="gemini-3.5-flash",
            output_dir=workspace / "secondary",
            model_client=FakeModelClient(score=0.8),
        )
        first_path, primary_metric = merge_rejudged_summary(
            summary_path,
            primary.output_dir,
            output_path=workspace / "first.json",
        )
        final_path, secondary_metric = merge_rejudged_summary(
            first_path,
            secondary.output_dir,
            output_path=workspace / "final.json",
        )

        final = json.loads(final_path.read_text(encoding="utf-8"))
        c5 = final["c5_rejudge"]
        schema = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "experiments/schemas/c5_summary_provenance.schema.json"
            ).read_text(encoding="utf-8")
        )
        validate(instance=c5, schema=schema)
        assert final["original_summary_hash"] == source["summary_hash"]
        assert set(c5["judges"]) == {
            "visual-form-primary-v1",
            "visual-form-secondary-v1",
        }
        assert c5["input_manifest_hash"] == json.loads(
            primary.batch_path.read_text(encoding="utf-8")
        )["input_manifest_hash"]
        assert c5["judges"]["visual-form-primary-v1"]["batch_hash"] == json.loads(
            primary.batch_path.read_text(encoding="utf-8")
        )["batch_hash"]
        assert c5["judges"]["visual-form-secondary-v1"]["batch_hash"] == json.loads(
            secondary.batch_path.read_text(encoding="utf-8")
        )["batch_hash"]
        assert final["runs"][0][primary_metric] == 0.7
        assert final["runs"][0][secondary_metric] == 0.8
        assert load_provenance_summary(final_path).summary_hash == final["summary_hash"]
