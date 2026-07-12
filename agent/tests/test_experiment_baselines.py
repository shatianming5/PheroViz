from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

import pytest

from experiments.baseline_registry import (
    LICENSE_NOT_DECLARED,
    BaselineDefinition,
    BaselinePreflightError,
    preflight_baseline,
    require_preflight,
)
from experiments.external_baselines import (
    BaselineInvocation,
    ChartCoderProvider,
    ExternalBaselineProvider,
    MatPlotAgentProvider,
    NvAgentProvider,
    _NVAGENT_DRIVER,
)
from experiments.harness import execute_experiment
from experiments.manifest import DatasetCase
from experiments.providers import GenerationRequest, ProviderExecutionError
from tests.test_experiment_support import (
    experiment_workspace,
    make_spec,
    write_manifest,
)


_FAKE_ENTRY = r'''
import json
import os
import sys
import time
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
secret = os.environ.get("FAKE_SECRET", "")
home = Path.home()
print(f"stdout secret={secret}")
print(f"stderr secret={secret}", file=sys.stderr)
if config["behavior"] == "fail":
    raise SystemExit(7)
if config["behavior"] == "sleep":
    time.sleep(0.2)

work_dir = Path(config["work_dir"])
code_path = work_dir / "generated.py"
code_path.write_text("test_only = True\n", encoding="utf-8")
if config["behavior"] == "escape":
    image_path = work_dir.parent / "escaped.png"
else:
    image_path = work_dir / "figure.png"
image_path.write_bytes(b"test-only-image")
log_path = work_dir / "baseline.log"
log_path.write_text(f"secret={secret}\nhome={home}\n", encoding="utf-8")
Path(config["result_path"]).write_text(
    json.dumps(
        {
            "code_path": str(code_path),
            "image_path": str(image_path),
            "log_path": str(log_path),
            "test_only": True,
        }
    ),
    encoding="utf-8",
)
'''


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fake_repo(
    workspace: Path,
    *,
    checkpoint_env: Optional[str] = None,
    track: str = "table_instruction",
) -> tuple[Path, BaselineDefinition]:
    repo = workspace / "fake-baseline"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Experiment Tests")
    (repo / "fake_entry.py").write_text(_FAKE_ENTRY, encoding="utf-8")
    _git(repo, "add", "fake_entry.py")
    _git(repo, "commit", "-q", "-m", "fake baseline")
    repo_url = "https://example.invalid/fake-baseline"
    _git(repo, "remote", "add", "origin", repo_url)
    commit = _git(repo, "rev-parse", "HEAD")
    definition = BaselineDefinition(
        name="FakeBaseline",
        repo_url=repo_url,
        commit=commit,
        input_track=track,
        license_status=LICENSE_NOT_DECLARED,
        entrypoints={"fake": "fake_entry.py"},
        required_files=("fake_entry.py",),
        dependency_modules=(),
        required_env=("FAKE_SECRET",),
        checkpoint_env=checkpoint_env,
    )
    return repo, definition


class FakeExternalProvider(ExternalBaselineProvider):
    test_only = True

    def _prepare_invocation(
        self,
        request: GenerationRequest,
        case,
    ) -> BaselineInvocation:
        result_path = request.output_dir / "driver_result.json"
        config_path = request.output_dir / "fake_input.json"
        config_path.write_text(
            json.dumps(
                {
                    "behavior": request.spec.method_config.get(
                        "behavior",
                        "success",
                    ),
                    "work_dir": str(request.output_dir),
                    "result_path": str(result_path),
                    "test_only": True,
                }
            ),
            encoding="utf-8",
        )
        return BaselineInvocation(
            argv=(
                self.python_executable,
                str(self.repo_path / "fake_entry.py"),
                str(config_path),
            ),
            environment={
                "FAKE_SECRET": self.environ.get("FAKE_SECRET", ""),
            },
            result_manifest=result_path,
            served_model=request.spec.backbone,
            entrypoint=self.definition.entrypoints["fake"],
        )


def _external_spec(
    workspace: Path,
    *,
    behavior: str = "success",
    input_track: str = "table_instruction",
):
    table = workspace / "table.csv"
    table.write_text("x,y\n1,2\n", encoding="utf-8")
    write_manifest(
        workspace,
        [
            {
                "case_id": "baseline-case",
                "panel_count": 1,
                "split": "test",
                "input_track": input_track,
                "table_path": str(table),
                "instruction": "Plot y against x.",
            }
        ],
    )
    spec = make_spec(
        workspace,
        run_name=f"external-{behavior}",
        case_id="baseline-case",
        panel_count=1,
        split="test",
        budget_value=1,
        selection_metric="execution_success",
    )
    return replace(spec, method_config={"behavior": behavior})


def _request(spec, workspace: Path) -> GenerationRequest:
    output = workspace / "provider-output"
    output.mkdir()
    return GenerationRequest(
        spec=spec,
        dataset_manifest_path=Path(spec.dataset_manifest_path),
        output_dir=output,
        call_index=1,
        remaining_renders=1,
        remaining_seconds=None,
        deadline_monotonic=None,
        history=(),
        previous_candidate=None,
    )


def test_preflight_rejects_commit_mismatch() -> None:
    with experiment_workspace("baseline-commit") as workspace:
        repo, definition = _fake_repo(workspace)
        mismatched = replace(definition, commit="0" * 40)

        report = preflight_baseline(
            mismatched,
            repo,
            check_dependencies=False,
            environ={"FAKE_SECRET": "set"},
        )

        assert not report.ready
        assert not next(
            check for check in report.checks if check.name == "commit"
        ).ok
        with pytest.raises(BaselinePreflightError):
            require_preflight(report)


def test_matplotagent_injects_standard_openai_environment() -> None:
    with experiment_workspace("matplot-openai-env") as workspace:
        table = workspace / "table.csv"
        table.write_text("x,y\n1,2\n", encoding="utf-8")
        spec = _external_spec(workspace)
        request = _request(spec, workspace)
        case = DatasetCase(
            case_id="baseline-case",
            panel_count=1,
            split="test",
            payload={
                "input_track": "table_instruction",
                "data_path": str(table),
                "instruction": "Plot y against x.",
            },
        )
        provider = MatPlotAgentProvider(
            repo_path=workspace,
            check_dependencies=False,
            environ={
                "MATPLOTAGENT_API_KEY": "test-key",
                "MATPLOTAGENT_BASE_URL": "https://example.test/v1",
            },
        )

        invocation = provider._prepare_invocation(request, case)

        assert invocation.environment["OPENAI_API_KEY"] == "test-key"
        assert (
            invocation.environment["OPENAI_BASE_URL"]
            == "https://example.test/v1/"
        )
        assert (request.output_dir / "workspace" / "data.csv").is_file()


def test_nvagent_openai_compatible_mode_is_explicit() -> None:
    with experiment_workspace("nvagent-openai-compatible") as workspace:
        table = workspace / "table.csv"
        table.write_text("x,y\n1,2\n", encoding="utf-8")
        spec = replace(
            _external_spec(workspace),
            method_config={"openai_compatible": True},
        )
        request = _request(spec, workspace)
        case = DatasetCase(
            case_id="baseline-case",
            panel_count=1,
            split="test",
            payload={
                "input_track": "table_nl_instruction",
                "data_path": str(table),
                "instruction": "Plot y against x.",
            },
        )
        provider = NvAgentProvider(
            repo_path=workspace,
            check_dependencies=False,
            environ={
                "NVAGENT_AZURE_OPENAI_API_KEY": "test-key",
                "NVAGENT_AZURE_OPENAI_ENDPOINT": "https://example.test/v1",
                "NVAGENT_OPENAI_API_VERSION": "2024-02-01",
            },
        )

        invocation = provider._prepare_invocation(request, case)
        config = json.loads(
            (request.output_dir / "baseline_input.json").read_text(
                encoding="utf-8"
            )
        )

        assert config["openai_compatible"] is True
        assert invocation.environment["PHEROVIZ_AZURE_OPENAI_API_KEY"] == (
            "test-key"
        )
        assert "PROVIDER_NAME = core_const.PROCESSOR_NAME" in _NVAGENT_DRIVER


def test_preflight_rejects_dirty_repo() -> None:
    with experiment_workspace("baseline-dirty") as workspace:
        repo, definition = _fake_repo(workspace)
        (repo / "untracked.txt").write_text("dirty", encoding="utf-8")

        report = preflight_baseline(
            definition,
            repo,
            check_dependencies=False,
            environ={"FAKE_SECRET": "set"},
        )

        assert not report.ready
        assert not next(
            check for check in report.checks if check.name == "clean_worktree"
        ).ok


def test_missing_license_is_recorded_and_checkpoint_is_required() -> None:
    with experiment_workspace("baseline-license-checkpoint") as workspace:
        repo, definition = _fake_repo(
            workspace,
            checkpoint_env="FAKE_CHECKPOINT",
        )

        report = preflight_baseline(
            definition,
            repo,
            check_dependencies=False,
            environ={"FAKE_SECRET": "set"},
            checkpoint=None,
        )

        license_check = next(
            check for check in report.checks if check.name == "license"
        )
        checkpoint_check = next(
            check
            for check in report.checks
            if check.name == "checkpoint:FAKE_CHECKPOINT"
        )
        assert license_check.ok
        assert "not_declared" in license_check.detail
        assert not checkpoint_check.ok
        assert report.license_status == "not_declared"


def test_track_mismatch_fails_before_subprocess() -> None:
    with experiment_workspace("baseline-track") as workspace:
        repo, definition = _fake_repo(workspace)
        spec = _external_spec(
            workspace,
            input_track="image_to_code_reconstruction",
        )
        provider = FakeExternalProvider(
            definition,
            repo,
            check_dependencies=False,
            environ={"FAKE_SECRET": "secret-value"},
        )

        with pytest.raises(ProviderExecutionError, match="requires input_track"):
            provider.generate(_request(spec, workspace))


def test_secret_redaction_and_success_artifacts() -> None:
    with experiment_workspace("baseline-success") as workspace:
        repo, definition = _fake_repo(workspace)
        spec = _external_spec(workspace)
        provider = FakeExternalProvider(
            definition,
            repo,
            check_dependencies=False,
            environ={"FAKE_SECRET": "super-secret-value"},
        )

        outcome = execute_experiment(
            spec,
            provider_loader=lambda import_path, options: provider,
        )

        assert outcome.record.status == "completed"
        assert outcome.record.test_only
        candidate = outcome.record.candidates[0]
        metadata = candidate["provider_metadata"]
        assert metadata["external_repo_commit"] == definition.commit
        assert metadata["track"] == definition.input_track
        assert metadata["exit_code"] == 0
        assert metadata["served_model"] == spec.backbone
        serialized = json.dumps(metadata)
        assert "super-secret-value" not in serialized
        assert "***REDACTED***" in serialized
        assert {"code", "image", "stdout", "stderr"} <= set(
            candidate["artifact_paths"]
        )
        run_dir = Path(spec.artifact_root) / spec.run_name
        text_artifacts = ("code", "log", "stdout", "stderr", "subprocess_result")
        for label in text_artifacts:
            artifact = run_dir / candidate["artifact_paths"][label]
            assert "super-secret-value" not in artifact.read_text(
                encoding="utf-8"
            )
        assert _git(repo, "status", "--porcelain") == ""
        log = (
            run_dir / candidate["artifact_paths"]["log"]
        ).read_text(encoding="utf-8")
        assert "isolated_home" in log


def test_subprocess_failure_is_failed_with_redacted_logs() -> None:
    with experiment_workspace("baseline-failure") as workspace:
        repo, definition = _fake_repo(workspace)
        spec = _external_spec(workspace, behavior="fail")
        provider = FakeExternalProvider(
            definition,
            repo,
            check_dependencies=False,
            environ={"FAKE_SECRET": "failure-secret"},
        )

        outcome = execute_experiment(
            spec,
            provider_loader=lambda import_path, options: provider,
        )

        assert outcome.record.status == "failed"
        assert outcome.record.error["type"] == "ProviderExecutionError"
        failure_keys = {
            key
            for key in outcome.record.artifact_paths
            if key.startswith("failure.call_0001")
        }
        assert failure_keys
        run_dir = Path(spec.artifact_root) / spec.run_name
        evidence = "\n".join(
            (run_dir / outcome.record.artifact_paths[key]).read_text(
                encoding="utf-8"
            )
            for key in failure_keys
        )
        assert "failure-secret" not in evidence
        assert "***REDACTED***" in evidence
        assert '"exit_code": 7' in evidence


def test_external_timeout_recomputes_absolute_deadline_before_launch() -> None:
    with experiment_workspace("baseline-timeout") as workspace:
        repo, definition = _fake_repo(workspace)
        spec = _external_spec(workspace, behavior="sleep")
        request = replace(
            _request(spec, workspace),
            remaining_renders=None,
            remaining_seconds=10.0,
            deadline_monotonic=100.01,
        )
        provider = FakeExternalProvider(
            definition,
            repo,
            timeout_seconds=10.0,
            check_dependencies=False,
            environ={"FAKE_SECRET": "set"},
            monotonic=lambda: 100.0,
        )

        with pytest.raises(ProviderExecutionError):
            provider.generate(request)

        metadata = json.loads(
            (request.output_dir / "subprocess_result.json").read_text(
                encoding="utf-8"
            )
        )
        assert metadata["timeout_seconds"] == pytest.approx(0.01)
        assert metadata["exit_code"] == -1


def test_external_deadline_expired_during_setup_prevents_launch() -> None:
    with experiment_workspace("baseline-expired") as workspace:
        repo, definition = _fake_repo(workspace)
        spec = _external_spec(workspace)
        request = replace(
            _request(spec, workspace),
            remaining_renders=None,
            remaining_seconds=10.0,
            deadline_monotonic=99.0,
        )
        provider = FakeExternalProvider(
            definition,
            repo,
            timeout_seconds=10.0,
            check_dependencies=False,
            environ={"FAKE_SECRET": "set"},
            monotonic=lambda: 100.0,
        )

        with pytest.raises(ProviderExecutionError) as error:
            provider.generate(request)

        metadata = json.loads(
            (request.output_dir / "subprocess_result.json").read_text(
                encoding="utf-8"
            )
        )
        assert metadata["timeout_seconds"] == 0.0
        assert metadata["exit_code"] == -1
        assert "expired before subprocess launch" in metadata["stderr_summary"]
        assert not (request.output_dir / "driver_result.json").exists()
        assert error.value.failure_attribution == "method"


def test_artifact_escape_is_rejected() -> None:
    with experiment_workspace("baseline-escape") as workspace:
        repo, definition = _fake_repo(workspace)
        spec = _external_spec(workspace, behavior="escape")
        provider = FakeExternalProvider(
            definition,
            repo,
            check_dependencies=False,
            environ={"FAKE_SECRET": "set"},
        )

        outcome = execute_experiment(
            spec,
            provider_loader=lambda import_path, options: provider,
        )

        assert outcome.record.status == "failed"
        assert "escaped its work directory" in outcome.record.error["message"]


def test_chartcoder_rejects_missing_checkpoint_reference_and_wrong_track() -> None:
    with experiment_workspace("chartcoder-guards") as workspace:
        repo, definition = _fake_repo(
            workspace,
            checkpoint_env="CHARTCODER_CHECKPOINT",
            track="image_to_code_reconstruction",
        )
        missing_checkpoint = ChartCoderProvider(
            repo,
            definition=definition,
            check_dependencies=False,
            environ={"FAKE_SECRET": "set"},
        )
        with pytest.raises(BaselinePreflightError):
            missing_checkpoint.check_available()

        checkpoint = workspace / "checkpoint"
        checkpoint.mkdir()
        provider = ChartCoderProvider(
            repo,
            definition=definition,
            check_dependencies=False,
            environ={
                "FAKE_SECRET": "set",
                "CHARTCODER_CHECKPOINT": str(checkpoint),
            },
        )
        spec = _external_spec(
            workspace,
            input_track="image_to_code_reconstruction",
        )
        request = _request(spec, workspace)
        with pytest.raises(ProviderExecutionError, match="reference_image"):
            provider.generate(request)

    with experiment_workspace("chartcoder-track") as workspace:
        repo, definition = _fake_repo(
            workspace,
            checkpoint_env="CHARTCODER_CHECKPOINT",
            track="image_to_code_reconstruction",
        )
        checkpoint = workspace / "checkpoint"
        checkpoint.mkdir()
        provider = ChartCoderProvider(
            repo,
            definition=definition,
            check_dependencies=False,
            environ={
                "FAKE_SECRET": "set",
                "CHARTCODER_CHECKPOINT": str(checkpoint),
            },
        )
        spec = _external_spec(workspace, input_track="table_instruction")
        with pytest.raises(ProviderExecutionError, match="requires input_track"):
            provider.generate(_request(spec, workspace))
