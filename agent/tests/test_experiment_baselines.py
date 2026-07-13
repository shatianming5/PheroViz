from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

import pytest

from app.evaluation import EXPECTATION_SCHEMA_VERSION
from experiments.aggregate import aggregate_runs
from experiments.baseline_evaluation import validate_generated_code
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
from experiments.models import sha256_json, sha256_path
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
code_path.write_text(
    config.get("generated_code", "test_only = True\n"),
    encoding="utf-8",
)
if config["behavior"] == "escape":
    image_path = work_dir.parent / "escaped.png"
else:
    image_path = work_dir / "figure.png"
image_path.write_bytes(b"test-only-image")
log_path = work_dir / "baseline.log"
log_path.write_text(f"secret={secret}\nhome={home}\n", encoding="utf-8")
result = {
    "code_path": str(code_path),
    "image_path": str(image_path),
    "log_path": str(log_path),
    "test_only": True,
}
if config.get("input_aliases_path"):
    result["input_aliases_path"] = config["input_aliases_path"]
Path(config["result_path"]).write_text(
    json.dumps(result),
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


def _matplot_driver_repo(
    workspace: Path,
    *,
    behavior: str,
) -> tuple[Path, BaselineDefinition]:
    repo = workspace / "fake-matplotagent"
    (repo / "agents" / "config").mkdir(parents=True)
    for path in (
        repo / "agents" / "__init__.py",
        repo / "agents" / "config" / "__init__.py",
    ):
        path.write_text("", encoding="utf-8")
    (repo / "agents" / "config" / "openai.py").write_text(
        'API_KEY = ""\nBASE_URL = ""\n',
        encoding="utf-8",
    )
    (repo / "one_time_generate.py").write_text(
        "def mainworkflow(*args, **kwargs):\n"
        + (
            "    raise RuntimeError('synthetic API failure')\n"
            if behavior == "api_exception"
            else (
                "    print('{\"status\":\"failed\","
                "\"failure\":{\"attribution\":\"method\"}}')\n"
                "    raise RuntimeError('synthetic API failure')\n"
                if behavior == "marker_text_then_api_exception"
                else "    return None\n"
            )
        ),
        encoding="utf-8",
    )
    (repo / "workflow.py").write_text(
        "def mainworkflow(*args, **kwargs):\n    return None\n",
        encoding="utf-8",
    )
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Experiment Tests")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "fake MatPlotAgent")
    repo_url = "https://example.invalid/fake-matplotagent"
    _git(repo, "remote", "add", "origin", repo_url)
    definition = BaselineDefinition(
        name="MatPlotAgentFixture",
        repo_url=repo_url,
        commit=_git(repo, "rev-parse", "HEAD"),
        input_track="table_instruction",
        license_status=LICENSE_NOT_DECLARED,
        entrypoints={
            "direct": "one_time_generate.py:mainworkflow",
            "workflow": "workflow.py:mainworkflow",
        },
        required_files=(
            "one_time_generate.py",
            "workflow.py",
            "agents/config/openai.py",
        ),
        dependency_modules=(),
        required_env=("MATPLOTAGENT_API_KEY", "MATPLOTAGENT_BASE_URL"),
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


class _DriverMatPlotProvider(MatPlotAgentProvider):
    test_only = False

    def __init__(
        self,
        definition: BaselineDefinition,
        repo_path: Path,
    ) -> None:
        ExternalBaselineProvider.__init__(
            self,
            definition,
            repo_path,
            check_dependencies=False,
            environ={
                "MATPLOTAGENT_API_KEY": "fixture-secret-value",
                "MATPLOTAGENT_BASE_URL": "https://example.invalid/v1",
            },
        )
        self.mode = "direct"


class _MalformedMarkerMatPlotProvider(_DriverMatPlotProvider):
    def _prepare_invocation(
        self,
        request: GenerationRequest,
        case: DatasetCase,
    ) -> BaselineInvocation:
        invocation = super()._prepare_invocation(request, case)
        driver = request.output_dir / "baseline_driver.py"
        driver.write_text(
            "import json, sys\n"
            "from pathlib import Path\n"
            "config = json.loads(Path(sys.argv[1]).read_text())\n"
            "Path(config['result_path']).write_text("
            "json.dumps({'status': 'failed'}))\n"
            "raise SystemExit(7)\n",
            encoding="utf-8",
        )
        return invocation


class _FixtureMatPlotAgentProvider(MatPlotAgentProvider):
    test_only = True

    def __init__(
        self,
        definition: BaselineDefinition,
        repo_path: Path,
        *,
        timeout_seconds: float = 1800.0,
    ) -> None:
        ExternalBaselineProvider.__init__(
            self,
            definition,
            repo_path,
            timeout_seconds=timeout_seconds,
            check_dependencies=False,
            environ={"FAKE_SECRET": "fixture-secret-value"},
        )

    def _prepare_invocation(
        self,
        request: GenerationRequest,
        case: DatasetCase,
    ) -> BaselineInvocation:
        return _fixture_invocation(self, request, case, include_aliases=False)


class _FixtureNvAgentProvider(NvAgentProvider):
    test_only = True

    def __init__(
        self,
        definition: BaselineDefinition,
        repo_path: Path,
    ) -> None:
        ExternalBaselineProvider.__init__(
            self,
            definition,
            repo_path,
            check_dependencies=False,
            environ={"FAKE_SECRET": "fixture-secret-value"},
        )

    def _prepare_invocation(
        self,
        request: GenerationRequest,
        case: DatasetCase,
    ) -> BaselineInvocation:
        return _fixture_invocation(self, request, case, include_aliases=True)


def _fixture_invocation(
    provider: ExternalBaselineProvider,
    request: GenerationRequest,
    case: DatasetCase,
    *,
    include_aliases: bool,
) -> BaselineInvocation:
    source = Path(str(case.payload["data_path"]))
    copied_source = request.output_dir / f"eval_data{source.suffix}"
    copied_source.write_bytes(source.read_bytes())
    result_path = request.output_dir / "driver_result.json"
    config_path = request.output_dir / "fake_input.json"
    alias_path = request.output_dir / "input_aliases.json"
    alias_payload = request.spec.method_config.get("alias_payload")
    if include_aliases:
        if not isinstance(alias_payload, dict):
            raise AssertionError("nvAgent fixture requires alias_payload")
        alias_path.write_text(
            json.dumps(alias_payload, sort_keys=True),
            encoding="utf-8",
        )
    config_path.write_text(
        json.dumps(
            {
                "behavior": "success",
                "work_dir": str(request.output_dir),
                "result_path": str(result_path),
                "generated_code": request.spec.method_config[
                    "generated_code"
                ],
                "input_aliases_path": (
                    str(alias_path) if include_aliases else None
                ),
            }
        ),
        encoding="utf-8",
    )
    return BaselineInvocation(
        argv=(
            provider.python_executable,
            str(provider.repo_path / "fake_entry.py"),
            str(config_path),
        ),
        environment={"FAKE_SECRET": "fixture-secret-value"},
        result_manifest=result_path,
        served_model=request.spec.backbone,
        entrypoint=provider.definition.entrypoints["fake"],
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


def _line_expectation(
    *,
    x: str = "x",
    y: str = "y",
    label: str = "signal",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> dict:
    panel = {
        "panel_id": "panel-a",
        "axis_index": 0,
        "series": [
            {
                "series_id": label,
                "kind": "line",
                "label": label,
                "x": x,
                "y": y,
            }
        ],
    }
    if xlabel is not None:
        panel["xlabel"] = xlabel
    if ylabel is not None:
        panel["ylabel"] = ylabel
    return {
        "schema_version": EXPECTATION_SCHEMA_VERSION,
        "panels": [panel],
        "panel_groups": [],
    }


def _programmatic_spec(
    workspace: Path,
    *,
    generated_code: str,
    expectation: Optional[dict],
    source_path: Optional[Path] = None,
    sheet: str | int | None = None,
    alias_payload: Optional[dict] = None,
):
    source = source_path or workspace / "source.csv"
    if source_path is None:
        source.write_text(
            "x,y\n1,10\n2,20\n3,30\n",
            encoding="utf-8",
        )
    case = {
        "case_id": "baseline-case",
        "panel_count": 1,
        "split": "test",
        "input_track": "table_instruction",
        "data_path": str(source),
        "instruction": "Plot the requested series.",
        "sheet": sheet,
    }
    if expectation is not None:
        case["evaluation_expectation"] = expectation
    write_manifest(workspace, [case])
    selection_metric = (
        "data_fidelity" if expectation is not None else "execution_success"
    )
    spec = make_spec(
        workspace,
        run_name="external-programmatic",
        case_id="baseline-case",
        panel_count=1,
        split="test",
        budget_value=1,
        selection_metric=selection_metric,
    )
    metric_config = dict(spec.metric_config)
    metric_config["evaluator"] = {
        "metric_version": "1.0.0",
        "numeric_tolerance": {
            "absolute": 0.5,
            "relative": 0.0,
        },
    }
    return replace(
        spec,
        metric_config=metric_config,
        metric_config_hash=sha256_json(metric_config),
        method_config={
            "generated_code": generated_code,
            "alias_payload": alias_payload,
        },
    )


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


def test_matplotagent_missing_code_and_render_is_method_failure() -> None:
    with experiment_workspace("matplot-output-contract") as workspace:
        repo, definition = _matplot_driver_repo(
            workspace,
            behavior="missing_output",
        )
        spec = _external_spec(workspace)
        provider = _DriverMatPlotProvider(definition, repo)

        outcome = execute_experiment(
            spec,
            provider_loader=lambda import_path, options: provider,
        )

        assert outcome.record.status == "failed"
        assert outcome.record.error["attribution"] == "method"
        assert outcome.record.render_count == 0
        assert outcome.record.candidates == []
        failure_paths = {
            key: value
            for key, value in outcome.record.artifact_paths.items()
            if key.startswith("failure.call_0001")
        }
        assert any(key.endswith(".driver_result") for key in failure_paths)
        assert any(key.endswith(".matplot_workspace") for key in failure_paths)
        assert any(key.endswith(".baseline_input") for key in failure_paths)
        assert any(key.endswith(".baseline_driver") for key in failure_paths)
        run_dir = Path(spec.artifact_root) / spec.run_name
        marker_key = next(
            key for key in failure_paths if key.endswith(".driver_result")
        )
        marker = json.loads(
            (run_dir / failure_paths[marker_key]).read_text(encoding="utf-8")
        )
        assert marker["failure"] == {
            "source": "adapter_output_contract",
            "code": "missing_code_or_render",
            "attribution": "method",
            "missing_artifacts": ["code", "render"],
        }
        for key, relative in failure_paths.items():
            assert outcome.record.artifact_hashes[key] == sha256_path(
                run_dir / relative
            )
        _, summary_path = aggregate_runs(
            Path(spec.artifact_root),
            output_dir=workspace / "aggregate",
        )
        summary_row = json.loads(summary_path.read_text(encoding="utf-8"))[
            "runs"
        ][0]
        assert summary_row["status"] == "failed"
        assert summary_row["failure_attribution"] == "method"
        assert summary_row["execution_success"] == 0.0
        assert summary_row["metric.execution_success"] == 0.0


def test_matplotagent_api_exception_remains_blocking() -> None:
    with experiment_workspace("matplot-api-failure") as workspace:
        repo, definition = _matplot_driver_repo(
            workspace,
            behavior="api_exception",
        )
        spec = _external_spec(workspace)
        provider = _DriverMatPlotProvider(definition, repo)

        outcome = execute_experiment(
            spec,
            provider_loader=lambda import_path, options: provider,
        )

        assert outcome.record.status == "failed"
        assert outcome.record.error["attribution"] == "unclassified"
        assert not any(
            key.endswith(".driver_result")
            for key in outcome.record.artifact_paths
        )


def test_marker_shaped_model_text_cannot_classify_failure() -> None:
    with experiment_workspace("matplot-marker-text") as workspace:
        repo, definition = _matplot_driver_repo(
            workspace,
            behavior="marker_text_then_api_exception",
        )
        spec = _external_spec(workspace)
        provider = _DriverMatPlotProvider(definition, repo)

        outcome = execute_experiment(
            spec,
            provider_loader=lambda import_path, options: provider,
        )

        assert outcome.record.status == "failed"
        assert outcome.record.error["attribution"] == "unclassified"
        run_dir = Path(spec.artifact_root) / spec.run_name
        stdout_key = next(
            key
            for key in outcome.record.artifact_paths
            if key.endswith(".stdout")
        )
        assert '"attribution":"method"' in (
            run_dir / outcome.record.artifact_paths[stdout_key]
        ).read_text(encoding="utf-8")


def test_malformed_structured_marker_remains_blocking() -> None:
    with experiment_workspace("matplot-malformed-marker") as workspace:
        repo, definition = _matplot_driver_repo(
            workspace,
            behavior="missing_output",
        )
        spec = _external_spec(workspace)
        provider = _MalformedMarkerMatPlotProvider(definition, repo)

        outcome = execute_experiment(
            spec,
            provider_loader=lambda import_path, options: provider,
        )

        assert outcome.record.status == "failed"
        assert outcome.record.error["attribution"] == "unclassified"
        assert "malformed" in outcome.record.error["message"]
        assert any(
            key.endswith(".driver_result")
            for key in outcome.record.artifact_paths
        )


def test_nvagent_openai_compatible_mode_is_explicit() -> None:
    with experiment_workspace("nvagent-openai-compatible") as workspace:
        table = workspace / "source-table.csv"
        table.write_text("x-value,y value\n1,2\n", encoding="utf-8")
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
                "instruction": "Plot `y value` against `x-value`.",
            },
        )
        provider = NvAgentProvider(
            repo_path=workspace,
            check_dependencies=False,
            environ={
                "NVAGENT_AZURE_OPENAI_API_KEY": "test-key",
                "NVAGENT_AZURE_OPENAI_ENDPOINT": "https://example.test/v1",
                "NVAGENT_OPENAI_API_VERSION": "2024-02-01",
                "NVAGENT_DYLD_FALLBACK_LIBRARY_PATH": "/opt/cairo/lib",
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
        assert (
            invocation.environment["DYLD_FALLBACK_LIBRARY_PATH"]
            == "/opt/cairo/lib"
        )
        assert config["instruction"] == (
            "Plot `c_y_value` against `c_x_value`."
        )
        assert Path(config["tables"][0]).name == "t_source_table.csv"
        assert Path(config["image_path"]).suffix == ".png"
        aliases = json.loads(
            Path(config["input_aliases_path"]).read_text(encoding="utf-8")
        )
        assert aliases["tables"][0]["column_aliases"] == {
            "x-value": "c_x_value",
            "y value": "c_y_value",
        }
        assert "PROVIDER_NAME = core_const.PROCESSOR_NAME" in _NVAGENT_DRIVER


def test_nvagent_generated_read_only_duckdb_pattern_passes_static_filter() -> None:
    with experiment_workspace("nvagent-static-pattern") as workspace:
        data_dir = workspace / "database"
        data_dir.mkdir()
        (data_dir / "t_source.csv").write_text(
            "c_x,c_y\n1,2\n",
            encoding="utf-8",
        )
        code = f"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import duckdb

data_folder = {str(data_dir)!r}
con = duckdb.connect(database=":memory:")
csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
for file in csv_files:
    table_name = os.path.splitext(file)[0]
    con.execute(f"CREATE VIEW {{table_name}} AS SELECT * FROM read_csv_auto('{{os.path.join(data_folder, file)}}')")
sql = "SELECT c_x, c_y FROM t_source"
df = con.execute(sql).fetchdf()
fig, ax = plt.subplots()
ax.plot(df["c_x"], df["c_y"])
plt.show()
"""

        validate_generated_code(
            code,
            allowed_root=workspace,
            execution_cwd=workspace,
        )


def test_static_filter_rejects_module_import_escape() -> None:
    with experiment_workspace("baseline-static-module-escape") as workspace:
        code = """
import matplotlib
sp = matplotlib.importlib.import_module("subprocess")
sp.getoutput("id")
"""
        with pytest.raises(ValueError, match="matplotlib.importlib"):
            validate_generated_code(
                code,
                allowed_root=workspace,
                execution_cwd=workspace,
            )


def test_matplotagent_archives_programmatic_fidelity_and_hashes() -> None:
    with experiment_workspace("matplot-programmatic") as workspace:
        repo, definition = _fake_repo(workspace)
        code = """
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("eval_data.csv")
fig, ax = plt.subplots()
ax.plot(df["x"], df["y"] + 0.25, label="signal")
ax.legend()
plt.savefig("ignored-by-evaluator.png")
"""
        spec = _programmatic_spec(
            workspace,
            generated_code=code,
            expectation=_line_expectation(),
        )
        provider = _FixtureMatPlotAgentProvider(definition, repo)

        outcome = execute_experiment(
            spec,
            provider_loader=lambda import_path, options: provider,
        )

        assert outcome.record.status == "completed"
        candidate = outcome.record.candidates[0]
        assert candidate["metrics"] == {
            "data_fidelity": 1.0,
            "execution_success": 1.0,
        }
        assert "series_cohesion" not in candidate["metrics"]
        required = {
            "programmatic_evaluation",
            "programmatic_evaluation_input",
            "programmatic_evaluation_render",
            "programmatic_evaluation_validation",
            "programmatic_evaluation_stdout",
            "programmatic_evaluation_stderr",
            "programmatic_evaluation_subprocess",
        }
        assert required <= set(candidate["artifact_paths"])
        run_dir = Path(spec.artifact_root) / spec.run_name
        for label in required:
            artifact = run_dir / candidate["artifact_paths"][label]
            assert candidate["artifact_hashes"][label] == sha256_path(artifact)
        evaluation = json.loads(
            (
                run_dir
                / candidate["artifact_paths"]["programmatic_evaluation"]
            ).read_text(encoding="utf-8")
        )
        assert evaluation["fidelity"]["ratio"] == 1.0
        assert evaluation["cohesion"]["applicable"] is False
        assert evaluation["metric_config"]["numeric_tolerance"] == {
            "absolute": 0.5,
            "relative": 0.0,
        }
        render = run_dir / candidate["artifact_paths"]["render"]
        assert render.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert candidate["artifact_paths"]["render"] == candidate[
            "artifact_paths"
        ]["programmatic_evaluation_render"]
        assert candidate["artifact_paths"]["native_image"] == candidate[
            "artifact_paths"
        ]["image"]
        assert candidate["provider_metadata"]["programmatic_metric_render"] == (
            "render"
        )
        assert candidate["provider_metadata"]["programmatic_evaluation"][
            "render_sha256"
        ] == sha256_path(render)
        assert not (
            run_dir
            / candidate["artifact_paths"]["code"]
        ).parent.joinpath("ignored-by-evaluator.png").exists()


def test_engineering_manifest_without_expectation_stays_execution_only() -> None:
    with experiment_workspace("matplot-engineering") as workspace:
        repo, definition = _fake_repo(workspace)
        spec = _programmatic_spec(
            workspace,
            generated_code="test_only = True\n",
            expectation=None,
        )
        provider = _FixtureMatPlotAgentProvider(definition, repo)

        outcome = execute_experiment(
            spec,
            provider_loader=lambda import_path, options: provider,
        )

        assert outcome.record.status == "completed"
        candidate = outcome.record.candidates[0]
        assert candidate["metrics"] == {"execution_success": 1.0}
        assert candidate["artifact_paths"]["render"] == candidate[
            "artifact_paths"
        ]["image"]
        assert not any(
            label.startswith("programmatic_evaluation")
            for label in candidate["artifact_paths"]
        )


@pytest.mark.parametrize(
    ("code", "expected_error", "expected_attribution"),
    [
        (
            "import os\nos.system('touch escaped.txt')\n",
            "os.system",
            "method",
        ),
        (
            "import matplotlib.pyplot as plt\nplt.figure()\nplt.figure()\n",
            "found 2",
            "unclassified",
        ),
        (
            "import matplotlib.pyplot as plt\nvalue = 1\n",
            "found 0",
            "unclassified",
        ),
        (
            "import matplotlib\nbridge = matplotlib.cbook\n",
            "module graph access",
            "unclassified",
        ),
        (
            'import matplotlib\nmatplotlib.use("module://subprocess")\n',
            "Agg backend",
            "unclassified",
        ),
    ],
)
def test_programmatic_evaluation_fails_closed_with_provenance(
    code: str,
    expected_error: str,
    expected_attribution: str,
) -> None:
    with experiment_workspace("baseline-evaluation-rejected") as workspace:
        repo, definition = _fake_repo(workspace)
        spec = _programmatic_spec(
            workspace,
            generated_code=code,
            expectation=_line_expectation(),
        )
        provider = _FixtureMatPlotAgentProvider(definition, repo)

        outcome = execute_experiment(
            spec,
            provider_loader=lambda import_path, options: provider,
        )

        assert outcome.record.status == "failed"
        assert outcome.record.error["type"] == "ProviderExecutionError"
        assert expected_error in outcome.record.error["message"]
        assert outcome.record.error["attribution"] == expected_attribution
        assert not (workspace / "escaped.txt").exists()
        failure_artifacts = {
            key: value
            for key, value in outcome.record.artifact_paths.items()
            if key.startswith("failure.call_0001")
        }
        assert any(
            key.endswith("programmatic_evaluation_validation")
            for key in failure_artifacts
        )
        assert any(key.endswith(".code") for key in failure_artifacts)
        run_dir = Path(spec.artifact_root) / spec.run_name
        for key, relative_path in failure_artifacts.items():
            artifact = run_dir / relative_path
            assert outcome.record.artifact_hashes[key] == sha256_path(artifact)


def test_programmatic_evaluation_timeout_without_phase_proof_is_blocking() -> None:
    with experiment_workspace("baseline-evaluation-timeout") as workspace:
        repo, definition = _fake_repo(workspace)
        spec = _programmatic_spec(
            workspace,
            generated_code=(
                "import matplotlib.pyplot as plt\n"
                "fig = plt.figure()\n"
                "for item in range(1000000000):\n"
                "    value = item\n"
            ),
            expectation=_line_expectation(),
        )
        provider = _FixtureMatPlotAgentProvider(
            definition,
            repo,
            timeout_seconds=1.0,
        )

        outcome = execute_experiment(
            spec,
            provider_loader=lambda import_path, options: provider,
        )

        assert outcome.record.status == "failed"
        assert outcome.record.error["attribution"] == "unclassified"
        assert "timed out" in outcome.record.error["message"]
        assert not any(
            key.endswith("programmatic_evaluation_render")
            for key in outcome.record.artifact_paths
        )


def test_programmatic_evaluator_config_error_remains_blocking() -> None:
    with experiment_workspace("baseline-evaluation-config") as workspace:
        repo, definition = _fake_repo(workspace)
        spec = _programmatic_spec(
            workspace,
            generated_code=(
                "import matplotlib.pyplot as plt\n"
                "fig = plt.figure()\n"
            ),
            expectation=_line_expectation(),
        )
        metric_config = dict(spec.metric_config)
        metric_config["evaluator"] = "invalid"
        spec = replace(
            spec,
            metric_config=metric_config,
            metric_config_hash=sha256_json(metric_config),
        )
        provider = _FixtureMatPlotAgentProvider(definition, repo)

        outcome = execute_experiment(
            spec,
            provider_loader=lambda import_path, options: provider,
        )

        assert outcome.record.status == "failed"
        assert outcome.record.error["attribution"] == "unclassified"
        assert "metric_config.evaluator" in outcome.record.error["message"]


def test_nvagent_reverses_exact_alias_labels_and_preserves_mapping() -> None:
    with experiment_workspace("nvagent-alias-evaluation") as workspace:
        repo, definition = _fake_repo(workspace)
        source = workspace / "source-table.csv"
        source.write_text(
            "x-value,y value\n1,10\n2,20\n3,30\n",
            encoding="utf-8",
        )
        aliases = {
            "schema_version": "1.0",
            "tables": [
                {
                    "source_name": source.name,
                    "table_alias": "t_source_table",
                    "column_aliases": {
                        "x-value": "c_x_value",
                        "y value": "c_y_value",
                    },
                }
            ],
            "original_instruction": "Plot y value against x-value.",
            "normalized_instruction": (
                "Plot c_y_value against c_x_value."
            ),
        }
        code = """
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [10, 20, 30], label="c_y_value")
ax.set_xlabel("c_x_value")
ax.set_ylabel("c_y_value")
ax.legend()
"""
        spec = _programmatic_spec(
            workspace,
            generated_code=code,
            expectation=_line_expectation(
                x="x-value",
                y="y value",
                label="y value",
                xlabel="x-value",
                ylabel="y value",
            ),
            source_path=source,
            alias_payload=aliases,
        )
        provider = _FixtureNvAgentProvider(definition, repo)

        outcome = execute_experiment(
            spec,
            provider_loader=lambda import_path, options: provider,
        )

        assert outcome.record.status == "completed"
        candidate = outcome.record.candidates[0]
        assert candidate["metrics"]["data_fidelity"] == 1.0
        assert "series_cohesion" not in candidate["metrics"]
        assert candidate["provider_metadata"]["programmatic_evaluation"][
            "alias_reversal"
        ]
        assert "input_aliases" in candidate["artifact_paths"]
        run_dir = Path(spec.artifact_root) / spec.run_name
        alias_artifact = run_dir / candidate["artifact_paths"]["input_aliases"]
        assert json.loads(alias_artifact.read_text(encoding="utf-8")) == aliases
        assert (
            candidate["artifact_hashes"]["input_aliases"]
            == sha256_path(alias_artifact)
        )
        evaluation = json.loads(
            (
                run_dir
                / candidate["artifact_paths"]["programmatic_evaluation"]
            ).read_text(encoding="utf-8")
        )
        axis = evaluation["figure_manifest"]["axes"][0]
        assert axis["x_axis"]["label"] == "x-value"
        assert axis["y_axis"]["label"] == "y value"
        assert axis["series"][0]["label"] == "y value"


def test_nvagent_does_not_evaluate_multi_panel_cohesion() -> None:
    with experiment_workspace("nvagent-no-multi-cohesion") as workspace:
        repo, definition = _fake_repo(workspace)
        source = workspace / "source.csv"
        source.write_text("x,y\n1,2\n", encoding="utf-8")
        write_manifest(
            workspace,
            [
                {
                    "case_id": "baseline-case",
                    "panel_count": 2,
                    "split": "test",
                    "input_track": "table_instruction",
                    "panels": [
                        {"id": "left", "data_path": str(source)},
                        {"id": "right", "data_path": str(source)},
                    ],
                    "evaluation_expectation": {
                        "schema_version": EXPECTATION_SCHEMA_VERSION,
                        "panels": [
                            {
                                "panel_id": "left",
                                "axis_index": 0,
                                "series": [],
                            },
                            {
                                "panel_id": "right",
                                "axis_index": 1,
                                "series": [],
                            },
                        ],
                        "panel_groups": [],
                    },
                }
            ],
        )
        spec = make_spec(
            workspace,
            run_name="nvagent-multi-rejected",
            case_id="baseline-case",
            panel_count=2,
            split="test",
            budget_value=2,
            selection_metric="execution_success",
        )
        provider = _FixtureNvAgentProvider(definition, repo)

        with pytest.raises(
            ProviderExecutionError,
            match="single-panel cases only",
        ):
            provider.generate(_request(spec, workspace))

        assert not (
            workspace / "provider-output" / "driver_result.json"
        ).exists()


def test_programmatic_evaluation_loads_selected_excel_sheet() -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    with experiment_workspace("baseline-evaluation-sheet") as workspace:
        repo, definition = _fake_repo(workspace)
        source = workspace / "source.xlsx"
        with pd.ExcelWriter(source) as writer:
            pd.DataFrame(
                {"x": [1, 2, 3], "y": [100, 200, 300]}
            ).to_excel(writer, sheet_name="wrong", index=False)
            pd.DataFrame(
                {"x": [1, 2, 3], "y": [10, 20, 30]}
            ).to_excel(writer, sheet_name="target", index=False)
        code = """
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [10, 20, 30], label="signal")
ax.legend()
"""
        spec = _programmatic_spec(
            workspace,
            generated_code=code,
            expectation=_line_expectation(),
            source_path=source,
            sheet="target",
        )
        provider = _FixtureMatPlotAgentProvider(definition, repo)

        outcome = execute_experiment(
            spec,
            provider_loader=lambda import_path, options: provider,
        )

        assert outcome.record.status == "completed"
        assert outcome.record.candidates[0]["metrics"]["data_fidelity"] == 1.0


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

        with pytest.raises(ProviderExecutionError) as error:
            provider.generate(request)

        metadata = json.loads(
            (request.output_dir / "subprocess_result.json").read_text(
                encoding="utf-8"
            )
        )
        assert metadata["timeout_seconds"] == pytest.approx(0.01)
        assert metadata["exit_code"] == -1
        assert error.value.failure_attribution == "method"


def test_external_safety_timeout_without_deadline_remains_blocking() -> None:
    with experiment_workspace("baseline-safety-timeout") as workspace:
        repo, definition = _fake_repo(workspace)
        spec = _external_spec(workspace, behavior="sleep")
        provider = FakeExternalProvider(
            definition,
            repo,
            timeout_seconds=0.01,
            check_dependencies=False,
            environ={"FAKE_SECRET": "set"},
        )

        outcome = execute_experiment(
            spec,
            provider_loader=lambda import_path, options: provider,
        )

        assert outcome.record.status == "failed"
        assert outcome.record.error["attribution"] == "unclassified"
        assert outcome.record.error["type"] == "ProviderExecutionError"


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
