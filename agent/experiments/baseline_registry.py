from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


LICENSE_NOT_DECLARED = "not_declared"


@dataclass(frozen=True)
class BaselineDefinition:
    name: str
    repo_url: str
    commit: str
    input_track: str
    license_status: str
    entrypoints: Dict[str, str]
    required_files: tuple[str, ...]
    dependency_modules: tuple[str, ...]
    required_env: tuple[str, ...] = ()
    checkpoint_env: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


BASELINE_REGISTRY: Dict[str, BaselineDefinition] = {
    "chartcoder": BaselineDefinition(
        name="ChartCoder",
        repo_url="https://github.com/thunlp/ChartCoder",
        commit="47d965c971be15eebc5bd495c87efcfa34b2016a",
        input_track="image_to_code_reconstruction",
        license_status=LICENSE_NOT_DECLARED,
        entrypoints={"reconstruction": "inference.py:ChartCoder"},
        required_files=(
            "inference.py",
            "llava/model/builder.py",
            "requirements.txt",
        ),
        dependency_modules=("torch", "PIL", "transformers"),
        checkpoint_env="CHARTCODER_CHECKPOINT",
    ),
    "matplotagent": BaselineDefinition(
        name="MatPlotAgent",
        repo_url="https://github.com/thunlp/MatPlotAgent",
        commit="9cafa262aae7bdf85fccf6d02b2153fb772bc376",
        input_track="table_instruction",
        license_status=LICENSE_NOT_DECLARED,
        entrypoints={
            "direct": "one_time_generate.py:mainworkflow",
            "one_time": "one_time_generate.py:mainworkflow",
            "workflow": "workflow.py:mainworkflow",
        },
        required_files=(
            "one_time_generate.py",
            "workflow.py",
            "agents/plot_agent/agent.py",
            "agents/config/openai.py",
            "requirements.txt",
        ),
        dependency_modules=("openai", "tenacity", "matplotlib"),
        required_env=("MATPLOTAGENT_API_KEY", "MATPLOTAGENT_BASE_URL"),
    ),
    "nvagent": BaselineDefinition(
        name="nvAgent",
        repo_url="https://github.com/geliang0114/nvAgent",
        commit="a37209e675813a25241e83f2fe56a87657a49ef6",
        input_track="table_nl_instruction",
        license_status=LICENSE_NOT_DECLARED,
        entrypoints={"workflow": "core.chat_manager:ChatManager"},
        required_files=(
            "core/chat_manager.py",
            "core/agents.py",
            "core/api_config.py",
            "requirements.txt",
        ),
        dependency_modules=(
            "openai",
            "duckdb",
            "pandas",
            "matplotlib",
            "seaborn",
            "sqlglot",
            "func_timeout",
            "attr",
            "tqdm",
        ),
        required_env=(
            "NVAGENT_AZURE_OPENAI_API_KEY",
            "NVAGENT_AZURE_OPENAI_ENDPOINT",
            "NVAGENT_OPENAI_API_VERSION",
        ),
    ),
}


class BaselinePreflightError(RuntimeError):
    def __init__(self, report: "PreflightReport") -> None:
        self.report = report
        failed = [
            check.name for check in report.checks if not check.ok
        ]
        super().__init__(
            f"{report.baseline} preflight failed: {', '.join(failed)}"
        )


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class PreflightReport:
    baseline: str
    repo_url: str
    expected_commit: str
    actual_commit: Optional[str]
    input_track: str
    license_status: str
    entrypoints: Dict[str, str]
    checks: tuple[PreflightCheck, ...]

    @property
    def ready(self) -> bool:
        return all(check.ok for check in self.checks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline,
            "repo_url": self.repo_url,
            "expected_commit": self.expected_commit,
            "actual_commit": self.actual_commit,
            "input_track": self.input_track,
            "license_status": self.license_status,
            "entrypoints": dict(self.entrypoints),
            "ready": self.ready,
            "checks": [asdict(check) for check in self.checks],
        }


def _normalized_repo_url(url: str) -> str:
    normalized = url.strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized.casefold()


def _git_output(checkout: Path, *args: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, type(exc).__name__
    if result.returncode != 0:
        return False, (result.stderr or result.stdout).strip()[:300]
    return True, result.stdout.strip()


def _dependency_checks(
    modules: Sequence[str],
    *,
    python_executable: str,
) -> list[PreflightCheck]:
    executable = shutil.which(python_executable)
    if executable is None and Path(python_executable).is_file():
        executable = str(Path(python_executable).resolve())
    if executable is None:
        return [
            PreflightCheck(
                name="python_executable",
                ok=False,
                detail="configured Python executable was not found",
            )
        ]
    script = (
        "import importlib.util,json,sys;"
        "print(json.dumps({name: importlib.util.find_spec(name) is not None "
        "for name in sys.argv[1:]}))"
    )
    try:
        result = subprocess.run(
            [executable, "-c", script, *modules],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        discovered = json.loads(result.stdout) if result.returncode == 0 else {}
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        discovered = {}
    checks = [
        PreflightCheck(
            name=f"dependency:{module}",
            ok=bool(discovered.get(module)),
            detail="available" if discovered.get(module) else "missing",
        )
        for module in modules
    ]
    checks.insert(
        0,
        PreflightCheck(
            name="python_executable",
            ok=True,
            detail=Path(executable).name,
        ),
    )
    return checks


def preflight_baseline(
    definition: BaselineDefinition,
    checkout: Path,
    *,
    python_executable: str = "python",
    required_env: Optional[Sequence[str]] = None,
    checkpoint: Optional[Path] = None,
    check_dependencies: bool = True,
    environ: Optional[Mapping[str, str]] = None,
) -> PreflightReport:
    checkout = checkout.expanduser().resolve()
    environment = os.environ if environ is None else environ
    checks: list[PreflightCheck] = []

    root_ok, root_output = _git_output(checkout, "rev-parse", "--show-toplevel")
    root_matches = root_ok and Path(root_output).resolve() == checkout
    checks.append(
        PreflightCheck(
            name="git_repository",
            ok=root_matches,
            detail="root matches checkout" if root_matches else "not the git root",
        )
    )

    remote_ok, remote = _git_output(checkout, "remote", "get-url", "origin")
    remote_matches = remote_ok and _normalized_repo_url(
        remote
    ) == _normalized_repo_url(definition.repo_url)
    checks.append(
        PreflightCheck(
            name="origin",
            ok=remote_matches,
            detail="matches registry" if remote_matches else "origin mismatch",
        )
    )

    head_ok, actual_commit = _git_output(checkout, "rev-parse", "HEAD")
    commit_matches = head_ok and actual_commit == definition.commit
    checks.append(
        PreflightCheck(
            name="commit",
            ok=commit_matches,
            detail=actual_commit if head_ok else "unavailable",
        )
    )

    clean_ok, status = _git_output(
        checkout,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    clean = clean_ok and not status
    checks.append(
        PreflightCheck(
            name="clean_worktree",
            ok=clean,
            detail="clean" if clean else "tracked or untracked changes present",
        )
    )

    for relative in definition.required_files:
        exists = (checkout / relative).is_file()
        checks.append(
            PreflightCheck(
                name=f"required_file:{relative}",
                ok=exists,
                detail="present" if exists else "missing",
            )
        )

    root_license_files = [
        child.name
        for child in checkout.iterdir()
        if child.is_file()
        and (
            child.name.casefold().startswith("license")
            or child.name.casefold().startswith("copying")
        )
    ] if checkout.is_dir() else []
    if definition.license_status == LICENSE_NOT_DECLARED:
        license_ok = not root_license_files
        license_detail = (
            "no root license file; registry records not_declared"
            if license_ok
            else f"unexpected root license files: {sorted(root_license_files)}"
        )
    else:
        license_ok = bool(root_license_files)
        license_detail = (
            f"root license files: {sorted(root_license_files)}"
            if license_ok
            else "declared license file missing"
        )
    checks.append(
        PreflightCheck(
            name="license",
            ok=license_ok,
            detail=license_detail,
        )
    )

    env_names = tuple(
        definition.required_env if required_env is None else required_env
    )
    for name in env_names:
        present = bool(environment.get(name))
        checks.append(
            PreflightCheck(
                name=f"env:{name}",
                ok=present,
                detail="set" if present else "missing",
            )
        )

    if definition.checkpoint_env is not None:
        checkpoint_ok = checkpoint is not None and checkpoint.exists()
        checks.append(
            PreflightCheck(
                name=f"checkpoint:{definition.checkpoint_env}",
                ok=checkpoint_ok,
                detail=(
                    "present"
                    if checkpoint_ok
                    else "required checkpoint path is missing"
                ),
            )
        )

    if check_dependencies:
        checks.extend(
            _dependency_checks(
                definition.dependency_modules,
                python_executable=python_executable,
            )
        )

    return PreflightReport(
        baseline=definition.name,
        repo_url=definition.repo_url,
        expected_commit=definition.commit,
        actual_commit=actual_commit if head_ok else None,
        input_track=definition.input_track,
        license_status=definition.license_status,
        entrypoints=dict(definition.entrypoints),
        checks=tuple(checks),
    )


def require_preflight(report: PreflightReport) -> None:
    if not report.ready:
        raise BaselinePreflightError(report)
