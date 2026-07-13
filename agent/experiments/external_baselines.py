from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import pandas as pd

from .baseline_evaluation import (
    BaselineEvaluationError,
    run_programmatic_evaluation,
)
from .baseline_registry import (
    BASELINE_REGISTRY,
    BaselineDefinition,
    BaselinePreflightError,
    PreflightReport,
    preflight_baseline,
    require_preflight,
)
from .manifest import (
    DatasetCase,
    ManifestError,
    load_dataset_manifest,
    select_case,
    verify_case_data_files,
    verify_case_metadata,
)
from .models import sha256_file, slug_identifier, write_json_atomic
from .providers import (
    CandidateResult,
    GenerationRequest,
    ProviderExecutionError,
)


_DEFAULT_BASELINE_ROOT = Path(__file__).resolve().parents[3] / "baseline_repos"
_SECRET_MARKER = "***REDACTED***"
_METHOD_OUTPUT_FAILURE_CODE = "missing_code_or_render"


@dataclass(frozen=True)
class BaselineInvocation:
    argv: tuple[str, ...]
    environment: Dict[str, str]
    result_manifest: Path
    served_model: str
    entrypoint: str
    method_failure_token: Optional[str] = None
    failure_artifacts: Dict[str, str] = field(default_factory=dict)


def _redact_text(text: str, secret_values: Sequence[str]) -> str:
    redacted = text
    for value in sorted(
        {item for item in secret_values if item},
        key=len,
        reverse=True,
    ):
        redacted = redacted.replace(value, _SECRET_MARKER)
    return redacted


def _redact_argv(
    argv: Sequence[str],
    *,
    secret_values: Sequence[str],
    checkout: Path,
    work_dir: Path,
) -> list[str]:
    redacted: list[str] = []
    replacements = (
        (str(checkout), "<external_repo>"),
        (str(work_dir), "<workdir>"),
    )
    for raw in argv:
        value = _redact_text(str(raw), secret_values)
        for original, replacement in replacements:
            value = value.replace(original, replacement)
        redacted.append(value)
    return redacted


def _redact_text_artifact(path: Path, secret_values: Sequence[str]) -> None:
    if not secret_values:
        return
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    redacted = _redact_text(text, secret_values)
    if redacted != text:
        path.write_text(redacted, encoding="utf-8")


def _redact_failure_artifact(
    path: Path,
    secret_values: Sequence[str],
) -> None:
    if path.is_dir() and not path.is_symlink():
        for child in path.rglob("*"):
            if child.is_file() and not child.is_symlink():
                _redact_text_artifact(child, secret_values)
        return
    _redact_text_artifact(path, secret_values)


def _structured_method_failure(
    invocation: BaselineInvocation,
    payload: Any,
) -> tuple[bool, str]:
    token = invocation.method_failure_token
    if token is None:
        return False, "structured marker is not enabled for this adapter"
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "status",
        "adapter_invocation_id",
        "failure",
    }:
        return False, "structured adapter failure marker is malformed: invalid fields"
    failure = payload.get("failure")
    if (
        payload.get("schema_version") != "1.0"
        or payload.get("status") != "failed"
        or payload.get("adapter_invocation_id") != token
        or not isinstance(failure, Mapping)
        or set(failure)
        != {
            "source",
            "code",
            "attribution",
            "missing_artifacts",
        }
    ):
        return False, "structured adapter failure marker is malformed"
    missing = failure.get("missing_artifacts")
    if (
        failure.get("source") != "adapter_output_contract"
        or failure.get("code") != _METHOD_OUTPUT_FAILURE_CODE
        or failure.get("attribution") != "method"
        or not isinstance(missing, list)
        or not missing
        or any(item not in {"code", "render"} for item in missing)
        or missing != sorted(set(missing))
    ):
        return False, "structured adapter failure marker is not allowlisted"
    return True, _METHOD_OUTPUT_FAILURE_CODE


def _evaluator_failure_is_method_output(
    exc: BaselineEvaluationError,
    *,
    generated_code: Path,
) -> bool:
    validation = exc.artifacts.get("programmatic_evaluation_validation")
    if validation is None or not validation.is_file() or validation.is_symlink():
        return False
    try:
        payload = json.loads(validation.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, Mapping)
        and set(payload) == {
            "schema_version",
            "status",
            "code_sha256",
            "error",
        }
        and payload.get("schema_version") == "1.0"
        and payload.get("status") == "rejected"
        and isinstance(payload.get("error"), str)
        and bool(str(payload["error"]).strip())
        and payload.get("code_sha256") == sha256_file(generated_code)
    )


def _minimal_subprocess_environment(
    updates: Mapping[str, str],
    *,
    home_dir: Path,
) -> Dict[str, str]:
    allowed = (
        "PATH",
        "LANG",
        "LC_ALL",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CUDA_VISIBLE_DEVICES",
    )
    environment = {
        name: os.environ[name] for name in allowed if name in os.environ
    }
    environment.update(updates)
    home_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = home_dir / ".cache"
    mpl_dir = home_dir / ".matplotlib"
    cache_dir.mkdir(exist_ok=True)
    mpl_dir.mkdir(exist_ok=True)
    environment["HOME"] = str(home_dir)
    environment["XDG_CACHE_HOME"] = str(cache_dir)
    environment["MPLCONFIGDIR"] = str(mpl_dir)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["MPLBACKEND"] = "Agg"
    return environment


def _resolve_case_file(
    raw: Any,
    *,
    manifest_source: Path,
    field_name: str,
) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ProviderExecutionError(
            f"Case field {field_name!r} must be a non-empty path"
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = manifest_source.parent / path
    path = path.resolve()
    if not path.is_file():
        raise ProviderExecutionError(
            f"Case field {field_name!r} does not exist: {path}"
        )
    return path


def _case_instruction(case: DatasetCase) -> str:
    for key in ("instruction", "user_goal", "nl_instruction", "query"):
        value = case.payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ProviderExecutionError(
        f"case_id {case.case_id!r} requires an instruction/user_goal"
    )


def _table_inputs(
    case: DatasetCase,
    *,
    manifest_source: Path,
) -> list[Path]:
    raw_paths = case.payload.get("table_paths")
    if raw_paths is None:
        single = case.payload.get("table_path", case.payload.get("data_path"))
        raw_paths = [single] if single is not None else None
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ProviderExecutionError(
            f"case_id {case.case_id!r} requires table_path(s)"
        )
    tables = [
        _resolve_case_file(
            raw,
            manifest_source=manifest_source,
            field_name=f"table_paths[{index}]",
        )
        for index, raw in enumerate(raw_paths)
    ]
    allowed_suffixes = {".csv", ".tsv", ".xls", ".xlsx", ".xlsm"}
    for table in tables:
        if table.suffix.lower() not in allowed_suffixes:
            raise ProviderExecutionError(
                f"Unsupported table format for external baseline: {table.suffix}"
            )
    return tables


def _single_panel_evaluation_source(
    case: DatasetCase,
    *,
    manifest_source: Path,
) -> tuple[Path, str | int | None]:
    raw_path = case.payload.get(
        "data_path",
        case.payload.get("table_path"),
    )
    if raw_path is None:
        raw_paths = case.payload.get("table_paths")
        if not isinstance(raw_paths, list) or len(raw_paths) != 1:
            raise ProviderExecutionError(
                "Single-panel programmatic evaluation requires exactly one "
                "original source table"
            )
        raw_path = raw_paths[0]
    source = _resolve_case_file(
        raw_path,
        manifest_source=manifest_source,
        field_name="data_path",
    )
    sheet = case.payload.get("sheet")
    if isinstance(sheet, bool) or (
        sheet is not None and not isinstance(sheet, (str, int))
    ):
        raise ProviderExecutionError(
            "Single-panel programmatic evaluation sheet must be a string, "
            "integer, or null"
        )
    return source, sheet


def _copy_inputs(paths: Sequence[Path], destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=False)
    copied: list[Path] = []
    seen_names: set[str] = set()
    for index, source in enumerate(paths):
        name = source.name
        if name in seen_names:
            name = f"{index:03d}_{name}"
        seen_names.add(name)
        target = destination / name
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def _sql_safe_identifier(
    value: str,
    *,
    prefix: str,
    used: set[str],
) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    stem = stem or "value"
    candidate = f"{prefix}_{stem}"
    index = 2
    while candidate.casefold() in used:
        candidate = f"{prefix}_{stem}_{index}"
        index += 1
    used.add(candidate.casefold())
    return candidate


def _prepare_nvagent_tables(
    paths: Sequence[Path],
    destination: Path,
    *,
    instruction: str,
    sheet: str | int | None = None,
) -> tuple[list[Path], str, Path]:
    destination.mkdir(parents=True, exist_ok=False)
    copied: list[Path] = []
    replacements: Dict[str, str] = {}
    tables: list[Dict[str, Any]] = []
    used_tables: set[str] = set()
    for source in paths:
        suffix = source.suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(source)
        elif suffix == ".tsv":
            frame = pd.read_csv(source, sep="\t")
        elif suffix in {".xls", ".xlsx", ".xlsm"}:
            frame = pd.read_excel(
                source,
                sheet_name=0 if sheet is None else sheet,
            )
        else:
            raise ProviderExecutionError(
                f"nvAgent cannot normalize table format: {source.suffix}"
            )

        table_alias = _sql_safe_identifier(
            source.stem,
            prefix="t",
            used=used_tables,
        )
        used_columns: set[str] = set()
        column_aliases = {
            str(column): _sql_safe_identifier(
                str(column),
                prefix="c",
                used=used_columns,
            )
            for column in frame.columns
        }
        target = destination / f"{table_alias}.csv"
        frame.rename(columns=column_aliases).to_csv(target, index=False)
        copied.append(target)
        replacements[source.stem] = table_alias
        replacements.update(column_aliases)
        tables.append(
            {
                "source_name": source.name,
                "table_alias": table_alias,
                "column_aliases": column_aliases,
                "sheet": sheet if suffix in {".xls", ".xlsx", ".xlsm"} else None,
            }
        )

    normalized_instruction = instruction
    for original, alias in sorted(
        replacements.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        normalized_instruction = normalized_instruction.replace(original, alias)
    alias_path = destination.parent.parent / "input_aliases.json"
    write_json_atomic(
        alias_path,
        {
            "schema_version": "1.0",
            "tables": tables,
            "original_instruction": instruction,
            "normalized_instruction": normalized_instruction,
        },
    )
    return copied, normalized_instruction, alias_path


class ExternalBaselineProvider:
    """Base provider that executes a pinned external checkout in a run workspace."""

    test_only = False

    def __init__(
        self,
        definition: BaselineDefinition,
        repo_path: Path | str,
        *,
        python_executable: str = sys.executable,
        timeout_seconds: float = 1800.0,
        check_dependencies: bool = True,
        environ: Optional[Mapping[str, str]] = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.definition = definition
        self.name = f"external_{definition.name}"
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.python_executable = python_executable
        self.timeout_seconds = float(timeout_seconds)
        self.check_dependencies = check_dependencies
        self.environ = dict(os.environ if environ is None else environ)
        self.monotonic = monotonic
        self.last_preflight: Optional[PreflightReport] = None

    def _required_env(self) -> Sequence[str]:
        return self.definition.required_env

    def _checkpoint_path(self) -> Optional[Path]:
        if self.definition.checkpoint_env is None:
            return None
        raw = self.environ.get(self.definition.checkpoint_env)
        return Path(raw).expanduser().resolve() if raw else None

    def _preflight(self) -> PreflightReport:
        report = preflight_baseline(
            self.definition,
            self.repo_path,
            python_executable=self.python_executable,
            required_env=self._required_env(),
            checkpoint=self._checkpoint_path(),
            check_dependencies=self.check_dependencies,
            environ=self.environ,
        )
        self.last_preflight = report
        return report

    def check_available(self) -> None:
        require_preflight(self._preflight())

    def _case(self, request: GenerationRequest) -> DatasetCase:
        try:
            cases = load_dataset_manifest(
                request.dataset_manifest_path,
                dataset_mode=request.spec.dataset_mode,
            )
            case = select_case(cases, request.spec.case_id)
            verify_case_metadata(
                case,
                panel_count=request.spec.panel_count,
                split=request.spec.split,
            )
            verify_case_data_files(
                case,
                manifest_path=request.dataset_manifest_path,
            )
        except ManifestError as exc:
            raise ProviderExecutionError(str(exc)) from exc
        track = case.payload.get("input_track")
        if track != self.definition.input_track:
            raise ProviderExecutionError(
                f"{self.definition.name} requires input_track="
                f"{self.definition.input_track!r}; got {track!r}"
            )
        return case

    def _prepare_invocation(
        self,
        request: GenerationRequest,
        case: DatasetCase,
    ) -> BaselineInvocation:
        raise NotImplementedError

    def _programmatic_evaluation_kind(self) -> Optional[str]:
        return None

    def _secret_values(self) -> list[str]:
        return [
            self.environ[name]
            for name in self._required_env()
            if self.environ.get(name)
        ]

    def _failure(
        self,
        message: str,
        *,
        artifacts: Mapping[str, Path],
    ) -> ProviderExecutionError:
        return ProviderExecutionError(
            message,
            artifacts={name: str(path) for name, path in artifacts.items()},
        )

    def generate(self, request: GenerationRequest) -> CandidateResult:
        try:
            require_preflight(self._preflight())
        except BaselinePreflightError as exc:
            raise ProviderExecutionError(str(exc)) from exc
        case = self._case(request)
        evaluation_kind = self._programmatic_evaluation_kind()
        expectation = case.payload.get("evaluation_expectation")
        if evaluation_kind is not None and expectation is not None:
            if not isinstance(expectation, Mapping):
                raise ProviderExecutionError(
                    "evaluation_expectation must be an object"
                )
            if request.spec.panel_count != 1:
                raise ProviderExecutionError(
                    f"{self.definition.name} programmatic evaluation supports "
                    "single-panel cases only"
                )
        invocation = self._prepare_invocation(request, case)
        work_dir = request.output_dir.resolve()
        secret_values = self._secret_values()
        redacted_argv = _redact_argv(
            invocation.argv,
            secret_values=secret_values,
            checkout=self.repo_path,
            work_dir=work_dir,
        )

        stdout_path = work_dir / "subprocess.stdout.log"
        stderr_path = work_dir / "subprocess.stderr.log"
        metadata_path = work_dir / "subprocess_result.json"
        started_environment = _minimal_subprocess_environment(
            invocation.environment,
            home_dir=work_dir / "isolated_home",
        )
        python_bin = str(
            Path(self.python_executable).expanduser().absolute().parent
        )
        started_environment["PATH"] = os.pathsep.join(
            [
                python_bin,
                started_environment.get("PATH", ""),
            ]
        ).rstrip(os.pathsep)
        deadline_limited = False
        deadline_expired = False
        if request.deadline_monotonic is not None:
            fresh_remaining = (
                float(request.deadline_monotonic) - self.monotonic()
            )
            deadline_limited = fresh_remaining <= self.timeout_seconds
            deadline_expired = fresh_remaining <= 0
            effective_timeout = (
                0.0
                if deadline_expired
                else min(self.timeout_seconds, fresh_remaining)
            )
        elif request.remaining_seconds is not None:
            effective_timeout = min(
                self.timeout_seconds,
                max(float(request.remaining_seconds), 0.001),
            )
        else:
            effective_timeout = self.timeout_seconds

        timed_out = deadline_expired
        if deadline_expired:
            exit_code = -1
            stdout = ""
            stderr = "Absolute wall-clock deadline expired before subprocess launch"
        else:
            try:
                completed = subprocess.run(
                    list(invocation.argv),
                    cwd=work_dir,
                    env=started_environment,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=effective_timeout,
                )
                exit_code = completed.returncode
                stdout = completed.stdout
                stderr = completed.stderr
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                exit_code = -1
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", errors="replace")
                stderr = f"{stderr}\nTimeoutExpired after {effective_timeout}s"

        stdout = _redact_text(stdout, secret_values)
        stderr = _redact_text(stderr, secret_values)
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        metadata = {
            "external_repo": self.definition.repo_url,
            "external_repo_commit": self.definition.commit,
            "license_status": self.definition.license_status,
            "track": self.definition.input_track,
            "entrypoint": invocation.entrypoint,
            "command_argv": redacted_argv,
            "exit_code": exit_code,
            "served_model": invocation.served_model,
            "timeout_seconds": effective_timeout,
            "stdout_summary": stdout[:2000],
            "stderr_summary": stderr[:2000],
        }
        write_json_atomic(metadata_path, metadata)
        failure_artifacts = {
            "stdout": stdout_path,
            "stderr": stderr_path,
            "subprocess_result": metadata_path,
        }
        for label, raw_path in invocation.failure_artifacts.items():
            path = Path(raw_path)
            if not path.is_absolute():
                path = work_dir / path
            if path.is_symlink():
                continue
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(work_dir)
            except (OSError, ValueError):
                continue
            _redact_failure_artifact(resolved, secret_values)
            failure_artifacts[label] = resolved

        structured_payload: Any = None
        structured_error: Optional[str] = None
        if invocation.result_manifest.is_file():
            try:
                result_path = invocation.result_manifest.resolve(strict=True)
                result_path.relative_to(work_dir)
                _redact_text_artifact(result_path, secret_values)
                failure_artifacts["driver_result"] = result_path
                structured_payload = json.loads(
                    result_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                structured_error = (
                    "structured adapter failure marker is unreadable: "
                    f"{type(exc).__name__}"
                )
        if exit_code != 0:
            structured_method = False
            structured_reason = structured_error
            if not timed_out and structured_payload is not None:
                structured_method, structured_reason = (
                    _structured_method_failure(
                        invocation,
                        structured_payload,
                    )
                )
            metadata["structured_failure_marker"] = {
                "present": structured_payload is not None
                or structured_error is not None,
                "valid_method_failure": structured_method,
                "reason": structured_reason,
            }
            write_json_atomic(metadata_path, metadata)
            failure = self._failure(
                f"{self.definition.name} subprocess failed with exit code "
                f"{exit_code}"
                + (
                    f": {structured_reason}"
                    if structured_reason is not None
                    else ""
                ),
                artifacts=failure_artifacts,
            )
            if structured_method or (timed_out and deadline_limited):
                failure.failure_attribution = "method"
            raise failure

        if not invocation.result_manifest.is_file():
            raise self._failure(
                f"{self.definition.name} produced no adapter result manifest",
                artifacts=failure_artifacts,
            )
        try:
            result_data = json.loads(
                invocation.result_manifest.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise self._failure(
                f"Invalid adapter result manifest: {exc}",
                artifacts=failure_artifacts,
            ) from exc
        if not isinstance(result_data, Mapping):
            raise self._failure(
                "Adapter result manifest must be an object",
                artifacts=failure_artifacts,
            )
        _redact_text_artifact(invocation.result_manifest, secret_values)

        output_artifacts: Dict[str, str] = {
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "subprocess_result": str(metadata_path),
            "driver_result": str(invocation.result_manifest),
        }
        for label in (
            "code",
            "image",
            "log",
            "raw_response",
            "input_aliases",
        ):
            raw_path = result_data.get(f"{label}_path")
            if raw_path is None:
                continue
            path = Path(str(raw_path))
            if not path.is_absolute():
                path = work_dir / path
            if path.is_symlink():
                raise self._failure(
                    f"External artifact {label!r} cannot be a symlink",
                    artifacts=failure_artifacts,
                )
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(work_dir)
            except (OSError, ValueError) as exc:
                raise self._failure(
                    f"External artifact {label!r} escaped its work directory",
                    artifacts=failure_artifacts,
                ) from exc
            if label in {"code", "log", "raw_response", "input_aliases"}:
                _redact_text_artifact(resolved, secret_values)
            output_artifacts[label] = str(resolved)

        if "code" not in output_artifacts or "image" not in output_artifacts:
            raise self._failure(
                f"{self.definition.name} must produce code and image artifacts",
                artifacts=failure_artifacts,
            )
        output_artifacts["render"] = output_artifacts["image"]
        metadata["native_image_artifact"] = "render"
        metrics = {"execution_success": 1.0}
        if evaluation_kind is not None and expectation is not None:
            evaluation_failure_artifacts = {
                **failure_artifacts,
                **{
                    label: Path(path)
                    for label, path in output_artifacts.items()
                },
            }
            try:
                source_path, sheet = _single_panel_evaluation_source(
                    case,
                    manifest_source=Path(request.spec.dataset_manifest_path),
                )
            except ProviderExecutionError as exc:
                raise self._failure(
                    str(exc),
                    artifacts=evaluation_failure_artifacts,
                ) from exc
            evaluator_config = request.spec.metric_config.get("evaluator")
            if evaluator_config is not None and not isinstance(
                evaluator_config,
                Mapping,
            ):
                raise self._failure(
                    "metric_config.evaluator must be an object",
                    artifacts=evaluation_failure_artifacts,
                )
            alias_path = (
                Path(output_artifacts["input_aliases"])
                if evaluation_kind == "nvagent"
                and "input_aliases" in output_artifacts
                else None
            )
            if evaluation_kind == "nvagent" and alias_path is None:
                raise self._failure(
                    "nvAgent programmatic evaluation requires input_aliases",
                    artifacts=evaluation_failure_artifacts,
                )

            evaluation_environment = _minimal_subprocess_environment(
                {},
                home_dir=work_dir / "programmatic_evaluation_home",
            )
            evaluation_environment["CUDA_VISIBLE_DEVICES"] = ""
            evaluation_python_paths = [
                str(Path(__file__).resolve().parents[1])
            ]
            evaluation_python_paths.extend(
                path
                for path in sys.path
                if path
                and Path(path).is_dir()
                and (
                    "site-packages" in Path(path).parts
                    or "dist-packages" in Path(path).parts
                )
            )
            evaluation_environment["PYTHONPATH"] = os.pathsep.join(
                dict.fromkeys(evaluation_python_paths)
            )
            evaluation_python_bin = str(
                Path(sys.executable).expanduser().absolute().parent
            )
            evaluation_environment["PATH"] = os.pathsep.join(
                [
                    evaluation_python_bin,
                    evaluation_environment.get("PATH", ""),
                ]
            ).rstrip(os.pathsep)
            evaluation_timeout = min(self.timeout_seconds, 120.0)
            if request.deadline_monotonic is not None:
                evaluation_timeout = min(
                    evaluation_timeout,
                    float(request.deadline_monotonic) - self.monotonic(),
                )
            elif request.remaining_seconds is not None:
                evaluation_timeout = min(
                    evaluation_timeout,
                    float(request.remaining_seconds),
                )
            try:
                evaluation = run_programmatic_evaluation(
                    python_executable=sys.executable,
                    work_dir=work_dir,
                    code_path=Path(output_artifacts["code"]),
                    source_path=source_path,
                    sheet=sheet,
                    expectation=expectation,
                    metric_config=(
                        evaluator_config
                        if isinstance(evaluator_config, Mapping)
                        else None
                    ),
                    environment=evaluation_environment,
                    timeout_seconds=evaluation_timeout,
                    alias_path=alias_path,
                )
            except BaselineEvaluationError as exc:
                for path in exc.artifacts.values():
                    _redact_text_artifact(path, secret_values)
                method_output_failure = _evaluator_failure_is_method_output(
                    exc,
                    generated_code=Path(output_artifacts["code"]),
                )
                metadata["programmatic_evaluation_failure_classification"] = {
                    "method_output_failure": method_output_failure,
                    "timed_out": exc.timed_out,
                    "basis": (
                        "hash-bound static-validation rejection"
                        if method_output_failure
                        else (
                            "evaluator timeout without phase proof"
                            if exc.timed_out
                            else "not proven to originate from method output"
                        )
                    ),
                }
                metadata["programmatic_evaluation_error"] = str(exc)
                write_json_atomic(metadata_path, metadata)
                failure = self._failure(
                    f"{self.definition.name} programmatic evaluation failed: "
                    f"{exc}",
                    artifacts={
                        **evaluation_failure_artifacts,
                        **exc.artifacts,
                    },
                )
                if method_output_failure:
                    failure.failure_attribution = "method"
                raise failure from exc
            except Exception as exc:
                discovered = {
                    path.name: path
                    for path in work_dir.glob("programmatic_evaluation*")
                    if path.is_file() and not path.is_symlink()
                }
                metadata["programmatic_evaluation_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                write_json_atomic(metadata_path, metadata)
                raise self._failure(
                    f"{self.definition.name} programmatic evaluation setup "
                    f"failed: {type(exc).__name__}: {exc}",
                    artifacts={
                        **evaluation_failure_artifacts,
                        **discovered,
                    },
                ) from exc
            metrics.update(evaluation.metrics)
            output_artifacts.update(evaluation.artifacts)
            output_artifacts["native_image"] = output_artifacts["image"]
            output_artifacts["render"] = evaluation.artifacts[
                "programmatic_evaluation_render"
            ]
            metadata["programmatic_evaluation"] = evaluation.metadata
            metadata["programmatic_metric_render"] = "render"
            metadata["native_image_artifact"] = "native_image"
            write_json_atomic(metadata_path, metadata)
        return CandidateResult(
            metrics=metrics,
            render_count=1,
            artifacts=output_artifacts,
            metadata=metadata,
        )


def _write_driver(work_dir: Path, source: str) -> Path:
    driver = work_dir / "baseline_driver.py"
    driver.write_text(source, encoding="utf-8")
    return driver


class MatPlotAgentProvider(ExternalBaselineProvider):
    def __init__(
        self,
        repo_path: Path | str = _DEFAULT_BASELINE_ROOT / "MatPlotAgent",
        *,
        mode: str = "direct",
        python_executable: str = sys.executable,
        timeout_seconds: float = 1800.0,
        check_dependencies: bool = True,
        environ: Optional[Mapping[str, str]] = None,
    ) -> None:
        super().__init__(
            BASELINE_REGISTRY["matplotagent"],
            repo_path,
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
            check_dependencies=check_dependencies,
            environ=environ,
        )
        self.mode = mode

    def _programmatic_evaluation_kind(self) -> Optional[str]:
        return "matplotagent"

    def _prepare_invocation(
        self,
        request: GenerationRequest,
        case: DatasetCase,
    ) -> BaselineInvocation:
        mode = str(request.spec.method_config.get("mode", self.mode))
        if mode not in self.definition.entrypoints:
            raise ProviderExecutionError(
                f"Unsupported MatPlotAgent mode: {mode!r}"
            )
        tables = _table_inputs(
            case,
            manifest_source=Path(request.spec.dataset_manifest_path),
        )
        copied_tables = _copy_inputs(tables, request.output_dir / "workspace")
        if len(copied_tables) == 1:
            shutil.copy2(
                copied_tables[0],
                request.output_dir / "workspace" / "data.csv",
            )
        instruction = _case_instruction(case)
        config_path = request.output_dir / "baseline_input.json"
        result_path = request.output_dir / "driver_result.json"
        adapter_invocation_id = uuid.uuid4().hex
        write_json_atomic(
            config_path,
            {
                "mode": mode,
                "workspace": str(request.output_dir / "workspace"),
                "model": request.spec.backbone,
                "simple_instruction": instruction,
                "expert_instruction": str(
                    case.payload.get("expert_instruction") or instruction
                ),
                "no_sysprompt": bool(
                    request.spec.method_config.get("no_sysprompt", False)
                ),
                "visual_refine": bool(
                    request.spec.method_config.get("visual_refine", True)
                ),
                "table_files": [str(path) for path in copied_tables],
                "result_path": str(result_path),
                "adapter_invocation_id": adapter_invocation_id,
            },
        )
        driver = _write_driver(request.output_dir, _MATPLOT_DRIVER)
        api_key = self.environ.get("MATPLOTAGENT_API_KEY", "")
        base_url = self.environ["MATPLOTAGENT_BASE_URL"].rstrip("/") + "/"
        environment = {
            "PYTHONPATH": str(self.repo_path),
            "PHEROVIZ_BASELINE_API_KEY": api_key,
            "PHEROVIZ_BASELINE_BASE_URL": base_url,
            "OPENAI_API_KEY": api_key,
            "OPENAI_BASE_URL": base_url,
        }
        visual_refine = bool(
            request.spec.method_config.get("visual_refine", True)
        )
        served_model = request.spec.backbone
        if mode == "workflow" and visual_refine:
            served_model = (
                f"{request.spec.backbone}; visual=gpt-4-vision-preview"
            )
        return BaselineInvocation(
            argv=(self.python_executable, str(driver), str(config_path)),
            environment=environment,
            result_manifest=result_path,
            served_model=served_model,
            entrypoint=self.definition.entrypoints[mode],
            method_failure_token=adapter_invocation_id,
            failure_artifacts={
                "matplot_workspace": str(
                    request.output_dir / "workspace"
                ),
                "baseline_input": str(config_path),
                "baseline_driver": str(driver),
            },
        )


class NvAgentProvider(ExternalBaselineProvider):
    def __init__(
        self,
        repo_path: Path | str = _DEFAULT_BASELINE_ROOT / "nvAgent",
        *,
        python_executable: str = sys.executable,
        timeout_seconds: float = 1800.0,
        check_dependencies: bool = True,
        environ: Optional[Mapping[str, str]] = None,
        openai_compatible: bool = False,
    ) -> None:
        super().__init__(
            BASELINE_REGISTRY["nvagent"],
            repo_path,
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
            check_dependencies=check_dependencies,
            environ=environ,
        )
        self.openai_compatible = bool(openai_compatible)

    def _programmatic_evaluation_kind(self) -> Optional[str]:
        return "nvagent"

    def _prepare_invocation(
        self,
        request: GenerationRequest,
        case: DatasetCase,
    ) -> BaselineInvocation:
        tables = _table_inputs(
            case,
            manifest_source=Path(request.spec.dataset_manifest_path),
        )
        db_id = slug_identifier(case.case_id)
        dataset_root = request.output_dir / "dataset"
        database_dir = dataset_root / "databases" / db_id
        (
            copied_tables,
            normalized_instruction,
            alias_path,
        ) = _prepare_nvagent_tables(
            tables,
            database_dir,
            instruction=_case_instruction(case),
            sheet=case.payload.get("sheet"),
        )
        logs_dir = request.output_dir / "logs"
        logs_dir.mkdir()
        result_path = request.output_dir / "driver_result.json"
        config_path = request.output_dir / "baseline_input.json"
        write_json_atomic(
            config_path,
            {
                "dataset_root": str(dataset_root),
                "db_id": db_id,
                "tables": [str(path) for path in copied_tables],
                "instruction": normalized_instruction,
                "input_aliases_path": str(alias_path),
                "model": request.spec.backbone,
                "openai_compatible": bool(
                    request.spec.method_config.get(
                        "openai_compatible",
                        self.openai_compatible,
                    )
                ),
                "log_path": str(logs_dir / "nvagent.log"),
                "code_path": str(request.output_dir / "generated.py"),
                "image_path": str(request.output_dir / "figure.png"),
                "result_path": str(result_path),
            },
        )
        driver = _write_driver(request.output_dir, _NVAGENT_DRIVER)
        environment = {
            "PYTHONPATH": str(self.repo_path),
            "DYLD_FALLBACK_LIBRARY_PATH": self.environ.get(
                "NVAGENT_DYLD_FALLBACK_LIBRARY_PATH",
                "",
            ),
            "PHEROVIZ_AZURE_OPENAI_API_KEY": self.environ.get(
                "NVAGENT_AZURE_OPENAI_API_KEY",
                "",
            ),
            "PHEROVIZ_AZURE_OPENAI_ENDPOINT": self.environ.get(
                "NVAGENT_AZURE_OPENAI_ENDPOINT",
                "",
            ),
            "PHEROVIZ_OPENAI_API_VERSION": self.environ.get(
                "NVAGENT_OPENAI_API_VERSION",
                "",
            ),
        }
        return BaselineInvocation(
            argv=(self.python_executable, str(driver), str(config_path)),
            environment=environment,
            result_manifest=result_path,
            served_model=request.spec.backbone,
            entrypoint=self.definition.entrypoints["workflow"],
        )


class ChartCoderProvider(ExternalBaselineProvider):
    def __init__(
        self,
        repo_path: Path | str = _DEFAULT_BASELINE_ROOT / "ChartCoder",
        *,
        definition: Optional[BaselineDefinition] = None,
        python_executable: str = sys.executable,
        timeout_seconds: float = 1800.0,
        check_dependencies: bool = True,
        environ: Optional[Mapping[str, str]] = None,
    ) -> None:
        super().__init__(
            definition or BASELINE_REGISTRY["chartcoder"],
            repo_path,
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
            check_dependencies=check_dependencies,
            environ=environ,
        )

    def _prepare_invocation(
        self,
        request: GenerationRequest,
        case: DatasetCase,
    ) -> BaselineInvocation:
        reference = _resolve_case_file(
            case.payload.get("reference_image"),
            manifest_source=Path(request.spec.dataset_manifest_path),
            field_name="reference_image",
        )
        copied_reference = _copy_inputs(
            [reference],
            request.output_dir / "inputs",
        )[0]
        result_path = request.output_dir / "driver_result.json"
        config_path = request.output_dir / "baseline_input.json"
        write_json_atomic(
            config_path,
            {
                "reference_image": str(copied_reference),
                "instruction": str(
                    case.payload.get("instruction")
                    or "Generate Python plotting code that reconstructs this chart."
                ),
                "code_path": str(request.output_dir / "generated.py"),
                "raw_response_path": str(
                    request.output_dir / "raw_response.txt"
                ),
                "execution_log_path": str(
                    request.output_dir / "generated_code.log"
                ),
                "result_path": str(result_path),
            },
        )
        checkpoint = self._checkpoint_path()
        if checkpoint is None:
            raise ProviderExecutionError(
                "ChartCoder checkpoint is required"
            )
        driver = _write_driver(request.output_dir, _CHARTCODER_DRIVER)
        environment = {
            "PYTHONPATH": str(self.repo_path),
            "PHEROVIZ_CHARTCODER_CHECKPOINT": str(checkpoint),
        }
        return BaselineInvocation(
            argv=(self.python_executable, str(driver), str(config_path)),
            environment=environment,
            result_manifest=result_path,
            served_model=request.spec.backbone,
            entrypoint=self.definition.entrypoints["reconstruction"],
        )


_MATPLOT_DRIVER = r'''
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
workspace = Path(config["workspace"])
from agents.config import openai as api_config
api_config.API_KEY = os.environ["PHEROVIZ_BASELINE_API_KEY"]
api_config.BASE_URL = os.environ["PHEROVIZ_BASELINE_BASE_URL"]
logging.basicConfig(
    level=logging.INFO,
    filename=str(workspace / "adapter.log"),
    filemode="w",
)

mode = config["mode"]
entry_file = "workflow.py" if mode == "workflow" else "one_time_generate.py"
entry_path = Path(os.environ["PYTHONPATH"].split(os.pathsep)[0]) / entry_file
sys.argv = [str(entry_path)]
spec = importlib.util.spec_from_file_location("external_matplot_entry", entry_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if mode == "workflow":
    module.args.model_type = config["model"]
    module.args.visual_refine = config["visual_refine"]
    module.mainworkflow(
        config["expert_instruction"],
        config["simple_instruction"],
        workspace=str(workspace),
    )
else:
    module.mainworkflow(
        config["expert_instruction"],
        config["simple_instruction"],
        workspace=str(workspace),
        model_type=config["model"],
        no_sysprompt=config["no_sysprompt"],
    )

code_files = sorted(workspace.glob("code_action_*.py"))
image_files = sorted(workspace.glob("*.png"))
if not code_files or not image_files:
    missing = []
    if not code_files:
        missing.append("code")
    if not image_files:
        missing.append("render")
    Path(config["result_path"]).write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "status": "failed",
                "adapter_invocation_id": config["adapter_invocation_id"],
                "failure": {
                    "source": "adapter_output_contract",
                    "code": "missing_code_or_render",
                    "attribution": "method",
                    "missing_artifacts": sorted(missing),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    raise RuntimeError("MatPlotAgent did not produce both code and PNG output")
Path(config["result_path"]).write_text(
    json.dumps(
        {
            "schema_version": "1.0",
            "status": "completed",
            "adapter_invocation_id": config["adapter_invocation_id"],
            "code_path": str(code_files[-1]),
            "image_path": str(image_files[-1]),
            "log_path": str(workspace / "adapter.log"),
        }
    ),
    encoding="utf-8",
)
'''


_NVAGENT_DRIVER = r'''
import json
import os
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
import core.api_config as api_config
api_config.API_KEY = os.environ["PHEROVIZ_AZURE_OPENAI_API_KEY"]
api_config.AZURE_OPENAI_ENDPOINT = os.environ["PHEROVIZ_AZURE_OPENAI_ENDPOINT"]
api_config.OPENAI_API_VERSION = os.environ["PHEROVIZ_OPENAI_API_VERSION"]
api_config.MODEL_NAME = config["model"]
os.environ["AZURE_OPENAI_API_KEY"] = api_config.API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = api_config.AZURE_OPENAI_ENDPOINT
os.environ["OPENAI_API_VERSION"] = api_config.OPENAI_API_VERSION

from core import llm
llm.MODEL_NAME = config["model"]
if config.get("openai_compatible"):
    from openai import OpenAI
    base_url = api_config.AZURE_OPENAI_ENDPOINT.rstrip("/") + "/"
    llm.AzureOpenAI = lambda *args, **kwargs: OpenAI(
        api_key=api_config.API_KEY,
        base_url=base_url,
    )
import core.const as core_const
if not hasattr(core_const, "PROVIDER_NAME"):
    core_const.PROVIDER_NAME = core_const.PROCESSOR_NAME
from core.chat_manager import ChatManager
from core.const import SYSTEM_NAME

manager = ChatManager(
    data_path=config["dataset_root"],
    log_path=config["log_path"],
)
message = {
    "db_id": config["db_id"],
    "query": config["instruction"],
    "tables": config["tables"],
    "send_to": SYSTEM_NAME,
    "library": "matplotlib",
}
code = manager.start(message)
Path(config["code_path"]).write_text(code, encoding="utf-8")
result = manager.execute_to_svg(code, log_name=config["image_path"])
if not result.status:
    raise RuntimeError(str(result.error_msg))
Path(config["result_path"]).write_text(
    json.dumps(
        {
            "code_path": config["code_path"],
            "image_path": config["image_path"],
            "log_path": config["log_path"],
            "input_aliases_path": config["input_aliases_path"],
        }
    ),
    encoding="utf-8",
)
'''


_CHARTCODER_DRIVER = r'''
import json
import os
import re
import subprocess
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
import inference
original_loader = inference.load_pretrained_model
checkpoint = os.environ["PHEROVIZ_CHARTCODER_CHECKPOINT"]

def pinned_loader(pretrained, model_base, model_name, *args, **kwargs):
    return original_loader(checkpoint, model_base, model_name, *args, **kwargs)

inference.load_pretrained_model = pinned_loader
model = inference.ChartCoder()
response = model.generate(config["instruction"], config["reference_image"])
Path(config["raw_response_path"]).write_text(response, encoding="utf-8")
blocks = re.findall(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
code = blocks[-1].strip() if blocks else response.strip()
if not code:
    raise RuntimeError("ChartCoder returned empty code")
Path(config["code_path"]).write_text(code, encoding="utf-8")
execution = subprocess.run(
    [sys.executable, config["code_path"]],
    cwd=Path(config["code_path"]).parent,
    capture_output=True,
    text=True,
    check=False,
)
Path(config["execution_log_path"]).write_text(
    execution.stdout + "\n" + execution.stderr,
    encoding="utf-8",
)
if execution.returncode != 0:
    raise RuntimeError(
        f"Generated ChartCoder code failed with exit code {execution.returncode}"
    )
workspace = Path(config["code_path"]).parent
reference = Path(config["reference_image"]).resolve()
images = [
    path for path in sorted(workspace.glob("*.png"))
    if path.resolve() != reference
]
if not images:
    raise RuntimeError("Generated ChartCoder code produced no PNG")
Path(config["result_path"]).write_text(
    json.dumps(
        {
            "code_path": config["code_path"],
            "image_path": str(images[-1]),
            "log_path": config["execution_log_path"],
            "raw_response_path": config["raw_response_path"],
        }
    ),
    encoding="utf-8",
)
'''
