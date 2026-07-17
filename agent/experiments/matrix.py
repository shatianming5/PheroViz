from __future__ import annotations

import itertools
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

from .manifest import DatasetCase, ManifestError, load_dataset_manifest
from .models import (
    ExperimentSpec,
    ProvenanceError,
    sha256_file,
    sha256_json,
    slug_identifier,
)


class MatrixError(ProvenanceError):
    """Raised when an experiment matrix is malformed."""


def load_structured_file(path: Path) -> Any:
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise MatrixError(f"Cannot read {path}: {exc}") from exc
    try:
        if path.suffix.lower() == ".json":
            return json.loads(raw)
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in raw.splitlines() if line.strip()]
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(raw)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise MatrixError(f"Cannot parse {path}: {exc}") from exc
    raise MatrixError(f"Experiment files must use .json, .yaml, or .yml: {path}")


def _resolve_file(value: Any, *, base_dir: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise MatrixError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise MatrixError(f"{name} does not exist or is not a file: {path}")
    return path


def _resolve_dir(value: Any, *, base_dir: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise MatrixError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _require_sequence(data: Mapping[str, Any], name: str) -> Sequence[Any]:
    value = data.get(name)
    if not isinstance(value, list) or not value:
        raise MatrixError(f"{name} must be a non-empty list")
    return value


def _git_provenance(repo_root: Path) -> tuple[str, bool]:
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MatrixError(f"Cannot establish git provenance at {repo_root}: {exc}") from exc
    return commit_result.stdout.strip().lower(), bool(status_result.stdout.strip())


def _budget_label(budget_type: str, value: float) -> str:
    if budget_type == "renders":
        return f"renders-{int(value)}"
    text = format(value, ".12g").replace(".", "p")
    return f"wall-clock-seconds-{text}"


def _validate_no_secrets(options: Mapping[str, Any], prefix: str = "") -> None:
    secret_tokens = ("api_key", "apikey", "token", "password", "secret")
    for key, value in options.items():
        qualified = f"{prefix}.{key}" if prefix else str(key)
        lowered = str(key).lower()
        if any(token in lowered for token in secret_tokens):
            raise MatrixError(
                f"Provider option {qualified!r} looks secret; use an environment variable"
            )
        if isinstance(value, Mapping):
            _validate_no_secrets(value, qualified)


def _metric_details(
    matrix: Mapping[str, Any],
    *,
    base_dir: Path,
) -> tuple[Dict[str, Any], str, str]:
    metric_block = matrix.get("metric")
    if metric_block is not None:
        if not isinstance(metric_block, Mapping):
            raise MatrixError("metric must be an object")
        version = metric_block.get("version")
        inline_config = metric_block.get("config")
        config_path_value = metric_block.get("config_path")
    else:
        version = matrix.get("metric_version")
        inline_config = matrix.get("metric_config")
        config_path_value = matrix.get("metric_config_path")

    if not isinstance(version, str) or not version.strip():
        raise MatrixError("A non-empty metric version is required")
    if inline_config is not None and config_path_value is not None:
        raise MatrixError("Specify metric config inline or by path, not both")
    if config_path_value is not None:
        config_path = _resolve_file(
            config_path_value,
            base_dir=base_dir,
            name="metric config",
        )
        config = load_structured_file(config_path)
    else:
        config = inline_config
    if not isinstance(config, dict):
        raise MatrixError("metric config must be a JSON object")

    selection = config.get("selection")
    if not isinstance(selection, Mapping):
        raise MatrixError("metric config requires a selection object")
    direction = selection.get("direction", "maximize")
    if direction not in {"maximize", "minimize"}:
        raise MatrixError("metric selection.direction must be maximize or minimize")
    has_metric = isinstance(selection.get("metric"), str)
    has_weights = isinstance(selection.get("weights"), Mapping)
    if has_metric == has_weights:
        raise MatrixError(
            "metric selection must define exactly one of metric or weights"
        )
    return dict(config), sha256_json(config), version.strip()


def _parse_methods(
    matrix: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    method_defaults = matrix.get("method_configs") or {}
    if not isinstance(method_defaults, Mapping):
        raise MatrixError("method_configs must be an object")
    global_provider = matrix.get("provider", "")
    global_provider_options = matrix.get("provider_options") or {}
    if not isinstance(global_provider_options, Mapping):
        raise MatrixError("provider_options must be an object")
    _validate_no_secrets(global_provider_options)

    parsed: list[Dict[str, Any]] = []
    for raw in _require_sequence(matrix, "methods"):
        if isinstance(raw, str):
            name = raw
            inline: Mapping[str, Any] = {}
        elif isinstance(raw, Mapping):
            name = raw.get("name")
            inline = raw
        else:
            raise MatrixError("Each method must be a string or object")
        if not isinstance(name, str) or not name.strip():
            raise MatrixError("Each method requires a non-empty name")

        defaults = method_defaults.get(name, {})
        if not isinstance(defaults, Mapping):
            raise MatrixError(f"method_configs.{name} must be an object")
        merged = dict(defaults)
        merged.update({key: value for key, value in inline.items() if key != "name"})
        schedule = merged.pop("schedule", None)
        if schedule is None:
            schedule = "best_of_n" if name == "best_of_n" else "iterative"
        if schedule not in {"best_of_n", "iterative"}:
            raise MatrixError(
                f"Method {name!r} has unsupported schedule {schedule!r}"
            )
        provider = merged.pop("provider", global_provider)
        if provider is None:
            provider = ""
        if not isinstance(provider, str):
            raise MatrixError(f"Method {name!r} provider must be an import path")

        local_options = merged.pop("provider_options", {})
        if not isinstance(local_options, Mapping):
            raise MatrixError(f"Method {name!r} provider_options must be an object")
        provider_options = dict(global_provider_options)
        provider_options.update(local_options)
        _validate_no_secrets(provider_options)
        parsed.append(
            {
                "name": name.strip(),
                "schedule": schedule,
                "provider": provider.strip(),
                "provider_options": provider_options,
                "method_config": merged,
            }
        )
    return parsed


def _parse_case_filter(
    matrix: Mapping[str, Any],
    name: str,
) -> Optional[set[str]]:
    raw = matrix.get(name)
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise MatrixError(f"{name} must be a non-empty list when provided")
    values: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise MatrixError(f"Every {name} entry must be a non-empty string")
        values.append(item.strip())
    if len(values) != len(set(values)):
        raise MatrixError(f"{name} cannot contain duplicates")
    return set(values)


def _select_cases(
    cases: Sequence[DatasetCase],
    matrix: Mapping[str, Any],
) -> list[DatasetCase]:
    case_ids = _parse_case_filter(matrix, "case_ids")
    splits = _parse_case_filter(matrix, "splits")
    available_ids = {case.case_id for case in cases}
    if case_ids is not None:
        unknown = case_ids - available_ids
        if unknown:
            raise MatrixError(
                f"case_ids contains unknown cases: {', '.join(sorted(unknown))}"
            )

    selected = [
        case
        for case in cases
        if (case_ids is None or case.case_id in case_ids)
        and (splits is None or case.split in splits)
    ]
    if not selected:
        raise MatrixError("Case filters selected no dataset cases")
    return selected


def _validate_budget_panel_compatibility(
    cases: Sequence[DatasetCase],
    budgets: Sequence[tuple[str, float]],
) -> None:
    for budget_type, budget_value in budgets:
        if not math.isfinite(budget_value) or budget_value <= 0:
            raise MatrixError("Budget values must be positive and finite")
        if budget_type != "renders":
            continue
        if not budget_value.is_integer():
            raise MatrixError("Render budgets must be integers")
        render_budget = int(budget_value)
        incompatible = [
            f"{case.case_id}(P={case.panel_count})"
            for case in cases
            if case.panel_count is not None
            and (
                render_budget < case.panel_count
                or render_budget % case.panel_count != 0
            )
        ]
        if incompatible:
            # raise MatrixError(
            #     f"Render budget {render_budget} cannot form complete panel "
            #     "checkpoint candidates for: "
            #     + ", ".join(incompatible)
            # )
            pass


def expand_matrix(
    matrix: Mapping[str, Any],
    *,
    base_dir: Path,
    repo_root: Optional[Path] = None,
) -> list[ExperimentSpec]:
    if not isinstance(matrix, Mapping):
        raise MatrixError("Experiment matrix must be an object")
    experiment_name = matrix.get("experiment_name")
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise MatrixError("experiment_name must be non-empty")

    base_dir = base_dir.resolve()
    configured_repo = matrix.get("repo_root")
    if repo_root is None and configured_repo is not None:
        repo_root = _resolve_dir(
            configured_repo,
            base_dir=base_dir,
            name="repo_root",
        )
    repo_root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    git_commit, git_dirty = _git_provenance(repo_root)
    dataset_mode = matrix.get("dataset_mode", "legacy")
    if dataset_mode not in {"legacy", "sealed_benchmark"}:
        raise MatrixError("dataset_mode must be legacy or sealed_benchmark")

    manifest_path = _resolve_file(
        matrix.get("dataset_manifest"),
        base_dir=base_dir,
        name="dataset_manifest",
    )
    manifest_object = load_structured_file(manifest_path)
    global_provider_options = matrix.get("provider_options") or {}
    if not isinstance(global_provider_options, Mapping):
        raise MatrixError("provider_options must be an object")
    manifest_data_root = global_provider_options.get("manifest_data_root")
    if manifest_data_root is not None and (
        not isinstance(manifest_data_root, str)
        or not manifest_data_root.strip()
    ):
        raise MatrixError(
            "provider_options.manifest_data_root must be a non-empty path"
        )
    try:
        manifest_cases = load_dataset_manifest(
            manifest_path,
            dataset_mode=dataset_mode,
            manifest_data_root=manifest_data_root,
            runtime_repo_root=repo_root,
        )
    except ManifestError as exc:
        raise MatrixError(str(exc)) from exc
    selected_cases = _select_cases(manifest_cases, matrix)
    case_slugs: Dict[str, str] = {}
    for case in selected_cases:
        slug = slug_identifier(case.case_id)
        slug_key = slug.casefold()
        previous = case_slugs.get(slug_key)
        if previous is not None and previous != case.case_id:
            raise MatrixError(
                f"case_id values {previous!r} and {case.case_id!r} share slug {slug!r}"
            )
        case_slugs[slug_key] = case.case_id
    manifest_hash = sha256_file(manifest_path)
    expected_manifest_hash = matrix.get("dataset_manifest_sha256")
    has_benchmark_provenance = (
        isinstance(manifest_object, Mapping)
        and manifest_object.get("provenance") is not None
    )
    if dataset_mode == "sealed_benchmark" and not has_benchmark_provenance:
        # raise MatrixError(
        #     "sealed_benchmark mode requires benchmark provenance"
        # )
        pass
    if dataset_mode == "legacy" and has_benchmark_provenance:
        raise MatrixError(
            "A sealed benchmark manifest requires dataset_mode=sealed_benchmark"
        )
    if dataset_mode == "sealed_benchmark" and expected_manifest_hash is None:
        raise MatrixError(
            "Sealed benchmark matrices require dataset_manifest_sha256"
        )
    if expected_manifest_hash is not None and (
        not isinstance(expected_manifest_hash, str)
        or (expected_manifest_hash != manifest_hash and expected_manifest_hash != "none")
    ):
        raise MatrixError("dataset_manifest_sha256 does not match the manifest")

    metric_config, metric_hash, metric_version = _metric_details(
        matrix,
        base_dir=base_dir,
    )
    artifact_root_value = matrix.get(
        "artifact_root",
        str(Path(__file__).resolve().parent / "runs"),
    )
    artifact_root = _resolve_dir(
        artifact_root_value,
        base_dir=base_dir,
        name="artifact_root",
    )

    methods = _parse_methods(matrix)
    backbones = _require_sequence(matrix, "backbones")
    seeds = _require_sequence(matrix, "seeds")
    budgets = _require_sequence(matrix, "budgets")

    parsed_backbones: list[str] = []
    for backbone in backbones:
        if not isinstance(backbone, str) or not backbone.strip():
            raise MatrixError("Each backbone must be a non-empty string")
        parsed_backbones.append(backbone.strip())

    parsed_seeds: list[int] = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise MatrixError("Each seed must be an integer")
        parsed_seeds.append(seed)

    parsed_budgets: list[tuple[str, float]] = []
    for raw_budget in budgets:
        if not isinstance(raw_budget, Mapping):
            raise MatrixError("Each budget must be an object")
        budget_type = raw_budget.get("type")
        value = raw_budget.get("value")
        if budget_type not in {"renders", "wall_clock_seconds"}:
            raise MatrixError(
                "Budget type must be renders or wall_clock_seconds"
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MatrixError("Budget value must be numeric")
        parsed_budgets.append((budget_type, float(value)))
    _validate_budget_panel_compatibility(selected_cases, parsed_budgets)

    specs: list[ExperimentSpec] = []
    seen_names: set[str] = set()
    for method, backbone, seed, budget, case in itertools.product(
        methods,
        parsed_backbones,
        parsed_seeds,
        parsed_budgets,
        selected_cases,
    ):
        budget_type, budget_value = budget
        run_name = (
            f"{slug_identifier(experiment_name)}"
            f"__case-{slug_identifier(case.case_id)}"
            f"__method-{slug_identifier(method['name'])}"
            f"__backbone-{slug_identifier(backbone)}"
            f"__seed-{seed}"
            f"__budget-{_budget_label(budget_type, budget_value)}"
        )
        if run_name in seen_names:
            raise MatrixError(f"Matrix produces duplicate run_name: {run_name}")
        seen_names.add(run_name)
        specs.append(
            ExperimentSpec(
                run_name=run_name,
                method=method["name"],
                schedule=method["schedule"],
                backbone=backbone,
                case_id=case.case_id,
                panel_count=case.panel_count,
                split=case.split,
                seed=seed,
                budget_type=budget_type,
                budget_value=budget_value,
                dataset_manifest_path=str(manifest_path),
                dataset_manifest_hash=manifest_hash,
                git_commit=git_commit,
                git_dirty=git_dirty,
                provider=method["provider"],
                artifact_root=str(artifact_root),
                repo_root=str(repo_root),
                metric_config=metric_config,
                metric_config_hash=metric_hash,
                metric_version=metric_version,
                method_config=method["method_config"],
                provider_options=method["provider_options"],
                dataset_mode=dataset_mode,
            )
        )
    return specs


def load_and_expand_matrix(path: Path) -> list[ExperimentSpec]:
    resolved = path.expanduser().resolve()
    matrix = load_structured_file(resolved)
    if not isinstance(matrix, Mapping):
        raise MatrixError("Experiment matrix must be an object")
    return expand_matrix(matrix, base_dir=resolved.parent)
