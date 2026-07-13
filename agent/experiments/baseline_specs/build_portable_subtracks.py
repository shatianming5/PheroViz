from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml


BASELINES = ("matplotagent", "nvagent")
PATH_MARKERS = ("agent", "nature_download")
_MODULE_PROBE_CACHE: set[tuple[str, tuple[str, ...]]] = set()


class MaterializationError(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise MaterializationError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _discover_source_repo_root(parent: Mapping[str, Any]) -> Path:
    cases = parent.get("cases")
    if not isinstance(cases, list):
        raise MaterializationError("parent cases are invalid")
    candidates: set[Path] = set()
    for case in cases:
        if not isinstance(case, Mapping):
            continue
        raw = case.get("data_path")
        if not isinstance(raw, str) or not Path(raw).is_absolute():
            continue
        parts = Path(raw).parts
        for marker in PATH_MARKERS:
            if marker in parts:
                candidates.add(Path(*parts[: parts.index(marker)]))
                break
    if len(candidates) != 1:
        raise MaterializationError(
            "cannot determine one parent manifest repository root"
        )
    return next(iter(candidates))


def _remap_path(value: str, *, source_root: Path, target_root: Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        return value
    try:
        relative = path.relative_to(source_root)
    except ValueError as exc:
        raise MaterializationError(
            f"absolute path is outside parent repository root: {value}"
        ) from exc
    remapped = (target_root / relative).resolve()
    try:
        remapped.relative_to(target_root)
    except ValueError as exc:
        raise MaterializationError(
            f"remapped path escaped runtime repository root: {value}"
        ) from exc
    return str(remapped)


def _remap_tree(value: Any, *, source_root: Path, target_root: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: _remap_tree(
                child,
                source_root=source_root,
                target_root=target_root,
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _remap_tree(
                child,
                source_root=source_root,
                target_root=target_root,
            )
            for child in value
        ]
    if isinstance(value, str) and Path(value).is_absolute():
        return _remap_path(
            value,
            source_root=source_root,
            target_root=target_root,
        )
    return value


def _compatible(case: Mapping[str, Any], spec: Mapping[str, Any]) -> bool:
    required = spec["selection"]["required_case_metadata"]
    instruction = next(
        (
            case.get(name)
            for name in required["nonempty_instruction_fields"]
            if isinstance(case.get(name), str) and str(case[name]).strip()
        ),
        None,
    )
    expectation = case.get("evaluation_expectation")
    return (
        case.get("split") == required["split"]
        and case.get("panel_count") == required["panel_count"]
        and isinstance(case.get("data_path"), str)
        and Path(str(case["data_path"])).suffix.lower()
        in set(required["data_path_suffixes"])
        and instruction is not None
        and isinstance(expectation, Mapping)
        and expectation.get("panel_groups") == []
    )


def _git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_python_modules(python: Path, modules: Sequence[str]) -> None:
    cache_key = (str(python), tuple(modules))
    if cache_key in _MODULE_PROBE_CACHE:
        return
    probe = subprocess.run(
        [
            str(python),
            "-c",
            ";".join(f"import {module}" for module in modules),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        detail = probe.stderr.strip().splitlines()
        raise MaterializationError(
            "Python dependency preflight failed: "
            + (detail[-1] if detail else "unknown import error")
        )
    _MODULE_PROBE_CACHE.add(cache_key)


def _validate_runtime(
    spec: Mapping[str, Any],
    *,
    repo_root: Path,
    workspace_root: Path,
) -> tuple[Path, Path]:
    runtime = spec["external_runtime"]
    repo_binding = runtime["repo"]
    python_binding = runtime["python"]
    checkout = (
        workspace_root / str(repo_binding["relative_path"])
    ).resolve()
    python = (
        workspace_root / str(python_binding["relative_path"])
    ).absolute()
    if not checkout.is_dir() or _git_commit(checkout) != repo_binding["commit"]:
        raise MaterializationError("external checkout binding mismatch")
    if not python.is_file():
        raise MaterializationError("external Python binding is missing")
    _require_python_modules(python, runtime["dependency_modules"])
    _require_python_modules(
        Path(sys.executable),
        runtime["agent_evaluator_dependency_modules"],
    )
    audit = runtime["audit"]
    audit_path = repo_root / str(audit["path"])
    if _file_hash(audit_path) != audit["sha256"]:
        raise MaterializationError("external audit binding mismatch")
    for binding in runtime["evaluator_files"]:
        path = repo_root / str(binding["path"])
        if _file_hash(path) != binding["sha256"]:
            raise MaterializationError("external evaluator binding mismatch")
    return checkout, python


def _materialize_manifest(
    spec: Mapping[str, Any],
    *,
    spec_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    builder = spec["builder_contract"]
    builder_path = repo_root / str(builder["builder_path"])
    if _file_hash(builder_path) != builder["builder_sha256"]:
        raise MaterializationError("portable builder binding mismatch")
    parent_binding = spec["parent_manifest"]
    parent_path = repo_root / str(parent_binding["path"])
    if _file_hash(parent_path) != parent_binding["sha256"]:
        raise MaterializationError("parent manifest SHA-256 mismatch")
    parent = _read_json(parent_path)
    if parent.get("manifest_hash") != parent_binding["manifest_hash"]:
        raise MaterializationError("parent semantic manifest hash mismatch")
    source_root = _discover_source_repo_root(parent)
    parent_cases = parent.get("cases")
    if not isinstance(parent_cases, list):
        raise MaterializationError("parent cases are invalid")
    selected = [
        case
        for case in parent_cases
        if isinstance(case, dict) and _compatible(case, spec)
    ]
    selected_ids = [case["case_id"] for case in selected]
    declared = spec["selection"]
    if (
        selected_ids != declared["selected_case_ids"]
        or _json_hash(sorted(selected_ids))
        != declared["selected_case_set_sha256"]
    ):
        raise MaterializationError("compatibility case selection changed")
    parent_bindings = [
        {
            "case_id": case["case_id"],
            "parent_case_sha256": _json_hash(case),
        }
        for case in selected
    ]
    if _json_hash(parent_bindings) != declared["selected_parent_cases_sha256"]:
        raise MaterializationError("selected parent case bindings changed")

    input_track = declared["declared_field_addition"]["input_track"]
    derived_cases: list[dict[str, Any]] = []
    case_bindings: list[dict[str, Any]] = []
    for parent_case in selected:
        remapped_parent = _remap_tree(
            parent_case,
            source_root=source_root,
            target_root=repo_root,
        )
        derived = deepcopy(remapped_parent)
        derived["input_track"] = input_track
        derived_cases.append(derived)
        case_bindings.append(
            {
                "case_id": parent_case["case_id"],
                "parent_case_sha256": _json_hash(parent_case),
                "runtime_case_sha256": _json_hash(derived),
                "source_sha256": parent_case["data_sha256"],
                "instruction_sha256": _json_hash(parent_case["user_goal"]),
                "expectation_sha256": _json_hash(
                    parent_case["evaluation_expectation"]
                ),
                "runtime_data_path": derived["data_path"],
            }
        )

    provenance = _remap_tree(
        parent["provenance"],
        source_root=source_root,
        target_root=repo_root,
    )
    provenance["source_binding_hash"] = _json_hash(
        provenance["source_binding"]
    )
    derivation = {
        "schema_version": "1.0",
        "compatibility_id": spec["compatibility_id"],
        "spec_path": str(spec_path.relative_to(repo_root)),
        "spec_sha256": _file_hash(spec_path),
        "parent_manifest_path": str(parent_path.relative_to(repo_root)),
        "parent_manifest_sha256": parent_binding["sha256"],
        "parent_manifest_hash": parent_binding["manifest_hash"],
        "parent_manifest_source_root_sha256": hashlib.sha256(
            str(source_root).encode("utf-8")
        ).hexdigest(),
        "runtime_repo_root": str(repo_root),
        "runtime_root_remap": "parent repository-relative suffix preserved",
        "declared_field_addition": {"input_track": input_track},
        "allowed_case_mutations": ["input_track", "runtime_root_remap"],
        "selected_case_ids": selected_ids,
        "selected_case_set_sha256": declared["selected_case_set_sha256"],
        "case_bindings": case_bindings,
        "forbidden_inputs": declared["forbidden_inputs"],
        "license_status": "not_declared",
    }
    derivation["derivation_hash"] = _json_hash(derivation)
    provenance["compatibility_derivation"] = derivation
    manifest = {
        "schema_version": parent["schema_version"],
        "cases": derived_cases,
        "provenance": provenance,
    }
    manifest["manifest_hash"] = _json_hash(manifest)
    return manifest


def materialize_subtracks(
    *,
    repo_root: Path,
    workspace_root: Path,
    output_root: Path,
    require_clean: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve(strict=True)
    workspace_root = workspace_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve()
    code_commit = _git_commit(repo_root)
    code_dirty = bool(
        subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if require_clean and code_dirty:
        raise MaterializationError("repository is dirty")
    output_root.mkdir(parents=True, exist_ok=True)
    script_root = repo_root / "agent" / "experiments" / "baseline_specs"
    matrix_root = repo_root / "agent" / "experiments" / "matrices"

    outputs: dict[str, Any] = {}
    for baseline in BASELINES:
        spec_path = script_root / f"{baseline}-single-test-renderable-v1.json"
        template_path = (
            matrix_root / f"baseline_{baseline}_single_renderable_v1.yaml"
        )
        spec = _read_json(spec_path)
        checkout, python = _validate_runtime(
            spec,
            repo_root=repo_root,
            workspace_root=workspace_root,
        )
        manifest = _materialize_manifest(
            spec,
            spec_path=spec_path,
            repo_root=repo_root,
        )
        manifest_path = output_root / f"{baseline}.manifest.json"
        _write_json(manifest_path, manifest)
        manifest_sha256 = _file_hash(manifest_path)

        matrix = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        if not isinstance(matrix, dict) or not matrix.get("portable_template"):
            raise MaterializationError("matrix is not a portable template")
        matrix["dataset_manifest"] = str(manifest_path)
        matrix["dataset_manifest_sha256"] = manifest_sha256
        matrix["artifact_root"] = str(
            repo_root
            / "agent"
            / "experiments"
            / "runs"
            / "production"
            / matrix["portable_template"]["artifact_root_name"]
        )
        matrix["repo_root"] = str(repo_root)
        for method in matrix["methods"]:
            options = method.get("provider_options") or {}
            if method["name"].endswith("_native"):
                options["repo_path"] = str(checkout)
                options["python_executable"] = str(python)
            else:
                options["manifest_data_root"] = str(repo_root)
            method["provider_options"] = options
        matrix["compatibility_subtrack"]["materialized_manifest_sha256"] = (
            manifest_sha256
        )
        matrix["compatibility_subtrack"]["runtime_repo_root"] = str(repo_root)
        matrix_path = output_root / f"{baseline}.matrix.yaml"
        matrix_path.write_text(
            yaml.safe_dump(matrix, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        outputs[baseline] = {
            "spec": str(spec_path),
            "spec_sha256": _file_hash(spec_path),
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "matrix": str(matrix_path),
            "matrix_sha256": _file_hash(matrix_path),
            "cases": len(manifest["cases"]),
            "input_track": spec["public_interface"]["input_track"],
        }

    summary = {
        "schema_version": "1.0",
        "repo_root": str(repo_root),
        "workspace_root": str(workspace_root),
        "output_root": str(output_root),
        "code_commit": code_commit,
        "code_dirty": code_dirty,
        "subtracks": outputs,
        "model_calls": 0,
    }
    summary["summary_hash"] = _json_hash(summary)
    _write_json(output_root / "materialization_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    default_repo = Path(__file__).resolve().parents[3]
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=default_repo.parent,
    )
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            default_repo
            / "agent"
            / "experiments"
            / "runs"
            / "preflight"
            / "external_baseline_subtracks"
        ),
    )
    args = parser.parse_args(argv)
    summary = materialize_subtracks(
        repo_root=args.repo_root,
        workspace_root=args.workspace_root,
        output_root=args.out,
        require_clean=args.require_clean,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
