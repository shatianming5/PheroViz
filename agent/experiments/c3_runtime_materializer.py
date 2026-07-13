from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

import yaml

from .c3_matrix_builder import (
    AUTHORITATIVE_RENDERABLE_MANIFEST_SHA256,
    C3MatrixError,
    verify_c3_matrix,
)
from .manifest import ManifestError, normalize_manifest_data_root
from .matrix import expand_matrix
from .models import sha256_file, sha256_json


RUNTIME_VERSION = "1.1"
_SECRET_TOKENS = ("api_key", "apikey", "password", "secret", "auth_token")


class C3RuntimeMaterializationError(C3MatrixError):
    """Raised when a portable C3 template cannot be bound to a runtime root."""


def _normalize_original_manifest_data_root(value: Path) -> Path:
    try:
        return normalize_manifest_data_root(value)
    except ManifestError as exc:
        raise C3RuntimeMaterializationError(
            f"Invalid original_manifest_data_root: {exc}"
        ) from exc


def _git_output(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise C3RuntimeMaterializationError(
            f"Cannot establish runtime git provenance: {exc}"
        ) from exc


def _inside(path: Path, root: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve()
        resolved.relative_to(root.resolve(strict=True))
        return resolved
    except (OSError, ValueError) as exc:
        raise C3RuntimeMaterializationError(
            f"{label} must remain inside runtime_repo_root"
        ) from exc


def _require_ignored(repo_root: Path, path: Path, label: str) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", str(path)],
        cwd=repo_root,
        check=False,
    )
    if result.returncode != 0:
        raise C3RuntimeMaterializationError(
            f"{label} must be covered by repository ignore policy"
        )


def _assert_secret_free(value: Any, path: str = "runtime_matrix") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).casefold()
            if any(token in lowered for token in _SECRET_TOKENS):
                raise C3RuntimeMaterializationError(
                    f"Secret-like key is forbidden: {path}.{key}"
                )
            _assert_secret_free(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_secret_free(item, f"{path}[{index}]")


def _runtime_contract(matrix: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in matrix.items()
        if key != "runtime_materialization"
    }


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise C3RuntimeMaterializationError(
            f"Cannot read runtime matrix: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise C3RuntimeMaterializationError("Runtime matrix must be an object")
    return payload


def _portable_projection(
    runtime_matrix: Mapping[str, Any],
    template: Mapping[str, Any],
) -> dict[str, Any]:
    projected = deepcopy(dict(runtime_matrix))
    projected.pop("runtime_materialization", None)
    projected.pop("repo_root", None)
    projected.pop("provider_options", None)
    projected["dataset_manifest"] = template["dataset_manifest"]
    projected["artifact_root"] = template["artifact_root"]
    return projected


def verify_runtime_c3_matrix(
    runtime_matrix_path: Path,
    *,
    template_path: Path,
    runtime_repo_root: Path,
    original_manifest_data_root: Path,
    expected_manifest_sha256: str = (
        AUTHORITATIVE_RENDERABLE_MANIFEST_SHA256
    ),
) -> dict[str, Any]:
    runtime_repo_root = runtime_repo_root.expanduser().resolve(strict=True)
    runtime_matrix_path = _inside(
        runtime_matrix_path,
        runtime_repo_root,
        "runtime matrix",
    )
    template_path = _inside(
        template_path,
        runtime_repo_root,
        "portable template",
    )
    original_manifest_data_root = _normalize_original_manifest_data_root(
        original_manifest_data_root
    )
    template_result = verify_c3_matrix(
        template_path,
        repo_root=runtime_repo_root,
        manifest_data_root=original_manifest_data_root,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    template = _read_yaml(template_path)
    runtime = _read_yaml(runtime_matrix_path)
    _assert_secret_free(runtime)
    materialization = runtime.get("runtime_materialization")
    if not isinstance(materialization, Mapping):
        raise C3RuntimeMaterializationError(
            "Runtime matrix lacks materialization provenance"
        )
    expected_template_path = template_path.relative_to(
        runtime_repo_root
    ).as_posix()
    if (
        materialization.get("schema_version") != RUNTIME_VERSION
        or materialization.get("portable_template_path")
        != expected_template_path
        or materialization.get("portable_template_sha256")
        != sha256_file(template_path)
        or materialization.get("portable_contract_hash")
        != template_result["contract_hash"]
        or _portable_projection(runtime, template) != template
    ):
        raise C3RuntimeMaterializationError(
            "Runtime matrix no longer matches the portable template"
        )
    runtime_manifest = Path(str(runtime.get("dataset_manifest") or ""))
    runtime_artifact_root = Path(str(runtime.get("artifact_root") or ""))
    declared_repo_root = Path(str(runtime.get("repo_root") or ""))
    provider_options = runtime.get("provider_options")
    if (
        not runtime_manifest.is_absolute()
        or not runtime_artifact_root.is_absolute()
        or declared_repo_root != runtime_repo_root
        or not isinstance(provider_options, Mapping)
        or provider_options
        != {"manifest_data_root": str(original_manifest_data_root)}
    ):
        raise C3RuntimeMaterializationError(
            "Runtime matrix path remapping contract changed"
        )
    _inside(runtime_manifest, runtime_repo_root, "runtime manifest")
    _inside(runtime_artifact_root, runtime_repo_root, "runtime artifact root")
    _require_ignored(
        runtime_repo_root,
        runtime_artifact_root,
        "runtime artifact root",
    )
    if (
        materialization.get("runtime_repo_root") != str(runtime_repo_root)
        or materialization.get("original_manifest_data_root")
        != str(original_manifest_data_root)
        or materialization.get("runtime_manifest_path")
        != str(runtime_manifest)
        or materialization.get("runtime_manifest_sha256")
        != sha256_file(runtime_manifest)
        or materialization.get("runtime_artifact_root")
        != str(runtime_artifact_root)
        or materialization.get("runtime_matrix_contract_hash")
        != sha256_json(_runtime_contract(runtime))
    ):
        raise C3RuntimeMaterializationError(
            "Runtime matrix provenance binding changed"
        )
    runtime_builder = runtime_repo_root / str(
        materialization.get("runtime_builder_path") or ""
    )
    if (
        not runtime_builder.is_file()
        or materialization.get("runtime_builder_sha256")
        != sha256_file(runtime_builder)
    ):
        raise C3RuntimeMaterializationError(
            "Runtime materializer hash changed"
        )
    specs = expand_matrix(
        runtime,
        base_dir=runtime_matrix_path.parent,
        repo_root=runtime_repo_root,
    )
    commit = _git_output(runtime_repo_root, "rev-parse", "HEAD")
    if (
        materialization.get("runtime_commit") != commit
        or materialization.get("runtime_clean") is not True
        or len(specs) != template_result["run_count"]
        or len({spec.run_name for spec in specs}) != len(specs)
        or any(spec.git_dirty for spec in specs)
        or {spec.git_commit for spec in specs} != {commit}
        or {spec.repo_root for spec in specs} != {str(runtime_repo_root)}
        or {
            spec.provider_options.get("manifest_data_root")
            for spec in specs
        }
        != {str(original_manifest_data_root)}
    ):
        raise C3RuntimeMaterializationError(
            "Runtime matrix expansion provenance changed"
        )
    return {
        "run_count": len(specs),
        "unique_run_names": len({spec.run_name for spec in specs}),
        "runtime_commit": commit,
        "runtime_matrix_sha256": sha256_file(runtime_matrix_path),
        "runtime_contract_hash": materialization[
            "runtime_matrix_contract_hash"
        ],
        "specs": specs,
    }


def materialize_runtime_c3_matrix(
    *,
    template_path: Path,
    runtime_repo_root: Path,
    original_manifest_data_root: Path,
    output_root: Path,
    artifact_root: Path | None = None,
    expected_manifest_sha256: str = (
        AUTHORITATIVE_RENDERABLE_MANIFEST_SHA256
    ),
) -> dict[str, Any]:
    runtime_repo_root = runtime_repo_root.expanduser().resolve(strict=True)
    template_path = _inside(
        template_path,
        runtime_repo_root,
        "portable template",
    )
    original_manifest_data_root = _normalize_original_manifest_data_root(
        original_manifest_data_root
    )
    commit = _git_output(runtime_repo_root, "rev-parse", "HEAD")
    dirty = bool(
        _git_output(
            runtime_repo_root,
            "status",
            "--porcelain",
            "--untracked-files=all",
        )
    )
    if dirty:
        raise C3RuntimeMaterializationError("runtime repository is dirty")
    template_result = verify_c3_matrix(
        template_path,
        repo_root=runtime_repo_root,
        manifest_data_root=original_manifest_data_root,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    template = _read_yaml(template_path)
    runtime_manifest = (
        template_path.parent / str(template["dataset_manifest"])
    ).resolve(strict=True)
    output_root = _inside(
        output_root,
        runtime_repo_root,
        "runtime output root",
    )
    _require_ignored(runtime_repo_root, output_root, "runtime output root")
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_root = _inside(
        artifact_root
        or (
            runtime_repo_root
            / "agent/experiments/runs/production/"
            "c3_memory_modes_final_benchmark_v2_seed0_br6_gpt56sol"
        ),
        runtime_repo_root,
        "runtime artifact root",
    )
    _require_ignored(
        runtime_repo_root,
        artifact_root,
        "runtime artifact root",
    )
    runtime_matrix = deepcopy(template)
    runtime_matrix["dataset_manifest"] = str(runtime_manifest)
    runtime_matrix["artifact_root"] = str(artifact_root)
    runtime_matrix["repo_root"] = str(runtime_repo_root)
    runtime_matrix["provider_options"] = {
        "manifest_data_root": str(original_manifest_data_root)
    }
    builder_path = (
        runtime_repo_root
        / "agent/experiments/c3_runtime_materializer.py"
    ).resolve(strict=True)
    runtime_matrix["runtime_materialization"] = {
        "schema_version": RUNTIME_VERSION,
        "portable_template_path": template_path.relative_to(
            runtime_repo_root
        ).as_posix(),
        "portable_template_sha256": sha256_file(template_path),
        "portable_contract_hash": template_result["contract_hash"],
        "runtime_builder_path": builder_path.relative_to(
            runtime_repo_root
        ).as_posix(),
        "runtime_builder_sha256": sha256_file(builder_path),
        "runtime_commit": commit,
        "runtime_clean": not dirty,
        "runtime_repo_root": str(runtime_repo_root),
        "original_manifest_data_root": str(original_manifest_data_root),
        "runtime_manifest_path": str(runtime_manifest),
        "runtime_manifest_sha256": sha256_file(runtime_manifest),
        "runtime_artifact_root": str(artifact_root),
    }
    runtime_matrix["runtime_materialization"][
        "runtime_matrix_contract_hash"
    ] = sha256_json(_runtime_contract(runtime_matrix))
    _assert_secret_free(runtime_matrix)
    matrix_path = output_root / "c3.runtime.matrix.yaml"
    matrix_path.write_text(
        yaml.safe_dump(runtime_matrix, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    verified = verify_runtime_c3_matrix(
        matrix_path,
        template_path=template_path,
        runtime_repo_root=runtime_repo_root,
        original_manifest_data_root=original_manifest_data_root,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    entries = [
        {
            "run_name": spec.run_name,
            "spec_hash": spec.spec_hash,
            "method": spec.method,
            "case_id": spec.case_id,
            "panel_count": spec.panel_count,
            "seed": spec.seed,
        }
        for spec in verified.pop("specs")
    ]
    spec_manifest = {
        "schema_version": "1.0",
        "runtime_commit": commit,
        "runtime_clean": not dirty,
        "runtime_matrix_sha256": sha256_file(matrix_path),
        "run_count": len(entries),
        "specs": entries,
        "spec_set_hash": sha256_json(entries),
    }
    spec_manifest["manifest_hash"] = sha256_json(spec_manifest)
    spec_path = output_root / "c3.runtime.specs.json"
    spec_path.write_text(
        json.dumps(spec_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": "1.0",
        "runtime_commit": commit,
        "runtime_clean": not dirty,
        "portable_template_path": runtime_matrix["runtime_materialization"][
            "portable_template_path"
        ],
        "portable_template_sha256": sha256_file(template_path),
        "runtime_matrix_path": str(matrix_path),
        "runtime_matrix_sha256": sha256_file(matrix_path),
        "runtime_matrix_contract_hash": runtime_matrix[
            "runtime_materialization"
        ]["runtime_matrix_contract_hash"],
        "runtime_spec_manifest_path": str(spec_path),
        "runtime_spec_manifest_sha256": sha256_file(spec_path),
        "runtime_spec_manifest_hash": spec_manifest["manifest_hash"],
        "runtime_spec_set_hash": spec_manifest["spec_set_hash"],
        "run_count": len(entries),
        "runtime_manifest_sha256": sha256_file(runtime_manifest),
        "runtime_repo_root_sha256": hashlib.sha256(
            str(runtime_repo_root).encode("utf-8")
        ).hexdigest(),
        "original_manifest_data_root_sha256": hashlib.sha256(
            str(original_manifest_data_root).encode("utf-8")
        ).hexdigest(),
        "runtime_artifact_root_sha256": hashlib.sha256(
            str(artifact_root).encode("utf-8")
        ).hexdigest(),
    }
    summary["summary_hash"] = sha256_json(summary)
    summary_path = output_root / "c3.runtime.summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **verified,
        "matrix_path": str(matrix_path),
        "spec_manifest_path": str(spec_path),
        "summary_path": str(summary_path),
        "spec_manifest_hash": spec_manifest["manifest_hash"],
        "summary_hash": summary["summary_hash"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bind a portable C3 template to one clean runtime root"
    )
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--runtime-repo-root", type=Path, required=True)
    parser.add_argument("--original-manifest-data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args(argv)
    result = materialize_runtime_c3_matrix(
        template_path=args.template,
        runtime_repo_root=args.runtime_repo_root,
        original_manifest_data_root=args.original_manifest_data_root,
        output_root=args.out,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
