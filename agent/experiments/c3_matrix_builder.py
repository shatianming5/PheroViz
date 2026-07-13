from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .manifest import ManifestError, load_dataset_manifest
from .matrix import expand_matrix
from .models import sha256_file, sha256_json


BUILDER_VERSION = "1.1"
RENDER_BUDGET = 6
BACKBONE = "gpt-5.6-sol"
SEEDS = (0, 1, 2)
PROVIDER = "experiments.providers:UnifiedBenchmarkProvider"
AUTHORITATIVE_RENDERABLE_MANIFEST_SHA256 = (
    "6c6c5cb50603d899a9be0f41c9e562517c92ae15477610e9111f6504cd1757b3"
)
MEMORY_MODES = (
    ("memory_none", "none"),
    ("memory_ephemeral", "ephemeral"),
    ("memory_untyped", "untyped"),
    ("memory_constraints", "constraints"),
    ("memory_patches", "patches"),
    ("memory_full", "full"),
)
_DOI_RE = re.compile(r"^10\.\S+/\S+$")
_SECRET_TOKENS = ("api_key", "apikey", "password", "secret", "auth_token")


class C3MatrixError(ValueError):
    """Raised when a C3 matrix cannot be materialized provenance-safely."""


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
        raise C3MatrixError(f"Cannot establish git provenance: {exc}") from exc


def _repo_relative(path: Path, repo_root: Path, label: str) -> str:
    try:
        return path.resolve(strict=True).relative_to(
            repo_root.resolve(strict=True)
        ).as_posix()
    except (OSError, ValueError) as exc:
        raise C3MatrixError(f"{label} must be inside the repository") from exc


def _portable_relative(value: str, label: str) -> str:
    path = Path(value)
    if path.is_absolute() or not value.strip():
        raise C3MatrixError(f"{label} must be a non-empty relative path")
    return path.as_posix()


def _assert_portable_and_secret_free(
    value: Any,
    *,
    path: str = "matrix",
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).casefold()
            if any(token in lowered for token in _SECRET_TOKENS):
                raise C3MatrixError(f"Secret-like key is forbidden: {path}.{key}")
            _assert_portable_and_secret_free(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_portable_and_secret_free(item, path=f"{path}[{index}]")
    elif isinstance(value, str):
        if value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value):
            raise C3MatrixError(f"Absolute tracked path is forbidden: {path}")


def validate_selected_case_payloads(
    cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for case in cases:
        if case.get("split") != "test":
            continue
        panel_count = case.get("panel_count")
        if (
            isinstance(panel_count, bool)
            or not isinstance(panel_count, int)
            or panel_count < 2
        ):
            continue
        if RENDER_BUDGET % panel_count:
            raise C3MatrixError(
                f"case {case.get('case_id')!r} panel_count={panel_count} "
                f"does not divide B_R={RENDER_BUDGET}"
            )
        case_id = case.get("case_id")
        doi = case.get("doi")
        if not isinstance(case_id, str) or not case_id.strip():
            raise C3MatrixError("Selected C3 case has no case_id")
        if not isinstance(doi, str):
            raise C3MatrixError(f"Selected C3 case {case_id!r} has no DOI")
        normalized_doi = doi.strip().casefold()
        if normalized_doi != doi or not _DOI_RE.fullmatch(normalized_doi):
            raise C3MatrixError(f"Selected C3 case {case_id!r} has invalid DOI")
        selected.append(
            {
                "case_id": case_id,
                "doi": normalized_doi,
                "panel_count": panel_count,
            }
        )
    selected.sort(key=lambda item: item["case_id"])
    if sorted(item["panel_count"] for item in selected) != [2, 3, 6]:
        raise C3MatrixError(
            "C3 matrix requires exactly three test cases with P=2, P=3, and P=6"
        )
    if len({item["doi"] for item in selected}) < 2:
        raise C3MatrixError(
            "C3 matrix requires at least two independent DOI clusters"
        )
    return selected


def _contract(matrix: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in matrix.items()
        if key != "materialization"
    }


def _code_hashes(repo_root: Path) -> dict[str, str]:
    paths = (
        "agent/experiments/c3_matrix_builder.py",
        "agent/experiments/c3_runtime_materializer.py",
        "agent/experiments/matrix.py",
        "agent/experiments/manifest.py",
        "agent/experiments/providers.py",
    )
    return {
        relative: sha256_file(repo_root / relative)
        for relative in paths
    }


def _matrix_methods() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "schedule": "iterative",
            "memory_mode": memory_mode,
            "initial_generation": "model_spec",
            "render_timeout_seconds": 120,
        }
        for name, memory_mode in MEMORY_MODES
    ]


def materialize_c3_matrix(
    *,
    manifest_path: Path,
    output_path: Path,
    repo_root: Path,
    artifact_root: str = (
        "../runs/production/"
        "c3_memory_modes_final_benchmark_v2_seed0_br6_gpt56sol"
    ),
    expected_manifest_sha256: str = (
        AUTHORITATIVE_RENDERABLE_MANIFEST_SHA256
    ),
) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve(strict=True)
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    output_path = output_path.expanduser().resolve()
    if output_path.parent.resolve().is_relative_to(repo_root) is False:
        raise C3MatrixError("Matrix output must be inside the repository")
    _repo_relative(manifest_path, repo_root, "benchmark manifest")
    artifact_root = _portable_relative(artifact_root, "artifact_root")
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != expected_manifest_sha256:
        raise C3MatrixError(
            "C3 matrix must bind the authoritative renderable derivative "
            f"manifest SHA-256 {expected_manifest_sha256}, got {manifest_sha256}"
        )
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise C3MatrixError(f"Cannot read sealed benchmark manifest: {exc}") from exc
    if not isinstance(manifest_payload, dict):
        raise C3MatrixError("Sealed benchmark manifest must be an object")
    provenance = manifest_payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise C3MatrixError("Benchmark manifest lacks sealed provenance")
    source_binding_hash = provenance.get("source_binding_hash")
    if not isinstance(source_binding_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}",
        source_binding_hash,
    ):
        raise C3MatrixError("Benchmark manifest has no source_binding_hash")

    try:
        manifest_cases = load_dataset_manifest(
            manifest_path,
            dataset_mode="sealed_benchmark",
        )
    except ManifestError as exc:
        raise C3MatrixError(
            f"Benchmark manifest is not fully sealed: {exc}"
        ) from exc
    selected = validate_selected_case_payloads(
        [case.payload for case in manifest_cases]
    )
    case_ids = [item["case_id"] for item in selected]
    dois = sorted({item["doi"] for item in selected})
    manifest_relative = os.path.relpath(
        manifest_path,
        start=output_path.parent,
    )
    builder_path = (
        repo_root / "agent/experiments/c3_matrix_builder.py"
    ).resolve(strict=True)
    code_hashes = _code_hashes(repo_root)
    matrix: dict[str, Any] = {
        "experiment_name": (
            "c3-memory-modes-final-benchmark-v2-seed0-br6"
        ),
        "dataset_manifest": Path(manifest_relative).as_posix(),
        "dataset_manifest_sha256": manifest_sha256,
        "dataset_mode": "sealed_benchmark",
        "artifact_root": artifact_root,
        "provider": PROVIDER,
        "methods": _matrix_methods(),
        "backbones": [BACKBONE],
        "seeds": list(SEEDS),
        "budgets": [{"type": "renders", "value": RENDER_BUDGET}],
        "case_ids": case_ids,
        "splits": ["test"],
        "metric": {
            "version": "programmatic-v2",
            "config": {
                "selection": {
                    "metric": "data_fidelity",
                    "direction": "maximize",
                }
            },
        },
    }
    materialization = {
        "schema_version": BUILDER_VERSION,
        "outcome_inputs_read": False,
        "repository_commit": _git_output(repo_root, "rev-parse", "HEAD"),
        "builder_path": _repo_relative(
            builder_path,
            repo_root,
            "builder",
        ),
        "builder_sha256": sha256_file(builder_path),
        "code_file_hashes": code_hashes,
        "code_bundle_hash": sha256_json(code_hashes),
        "parent_manifest_path": _repo_relative(
            manifest_path,
            repo_root,
            "benchmark manifest",
        ),
        "parent_manifest_file_sha256": sha256_file(manifest_path),
        "parent_manifest_semantic_sha256": sha256_json(manifest_payload),
        "parent_source_binding_hash": source_binding_hash,
        "selected_cases": selected,
        "selected_case_count": len(selected),
        "selected_case_set_hash": sha256_json(case_ids),
        "selected_dois": dois,
        "selected_doi_count": len(dois),
        "selected_doi_set_hash": sha256_json(dois),
        "expected_run_count": len(selected) * len(SEEDS) * len(MEMORY_MODES),
        "contract_hash": sha256_json(matrix),
    }
    matrix["materialization"] = materialization
    _assert_portable_and_secret_free(matrix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(matrix, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    verify_c3_matrix(
        output_path,
        repo_root=repo_root,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    return matrix


def verify_c3_matrix(
    matrix_path: Path,
    *,
    repo_root: Path,
    manifest_data_root: Path | None = None,
    expected_manifest_sha256: str = (
        AUTHORITATIVE_RENDERABLE_MANIFEST_SHA256
    ),
) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve(strict=True)
    matrix_path = matrix_path.expanduser().resolve(strict=True)
    try:
        matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise C3MatrixError(f"Cannot read C3 matrix: {exc}") from exc
    if not isinstance(matrix, dict):
        raise C3MatrixError("C3 matrix must be an object")
    _assert_portable_and_secret_free(matrix)
    materialization = matrix.get("materialization")
    if not isinstance(materialization, Mapping):
        raise C3MatrixError("C3 matrix lacks materialization provenance")
    if materialization.get("outcome_inputs_read") is not False:
        raise C3MatrixError("C3 materialization must not read run outcomes")
    if materialization.get("schema_version") != BUILDER_VERSION:
        raise C3MatrixError("C3 materialization schema version changed")
    repository_commit = materialization.get("repository_commit")
    if not isinstance(repository_commit, str) or not re.fullmatch(
        r"[0-9a-f]{7,64}",
        repository_commit,
    ):
        raise C3MatrixError("C3 repository commit binding is invalid")
    if matrix.get("methods") != _matrix_methods():
        raise C3MatrixError("C3 memory-mode mapping changed")
    if (
        matrix.get("backbones") != [BACKBONE]
        or matrix.get("seeds") != list(SEEDS)
        or matrix.get("budgets")
        != [{"type": "renders", "value": RENDER_BUDGET}]
        or matrix.get("provider") != PROVIDER
        or matrix.get("splits") != ["test"]
    ):
        raise C3MatrixError("C3 matrix contract changed")
    manifest_raw = matrix.get("dataset_manifest")
    if not isinstance(manifest_raw, str):
        raise C3MatrixError("C3 matrix dataset_manifest is invalid")
    manifest_path = (matrix_path.parent / manifest_raw).resolve(strict=True)
    _repo_relative(manifest_path, repo_root, "benchmark manifest")
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        manifest_cases = load_dataset_manifest(
            manifest_path,
            dataset_mode="sealed_benchmark",
            manifest_data_root=manifest_data_root,
            runtime_repo_root=repo_root,
        )
    except ManifestError as exc:
        raise C3MatrixError(
            f"Benchmark manifest is not fully sealed: {exc}"
        ) from exc
    selected = validate_selected_case_payloads(
        [case.payload for case in manifest_cases]
    )
    case_ids = [item["case_id"] for item in selected]
    dois = sorted({item["doi"] for item in selected})
    if matrix.get("case_ids") != case_ids:
        raise C3MatrixError("C3 case set is incomplete or reordered")
    manifest_sha256 = sha256_file(manifest_path)
    if (
        matrix.get("dataset_manifest_sha256") != manifest_sha256
        or manifest_sha256 != expected_manifest_sha256
    ):
        raise C3MatrixError(
            "C3 authoritative renderable derivative manifest hash changed"
        )
    source_hash = (manifest_payload.get("provenance") or {}).get(
        "source_binding_hash"
    )
    expected = {
        "parent_manifest_path": _repo_relative(
            manifest_path,
            repo_root,
            "benchmark manifest",
        ),
        "parent_manifest_file_sha256": sha256_file(manifest_path),
        "parent_manifest_semantic_sha256": sha256_json(manifest_payload),
        "parent_source_binding_hash": source_hash,
        "selected_cases": selected,
        "selected_case_count": len(selected),
        "selected_case_set_hash": sha256_json(case_ids),
        "selected_dois": dois,
        "selected_doi_count": len(dois),
        "selected_doi_set_hash": sha256_json(dois),
        "expected_run_count": len(selected) * len(SEEDS) * len(MEMORY_MODES),
    }
    for key, value in expected.items():
        if materialization.get(key) != value:
            raise C3MatrixError(f"C3 materialization binding changed: {key}")
    builder_relative = materialization.get("builder_path")
    if not isinstance(builder_relative, str):
        raise C3MatrixError("C3 builder_path is invalid")
    builder_path = (repo_root / builder_relative).resolve(strict=True)
    if materialization.get("builder_sha256") != sha256_file(builder_path):
        raise C3MatrixError("C3 builder hash changed")
    code_hashes = _code_hashes(repo_root)
    if (
        materialization.get("code_file_hashes") != code_hashes
        or materialization.get("code_bundle_hash") != sha256_json(code_hashes)
    ):
        raise C3MatrixError("C3 code bundle hash changed")
    if materialization.get("contract_hash") != sha256_json(_contract(matrix)):
        raise C3MatrixError("C3 matrix contract hash changed")
    expansion_matrix = dict(matrix)
    if manifest_data_root is not None:
        expansion_matrix["provider_options"] = {
            "manifest_data_root": str(
                manifest_data_root.expanduser().resolve(strict=True)
            )
        }
    specs = expand_matrix(
        expansion_matrix,
        base_dir=matrix_path.parent,
        repo_root=repo_root,
    )
    if (
        len(specs) != expected["expected_run_count"]
        or len({spec.run_name for spec in specs}) != len(specs)
    ):
        raise C3MatrixError("C3 matrix dry-run count or names changed")
    return {
        "case_count": len(selected),
        "doi_count": len(dois),
        "run_count": len(specs),
        "contract_hash": materialization["contract_hash"],
        "matrix_file_sha256": sha256_file(matrix_path),
    }


def write_spec_hash_manifest(
    matrix_path: Path,
    *,
    output_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve(strict=True)
    matrix_path = matrix_path.expanduser().resolve(strict=True)
    output_path = output_path.expanduser().resolve()
    verified = verify_c3_matrix(matrix_path, repo_root=repo_root)
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    specs = expand_matrix(
        matrix,
        base_dir=matrix_path.parent,
        repo_root=repo_root,
    )
    source_commit = matrix["materialization"]["repository_commit"]
    if (
        any(spec.git_dirty for spec in specs)
        or {spec.git_commit for spec in specs} != {source_commit}
    ):
        raise C3MatrixError(
            "Spec hashes may only be frozen from the clean bound commit"
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
        for spec in specs
    ]
    payload = {
        "schema_version": "1.0",
        "source_commit": source_commit,
        "source_dirty": False,
        "matrix_path": _repo_relative(
            matrix_path,
            repo_root,
            "C3 matrix",
        ),
        "matrix_file_sha256": verified["matrix_file_sha256"],
        "matrix_contract_hash": verified["contract_hash"],
        "run_count": len(entries),
        "specs": entries,
        "spec_set_hash": sha256_json(entries),
    }
    payload["manifest_hash"] = sha256_json(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize an outcome-independent portable C3 matrix"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--artifact-root",
        default=(
            "../runs/production/"
            "c3_memory_modes_final_benchmark_v2_seed0_br6_gpt56sol"
        ),
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--spec-manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.verify_only:
        result = verify_c3_matrix(args.out, repo_root=args.repo_root)
    else:
        materialize_c3_matrix(
            manifest_path=args.manifest,
            output_path=args.out,
            repo_root=args.repo_root,
            artifact_root=args.artifact_root,
        )
        result = verify_c3_matrix(args.out, repo_root=args.repo_root)
    if args.spec_manifest is not None:
        spec_manifest = write_spec_hash_manifest(
            args.out,
            output_path=args.spec_manifest,
            repo_root=args.repo_root,
        )
        result["spec_manifest"] = str(args.spec_manifest)
        result["spec_manifest_hash"] = spec_manifest["manifest_hash"]
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
