"""Validate and attach the explicit frozen-V3 review-input provenance."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any, Callable, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _readonly(path: Path) -> bool:
    return not (path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def frozen_v3_input_binding(
    *,
    freeze_manifest: Path,
    expected_manifest_internal_sha256: str,
    frozen_input: Path,
    input_label: str,
) -> dict[str, Any]:
    """Return metadata only after validating the exact immutable V3 input."""
    manifest_path = freeze_manifest.resolve(strict=True)
    input_path = frozen_input.resolve(strict=True)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"frozen-v3-manifest-invalid:{manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("frozen-v3-manifest-invalid")
    internal_hash = manifest.get("manifest_sha256")
    unhashed_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if (
        not isinstance(internal_hash, str)
        or internal_hash != expected_manifest_internal_sha256
        or canonical_sha256(unhashed_manifest) != internal_hash
    ):
        raise RuntimeError("frozen-v3-manifest-hash-mismatch")
    entries = {
        str(item.get("label")): item
        for item in manifest.get("inputs", [])
        if isinstance(item, dict)
    }
    entry = entries.get(input_label)
    if entry is None:
        raise RuntimeError(f"frozen-v3-input-label-missing:{input_label}")
    declared_path = Path(str(entry.get("frozen_path") or "")).resolve(strict=True)
    declared_hash = entry.get("frozen_sha256")
    actual_hash = sha256_file(input_path)
    if (
        declared_path != input_path
        or not isinstance(declared_hash, str)
        or actual_hash != declared_hash
        or not _readonly(input_path)
    ):
        raise RuntimeError(f"frozen-v3-input-binding-mismatch:{input_label}")
    return {
        "schema_version": "c2-v3-frozen-review-input-binding-v1",
        "v3_freeze_manifest": {
            "path": str(manifest_path),
            "internal_sha256": internal_hash,
            "file_sha256": sha256_file(manifest_path),
        },
        "frozen_proposal_input": {
            "label": input_label,
            "path": str(input_path),
            "sha256": actual_hash,
            "records": entry.get("records"),
            "singles": entry.get("singles"),
            "multi_parents": entry.get("multi_parents"),
            "scope": entry.get("scope"),
            "readonly": True,
        },
        "requires_fresh_review": True,
    }


def decorate_binding(
    binding: Mapping[str, Any],
    *,
    frozen_input_binding: Mapping[str, Any],
    sha256_json: Callable[[Any], str],
) -> dict[str, Any]:
    """Add freeze provenance before recomputing the binding resume hash."""
    decorated = json.loads(json.dumps(binding, ensure_ascii=False))
    decorated.pop("resume_binding_hash", None)
    decorated["frozen_v3_input_binding"] = json.loads(
        json.dumps(frozen_input_binding, ensure_ascii=False)
    )
    decorated["resume_binding_hash"] = sha256_json(decorated)
    return decorated
