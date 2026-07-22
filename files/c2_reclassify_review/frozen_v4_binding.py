"""Validate and attach explicit frozen-V4 review-input provenance."""

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
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def frozen_v4_input_binding(
    *,
    freeze_manifest: Path,
    expected_manifest_internal_sha256: str,
    frozen_input: Path,
    input_label: str,
) -> dict[str, Any]:
    manifest_path = freeze_manifest.resolve(strict=True)
    input_path = frozen_input.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("frozen-v4-manifest-invalid")
    internal_hash = manifest.get("manifest_sha256")
    if (
        not isinstance(internal_hash, str)
        or internal_hash != expected_manifest_internal_sha256
        or canonical_sha256(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        )
        != internal_hash
        or manifest.get("proposal_rule_version") != "simple-2d-v4"
        or manifest.get("review_rubric_required") != "proposal-external-validation-v4"
        or manifest.get("requires_fresh_review") is not True
    ):
        raise RuntimeError("frozen-v4-manifest-binding-mismatch")

    manifest_schema = str(manifest.get("schema_version") or "")
    if isinstance(manifest.get("inputs"), list):
        entries = {
            str(item.get("label") or ""): item
            for item in manifest["inputs"]
            if isinstance(item, dict)
        }
        entry = entries.get(input_label)
        if entry is None:
            raise RuntimeError(f"frozen-v4-input-label-missing:{input_label}")
        scope = entry.get("scope")
        if scope != "historic-reject-selected diagnostic subset":
            raise RuntimeError(f"frozen-v4-input-scope-mismatch:{input_label}")
    elif manifest_schema == "c2-direct-raw-p5-v4-review-freeze-v1":
        entry = manifest.get("input")
        if not isinstance(entry, dict) or input_label != "direct_raw_p5_v4":
            raise RuntimeError(f"frozen-v4-input-label-missing:{input_label}")
        if (
            manifest.get("exploratory_normalizer_inputs_forbidden") is not True
            or entry.get("exploratory_normalizer_path_records") != 0
            or entry.get("source_normalization_records") != 0
            or entry.get("overlapping_component_source_ids") != 0
        ):
            raise RuntimeError("frozen-v4-direct-raw-integrity-mismatch")
        entry = {
            **entry,
            "singles": entry.get("single_panels"),
            "multi_parents": entry.get("multi_panel_parents"),
        }
        scope = "all-direct raw P5 C2-extreme universe (K62-insufficient)"
    else:
        raise RuntimeError(f"frozen-v4-manifest-schema-unsupported:{manifest_schema}")

    declared_path = Path(str(entry.get("frozen_path") or "")).resolve(strict=True)
    actual_hash = sha256_file(input_path)
    readonly = not (
        input_path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    )
    if (
        declared_path != input_path
        or actual_hash != entry.get("frozen_sha256")
        or not readonly
    ):
        raise RuntimeError(f"frozen-v4-input-binding-mismatch:{input_label}")
    return {
        "schema_version": "c2-v4-frozen-review-input-binding-v1",
        "v4_freeze_manifest": {
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
            "scope": scope,
            "readonly": True,
            "proposal_rule_version": "simple-2d-v4",
        },
        "requires_fresh_review": True,
    }


def decorate_binding(
    binding: Mapping[str, Any],
    *,
    frozen_input_binding: Mapping[str, Any],
    sha256_json: Callable[[Any], str],
) -> dict[str, Any]:
    decorated = json.loads(json.dumps(binding, ensure_ascii=False))
    decorated.pop("resume_binding_hash", None)
    decorated["frozen_v4_input_binding"] = json.loads(
        json.dumps(frozen_input_binding, ensure_ascii=False)
    )
    decorated["resume_binding_hash"] = sha256_json(decorated)
    return decorated
