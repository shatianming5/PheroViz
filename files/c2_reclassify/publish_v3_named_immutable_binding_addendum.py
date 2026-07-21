#!/usr/bin/env python3
"""Publish a named-immutable-binding-only V3 integrity addendum.

This publisher reads only manifests, frozen proposal inputs, and hashes.  It
does not inspect reviews, evidence, or judge/model output.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
V3_FREEZE_MANIFEST = (
    ROOT / "files/c2_reclassify/frozen_v3_20260722T024015_0800/freeze_manifest.json"
)
HISTORICAL_CLARIFICATION = (
    ROOT
    / "files/c2_reclassify"
    / "v3_mutable_predecessor_path_clarification_20260722T035220_0800.json"
)
NAMED_BINDING_MANIFEST = (
    ROOT
    / "files/c2_reclassify_review"
    / "frozen_v3_priority_review_inputs_20260722T035220_0800"
    / "input_binding_manifest.json"
)
EXPECTED_INPUTS = {
    "casecount_strict": "priority_casecount_strict",
    "full_strict": "priority_full_strict",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def readonly(path: Path) -> bool:
    return not bool(path.stat().st_mode & 0o222)


def read_hash_bound_manifest(path: Path) -> tuple[dict[str, Any], str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    internal_hash = payload.pop("manifest_sha256")
    if internal_hash != canonical_hash(payload):
        raise SystemExit(f"internal manifest hash mismatch: {path}")
    sidecars = (path.with_suffix(path.suffix + ".sha256"), path.with_suffix(".sha256"))
    if not any(
        sidecar.is_file()
        and sidecar.read_text(encoding="utf-8").strip() == internal_hash
        for sidecar in sidecars
    ):
        raise SystemExit(f"manifest sidecar hash mismatch: {path}")
    payload["manifest_sha256"] = internal_hash
    return payload, internal_hash, sha256_file(path)


def write_hash_bound_manifest(path: Path, payload: dict[str, Any]) -> tuple[str, str]:
    if path.exists():
        raise SystemExit(f"refusing to overwrite immutable artifact: {path}")
    payload["manifest_sha256"] = canonical_hash(payload)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(payload["manifest_sha256"] + "\n", encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    sidecar.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return payload["manifest_sha256"], sha256_file(path)


def path_from_manifest(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite immutable artifact: {output}")

    freeze, freeze_internal_hash, freeze_file_hash = read_hash_bound_manifest(
        V3_FREEZE_MANIFEST
    )
    freeze_entries = {str(entry["label"]): entry for entry in freeze["inputs"]}
    historical, historical_internal_hash, historical_file_hash = (
        read_hash_bound_manifest(HISTORICAL_CLARIFICATION)
    )
    observations = historical.get("historical_mutable_path_observations")
    if not isinstance(observations, list) or len(observations) != 2:
        raise SystemExit("unexpected historical predecessor observations")

    binding, binding_internal_hash, binding_file_hash = read_hash_bound_manifest(
        NAMED_BINDING_MANIFEST
    )
    if binding.get("source_v3_freeze_manifest_sha256") != freeze_file_hash:
        raise SystemExit("named binding source freeze hash mismatch")
    binding_entries = {
        str(entry["label"]): entry
        for entry in binding.get("authoritative_proposal_inputs", [])
    }
    verified_inputs = []
    for binding_label, freeze_label in EXPECTED_INPUTS.items():
        entry = binding_entries.get(binding_label)
        frozen = freeze_entries.get(freeze_label)
        if not isinstance(entry, dict) or not isinstance(frozen, dict):
            raise SystemExit(f"missing named binding input: {binding_label}")
        named_path = path_from_manifest(str(entry["review_input_path"]))
        frozen_path = path_from_manifest(str(entry["source_v3_frozen_path"]))
        expected_frozen_path = Path(str(frozen["frozen_path"]))
        named_hash = sha256_file(named_path)
        frozen_hash = sha256_file(frozen_path)
        if (
            frozen_path != expected_frozen_path
            or named_hash != entry.get("review_input_sha256")
            or frozen_hash != entry.get("source_v3_frozen_sha256")
            or frozen_hash != frozen.get("frozen_sha256")
            or named_hash != frozen_hash
            or not readonly(named_path)
            or not readonly(frozen_path)
        ):
            raise SystemExit(f"named immutable input verification failed: {binding_label}")
        verified_inputs.append(
            {
                "label": binding_label,
                "named_readonly_input": str(named_path),
                "sha256": named_hash,
                "frozen_v3_source": str(frozen_path),
                "byte_identical_to_frozen_v3_source": True,
                "readonly": True,
                "records": entry["records"],
                "singles": entry["singles"],
                "multi_parents": entry["multi_parents"],
            }
        )

    historical_entries = [
        {
            "former_alias_path": observation["former_mutable_path"],
            "sha256_observed_at_original_v3_freeze": observation[
                "sha256_observed_at_original_v3_freeze"
            ],
        }
        for observation in observations
    ]
    addendum = {
        "schema_version": "c2-v3-named-immutable-review-binding-addendum-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_v3_freeze": {
            "path": str(V3_FREEZE_MANIFEST),
            "file_sha256": freeze_file_hash,
            "internal_sha256": freeze_internal_hash,
        },
        "historical_predecessor_observation_context": {
            "clarification_path": str(HISTORICAL_CLARIFICATION),
            "clarification_file_sha256": historical_file_hash,
            "clarification_internal_sha256": historical_internal_hash,
            "semantics": (
                "These hashes are historical predecessor observations only, not "
                "bindings for a later review input."
            ),
            "observations": historical_entries,
        },
        "canonical_named_immutable_review_binding": {
            "binding_manifest_path": str(NAMED_BINDING_MANIFEST),
            "binding_manifest_file_sha256": binding_file_hash,
            "binding_manifest_internal_sha256": binding_internal_hash,
            "inputs": verified_inputs,
        },
        "new_review_binding_requirement": (
            "Every new V3 review record must bind the named immutable binding "
            "manifest SHA-256 and the exact named readonly proposal-input SHA-256 "
            "listed above. No unversioned alias path is a canonical review input."
        ),
        "requires_fresh_review": True,
        "stale_review_or_evidence_reuse_forbidden": True,
        "prior_review_or_evidence_status": (
            "rejected: all review/evidence records created before this named "
            "immutable binding, including records tied to the historical 485e/f2b3 "
            "predecessor observations, are stale and must not be reused"
        ),
        "no_in_place_overwrite": True,
        "sealed_status": "diagnostic_only_stale_v3_strict_scope",
        "integrity_note": (
            "This addendum uses only hash-bound source proposal manifests and "
            "frozen proposal files; it does not read reviews, evidence, or "
            "judge/model outputs."
        ),
    }
    internal_hash, file_hash = write_hash_bound_manifest(output, addendum)
    print(
        json.dumps(
            {
                "addendum": str(output),
                "addendum_file_sha256": file_hash,
                "addendum_internal_sha256": internal_hash,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
