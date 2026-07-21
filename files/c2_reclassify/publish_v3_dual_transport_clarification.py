#!/usr/bin/env python3
"""Publish a source-only clarification of V3's two valid diagnostic transports."""

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
CONTROLLED_ALIAS_BINDING = (
    ROOT
    / "files/c2_reclassify"
    / "v3_controlled_live_input_binding_20260722T035556_0800.json"
)
NAMED_MIRROR_BINDING = (
    ROOT
    / "files/c2_reclassify_review"
    / "frozen_v3_priority_review_inputs_20260722T035220_0800"
    / "input_binding_manifest.json"
)
NAMED_ONLY_ADDENDUM = (
    ROOT
    / "files/c2_reclassify"
    / "v3_named_immutable_review_binding_addendum_20260722T040852_0800.json"
)
EXPECTED_LABELS = {
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


def path_from_manifest(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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
        raise SystemExit(f"manifest sidecar mismatch: {path}")
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


def frozen_inputs(freeze: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(entry["label"]): entry for entry in freeze["inputs"]}


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
    frozen_by_label = frozen_inputs(freeze)
    historical, historical_internal_hash, historical_file_hash = (
        read_hash_bound_manifest(HISTORICAL_CLARIFICATION)
    )
    historical_observations = historical.get("historical_mutable_path_observations")
    if not isinstance(historical_observations, list) or len(historical_observations) != 2:
        raise SystemExit("unexpected V3 historical predecessor observations")

    controlled, controlled_internal_hash, controlled_file_hash = (
        read_hash_bound_manifest(CONTROLLED_ALIAS_BINDING)
    )
    live_binding = controlled.get("controlled_live_input_manifest")
    if not isinstance(live_binding, dict):
        raise SystemExit("controlled alias binding lacks input manifest")
    input_manifest = path_from_manifest(str(live_binding["path"]))
    input_manifest_hash = sha256_file(input_manifest)
    input_sidecar = path_from_manifest(str(live_binding["file_sha256_sidecar"]))
    if (
        input_manifest_hash != live_binding.get("file_sha256")
        or not input_sidecar.is_file()
        or input_sidecar.read_text(encoding="utf-8").strip() != input_manifest_hash
    ):
        raise SystemExit("controlled alias input manifest mismatch")
    if live_binding.get("source_freeze_manifest_sha256") != freeze_file_hash:
        raise SystemExit("controlled alias source freeze mismatch")
    controlled_by_label = {
        str(entry["batch"]): entry for entry in live_binding.get("inputs", [])
    }

    named, named_internal_hash, named_file_hash = read_hash_bound_manifest(
        NAMED_MIRROR_BINDING
    )
    if named.get("source_v3_freeze_manifest_sha256") != freeze_file_hash:
        raise SystemExit("named mirror source freeze mismatch")
    named_by_label = {
        str(entry["label"]): entry
        for entry in named.get("authoritative_proposal_inputs", [])
    }

    verified_transports = []
    for label, frozen_label in EXPECTED_LABELS.items():
        frozen = frozen_by_label.get(frozen_label)
        controlled_entry = controlled_by_label.get(label)
        named_entry = named_by_label.get(label)
        if not all(isinstance(entry, dict) for entry in (frozen, controlled_entry, named_entry)):
            raise SystemExit(f"missing V3 transport entry: {label}")
        frozen_path = Path(str(frozen["frozen_path"]))
        live_path = path_from_manifest(
            str(controlled_entry["controlled_live_input_path"])
        )
        named_path = path_from_manifest(str(named_entry["review_input_path"]))
        frozen_hash = sha256_file(frozen_path)
        live_hash = sha256_file(live_path)
        named_hash = sha256_file(named_path)
        if (
            frozen_hash != frozen.get("frozen_sha256")
            or frozen_hash != controlled_entry.get("frozen_v3_input_sha256")
            or frozen_hash != named_entry.get("source_v3_frozen_sha256")
            or path_from_manifest(str(controlled_entry["frozen_v3_input_path"]))
            != frozen_path
            or path_from_manifest(str(named_entry["source_v3_frozen_path"]))
            != frozen_path
            or live_hash != controlled_entry.get("controlled_live_input_sha256")
            or named_hash != named_entry.get("review_input_sha256")
            or frozen_hash != live_hash
            or frozen_hash != named_hash
            or not readonly(frozen_path)
            or not readonly(live_path)
            or not readonly(named_path)
        ):
            raise SystemExit(f"V3 transport parity mismatch: {label}")
        verified_transports.append(
            {
                "label": label,
                "sha256": frozen_hash,
                "frozen_v3_source": str(frozen_path),
                "controlled_alias_input": str(live_path),
                "named_readonly_mirror_input": str(named_path),
                "all_three_byte_identical": True,
                "all_three_readonly": True,
            }
        )

    named_only, named_only_internal_hash, named_only_file_hash = (
        read_hash_bound_manifest(NAMED_ONLY_ADDENDUM)
    )
    addendum = {
        "schema_version": "c2-v3-dual-diagnostic-transport-clarification-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_v3_freeze": {
            "path": str(V3_FREEZE_MANIFEST),
            "file_sha256": freeze_file_hash,
            "internal_sha256": freeze_internal_hash,
        },
        "historical_predecessor_observation_context": {
            "path": str(HISTORICAL_CLARIFICATION),
            "file_sha256": historical_file_hash,
            "internal_sha256": historical_internal_hash,
            "semantics": (
                "The 485e/f2b3 values are historical predecessor observations, "
                "not a finding that later hash-bound aliases are invalid."
            ),
            "observations": [
                {
                    "former_alias_path": entry["former_mutable_path"],
                    "sha256_observed_at_original_v3_freeze": entry[
                        "sha256_observed_at_original_v3_freeze"
                    ],
                }
                for entry in historical_observations
            ],
        },
        "valid_diagnostic_transports": {
            "controlled_alias_binding": {
                "path": str(CONTROLLED_ALIAS_BINDING),
                "file_sha256": controlled_file_hash,
                "internal_sha256": controlled_internal_hash,
                "input_manifest_path": str(input_manifest),
                "input_manifest_file_sha256": input_manifest_hash,
                "input_manifest_sha256_sidecar": str(input_sidecar),
                "validity_condition": (
                    "valid only when the input-manifest hash and matching frozen "
                    "V3 source hash are bound into the review record"
                ),
            },
            "named_readonly_mirror_binding": {
                "path": str(NAMED_MIRROR_BINDING),
                "file_sha256": named_file_hash,
                "internal_sha256": named_internal_hash,
                "validity_condition": (
                    "valid only when the named binding-manifest hash and matching "
                    "named readonly input hash are bound into the review record"
                ),
            },
            "verified_equivalent_inputs": verified_transports,
        },
        "supersedes_exclusive_named_only_interpretation": {
            "path": str(NAMED_ONLY_ADDENDUM),
            "file_sha256": named_only_file_hash,
            "internal_sha256": named_only_internal_hash,
            "correction": (
                "The named immutable mirror is valid, but it is not the only "
                "valid diagnostic transport: the controlled aliases are also "
                "valid when bound through their input manifest and frozen source."
            ),
        },
        "new_review_binding_requirement": (
            "A fresh V3 diagnostic review must bind one complete valid transport: "
            "either the controlled alias manifest plus frozen source, or the named "
            "immutable binding manifest plus named readonly input and frozen source. "
            "An alias path alone is insufficient."
        ),
        "requires_fresh_review": True,
        "stale_review_or_evidence_reuse_forbidden": True,
        "prior_review_or_evidence_status": (
            "rejected: all prior review/evidence remains stale, including evidence "
            "bound to the historical 485e/f2b3 predecessor inputs"
        ),
        "no_in_place_overwrite": True,
        "sealed_status": "diagnostic_only_stale_v3_strict_scope",
        "integrity_note": (
            "This clarification reads only manifests, proposal files, file modes, "
            "and SHA-256 values. It does not read reviews, evidence, or judge/model "
            "outputs."
        ),
    }
    internal_hash, file_hash = write_hash_bound_manifest(output, addendum)
    print(
        json.dumps(
            {
                "clarification": str(output),
                "clarification_file_sha256": file_hash,
                "clarification_internal_sha256": internal_hash,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
