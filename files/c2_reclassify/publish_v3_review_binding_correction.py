#!/usr/bin/env python3
"""Publish an immutable correction for V3's predecessor-path observations.

The original V3 freeze remains unchanged.  Its predecessor-path hashes are
historical observations from creation time, not assertions about mutable paths
at a later time.  A later live alias requires its own hash-bound manifest and
immutable frozen source.  This publisher creates an immutable proposal-input
mirror and a hash-bound historical clarification.
It never reads review or judge output.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import stat
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
V3_FREEZE = ROOT / "files/c2_reclassify/frozen_v3_20260722T024015_0800"
V3_MANIFEST = V3_FREEZE / "freeze_manifest.json"

MUTABLE_PATHS = {
    "casecount_strict": (
        ROOT / "files/c2_reclassify_review/inputs/casecount_strict.proposed.jsonl",
        "priority_casecount_strict",
    ),
    "full_strict": (
        ROOT / "files/c2_reclassify_review/inputs/full_strict.proposed.jsonl",
        "priority_full_strict",
    ),
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


def write_immutable_json(path: Path, payload: dict[str, Any]) -> tuple[str, str]:
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


def immutable_input_copy(source: Path, destination: Path) -> str:
    if destination.exists():
        raise SystemExit(f"refusing to overwrite immutable input: {destination}")
    shutil.copyfile(source, destination)
    destination.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return sha256_file(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding-out", type=Path, required=True)
    parser.add_argument("--clarification-out", type=Path, required=True)
    args = parser.parse_args()

    binding_out = args.binding_out.resolve()
    clarification_out = args.clarification_out.resolve()
    if binding_out.exists():
        raise SystemExit(f"refusing to overwrite binding directory: {binding_out}")
    if clarification_out.exists():
        raise SystemExit(
            f"refusing to overwrite clarification artifact: {clarification_out}"
        )

    frozen_manifest = json.loads(V3_MANIFEST.read_text(encoding="utf-8"))
    frozen_manifest_without_hash = {
        key: value
        for key, value in frozen_manifest.items()
        if key != "manifest_sha256"
    }
    if frozen_manifest["manifest_sha256"] != canonical_hash(
        frozen_manifest_without_hash
    ):
        raise SystemExit(f"V3 freeze manifest internal hash mismatch: {V3_MANIFEST}")

    inputs_by_label = {
        str(entry["label"]): entry for entry in frozen_manifest.get("inputs", [])
    }
    predecessor_observations = frozen_manifest.get("known_stale_predecessor_inputs")
    if not isinstance(predecessor_observations, dict):
        raise SystemExit("V3 freeze lacks historical predecessor observations")

    created_at = datetime.now(timezone.utc).isoformat()
    binding_out.mkdir(parents=True, exist_ok=False)
    authoritative_inputs: list[dict[str, Any]] = []
    historical_paths: list[dict[str, Any]] = []

    for review_label, (mutable_path, frozen_label) in MUTABLE_PATHS.items():
        frozen_entry = inputs_by_label.get(frozen_label)
        if not isinstance(frozen_entry, dict):
            raise SystemExit(f"V3 freeze lacks {frozen_label!r}")
        source = Path(str(frozen_entry["frozen_path"]))
        source_hash = sha256_file(source)
        if source_hash != frozen_entry["frozen_sha256"]:
            raise SystemExit(f"V3 frozen input hash mismatch: {source}")

        destination = binding_out / f"{review_label}.proposed.jsonl"
        bound_hash = immutable_input_copy(source, destination)
        if bound_hash != source_hash:
            raise SystemExit(f"immutable copy hash mismatch: {destination}")

        old_observation = predecessor_observations.get(str(mutable_path))
        if not isinstance(old_observation, dict):
            raise SystemExit(
                f"V3 freeze lacks historical observation for mutable path: {mutable_path}"
            )
        historical_hash = old_observation.get("recorded_sha256")
        if not isinstance(historical_hash, str):
            raise SystemExit(f"missing historical hash for {mutable_path}")

        mutable_observed_now = (
            sha256_file(mutable_path) if mutable_path.is_file() else None
        )
        authoritative_inputs.append(
            {
                "label": review_label,
                "review_input_path": str(destination),
                "review_input_sha256": bound_hash,
                "source_v3_frozen_path": str(source),
                "source_v3_frozen_sha256": source_hash,
                "records": frozen_entry["records"],
                "singles": frozen_entry["singles"],
                "multi_parents": frozen_entry["multi_parents"],
                "eligible_for_experiment_true": frozen_entry[
                    "eligible_for_experiment_true"
                ],
                "scope": frozen_entry["scope"],
            }
        )
        historical_paths.append(
            {
                "former_mutable_path": str(mutable_path),
                "original_manifest_field": "known_stale_predecessor_inputs.current_sha256",
                "sha256_observed_at_original_v3_freeze": historical_hash,
                "sha256_observed_at_clarification_publication": mutable_observed_now,
                "clarification_observed_at_utc": created_at,
                "path_requires_a_separate_hash_bound_review_manifest": True,
                "path_is_not_self_authenticating": True,
                "authoritative_named_review_input": str(destination),
                "authoritative_named_review_input_sha256": bound_hash,
            }
        )

    binding_manifest = {
        "schema_version": "c2-v3-immutable-review-input-binding-v1",
        "created_at_utc": created_at,
        "binding_id": binding_out.name,
        "proposal_rule_version": "simple-2d-v3",
        "review_rubric_required": "proposal-external-validation-v3",
        "source_v3_freeze_manifest": str(V3_MANIFEST),
        "source_v3_freeze_manifest_sha256": sha256_file(V3_MANIFEST),
        "source_v3_freeze_manifest_internal_sha256": frozen_manifest[
            "manifest_sha256"
        ],
        "authoritative_proposal_inputs": authoritative_inputs,
        "superseded_mutable_path_observations": historical_paths,
        "requires_fresh_review": True,
        "stale_review_or_evidence_reuse_forbidden": True,
        "prior_review_or_evidence_status": (
            "rejected: all review/evidence records predating this binding or bound "
            "to the mutable unversioned input paths are stale and must not be reused"
        ),
        "sealed_status": "diagnostic_only_stale_v3_strict_scope",
        "integrity_note": (
            "This binding copies only the already hash-bound V3 proposal inputs. "
            "It does not read review records, judge/model output, or evidence. "
            "It is a named immutable proposal-input binding for a fresh review, "
            "not a validation of any earlier review."
        ),
    }
    binding_manifest_path = binding_out / "input_binding_manifest.json"
    binding_internal_hash, binding_file_hash = write_immutable_json(
        binding_manifest_path, binding_manifest
    )
    binding_out.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)

    clarification = {
        "schema_version": "c2-v3-mutable-predecessor-path-clarification-v1",
        "created_at_utc": created_at,
        "clarifies_v3_freeze_manifest": str(V3_MANIFEST),
        "clarifies_v3_freeze_manifest_sha256": sha256_file(V3_MANIFEST),
        "clarifies_v3_freeze_manifest_internal_sha256": frozen_manifest[
            "manifest_sha256"
        ],
        "correction": (
            "The original V3 manifest's fields named "
            "`known_stale_predecessor_inputs.*.current_sha256` are observations "
            "made when that manifest was created. They are not assertions about "
            "the hash of mutable unversioned paths at any later time. A later "
            "alias may be controlled only by a separate hash-bound input manifest "
            "and the matching immutable frozen proposal source."
        ),
        "historical_mutable_path_observations": historical_paths,
        "immutable_rebinding": {
            "binding_directory": str(binding_out),
            "binding_manifest": str(binding_manifest_path),
            "binding_manifest_sha256": binding_file_hash,
            "binding_manifest_internal_sha256": binding_internal_hash,
        },
        "requires_fresh_review": True,
        "stale_review_or_evidence_reuse_forbidden": True,
        "prior_review_or_evidence_status": binding_manifest[
            "prior_review_or_evidence_status"
        ],
        "sealed_status": "diagnostic_only_stale_v3_strict_scope",
    }
    clarification_internal_hash, clarification_file_hash = write_immutable_json(
        clarification_out, clarification
    )
    print(
        json.dumps(
            {
                "binding_directory": str(binding_out),
                "binding_manifest": str(binding_manifest_path),
                "binding_manifest_sha256": binding_file_hash,
                "binding_manifest_internal_sha256": binding_internal_hash,
                "clarification": str(clarification_out),
                "clarification_sha256": clarification_file_hash,
                "clarification_internal_sha256": clarification_internal_hash,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
