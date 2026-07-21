#!/usr/bin/env python3
"""Publish a final, cross-bound V3 review-input integrity addendum.

This is metadata-only: it reads frozen proposal inputs and binding manifests,
never review records, evidence, or judge/model output.  It preserves earlier
immutable artifacts and makes their relationship explicit for audit.
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
NAMED_MIRROR_MANIFEST = (
    ROOT
    / "files/c2_reclassify_review"
    / "frozen_v3_priority_review_inputs_20260722T035220_0800"
    / "input_binding_manifest.json"
)
CONTROLLED_LIVE_BINDING = (
    ROOT
    / "files/c2_reclassify"
    / "v3_controlled_live_input_binding_20260722T035556_0800.json"
)


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


def path_from_manifest(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def source_freeze_metadata(
    freeze: dict[str, Any], internal_hash: str, file_hash: str
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    by_label = {str(entry["label"]): entry for entry in freeze["inputs"]}
    return by_label, {
        "path": str(V3_FREEZE_MANIFEST),
        "file_sha256": file_hash,
        "internal_sha256": internal_hash,
    }


def historical_observations(
    clarification: dict[str, Any], internal_hash: str, file_hash: str
) -> dict[str, Any]:
    observations = clarification.get("historical_mutable_path_observations")
    if not isinstance(observations, list) or len(observations) != 2:
        raise SystemExit("unexpected historical V3 predecessor observations")
    entries = []
    for observation in observations:
        historical = observation.get("sha256_observed_at_original_v3_freeze")
        if not isinstance(historical, str):
            raise SystemExit("missing historical predecessor hash")
        entries.append(
            {
                "former_live_alias": observation["former_mutable_path"],
                "sha256_observed_at_original_v3_freeze": historical,
                "controlled_replacement_sha256_observed_at_clarification": observation[
                    "sha256_observed_at_clarification_publication"
                ],
            }
        )
    return {
        "path": str(HISTORICAL_CLARIFICATION),
        "file_sha256": file_hash,
        "internal_sha256": internal_hash,
        "semantics": (
            "The 485e/f2b3 hashes are historical observations of predecessor "
            "aliases, not assertions about later controlled live aliases."
        ),
        "observations": entries,
    }


def named_mirror_metadata(
    binding: dict[str, Any],
    internal_hash: str,
    file_hash: str,
    freeze_entries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if binding.get("source_v3_freeze_manifest_sha256") != sha256_file(
        V3_FREEZE_MANIFEST
    ):
        raise SystemExit("named mirror source V3 freeze hash mismatch")
    expected_labels = {
        "casecount_strict": "priority_casecount_strict",
        "full_strict": "priority_full_strict",
    }
    inputs = binding.get("authoritative_proposal_inputs")
    if not isinstance(inputs, list) or len(inputs) != len(expected_labels):
        raise SystemExit("unexpected named V3 mirror inputs")
    verified = []
    for entry in inputs:
        label = str(entry["label"])
        source_label = expected_labels.get(label)
        if source_label is None:
            raise SystemExit(f"unexpected named V3 mirror label: {label}")
        mirror = path_from_manifest(str(entry["review_input_path"]))
        frozen = path_from_manifest(str(entry["source_v3_frozen_path"]))
        expected_frozen = Path(str(freeze_entries[source_label]["frozen_path"]))
        if frozen != expected_frozen:
            raise SystemExit(f"{label}: named mirror frozen path mismatch")
        mirror_hash = sha256_file(mirror)
        frozen_hash = sha256_file(frozen)
        if (
            mirror_hash != entry["review_input_sha256"]
            or frozen_hash != entry["source_v3_frozen_sha256"]
            or frozen_hash != freeze_entries[source_label]["frozen_sha256"]
            or mirror_hash != frozen_hash
        ):
            raise SystemExit(f"{label}: named mirror hash mismatch")
        verified.append(
            {
                "label": label,
                "named_readonly_input": str(mirror),
                "sha256": mirror_hash,
                "frozen_v3_source": str(frozen),
                "byte_identical_to_frozen_v3_source": True,
                "readonly": readonly(mirror) and readonly(frozen),
            }
        )
    if not all(entry["readonly"] for entry in verified):
        raise SystemExit("named mirror inputs must be readonly")
    return {
        "binding_manifest": str(NAMED_MIRROR_MANIFEST),
        "binding_manifest_file_sha256": file_hash,
        "binding_manifest_internal_sha256": internal_hash,
        "inputs": verified,
        "role": (
            "immutable named audit mirror; fresh live review records remain bound "
            "to the controlled live input manifest below"
        ),
    }


def controlled_live_metadata(
    controlled: dict[str, Any],
    internal_hash: str,
    file_hash: str,
    freeze_entries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    live_binding = controlled.get("controlled_live_input_manifest")
    if not isinstance(live_binding, dict):
        raise SystemExit("controlled binding lacks live input manifest")
    input_manifest_path = path_from_manifest(str(live_binding["path"]))
    input_manifest_hash = sha256_file(input_manifest_path)
    sidecar = path_from_manifest(str(live_binding["file_sha256_sidecar"]))
    if (
        input_manifest_hash != live_binding.get("file_sha256")
        or not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").strip() != input_manifest_hash
    ):
        raise SystemExit("controlled live input manifest hash mismatch")
    live_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    batches = live_manifest.get("batches")
    if not isinstance(batches, dict):
        raise SystemExit("controlled live input manifest lacks batches")
    expected_labels = {
        "casecount_strict": "priority_casecount_strict",
        "full_strict": "priority_full_strict",
    }
    controlled_entries = live_binding.get("inputs")
    if not isinstance(controlled_entries, list) or len(controlled_entries) != 2:
        raise SystemExit("unexpected controlled live input entries")
    verified = []
    for entry in controlled_entries:
        label = str(entry["batch"])
        source_label = expected_labels.get(label)
        live = batches.get(label)
        if source_label is None or not isinstance(live, dict):
            raise SystemExit(f"unexpected controlled live input label: {label}")
        live_path = path_from_manifest(str(entry["controlled_live_input_path"]))
        frozen_path = path_from_manifest(str(entry["frozen_v3_input_path"]))
        expected_frozen = Path(str(freeze_entries[source_label]["frozen_path"]))
        live_hash = sha256_file(live_path)
        frozen_hash = sha256_file(frozen_path)
        if (
            frozen_path != expected_frozen
            or live_path != path_from_manifest(str(live["path"]))
            or live_hash != entry["controlled_live_input_sha256"]
            or live_hash != live["sha256"]
            or frozen_hash != entry["frozen_v3_input_sha256"]
            or frozen_hash != live["frozen_source_sha256"]
            or frozen_hash != freeze_entries[source_label]["frozen_sha256"]
            or live_hash != frozen_hash
        ):
            raise SystemExit(f"{label}: controlled live input hash mismatch")
        verified.append(
            {
                "label": label,
                "controlled_live_input": str(live_path),
                "sha256": live_hash,
                "frozen_v3_source": str(frozen_path),
                "byte_identical_to_frozen_v3_source": True,
                "readonly": readonly(live_path) and readonly(frozen_path),
            }
        )
    if not all(entry["readonly"] for entry in verified):
        raise SystemExit("controlled live inputs must be readonly")
    return {
        "binding_addendum": str(CONTROLLED_LIVE_BINDING),
        "binding_addendum_file_sha256": file_hash,
        "binding_addendum_internal_sha256": internal_hash,
        "input_manifest": str(input_manifest_path),
        "input_manifest_file_sha256": input_manifest_hash,
        "input_manifest_sha256_sidecar": str(sidecar),
        "inputs": verified,
        "role": (
            "the only controlled V3 live aliases; each fresh review record must "
            "bind this input-manifest hash and the matching frozen source hash"
        ),
    }


def write_addendum(path: Path, payload: dict[str, Any]) -> tuple[str, str]:
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
    freeze_entries, freeze_metadata = source_freeze_metadata(
        freeze, freeze_internal_hash, freeze_file_hash
    )
    historical, historical_internal_hash, historical_file_hash = (
        read_hash_bound_manifest(HISTORICAL_CLARIFICATION)
    )
    named_mirror, mirror_internal_hash, mirror_file_hash = read_hash_bound_manifest(
        NAMED_MIRROR_MANIFEST
    )
    controlled, controlled_internal_hash, controlled_file_hash = (
        read_hash_bound_manifest(CONTROLLED_LIVE_BINDING)
    )
    if controlled.get("requires_fresh_review") is not True:
        raise SystemExit("controlled binding must require fresh review")
    if controlled.get("stale_review_or_evidence_reuse_forbidden") is not True:
        raise SystemExit("controlled binding must forbid stale evidence reuse")

    addendum = {
        "schema_version": "c2-v3-final-review-input-binding-addendum-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_v3_freeze": freeze_metadata,
        "historical_predecessor_observations": historical_observations(
            historical, historical_internal_hash, historical_file_hash
        ),
        "named_readonly_input_mirror": named_mirror_metadata(
            named_mirror,
            mirror_internal_hash,
            mirror_file_hash,
            freeze_entries,
        ),
        "controlled_live_input_binding": controlled_live_metadata(
            controlled,
            controlled_internal_hash,
            controlled_file_hash,
            freeze_entries,
        ),
        "live_alias_interpretation": (
            "Historical 485e/f2b3 values are predecessor observations only. "
            "Current c3ea/737c aliases are controlled only through the listed "
            "input manifest and byte-identical frozen V3 sources."
        ),
        "new_review_binding_requirement": (
            "Every new V3 review record must bind both the controlled live "
            "input-manifest file SHA-256 and its matching frozen V3 proposal "
            "source SHA-256. No alias path alone is a sufficient binding."
        ),
        "requires_fresh_review": True,
        "stale_review_or_evidence_reuse_forbidden": True,
        "prior_review_or_evidence_status": (
            "rejected: all review/evidence records created before this final "
            "cross-bound addendum, including records bound to the 485e/f2b3 "
            "predecessor inputs, are stale and must not be reused"
        ),
        "no_in_place_overwrite": True,
        "sealed_status": "diagnostic_only_stale_v3_strict_scope",
        "integrity_note": (
            "This addendum reads only manifests, frozen proposal files, paths, "
            "permissions, and SHA-256 values. It does not read review records, "
            "evidence, judge/model outputs, or use them for proposal labels."
        ),
    }
    internal_hash, file_hash = write_addendum(output, addendum)
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
