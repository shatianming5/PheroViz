#!/usr/bin/env python3
"""Hash-bind V3's controlled live review aliases without reading reviews.

The live aliases are valid only through their input manifest and the immutable
V3 freeze.  This publisher records that relationship, locks those binding
artifacts, and separately preserves the rule that pre-existing review evidence
cannot be reused.
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
INPUT_MANIFEST = ROOT / "files/c2_reclassify_review/input_manifest.json"
V3_FREEZE = ROOT / "files/c2_reclassify/frozen_v3_20260722T024015_0800"
V3_FREEZE_MANIFEST = V3_FREEZE / "freeze_manifest.json"
PREVIOUS_CLARIFICATION = (
    ROOT
    / "files/c2_reclassify"
    / "v3_mutable_predecessor_path_clarification_20260722T035220_0800.json"
)
EXPECTED_BATCHES = {
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


def live_binding_entries(
    live_manifest: dict[str, Any], frozen_manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    if live_manifest.get("schema_version") != "c2-v3-live-review-input-v2":
        raise SystemExit("unexpected live V3 input manifest schema")
    freeze_path = path_from_manifest(str(live_manifest["source_freeze"]))
    if freeze_path != V3_FREEZE_MANIFEST:
        raise SystemExit(f"unexpected source freeze: {freeze_path}")
    if live_manifest.get("source_freeze_sha256") != sha256_file(V3_FREEZE_MANIFEST):
        raise SystemExit("live V3 input manifest source freeze hash mismatch")

    frozen_by_label = {
        str(entry["label"]): entry for entry in frozen_manifest.get("inputs", [])
    }
    batches = live_manifest.get("batches")
    if not isinstance(batches, dict) or set(batches) != set(EXPECTED_BATCHES):
        raise SystemExit("unexpected controlled V3 live input batches")

    entries = []
    for batch, frozen_label in EXPECTED_BATCHES.items():
        live = batches[batch]
        frozen = frozen_by_label.get(frozen_label)
        if not isinstance(live, dict) or not isinstance(frozen, dict):
            raise SystemExit(f"missing V3 input binding for {batch}")
        live_path = path_from_manifest(str(live["path"]))
        frozen_path = path_from_manifest(str(live["frozen_source"]))
        live_hash = sha256_file(live_path)
        frozen_hash = sha256_file(frozen_path)
        if (
            live_hash != live.get("sha256")
            or frozen_hash != live.get("frozen_source_sha256")
            or frozen_hash != frozen.get("frozen_sha256")
            or live_hash != frozen_hash
        ):
            raise SystemExit(f"{batch}: controlled live binding hash mismatch")
        if frozen_path != Path(str(frozen["frozen_path"])):
            raise SystemExit(f"{batch}: frozen source path mismatch")
        entries.append(
            {
                "batch": batch,
                "controlled_live_input_path": str(live_path),
                "controlled_live_input_sha256": live_hash,
                "frozen_v3_input_path": str(frozen_path),
                "frozen_v3_input_sha256": frozen_hash,
                "records": live["records"],
                "single_panel": live["single_panel"],
                "multi_panel": live["multi_panel"],
                "live_input_readonly": not bool(live_path.stat().st_mode & 0o222),
                "frozen_input_readonly": not bool(
                    frozen_path.stat().st_mode & 0o222
                ),
            }
        )
    return entries


def lock_live_binding() -> tuple[str, Path]:
    sidecar = INPUT_MANIFEST.with_suffix(INPUT_MANIFEST.suffix + ".sha256")
    manifest_hash = sha256_file(INPUT_MANIFEST)
    if sidecar.exists() and sidecar.read_text(encoding="utf-8").strip() != manifest_hash:
        raise SystemExit(f"refusing to replace conflicting input-manifest sidecar: {sidecar}")
    if not sidecar.exists():
        sidecar.write_text(manifest_hash + "\n", encoding="utf-8")
    INPUT_MANIFEST.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    sidecar.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    inputs_directory = INPUT_MANIFEST.parent / "inputs"
    inputs_directory.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )
    return manifest_hash, sidecar


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite immutable artifact: {output}")

    frozen_manifest, frozen_internal_hash, frozen_file_hash = read_hash_bound_manifest(
        V3_FREEZE_MANIFEST
    )
    previous, previous_internal_hash, previous_file_hash = read_hash_bound_manifest(
        PREVIOUS_CLARIFICATION
    )
    live_manifest = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
    entries = live_binding_entries(live_manifest, frozen_manifest)
    input_manifest_hash, sidecar = lock_live_binding()
    created_at = datetime.now(timezone.utc).isoformat()

    clarification = {
        "schema_version": "c2-v3-controlled-live-input-binding-v1",
        "created_at_utc": created_at,
        "supersedes_interpretation_of": {
            "path": str(PREVIOUS_CLARIFICATION),
            "file_sha256": previous_file_hash,
            "internal_sha256": previous_internal_hash,
            "superseded_claim": (
                "The current c3ea/737c aliases are categorically non-authoritative."
            ),
            "replacement_interpretation": (
                "The aliases are controlled review inputs only when verified against "
                "this input manifest and their immutable frozen V3 sources."
            ),
        },
        "controlled_live_input_manifest": {
            "path": str(INPUT_MANIFEST),
            "file_sha256": input_manifest_hash,
            "file_sha256_sidecar": str(sidecar),
            "source_freeze_manifest": str(V3_FREEZE_MANIFEST),
            "source_freeze_manifest_sha256": frozen_file_hash,
            "source_freeze_manifest_internal_sha256": frozen_internal_hash,
            "inputs": entries,
        },
        "new_review_binding_requirement": (
            "Every new V3 review record must bind its input SHA-256 to this "
            "controlled live input manifest and to the matching immutable frozen "
            "V3 proposal copy; unversioned aliases alone are insufficient."
        ),
        "requires_fresh_review": True,
        "stale_review_or_evidence_reuse_forbidden": True,
        "prior_review_or_evidence_status": (
            "rejected: all review/evidence records created before this controlled "
            "binding, including records tied to the 485e/f2b3 predecessor inputs, "
            "are stale and must not be reused"
        ),
        "sealed_status": "diagnostic_only_stale_v3_strict_scope",
        "integrity_note": (
            "This publication reads only proposal-input manifests, frozen proposal "
            "copies, and hashes. It does not read review records, judge/model "
            "output, or evidence."
        ),
    }
    internal_hash, file_hash = write_hash_bound_manifest(output, clarification)
    print(
        json.dumps(
            {
                "clarification": str(output),
                "clarification_file_sha256": file_hash,
                "clarification_internal_sha256": internal_hash,
                "input_manifest_sha256": input_manifest_hash,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
