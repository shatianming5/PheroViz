#!/usr/bin/env python3
"""Freeze hash-bound V3 review inputs without reading review or judge outputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import stat
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from improved_proposals import read_jsonl, validate_review_input  # noqa: E402


SOURCE_ROOT = ROOT / "files" / "c2_reclassify"
INPUTS = (
    (
        "priority_casecount_strict",
        SOURCE_ROOT / "reproposed_strict_rejects_casecount_strict.jsonl",
        "historic-reject-selected diagnostic subset",
    ),
    (
        "priority_full_strict",
        SOURCE_ROOT / "reproposed_strict_rejects_full_strict.jsonl",
        "historic-reject-selected diagnostic subset",
    ),
    (
        "full_casecount_strict",
        SOURCE_ROOT / "reproposed_strict_full_casecount_strict.jsonl",
        "full supplied strict batch; not the sealed C2 universe",
    ),
    (
        "full_full_strict",
        SOURCE_ROOT / "reproposed_strict_full_full_strict.jsonl",
        "full supplied strict batch; not the sealed C2 universe",
    ),
)
KNOWN_STALE_PREDECESSORS_AT_FREEZE = {
    str(
        ROOT
        / "files/c2_reclassify_review/inputs/casecount_strict.proposed.jsonl"
    ): "485e36ca4e32be0e9b58a0327088301813928a47627c38518ee02fac3a32f353",
    str(
        ROOT / "files/c2_reclassify_review/inputs/full_strict.proposed.jsonl"
    ): "f2b364455fa69d5360a5925b8fc2a9516f1c2da99e099e46c36b73cfdbf618d6",
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


def record_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "records": len(records),
        "singles": sum(
            record.get("proposal_type") == "single_panel" for record in records
        ),
        "multi_parents": sum(
            record.get("proposal_type") == "multi_panel" for record in records
        ),
        "eligible_for_experiment_true": sum(
            record.get("eligible_for_experiment") is True for record in records
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty freeze directory: {output}")
    output.mkdir(parents=True, exist_ok=False)

    frozen_inputs = []
    for label, source, scope in INPUTS:
        if not source.is_file():
            raise SystemExit(f"missing current V3 input: {source}")
        records = read_jsonl(source)
        validate_review_input(records)
        counts = record_counts(records)
        if counts["eligible_for_experiment_true"]:
            raise SystemExit(f"{source}: proposal unexpectedly experiment eligible")
        destination = output / f"{label}.proposed.jsonl"
        shutil.copyfile(source, destination)
        destination.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        source_hash = sha256_file(source)
        frozen_hash = sha256_file(destination)
        if source_hash != frozen_hash:
            raise SystemExit(f"{source}: copy hash mismatch")
        frozen_inputs.append(
            {
                "label": label,
                "scope": scope,
                "source_path": str(source),
                "source_sha256": source_hash,
                "frozen_path": str(destination),
                "frozen_sha256": frozen_hash,
                **counts,
            }
        )

    stale_predecessors_at_freeze = {
        path: {
            "sha256_observed_at_freeze": expected,
            "path_requires_a_separate_hash_bound_review_manifest": True,
            "not_a_live_path_hash_assertion": True,
        }
        for path, expected in KNOWN_STALE_PREDECESSORS_AT_FREEZE.items()
    }
    manifest = {
        "schema_version": "c2-v3-review-freeze-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "proposal_rule_version": "simple-2d-v3",
        "review_rubric_required": "proposal-external-validation-v3",
        "inputs": frozen_inputs,
        "known_stale_predecessor_inputs_at_freeze": stale_predecessors_at_freeze,
        "requires_fresh_review": True,
        "stale_review_or_evidence_reuse_forbidden": True,
        "integrity_note": (
            "This freeze copies and hash-binds current V3 proposal inputs only. "
            "It does not read review records, judge/model outputs, or evidence. "
            "Predecessor-path hashes are historical observations made at freeze "
            "creation and must never be interpreted as hashes of mutable paths "
            "at a later time. "
            "Priority inputs are historic-reject-selected diagnostic subsets and "
            "cannot support a final sealed-C2 universe claim."
        ),
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    manifest_path = output / "freeze_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "freeze_manifest.sha256").write_text(
        manifest["manifest_sha256"] + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
