#!/usr/bin/env python3
"""Promote a validated fresh frozen-V3 run while retaining stale artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARTIFACTS = (
    "probe_casecount",
    "casecount_strict",
    "full_strict",
    "reviews.jsonl",
    "summary.json",
    "review_partitions.json",
    "lift_measurement.json",
    "independent_crosscheck.json",
    "frozen_v3_artifact_validation.json",
    "verdict.md",
)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".next")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--archive-name", required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve(strict=True)
    validation = json.loads(
        (run_root / "frozen_v3_artifact_validation.json").read_text(encoding="utf-8")
    )
    crosscheck = json.loads(
        (run_root / "independent_crosscheck.json").read_text(encoding="utf-8")
    )
    if validation.get("discrepancy_count") != 0 or crosscheck.get("discrepancy_count") != 0:
        raise SystemExit("refusing to promote a nonzero-discrepancy run")
    if any(not (run_root / name).exists() for name in ARTIFACTS):
        raise SystemExit("fresh run is missing a required artifact")

    archive = ROOT / args.archive_name
    if archive.exists():
        raise SystemExit(f"archive already exists: {archive}")
    archive.mkdir()
    moved: list[str] = []
    for name in ARTIFACTS:
        source = ROOT / name
        if source.exists():
            shutil.move(str(source), str(archive / name))
            moved.append(name)
    for name in ARTIFACTS:
        source = run_root / name
        destination = ROOT / name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)

    status = {
        "status": "completed_current_frozen_v3",
        "scope": "diagnostic-only historic-reject-selected V3 subset",
        "current_run_root": str(run_root),
        "freeze_manifest": validation["freeze_manifest"],
        "freeze_manifest_internal_sha256": validation[
            "freeze_manifest_internal_sha256"
        ],
        "validation_discrepancy_count": validation["discrepancy_count"],
        "measurement_discrepancy_count": crosscheck["discrepancy_count"],
        "prior_artifacts_archived_to": str(archive),
        "prior_artifact_names": moved,
    }
    write_json(ROOT / "RUN_STATUS.json", status)
    write_json(ROOT / "CURRENT_FROZEN_V3_RUN.json", status)
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
