#!/usr/bin/env python3
"""Freeze the complete non-overlapping V4 pool for a fresh dual-model review."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RECLASSIFIER_DIR = ROOT / "files" / "c2_reclassify"
if str(RECLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(RECLASSIFIER_DIR))

from improved_proposals import read_jsonl, validate_review_input  # noqa: E402


SOURCE = ROOT / "files/c2_reclassify_pool/reproposed_pool_multi.jsonl"
SOURCE_RELATIVE = SOURCE.relative_to(ROOT).as_posix()


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


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def pool_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    singles = [
        record for record in records if record.get("proposal_type") == "single_panel"
    ]
    multis = [
        record for record in records if record.get("proposal_type") == "multi_panel"
    ]
    if len(singles) + len(multis) != len(records):
        raise ValueError("unsupported proposal type in full pool")
    candidate_ids = [str(record.get("candidate_id") or "") for record in records]
    if not all(candidate_ids) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("full pool has missing or duplicate candidate IDs")
    if any(record.get("eligible_for_experiment") is not False for record in records):
        raise ValueError("full pool contains experiment-eligible record")
    rule_versions = {str(record.get("proposal_rule_version") or "") for record in records}
    if rule_versions != {"simple-2d-v4"}:
        raise ValueError(f"full pool is not uniformly V4: {sorted(rule_versions)}")

    single_ids = {str(record["candidate_id"]) for record in singles}
    component_counts = Counter(
        str(component_id)
        for record in multis
        for component_id in record.get("source_candidate_ids") or []
    )
    if set(component_counts) != single_ids:
        raise ValueError("multi components do not exactly cover full-pool singles")
    if any(count != 1 for count in component_counts.values()):
        raise ValueError("full pool has overlapping multi-panel components")

    multi_dois = {str(record.get("doi") or "") for record in multis}
    single_dois = {str(record.get("doi") or "") for record in singles}
    if "" in multi_dois or "" in single_dois or multi_dois != single_dois:
        raise ValueError("single/multi DOI coverage mismatch")
    return {
        "records": len(records),
        "single_panels": len(singles),
        "multi_panel_parents": len(multis),
        "distinct_candidate_ids": len(set(candidate_ids)),
        "distinct_doi_clusters": len(multi_dois),
        "component_source_ids": len(component_counts),
        "overlapping_component_source_ids": sum(
            count > 1 for count in component_counts.values()
        ),
        "proposal_rule_versions": sorted(rule_versions),
        "preserved_proposal_code_commits": sorted(
            {str(record.get("code_commit") or "") for record in records}
        ),
        "eligible_for_experiment_true": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty freeze directory: {output}")
    output.mkdir(parents=True, exist_ok=False)

    if not SOURCE.is_file():
        raise SystemExit(f"missing V4 full pool: {SOURCE}")
    source_bytes = SOURCE.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    head_bytes = subprocess.check_output(
        ["git", "show", f"HEAD:{SOURCE_RELATIVE}"],
        cwd=ROOT,
    )
    if head_bytes != source_bytes:
        raise SystemExit("full-pool bytes differ from HEAD; refusing to freeze")

    records = read_jsonl(SOURCE)
    validate_review_input(records)
    summary = pool_summary(records)

    destination = output / "v4_full_nonoverlap.proposed.jsonl"
    shutil.copyfile(SOURCE, destination)
    destination.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    frozen_hash = sha256_file(destination)
    if frozen_hash != source_hash:
        raise SystemExit("full-pool copy hash mismatch")

    manifest = {
        "schema_version": "c2-v4-full-pool-review-freeze-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_from_git_head": git_output("rev-parse", "HEAD"),
        "proposal_rule_version": "simple-2d-v4",
        "review_rubric_required": "proposal-external-validation-v4",
        "required_review_models": ["claude-sonnet-4.6", "gemini-3.5-flash"],
        "input": {
            "source_path": str(SOURCE),
            "source_sha256": source_hash,
            "frozen_path": str(destination),
            "frozen_sha256": frozen_hash,
            **summary,
        },
        "requires_fresh_review": True,
        "stale_review_or_evidence_reuse_forbidden": True,
        "priority_or_overlapping_strict_batches_forbidden": True,
        "integrity_note": (
            "This is the complete source-structure-only V4 pool, copied from a "
            "HEAD-identical artifact after offline review preflight. Candidate IDs "
            "are unique and every single-panel component belongs to exactly one "
            "multi parent. It contains 224 multi parents across 204 DOI clusters; "
            "downstream DOI-cluster inference must select at most one qualified "
            "parent per DOI. No review, judge, or evidence output was read here."
        ),
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    (output / "freeze_manifest.json").write_text(
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
