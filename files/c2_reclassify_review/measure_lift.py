#!/usr/bin/env python3
"""Measure the completed V3 casecount diagnostic conversion without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = Path(
    "/Users/tommy/.copilot/session-state/"
    "7b81726b-937c-41cb-9392-fead4d53250b/files/c2_review_scale"
)
FREEZE = ROOT / "files/c2_reclassify/frozen_v3_20260722T024015_0800"
INPUT = FREEZE / "priority_casecount_strict.proposed.jsonl"
INPUT_SHA256 = "c3ea1e106e8aa0307f5911bb5dc8323aa096267e2777bf315791c2c61d8fca49"
MANIFEST_INTERNAL_SHA256 = (
    "f2d7417a5df6171bc2dca24192fec2ae73fd0e5c372980d99ce57dfc8acf6999"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".next")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root", type=Path, default=ROOT / "files/c2_reclassify_review"
    )
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    if sha256_file(INPUT) != INPUT_SHA256:
        raise SystemExit("frozen V3 casecount input SHA-256 mismatch")
    proposals = jsonl(INPUT)
    reviews = jsonl(run_root / "casecount_strict/reviews.jsonl")
    baseline_reviews = jsonl(BASELINE_ROOT / "casecount_strict/reviews.jsonl")
    sys.path.insert(0, str(ROOT / "nature_download"))
    from corpus.reviews import _validate_review_hash

    invalid_review_hashes = []
    for row in reviews:
        try:
            _validate_review_hash(row)
        except Exception:
            invalid_review_hashes.append(str(row.get("candidate_id") or ""))
    proposal_ids = {str(row.get("candidate_id") or "") for row in proposals}
    review_ids = {str(row.get("candidate_id") or "") for row in reviews}
    former_reject_ids = {
        str(row.get("candidate_id") or "")
        for row in baseline_reviews
        if row.get("proposal_type") == "single_panel"
        and row.get("status") == "rejected"
    }
    single_reviews = [
        row for row in reviews if row.get("proposal_type") == "single_panel"
    ]
    accepted_singles = sum(row.get("status") == "accepted" for row in single_reviews)
    multi_reviews = [
        row for row in reviews if row.get("proposal_type") == "multi_panel"
    ]
    accepted_multis = sum(row.get("status") == "accepted" for row in multi_reviews)
    operational_errors = [
        {
            "candidate_id": row.get("candidate_id"),
            "failure_code": model.get("failure_code"),
        }
        for row in single_reviews
        for model in row.get("model_reviews") or []
        if model.get("failure_code") in {"model-call-failed", "model-client-init-failed"}
    ]
    result = {
        "status": "diagnostic_only_superseded_by_v4",
        "purpose": (
            "Completed V3 casecount conversion requested before V4; not a final "
            "sealed-C2 or N-prime gate measurement."
        ),
        "freeze_manifest": str(FREEZE / "freeze_manifest.json"),
        "freeze_manifest_internal_sha256": MANIFEST_INTERNAL_SHA256,
        "input": str(INPUT),
        "input_sha256": INPUT_SHA256,
        "casecount": {
            "proposals": len(proposals),
            "reviews": len(reviews),
            "former_single_rejects": len(former_reject_ids),
            "single_accepted": accepted_singles,
            "former_reject_to_accept_rate": (
                accepted_singles / len(former_reject_ids)
                if former_reject_ids
                else None
            ),
            "multi_reviewed": len(multi_reviews),
            "multi_accepted": accepted_multis,
            "multi_pass_rate": (
                accepted_multis / len(multi_reviews) if multi_reviews else None
            ),
        },
        "checks": {
            "proposal_review_id_sets_match": proposal_ids == review_ids,
            "all_review_hashes_valid": not invalid_review_hashes,
            "invalid_review_hash_candidate_ids": invalid_review_hashes,
            "former_reject_single_id_set_matches": {
                str(row.get("candidate_id") or "")
                for row in proposals
                if row.get("proposal_type") == "single_panel"
            }
            == former_reject_ids,
            "unresolved_operational_errors": operational_errors,
        },
    }
    atomic_json(run_root / "v3_casecount_lift.json", result)
    print(json.dumps(result, ensure_ascii=False))
    if not all(
        (
            result["checks"]["proposal_review_id_sets_match"],
            result["checks"]["former_reject_single_id_set_matches"],
            result["checks"]["all_review_hashes_valid"],
            not operational_errors,
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
