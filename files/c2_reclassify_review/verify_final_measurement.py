#!/usr/bin/env python3
"""Independently recount final C2 V3 artifacts and attach the cross-check."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "files/c2_reclassify_review"
BASELINE = Path(
    "/Users/tommy/.copilot/session-state/"
    "7b81726b-937c-41cb-9392-fead4d53250b/files/c2_review_scale"
)
SOURCES = {
    "casecount_strict": ROOT
    / "nature_download/outputs/c2_full_casecount_exploratory_20260720/"
    "corrected_sheet_binding/strict/proposals/proposed.jsonl",
    "full_strict": ROOT / "nature_download/outputs/c2_full_proposals/proposed.jsonl",
}
FROZEN_V3 = ROOT / "files/c2_reclassify/frozen_v3_20260722T024015_0800"


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    discrepancies: list[str] = []
    batches: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    baseline_single = Counter()
    baseline_multi = Counter()
    baseline_dois: set[str] = set()
    reclassified_dois: set[str] = set()
    all_batch_reviews: list[dict[str, Any]] = []
    input_single = input_multi = accepted_single = accepted_multi = 0
    input_binding: dict[str, Any] = {}

    sys.path.insert(0, str(ROOT / "nature_download"))
    from corpus.reviews import _validate_review_hash

    for label, source in SOURCES.items():
        proposals = jsonl(OUT / "inputs" / f"{label}.proposed.jsonl")
        reviews = jsonl(OUT / label / "reviews.jsonl")
        summary = json.loads((OUT / label / "summary.json").read_text())
        batches[label] = (proposals, reviews)
        proposal_ids = {item["candidate_id"] for item in proposals}
        review_ids = {item["candidate_id"] for item in reviews}
        if proposal_ids != review_ids:
            discrepancies.append(f"{label}:proposal-review-set")
        input_hash = sha256_file(OUT / "inputs" / f"{label}.proposed.jsonl")
        frozen_hash = sha256_file(FROZEN_V3 / f"priority_{label}.proposed.jsonl")
        review_input_hashes = {
            str(review.get("binding", {}).get("input_proposed_sha256") or "")
            for review in reviews
        }
        input_binding[label] = {
            "input_sha256": input_hash,
            "frozen_priority_sha256": frozen_hash,
            "review_binding_input_sha256s": sorted(review_input_hashes),
            "summary_input_sha256": summary.get("input_proposed_sha256"),
        }
        if (
            input_hash != frozen_hash
            or review_input_hashes != {input_hash}
            or summary.get("input_proposed_sha256") != input_hash
        ):
            discrepancies.append(f"{label}:frozen-input-binding")
        for review in reviews:
            try:
                _validate_review_hash(review)
            except Exception:
                discrepancies.append(f"{label}:review-hash:{review.get('candidate_id')}")
        input_single += sum(item["proposal_type"] == "single_panel" for item in proposals)
        input_multi += sum(item["proposal_type"] == "multi_panel" for item in proposals)
        accepted_single += sum(
            review["proposal_type"] == "single_panel" and review["status"] == "accepted"
            for review in reviews
        )
        accepted_multi += sum(
            review["proposal_type"] == "multi_panel" and review["status"] == "accepted"
            for review in reviews
        )
        proposal_by_id = {item["candidate_id"]: item for item in proposals}
        reclassified_dois.update(
            proposal_by_id[review["candidate_id"]]["doi"]
            for review in reviews
            if review["proposal_type"] == "multi_panel"
            and review["status"] == "accepted"
        )
        baseline_reviews = jsonl(BASELINE / label / "reviews.jsonl")
        baseline_proposals = {
            item["candidate_id"]: item for item in jsonl(source)
        }
        baseline_reject_ids = {
            item["candidate_id"]
            for item in baseline_reviews
            if item["proposal_type"] == "single_panel" and item["status"] == "rejected"
        }
        reclassified_ids = {
            item["candidate_id"]
            for item in proposals
            if item["proposal_type"] == "single_panel"
        }
        if baseline_reject_ids != reclassified_ids:
            discrepancies.append(f"{label}:baseline-pair-set")
        baseline_single.update(
            item["status"]
            for item in baseline_reviews
            if item["proposal_type"] == "single_panel"
        )
        baseline_multi.update(
            item["status"]
            for item in baseline_reviews
            if item["proposal_type"] == "multi_panel"
        )
        baseline_dois.update(
            baseline_proposals[item["candidate_id"]]["doi"]
            for item in baseline_reviews
            if item["proposal_type"] == "multi_panel" and item["status"] == "accepted"
        )
        all_batch_reviews.extend(reviews)

    root_reviews = jsonl(OUT / "reviews.jsonl")
    if root_reviews != all_batch_reviews:
        discrepancies.append("root:reviews-not-exact-concatenation")
    root_summary = json.loads((OUT / "summary.json").read_text())
    expected_summary = {
        "single_reviewed": input_single,
        "single_accepted": accepted_single,
        "multi_reviewed": input_multi,
        "multi_accepted": accepted_multi,
        "rejected": input_single + input_multi - accepted_single - accepted_multi,
    }
    for key, expected in expected_summary.items():
        if root_summary.get(key) != expected:
            discrepancies.append(f"root:summary:{key}")

    lift_path = OUT / "lift_measurement.json"
    lift = json.loads(lift_path.read_text())
    expected_values = {
        ("former_reject_to_accept", "accepted"): accepted_single,
        ("former_reject_to_accept", "former_rejects"): input_single,
        ("reclassified_set", "single_panel", "reviewed"): input_single,
        ("reclassified_set", "single_panel", "accepted"): accepted_single,
        ("reclassified_set", "multi_panel", "reviewed"): input_multi,
        ("reclassified_set", "multi_panel", "accepted"): accepted_multi,
        ("verified_multi_doi", "baseline_N"): len(baseline_dois),
        ("verified_multi_doi", "N_prime"): len(baseline_dois | reclassified_dois),
    }
    for keys, expected in expected_values.items():
        value: Any = lift
        for key in keys:
            value = value[key]
        if value != expected:
            discrepancies.append("lift:" + ".".join(keys))
    if (
        lift["baseline"]["aggregate_pass_rates"]["single_panel"]["accepted"]
        != baseline_single["accepted"]
        or lift["baseline"]["aggregate_pass_rates"]["single_panel"]["reviewed"]
        != sum(baseline_single.values())
        or lift["baseline"]["aggregate_pass_rates"]["multi_panel"]["accepted"]
        != baseline_multi["accepted"]
        or lift["baseline"]["aggregate_pass_rates"]["multi_panel"]["reviewed"]
        != sum(baseline_multi.values())
    ):
        discrepancies.append("lift:baseline-status-counts")

    residual = lift["residual_single_reject_primary_breakdown"]
    residual_sum = sum(
        value
        for key, value in residual.items()
        if key != "total_raw_rejected_singles"
    )
    if residual_sum != residual["total_raw_rejected_singles"]:
        discrepancies.append("lift:residual-buckets-do-not-sum")
    operational = lift["adjudication_triage"]["non_content_operational_errors"]
    if operational:
        discrepancies.append("lift:operational-errors-present")

    result = {
        "method": "Independent raw JSONL recount; does not call aggregate_lift helpers.",
        "recount": {
            "former_single_rejects": input_single,
            "accepted_reclassified_singles": accepted_single,
            "accepted_reclassified_multis": accepted_multi,
            "baseline_multi_N": len(baseline_dois),
            "N_prime": len(baseline_dois | reclassified_dois),
        },
        "frozen_input_binding": input_binding,
        "discrepancies": discrepancies,
        "discrepancy_count": len(discrepancies),
    }
    (OUT / "independent_crosscheck.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    lift["python_cross_validation"]["independent_raw_recount"] = result
    lift["python_cross_validation"]["discrepancy_count"] = len(
        lift["python_cross_validation"]["discrepancies"]
    ) + len(discrepancies)
    lift["python_cross_validation"]["discrepancies"] = (
        lift["python_cross_validation"]["discrepancies"] + discrepancies
    )
    lift_path.write_text(json.dumps(lift, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if discrepancies:
        raise SystemExit(json.dumps(result, ensure_ascii=False))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
