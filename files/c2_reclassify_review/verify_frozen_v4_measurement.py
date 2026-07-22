#!/usr/bin/env python3
"""Independently recount the fresh frozen-V4 lift measurement."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from frozen_v4_binding import frozen_v4_input_binding, sha256_file


ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = Path(
    "/Users/tommy/.copilot/session-state/"
    "7b81726b-937c-41cb-9392-fead4d53250b/files/c2_review_scale"
)
BASELINE_SOURCES = {
    "casecount_strict": ROOT
    / "nature_download/outputs/c2_full_casecount_exploratory_20260720/"
    "corrected_sheet_binding/strict/proposals/proposed.jsonl",
    "full_strict": ROOT / "nature_download/outputs/c2_full_proposals/proposed.jsonl",
}


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def get_path(value: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = value
    for key in keys:
        current = current[key]
    return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--freeze-manifest-internal-sha256", required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    freeze_manifest = args.freeze_manifest.resolve(strict=True)
    frozen_root = freeze_manifest.parent
    lift = json.loads((run_root / "lift_measurement.json").read_text(encoding="utf-8"))
    discrepancies: list[str] = []
    baseline_single = Counter()
    baseline_multi = Counter()
    baseline_dois: set[str] = set()
    converted_multi_dois: set[str] = set()
    total_single = accepted_single = total_multi = accepted_multi = 0
    frozen_bindings: dict[str, Any] = {}

    for label in ("casecount_strict", "full_strict"):
        proposed = frozen_root / f"priority_{label}.proposed.jsonl"
        binding = frozen_v4_input_binding(
            freeze_manifest=freeze_manifest,
            expected_manifest_internal_sha256=args.freeze_manifest_internal_sha256,
            frozen_input=proposed,
            input_label=f"priority_{label}",
        )
        frozen_bindings[label] = binding
        proposals = jsonl(proposed)
        reviews = jsonl(run_root / label / "reviews.jsonl")
        proposal_by_id = {str(row.get("candidate_id") or ""): row for row in proposals}
        review_by_id = {str(row.get("candidate_id") or ""): row for row in reviews}
        if set(proposal_by_id) != set(review_by_id):
            discrepancies.append(f"{label}:proposal-review-set")
        if any(
            row.get("binding", {}).get("frozen_v4_input_binding") != binding
            for row in reviews
        ):
            discrepancies.append(f"{label}:frozen-v4-binding")
        input_hash = sha256_file(proposed)
        if any(
            row.get("binding", {}).get("input_proposed_sha256") != input_hash
            for row in reviews
        ):
            discrepancies.append(f"{label}:input-hash")
        singles = [row for row in reviews if row.get("proposal_type") == "single_panel"]
        multis = [row for row in reviews if row.get("proposal_type") == "multi_panel"]
        total_single += len(singles)
        accepted_single += sum(row.get("status") == "accepted" for row in singles)
        total_multi += len(multis)
        accepted_multi += sum(row.get("status") == "accepted" for row in multis)
        converted_multi_dois.update(
            str(proposal_by_id[str(row["candidate_id"])]["doi"])
            for row in multis
            if row.get("status") == "accepted"
        )

        baseline_reviews = jsonl(BASELINE_ROOT / label / "reviews.jsonl")
        baseline_proposals = {
            str(row.get("candidate_id") or ""): row
            for row in jsonl(BASELINE_SOURCES[label])
        }
        frozen_single_ids = {
            str(row.get("candidate_id") or "")
            for row in proposals
            if row.get("proposal_type") == "single_panel"
        }
        former_reject_ids = {
            str(row.get("candidate_id") or "")
            for row in baseline_reviews
            if row.get("proposal_type") == "single_panel"
            and row.get("status") == "rejected"
        }
        if frozen_single_ids != former_reject_ids:
            discrepancies.append(f"{label}:baseline-pair-set")
        for row in baseline_reviews:
            if row.get("proposal_type") == "single_panel":
                baseline_single[str(row.get("status"))] += 1
            elif row.get("proposal_type") == "multi_panel":
                baseline_multi[str(row.get("status"))] += 1
                if row.get("status") == "accepted":
                    baseline_dois.add(
                        str(baseline_proposals[str(row["candidate_id"])]["doi"])
                    )

    expected = {
        ("former_reject_to_accept", "accepted"): accepted_single,
        ("former_reject_to_accept", "former_rejects"): total_single,
        ("reclassified_set", "single_panel", "reviewed"): total_single,
        ("reclassified_set", "single_panel", "accepted"): accepted_single,
        ("reclassified_set", "multi_panel", "reviewed"): total_multi,
        ("reclassified_set", "multi_panel", "accepted"): accepted_multi,
        ("verified_multi_doi", "baseline_N"): len(baseline_dois),
        ("verified_multi_doi", "N_prime"): len(baseline_dois | converted_multi_dois),
    }
    for keys, expected_value in expected.items():
        try:
            if get_path(lift, keys) != expected_value:
                discrepancies.append("lift:" + ".".join(keys))
        except (KeyError, TypeError):
            discrepancies.append("lift:missing:" + ".".join(keys))
    baseline_single_lift = lift.get("baseline", {}).get("aggregate_pass_rates", {}).get(
        "single_panel", {}
    )
    baseline_multi_lift = lift.get("baseline", {}).get("aggregate_pass_rates", {}).get(
        "multi_panel", {}
    )
    if (
        baseline_single_lift.get("accepted") != baseline_single["accepted"]
        or baseline_single_lift.get("reviewed") != sum(baseline_single.values())
        or baseline_multi_lift.get("accepted") != baseline_multi["accepted"]
        or baseline_multi_lift.get("reviewed") != sum(baseline_multi.values())
    ):
        discrepancies.append("lift:baseline-counts")
    if (
        lift.get("input", {}).get("frozen_v4_manifest_internal_sha256")
        != args.freeze_manifest_internal_sha256
    ):
        discrepancies.append("lift:freeze-manifest-internal-hash")
    result = {
        "method": (
            "Independent raw JSONL recount of the fresh frozen-V4 run; no aggregate "
            "helpers and no model calls."
        ),
        "freeze_manifest": str(freeze_manifest),
        "freeze_manifest_internal_sha256": args.freeze_manifest_internal_sha256,
        "frozen_input_bindings": frozen_bindings,
        "recount": {
            "former_single_rejects": total_single,
            "accepted_reclassified_singles": accepted_single,
            "accepted_reclassified_multis": accepted_multi,
            "baseline_multi_N": len(baseline_dois),
            "N_prime": len(baseline_dois | converted_multi_dois),
        },
        "discrepancies": discrepancies,
        "discrepancy_count": len(discrepancies),
    }
    (run_root / "independent_crosscheck.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lift.setdefault("python_cross_validation", {})["independent_raw_recount"] = result
    aggregate_discrepancies = lift["python_cross_validation"].get("discrepancies", [])
    lift["python_cross_validation"]["discrepancies"] = (
        aggregate_discrepancies + discrepancies
    )
    lift["python_cross_validation"]["discrepancy_count"] = len(
        lift["python_cross_validation"]["discrepancies"]
    )
    (run_root / "lift_measurement.json").write_text(
        json.dumps(lift, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    if discrepancies:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
