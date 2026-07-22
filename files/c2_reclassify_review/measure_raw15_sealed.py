#!/usr/bin/env python3
"""Produce the direct-raw P5 per-DOI C2-extreme measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from frozen_v4_binding import frozen_v4_input_binding


EXPECTED_INPUT_SHA256 = (
    "d7c655b97334186d75c99ff69e23615065535224e609b40e62e1c50e26cde801"
)


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(path.name + ".next")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def panel_id(proposal: Mapping[str, Any]) -> str | None:
    case = proposal.get("experiment_case")
    if isinstance(case, Mapping):
        value = case.get("panel_id")
        if isinstance(value, str):
            return value
    values = proposal.get("panel_ids")
    if isinstance(values, list) and len(values) == 1 and isinstance(values[0], str):
        return values[0]
    return None


def failure_categories(
    proposal: Mapping[str, Any], review: Mapping[str, Any]
) -> list[str]:
    categories: set[str] = set()
    reasons = [str(reason) for reason in review.get("rejection_reasons") or []]
    model_reviews = review.get("model_reviews") or []
    for reason in reasons:
        lowered = reason.casefold()
        if (
            "model-call" in lowered
            or "model-client" in lowered
            or "served-model" in lowered
            or "truncated" in lowered
            or "not-completed" in lowered
        ):
            categories.add("operational")
        elif "schema" in lowered or "json" in lowered:
            categories.add("schema_invalid")
        elif "render" in lowered:
            categories.add("render")
        elif (
            "asset" in lowered
            or "proposal-column" in lowered
            or "wide-melt-column" in lowered
            or "not-single-panel" in lowered
            or "not-proposed" in lowered
            or "already-eligible" in lowered
        ):
            categories.add("out_of_scope_or_preflight")
    expected_case = proposal.get("experiment_case")
    expected_family = (
        expected_case.get("chart_family") if isinstance(expected_case, Mapping) else None
    )
    expected_intent = (
        expected_case.get("intent") if isinstance(expected_case, Mapping) else None
    )
    expected_x = expected_intent.get("x") if isinstance(expected_intent, Mapping) else None
    expected_y = expected_intent.get("y") if isinstance(expected_intent, Mapping) else None
    for model_review in model_reviews:
        if not isinstance(model_review, Mapping):
            continue
        failure_code = model_review.get("failure_code")
        if isinstance(failure_code, str):
            lowered = failure_code.casefold()
            if "schema" in lowered or "json" in lowered:
                categories.add("schema_invalid")
            else:
                categories.add("operational")
        output = model_review.get("output")
        if not isinstance(output, Mapping):
            continue
        if output.get("chart_family") != expected_family:
            categories.add("chart_family")
        elif output.get("x") != expected_x or set(output.get("y") or []) != {
            expected_y
        }:
            categories.add("binding")
        elif output.get("valid") is False:
            categories.add("semantic_or_visual")
    if "model-proposed-correction" in reasons and not (
        {"chart_family", "binding"} & categories
    ):
        categories.add("binding_or_semantic_correction")
    if "model-valid-false" in reasons and not categories:
        categories.add("semantic_or_visual")
    if review.get("status") == "rejected" and not categories:
        categories.add("other_rejection")
    return sorted(categories)


def model_verdicts(review: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "request_model": item.get("request_model"),
            "served_model": item.get("served_model"),
            "status": item.get("status"),
            "failure_code": item.get("failure_code"),
            "valid": item.get("output", {}).get("valid")
            if isinstance(item.get("output"), Mapping)
            else None,
            "chart_family": item.get("output", {}).get("chart_family")
            if isinstance(item.get("output"), Mapping)
            else None,
            "x": item.get("output", {}).get("x")
            if isinstance(item.get("output"), Mapping)
            else None,
            "y": item.get("output", {}).get("y")
            if isinstance(item.get("output"), Mapping)
            else None,
            "reason": item.get("output", {}).get("reason")
            if isinstance(item.get("output"), Mapping)
            else None,
        }
        for item in review.get("model_reviews") or []
        if isinstance(item, Mapping)
    ]


def outcome_for_panels(panels: list[dict[str, Any]]) -> str:
    categories = {category for panel in panels for category in panel["failure_categories"]}
    if all(panel["status"] == "accepted" for panel in panels):
        return "verified"
    if "operational" in categories:
        return "unresolved_operational"
    if "schema_invalid" in categories:
        return "unresolved_schema"
    return "not_verified_content_rejection"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--proposed", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--freeze-manifest-internal-sha256", required=True)
    parser.add_argument("--freeze-input-label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    proposed = args.proposed.resolve(strict=True)
    if sha256_file(proposed) != EXPECTED_INPUT_SHA256:
        raise SystemExit("raw15 frozen proposal SHA-256 mismatch")
    validation = json.loads(
        (run_root / "raw15_artifact_validation.json").read_text(encoding="utf-8")
    )
    if validation.get("error_count") != 0:
        raise SystemExit("raw15 artifact validation did not pass")
    freeze_binding = frozen_v4_input_binding(
        freeze_manifest=args.freeze_manifest,
        expected_manifest_internal_sha256=args.freeze_manifest_internal_sha256,
        frozen_input=proposed,
        input_label=args.freeze_input_label,
    )
    if validation.get("frozen_v4_input_binding") != freeze_binding:
        raise SystemExit("raw15 artifact validation binding mismatch")

    proposals = jsonl(proposed)
    reviews = jsonl(run_root / "reviews.jsonl")
    proposal_by_id = {str(row["candidate_id"]): row for row in proposals}
    review_by_id = {str(row["candidate_id"]): row for row in reviews}
    singles = [
        row for row in proposals if row.get("proposal_type") == "single_panel"
    ]
    multis = [row for row in proposals if row.get("proposal_type") == "multi_panel"]
    if len(singles) != 124 or len(multis) != 15 or len({row["doi"] for row in multis}) != 15:
        raise SystemExit("raw15 frozen population mismatch")

    doi_results: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    raw_reason_counts: Counter[str] = Counter()
    operational_panel_ids: list[str] = []
    schema_panel_ids: list[str] = []
    for multi in sorted(multis, key=lambda row: (str(row["doi"]), str(row["candidate_id"]))):
        source_ids = [str(value) for value in multi.get("source_candidate_ids") or []]
        panels: list[dict[str, Any]] = []
        for source_id in source_ids:
            proposal = proposal_by_id[source_id]
            review = review_by_id[source_id]
            categories = failure_categories(proposal, review)
            category_counts.update(categories)
            raw_reason_counts.update(
                str(reason) for reason in review.get("rejection_reasons") or []
            )
            if "operational" in categories:
                operational_panel_ids.append(source_id)
            if "schema_invalid" in categories:
                schema_panel_ids.append(source_id)
            panels.append(
                {
                    "candidate_id": source_id,
                    "panel_id": panel_id(proposal),
                    "status": review.get("status"),
                    "review_hash": review.get("review_hash"),
                    "rejection_reasons": list(review.get("rejection_reasons") or []),
                    "failure_categories": categories,
                    "model_verdicts": model_verdicts(review),
                }
            )
        parent_review = review_by_id[str(multi["candidate_id"])]
        outcome = outcome_for_panels(panels)
        all_accepted = all(panel["status"] == "accepted" for panel in panels)
        if (parent_review.get("status") == "accepted") != all_accepted:
            raise SystemExit(f"raw15 parent derivation mismatch: {multi['candidate_id']}")
        doi_results.append(
            {
                "doi": multi["doi"],
                "multi_candidate_id": multi["candidate_id"],
                "parent_review_hash": parent_review.get("review_hash"),
                "parent_review_status": parent_review.get("status"),
                "all_child_panels_accepted": all_accepted,
                "verification_outcome": outcome,
                "parent_rejection_reasons": list(
                    parent_review.get("rejection_reasons") or []
                ),
                "failure_categories": sorted(
                    {category for panel in panels for category in panel["failure_categories"]}
                ),
                "panel_verdicts": panels,
            }
        )

    accepted_singles = sum(
        review_by_id[str(row["candidate_id"])].get("status") == "accepted"
        for row in singles
    )
    verified_dois = [
        result["doi"]
        for result in doi_results
        if result["verification_outcome"] == "verified"
    ]
    if len(verified_dois) != sum(
        review_by_id[str(row["candidate_id"])].get("status") == "accepted"
        for row in multis
    ):
        raise SystemExit("raw15 verified DOI count mismatch")
    result: dict[str, Any] = {
        "status": "completed_direct_raw_p5_v4_measurement",
        "scope": (
            "All-direct raw P5 15-DOI C2-extreme universe. This measurement cannot "
            "establish K=41-62 readiness because the universe ceiling is 15."
        ),
        "review_input_sha256": EXPECTED_INPUT_SHA256,
        "frozen_v4_input_binding": freeze_binding,
        "review_protocol": {
            "judge_models": ["claude-sonnet-4.6", "gemini-3.5-flash"],
            "rubric": "proposal-external-validation-v4",
            "rubric_hash": "46b3d7fe022f0ce17207dc822c2a8abbe309b2bf3308c1f0853b49cd8afc3c36",
        },
        "population": {
            "records": len(proposals),
            "single_panels": len(singles),
            "multi_parents": len(multis),
            "distinct_dois": len(doi_results),
        },
        "single_panel_pass_rate": {
            "accepted": accepted_singles,
            "reviewed": len(singles),
            "rate": accepted_singles / len(singles),
        },
        "N_raw": {
            "verified_distinct_multi_dois": len(verified_dois),
            "total_raw_multi_dois": len(doi_results),
            "verified_dois": verified_dois,
        },
        "gate_context": {
            "optimistic_K": 12,
            "holm_range_K": [41, 62],
            "meets_optimistic_K": len(verified_dois) >= 12,
            "can_meet_holm_range_given_universe_ceiling": False,
        },
        "per_doi": doi_results,
        "failed_doi_reason_category_counts": dict(sorted(category_counts.items())),
        "raw_panel_rejection_reason_occurrences": dict(sorted(raw_reason_counts.items())),
        "unresolved_noncontent": {
            "operational_panel_candidate_ids": sorted(set(operational_panel_ids)),
            "schema_invalid_panel_candidate_ids": sorted(set(schema_panel_ids)),
        },
        "artifact_validation": validation,
        "python_cross_validation": {
            "parent_count_matches_doi_count": len(doi_results) == 15,
            "all_parent_statuses_match_all_child_acceptance": True,
            "artifact_validation_discrepancies": validation.get("error_count"),
        },
    }
    write_json(args.out.resolve(), result)
    print(
        json.dumps(
            {
                "out": str(args.out.resolve()),
                "single_panel_pass_rate": result["single_panel_pass_rate"],
                "N_raw": result["N_raw"],
                "unresolved_noncontent": result["unresolved_noncontent"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
