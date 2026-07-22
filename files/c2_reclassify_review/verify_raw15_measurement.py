#!/usr/bin/env python3
"""Independently recount the direct-raw P5 V4 measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


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


def write_json(path: Path, value: dict[str, Any]) -> None:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(path.name + ".next")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--proposed", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    proposed = args.proposed.resolve(strict=True)
    result_path = args.result.resolve(strict=True)
    discrepancies: list[str] = []
    if sha256_file(proposed) != EXPECTED_INPUT_SHA256:
        discrepancies.append("frozen-input-sha256")
    proposals = jsonl(proposed)
    reviews = jsonl(run_root / "reviews.jsonl")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validation = json.loads(
        (run_root / "raw15_artifact_validation.json").read_text(encoding="utf-8")
    )
    proposal_by_id = {str(row.get("candidate_id") or ""): row for row in proposals}
    review_by_id = {str(row.get("candidate_id") or ""): row for row in reviews}
    if len(proposals) != 139 or len(reviews) != 139:
        discrepancies.append("record-count")
    if set(proposal_by_id) != set(review_by_id):
        discrepancies.append("proposal-review-id-set")
    singles = [row for row in proposals if row.get("proposal_type") == "single_panel"]
    multis = [row for row in proposals if row.get("proposal_type") == "multi_panel"]
    accepted_singles = sum(
        review_by_id[str(row["candidate_id"])].get("status") == "accepted"
        for row in singles
    )
    accepted_multis = sum(
        review_by_id[str(row["candidate_id"])].get("status") == "accepted"
        for row in multis
    )
    verified_dois = sorted(
        str(row["doi"])
        for row in multis
        if review_by_id[str(row["candidate_id"])].get("status") == "accepted"
    )
    if len(singles) != 124 or len(multis) != 15 or len({row["doi"] for row in multis}) != 15:
        discrepancies.append("raw15-population")
    if result.get("review_input_sha256") != EXPECTED_INPUT_SHA256:
        discrepancies.append("result-input-sha256")
    single_rate = result.get("single_panel_pass_rate") or {}
    if (
        single_rate.get("accepted") != accepted_singles
        or single_rate.get("reviewed") != len(singles)
        or single_rate.get("rate") != accepted_singles / len(singles)
    ):
        discrepancies.append("single-pass-rate")
    n_raw = result.get("N_raw") or {}
    if (
        n_raw.get("verified_distinct_multi_dois") != accepted_multis
        or n_raw.get("total_raw_multi_dois") != len(multis)
        or n_raw.get("verified_dois") != verified_dois
    ):
        discrepancies.append("N-raw")
    per_doi = result.get("per_doi") or []
    by_doi_result = {str(item.get("doi") or ""): item for item in per_doi}
    if len(by_doi_result) != len(multis):
        discrepancies.append("per-doi-cardinality")
    schema_panels: set[str] = set()
    operational_panels: set[str] = set()
    for multi in multis:
        doi = str(multi["doi"])
        measured = by_doi_result.get(doi)
        parent_review = review_by_id[str(multi["candidate_id"])]
        source_ids = [str(source_id) for source_id in multi.get("source_candidate_ids") or []]
        if measured is None:
            discrepancies.append(f"missing-doi:{doi}")
            continue
        panels = {str(item.get("candidate_id") or ""): item for item in measured.get("panel_verdicts") or []}
        if set(panels) != set(source_ids):
            discrepancies.append(f"panel-set:{doi}")
        all_accepted = all(
            review_by_id[source_id].get("status") == "accepted"
            for source_id in source_ids
        )
        if measured.get("all_child_panels_accepted") != all_accepted:
            discrepancies.append(f"all-children:{doi}")
        if (parent_review.get("status") == "accepted") != all_accepted:
            discrepancies.append(f"parent-rule:{doi}")
        for source_id in source_ids:
            panel = panels.get(source_id, {})
            review = review_by_id[source_id]
            if panel.get("status") != review.get("status"):
                discrepancies.append(f"panel-status:{source_id}")
            failures = {
                str(model.get("failure_code") or "")
                for model in review.get("model_reviews") or []
                if model.get("failure_code")
            }
            if "model-output-schema-invalid" in failures:
                schema_panels.add(source_id)
            if failures - {"model-output-schema-invalid"}:
                operational_panels.add(source_id)
    unresolved = result.get("unresolved_noncontent") or {}
    if set(unresolved.get("schema_invalid_panel_candidate_ids") or []) != schema_panels:
        discrepancies.append("schema-invalid-panels")
    if set(unresolved.get("operational_panel_candidate_ids") or []) != operational_panels:
        discrepancies.append("operational-panels")
    if validation.get("error_count") != 0:
        discrepancies.append("artifact-validation")

    output = {
        "method": "Independent direct raw JSONL recount of raw15 result; no measurement helpers and no model calls.",
        "review_input_sha256": EXPECTED_INPUT_SHA256,
        "recount": {
            "single_reviewed": len(singles),
            "single_accepted": accepted_singles,
            "single_panel_pass_rate": accepted_singles / len(singles),
            "multi_reviewed": len(multis),
            "verified_distinct_multi_dois_N_raw": accepted_multis,
            "verified_dois": verified_dois,
            "schema_invalid_panels": sorted(schema_panels),
            "operational_panels": sorted(operational_panels),
        },
        "discrepancies": discrepancies,
        "discrepancy_count": len(discrepancies),
    }
    write_json(run_root / "raw15_independent_crosscheck.json", output)
    print(json.dumps(output, ensure_ascii=False))
    if discrepancies:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
