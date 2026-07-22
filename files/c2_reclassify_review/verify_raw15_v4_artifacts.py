#!/usr/bin/env python3
"""Validate a completed direct-raw P5 V4 review without model calls."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from frozen_v4_binding import decorate_binding, frozen_v4_input_binding, sha256_file


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "files/c2_reclassify_review/v4_bound_runtime/nature_download"
sys.path.insert(0, str(RUNTIME))

import corpus.reviews as reviews


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def json_file(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(path.name + ".next")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def hash_valid(payload: Mapping[str, Any], field: str) -> bool:
    unhashed = dict(payload)
    declared = unhashed.pop(field, None)
    return isinstance(declared, str) and reviews._sha256_json(unhashed) == declared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--proposed", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--freeze-manifest-internal-sha256", required=True)
    parser.add_argument("--freeze-input-label", required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    proposed = args.proposed.resolve(strict=True)
    freeze_binding = frozen_v4_input_binding(
        freeze_manifest=args.freeze_manifest,
        expected_manifest_internal_sha256=args.freeze_manifest_internal_sha256,
        frozen_input=proposed,
        input_label=args.freeze_input_label,
    )
    input_hash = sha256_file(proposed)
    errors: list[str] = []
    proposals = jsonl(proposed)
    review_rows = jsonl(run_root / "reviews.jsonl")
    proposal_by_id = {str(row.get("candidate_id") or ""): row for row in proposals}
    review_by_id = {str(row.get("candidate_id") or ""): row for row in review_rows}
    if len(proposal_by_id) != len(proposals):
        errors.append("duplicate-proposal-id")
    if len(review_by_id) != len(review_rows):
        errors.append("duplicate-review-id")
    if set(proposal_by_id) != set(review_by_id):
        errors.append("proposal-review-id-set-mismatch")

    single_ids = {
        candidate_id
        for candidate_id, proposal in proposal_by_id.items()
        if proposal.get("proposal_type") == "single_panel"
    }
    accepted_single_ids = {
        candidate_id
        for candidate_id in single_ids
        if review_by_id.get(candidate_id, {}).get("status") == "accepted"
    }
    canonical_multi = reviews._canonical_multi_map(proposals)
    for candidate_id, record in review_by_id.items():
        try:
            reviews._validate_review_hash(record)
        except Exception:
            errors.append(f"invalid-review-hash:{candidate_id}")
            continue
        proposal = proposal_by_id.get(candidate_id)
        binding = record.get("binding")
        if proposal is None or not isinstance(binding, Mapping):
            errors.append(f"missing-proposal-or-binding:{candidate_id}")
            continue
        if (
            binding.get("frozen_v4_input_binding") != freeze_binding
            or binding.get("input_proposed_sha256") != input_hash
            or binding.get("rubric_hash") != reviews.REVIEW_RUBRIC_V4_HASH
        ):
            errors.append(f"freeze-or-rubric-binding-mismatch:{candidate_id}")
            continue
        try:
            state = reviews.GitState(
                commit=str(binding["code_commit"]), dirty=bool(binding["code_dirty"])
            )
            models = tuple(binding["request_models"])
            if models != ("claude-sonnet-4.6", "gemini-3.5-flash"):
                errors.append(f"judge-model-mismatch:{candidate_id}")
                continue
            if record.get("proposal_type") == "single_panel":
                rubric, families = reviews.REVIEW_RUBRICS[reviews.REVIEW_RUBRIC_V4_HASH]
                expected, _, _, _, _ = reviews._single_binding(
                    proposal,
                    input_hash=input_hash,
                    models=models,
                    git_state=state,
                    rubric=rubric,
                    rubric_hash=reviews.REVIEW_RUBRIC_V4_HASH,
                    allowed_chart_families=families,
                )
            elif record.get("proposal_type") == "multi_panel":
                expected = reviews._multi_binding(
                    proposal,
                    input_hash=input_hash,
                    models=models,
                    git_state=state,
                    rubric_hash=reviews.REVIEW_RUBRIC_V4_HASH,
                )
            else:
                errors.append(f"unsupported-proposal-type:{candidate_id}")
                continue
            expected = decorate_binding(
                expected,
                frozen_input_binding=freeze_binding,
                sha256_json=reviews._sha256_json,
            )
            if dict(binding) != expected:
                errors.append(f"recomputed-binding-mismatch:{candidate_id}")
        except Exception as exc:
            errors.append(f"binding-recompute-error:{candidate_id}:{type(exc).__name__}")

    for candidate_id, proposal in proposal_by_id.items():
        if proposal.get("proposal_type") != "multi_panel":
            continue
        record = review_by_id.get(candidate_id)
        if record is None or not isinstance(record.get("binding"), Mapping):
            continue
        try:
            recomputed = reviews._multi_review(
                proposal,
                binding=dict(record["binding"]),
                accepted_single_ids=accepted_single_ids,
                known_single_ids=single_ids,
                single_reviews=review_by_id,
                canonical_multi=canonical_multi,
            )
            if recomputed != record:
                errors.append(f"derived-multi-mismatch:{candidate_id}")
        except Exception as exc:
            errors.append(f"derived-multi-recompute-error:{candidate_id}:{type(exc).__name__}")

    evidence = json_file(run_root / "evidence.json")
    summary = json_file(run_root / "summary.json")
    for kind, payload, field in (
        ("evidence", evidence, "evidence_hash"),
        ("summary", summary, "summary_hash"),
    ):
        if not hash_valid(payload, field):
            errors.append(f"{kind}-hash-invalid")
        if payload.get("frozen_v4_input_binding") != freeze_binding:
            errors.append(f"{kind}-freeze-binding-mismatch")
        if payload.get("input_proposed_sha256") != input_hash:
            errors.append(f"{kind}-input-hash-mismatch")
        if payload.get("rubric_hash") != reviews.REVIEW_RUBRIC_V4_HASH:
            errors.append(f"{kind}-rubric-mismatch")
    sidecar = run_root / "summary.sha256"
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != summary.get(
        "summary_hash"
    ):
        errors.append("summary-sidecar-mismatch")

    counts = Counter(
        (str(row.get("proposal_type")), str(row.get("status"))) for row in review_rows
    )
    expected_summary = {
        "single_reviewed": counts[("single_panel", "accepted")]
        + counts[("single_panel", "rejected")],
        "single_accepted": counts[("single_panel", "accepted")],
        "multi_reviewed": counts[("multi_panel", "accepted")]
        + counts[("multi_panel", "rejected")],
        "multi_accepted": counts[("multi_panel", "accepted")],
        "rejected": counts[("single_panel", "rejected")]
        + counts[("multi_panel", "rejected")],
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            errors.append(f"summary-count-mismatch:{key}")
    if summary.get("review_hashes") != {
        str(row.get("candidate_id") or ""): str(row.get("review_hash") or "")
        for row in review_rows
    }:
        errors.append("summary-review-hashes-mismatch")
    accepted_ids = {
        str(row.get("candidate_id") or "")
        for row in review_rows
        if row.get("status") == "accepted"
    }
    evidence_ids = {
        str(item.get("candidate_id") or "")
        for item in evidence.get("verifications") or []
        if isinstance(item, Mapping)
    }
    if evidence_ids != accepted_ids:
        errors.append("evidence-accepted-id-set-mismatch")

    source_component_ids = [
        source_id
        for proposal in proposal_by_id.values()
        if proposal.get("proposal_type") == "multi_panel"
        for source_id in proposal.get("source_candidate_ids") or []
    ]
    if set(source_component_ids) != single_ids or len(source_component_ids) != len(
        set(source_component_ids)
    ):
        errors.append("multi-components-not-exactly-one-to-one")

    result = {
        "method": "Independent direct-raw V4 provenance, binding, review-hash, derived-multi, evidence, and summary validation; no model calls.",
        "frozen_v4_input_binding": freeze_binding,
        "review_count": len(review_rows),
        "counts": expected_summary,
        "review_sha256": sha256_file(run_root / "reviews.jsonl"),
        "errors": errors,
        "error_count": len(errors),
    }
    write_json(run_root / "raw15_artifact_validation.json", result)
    print(json.dumps(result, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
