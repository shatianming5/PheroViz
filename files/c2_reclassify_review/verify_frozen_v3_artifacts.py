#!/usr/bin/env python3
"""Independently validate fresh frozen-V3 review artifacts without model calls."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from frozen_v3_binding import decorate_binding, frozen_v3_input_binding, sha256_file


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "files/c2_reclassify_review/v3_bound_runtime/nature_download"
sys.path.insert(0, str(RUNTIME))

import corpus.reviews as reviews


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def json_file(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".next")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def hash_valid(payload: Mapping[str, Any], field: str) -> bool:
    sealed = payload.get(field)
    unhashed = dict(payload)
    unhashed.pop(field, None)
    return isinstance(sealed, str) and reviews._sha256_json(unhashed) == sealed


def verify_batch(
    *,
    label: str,
    proposed: Path,
    out: Path,
    freeze_manifest: Path,
    manifest_internal_hash: str,
) -> dict[str, Any]:
    errors: list[str] = []
    binding_metadata = frozen_v3_input_binding(
        freeze_manifest=freeze_manifest,
        expected_manifest_internal_sha256=manifest_internal_hash,
        frozen_input=proposed,
        input_label=f"priority_{label}",
    )
    input_hash = sha256_file(proposed)
    proposal_rows = jsonl(proposed)
    proposal_by_id = {str(row.get("candidate_id") or ""): row for row in proposal_rows}
    if len(proposal_by_id) != len(proposal_rows):
        errors.append("duplicate-proposal-id")
    review_rows = jsonl(out / label / "reviews.jsonl")
    review_by_id = {str(row.get("candidate_id") or ""): row for row in review_rows}
    if len(review_by_id) != len(review_rows):
        errors.append("duplicate-review-id")
    if set(proposal_by_id) != set(review_by_id):
        errors.append("proposal-review-id-set-mismatch")

    singles = {
        candidate_id
        for candidate_id, proposal in proposal_by_id.items()
        if proposal.get("proposal_type") == "single_panel"
    }
    accepted_singles = {
        candidate_id
        for candidate_id in singles
        if review_by_id.get(candidate_id, {}).get("status") == "accepted"
    }
    canonical_multi = reviews._canonical_multi_map(proposal_rows)
    for candidate_id, record in review_by_id.items():
        try:
            reviews._validate_review_hash(record)
        except Exception:
            errors.append(f"invalid-review-hash:{candidate_id}")
            continue
        proposal = proposal_by_id.get(candidate_id)
        if proposal is None:
            continue
        binding = record.get("binding")
        if not isinstance(binding, Mapping):
            errors.append(f"missing-binding:{candidate_id}")
            continue
        if binding.get("frozen_v3_input_binding") != binding_metadata:
            errors.append(f"freeze-binding-mismatch:{candidate_id}")
            continue
        if binding.get("input_proposed_sha256") != input_hash:
            errors.append(f"input-hash-mismatch:{candidate_id}")
            continue
        try:
            state = reviews.GitState(
                commit=str(binding["code_commit"]), dirty=bool(binding["code_dirty"])
            )
            models = tuple(binding["request_models"])
            rubric_hash = str(binding["rubric_hash"])
            if record.get("proposal_type") == "single_panel":
                rubric, families = reviews.REVIEW_RUBRICS[rubric_hash]
                expected, _, _, _, _ = reviews._single_binding(
                    proposal,
                    input_hash=input_hash,
                    models=models,
                    git_state=state,
                    rubric=rubric,
                    rubric_hash=rubric_hash,
                    allowed_chart_families=families,
                )
            elif record.get("proposal_type") == "multi_panel":
                expected = reviews._multi_binding(
                    proposal,
                    input_hash=input_hash,
                    models=models,
                    git_state=state,
                    rubric_hash=rubric_hash,
                )
            else:
                errors.append(f"unsupported-proposal-type:{candidate_id}")
                continue
            expected = decorate_binding(
                expected,
                frozen_input_binding=binding_metadata,
                sha256_json=reviews._sha256_json,
            )
            if dict(binding) != expected:
                errors.append(f"recomputed-binding-mismatch:{candidate_id}")
        except Exception as exc:
            errors.append(f"binding-recompute-error:{candidate_id}:{type(exc).__name__}")

    for candidate_id, proposal in proposal_by_id.items():
        if proposal.get("proposal_type") != "multi_panel" or candidate_id not in review_by_id:
            continue
        record = review_by_id[candidate_id]
        binding = record.get("binding")
        if not isinstance(binding, Mapping):
            continue
        try:
            recomputed = reviews._multi_review(
                proposal,
                binding=dict(binding),
                accepted_single_ids=accepted_singles,
                known_single_ids=singles,
                single_reviews=review_by_id,
                canonical_multi=canonical_multi,
            )
            if recomputed != record:
                errors.append(f"derived-multi-mismatch:{candidate_id}")
        except Exception as exc:
            errors.append(f"derived-multi-recompute-error:{candidate_id}:{type(exc).__name__}")

    evidence = json_file(out / label / "evidence.json")
    summary = json_file(out / label / "summary.json")
    for kind, payload, field in (
        ("evidence", evidence, "evidence_hash"),
        ("summary", summary, "summary_hash"),
    ):
        if not hash_valid(payload, field):
            errors.append(f"{kind}-hash-invalid")
        if payload.get("frozen_v3_input_binding") != binding_metadata:
            errors.append(f"{kind}-freeze-binding-mismatch")
        if payload.get("input_proposed_sha256") != input_hash:
            errors.append(f"{kind}-input-hash-mismatch")
    counts = Counter(
        (str(record.get("proposal_type")), str(record.get("status")))
        for record in review_rows
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
    expected_review_hashes = {
        str(record.get("candidate_id") or ""): str(record.get("review_hash") or "")
        for record in review_rows
    }
    if summary.get("review_hashes") != expected_review_hashes:
        errors.append("summary-review-hashes-mismatch")
    accepted_ids = {
        str(record.get("candidate_id") or "")
        for record in review_rows
        if record.get("status") == "accepted"
    }
    evidence_ids = {
        str(item.get("candidate_id") or "")
        for item in evidence.get("verifications") or []
        if isinstance(item, Mapping)
    }
    if evidence_ids != accepted_ids:
        errors.append("evidence-accepted-id-set-mismatch")
    return {
        "label": label,
        "frozen_v3_input_binding": binding_metadata,
        "review_count": len(review_rows),
        "counts": {
            "single_reviewed": expected_summary["single_reviewed"],
            "single_accepted": expected_summary["single_accepted"],
            "multi_reviewed": expected_summary["multi_reviewed"],
            "multi_accepted": expected_summary["multi_accepted"],
        },
        "review_sha256": sha256_file(out / label / "reviews.jsonl"),
        "errors": errors,
        "error_count": len(errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--freeze-manifest-internal-sha256", required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    frozen_root = args.freeze_manifest.resolve().parent
    results = {}
    for label in ("casecount_strict", "full_strict"):
        results[label] = verify_batch(
            label=label,
            proposed=frozen_root / f"priority_{label}.proposed.jsonl",
            out=run_root,
            freeze_manifest=args.freeze_manifest,
            manifest_internal_hash=args.freeze_manifest_internal_sha256,
        )
    errors = [
        f"{label}:{error}"
        for label, result in results.items()
        for error in result["errors"]
    ]
    result = {
        "method": (
            "Independent frozen-V3 provenance, hash, binding, derived-multi, "
            "evidence, and summary recount; no model calls."
        ),
        "freeze_manifest": str(args.freeze_manifest.resolve()),
        "freeze_manifest_internal_sha256": args.freeze_manifest_internal_sha256,
        "batches": results,
        "discrepancies": errors,
        "discrepancy_count": len(errors),
    }
    write_json(run_root / "frozen_v3_artifact_validation.json", result)
    print(json.dumps(result, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
