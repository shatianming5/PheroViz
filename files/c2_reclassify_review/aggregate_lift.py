#!/usr/bin/env python3
"""Consolidate two canonical re-review batches and measure the paired C2 lift."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from frozen_v3_binding import canonical_sha256

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "files/c2_reclassify_review"
BASELINE_ROOT = Path(
    "/Users/tommy/.copilot/session-state/"
    "7b81726b-937c-41cb-9392-fead4d53250b/files/c2_review_scale"
)
BATCHES = {
    "casecount_strict": {
        "baseline_reviews": BASELINE_ROOT / "casecount_strict/reviews.jsonl",
        "baseline_proposals": ROOT
        / "nature_download/outputs/c2_full_casecount_exploratory_20260720/"
        "corrected_sheet_binding/strict/proposals/proposed.jsonl",
    },
    "full_strict": {
        "baseline_reviews": BASELINE_ROOT / "full_strict/reviews.jsonl",
        "baseline_proposals": ROOT / "nature_download/outputs/c2_full_proposals/proposed.jsonl",
    },
}
OPERATIONAL_FAILURE_CODES = frozenset(
    {"model-call-failed", "model-client-init-failed"}
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        records.append(value)
    return records


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".next")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".next")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def status_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int | float | None]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        counts[str(record.get("proposal_type"))][str(record.get("status"))] += 1
    result: dict[str, dict[str, int | float | None]] = {}
    for proposal_type in ("single_panel", "multi_panel"):
        accepted = counts[proposal_type]["accepted"]
        rejected = counts[proposal_type]["rejected"]
        reviewed = accepted + rejected
        result[proposal_type] = {
            "reviewed": reviewed,
            "accepted": accepted,
            "rejected": rejected,
            "pass_rate": rate(accepted, reviewed),
        }
    return result


def expected_binding(proposal: Mapping[str, Any]) -> tuple[str, str, set[str]]:
    case = proposal.get("experiment_case") or {}
    intent = case.get("intent") if isinstance(case, Mapping) else {}
    if not isinstance(intent, Mapping):
        intent = {}
    raw_series = intent.get("series")
    y = raw_series if isinstance(raw_series, list) and raw_series else [intent.get("y")]
    return (
        str(case.get("chart_family") or ""),
        str(intent.get("x") or ""),
        {str(value) for value in y if isinstance(value, str) and value},
    )


def failure_codes(review: Mapping[str, Any]) -> set[str]:
    return {
        str(item.get("failure_code"))
        for item in review.get("model_reviews") or []
        if isinstance(item, Mapping)
        and item.get("status") != "completed"
        and item.get("failure_code")
    }


def triage(review: Mapping[str, Any]) -> str:
    if review.get("status") == "accepted":
        return "accepted"
    if failure_codes(review) & OPERATIONAL_FAILURE_CODES:
        return "operational_error_unresolved"
    if "model-output-schema-invalid" in set(review.get("rejection_reasons") or []):
        return "schema_invalid_unresolved"
    return "content_rejected"


def residual_bucket(proposal: Mapping[str, Any], review: Mapping[str, Any]) -> str:
    status = triage(review)
    if status == "operational_error_unresolved":
        return "operational_error"
    if status == "schema_invalid_unresolved":
        return "schema_invalid"
    family, x, y = expected_binding(proposal)
    outputs = [
        item.get("output")
        for item in review.get("model_reviews") or []
        if isinstance(item, Mapping)
        and item.get("status") == "completed"
        and isinstance(item.get("output"), Mapping)
    ]
    alternatives = {str(output.get("chart_family") or "") for output in outputs}
    if any(value and value not in {"line", "bar", "scatter"} for value in alternatives):
        return "genuinely_non_line_bar_scatter"
    if family in {"bar", "scatter"} and any(
        value in {"bar", "scatter"} and value != family for value in alternatives
    ):
        return "residual_bar_scatter_subtype"
    if any(value != family for value in alternatives):
        return "other_line_bar_scatter_family_mismatch"
    if any(
        str(output.get("x") or "") != x
        or not isinstance(output.get("y"), list)
        or {str(value) for value in output["y"]} != y
        for output in outputs
    ):
        return "still_wrong_binding"
    if "model-valid-false" in set(review.get("rejection_reasons") or []):
        return "other_visual_or_semantic_mismatch"
    return "other_rejection"


def verify_review_hashes(reviews: Iterable[Mapping[str, Any]]) -> list[str]:
    sys.path.insert(0, str(ROOT / "nature_download"))
    from corpus.reviews import _validate_review_hash

    discrepancies: list[str] = []
    for review in reviews:
        try:
            _validate_review_hash(review)
        except Exception as exc:
            discrepancies.append(
                f"invalid-review-hash:{review.get('candidate_id')}:{type(exc).__name__}"
            )
    return discrepancies


def baseline_for_batch(label: str, paths: Mapping[str, Path]) -> tuple[
    list[dict[str, Any]], dict[str, dict[str, Any]], set[str], list[str]
]:
    reviews = read_jsonl(paths["baseline_reviews"])
    proposals = read_jsonl(paths["baseline_proposals"])
    by_id = {str(proposal.get("candidate_id") or ""): proposal for proposal in proposals}
    discrepancies: list[str] = []
    if len(by_id) != len(proposals):
        discrepancies.append(f"{label}:baseline-duplicate-proposal-id")
    if {str(review.get("candidate_id") or "") for review in reviews} != set(by_id):
        discrepancies.append(f"{label}:baseline-proposal-review-id-set-mismatch")
    rejected_single = {
        str(review["candidate_id"]): review
        for review in reviews
        if review.get("proposal_type") == "single_panel"
        and review.get("status") == "rejected"
    }
    accepted_multi_dois = {
        str(by_id[str(review["candidate_id"])]["doi"])
        for review in reviews
        if review.get("proposal_type") == "multi_panel"
        and review.get("status") == "accepted"
        and isinstance(by_id.get(str(review.get("candidate_id") or ""), {}).get("doi"), str)
    }
    return reviews, rejected_single, accepted_multi_dois, discrepancies


def probe_result() -> dict[str, Any]:
    path = OUT / "probe_casecount/reviews.jsonl"
    if not path.is_file():
        return {"available": False}
    records = read_jsonl(path)
    model_reviews = [
        item
        for record in records
        for item in record.get("model_reviews") or []
        if isinstance(item, Mapping)
    ]
    passed = (
        len(records) == 3
        and len(model_reviews) == 6
        and all(
            item.get("status") == "completed"
            and item.get("failure_code") is None
            and isinstance(item.get("output"), Mapping)
            and {"valid", "chart_family", "x", "y"} <= set(item["output"])
            for item in model_reviews
        )
        and {item.get("request_model") for item in model_reviews}
        == {"claude-sonnet-4.6", "gemini-3.5-flash"}
    )
    return {
        "available": True,
        "passed": passed,
        "records": len(records),
        "model_reviews": len(model_reviews),
        "requested_models": sorted(
            {str(item.get("request_model")) for item in model_reviews}
        ),
    }


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--freeze-manifest-internal-sha256", required=True)
    args = parser.parse_args()
    OUT = args.run_root.resolve()
    manifest_path = args.freeze_manifest.resolve(strict=True)
    manifest = read_json(manifest_path)
    manifest_internal = manifest.get("manifest_sha256")
    if (
        not isinstance(manifest_internal, str)
        or manifest_internal != args.freeze_manifest_internal_sha256
        or canonical_sha256(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        )
        != manifest_internal
    ):
        raise SystemExit("frozen V3 manifest internal SHA-256 mismatch")
    frozen_entries = {
        str(item.get("label") or ""): item
        for item in manifest.get("inputs", [])
        if isinstance(item, Mapping)
    }
    all_reviews: list[dict[str, Any]] = []
    partition_index: dict[str, Any] = {}
    batch_results: dict[str, Any] = {}
    baseline_reviews_all: list[dict[str, Any]] = []
    baseline_multi_dois: set[str] = set()
    reclassified_multi_dois: set[str] = set()
    discrepancies: list[str] = []
    residuals: Counter[str] = Counter()
    triages: Counter[str] = Counter()
    raw_single_reasons: Counter[str] = Counter()
    raw_multi_reasons: Counter[str] = Counter()
    operational_errors: list[dict[str, Any]] = []

    for label, paths in BATCHES.items():
        frozen_entry = frozen_entries.get(f"priority_{label}")
        if not isinstance(frozen_entry, Mapping):
            raise SystemExit(f"frozen V3 manifest lacks priority_{label}")
        input_path = Path(str(frozen_entry.get("frozen_path") or "")).resolve(
            strict=True
        )
        review_dir = OUT / label
        review_path = review_dir / "reviews.jsonl"
        summary_path = review_dir / "summary.json"
        evidence_path = review_dir / "evidence.json"
        proposals = read_jsonl(input_path)
        reviews = read_jsonl(review_path)
        summary = read_json(summary_path)
        proposal_by_id = {str(item.get("candidate_id") or ""): item for item in proposals}
        review_by_id = {str(item.get("candidate_id") or ""): item for item in reviews}
        if len(proposal_by_id) != len(proposals):
            discrepancies.append(f"{label}:duplicate-proposal-candidate-id")
        if len(review_by_id) != len(reviews):
            discrepancies.append(f"{label}:duplicate-review-candidate-id")
        if set(proposal_by_id) != set(review_by_id):
            discrepancies.append(f"{label}:proposal-review-id-set-mismatch")
        batch_markers = {
            str(proposal["reproposal_batch"])
            for proposal in proposals
            if isinstance(proposal.get("reproposal_batch"), str)
        }
        if batch_markers and batch_markers != {label}:
            discrepancies.append(f"{label}:wrong-reproposal-batch-marker")
        if frozen_entry.get("frozen_sha256") != sha256_file(input_path):
            discrepancies.append(f"{label}:input-sha256-mismatch")
        expected_freeze_binding = {
            "schema_version": "c2-v3-frozen-review-input-binding-v1",
            "v3_freeze_manifest": {
                "path": str(manifest_path),
                "internal_sha256": manifest_internal,
                "file_sha256": sha256_file(manifest_path),
            },
            "frozen_proposal_input": {
                "label": f"priority_{label}",
                "path": str(input_path),
                "sha256": sha256_file(input_path),
                "records": frozen_entry.get("records"),
                "singles": frozen_entry.get("singles"),
                "multi_parents": frozen_entry.get("multi_parents"),
                "scope": frozen_entry.get("scope"),
                "readonly": True,
            },
            "requires_fresh_review": True,
        }
        if any(
            review.get("binding", {}).get("frozen_v3_input_binding")
            != expected_freeze_binding
            for review in reviews
        ):
            discrepancies.append(f"{label}:frozen-v3-review-binding-mismatch")
        if summary.get("frozen_v3_input_binding") != expected_freeze_binding:
            discrepancies.append(f"{label}:frozen-v3-summary-binding-mismatch")
        discrepancies.extend(f"{label}:{item}" for item in verify_review_hashes(reviews))

        counts = status_counts(reviews)
        expected_summary = {
            "single_reviewed": counts["single_panel"]["reviewed"],
            "single_accepted": counts["single_panel"]["accepted"],
            "multi_reviewed": counts["multi_panel"]["reviewed"],
            "multi_accepted": counts["multi_panel"]["accepted"],
            "rejected": counts["single_panel"]["rejected"]
            + counts["multi_panel"]["rejected"],
        }
        for field, expected in expected_summary.items():
            if summary.get(field) != expected:
                discrepancies.append(
                    f"{label}:summary-{field}-expected-{expected}-actual-{summary.get(field)}"
                )
        if not evidence_path.is_file():
            discrepancies.append(f"{label}:evidence-missing")
            evidence_ids: set[str] = set()
        else:
            evidence = read_json(evidence_path)
            if evidence.get("frozen_v3_input_binding") != expected_freeze_binding:
                discrepancies.append(f"{label}:frozen-v3-evidence-binding-mismatch")
            evidence_ids = {
                str(item.get("candidate_id") or "")
                for item in evidence.get("verifications") or []
                if isinstance(item, Mapping)
            }
            accepted_ids = {
                str(review.get("candidate_id") or "")
                for review in reviews
                if review.get("status") == "accepted"
            }
            if evidence_ids != accepted_ids:
                discrepancies.append(f"{label}:evidence-accepted-id-mismatch")

        baseline_reviews, baseline_rejects, baseline_dois, baseline_errors = baseline_for_batch(
            label, paths
        )
        baseline_reviews_all.extend(baseline_reviews)
        baseline_multi_dois.update(baseline_dois)
        discrepancies.extend(baseline_errors)
        single_ids = {
            str(proposal.get("candidate_id") or "")
            for proposal in proposals
            if proposal.get("proposal_type") == "single_panel"
        }
        if single_ids != set(baseline_rejects):
            discrepancies.append(f"{label}:paired-former-reject-id-set-mismatch")

        for proposal in proposals:
            candidate_id = str(proposal.get("candidate_id") or "")
            review = review_by_id.get(candidate_id)
            if review is None:
                continue
            if proposal.get("proposal_type") == "single_panel":
                outcome = triage(review)
                triages[outcome] += 1
                if review.get("status") != "accepted":
                    residuals[residual_bucket(proposal, review)] += 1
                    raw_single_reasons.update(
                        str(reason) for reason in review.get("rejection_reasons") or []
                    )
                codes = sorted(failure_codes(review) & OPERATIONAL_FAILURE_CODES)
                if codes:
                    operational_errors.append(
                        {
                            "batch": label,
                            "candidate_id": candidate_id,
                            "failure_codes": codes,
                            "raw_review_status": review.get("status"),
                        }
                    )
            elif (
                proposal.get("proposal_type") == "multi_panel"
                and review.get("status") == "accepted"
                and isinstance(proposal.get("doi"), str)
            ):
                reclassified_multi_dois.add(proposal["doi"])
            elif proposal.get("proposal_type") == "multi_panel":
                raw_multi_reasons.update(
                    str(reason) for reason in review.get("rejection_reasons") or []
                )

        all_reviews.extend(reviews)
        partition_index[label] = {
            "input": str(input_path),
            "input_sha256": sha256_file(input_path),
            "frozen_v3_input_binding": expected_freeze_binding,
            "reviews": str(review_path),
            "reviews_sha256": sha256_file(review_path),
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "records": len(reviews),
            "review_hashes": {
                str(review.get("candidate_id") or ""): str(review.get("review_hash") or "")
                for review in reviews
            },
        }
        batch_results[label] = {
            "reclassified_status_counts": counts,
            "baseline_status_counts": status_counts(baseline_reviews),
            "baseline_former_rejects": len(baseline_rejects),
            "accepted_reclassified_multis": sorted(
                proposal["doi"]
                for proposal in proposals
                if proposal.get("proposal_type") == "multi_panel"
                and isinstance(proposal.get("doi"), str)
                and review_by_id.get(str(proposal.get("candidate_id") or ""), {}).get(
                    "status"
                )
                == "accepted"
            ),
        }

    aggregate_counts = status_counts(all_reviews)
    baseline_counts = status_counts(baseline_reviews_all)
    n_prime_dois = baseline_multi_dois | reclassified_multi_dois
    root_summary = {
        "schema_version": "c2-reclassify-composite-v1",
        "composite_reason": (
            "The source aggregate contains duplicate candidate IDs across two distinct "
            "baseline provenance batches. Each batch was canonically reviewed separately; "
            "this root is a read-only concatenation keyed by (reproposal_batch, candidate_id)."
        ),
        "input_manifest_sha256": sha256_file(manifest_path),
        "frozen_v3_manifest": str(manifest_path),
        "frozen_v3_manifest_internal_sha256": manifest_internal,
        "review_partitions": partition_index,
        "single_reviewed": aggregate_counts["single_panel"]["reviewed"],
        "single_accepted": aggregate_counts["single_panel"]["accepted"],
        "multi_reviewed": aggregate_counts["multi_panel"]["reviewed"],
        "multi_accepted": aggregate_counts["multi_panel"]["accepted"],
        "rejected": aggregate_counts["single_panel"]["rejected"]
        + aggregate_counts["multi_panel"]["rejected"],
        "evidence_records": aggregate_counts["single_panel"]["accepted"]
        + aggregate_counts["multi_panel"]["accepted"],
        "judge_models": ["claude-sonnet-4.6", "gemini-3.5-flash"],
    }
    atomic_write_jsonl(OUT / "reviews.jsonl", all_reviews)
    atomic_write_json(OUT / "review_partitions.json", partition_index)
    atomic_write_json(OUT / "summary.json", root_summary)

    total_single = int(aggregate_counts["single_panel"]["reviewed"])
    accepted_single = int(aggregate_counts["single_panel"]["accepted"])
    total_multi = int(aggregate_counts["multi_panel"]["reviewed"])
    accepted_multi = int(aggregate_counts["multi_panel"]["accepted"])
    total_baseline_rejects = sum(
        result["baseline_former_rejects"] for result in batch_results.values()
    )
    if total_single != total_baseline_rejects:
        discrepancies.append(
            f"paired:single-count-{total_single}-does-not-equal-baseline-reject-count-"
            f"{total_baseline_rejects}"
        )

    lift = {
        "measurement_method": (
            "The two immutable V3 priority batches are separately reviewed because "
            "candidate IDs are only unique within their baseline provenance batch. "
            "All 203 former-reject single records are paired one-to-one with their "
            "same-batch baseline reject by candidate_id."
        ),
        "measurement_scope": {
            "classification": "diagnostic-only historic-reject-selected V3 subset",
            "not_a_sealed_C2_universe_claim": True,
            "reason": (
                "A1's frozen V3 priority inputs contain the 203 historic baseline "
                "single-panel rejects and their parents. The measured rates describe "
                "only that reclassified subset, not a final sealed-C2 universe."
            ),
        },
        "probe": probe_result(),
        "input": {
            "frozen_v3_manifest": str(manifest_path),
            "frozen_v3_manifest_file_sha256": sha256_file(manifest_path),
            "frozen_v3_manifest_internal_sha256": manifest_internal,
            "partitions": partition_index,
        },
        "baseline": {
            "aggregate_pass_rates": baseline_counts,
            "verified_multi_unique_doi_count_N": len(baseline_multi_dois),
            "verified_multi_dois": sorted(baseline_multi_dois),
            "former_single_rejects": total_baseline_rejects,
            "per_batch": {
                label: {
                    "former_single_rejects": result["baseline_former_rejects"],
                    "pass_rates": result["baseline_status_counts"],
                }
                for label, result in batch_results.items()
            },
        },
        "reclassified_set": {
            "aggregate_pass_rates": aggregate_counts,
            "single_panel": {
                "reviewed": total_single,
                "accepted": accepted_single,
                "raw_pass_rate": rate(accepted_single, total_single),
            },
            "multi_panel": {
                "reviewed": total_multi,
                "accepted": accepted_multi,
                "raw_pass_rate": rate(accepted_multi, total_multi),
            },
            "per_batch": {
                label: result["reclassified_status_counts"]
                for label, result in batch_results.items()
            },
        },
        "former_reject_to_accept": {
            "accepted": accepted_single,
            "former_rejects": total_baseline_rejects,
            "conversion_rate": rate(accepted_single, total_baseline_rejects),
            "raw_review_status_counts": {
                "accepted": accepted_single,
                "rejected": total_single - accepted_single,
            },
        },
        "adjudication_triage": {
            "single_panel": dict(sorted(triages.items())),
            "non_content_operational_errors": operational_errors,
            "note": (
                "Gateway/model-call failures are unresolved operational errors, not "
                "content rejects. Model-output-schema-invalid is also reported "
                "separately rather than silently relabeled as a content rejection."
            ),
        },
        "residual_single_reject_primary_breakdown": {
            "residual_bar_scatter_subtype": residuals[
                "residual_bar_scatter_subtype"
            ],
            "still_wrong_binding": residuals["still_wrong_binding"],
            "schema_invalid": residuals["schema_invalid"],
            "genuinely_non_line_bar_scatter": residuals[
                "genuinely_non_line_bar_scatter"
            ],
            "other_line_bar_scatter_family_mismatch": residuals[
                "other_line_bar_scatter_family_mismatch"
            ],
            "other_visual_or_semantic_mismatch": residuals[
                "other_visual_or_semantic_mismatch"
            ],
            "other_rejection": residuals["other_rejection"],
            "operational_error": residuals["operational_error"],
            "total_raw_rejected_singles": total_single - accepted_single,
        },
        "raw_rejection_reason_occurrences": {
            "single_panel": dict(sorted(raw_single_reasons.items())),
            "multi_panel": dict(sorted(raw_multi_reasons.items())),
        },
        "verified_multi_doi": {
            "baseline_N": len(baseline_multi_dois),
            "baseline_dois": sorted(baseline_multi_dois),
            "reclassified_accepted_multi_doi_count": len(reclassified_multi_dois),
            "reclassified_accepted_multi_dois": sorted(reclassified_multi_dois),
            "N_prime": len(n_prime_dois),
            "N_prime_dois": sorted(n_prime_dois),
            "K": 12,
            "reaches_K": len(n_prime_dois) >= 12,
        },
        "python_cross_validation": {
            "checks": {
                "all_review_records_hash_valid": not any(
                    "invalid-review-hash:" in item for item in discrepancies
                ),
                "all_partitions_pair_exactly_to_baseline_reject_ids": not any(
                    "paired-former-reject-id-set-mismatch" in item
                    for item in discrepancies
                ),
                "composite_single_count_equals_former_reject_count": total_single
                == total_baseline_rejects,
            },
            "discrepancies": discrepancies,
            "discrepancy_count": len(discrepancies),
        },
    }
    atomic_write_json(OUT / "lift_measurement.json", lift)

    conversion = lift["former_reject_to_accept"]
    multi = lift["verified_multi_doi"]
    verdict = (
        "# C2 reclassification verdict\n\n"
        "- Scope: diagnostic-only historic-reject-selected V3 subset; not a sealed-C2 "
        "universe pass-rate claim.\n"
        f"- Fresh review binding: frozen V3 manifest internal SHA-256 {manifest_internal}.\n"
        f"- Former-reject → accept: {conversion['accepted']}/{conversion['former_rejects']} "
        f"({conversion['conversion_rate']:.3%}).\n"
        f"- Reclassified single pass rate: {accepted_single}/{total_single} "
        f"({rate(accepted_single, total_single):.3%}).\n"
        f"- Reclassified multi pass rate: {accepted_multi}/{total_multi} "
        f"({rate(accepted_multi, total_multi):.3%}).\n"
        f"- N′={multi['N_prime']} versus K=12: "
        f"{'REACHES' if multi['reaches_K'] else 'does not reach'} the gate.\n"
        f"- Python cross-validation discrepancies: {len(discrepancies)}.\n"
    )
    (OUT / "verdict.md").write_text(verdict, encoding="utf-8")
    print(json.dumps(lift["python_cross_validation"], ensure_ascii=False))


if __name__ == "__main__":
    main()
