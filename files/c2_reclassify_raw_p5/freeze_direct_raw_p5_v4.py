#!/usr/bin/env python3
"""Freeze the P5+ subset from an all-direct V4 raw candidate proposal run."""

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
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
RECLASSIFIER_DIR = ROOT / "files" / "c2_reclassify"
if str(RECLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(RECLASSIFIER_DIR))

from improved_proposals import read_jsonl, sha256_file, validate_review_input  # noqa: E402
from nature_download.corpus.proposals import PROPOSAL_RULE_V4  # noqa: E402


RAW_CANDIDATES = (
    ROOT / "nature_download/outputs/c2_full_casecount_20260720/combined/all_candidates.jsonl"
)
FULL_PROPOSED = (
    ROOT / "files/c2_reclassify_raw_p5/direct_all_candidates_v4/proposed.jsonl"
)
OUTPUT = ROOT / "files/c2_reclassify_raw_p5"


def canonical_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    dict(record),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def is_direct_raw(record: Mapping[str, Any]) -> bool:
    if record.get("source_normalization") not in (None, {}):
        return False
    if record.get("raw_source_table") not in (None, {}):
        return False
    source = record.get("source_table")
    if not isinstance(source, Mapping):
        return False
    path = str(source.get("path") or "")
    return bool(
        path
        and "exploratory_normalizer" not in path
        and source.get("format") in {"csv", "xlsx"}
        and source.get("sha256")
        and source.get("size_bytes") is not None
    )


def freeze(proposed: Path, report: dict[str, Any], output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty freeze directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "direct_raw_p5_v4.proposed.jsonl"
    shutil.copyfile(proposed, destination)
    destination.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    proposed_hash = sha256_file(proposed)
    frozen_hash = sha256_file(destination)
    if proposed_hash != frozen_hash:
        raise ValueError("direct raw P5 frozen copy hash mismatch")
    manifest = {
        "schema_version": "c2-direct-raw-p5-v4-review-freeze-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_from_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "proposal_rule_version": PROPOSAL_RULE_V4,
        "review_rubric_required": "proposal-external-validation-v4",
        "required_review_models": ["claude-sonnet-4.6", "gemini-3.5-flash"],
        "input": {
            "source_path": str(proposed),
            "source_sha256": proposed_hash,
            "frozen_path": str(destination),
            "frozen_sha256": frozen_hash,
            **report["direct_raw_p5_pool"],
        },
        "requires_fresh_review": True,
        "stale_review_or_evidence_reuse_forbidden": True,
        "exploratory_normalizer_inputs_forbidden": True,
        "k62_status": report["k62_status"]["status"],
        "integrity_note": (
            "This is a P5+ subset of the all-direct raw candidate inventory. Every "
            "component was re-proposed by V4 from a hash-bound published CSV/XLSX "
            "table and is rejected if normalization metadata, raw_source_table, or "
            "an exploratory_normalizer path appears. It is not K=62 sufficient."
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
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-out", type=Path, required=True)
    args = parser.parse_args()
    if not RAW_CANDIDATES.is_file() or not FULL_PROPOSED.is_file():
        raise SystemExit("missing raw candidates or all-direct V4 proposal artifact")

    raw_candidate_hash = sha256_file(RAW_CANDIDATES)
    full_records = read_jsonl(FULL_PROPOSED)
    singles = [
        record
        for record in full_records
        if record.get("proposal_type") == "single_panel"
    ]
    multis = [
        record
        for record in full_records
        if record.get("proposal_type") == "multi_panel"
    ]
    if len(singles) + len(multis) != len(full_records):
        raise SystemExit("full proposal artifact has unsupported proposal types")
    if not all(
        record.get("proposal_rule_version") == PROPOSAL_RULE_V4
        and record.get("input_candidates_sha256") == raw_candidate_hash
        for record in full_records
    ):
        raise SystemExit("full proposal artifact is not bound to the raw V4 input")
    if not all(is_direct_raw(record) for record in singles):
        raise SystemExit("normalizer-backed component found in all-direct V4 artifact")

    p5_multis = [
        record
        for record in multis
        if len(record.get("source_candidate_ids") or []) >= 5
    ]
    component_counts = Counter(
        str(component_id)
        for record in p5_multis
        for component_id in record["source_candidate_ids"]
    )
    component_ids = set(component_counts)
    p5_singles = [
        record for record in singles if str(record["candidate_id"]) in component_ids
    ]
    if {str(record["candidate_id"]) for record in p5_singles} != component_ids:
        raise SystemExit("P5 component/single mismatch")
    if any(count != 1 for count in component_counts.values()):
        raise SystemExit("P5 components overlap across multi parents")
    if any(record.get("eligible_for_experiment") is not False for record in p5_singles):
        raise SystemExit("P5 single became experiment eligible")
    if any(record.get("eligible_for_experiment") is not False for record in p5_multis):
        raise SystemExit("P5 multi became experiment eligible")

    records = sorted(
        p5_singles + p5_multis,
        key=lambda record: (
            str(record["proposal_type"]),
            str(record["candidate_id"]),
        ),
    )
    validate_review_input(records)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    proposed_path = OUTPUT / "reproposed_direct_raw_p5_v4.jsonl"
    write_jsonl(proposed_path, records)
    doi_count = len({str(record["doi"]) for record in p5_multis})
    path_roots = Counter(
        str((record.get("source_table") or {}).get("path_root") or "")
        for record in p5_singles
    )
    report = {
        "schema_version": "c2-direct-raw-p5-v4-reproposal-v1",
        "proposal_rule_version": PROPOSAL_RULE_V4,
        "source": {
            "raw_candidates_path": str(RAW_CANDIDATES),
            "raw_candidates_sha256": raw_candidate_hash,
            "all_direct_v4_proposed_path": str(FULL_PROPOSED),
            "all_direct_v4_proposed_sha256": sha256_file(FULL_PROPOSED),
            "all_direct_v4_records": len(full_records),
            "all_direct_v4_singles": len(singles),
            "all_direct_v4_multis": len(multis),
        },
        "direct_raw_p5_pool": {
            "proposed_path": str(proposed_path),
            "proposed_sha256": sha256_file(proposed_path),
            "records": len(records),
            "single_panels": len(p5_singles),
            "multi_panel_parents": len(p5_multis),
            "independent_doi_clusters": doi_count,
            "component_source_ids": len(component_ids),
            "overlapping_component_source_ids": sum(
                count > 1 for count in component_counts.values()
            ),
            "source_normalization_records": sum(
                record.get("source_normalization") not in (None, {})
                for record in p5_singles
            ),
            "raw_source_table_records": sum(
                record.get("raw_source_table") not in (None, {})
                for record in p5_singles
            ),
            "exploratory_normalizer_path_records": sum(
                "exploratory_normalizer"
                in str((record.get("source_table") or {}).get("path") or "")
                for record in p5_singles
            ),
            "source_table_path_roots": dict(sorted(path_roots.items())),
            "eligible_for_experiment_true": 0,
        },
        "review_status": "not_reviewed_fresh_v4_review_required",
        "k62_status": {
            "status": "blocked_insufficient_direct_raw_p5_doi_clusters",
            "required_independent_verified_p5plus_doi_clusters": 62,
            "direct_raw_p5plus_doi_clusters_available": doi_count,
            "satisfied": False,
        },
        "integrity_note": (
            "V4 proposal construction used only the direct raw candidate records "
            "and each hash-bound source table. No judge/review output was read. "
            "This direct raw P5 pool is eligible for a fresh diagnostic review but "
            "cannot support a K=62 sealed benchmark claim."
        ),
    }
    report["report_sha256"] = canonical_hash(report)
    report_path = OUTPUT / "direct_raw_p5_v4_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = freeze(proposed_path, report, args.freeze_out.resolve())
    print(
        json.dumps(
            {
                "direct_raw_p5_pool": report["direct_raw_p5_pool"],
                "report_path": str(report_path),
                "freeze_manifest_sha256": manifest["manifest_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
