#!/usr/bin/env python3
"""Re-propose and freeze the direct, non-normalized raw P5+ V4 pool."""

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

from improved_proposals import (  # noqa: E402
    read_jsonl,
    reclassify_proposals,
    sha256_file,
    validate_review_input,
)
from nature_download.corpus.proposals import PROPOSAL_RULE_V4  # noqa: E402


SOURCE = (
    ROOT / "files/c2_widen_pool/raw_materialized_strict_proposer/proposed.jsonl"
)
MATERIALIZATION_CENSUS = ROOT / "files/c2_widen_pool/materialization_census.json"
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


def is_direct_raw_single(record: Mapping[str, Any]) -> bool:
    """Reject any transformed/normalizer-backed component from sealed raw scope."""

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
        and source.get("sha256")
        and source.get("size_bytes") is not None
        and source.get("format") in {"csv", "xlsx"}
    )


def freeze(
    *,
    proposed: Path,
    report: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty freeze directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "raw_p5_v4.proposed.jsonl"
    shutil.copyfile(proposed, destination)
    destination.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    proposed_hash = sha256_file(proposed)
    frozen_hash = sha256_file(destination)
    if proposed_hash != frozen_hash:
        raise ValueError("frozen raw P5 copy hash mismatch")
    manifest = {
        "schema_version": "c2-raw-p5-v4-review-freeze-v1",
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
            **report["raw_p5_pool"],
        },
        "requires_fresh_review": True,
        "stale_review_or_evidence_reuse_forbidden": True,
        "exploratory_normalizer_inputs_forbidden": True,
        "k62_status": "blocked_only_7_independent_raw_p5_doi_clusters",
        "integrity_note": (
            "This freeze contains only directly read, hash-bound published CSV/XLSX "
            "source tables. Components with source_normalization, raw_source_table, "
            "or exploratory_normalizer paths are rejected before writing. It is a "
            "fresh-review-ready raw diagnostic pool, not sufficient for K=62."
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

    if not SOURCE.is_file():
        raise SystemExit(f"missing raw source proposal set: {SOURCE}")
    source_bytes = SOURCE.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if not MATERIALIZATION_CENSUS.is_file():
        raise SystemExit(f"missing materialization census: {MATERIALIZATION_CENSUS}")
    census = json.loads(MATERIALIZATION_CENSUS.read_text(encoding="utf-8"))
    expected_source_hash = (
        census.get("strict_proposer_runs", {})
        .get("raw_full_materialized_input", {})
        .get("proposed_jsonl_sha256")
    )
    if source_hash != expected_source_hash:
        raise SystemExit("raw source proposal hash differs from materialization census")

    rebuilt = reclassify_proposals(
        "raw-materialized-p5",
        SOURCE,
        preserve_existing_multi_scope=True,
        rule_version=PROPOSAL_RULE_V4,
    )
    p5_multis = [
        record
        for record in rebuilt["multis"]
        if len(record.get("source_candidate_ids") or []) >= 5
    ]
    component_ids = {
        str(component_id)
        for record in p5_multis
        for component_id in record["source_candidate_ids"]
    }
    component_counts = Counter(
        str(component_id)
        for record in p5_multis
        for component_id in record["source_candidate_ids"]
    )
    p5_singles = [
        record
        for record in rebuilt["singles"]
        if str(record["candidate_id"]) in component_ids
    ]
    if {str(record["candidate_id"]) for record in p5_singles} != component_ids:
        raise SystemExit("raw P5 component/single mismatch")
    if any(count != 1 for count in component_counts.values()):
        raise SystemExit("raw P5 pool has overlapping component panels")
    if not all(is_direct_raw_single(record) for record in p5_singles):
        raise SystemExit("normalized or non-raw component entered raw P5 pool")
    if any(record.get("eligible_for_experiment") is not False for record in p5_singles):
        raise SystemExit("raw P5 single became experiment eligible")
    if any(record.get("eligible_for_experiment") is not False for record in p5_multis):
        raise SystemExit("raw P5 multi became experiment eligible")

    records = sorted(
        p5_singles + p5_multis,
        key=lambda record: (
            str(record["proposal_type"]),
            str(record["candidate_id"]),
        ),
    )
    validate_review_input(records)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    proposed_path = OUTPUT / "reproposed_raw_p5_v4.jsonl"
    write_jsonl(proposed_path, records)
    raw_paths = Counter(
        str((record.get("source_table") or {}).get("path_root") or "")
        for record in p5_singles
    )
    report = {
        "schema_version": "c2-raw-p5-v4-reproposal-v1",
        "proposal_rule_version": PROPOSAL_RULE_V4,
        "source": {
            "path": str(SOURCE),
            "sha256": source_hash,
            "materialization_census_path": str(MATERIALIZATION_CENSUS),
            "materialization_census_sha256": sha256_file(MATERIALIZATION_CENSUS),
            "materialization_census_hash_verified": True,
            "baseline_rule_versions": sorted(
                {
                    str(record.get("proposal_rule_version") or "")
                    for record in rebuilt["original"]
                }
            ),
        },
        "raw_p5_pool": {
            "proposed_path": str(proposed_path),
            "proposed_sha256": sha256_file(proposed_path),
            "records": len(records),
            "single_panels": len(p5_singles),
            "multi_panel_parents": len(p5_multis),
            "independent_doi_clusters": len({record["doi"] for record in p5_multis}),
            "component_source_ids": len(component_ids),
            "overlapping_component_source_ids": sum(
                count > 1 for count in component_counts.values()
            ),
            "source_table_path_roots": dict(sorted(raw_paths.items())),
            "source_normalization_records": sum(
                record.get("source_normalization") not in (None, {})
                for record in p5_singles
            ),
            "exploratory_normalizer_path_records": sum(
                "exploratory_normalizer"
                in str((record.get("source_table") or {}).get("path") or "")
                for record in p5_singles
            ),
            "eligible_for_experiment_true": 0,
        },
        "review_status": "not_reviewed_fresh_v4_review_required",
        "k62_status": {
            "required_independent_verified_p5plus_doi_clusters": 62,
            "raw_p5plus_doi_clusters_available": len(
                {record["doi"] for record in p5_multis}
            ),
            "satisfied": False,
            "reason": (
                "This direct raw materialized source contains only seven P5+ DOI "
                "clusters; it cannot establish the sealed K=62 gate."
            ),
        },
        "integrity_note": (
            "V4 construction uses only original proposed records and their "
            "hash-bound source-table structure. No review/judge output is read. "
            "The raw P5 filter is structural/provenance-based and rejects all "
            "normalizer-backed components."
        ),
    }
    report["report_sha256"] = canonical_hash(report)
    report_path = OUTPUT / "raw_p5_v4_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = freeze(
        proposed=proposed_path,
        report=report,
        output=args.freeze_out.resolve(),
    )
    print(
        json.dumps(
            {
                "proposed": report["raw_p5_pool"],
                "report": str(report_path),
                "freeze_manifest": manifest["manifest_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
