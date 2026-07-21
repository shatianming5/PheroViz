#!/usr/bin/env python3
"""Re-propose C2 strict batches with the structure-only simple-2d-v4 rule.

This tool intentionally has two phases.  Proposal construction reads only the
proposed record and its hash-bound source table.  Old review records are read
only after construction, to select the priority transport set and to produce
diagnostic metrics; they never influence a proposed chart family or binding.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nature_download.corpus.proposals import (  # noqa: E402
    DEFAULT_MAX_COLUMNS,
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_ROWS,
    PROPOSAL_RULE_V4,
    WIDE_MELT_BINDING_MODE,
    _multi_panel_proposals,
    propose_single_candidate,
    read_candidate_table,
)
from nature_download.corpus.reviews import (  # noqa: E402
    GitState,
    REVIEW_RUBRIC_V4,
    REVIEW_RUBRIC_V4_HASH,
    _canonical_multi_map,
    _single_binding,
)


REVIEW_ROOT = Path(
    "/Users/tommy/.copilot/session-state/"
    "7b81726b-937c-41cb-9392-fead4d53250b/files/c2_review_scale"
)
OUTPUT = ROOT / "files" / "c2_reclassify"
RULE_VERSION = PROPOSAL_RULE_V4
RULE_TAG = "v4"

AXIS = (
    "time",
    "day",
    "hour",
    "min",
    "sec",
    "conc",
    "dose",
    "distance",
    "wavelength",
    "temp",
    "week",
    "month",
    "age",
    "cycle",
    "freq",
    "voltage",
    "current",
    "position",
    "depth",
    "ph ",
    "passage",
    "generation",
    "branchpoint",
    "length",
    "ratio",
    "mass",
    "weight",
    "diameter",
    "angle",
)
INDEX = (
    "replicate",
    "sample",
    "animal",
    "id",
    "index",
    "no.",
    "subject",
    "cell",
    "clone",
    "mouse",
    "rep",
)


Batch = tuple[str, Path, Path]
BATCHES: tuple[Batch, ...] = (
    (
        "casecount_strict",
        ROOT
        / "nature_download/outputs/c2_full_casecount_exploratory_20260720/"
        "corrected_sheet_binding/strict/proposals/proposed.jsonl",
        REVIEW_ROOT / "casecount_strict/reviews.jsonl",
    ),
    (
        "full_strict",
        ROOT / "nature_download/outputs/c2_full_proposals/proposed.jsonl",
        REVIEW_ROOT / "full_strict/reviews.jsonl",
    ),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected object")
        records.append(value)
    return records


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(record: Mapping[str, Any]) -> tuple[Any, ...]:
    case = record["experiment_case"]
    intent = case["intent"]
    return (
        case["chart_family"],
        intent["x"],
        tuple(intent["series"]),
        intent["x_scale"],
        intent.get("binding_mode", "direct"),
    )


def invariant_fields(original: Mapping[str, Any], rebuilt: Mapping[str, Any]) -> None:
    """Ensure construction did not mutate source/figure/provenance identities."""

    fields = (
        "candidate_id",
        "doi",
        "figure_no",
        "panel_ids",
        "source_table",
        "figure",
        "caption",
        "license",
        "license_evidence_sha256",
        "corpus_manifest_sha256",
        "provenance_manifest_sha256",
        "verification_evidence",
        "verification_evidence_file_sha256",
        "input_candidates_sha256",
        "code_commit",
    )
    changed = [
        field
        for field in fields
        if original.get(field) != rebuilt.get(field)
    ]
    if changed:
        raise AssertionError(
            f"{original.get('candidate_id')}: provenance changed: {changed}"
        )
    if rebuilt.get("eligible_for_experiment") is not False:
        raise AssertionError("re-proposal became experiment eligible")


def reclassify_proposals(
    name: str,
    proposal_path: Path,
    *,
    preserve_existing_multi_scope: bool = True,
    rule_version: str = RULE_VERSION,
) -> dict[str, Any]:
    """Construct source-structure-only proposals from old records and source tables.

    Set ``preserve_existing_multi_scope=False`` for a new frozen proposal pool to
    retain every newly canonical multi parent. The default preserves the supplied
    strict review-set cardinality by retaining only pre-existing multi parents.
    """

    original = read_jsonl(proposal_path)
    original_singles = [
        record for record in original if record.get("proposal_type") == "single_panel"
    ]
    original_multis = [
        record for record in original if record.get("proposal_type") == "multi_panel"
    ]
    if len(original_singles) + len(original_multis) != len(original):
        raise ValueError(f"{name}: unsupported proposal type")
    input_hashes = {str(record.get("input_candidates_sha256") or "") for record in original}
    commits = {str(record.get("code_commit") or "") for record in original}
    if len(input_hashes) != 1 or len(commits) != 1:
        raise ValueError(f"{name}: mixed proposal provenance")
    input_hash = next(iter(input_hashes))
    code_commit = next(iter(commits))

    rebuilt_singles = []
    for record in sorted(original_singles, key=lambda item: str(item["candidate_id"])):
        rebuilt = propose_single_candidate(
            record,
            input_candidates_sha256=input_hash,
            code_commit=code_commit,
            max_file_bytes=DEFAULT_MAX_FILE_BYTES,
            max_rows=DEFAULT_MAX_ROWS,
            max_columns=DEFAULT_MAX_COLUMNS,
            rule_version=rule_version,
        )
        invariant_fields(record, rebuilt)
        rebuilt_singles.append(rebuilt)

    generated_multis = _multi_panel_proposals(
        rebuilt_singles,
        input_candidates_sha256=input_hash,
        code_commit=code_commit,
    )
    original_multi_ids = {record["candidate_id"] for record in original_multis}
    generated_by_id = {
        record["candidate_id"]: record for record in generated_multis
    }
    if preserve_existing_multi_scope:
        missing_multi_ids = original_multi_ids - set(generated_by_id)
        if missing_multi_ids:
            raise AssertionError(
                f"{name}: {rule_version} lost multi parents: {sorted(missing_multi_ids)}"
            )
        # Keep the original strict-candidate scope. A newer rule can make an additional
        # parent canonical when an upstream raw candidate was renderability-blocked,
        # but adding it here would silently change the supplied review set.
        rebuilt_multis = [
            generated_by_id[record["candidate_id"]]
            for record in sorted(
                original_multis,
                key=lambda item: str(item["candidate_id"]),
            )
        ]
    else:
        rebuilt_multis = generated_multis
    if any(record.get("eligible_for_experiment") is not False for record in rebuilt_multis):
        raise AssertionError(f"{name}: multi proposal became experiment eligible")
    records = sorted(
        rebuilt_singles + rebuilt_multis,
        key=lambda item: (str(item["proposal_type"]), str(item["candidate_id"])),
    )
    if preserve_existing_multi_scope and len(records) != len(original):
        raise AssertionError(f"{name}: proposal count changed")
    return {
        "name": name,
        "original": original,
        "singles": rebuilt_singles,
        "multis": rebuilt_multis,
        "records": records,
        "newly_canonical_multi_parents_omitted": (
            sorted(set(generated_by_id) - original_multi_ids)
            if preserve_existing_multi_scope
            else []
        ),
    }


def reclassify_batch(name: str, proposal_path: Path) -> dict[str, Any]:
    """Backward-compatible strict-set wrapper for the published reclassifier."""

    return reclassify_proposals(
        name,
        proposal_path,
        preserve_existing_multi_scope=True,
    )


def rejected_single_ids(reviews_path: Path) -> set[str]:
    return {
        str(review["candidate_id"])
        for review in read_jsonl(reviews_path)
        if review.get("proposal_type") == "single_panel"
        and review.get("status") == "rejected"
    }


def priority_records(batch: Mapping[str, Any], reject_ids: set[str]) -> list[dict[str, Any]]:
    singles = [
        record for record in batch["singles"] if record["candidate_id"] in reject_ids
    ]
    if len(singles) != len(reject_ids):
        raise AssertionError(f"{batch['name']}: missing rejected singles")
    # A multi parent is reviewable only when every constituent appears in this
    # standalone priority input. Parents with already-accepted constituents remain
    # in the complete batch input, where they can be canonically re-reviewed.
    multis = [
        record
        for record in batch["multis"]
        if set(record["source_candidate_ids"]).issubset(reject_ids)
    ]
    return sorted(
        singles + multis,
        key=lambda item: (str(item["proposal_type"]), str(item["candidate_id"])),
    )


def validate_review_input(records: list[dict[str, Any]]) -> None:
    """Run the local, no-model review preflight and canonical multi check."""

    ids = [str(record.get("candidate_id") or "") for record in records]
    if not all(ids) or len(ids) != len(set(ids)):
        raise AssertionError("review input has missing or duplicate candidate IDs")
    state = GitState(commit="abcdef1", dirty=False)
    singles = [record for record in records if record["proposal_type"] == "single_panel"]
    for record in singles:
        _, reasons, _, _, expected = _single_binding(
            record,
            input_hash="offline-reclassify-preflight",
            models=("offline-a", "offline-b"),
            git_state=state,
            rubric=REVIEW_RUBRIC_V4,
            rubric_hash=REVIEW_RUBRIC_V4_HASH,
        )
        if reasons or expected is None:
            raise AssertionError(
                f"{record['candidate_id']}: review preflight failed: {reasons}"
            )
    canonical = _canonical_multi_map(records)
    for record in records:
        if record["proposal_type"] != "multi_panel":
            continue
        expected = canonical.get(record["candidate_id"])
        fields = (
            "schema_version",
            "candidate_id",
            "proposal_type",
            "source_candidate_ids",
            "doi",
            "figure_no",
            "panel_ids",
            "curation_status",
            "eligible_for_experiment",
            "eligibility_reasons",
            "experiment_case",
        )
        if expected is None or any(
            record.get(field) != expected.get(field) for field in fields
        ):
            raise AssertionError(
                f"{record['candidate_id']}: non-canonical priority multi parent"
            )


def proto_colkind(name: str, series: pd.Series) -> str:
    normalized = str(name).lower().strip()
    values = series.dropna()
    integer_values = values.map(
        lambda value: isinstance(value, (int, np.integer))
        or (isinstance(value, float) and float(value).is_integer())
    ).all()
    if any(token in normalized for token in INDEX) and integer_values:
        return "index"
    if series.dtype == object or str(series.dtype).startswith("str"):
        return "label"
    return "measure"


def prototype_classify(frame: pd.DataFrame) -> tuple[str, str]:
    """Literal structure-only classifier from c2_melt_classifier_prototype.py."""

    columns = list(frame.columns)
    kinds = {column: proto_colkind(column, frame[column]) for column in columns}
    labels = [column for column in columns if kinds[column] == "label"]
    measures = [column for column in columns if kinds[column] == "measure"]
    indexes = [column for column in columns if kinds[column] == "index"]
    axis_measures = [
        column
        for column in measures
        if any(token in str(column).lower() for token in AXIS)
    ]
    if labels:
        return "bar", f"x={labels[0]}(categorical)"
    nonaxis_measures = [
        column
        for column in measures
        if not any(token in str(column).lower() for token in AXIS)
    ]
    if len(measures) >= 2 and not axis_measures and not indexes:
        if len(measures) >= 3:
            return "scatter", f"melt-wide-{len(measures)}groups"
        return "scatter", "two-measure-correlation"
    if axis_measures and len(measures) >= 2:
        x_column = axis_measures[0]
        x_values = frame[x_column].dropna()
        if x_values.duplicated().mean() > 0.3:
            return "scatter", "axis-x-with-duplicates"
        return "line", f"axis-x={x_column}"
    if len(measures) == 1 and indexes:
        return "bar", "index-x-single-measure"
    return "unknown", f"cols={[kinds[column] for column in columns]}"


def consensus_family(review: Mapping[str, Any]) -> str | None:
    families = [
        str((model.get("output") or {}).get("chart_family")).lower()
        for model in review.get("model_reviews") or []
        if isinstance(model.get("output"), dict)
        and (model.get("output") or {}).get("chart_family")
    ]
    return families[0] if len(families) == 2 and families[0] == families[1] else None


def binding_class(family: str) -> str:
    return "grouped" if family in {"bar", "scatter"} else family


def rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "match": numerator,
        "total": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def diagnostic(
    batch: Mapping[str, Any],
    reviews_path: Path,
) -> dict[str, Any]:
    """Score finished source-only proposals; reviews are diagnostics only."""

    review_by_id = {
        str(review["candidate_id"]): review for review in read_jsonl(reviews_path)
    }
    proposals = {record["candidate_id"]: record for record in batch["singles"]}
    rejected = [
        review
        for review in review_by_id.values()
        if review.get("proposal_type") == "single_panel"
        and review.get("status") == "rejected"
    ]
    accepted = [
        review
        for review in review_by_id.values()
        if review.get("proposal_type") == "single_panel"
        and review.get("status") == "accepted"
    ]
    rejected_ids = {str(review["candidate_id"]) for review in rejected}
    v4_consensus_exact = v4_consensus_binding = 0
    prototype_consensus_exact = prototype_consensus_binding = 0
    v4_prototype_exact = v4_prototype_binding = 0
    evaluated = prototype_nonunknown = unreadable = 0
    prototype_unknown = 0
    changes = collections.Counter()

    original_by_id = {
        record["candidate_id"]: record
        for record in batch["original"]
        if record.get("proposal_type") == "single_panel"
    }
    for proposal in batch["singles"]:
        if proposal["candidate_id"] not in rejected_ids:
            continue
        old = original_by_id[proposal["candidate_id"]]
        new_analysis = proposal["proposal_analysis"]
        old_binding = binding(old)
        new_binding = binding(proposal)
        if new_analysis["binding_mode"] == WIDE_MELT_BINDING_MODE:
            changes["wide_melt"] += 1
        if (
            old_binding[0] != "scatter"
            and new_binding[0] == "scatter"
            and new_binding[3] == "categorical"
        ):
            changes["grouped_dot_plot_scatter"] += 1
        if (
            old_binding != new_binding
            and new_binding[0] == "line"
            and new_binding[3] == "temporal"
        ):
            changes["temporal_priority_line"] += 1
        if (
            old_binding[0] != "scatter"
            and new_binding[0] == "scatter"
            and new_analysis["binding_mode"] == "direct"
        ):
            changes["correlation_scatter"] += 1
            if new_binding[3] == "linear":
                changes["repeated_x_grid_scatter"] += 1
        if new_analysis["dropped_index_columns"] and (
            tuple(old_binding[2]) != tuple(new_binding[2])
        ):
            changes["index_identifier_y_exclusion"] += 1
        if old_binding != new_binding:
            changes["any_binding_changed"] += 1

    for review in rejected:
        candidate_id = str(review["candidate_id"])
        proposal = proposals[candidate_id]
        consensus = consensus_family(review)
        if consensus is None:
            continue
        try:
            frame = read_candidate_table(proposal)
        except Exception:
            unreadable += 1
            continue
        predicted = proposal["experiment_case"]["chart_family"]
        prototype, _ = prototype_classify(frame)
        evaluated += 1
        if predicted == consensus:
            v4_consensus_exact += 1
        if binding_class(predicted) == binding_class(consensus):
            v4_consensus_binding += 1
        if prototype != "unknown":
            prototype_nonunknown += 1
            if prototype == consensus:
                prototype_consensus_exact += 1
            if binding_class(prototype) == binding_class(consensus):
                prototype_consensus_binding += 1
            if predicted == prototype:
                v4_prototype_exact += 1
            if binding_class(predicted) == binding_class(prototype):
                v4_prototype_binding += 1
        else:
            prototype_unknown += 1

    regression = []
    for review in accepted:
        candidate_id = str(review["candidate_id"])
        old = original_by_id[candidate_id]
        new = proposals[candidate_id]
        if binding(old) != binding(new):
            regression.append(
                {
                    "candidate_id": candidate_id,
                    "old": {
                        "chart_family": old["experiment_case"]["chart_family"],
                        "intent": old["experiment_case"]["intent"],
                    },
                    "new": {
                        "chart_family": new["experiment_case"]["chart_family"],
                        "intent": new["experiment_case"]["intent"],
                    },
                }
            )
    return {
        "single_rejects": len(rejected),
        "single_accepted": len(accepted),
        "reclassified_per_fix_type": dict(sorted(changes.items())),
        "consensus_evaluable_rejects": evaluated,
        "unreadable_rejects": unreadable,
        "v4_vs_two_judge_consensus": {
            "exact_family": rate(v4_consensus_exact, evaluated),
            "binding_class": rate(v4_consensus_binding, evaluated),
        },
        "prototype_vs_two_judge_consensus": {
            "exact_family": rate(prototype_consensus_exact, evaluated),
            "binding_class": rate(prototype_consensus_binding, evaluated),
            "unknown": prototype_unknown,
        },
        "v4_vs_prototype": {
            "exact_family": rate(v4_prototype_exact, evaluated),
            "binding_class": rate(v4_prototype_binding, evaluated),
            "prototype_nonunknown": prototype_nonunknown,
        },
        "accepted_regression": {
            "checked": len(accepted),
            "changed": len(regression),
            "unchanged": len(accepted) - len(regression),
            "records": regression,
        },
    }


def aggregate_diagnostics(diagnostics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    def aggregate_rate(path: tuple[str, ...]) -> dict[str, Any]:
        leaves = [_walk(entry, path) for entry in diagnostics.values()]
        match = sum(int(leaf["match"]) for leaf in leaves)
        total = sum(int(leaf["total"]) for leaf in leaves)
        return rate(match, total)

    def _fix_count(key: str) -> int:
        return sum(
            int(entry["reclassified_per_fix_type"].get(key, 0))
            for entry in diagnostics.values()
        )

    return {
        "single_rejects": sum(
            int(entry["single_rejects"]) for entry in diagnostics.values()
        ),
        "consensus_evaluable_rejects": sum(
            int(entry["consensus_evaluable_rejects"])
            for entry in diagnostics.values()
        ),
        "v4_vs_two_judge_consensus": {
            "exact_family": aggregate_rate(
                ("v4_vs_two_judge_consensus", "exact_family")
            ),
            "binding_class": aggregate_rate(
                ("v4_vs_two_judge_consensus", "binding_class")
            ),
        },
        "prototype_vs_two_judge_consensus": {
            "exact_family": aggregate_rate(
                ("prototype_vs_two_judge_consensus", "exact_family")
            ),
            "binding_class": aggregate_rate(
                ("prototype_vs_two_judge_consensus", "binding_class")
            ),
            "unknown": sum(
                int(entry["prototype_vs_two_judge_consensus"]["unknown"])
                for entry in diagnostics.values()
            ),
        },
        "v4_vs_prototype": {
            "exact_family": aggregate_rate(
                ("v4_vs_prototype", "exact_family")
            ),
            "binding_class": aggregate_rate(
                ("v4_vs_prototype", "binding_class")
            ),
            "prototype_nonunknown": sum(
                int(entry["v4_vs_prototype"]["prototype_nonunknown"])
                for entry in diagnostics.values()
            ),
        },
        "reclassified_per_fix_type": {
            key: _fix_count(key)
            for key in (
                "wide_melt",
                "grouped_dot_plot_scatter",
                "temporal_priority_line",
                "correlation_scatter",
                "repeated_x_grid_scatter",
                "index_identifier_y_exclusion",
                "any_binding_changed",
            )
        },
        "accepted_regression": {
            "checked": sum(
                int(entry["accepted_regression"]["checked"])
                for entry in diagnostics.values()
            ),
            "changed": sum(
                int(entry["accepted_regression"]["changed"])
                for entry in diagnostics.values()
            ),
        },
    }


def _walk(value: Mapping[str, Any], path: tuple[str, ...]) -> Mapping[str, Any]:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            raise ValueError(f"invalid diagnostic path: {path}")
        current = current[key]
    if not isinstance(current, Mapping):
        raise ValueError(f"diagnostic leaf is not an object: {path}")
    return current


def tagged(records: Iterable[Mapping[str, Any]], batch: str) -> list[dict[str, Any]]:
    return [{**dict(record), "reproposal_batch": batch} for record in records]


def write_priority(rebuilt: list[dict[str, Any]]) -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for batch, (_, _, reviews_path) in zip(rebuilt, BATCHES, strict=True):
        reject_ids = rejected_single_ids(reviews_path)
        records = priority_records(batch, reject_ids)
        validate_review_input(records)
        native_path = OUTPUT / (
            f"reproposed_strict_rejects_{RULE_TAG}_{batch['name']}.jsonl"
        )
        write_jsonl(native_path, records)
        all_records.extend(tagged(records, batch["name"]))
        details[batch["name"]] = {
            "rejected_singles": len(reject_ids),
            "reviewable_multi_parents": sum(
                record["proposal_type"] == "multi_panel" for record in records
            ),
            "native_review_input": str(native_path),
            "sha256": sha256_file(native_path),
        }
    combined_path = OUTPUT / f"reproposed_strict_rejects_{RULE_TAG}.jsonl"
    write_jsonl(combined_path, all_records)
    return {
        "combined_path": str(combined_path),
        "combined_records": len(all_records),
        "combined_sha256": sha256_file(combined_path),
        "batches": details,
    }


def write_full(rebuilt: list[dict[str, Any]]) -> dict[str, Any]:
    all_records: list[dict[str, Any]] = []
    all_multis: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for batch in rebuilt:
        records = batch["records"]
        validate_review_input(records)
        native_path = OUTPUT / (
            f"reproposed_strict_full_{RULE_TAG}_{batch['name']}.jsonl"
        )
        multi_path = OUTPUT / (
            f"reproposed_strict_multi_parents_{RULE_TAG}_{batch['name']}.jsonl"
        )
        write_jsonl(native_path, records)
        write_jsonl(multi_path, batch["multis"])
        all_records.extend(tagged(records, batch["name"]))
        all_multis.extend(tagged(batch["multis"], batch["name"]))
        details[batch["name"]] = {
            "records": len(records),
            "singles": len(batch["singles"]),
            "multi_parents": len(batch["multis"]),
            "newly_canonical_multi_parents_omitted_to_preserve_input_scope": batch[
                "newly_canonical_multi_parents_omitted"
            ],
            "native_review_input": str(native_path),
            "sha256": sha256_file(native_path),
        }
    combined_path = OUTPUT / f"reproposed_strict_full_{RULE_TAG}.jsonl"
    combined_multis_path = OUTPUT / (
        f"reproposed_strict_multi_parents_{RULE_TAG}.jsonl"
    )
    write_jsonl(combined_path, all_records)
    write_jsonl(combined_multis_path, all_multis)
    return {
        "combined_path": str(combined_path),
        "combined_records": len(all_records),
        "combined_sha256": sha256_file(combined_path),
        "combined_multi_parents_path": str(combined_multis_path),
        "combined_multi_parents": len(all_multis),
        "batches": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--priority-only",
        action="store_true",
        help="write only the priority rejected-single review inputs",
    )
    args = parser.parse_args()
    rebuilt = [
        reclassify_batch(name, proposal_path)
        for name, proposal_path, _ in BATCHES
    ]
    priority = write_priority(rebuilt)
    print(
        json.dumps(
            {"status": "priority-written", **priority},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if args.priority_only:
        return 0

    full = write_full(rebuilt)
    diagnostics = {
        batch["name"]: diagnostic(batch, reviews_path)
        for batch, (_, _, reviews_path) in zip(rebuilt, BATCHES, strict=True)
    }
    report = {
        "schema_version": "c2-reclassify-v1",
        "proposal_rule_version": RULE_VERSION,
        "classifier_inputs": [
            "hash-bound source-table headers",
            "hash-bound source-table dtypes",
            "hash-bound source-table values",
        ],
        "integrity_note": (
            "Proposal construction is source-table-structure-only. Historical "
            "judge records are read only after construction for priority selection, "
            "regression measurement, and diagnostic scoring; no judge output is "
            "used to set a proposal family, x, y, or source transform."
        ),
        "wide_melt_contract": {
            "binding_mode": WIDE_MELT_BINDING_MODE,
            "group_column": "__wide_group__",
            "value_column": "__wide_value__",
            "source_mutation": "none; an explicit deterministic in-memory transform",
            "review_rubric": "proposal-external-validation-v4",
        },
        "priority": priority,
        "full": full,
        "diagnostics": diagnostics,
        "diagnostics_aggregate": aggregate_diagnostics(diagnostics),
        "diagnostics_aggregate_scope": (
            "Supplied review-batch instances (127 casecount_strict + 76 "
            "full_strict rejects); overlapping candidate IDs are intentionally "
            "counted once per source-root batch."
        ),
        "combined_transport_note": (
            "The two combined JSONL files retain all batch instances and add "
            "reproposal_batch. They contain duplicate candidate_id values because "
            "full_strict is a source-root variant/subset of casecount_strict, so "
            "they must be split by reproposal_batch before native review. The "
            "per-batch files named native_review_input are independently valid "
            "review_proposals inputs and have been offline preflighted."
        ),
    }
    report_path = OUTPUT / f"reclassify_report_{RULE_TAG}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # Re-read each written aggregate so reported counts are not inferred.
    if len(
        read_jsonl(OUTPUT / f"reproposed_strict_rejects_{RULE_TAG}.jsonl")
    ) != priority[
        "combined_records"
    ]:
        raise AssertionError("priority output count mismatch")
    if len(read_jsonl(OUTPUT / f"reproposed_strict_full_{RULE_TAG}.jsonl")) != full[
        "combined_records"
    ]:
        raise AssertionError("full output count mismatch")
    print(
        json.dumps(
            {
                "status": "full-written",
                "report": str(report_path),
                "priority_records": priority["combined_records"],
                "full_records": full["combined_records"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
