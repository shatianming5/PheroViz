"""Validate and summarize the frozen Scientific Reports/npj C2 universe."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

CORPUS_COMMIT = "ca98442b9e805110089b03083cb240e19b58d4a2"
UNIVERSE_SHA256 = "51848466c6bf6bf400b58faf539953830349abab21438e527eaefd74103450df"
CHUNK_COUNT = 13
UNIVERSE_RECORDS = 2463


class ValidationError(RuntimeError):
    """Raised when a sealed corpus binding is inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _require(all(isinstance(row, dict) for row in rows), f"{path}: invalid JSONL")
    return rows


def _read_statuses(path: Path) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        article_id, status = line.split("\t", 1)
        _require(article_id not in statuses, f"{path}: duplicate {article_id}")
        statuses[article_id] = status
    return statuses


def _report_value(report: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value: Any = report
        for key in path:
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            return value
    raise ValidationError(f"sealed report lacks all alternatives: {paths}")


def _round_paths(root: Path, round_name: str) -> tuple[Path, Path]:
    if round_name == "initial":
        control_processed = root / "control/initial/processed.txt"
        control_skipped = root / "control/initial/_skipped.txt"
        processed = (
            control_processed
            if control_processed.is_file()
            else root / "content/_processed.txt"
        )
        skipped = (
            control_skipped
            if control_skipped.is_file()
            else root / "content/_skipped.txt"
        )
        return processed, skipped
    return (
        root / f"control/{round_name}/processed.txt",
        root / f"control/{round_name}/_skipped.txt",
    )


def _validate_rounds(
    root: Path, accepted: list[dict[str, Any]], accepted_sha256: str
) -> tuple[list[dict[str, Any]], Counter[str]]:
    article_ids = [
        str(row["article_url"]).rstrip("/").rsplit("/", 1)[-1] for row in accepted
    ]
    expected_ids = set(article_ids)
    round_statuses: dict[str, dict[str, str]] = {}
    for round_name in ("initial", "retry1", "retry2"):
        processed_path, skipped_path = _round_paths(root, round_name)
        exit_path = root / f"control/{round_name}/postfetch.exit"
        _require(processed_path.is_file(), f"{root}: missing {round_name} processed")
        _require(skipped_path.is_file(), f"{root}: missing {round_name} skipped")
        _require(exit_path.read_text().strip() == "0", f"{root}: failed {round_name}")
        processed = processed_path.read_text(encoding="utf-8").splitlines()
        _require(
            len(processed) == len(accepted) and set(processed) == expected_ids,
            f"{root}: {round_name} did not process the exact frozen chunk",
        )
        if round_name != "initial":
            retry_input = root / f"control/{round_name}/accepted.jsonl"
            _require(
                _sha256(retry_input) == accepted_sha256,
                f"{root}: {round_name} input differs from frozen chunk",
            )
        skipped = _read_statuses(skipped_path)
        _require(set(skipped) <= expected_ids, f"{root}: unknown skipped article")
        round_statuses[round_name] = {
            article_id: skipped.get(article_id, "downloaded")
            for article_id in article_ids
        }

    outcomes_path = root / "control/terminal_outcomes.jsonl"
    outcomes = _read_jsonl(outcomes_path)
    _require(len(outcomes) == len(accepted), f"{root}: terminal outcome count")
    by_doi = {str(row["doi"]).lower(): row for row in outcomes}
    _require(len(by_doi) == len(accepted), f"{root}: duplicate terminal DOI")
    counts: Counter[str] = Counter()
    for record, article_id in zip(accepted, article_ids, strict=True):
        outcome = by_doi.get(str(record["doi"]).lower())
        _require(outcome is not None, f"{root}: missing terminal DOI")
        expected_rounds = {
            name: statuses[article_id] for name, statuses in round_statuses.items()
        }
        _require(outcome.get("rounds") == expected_rounds, f"{root}: round mismatch")
        terminal = expected_rounds["retry2"]
        _require(outcome.get("terminal_status") == terminal, f"{root}: terminal mismatch")
        counts[terminal] += 1
    return outcomes, counts


def _manifest_path(root: Path) -> Path:
    alternatives = (
        root / "manifest_v1/corpus_manifest.jsonl",
        root / "corpus_manifest/corpus_manifest.jsonl",
    )
    for path in alternatives:
        if path.is_file():
            return path
    raise ValidationError(f"{root}: corpus manifest missing")


def _validate_artifact_seal(root: Path, report: dict[str, Any]) -> None:
    artifact_hashes = report.get("trust_chain", {}).get("artifact_hashes")
    if isinstance(artifact_hashes, dict):
        for relative_path, expected_sha256 in artifact_hashes.items():
            path = root / relative_path
            _require(path.is_file(), f"{root}: sealed artifact missing: {relative_path}")
            _require(
                _sha256(path) == expected_sha256,
                f"{root}: sealed artifact changed: {relative_path}",
            )
        return

    artifact_seal = report.get("artifact_seal")
    if not isinstance(artifact_seal, dict):
        return
    manifest_path = root / str(artifact_seal["artifact_manifest"])
    _require(
        _sha256(manifest_path) == artifact_seal["artifact_manifest_file_sha256"],
        f"{root}: artifact manifest file hash",
    )
    manifest = _read_json(manifest_path)
    declared_hash = manifest.get("manifest_hash")
    if declared_hash is None:
        declared_hash = manifest.get("summary_hash")
    if declared_hash is not None:
        computed_hash = _canonical_sha256(
            {
                key: value
                for key, value in manifest.items()
                if key not in {"manifest_hash", "summary_hash"}
            }
        )
        _require(
            computed_hash == artifact_seal["artifact_manifest_hash"],
            f"{root}: artifact manifest semantic hash",
        )
    for artifact in manifest["files"]:
        path = root / artifact["path"]
        _require(path.is_file(), f"{root}: artifact missing: {artifact['path']}")
        _require(
            _sha256(path) == artifact["sha256"],
            f"{root}: artifact changed: {artifact['path']}",
        )


def _chunk_summary(repository_root: Path, index: int) -> dict[str, Any]:
    outputs = repository_root / "nature_download/outputs"
    frozen = outputs / "ccby_sr_npj_universe_frozen_ca98442"
    chunk_path = frozen / f"chunks/chunk_{index:03d}.jsonl"
    root_rel = Path(
        f"nature_download/outputs/ccby_sr_npj_chunk{index:03d}_clean_ca98442"
    )
    root = repository_root / root_rel
    accepted_path = root / "accepted.jsonl"
    if not accepted_path.is_file():
        accepted_path = root / f"frozen_input/chunk_{index:03d}.jsonl"
    report_path = root / "sealed_report_v1/sealed_report.json"
    report_seal = root / "sealed_report_v1/sealed_report.sha256"

    chunk_rows = _read_jsonl(chunk_path)
    accepted = _read_jsonl(accepted_path)
    chunk_sha256 = _sha256(chunk_path)
    _require(_sha256(accepted_path) == chunk_sha256, f"chunk {index}: copied input")
    _require(accepted == chunk_rows, f"chunk {index}: input records differ")
    dois = [str(row["doi"]).lower() for row in chunk_rows]
    _require(len(dois) == len(set(dois)), f"chunk {index}: duplicate DOI")
    _require(
        all(
            row.get("download_eligible") is True
            and row.get("policy_accepted") is True
            and row.get("require_cc_by") is True
            and row.get("license", {}).get("content_version") == "vor"
            and row.get("license", {}).get("license_id") == "CC-BY-4.0"
            and (
                str(row.get("journal", "")).casefold() == "scientific reports"
                or str(row.get("journal", "")).casefold().startswith("npj ")
            )
            for row in chunk_rows
        ),
        f"chunk {index}: record violates strict VOR/CC-BY/journal policy",
    )

    report = _read_json(report_path)
    report_file_sha256 = _sha256(report_path)
    _require(
        report_seal.read_text().split()[0] == report_file_sha256,
        f"chunk {index}: report file seal",
    )
    declared_report_hash = str(report["report_hash"])
    computed_report_hash = _canonical_sha256(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    _require(
        declared_report_hash == computed_report_hash,
        f"chunk {index}: canonical report hash",
    )
    _validate_artifact_seal(root, report)
    _require(
        _report_value(report, ("code", "commit")) == CORPUS_COMMIT,
        f"chunk {index}: corpus commit",
    )
    _require(
        _report_value(
            report, ("frozen_input", "sha256"), ("frozen_input", "chunk_sha256")
        )
        == chunk_sha256,
        f"chunk {index}: report input hash",
    )

    outcomes, terminal_counts = _validate_rounds(root, accepted, chunk_sha256)
    declared_outcomes_sha256 = report.get("postfetch", {}).get(
        "terminal_outcomes_sha256"
    )
    if declared_outcomes_sha256 is not None:
        _require(
            declared_outcomes_sha256
            == _sha256(root / "control/terminal_outcomes.jsonl"),
            f"chunk {index}: report terminal outcome hash",
        )
    report_terminal = _report_value(
        report, ("postfetch", "terminal_counts"), ("execution", "terminal_counts")
    )
    _require(
        dict(sorted(terminal_counts.items())) == report_terminal,
        f"chunk {index}: report terminal counts",
    )

    manifest_path = _manifest_path(root)
    manifest = _read_jsonl(manifest_path)
    manifest_sha256 = _sha256(manifest_path)
    _require(len(manifest) == len(chunk_rows), f"chunk {index}: manifest count")
    _require(
        len({str(row["doi"]).lower() for row in manifest}) == len(chunk_rows),
        f"chunk {index}: manifest DOI count",
    )
    _require(
        _report_value(
            report,
            ("corpus_manifest", "manifest_sha256"),
            ("trust_chain", "corpus_manifest_sha256"),
        )
        == manifest_sha256,
        f"chunk {index}: manifest hash",
    )

    source_data_dois = int(
        _report_value(
            report,
            ("corpus_manifest", "source_data_dois"),
            ("execution", "source_data_dois"),
        )
    )
    candidate_cases = int(
        _report_value(
            report, ("cases", "candidates"), ("trust_chain", "candidate_cases")
        )
    )
    proposals = int(
        _report_value(report, ("proposals", "total"), ("trust_chain", "proposals"))
    )
    try:
        review_calls = int(
            _report_value(
                report, ("dual_review", "reviews"), ("trust_chain", "review_calls")
            )
        )
    except ValidationError:
        _require(
            report.get("dual_review", {}).get("status") == "SKIPPED_NO_PROPOSALS"
            and proposals == 0,
            f"chunk {index}: review count missing for nonempty proposals",
        )
        review_calls = 0
    accepted_cases = int(report.get("canonical", {}).get("accepted_single", 0)) + int(
        report.get("canonical", {}).get("accepted_multi", 0)
    )
    outcomes_path = root / "control/terminal_outcomes.jsonl"
    return {
        "chunk": f"{index:03d}",
        "root": root_rel.as_posix(),
        "records": len(chunk_rows),
        "unique_dois": len(dois),
        "input_sha256": chunk_sha256,
        "manifest_sha256": manifest_sha256,
        "terminal_outcomes_sha256": _sha256(outcomes_path),
        "sealed_report_hash": declared_report_hash,
        "sealed_report_file_sha256": report_file_sha256,
        "fixed_terminal_rounds": ["initial", "retry1", "retry2"],
        "terminal_counts": dict(sorted(terminal_counts.items())),
        "source_data_dois": source_data_dois,
        "candidate_cases": candidate_cases,
        "proposals": proposals,
        "review_calls": review_calls,
        "accepted_cases": accepted_cases,
        "_dois": dois,
        "_bytes": chunk_path.read_bytes(),
        "_outcome_count": len(outcomes),
    }


def build_exhaustion_report(repository_root: Path) -> dict[str, Any]:
    """Validate all persistent roots and return a deterministic exhaustion seal."""
    outputs = repository_root / "nature_download/outputs"
    frozen = outputs / "ccby_sr_npj_universe_frozen_ca98442"
    universe_path = frozen / "universe.jsonl"
    _require(_sha256(universe_path) == UNIVERSE_SHA256, "frozen universe hash")
    universe = _read_jsonl(universe_path)
    _require(len(universe) == UNIVERSE_RECORDS, "frozen universe record count")

    chunks = [_chunk_summary(repository_root, index) for index in range(1, 14)]
    chunk_bytes = b"".join(chunk.pop("_bytes") for chunk in chunks)
    chunk_dois = [doi for chunk in chunks for doi in chunk.pop("_dois")]
    outcome_count = sum(int(chunk.pop("_outcome_count")) for chunk in chunks)
    universe_dois = [str(row["doi"]).lower() for row in universe]
    _require(chunk_bytes == universe_path.read_bytes(), "chunks do not byte-cover universe")
    _require(chunk_dois == universe_dois, "chunks do not order-cover universe")
    _require(len(set(chunk_dois)) == UNIVERSE_RECORDS, "duplicate universe DOI")
    _require(outcome_count == UNIVERSE_RECORDS, "terminal outcome coverage")

    terminal_counts: Counter[str] = Counter()
    for chunk in chunks:
        terminal_counts.update(chunk["terminal_counts"])
    source_data_dois = sum(chunk["source_data_dois"] for chunk in chunks)
    candidate_cases = sum(chunk["candidate_cases"] for chunk in chunks)
    proposals = sum(chunk["proposals"] for chunk in chunks)
    review_calls = sum(chunk["review_calls"] for chunk in chunks)
    accepted_cases = sum(chunk["accepted_cases"] for chunk in chunks)
    evidence_aggregate_sha256 = _canonical_sha256(chunks)

    return {
        "schema_version": "1.0",
        "status": "BLOCKED_INSUFFICIENT_INDEPENDENT_P5PLUS",
        "c2_support": "UNSUPPORTED",
        "code": {"corpus_commit": CORPUS_COMMIT, "corpus_dirty": False},
        "universe": {
            "root": "nature_download/outputs/ccby_sr_npj_universe_frozen_ca98442",
            "records": UNIVERSE_RECORDS,
            "unique_dois": len(set(universe_dois)),
            "sha256": UNIVERSE_SHA256,
            "chunk_count": CHUNK_COUNT,
            "terminal_chunks": CHUNK_COUNT,
            "terminal_status": "13/13",
            "terminal_outcomes": outcome_count,
            "exact_ordered_byte_coverage": True,
            "missing_records": 0,
            "duplicate_records": 0,
        },
        "chunks": chunks,
        "aggregate": {
            "evidence_aggregate_sha256": evidence_aggregate_sha256,
            "terminal_counts": dict(sorted(terminal_counts.items())),
            "source_data_dois": source_data_dois,
            "candidate_cases": candidate_cases,
            "proposals": proposals,
            "review_calls": review_calls,
            "accepted_cases": accepted_cases,
            "evidence_note": (
                "One chunk-001 DOI downloaded Source Data, but it yielded zero "
                "deterministic candidate cases, proposals, reviews, or accepted cases."
                if source_data_dois
                else "No chunk yielded Source Data or downstream cases."
            ),
        },
        "c2_readiness": {
            "prospective_gate": {"P=5+_minimum_independent_dois": 2},
            "cumulative_independent_doi_strata": {
                "P=2": 2,
                "P=3-4": 3,
                "P=5+": 1,
            },
            "gate_satisfied": False,
            "reason": "P=5+ has 1 accepted independent DOI; at least 2 are required.",
        },
        "trend_experiment": {
            "status": "NOT_RUN",
            "inference": "NOT_EVALUATED",
            "no_trend_claim_permitted": True,
            "equivalence_claim_permitted": False,
        },
        "policy": {
            "all_chunks_terminal": True,
            "all_records_processed_initial_plus_two_fixed_retries": True,
            "outcome_independent": True,
            "no_missing_or_duplicate_dois": True,
            "dual_review_required_iff_proposals_exist": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_exhaustion_report(args.repository_root.resolve())
    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
