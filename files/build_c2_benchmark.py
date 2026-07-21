#!/usr/bin/env python3
"""Assemble a verified C2 benchmark without weakening the sealed-corpus gate.

This is intentionally a thin, offline wrapper around the canonical
``assemble-benchmark`` CLI.  It never edits a candidate or evidence artifact:
review evidence must first have been rebound by ``build-cases --evidence`` into
verified ``candidates.jsonl`` files.  That prevents an unreviewed proposal from
being relabelled as curation-verified by this convenience script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


class C2BenchmarkBuildError(RuntimeError):
    """Raised when inputs cannot safely enter the sealed benchmark builder."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve(strict=False)
    if path.is_symlink() or not path.is_file():
        raise C2BenchmarkBuildError(f"{label} is not a regular file: {path}")
    return path


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise C2BenchmarkBuildError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise C2BenchmarkBuildError(f"{label} must be a JSON object: {path}")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise C2BenchmarkBuildError(f"Cannot read {label}: {path}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise C2BenchmarkBuildError(
                f"{label} has invalid JSONL at {path}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise C2BenchmarkBuildError(
                f"{label} record is not an object at {path}:{line_number}"
            )
        records.append(value)
    if not records:
        raise C2BenchmarkBuildError(f"{label} is empty: {path}")
    return records


def _candidate_id(record: Mapping[str, Any], label: str) -> str:
    value = record.get("candidate_id")
    if not isinstance(value, str) or not value.strip():
        raise C2BenchmarkBuildError(f"{label} has no non-empty candidate_id")
    return value


def _assert_clean_worktree(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise C2BenchmarkBuildError(
            f"Cannot inspect Git worktree: {completed.stderr.strip()}"
        )
    if completed.stdout:
        raise C2BenchmarkBuildError(
            "Refusing sealed assembly from a dirty worktree. Commit/stash tracked "
            "and untracked changes first; this matches corpus assemble-benchmark."
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if commit.returncode != 0 or not commit.stdout.strip():
        raise C2BenchmarkBuildError("Cannot determine the Git commit for assembly.")
    return commit.stdout.strip()


def _bundle_evidence(
    proposed: Path,
    reviews: Path,
    evidence: Path,
    *,
    bundle_number: int,
) -> dict[str, dict[str, Any]]:
    """Perform a cheap preflight; canonical code repeats the full validation."""

    proposed_records = _load_jsonl(proposed, "proposed")
    review_records = _load_jsonl(reviews, "reviews")
    evidence_payload = _load_json(evidence, "evidence")
    if evidence_payload.get("input_proposed_sha256") != _sha256_file(proposed):
        raise C2BenchmarkBuildError(
            f"review bundle {bundle_number} evidence is not bound to proposed.jsonl"
        )

    proposed_ids = {_candidate_id(record, "proposal") for record in proposed_records}
    review_ids = {_candidate_id(record, "review") for record in review_records}
    if review_ids != proposed_ids:
        raise C2BenchmarkBuildError(
            f"review bundle {bundle_number} review/proposal candidate sets differ"
        )

    raw_verifications = evidence_payload.get("verifications")
    if not isinstance(raw_verifications, list) or not raw_verifications:
        raise C2BenchmarkBuildError(
            f"review bundle {bundle_number} has no verified evidence records"
        )
    result: dict[str, dict[str, Any]] = {}
    for record in raw_verifications:
        if not isinstance(record, dict):
            raise C2BenchmarkBuildError(
                f"review bundle {bundle_number} has a non-object verification"
            )
        candidate_id = _candidate_id(record, "verification")
        if candidate_id in result:
            raise C2BenchmarkBuildError(
                f"review bundle {bundle_number} duplicates verification {candidate_id}"
            )
        if (
            candidate_id not in proposed_ids
            or record.get("status") != "verified"
            or record.get("curation_status") != "verified"
        ):
            raise C2BenchmarkBuildError(
                f"review bundle {bundle_number} has an invalid verification "
                f"for {candidate_id}"
            )
        result[candidate_id] = record
    return result


def _preflight(
    *,
    candidate_paths: Iterable[Path],
    proposed_paths: Iterable[Path],
    review_paths: Iterable[Path],
    evidence_paths: Iterable[Path],
) -> dict[str, Any]:
    evidence_by_id: dict[str, dict[str, Any]] = {}
    bundle_details: list[dict[str, Any]] = []
    for index, (proposed, reviews, evidence) in enumerate(
        zip(proposed_paths, review_paths, evidence_paths, strict=True),
        1,
    ):
        bundle_evidence = _bundle_evidence(
            proposed,
            reviews,
            evidence,
            bundle_number=index,
        )
        overlap = sorted(set(evidence_by_id) & set(bundle_evidence))
        if overlap:
            raise C2BenchmarkBuildError(
                "candidate IDs cannot appear in multiple review bundles: "
                + ", ".join(overlap)
            )
        evidence_by_id.update(bundle_evidence)
        bundle_details.append(
            {
                "bundle": index,
                "proposed": str(proposed),
                "reviews": str(reviews),
                "evidence": str(evidence),
                "verified_evidence_records": len(bundle_evidence),
            }
        )

    seen_candidates: set[str] = set()
    eligible_candidates: list[str] = []
    for candidate_path in candidate_paths:
        summary_path = candidate_path.parent / "summary.json"
        digest_path = candidate_path.parent / "summary.sha256"
        if not summary_path.is_file() or not digest_path.is_file():
            raise C2BenchmarkBuildError(
                "verified candidate batch lacks its case-builder summary seal: "
                f"{candidate_path.parent}"
            )
        for candidate in _load_jsonl(candidate_path, "candidates"):
            candidate_id = _candidate_id(candidate, "candidate")
            if candidate_id in seen_candidates:
                raise C2BenchmarkBuildError(
                    f"candidate ID is duplicated across candidate inputs: {candidate_id}"
                )
            seen_candidates.add(candidate_id)
            if candidate.get("eligible_for_experiment") is not True:
                continue
            verification = evidence_by_id.get(candidate_id)
            if (
                candidate.get("curation_status") != "verified"
                or verification is None
                or candidate.get("verification_evidence") != verification
            ):
                raise C2BenchmarkBuildError(
                    "eligible candidate is not a rebuild bound to review evidence: "
                    f"{candidate_id}. Re-run build-cases --evidence; do not edit "
                    "curation_status or eligibility fields."
                )
            eligible_candidates.append(candidate_id)
    if not eligible_candidates:
        raise C2BenchmarkBuildError(
            "No curation-verified candidates were supplied. evidence.json alone "
            "cannot form a sealed benchmark; rebuild candidates with build-cases "
            "--evidence first."
        )
    return {
        "candidate_inputs": [str(path) for path in candidate_paths],
        "candidate_records": len(seen_candidates),
        "verified_candidate_records": len(eligible_candidates),
        "review_bundles": bundle_details,
    }


def _qualified_multi_cases(manifest: Mapping[str, Any], minimum_panels: int) -> list[dict[str, Any]]:
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list):
        raise C2BenchmarkBuildError("Assembled manifest has no cases array.")
    cases: list[dict[str, Any]] = []
    for case in raw_cases:
        if not isinstance(case, dict):
            continue
        panel_count = case.get("panel_count")
        if (
            isinstance(panel_count, int)
            and not isinstance(panel_count, bool)
            and panel_count >= minimum_panels
            and case.get("chart_family") == "multi_panel"
            and case.get("curation_status") == "verified"
            and case.get("eligible_for_experiment") is True
        ):
            cases.append(case)
    return cases


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight verified C2 review bundles and invoke the canonical "
            "assemble-benchmark command."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--proposed", action="append", required=True)
    parser.add_argument("--reviews", action="append", required=True)
    parser.add_argument("--evidence", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--minimum-multi-panels", type=int, default=5)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate local review/candidate bindings without writing a manifest.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        repo_root = args.repo_root.expanduser().resolve(strict=True)
        if not (repo_root / ".git").exists():
            raise C2BenchmarkBuildError(f"--repo-root is not a Git checkout: {repo_root}")
        if args.minimum_multi_panels < 2:
            raise C2BenchmarkBuildError("--minimum-multi-panels must be at least 2")
        if not (
            len(args.proposed) == len(args.reviews) == len(args.evidence)
        ):
            raise C2BenchmarkBuildError(
                "--proposed, --reviews, and --evidence must have equal counts"
            )

        candidate_paths = [_resolve_file(value, "candidate") for value in args.candidate]
        proposed_paths = [_resolve_file(value, "proposed") for value in args.proposed]
        review_paths = [_resolve_file(value, "reviews") for value in args.reviews]
        evidence_paths = [_resolve_file(value, "evidence") for value in args.evidence]
        commit = _assert_clean_worktree(repo_root)
        preflight = _preflight(
            candidate_paths=candidate_paths,
            proposed_paths=proposed_paths,
            review_paths=review_paths,
            evidence_paths=evidence_paths,
        )
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "preflight-ok-no-manifest-written",
                        "code_commit": commit,
                        "minimum_multi_panels": args.minimum_multi_panels,
                        **preflight,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        output = args.out.expanduser().resolve(strict=False)
        if output.exists() and any(output.iterdir()):
            raise C2BenchmarkBuildError(
                f"--out must be new or empty; refusing to reuse {output}"
            )
        corpus_cli = repo_root / "nature_download" / "nature_all_in_one.py"
        if not corpus_cli.is_file():
            raise C2BenchmarkBuildError(f"Canonical corpus CLI is missing: {corpus_cli}")
        command = [
            args.python,
            str(corpus_cli),
            "assemble-benchmark",
            *[
                value
                for path in candidate_paths
                for value in ("--candidates", str(path))
            ],
            *[
                value
                for proposed, reviews, evidence in zip(
                    proposed_paths,
                    review_paths,
                    evidence_paths,
                    strict=True,
                )
                for value in (
                    "--proposed",
                    str(proposed),
                    "--reviews",
                    str(reviews),
                    "--evidence",
                    str(evidence),
                )
            ],
            "--seed",
            str(args.seed),
            "--out",
            str(output),
        ]
        completed = subprocess.run(command, cwd=repo_root / "nature_download", check=False)
        if completed.returncode != 0:
            raise C2BenchmarkBuildError(
                f"canonical assemble-benchmark failed with exit code {completed.returncode}"
            )

        manifest_path = output / "benchmark_manifest.json"
        summary_path = output / "summary.json"
        manifest = _load_json(manifest_path, "assembled benchmark manifest")
        summary = _load_json(summary_path, "assembled benchmark summary")
        qualified = _qualified_multi_cases(manifest, args.minimum_multi_panels)
        qualified_dois = sorted(
            {
                str(case["doi"]).casefold()
                for case in qualified
                if isinstance(case.get("doi"), str)
            }
        )
        result = {
            "status": "assembled",
            "code_commit": commit,
            "benchmark_manifest": str(manifest_path),
            "benchmark_manifest_sha256": summary.get("benchmark_manifest_sha256"),
            "minimum_multi_panels": args.minimum_multi_panels,
            "qualified_multi_cases": len(qualified),
            "qualified_multi_dois": len(qualified_dois),
            "qualified_case_ids": sorted(str(case.get("case_id")) for case in qualified),
            **preflight,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        if not qualified:
            raise C2BenchmarkBuildError(
                "Assembly succeeded but contains no verified multi-panel case at "
                f"P>={args.minimum_multi_panels}; do not launch C2-extreme."
            )
        return 0
    except C2BenchmarkBuildError as exc:
        print(f"C2 sealed benchmark build refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
