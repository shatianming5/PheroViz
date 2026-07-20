#!/usr/bin/env python3
"""Build and run C2 cases from a frozen article-directory dataset.

The strict path preserves the existing fail-closed corpus builder and emits only
unverified proposals.  Those proposals can be used for explicitly labelled
legacy/exploratory runs, but they are not a sealed benchmark.  A separately
assembled sealed manifest can be supplied with ``--dataset-manifest``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = REPO_ROOT / "agent"
NATURE_ROOT = REPO_ROOT / "nature_download"
for import_root in (AGENT_ROOT, NATURE_ROOT):
    text = str(import_root)
    if text not in sys.path:
        sys.path.insert(0, text)

from corpus.cases import build_cases, write_case_outputs
from corpus.proposals import (
    ProposalRejected,
    propose_cases,
    write_proposal_outputs,
)
from corpus.provenance import sha256_file, validate_article_manifest
from corpus.source_table_normalizer import (
    NormalizerRejected,
    normalize_source_sheet,
)
from experiments.manifest import DatasetCase, load_dataset_manifest


class C2PipelineError(RuntimeError):
    """Raised when a frozen C2 pipeline stage cannot be safely completed."""


def _json_default(value: object) -> object:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    dict(record),
                    ensure_ascii=False,
                    sort_keys=True,
                    default=_json_default,
                )
                + "\n"
            )


def _git_text(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise C2PipelineError(f"Cannot establish repository provenance: {exc}") from exc
    return completed.stdout.strip()


def _require_clean_worktree() -> str:
    status = _git_text("status", "--porcelain")
    if status:
        raise C2PipelineError(
            "C2 corpus construction requires a clean worktree; commit or stash "
            "tracked changes before starting."
        )
    commit = _git_text("rev-parse", "HEAD")
    if not commit:
        raise C2PipelineError("Cannot determine the current git commit.")
    return commit


def _regular_source_files(article_dir: Path) -> list[Path]:
    source_dir = article_dir / "source_data"
    if source_dir.is_symlink() or not source_dir.is_dir():
        return []
    return sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and not path.is_symlink()
    )


def _frozen_records(
    input_root: Path,
    *,
    max_articles: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    try:
        entries = sorted(input_root.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise C2PipelineError(f"Cannot list frozen input {input_root}: {exc}") from exc

    for article_dir in entries:
        if (
            article_dir.name.startswith("_")
            or article_dir.is_symlink()
            or not article_dir.is_dir()
        ):
            continue
        provenance_path = article_dir / "meta" / "provenance.json"
        if not provenance_path.is_file() or provenance_path.is_symlink():
            skipped.append(
                {
                    "article_id": article_dir.name,
                    "reason": "provenance-missing",
                }
            )
            continue
        if not _regular_source_files(article_dir):
            skipped.append(
                {
                    "article_id": article_dir.name,
                    "reason": "source-data-missing",
                }
            )
            continue
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            skipped.append(
                {
                    "article_id": article_dir.name,
                    "reason": f"provenance-invalid:{type(exc).__name__}",
                }
            )
            continue
        if not isinstance(provenance, dict):
            skipped.append(
                {
                    "article_id": article_dir.name,
                    "reason": "provenance-not-object",
                }
            )
            continue
        errors = validate_article_manifest(provenance, content_root=input_root)
        if errors:
            skipped.append(
                {
                    "article_id": article_dir.name,
                    "reason": "provenance-validation-failed",
                    "errors": errors,
                }
            )
            continue
        selected.append(provenance)

    selected.sort(key=lambda record: str(record.get("doi") or ""))
    if max_articles is not None:
        selected = selected[:max_articles]
    if not selected:
        raise C2PipelineError(
            "No frozen article directories passed provenance and source-data checks."
        )
    return selected, skipped


def _propose(
    candidates_path: Path,
    output_dir: Path,
    *,
    code_commit: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    proposed, rejected, summary = propose_cases(
        candidates_path=candidates_path,
        code_commit=code_commit,
    )
    write_proposal_outputs(output_dir, proposed, rejected, summary)
    return proposed, rejected, summary


def _normalizer_candidates(
    candidates: list[dict[str, Any]],
    rejected_ids: set[str],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized_dir = output_dir / "normalized_source_tables"
    normalized_dir.mkdir(parents=True, exist_ok=False)
    transformed: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()

    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id or candidate_id not in rejected_ids:
            continue
        source = candidate.get("source_table")
        if not isinstance(source, Mapping):
            rejected["source-table-invalid"] += 1
            continue
        raw_path = source.get("path")
        sheet = source.get("sheet_name")
        if (
            not isinstance(raw_path, str)
            or Path(raw_path).suffix.casefold() != ".xlsx"
            or not isinstance(sheet, str)
            or not sheet.strip()
        ):
            rejected["normalizer-requires-xlsx-sheet"] += 1
            continue
        try:
            result = normalize_source_sheet(raw_path, sheet)
        except NormalizerRejected as exc:
            rejected[exc.reason] += 1
            continue
        except (OSError, ValueError) as exc:
            rejected[f"normalizer-error:{type(exc).__name__}"] += 1
            continue

        normalized_path = normalized_dir / f"{candidate_id}.csv"
        result.frame.to_csv(normalized_path, index=False, encoding="utf-8")
        normalized_hash = sha256_file(normalized_path)
        replacement = copy.deepcopy(candidate)
        replacement["raw_source_table"] = copy.deepcopy(dict(source))
        replacement["source_table"] = {
            "path": str(normalized_path.resolve()),
            "sha256": normalized_hash,
            "size_bytes": normalized_path.stat().st_size,
            "format": "csv",
            "sheet_name": None,
            "path_root": "exploratory_normalizer_output",
            "relative_path": normalized_path.relative_to(output_dir).as_posix(),
            "source_url": source.get("source_url"),
        }
        replacement["source_normalization"] = {
            "schema_version": "exploratory-c2-source-normalizer-v1",
            "implementation": (
                "nature_download.corpus.source_table_normalizer."
                "normalize_source_sheet"
            ),
            "raw_source_table": copy.deepcopy(dict(source)),
            "normalizer_output_sha256": normalized_hash,
            "orientation": result.orientation,
            "header_row_index": result.header_row_index,
            "dropped_side_blocks": result.dropped_side_blocks,
            "aggregated_replicates": result.aggregated_replicates,
            "notes": list(result.notes),
            "warning": (
                "EXPLORATORY_ONLY: this is a deterministic source-table "
                "derivative, not a sealed C2 benchmark input."
            ),
        }
        transformed.append(replacement)

    transformed.sort(key=lambda record: str(record.get("candidate_id") or ""))
    summary = {
        "schema_version": "exploratory-c2-source-normalizer-v1",
        "input_rejected_candidates": len(rejected_ids),
        "normalized_candidates": len(transformed),
        "normalizer_rejections": dict(sorted(rejected.items())),
        "warning": (
            "EXPLORATORY_ONLY: normalized tables are not eligible for sealed C2 "
            "benchmark assembly without a separately approved provenance contract."
        ),
    }
    _write_json(output_dir / "normalizer_summary.json", summary)
    return transformed, summary


def _select_proposals(
    proposals: list[dict[str, Any]],
    *,
    case_kind: str,
    max_cases: int | None,
) -> list[dict[str, Any]]:
    if case_kind == "single":
        proposals = [
            proposal
            for proposal in proposals
            if proposal.get("proposal_type") == "single_panel"
        ]
    elif case_kind == "multi":
        proposals = [
            proposal
            for proposal in proposals
            if proposal.get("proposal_type") == "multi_panel"
        ]
    selected = sorted(
        proposals,
        key=lambda proposal: str(proposal.get("candidate_id") or ""),
    )
    if max_cases is not None:
        selected = selected[:max_cases]
    if not selected:
        raise C2PipelineError("Case selection produced no proposals.")
    return selected


def _proposal_case_info(proposals: Iterable[Mapping[str, Any]]) -> list[tuple[str, int]]:
    cases: list[tuple[str, int]] = []
    for proposal in proposals:
        case = proposal.get("experiment_case")
        if not isinstance(case, Mapping):
            raise C2PipelineError("Proposal has no experiment_case object.")
        case_id = case.get("case_id")
        panel_count = case.get("panel_count")
        if (
            not isinstance(case_id, str)
            or not case_id
            or isinstance(panel_count, bool)
            or not isinstance(panel_count, int)
            or panel_count < 1
        ):
            raise C2PipelineError("Proposal has an invalid case_id or panel_count.")
        cases.append((case_id, panel_count))
    return cases


def _manifest_case_info(
    manifest: Path,
    *,
    manifest_data_root: Path,
    dataset_mode: str,
    case_kind: str,
    max_cases: int | None,
    min_panels: int | None,
) -> list[tuple[str, int]]:
    try:
        cases = load_dataset_manifest(
            manifest,
            dataset_mode=dataset_mode,
            manifest_data_root=manifest_data_root,
            runtime_repo_root=REPO_ROOT,
        )
    except Exception as exc:
        raise C2PipelineError(
            f"{dataset_mode} dataset manifest validation failed: {exc}"
        ) from exc
    selected: list[DatasetCase] = list(cases)
    if case_kind == "single":
        selected = [case for case in selected if case.panel_count == 1]
    elif case_kind == "multi":
        selected = [
            case
            for case in selected
            if isinstance(case.panel_count, int) and case.panel_count > 1
        ]
    if min_panels is not None:
        selected = [
            case
            for case in selected
            if isinstance(case.panel_count, int) and case.panel_count >= min_panels
        ]
    selected.sort(key=lambda case: case.case_id)
    if max_cases is not None:
        selected = selected[:max_cases]
    if not selected:
        raise C2PipelineError("Case selection produced no sealed benchmark cases.")
    return [
        (case.case_id, int(case.panel_count))
        for case in selected
        if isinstance(case.panel_count, int)
    ]


def _methods_for_profile(profile: str, *, offline_defaults: bool) -> list[dict[str, Any]]:
    if offline_defaults:
        return [
            {
                "name": "offline_defaults",
                "schedule": "iterative",
                "memory_mode": "none",
                "initial_generation": "defaults",
                "render_timeout_seconds": 120,
            }
        ]
    if profile == "smoke":
        return [
            {
                "name": "pheroviz_full",
                "schedule": "iterative",
                "memory_mode": "full",
                "initial_generation": "model_spec",
                "render_timeout_seconds": 120,
            }
        ]
    return [
        {
            "name": "best_of_n",
            "schedule": "best_of_n",
            "memory_mode": "none",
            "initial_generation": "model_spec",
            "render_timeout_seconds": 120,
        },
        {
            "name": "flat_iterative",
            "schedule": "iterative",
            "memory_mode": "none",
            "initial_generation": "model_spec",
            "render_timeout_seconds": 120,
        },
        {
            "name": "pheroviz_full",
            "schedule": "iterative",
            "memory_mode": "full",
            "initial_generation": "model_spec",
            "render_timeout_seconds": 120,
        },
    ]


def _write_matrices(
    *,
    output_dir: Path,
    dataset_manifest: Path,
    dataset_mode: str,
    cases: list[tuple[str, int]],
    profile: str,
    rounds_per_case: int,
    model: str,
    offline_defaults: bool,
    manifest_data_root: Path,
) -> list[Path]:
    grouped: dict[int, list[str]] = {}
    for case_id, panel_count in cases:
        grouped.setdefault(panel_count, []).append(case_id)
    matrices_dir = output_dir / "matrices"
    matrices_dir.mkdir(parents=True, exist_ok=False)
    manifest_hash = sha256_file(dataset_manifest)
    methods = _methods_for_profile(profile, offline_defaults=offline_defaults)
    matrices: list[Path] = []
    for panel_count, case_ids in sorted(grouped.items()):
        budget = panel_count * rounds_per_case
        provider_options: dict[str, Any] = {
            "manifest_data_root": str(manifest_data_root),
        }
        if offline_defaults:
            provider_options["offline_defaults"] = True
        matrix = {
            "experiment_name": (
                f"c2-{profile}-p{panel_count}"
                + ("-offline-defaults" if offline_defaults else "")
            ),
            "dataset_manifest": str(dataset_manifest),
            "dataset_manifest_sha256": manifest_hash,
            "dataset_mode": dataset_mode,
            "artifact_root": str((output_dir / "agent_runs" / f"p{panel_count}").resolve()),
            "provider": "experiments.providers:UnifiedBenchmarkProvider",
            "provider_options": provider_options,
            "methods": methods,
            "backbones": ["offline-default-v2" if offline_defaults else model],
            "seeds": [0] if profile == "smoke" or offline_defaults else [0, 1, 2],
            "budgets": [{"type": "renders", "value": budget}],
            "case_ids": sorted(case_ids),
            "metric": {
                "version": "programmatic-v2",
                "config": {
                    "selection": {
                        "metric": "data_fidelity",
                        "direction": "maximize",
                    }
                },
            },
        }
        path = matrices_dir / f"panel_count_{panel_count}.json"
        _write_json(path, matrix)
        matrices.append(path)
    return matrices


def _execute_matrices(
    matrices: Iterable[Path],
    *,
    python: str,
    dry_run: bool,
) -> list[dict[str, Any]]:
    executions: list[dict[str, Any]] = []
    for matrix in matrices:
        command = [python, "-m", "experiments", "run", str(matrix)]
        if dry_run:
            command.append("--dry-run")
        completed = subprocess.run(command, cwd=AGENT_ROOT, check=False)
        executions.append(
            {
                "matrix": str(matrix),
                "command": command,
                "returncode": completed.returncode,
            }
        )
    return executions


def _build_from_input(
    *,
    input_root: Path,
    output_dir: Path,
    max_articles: int | None,
    normalizer_exploratory: bool,
    code_commit: str,
) -> tuple[Path | None, list[dict[str, Any]], dict[str, Any]]:
    records, skipped = _frozen_records(input_root, max_articles=max_articles)
    corpus_manifest = output_dir / "corpus_manifest.jsonl"
    _write_jsonl(corpus_manifest, records)
    _write_json(
        output_dir / "input_selection.json",
        {
            "input_root": str(input_root),
            "selected_articles": len(records),
            "selected_dois": [record["doi"] for record in records],
            "skipped": skipped,
        },
    )

    strict_root = output_dir / "strict"
    candidates, ambiguous, case_summary = build_cases(
        corpus_manifest=corpus_manifest,
        content_root=input_root,
        output_root=strict_root / "cases",
    )
    case_summary = write_case_outputs(
        strict_root / "cases",
        candidates,
        ambiguous,
        case_summary,
        code_commit=code_commit,
        code_dirty=False,
    )
    strict_proposed, strict_rejected, strict_summary = _propose(
        strict_root / "cases" / "candidates.jsonl",
        strict_root / "proposals",
        code_commit=code_commit,
    )
    stage_report: dict[str, Any] = {
        "corpus_manifest": str(corpus_manifest),
        "case_summary": case_summary,
        "strict_proposal_summary": strict_summary,
        "strict_proposals": str(strict_root / "proposals" / "proposed.jsonl"),
    }
    if not normalizer_exploratory:
        return strict_root / "proposals" / "proposed.jsonl", strict_proposed, stage_report

    rejected_ids = {
        str(record.get("candidate_id") or "")
        for record in strict_rejected
        if record.get("candidate_id")
    }
    exploratory_root = output_dir / "exploratory_normalizer"
    transformed, normalizer_summary = _normalizer_candidates(
        candidates,
        rejected_ids,
        exploratory_root,
    )
    transformed_by_id = {
        str(candidate["candidate_id"]): candidate for candidate in transformed
    }
    combined = [
        transformed_by_id.get(str(candidate.get("candidate_id") or ""), candidate)
        for candidate in candidates
    ]
    combined_path = exploratory_root / "candidates.jsonl"
    _write_jsonl(combined_path, combined)
    proposed, _, proposal_summary = _propose(
        combined_path,
        exploratory_root / "proposals",
        code_commit=code_commit,
    )
    stage_report.update(
        {
            "mode": "exploratory-normalizer",
            "normalizer_summary": normalizer_summary,
            "exploratory_proposal_summary": proposal_summary,
            "exploratory_proposals": str(
                exploratory_root / "proposals" / "proposed.jsonl"
            ),
            "warning": (
                "EXPLORATORY_ONLY: normalized source-table derivatives cannot be "
                "used as a sealed C2 benchmark without an approved provenance "
                "and review contract."
            ),
        }
    )
    return exploratory_root / "proposals" / "proposed.jsonl", proposed, stage_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build C2 proposals from frozen article directories, then materialize "
            "compatible agent matrices."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        type=Path,
        help="Frozen article-directory root, such as data/c2_p5_4k.",
    )
    source.add_argument(
        "--dataset-manifest",
        type=Path,
        help="Previously assembled sealed benchmark_manifest.json to execute.",
    )
    source.add_argument(
        "--legacy-manifest",
        type=Path,
        help=(
            "Previously materialized legacy manifest to execute. This is useful "
            "for a hash-pinned multi-panel comparison view."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New empty output directory; it is never reused or overwritten.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--min-panels",
        type=int,
        default=None,
        help="Keep only cases with at least this many panels.",
    )
    parser.add_argument(
        "--case-kind",
        choices=("all", "single", "multi"),
        default="all",
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "benchmark"),
        default="benchmark",
    )
    parser.add_argument(
        "--rounds-per-case",
        type=int,
        default=1,
        help=(
            "Complete global rounds per panel-count shard. Each matrix budget is "
            "panel_count * rounds_per_case."
        ),
    )
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument(
        "--normalizer-exploratory",
        action="store_true",
        help=(
            "Explicitly use the candidate source-table normalizer after strict "
            "proposals fail. This mode is not a sealed benchmark."
        ),
    )
    parser.add_argument(
        "--offline-defaults",
        action="store_true",
        help=(
            "Run one deterministic default-slot render per selected case without "
            "model credentials; artifacts are marked test_only."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute generated experiment matrices.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Expand generated matrices without creating experiment runs.",
    )
    parser.add_argument(
        "--manifest-data-root",
        type=Path,
        default=REPO_ROOT,
        help="Original repository root encoded in a sealed manifest's paths.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_articles is not None and args.max_articles < 1:
        raise SystemExit("--max-articles must be positive")
    if args.max_cases is not None and args.max_cases < 1:
        raise SystemExit("--max-cases must be positive")
    if args.min_panels is not None and args.min_panels < 1:
        raise SystemExit("--min-panels must be positive")
    if args.rounds_per_case < 1:
        raise SystemExit("--rounds-per-case must be positive")
    if args.normalizer_exploratory and args.input is None:
        raise SystemExit(
            "--normalizer-exploratory cannot be used with an existing dataset manifest"
        )
    if args.offline_defaults and args.input is None:
        raise SystemExit("--offline-defaults cannot be used for an existing dataset manifest")
    if args.offline_defaults and args.profile != "smoke":
        raise SystemExit("--offline-defaults requires --profile smoke")
    if args.offline_defaults and args.rounds_per_case != 1:
        raise SystemExit("--offline-defaults requires --rounds-per-case 1")

    output_dir = args.output.expanduser().resolve(strict=False)
    if output_dir.exists():
        raise SystemExit(f"--output already exists; refusing to reuse it: {output_dir}")
    manifest_data_root = args.manifest_data_root.expanduser().resolve(strict=True)
    if not manifest_data_root.is_dir():
        raise SystemExit(
            f"--manifest-data-root is not a directory: {manifest_data_root}"
        )

    try:
        code_commit = _require_clean_worktree()
        if args.input is not None:
            input_root = args.input.expanduser().resolve(strict=True)
            if not input_root.is_dir():
                raise C2PipelineError(f"--input is not a directory: {input_root}")
            if input_root == output_dir or input_root in output_dir.parents:
                raise C2PipelineError("--output must not be inside --input.")
        else:
            input_root = None
        output_dir.mkdir(parents=True, exist_ok=False)

        if input_root is not None:
            dataset_manifest, proposals, stage_report = _build_from_input(
                input_root=input_root,
                output_dir=output_dir,
                max_articles=args.max_articles,
                normalizer_exploratory=args.normalizer_exploratory,
                code_commit=code_commit,
            )
            if not proposals or dataset_manifest is None:
                report = {
                    "status": "blocked-no-proposals",
                    "code_commit": code_commit,
                    "stages": stage_report,
                    "next_step": (
                        "Use --normalizer-exploratory only for an explicitly "
                        "unsealed diagnostic run, or curate simpler source tables "
                        "under the strict proposal contract."
                    ),
                }
                _write_json(output_dir / "pipeline_report.json", report)
                print(
                    "C2 strict proposer produced no runnable cases; see "
                    f"{output_dir / 'pipeline_report.json'}",
                    file=sys.stderr,
                )
                return 3
            selected_proposals = _select_proposals(
                proposals,
                case_kind=args.case_kind,
                max_cases=args.max_cases,
            )
            if args.min_panels is not None:
                selected_proposals = [
                    proposal
                    for proposal in selected_proposals
                    if isinstance(
                        (proposal.get("experiment_case") or {}).get("panel_count"),
                        int,
                    )
                    and (proposal.get("experiment_case") or {})["panel_count"]
                    >= args.min_panels
                ]
                if not selected_proposals:
                    raise C2PipelineError(
                        "Panel-count filtering produced no selected proposals."
                    )
            selected_manifest = output_dir / "selected_proposed.jsonl"
            _write_jsonl(selected_manifest, selected_proposals)
            case_info = _proposal_case_info(selected_proposals)
            dataset_mode = "legacy"
            stage_report["selected_proposals"] = str(selected_manifest)
            stage_report["selected_cases"] = len(selected_proposals)
        elif args.legacy_manifest is not None:
            dataset_manifest = args.legacy_manifest.expanduser().resolve(strict=True)
            if not dataset_manifest.is_file():
                raise C2PipelineError(
                    f"--legacy-manifest is not a file: {dataset_manifest}"
                )
            case_info = _manifest_case_info(
                dataset_manifest,
                manifest_data_root=manifest_data_root,
                dataset_mode="legacy",
                case_kind=args.case_kind,
                max_cases=args.max_cases,
                min_panels=args.min_panels,
            )
            selected_manifest = dataset_manifest
            dataset_mode = "legacy"
            stage_report = {
                "mode": "legacy-manifest-execution",
                "dataset_manifest": str(dataset_manifest),
                "selected_cases": len(case_info),
                "min_panels": args.min_panels,
            }
        else:
            dataset_manifest = args.dataset_manifest.expanduser().resolve(strict=True)
            if not dataset_manifest.is_file():
                raise C2PipelineError(
                    f"--dataset-manifest is not a file: {dataset_manifest}"
                )
            case_info = _manifest_case_info(
                dataset_manifest,
                manifest_data_root=manifest_data_root,
                dataset_mode="sealed_benchmark",
                case_kind=args.case_kind,
                max_cases=args.max_cases,
                min_panels=args.min_panels,
            )
            selected_manifest = dataset_manifest
            dataset_mode = "sealed_benchmark"
            stage_report = {
                "mode": "sealed-benchmark-execution",
                "dataset_manifest": str(dataset_manifest),
                "selected_cases": len(case_info),
            }

        matrices = _write_matrices(
            output_dir=output_dir,
            dataset_manifest=selected_manifest,
            dataset_mode=dataset_mode,
            cases=case_info,
            profile=args.profile,
            rounds_per_case=args.rounds_per_case,
            model=args.model,
            offline_defaults=args.offline_defaults,
            manifest_data_root=manifest_data_root,
        )
        report: dict[str, Any] = {
            "status": "matrices-materialized",
            "code_commit": code_commit,
            "dataset_mode": dataset_mode,
            "profile": args.profile,
            "offline_defaults": args.offline_defaults,
            "rounds_per_case": args.rounds_per_case,
            "matrices": [str(path) for path in matrices],
            "stages": stage_report,
        }
        _write_json(output_dir / "pipeline_report.json", report)

        if args.execute or args.dry_run:
            executions = _execute_matrices(
                matrices,
                python=args.python,
                dry_run=args.dry_run,
            )
            report["executions"] = executions
            report["status"] = (
                "completed"
                if all(item["returncode"] == 0 for item in executions)
                else "execution-failed"
            )
            _write_json(output_dir / "pipeline_report.json", report)
            return 0 if report["status"] == "completed" else 1

        print(
            "C2 matrices materialized. Re-run with --execute, or inspect "
            f"{output_dir / 'pipeline_report.json'}."
        )
        return 0
    except C2PipelineError as exc:
        if output_dir.exists():
            _write_json(
                output_dir / "pipeline_report.json",
                {"status": "failed", "error": str(exc)},
            )
        print(f"C2 pipeline failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
