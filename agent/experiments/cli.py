from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .aggregate import AggregationError, aggregate_runs
from .harness import ExistingRunError, execute_experiment
from .matrix import MatrixError, load_and_expand_matrix
from .models import ProvenanceError
from .production_statistics import (
    analyze_summary,
    build_holm_family,
    load_provenance_summary,
    load_trajectory_threshold_config,
    write_analysis_outputs,
    write_holm_family_output,
)
from .rejudge import merge_rejudged_summary, rejudge_batch


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phero-experiments",
        description="Provenance-strict PheroViz experiment harness",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Expand and execute a YAML/JSON experiment matrix",
    )
    run_parser.add_argument("spec", type=Path)
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print expanded immutable run specs without creating run directories",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip valid completed runs and retry recorded failures",
    )
    run_parser.add_argument(
        "--fail-fast-nonmethod",
        action="store_true",
        help=(
            "Stop after a failed run unless it has explicit method "
            "attribution"
        ),
    )

    aggregate_parser = subparsers.add_parser(
        "aggregate",
        help="Create strict summary.csv and summary.json files",
    )
    aggregate_parser.add_argument("run_root", type=Path)
    aggregate_parser.add_argument("--output-dir", type=Path, default=None)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run provenance-strict paired production statistics",
    )
    analyze_parser.add_argument("summary", type=Path)
    analyze_parser.add_argument("--reference", required=True)
    analyze_parser.add_argument("--methods", nargs="+", required=True)
    analyze_parser.add_argument("--metric", required=True)
    analyze_parser.add_argument(
        "--panel-scope",
        choices=("all", "single_panel", "multi_panel"),
        default="all",
    )
    analyze_parser.add_argument("--second-judge-metric", default=None)
    analyze_parser.add_argument(
        "--trajectory-threshold-config",
        type=Path,
        default=None,
        help=(
            "Explicit frozen C3 joint fidelity/cohesion threshold JSON; "
            "requires --panel-scope multi_panel"
        ),
    )
    analyze_parser.add_argument("--out", type=Path, required=True)
    analyze_parser.add_argument("--seed", type=int, default=17_029)
    analyze_parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10_000,
    )
    analyze_parser.add_argument(
        "--permutations",
        type=int,
        default=100_000,
    )
    analyze_parser.add_argument("--exact-max-n", type=int, default=16)

    holm_parser = subparsers.add_parser(
        "holm-family",
        help="Apply Holm adjustment to a declared cross-analysis family",
    )
    holm_parser.add_argument("manifest", type=Path)
    holm_parser.add_argument("--out", type=Path, required=True)

    rejudge_parser = subparsers.add_parser(
        "rejudge",
        help="Read sealed best-candidate renders and run a post-hoc visual judge",
    )
    rejudge_parser.add_argument("source", type=Path)
    rejudge_parser.add_argument("--judge-model", required=True)
    rejudge_parser.add_argument("--out", type=Path, default=None)
    rejudge_parser.add_argument("--resume", action="store_true")

    merge_parser = subparsers.add_parser(
        "merge-rejudge",
        help="Create rejudged_summary.json without changing the source summary",
    )
    merge_parser.add_argument("summary", type=Path)
    merge_parser.add_argument("sidecar_dir", type=Path)
    merge_parser.add_argument("--out", type=Path, default=None)
    return parser


def _run_command(args: argparse.Namespace) -> int:
    specs = load_and_expand_matrix(args.spec)
    if args.dry_run:
        payload = [
            {**spec.to_dict(), "spec_hash": spec.spec_hash}
            for spec in specs
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    failures = 0
    results = []
    for spec in specs:
        try:
            outcome = execute_experiment(spec, resume=args.resume)
        except ExistingRunError as exc:
            failures += 1
            results.append(
                {
                    "run_name": spec.run_name,
                    "status": "not_started",
                    "error": str(exc),
                }
            )
            continue
        record = outcome.record
        if record.status != "completed":
            failures += 1
        results.append(
            {
                "run_name": record.run_name,
                "status": record.status,
                "skipped": outcome.skipped,
                "record": str(
                    Path(spec.artifact_root)
                    / spec.run_name
                    / "run_record.json"
                ),
                "error": record.error,
            }
        )
        if (
            args.fail_fast_nonmethod
            and record.status == "failed"
            and (record.error or {}).get("attribution") != "method"
        ):
            break
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if failures else 0


def _aggregate_command(args: argparse.Namespace) -> int:
    csv_path, json_path = aggregate_runs(
        args.run_root,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "summary_csv": str(csv_path),
                "summary_json": str(json_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _analyze_command(args: argparse.Namespace) -> int:
    summary = load_provenance_summary(args.summary)
    analysis = analyze_summary(
        summary,
        reference=args.reference,
        methods=args.methods,
        metric=args.metric,
        panel_scope=args.panel_scope,
        second_judge_metric=args.second_judge_metric,
        trajectory_threshold=(
            load_trajectory_threshold_config(
                args.trajectory_threshold_config
            )
            if args.trajectory_threshold_config is not None
            else None
        ),
        seed=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
        monte_carlo_permutations=args.permutations,
        exact_max_n=args.exact_max_n,
    )
    json_path, csv_path = write_analysis_outputs(analysis, args.out)
    print(
        json.dumps(
            {
                "analysis_json": str(json_path),
                "analysis_csv": str(csv_path),
                "analysis_hash": analysis["analysis_hash"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _holm_family_command(args: argparse.Namespace) -> int:
    family = build_holm_family(args.manifest)
    path = write_holm_family_output(family, args.out)
    print(
        json.dumps(
            {
                "holm_family": str(path),
                "family_hash": family["family_hash"],
                "code_git_commit": family["code_git_commit"],
                "code_git_dirty": family["code_git_dirty"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _rejudge_command(args: argparse.Namespace) -> int:
    result = rejudge_batch(
        args.source,
        judge_model=args.judge_model,
        output_dir=args.out,
        resume=args.resume,
    )
    print(
        json.dumps(
            result.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return result.exit_code


def _merge_rejudge_command(args: argparse.Namespace) -> int:
    path, metric = merge_rejudged_summary(
        args.summary,
        args.sidecar_dir,
        output_path=args.out,
    )
    print(
        json.dumps(
            {
                "rejudged_summary": str(path),
                "second_judge_metric": metric,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _run_command(args)
        if args.command == "aggregate":
            return _aggregate_command(args)
        if args.command == "analyze":
            return _analyze_command(args)
        if args.command == "holm-family":
            return _holm_family_command(args)
        if args.command == "rejudge":
            return _rejudge_command(args)
        if args.command == "merge-rejudge":
            return _merge_rejudge_command(args)
    except (AggregationError, MatrixError, ProvenanceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"Unknown command: {args.command}")
    return 2
