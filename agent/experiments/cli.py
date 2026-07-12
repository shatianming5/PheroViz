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
    load_provenance_summary,
    write_analysis_outputs,
)


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
    analyze_parser.add_argument("--second-judge-metric", default=None)
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
        second_judge_metric=args.second_judge_metric,
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
    except (AggregationError, MatrixError, ProvenanceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"Unknown command: {args.command}")
    return 2
