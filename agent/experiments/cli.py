from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .aggregate import AggregationError, aggregate_runs
from .c2_terminal_finalizer import finalize_to_path as finalize_c2_to_path
from .decision_report import (
    build_decision_report,
    write_decision_report,
)
from .harness import ExistingRunError, execute_experiment
from .matrix import MatrixError, load_and_expand_matrix
from .models import ProvenanceError
from .production_statistics import (
    StatisticsError,
    analyze_summary,
    build_holm_family,
    load_render_only_trajectory_bundle,
    load_provenance_summary,
    load_trajectory_threshold_config,
    write_analysis_outputs,
    write_holm_family_output,
)
from .provenance_stage import (
    build_c5_provenance_index,
    build_generator_identity_index,
    validate_c5_provenance_index,
    validate_generator_identity_index,
    write_provenance_index,
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
    analyze_parser.add_argument(
        "--trajectory-threshold-status",
        type=Path,
        default=None,
        help=(
            "Frozen C3 wall-clock blocker status required for a render-only "
            "trajectory threshold"
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

    decision_parser = subparsers.add_parser(
        "decision-report",
        help="Build a provenance-strict outcome-independent C1/C4 decision report",
    )
    decision_parser.add_argument("family", type=Path)
    decision_parser.add_argument("analyses", nargs="+", type=Path)
    decision_parser.add_argument("--out", type=Path, required=True)

    rejudge_parser = subparsers.add_parser(
        "rejudge",
        help="Read sealed best-candidate renders and run a post-hoc visual judge",
    )
    rejudge_parser.add_argument("source", type=Path)
    rejudge_parser.add_argument(
        "--judge-model",
        required=True,
        choices=("claude-sonnet-4.6", "gemini-3.5-flash"),
        help="Exact frozen C5 judge request identity",
    )
    rejudge_parser.add_argument("--out", type=Path, default=None)
    rejudge_parser.add_argument("--resume", action="store_true")

    merge_parser = subparsers.add_parser(
        "merge-rejudge",
        help="Create rejudged_summary.json without changing the source summary",
    )
    merge_parser.add_argument("summary", type=Path)
    merge_parser.add_argument("sidecar_dir", type=Path)
    merge_parser.add_argument("--out", type=Path, default=None)

    provenance_parser = subparsers.add_parser(
        "provenance-stage",
        help="Build a fail-closed final C1/C4 or C5 provenance index",
    )
    provenance_parser.add_argument("manifest", type=Path)
    provenance_parser.add_argument("--out", type=Path, required=True)

    c2_finalizer_parser = subparsers.add_parser(
        "c2-terminal-finalize",
        help="Build a terminal-only C2 report from sealed chunk reports",
    )
    c2_finalizer_parser.add_argument("manifest", type=Path)
    c2_finalizer_parser.add_argument("--out", type=Path, required=True)
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
    trajectory_threshold = None
    trajectory_threshold_provenance = None
    if args.trajectory_threshold_config is not None:
        if args.trajectory_threshold_status is not None:
            (
                trajectory_threshold,
                trajectory_threshold_provenance,
            ) = load_render_only_trajectory_bundle(
                args.trajectory_threshold_config,
                args.trajectory_threshold_status,
            )
        else:
            trajectory_threshold = load_trajectory_threshold_config(
                args.trajectory_threshold_config
            )
    elif args.trajectory_threshold_status is not None:
        raise StatisticsError(
            "--trajectory-threshold-status requires "
            "--trajectory-threshold-config"
        )
    analysis = analyze_summary(
        summary,
        reference=args.reference,
        methods=args.methods,
        metric=args.metric,
        panel_scope=args.panel_scope,
        second_judge_metric=args.second_judge_metric,
        trajectory_threshold=trajectory_threshold,
        trajectory_threshold_provenance=(
            trajectory_threshold_provenance
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


def _decision_report_command(args: argparse.Namespace) -> int:
    report = build_decision_report(args.analyses, args.family)
    path = write_decision_report(report, args.out)
    print(
        json.dumps(
            {
                "decision_report": str(path),
                "decision_report_hash": report["decision_report_hash"],
                "c1_status": report["c1"]["status"],
                "c4_status": report["c4"]["status"],
                "code_git_commit": report["decision_code_git_commit"],
                "code_git_dirty": report["decision_code_git_dirty"],
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
                "merged_judge_metric": metric,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _provenance_stage_command(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    kind = manifest.get("kind")
    if kind == "generator":
        reports = {
            tier: Path(path)
            for tier, path in manifest.get("completion_reports", {}).items()
        }
        summary_paths = {
            tier: Path(path)
            for tier, path in manifest.get("summary_paths", {}).items()
        }
        record_roots = {
            tier: Path(path)
            for tier, path in manifest.get("record_roots", {}).items()
        }
        record_indexes = {
            tier: Path(path)
            for tier, path in manifest.get("record_indexes", {}).items()
        }
        payload = build_generator_identity_index(
            reports,
            summary_paths=summary_paths or None,
            record_roots=record_roots or None,
            record_indexes=record_indexes or None,
        )
        validate_generator_identity_index(payload)
    elif kind == "c5":
        batch_paths = {
            key: None if path is None else Path(path)
            for key, path in manifest.get("batch_paths", {}).items()
        }
        tier_artifacts = {
            tier: {
                name: None if path is None else Path(path)
                for name, path in paths.items()
            }
            for tier, paths in manifest.get("tier_artifacts", {}).items()
        }
        payload = build_c5_provenance_index(
            Path(manifest["final_report"]),
            batch_paths=batch_paths,
            tier_artifacts=tier_artifacts,
        )
        validate_c5_provenance_index(payload)
    else:
        raise ProvenanceError("Provenance manifest kind must be generator or c5")
    path = write_provenance_index(payload, args.out)
    print(
        json.dumps(
            {
                "path": str(path),
                "status": payload["status"],
                "index_hash": payload["index_hash"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "COMPLETE" else 3


def _c2_terminal_finalize_command(args: argparse.Namespace) -> int:
    report, path = finalize_c2_to_path(args.manifest, args.out)
    print(
        json.dumps(
            {
                "final_universe_report": str(path),
                "final_report_hash": report["final_report_hash"],
                "status": report["status"],
                "claim_status": report["claim_status"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "ADMITTED" else 3


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
        if args.command == "decision-report":
            return _decision_report_command(args)
        if args.command == "rejudge":
            return _rejudge_command(args)
        if args.command == "merge-rejudge":
            return _merge_rejudge_command(args)
        if args.command == "provenance-stage":
            return _provenance_stage_command(args)
        if args.command == "c2-terminal-finalize":
            return _c2_terminal_finalize_command(args)
    except (AggregationError, MatrixError, ProvenanceError, StatisticsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
