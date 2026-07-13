from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .models import sha256_file, sha256_json, utc_now, write_json_atomic
from .production_statistics import (
    StatisticsError,
    _family_member_slice,
    _git_provenance,
    _load_analysis_artifact,
    build_holm_family,
)


DECISION_REPORT_VERSION = "1.0"
FIDELITY_METRIC = "metric.data_fidelity"
COHESION_METRIC = "metric.series_cohesion"
METRIC_CONTRACT = {
    FIDELITY_METRIC: {"panel_scope": "all", "margin": 0.02, "symbol": "F"},
    COHESION_METRIC: {
        "panel_scope": "multi_panel",
        "margin": 0.25,
        "symbol": "C",
    },
}
REFERENCE_METHOD = "flat_iterative"
COMPARISON_METHODS = ("best_of_n", "pheroviz_full")
RENDER_BUDGET = 6.0
ALPHA = 0.05
D90_EXPERIMENT_COMMIT = "d90d655b96bfb3b95bfdc37665692957969d8968"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")


class DecisionReportError(StatisticsError):
    """Raised when C1/C4 decisions cannot be provenance-safe."""


def _reject_constant(value: str) -> None:
    raise DecisionReportError(f"Non-finite JSON value is forbidden: {value}")


def _read_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
        )
    except OSError as exc:
        raise DecisionReportError(f"Cannot read {label} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DecisionReportError(f"Invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise DecisionReportError(f"{label} must be a JSON object")
    return value


def _required_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise DecisionReportError(f"{label} must be a SHA-256 hash")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionReportError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DecisionReportError(f"{label} must be finite")
    return result


def _load_family(path: Path) -> Dict[str, Any]:
    family = _read_object(path, "Holm family")
    family_hash = _required_hash(family.get("family_hash"), "family_hash")
    unhashed = dict(family)
    unhashed.pop("family_hash", None)
    if sha256_json(unhashed) != family_hash:
        raise DecisionReportError("Holm family failed its integrity hash")
    if family.get("code_git_dirty") is not False:
        raise DecisionReportError("Decision report refuses a dirty Holm family")
    family_commit = family.get("code_git_commit")
    if not isinstance(family_commit, str) or not _COMMIT_RE.fullmatch(
        family_commit
    ):
        raise DecisionReportError("Holm family code commit is invalid")
    manifest_raw = family.get("manifest_path")
    if not isinstance(manifest_raw, str) or not manifest_raw.strip():
        raise DecisionReportError("Holm family lacks manifest_path")
    manifest_path = Path(manifest_raw).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = (path.parent / manifest_path).resolve()
    recomputed = build_holm_family(manifest_path)
    stable_fields = (
        "family_version",
        "family_name",
        "adjustment",
        "scope",
        "manifest_path",
        "manifest_hash",
        "family_config",
        "family_config_hash",
        "input_summary_hashes",
        "summary_hash_by_backbone",
        "experiment_git_commits",
        "experiment_git_commit_by_backbone",
        "common_experiment_provenance",
        "analysis_code_git_commit",
        "members",
    )
    if any(family.get(name) != recomputed.get(name) for name in stable_fields):
        raise DecisionReportError(
            "Holm family does not match manifest/analysis recomputation"
        )
    return family


def _comparison(
    selected_slice: Mapping[str, Any],
    method: str,
) -> Mapping[str, Any]:
    matches = [
        value
        for value in selected_slice.get("comparisons", [])
        if isinstance(value, Mapping) and value.get("method") == method
    ]
    if len(matches) != 1:
        raise DecisionReportError(
            f"Analysis slice requires exactly one {method} comparison"
        )
    return matches[0]


def _effect_fields(
    comparison: Mapping[str, Any],
    label: str,
) -> Dict[str, Any]:
    overall = comparison.get("overall")
    permutation = comparison.get("permutation")
    if not isinstance(overall, Mapping) or not isinstance(permutation, Mapping):
        raise DecisionReportError(f"{label} lacks overall/permutation results")
    effect = _finite(overall.get("mean_gap"), f"{label}.mean_gap")
    ci = overall.get("ci95")
    if not isinstance(ci, list) or len(ci) != 2:
        raise DecisionReportError(f"{label}.ci95 must contain two values")
    ci_lower = _finite(ci[0], f"{label}.ci95[0]")
    ci_upper = _finite(ci[1], f"{label}.ci95[1]")
    if ci_lower > ci_upper:
        raise DecisionReportError(f"{label}.ci95 is reversed")
    raw_p = _finite(permutation.get("p_value"), f"{label}.p_value")
    if not 0.0 <= raw_p <= 1.0:
        raise DecisionReportError(f"{label}.p_value must be in [0,1]")
    return {
        "effect": effect,
        "ci95": [ci_lower, ci_upper],
        "ci_excludes_zero_positive": ci_lower > 0.0,
        "raw_p_value": raw_p,
    }


def build_decision_report(
    analysis_paths: Sequence[Path],
    family_path: Path,
) -> Dict[str, Any]:
    code_commit, code_dirty = _git_provenance()
    if code_dirty:
        raise DecisionReportError(
            "Decision report must be produced from a clean worktree"
        )
    if not _COMMIT_RE.fullmatch(code_commit):
        raise DecisionReportError("Decision code commit is invalid")

    resolved_paths = [path.expanduser().resolve() for path in analysis_paths]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise DecisionReportError("Duplicate analysis artifact path")
    if len(resolved_paths) != 6:
        raise DecisionReportError(
            "C1/C4 requires exactly six analysis artifacts"
        )
    analyses = {
        path: _load_analysis_artifact(path) for path in resolved_paths
    }
    family_path = family_path.expanduser().resolve()
    family = _load_family(family_path)
    manifest_path = Path(str(family["manifest_path"])).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = (family_path.parent / manifest_path).resolve()
    manifest = _read_object(manifest_path, "Holm family manifest")

    members = family.get("members")
    if not isinstance(members, list) or len(members) != 6:
        raise DecisionReportError("C4 Holm family must contain exactly six members")
    family_analysis_paths = {
        Path(str(member.get("analysis_path"))).expanduser().resolve()
        for member in members
        if isinstance(member, Mapping)
    }
    if family_analysis_paths != set(resolved_paths):
        raise DecisionReportError(
            "Provided analyses must exactly cover Holm family analysis artifacts"
        )

    summary_hashes = {
        _required_hash(analysis.get("input_summary_hash"), "input_summary_hash")
        for analysis in analyses.values()
    }
    analysis_commits = {
        str(analysis.get("code_git_commit")) for analysis in analyses.values()
    }
    if len(analysis_commits) != 1:
        raise DecisionReportError("Decision inputs mix analysis provenance")
    if len(summary_hashes) != 3:
        raise DecisionReportError(
            "C1/C4 requires exactly three tier summary hashes"
        )
    if sorted(summary_hashes) != family.get("input_summary_hashes"):
        raise DecisionReportError("Analysis and Holm family summary hashes differ")
    if next(iter(analysis_commits)) != family.get("analysis_code_git_commit"):
        raise DecisionReportError("Analysis and Holm family code commits differ")

    analysis_by_hash: Dict[str, tuple[Path, Dict[str, Any]]] = {}
    for path, analysis in analyses.items():
        digest = _required_hash(analysis.get("analysis_hash"), "analysis_hash")
        if digest in analysis_by_hash:
            raise DecisionReportError("Duplicate analysis_hash across artifacts")
        config = analysis.get("analysis_config")
        if not isinstance(config, Mapping):
            raise DecisionReportError("Analysis lacks analysis_config")
        metric = config.get("metric")
        contract = METRIC_CONTRACT.get(str(metric))
        if contract is None:
            raise DecisionReportError(f"Unexpected decision metric: {metric!r}")
        if (
            config.get("panel_scope") != contract["panel_scope"]
            or config.get("reference") != REFERENCE_METHOD
            or not isinstance(config.get("methods"), list)
            or len(config["methods"]) != len(COMPARISON_METHODS)
            or set(config["methods"]) != set(COMPARISON_METHODS)
        ):
            raise DecisionReportError(
                f"Analysis config violates the frozen {metric} contract"
            )
        slices = analysis.get("slices")
        if not isinstance(slices, list) or len(slices) != 1:
            raise DecisionReportError(
                "Each tier/metric analysis must contain exactly one slice"
            )
        analysis_by_hash[digest] = (path, analysis)
    if len(analysis_by_hash) != 6:
        raise DecisionReportError("Each Holm member requires a distinct analysis")

    c4_cells: list[Dict[str, Any]] = []
    cell_keys: set[tuple[str, str]] = set()
    backbones: set[str] = set()
    summary_hashes_by_tier: Dict[str, set[str]] = {}
    experiment_commits_by_tier: Dict[str, set[str]] = {}
    for raw_member in members:
        if not isinstance(raw_member, Mapping):
            raise DecisionReportError("Every Holm family member must be an object")
        metric = str(raw_member.get("metric"))
        contract = METRIC_CONTRACT.get(metric)
        if contract is None:
            raise DecisionReportError(f"Unexpected C4 metric: {metric!r}")
        if (
            raw_member.get("method") != "pheroviz_full"
            or raw_member.get("reference") != REFERENCE_METHOD
            or raw_member.get("panel_scope") != contract["panel_scope"]
        ):
            raise DecisionReportError("C4 family selector violates frozen methods/scope")
        analysis_hash = _required_hash(
            raw_member.get("analysis_hash"),
            f"{raw_member.get('name')}.analysis_hash",
        )
        if analysis_hash not in analysis_by_hash:
            raise DecisionReportError("Holm member references an unprovided analysis")
        _, analysis = analysis_by_hash[analysis_hash]
        selector = next(
            (
                value
                for value in manifest.get("members", [])
                if isinstance(value, Mapping)
                and value.get("name") == raw_member.get("name")
            ),
            None,
        )
        if not isinstance(selector, Mapping):
            raise DecisionReportError("Holm member selector is missing")
        selected_slice, comparison = _family_member_slice(analysis, selector)
        backbone = str(selected_slice.get("backbone"))
        key = (backbone, metric)
        if key in cell_keys:
            raise DecisionReportError(f"Duplicate C4 tier/metric cell: {key}")
        cell_keys.add(key)
        backbones.add(backbone)
        summary_hashes_by_tier.setdefault(backbone, set()).add(
            analysis["input_summary_hash"]
        )
        experiment_commit = str(selected_slice.get("experiment_git_commit"))
        if not _COMMIT_RE.fullmatch(experiment_commit):
            raise DecisionReportError("Tier experiment commit is invalid")
        experiment_commits_by_tier.setdefault(backbone, set()).add(
            experiment_commit
        )
        if (
            selected_slice.get("budget_type") != "renders"
            or _finite(selected_slice.get("budget_value"), "budget_value")
            != RENDER_BUDGET
        ):
            raise DecisionReportError("C1/C4 decision requires render budget B_R=6")
        fields = _effect_fields(comparison, str(raw_member.get("name")))
        if fields["raw_p_value"] != _finite(
            raw_member.get("raw_p_value"),
            f"{raw_member.get('name')}.raw_p_value",
        ):
            raise DecisionReportError("Holm raw p-value disagrees with analysis")
        adjusted = _finite(
            raw_member.get("adjusted_p_value"),
            f"{raw_member.get('name')}.adjusted_p_value",
        )
        passes_margin = fields["effect"] > float(contract["margin"])
        passes_holm = adjusted < ALPHA
        passed = (
            passes_margin
            and fields["ci_excludes_zero_positive"]
            and passes_holm
        )
        c4_cells.append(
            {
                "backbone": backbone,
                "metric": metric,
                "metric_symbol": contract["symbol"],
                "panel_scope": contract["panel_scope"],
                "method": "pheroviz_full",
                "reference": REFERENCE_METHOD,
                "budget_type": "renders",
                "budget_value": RENDER_BUDGET,
                "practical_margin": contract["margin"],
                **fields,
                "holm_adjusted_p_value": adjusted,
                "passes_practical_margin": passes_margin,
                "passes_positive_ci": fields["ci_excludes_zero_positive"],
                "passes_cross_family_holm": passes_holm,
                "status": "pass" if passed else "fail",
            }
        )

    expected_cells = {
        (backbone, metric)
        for backbone in backbones
        for metric in METRIC_CONTRACT
    }
    if len(backbones) != 3 or cell_keys != expected_cells:
        raise DecisionReportError(
            "C4 requires exactly three backbones crossed with F and C"
        )
    if any(len(values) != 1 for values in summary_hashes_by_tier.values()):
        raise DecisionReportError(
            "F and C analyses within each tier must share one summary hash"
        )
    tier_summary_hashes = {
        tier: next(iter(summary_hashes_by_tier[tier]))
        for tier in sorted(backbones)
    }
    if len(set(tier_summary_hashes.values())) != 3:
        raise DecisionReportError(
            "Each of the three tiers must bind a distinct summary hash"
        )
    if tier_summary_hashes != family.get("summary_hash_by_backbone"):
        raise DecisionReportError("Tier summary map disagrees with Holm family")
    if any(len(values) != 1 for values in experiment_commits_by_tier.values()):
        raise DecisionReportError(
            "F and C analyses within each tier mix experiment commits"
        )
    tier_experiment_commits = {
        tier: next(iter(experiment_commits_by_tier[tier]))
        for tier in sorted(backbones)
    }
    experiment_commits = sorted(set(tier_experiment_commits.values()))
    if experiment_commits != [D90_EXPERIMENT_COMMIT]:
        raise DecisionReportError(
            "All tiers must use the frozen d90d655 experiment commit"
        )
    if (
        family.get("experiment_git_commits") != experiment_commits
        or family.get("experiment_git_commit_by_backbone")
        != tier_experiment_commits
    ):
        raise DecisionReportError(
            "Tier experiment commit map disagrees with Holm family"
        )

    c1_rows: list[Dict[str, Any]] = []
    c1_keys: set[tuple[str, str, str]] = set()
    for analysis in analyses.values():
        config = analysis["analysis_config"]
        metric = str(config["metric"])
        contract = METRIC_CONTRACT[metric]
        for selected_slice in analysis["slices"]:
            backbone = str(selected_slice.get("backbone"))
            if backbone not in backbones:
                raise DecisionReportError("C1/C4 analyses use different backbones")
            if (
                selected_slice.get("budget_type") != "renders"
                or _finite(selected_slice.get("budget_value"), "budget_value")
                != RENDER_BUDGET
            ):
                raise DecisionReportError("C1 report requires render budget B_R=6")
            for method in COMPARISON_METHODS:
                key = (backbone, metric, method)
                if key in c1_keys:
                    raise DecisionReportError(f"Duplicate C1 comparison: {key}")
                c1_keys.add(key)
                fields = _effect_fields(
                    _comparison(selected_slice, method),
                    f"C1.{backbone}.{metric}.{method}",
                )
                c1_rows.append(
                    {
                        "backbone": backbone,
                        "metric": metric,
                        "metric_symbol": contract["symbol"],
                        "panel_scope": contract["panel_scope"],
                        "method": method,
                        "reference": REFERENCE_METHOD,
                        "budget_type": "renders",
                        "budget_value": RENDER_BUDGET,
                        **fields,
                        "status": "reported",
                    }
                )
    expected_c1 = {
        (backbone, metric, method)
        for backbone in backbones
        for metric in METRIC_CONTRACT
        for method in COMPARISON_METHODS
    }
    if c1_keys != expected_c1:
        raise DecisionReportError("C1 comparisons lack exact tier/metric/method coverage")

    tier_decisions = []
    for backbone in sorted(backbones):
        cells = [value for value in c4_cells if value["backbone"] == backbone]
        tier_pass = len(cells) == 2 and all(value["status"] == "pass" for value in cells)
        tier_decisions.append(
            {
                "backbone": backbone,
                "status": "pass" if tier_pass else "report_effects_only",
                "both_metrics_pass": tier_pass,
            }
        )
    all_pass = all(value["status"] == "pass" for value in c4_cells)

    common = family.get("common_experiment_provenance")
    if not isinstance(common, Mapping):
        raise DecisionReportError("Holm family lacks common experiment provenance")
    for name in (
        "dataset_manifest_hash",
        "metric_config_hash",
    ):
        _required_hash(common.get(name), f"common_experiment_provenance.{name}")
    experiment_commit = str(common.get("experiment_git_commit"))
    if experiment_commit != D90_EXPERIMENT_COMMIT:
        raise DecisionReportError(
            "Common provenance is not the frozen d90d655 experiment commit"
        )
    if (
        common.get("budget_type") != "renders"
        or _finite(common.get("budget_value"), "common budget_value")
        != RENDER_BUDGET
    ):
        raise DecisionReportError("Common experiment provenance is not B_R=6")
    metric_version = common.get("metric_version")
    if not isinstance(metric_version, str) or not metric_version:
        raise DecisionReportError("Common experiment metric_version is invalid")

    report = {
        "decision_report_version": DECISION_REPORT_VERSION,
        "generated_at": utc_now(),
        "decision_code_git_commit": code_commit,
        "decision_code_git_dirty": False,
        "frozen_policy": {
            "delta_F": 0.02,
            "delta_C": 0.25,
            "alpha": ALPHA,
            "effect_comparison": "strictly_greater_than_margin",
            "ci_rule": "lower_bound_strictly_greater_than_zero",
            "multiple_comparisons": "six_member_cross_analysis_holm",
            "render_budget": RENDER_BUDGET,
            "experiment_git_commit": D90_EXPERIMENT_COMMIT,
        },
        "provenance": {
            "input_summary_hashes": sorted(summary_hashes),
            "tier_summary_hashes": tier_summary_hashes,
            "experiment_git_commits": experiment_commits,
            "tier_experiment_git_commits": tier_experiment_commits,
            "analysis_code_git_commit": next(iter(analysis_commits)),
            "holm_family_code_git_commit": family["code_git_commit"],
            "dataset_manifest_hash": common["dataset_manifest_hash"],
            "metric_config_hash": common["metric_config_hash"],
            "metric_version": common.get("metric_version"),
            "budget_type": common.get("budget_type"),
            "budget_value": common.get("budget_value"),
            "split": common.get("split"),
            "analysis_artifacts": [
                {
                    "path": str(path),
                    "file_sha256": sha256_file(path),
                    "analysis_hash": analysis["analysis_hash"],
                    "analysis_config_hash": analysis["analysis_config_hash"],
                    "input_summary_hash": analysis["input_summary_hash"],
                    "code_git_commit": analysis["code_git_commit"],
                    "metric": analysis["analysis_config"]["metric"],
                    "backbone": analysis["slices"][0]["backbone"],
                    "panel_scope": analysis["analysis_config"]["panel_scope"],
                    "reference": analysis["analysis_config"]["reference"],
                    "methods": analysis["analysis_config"]["methods"],
                }
                for path, analysis in sorted(
                    analyses.items(),
                    key=lambda item: str(item[0]),
                )
            ],
            "holm_family": {
                "path": str(family_path),
                "file_sha256": sha256_file(family_path),
                "family_hash": family["family_hash"],
                "family_config_hash": family["family_config_hash"],
                "manifest_hash": family["manifest_hash"],
            },
        },
        "c1": {
            "status": "render_budget_only",
            "claim_scope": (
                "Programmatic render-budget comparisons; visual-form judges "
                "and wall-clock performance are not correctness substitutes."
            ),
            "comparisons": sorted(
                c1_rows,
                key=lambda value: (
                    value["backbone"],
                    value["metric"],
                    value["method"],
                ),
            ),
            "wall_clock": {
                "status": "unavailable",
                "claim_permitted": False,
                "reason": (
                    "No frozen equal-hardware, equal-concurrency wall-clock "
                    "analysis is supplied to this render-budget report."
                ),
            },
        },
        "c4": {
            "status": "pass" if all_pass else "tier_scoped_fallback",
            "cross_tier_claim_permitted": all_pass,
            "required_cell_count": 6,
            "cells": sorted(
                c4_cells,
                key=lambda value: (value["backbone"], value["metric"]),
            ),
            "tier_decisions": tier_decisions,
            "fallback_policy": (
                None
                if all_pass
                else (
                    "Report tier-specific effects and failed criteria; do not "
                    "make the all-tier C4 claim."
                )
            ),
        },
    }
    report["decision_report_hash"] = sha256_json(report)
    return report


def write_decision_report(report: Mapping[str, Any], out: Path) -> Path:
    path = out.expanduser().resolve()
    if path.suffix.lower() != ".json":
        path = path / "c1_c4_decision_report.json"
    write_json_atomic(path, report)
    return path
