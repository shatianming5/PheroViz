from __future__ import annotations

import csv
import io
import json
import math
import os
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .aggregate import AggregationError, assert_paired_ready
from .models import (
    SCHEMA_VERSION,
    ProvenanceError,
    canonical_json,
    sha256_json,
    utc_now,
    write_json_atomic,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PANEL_STRATA = ("P=1", "P=2", "P=3-4", "P=5+")


class StatisticsError(ProvenanceError):
    """Raised when a production analysis cannot be validated."""


@dataclass(frozen=True)
class ValidatedSummary:
    path: Path
    summary_hash: str
    rows: tuple[Dict[str, Any], ...]


def _reject_json_constant(value: str) -> None:
    raise StatisticsError(f"Non-finite JSON value is forbidden: {value}")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StatisticsError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise StatisticsError(f"{name} must be finite")
    return number


def _required_string(row: Mapping[str, Any], name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value.strip():
        raise StatisticsError(f"Summary row requires non-empty {name}")
    return value


def load_provenance_summary(path: Path) -> ValidatedSummary:
    resolved = path.expanduser().resolve()
    try:
        data = json.loads(
            resolved.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except OSError as exc:
        raise StatisticsError(f"Cannot read summary {resolved}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise StatisticsError(f"Invalid summary JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise StatisticsError("Summary must be a JSON object")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise StatisticsError(
            f"Unsupported summary schema_version: {data.get('schema_version')!r}"
        )

    recorded_hash = data.get("summary_hash")
    if not isinstance(recorded_hash, str) or not _SHA256_RE.fullmatch(
        recorded_hash
    ):
        raise StatisticsError("Summary has no valid summary_hash")
    unhashed = dict(data)
    unhashed.pop("summary_hash", None)
    if sha256_json(unhashed) != recorded_hash:
        raise StatisticsError("Summary failed its integrity hash")

    rows = data.get("runs")
    if not isinstance(rows, list) or not rows:
        raise StatisticsError("Summary contains no run rows")
    if data.get("run_count") != len(rows):
        raise StatisticsError("summary.run_count does not match runs")
    record_hashes = data.get("input_record_hashes")
    if not isinstance(record_hashes, Mapping):
        raise StatisticsError("Summary lacks input_record_hashes")

    validated: list[Dict[str, Any]] = []
    run_names: set[str] = set()
    statistical_keys: set[tuple[Any, ...]] = set()
    for raw_row in rows:
        if not isinstance(raw_row, dict):
            raise StatisticsError("Every summary run must be an object")
        row = dict(raw_row)
        run_name = _required_string(row, "run_name")
        if run_name in run_names:
            raise StatisticsError(f"Duplicate run_name in summary: {run_name}")
        run_names.add(run_name)
        if row.get("status") != "completed":
            raise StatisticsError(f"Non-completed summary row: {run_name}")
        if row.get("test_only") is not False:
            raise StatisticsError(
                f"test_only or unlabelled row is forbidden: {run_name}"
            )

        method = _required_string(row, "method")
        backbone = _required_string(row, "backbone")
        case_id = _required_string(row, "case_id")
        metric_version = _required_string(row, "metric_version")
        budget_type = _required_string(row, "budget_type")
        if budget_type not in {"renders", "wall_clock_seconds"}:
            raise StatisticsError(f"Invalid budget_type in {run_name}")
        budget_value = _finite_number(
            row.get("budget_value"),
            f"{run_name}.budget_value",
        )
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise StatisticsError(f"{run_name}.seed must be an integer")
        panel_count = row.get("panel_count")
        if (
            isinstance(panel_count, bool)
            or not isinstance(panel_count, int)
            or panel_count < 1
        ):
            raise StatisticsError(
                f"{run_name}.panel_count must be a positive integer"
            )
        split = row.get("split")
        if split is not None and (
            not isinstance(split, str) or not split.strip()
        ):
            raise StatisticsError(f"{run_name}.split is invalid")

        for hash_name in (
            "dataset_manifest_hash",
            "metric_config_hash",
            "record_hash",
            "spec_hash",
        ):
            digest = row.get(hash_name)
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise StatisticsError(
                    f"{run_name}.{hash_name} must be SHA-256"
                )
        if record_hashes.get(run_name) != row["record_hash"]:
            raise StatisticsError(
                f"input_record_hashes disagrees for {run_name}"
            )

        key = (
            method,
            backbone,
            seed,
            budget_type,
            budget_value,
            row["dataset_manifest_hash"],
            row["metric_config_hash"],
            metric_version,
            split,
            case_id,
        )
        if key in statistical_keys:
            raise StatisticsError(
                f"Duplicate case within a method slice: {method}/{case_id}"
            )
        statistical_keys.add(key)
        validated.append(row)

    if set(record_hashes) != run_names:
        raise StatisticsError(
            "input_record_hashes does not exactly cover summary rows"
        )
    return ValidatedSummary(
        path=resolved,
        summary_hash=recorded_hash,
        rows=tuple(validated),
    )


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise StatisticsError("Cannot compute a percentile of no values")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def paired_bootstrap(
    gaps: Sequence[float],
    *,
    seed: int,
    resamples: int = 10_000,
) -> Dict[str, Any]:
    values = [
        _finite_number(value, f"gap[{index}]")
        for index, value in enumerate(gaps)
    ]
    if not values:
        raise StatisticsError("Paired bootstrap requires at least one case")
    if resamples < 1:
        raise StatisticsError("bootstrap resamples must be positive")
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        total = sum(values[rng.randrange(n)] for _ in range(n))
        means.append(total / n)
    means.sort()
    return {
        "n": n,
        "mean_gap": sum(values) / n,
        "ci95": [
            _percentile(means, 0.025),
            _percentile(means, 0.975),
        ],
        "resamples": resamples,
        "seed": seed,
        "sampling_unit": "case_id",
    }


def paired_sign_flip_permutation(
    gaps: Sequence[float],
    *,
    seed: int,
    monte_carlo_permutations: int = 100_000,
    exact_max_n: int = 16,
) -> Dict[str, Any]:
    values = [
        _finite_number(value, f"gap[{index}]")
        for index, value in enumerate(gaps)
    ]
    n = len(values)
    if n < 1:
        raise StatisticsError("Sign-flip test requires at least one case")
    observed = abs(sum(values) / n)
    epsilon = 1e-15
    if n <= exact_max_n:
        permutations = 1 << n
        extreme = 0
        for mask in range(permutations):
            signed_sum = sum(
                value if mask & (1 << index) else -value
                for index, value in enumerate(values)
            )
            if abs(signed_sum / n) + epsilon >= observed:
                extreme += 1
        p_value = extreme / permutations
        mode = "exact"
    else:
        if monte_carlo_permutations < 1:
            raise StatisticsError(
                "Monte Carlo permutation count must be positive"
            )
        rng = random.Random(seed)
        permutations = monte_carlo_permutations
        extreme = 0
        for _ in range(permutations):
            signed_sum = sum(
                value if rng.getrandbits(1) else -value
                for value in values
            )
            if abs(signed_sum / n) + epsilon >= observed:
                extreme += 1
        p_value = (extreme + 1) / (permutations + 1)
        mode = "monte_carlo"
    return {
        "n": n,
        "statistic": observed,
        "alternative": "two-sided",
        "p_value": p_value,
        "mode": mode,
        "permutations": permutations,
        "seed": seed,
        "sampling_unit": "case_id",
    }


def panel_stratum(panel_count: int) -> str:
    if panel_count == 1:
        return "P=1"
    if panel_count == 2:
        return "P=2"
    if panel_count <= 4:
        return "P=3-4"
    return "P=5+"


def stratified_bootstrap(
    case_gaps: Mapping[str, float],
    panel_counts: Mapping[str, int],
    *,
    seed: int,
    resamples: int,
) -> Dict[str, Any]:
    grouped: Dict[str, list[float]] = {
        stratum: [] for stratum in _PANEL_STRATA
    }
    for case_id, gap in case_gaps.items():
        if case_id not in panel_counts:
            raise StatisticsError(
                f"Missing panel_count for paired case {case_id!r}"
            )
        grouped[panel_stratum(panel_counts[case_id])].append(gap)
    result: Dict[str, Any] = {}
    for stratum in _PANEL_STRATA:
        values = grouped[stratum]
        if not values:
            result[stratum] = {
                "status": "NA",
                "n": 0,
                "mean_gap": None,
                "ci95": None,
                "resamples": resamples,
                "sampling_unit": "case_id",
            }
            continue
        result[stratum] = {
            "status": "ok",
            **paired_bootstrap(
                values,
                seed=seed,
                resamples=resamples,
            ),
        }
    return result


def kendall_tau_b(
    first: Sequence[float],
    second: Sequence[float],
) -> Optional[float]:
    if len(first) != len(second):
        raise StatisticsError("Kendall vectors must have equal length")
    if len(first) < 2:
        raise StatisticsError("Kendall tau-b requires at least two methods")
    concordant = 0
    discordant = 0
    ties_first = 0
    ties_second = 0
    for left in range(len(first)):
        for right in range(left + 1, len(first)):
            delta_first = first[left] - first[right]
            delta_second = second[left] - second[right]
            if delta_first == 0 and delta_second == 0:
                continue
            if delta_first == 0:
                ties_first += 1
            elif delta_second == 0:
                ties_second += 1
            elif delta_first * delta_second > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_first)
        * (concordant + discordant + ties_second)
    )
    if denominator == 0:
        return None
    return (concordant - discordant) / denominator


def method_ranking(scores: Mapping[str, float]) -> list[Dict[str, Any]]:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    output: list[Dict[str, Any]] = []
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        for method, score in ordered[index:end]:
            output.append(
                {
                    "method": method,
                    "mean": score,
                    "rank": average_rank,
                }
            )
        index = end
    return output


def _metric_value(row: Mapping[str, Any], metric: str) -> float:
    if metric not in row:
        raise StatisticsError(
            f"Metric {metric!r} missing from run {row.get('run_name')!r}"
        )
    return _finite_number(
        row[metric],
        f"{row.get('run_name')}.{metric}",
    )


def _slice_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["backbone"],
        row["seed"],
        row["budget_type"],
        float(row["budget_value"]),
        row.get("split"),
    )


def _git_provenance() -> tuple[str, bool]:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StatisticsError(f"Cannot record analysis git commit: {exc}") from exc
    return commit, dirty


def _ranking_analysis(
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    *,
    primary_metric: str,
    second_judge_metric: Optional[str],
) -> Dict[str, Any]:
    primary_scores: Dict[str, float] = {}
    second_scores: Dict[str, float] = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        primary_values = [
            _metric_value(row, primary_metric) for row in method_rows
        ]
        primary_scores[method] = sum(primary_values) / len(primary_values)
        if second_judge_metric is not None:
            second_values = [
                _metric_value(row, second_judge_metric) for row in method_rows
            ]
            second_scores[method] = sum(second_values) / len(second_values)
    result: Dict[str, Any] = {
        "primary_metric": primary_metric,
        "ranking_direction": "higher_is_better",
        "primary_ranking": method_ranking(primary_scores),
    }
    if second_judge_metric is None:
        result.update(
            {
                "second_judge_metric": None,
                "second_judge_ranking": None,
                "kendall_tau_b": None,
                "status": "not_requested",
            }
        )
        return result
    result.update(
        {
            "second_judge_metric": second_judge_metric,
            "second_judge_ranking": method_ranking(second_scores),
            "kendall_tau_b": kendall_tau_b(
                [primary_scores[method] for method in methods],
                [second_scores[method] for method in methods],
            ),
            "status": "ok",
        }
    )
    return result


def analyze_summary(
    summary: ValidatedSummary,
    *,
    reference: str,
    methods: Sequence[str],
    metric: str,
    second_judge_metric: Optional[str] = None,
    seed: int = 17_029,
    bootstrap_resamples: int = 10_000,
    monte_carlo_permutations: int = 100_000,
    exact_max_n: int = 16,
) -> Dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise StatisticsError("Analysis seed must be an integer")
    if bootstrap_resamples < 1:
        raise StatisticsError("bootstrap resamples must be positive")
    if monte_carlo_permutations < 1:
        raise StatisticsError("permutation count must be positive")
    if exact_max_n < 1:
        raise StatisticsError("exact_max_n must be positive")
    comparison_methods = list(methods)
    if not comparison_methods:
        raise StatisticsError("At least one comparison method is required")
    if reference in comparison_methods:
        raise StatisticsError("Reference must not also appear in --methods")
    if len(comparison_methods) != len(set(comparison_methods)):
        raise StatisticsError("Comparison methods must be unique")
    selected_methods = [reference, *comparison_methods]
    selected_rows = [
        row for row in summary.rows if row["method"] in selected_methods
    ]
    if not selected_rows:
        raise StatisticsError("No rows match the requested methods")
    available_methods = {row["method"] for row in selected_rows}
    missing_methods = set(selected_methods) - available_methods
    if missing_methods:
        raise StatisticsError(
            f"Requested methods are absent: {sorted(missing_methods)}"
        )

    by_slice: Dict[tuple[Any, ...], list[Dict[str, Any]]] = {}
    for row in selected_rows:
        by_slice.setdefault(_slice_key(row), []).append(row)

    slice_results = []
    for key in sorted(by_slice, key=lambda item: canonical_json(item)):
        backbone, slice_seed, budget_type, budget_value, split = key
        rows = by_slice[key]
        try:
            case_ids = assert_paired_ready(
                rows,
                methods=selected_methods,
                backbone=backbone,
                seed=slice_seed,
                budget_type=budget_type,
                budget_value=budget_value,
            )
        except AggregationError as exc:
            raise StatisticsError(str(exc)) from exc
        if len(case_ids) < 2:
            raise StatisticsError(
                f"Slice {key!r} has fewer than two paired cases"
            )
        manifest_hashes = {row["dataset_manifest_hash"] for row in rows}
        config_hashes = {row["metric_config_hash"] for row in rows}
        metric_versions = {row["metric_version"] for row in rows}
        if (
            len(manifest_hashes) != 1
            or len(config_hashes) != 1
            or len(metric_versions) != 1
        ):
            raise StatisticsError(
                f"Slice {key!r} mixes manifest or metric provenance"
            )

        by_method_case = {
            method: {
                row["case_id"]: row
                for row in rows
                if row["method"] == method
            }
            for method in selected_methods
        }
        reference_rows = by_method_case[reference]
        panel_counts = {
            case_id: int(reference_rows[case_id]["panel_count"])
            for case_id in case_ids
        }
        comparisons = []
        for method in comparison_methods:
            case_gaps = {
                case_id: (
                    _metric_value(by_method_case[method][case_id], metric)
                    - _metric_value(reference_rows[case_id], metric)
                )
                for case_id in sorted(case_ids)
            }
            gaps = list(case_gaps.values())
            comparisons.append(
                {
                    "method": method,
                    "gap_definition": f"{method} - {reference}",
                    "overall": paired_bootstrap(
                        gaps,
                        seed=seed,
                        resamples=bootstrap_resamples,
                    ),
                    "permutation": paired_sign_flip_permutation(
                        gaps,
                        seed=seed,
                        monte_carlo_permutations=monte_carlo_permutations,
                        exact_max_n=exact_max_n,
                    ),
                    "panel_strata": stratified_bootstrap(
                        case_gaps,
                        panel_counts,
                        seed=seed,
                        resamples=bootstrap_resamples,
                    ),
                }
            )

        slice_results.append(
            {
                "backbone": backbone,
                "seed": slice_seed,
                "budget_type": budget_type,
                "budget_value": budget_value,
                "split": split,
                "dataset_manifest_hash": next(iter(manifest_hashes)),
                "metric_config_hash": next(iter(config_hashes)),
                "metric_version": next(iter(metric_versions)),
                "reference": reference,
                "case_count": len(case_ids),
                "case_ids": sorted(case_ids),
                "comparisons": comparisons,
                "rankings": _ranking_analysis(
                    rows,
                    selected_methods,
                    primary_metric=metric,
                    second_judge_metric=second_judge_metric,
                ),
            }
        )

    config = {
        "reference": reference,
        "methods": comparison_methods,
        "metric": metric,
        "second_judge_metric": second_judge_metric,
        "seed": seed,
        "bootstrap_resamples": bootstrap_resamples,
        "monte_carlo_permutations": monte_carlo_permutations,
        "exact_max_n": exact_max_n,
        "sampling_unit": "case_id",
        "panel_strata": list(_PANEL_STRATA),
    }
    commit, dirty = _git_provenance()
    output = {
        "analysis_version": "1.0",
        "generated_at": utc_now(),
        "input_summary_path": str(summary.path),
        "input_summary_hash": summary.summary_hash,
        "analysis_config": config,
        "analysis_config_hash": sha256_json(config),
        "code_git_commit": commit,
        "code_git_dirty": dirty,
        "slices": slice_results,
    }
    output["analysis_hash"] = sha256_json(output)
    return output


def _analysis_csv_rows(analysis: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for slice_result in analysis["slices"]:
        ranking = slice_result["rankings"]
        for comparison in slice_result["comparisons"]:
            overall = comparison["overall"]
            permutation = comparison["permutation"]
            base = {
                "backbone": slice_result["backbone"],
                "seed": slice_result["seed"],
                "budget_type": slice_result["budget_type"],
                "budget_value": slice_result["budget_value"],
                "split": slice_result["split"],
                "dataset_manifest_hash": slice_result[
                    "dataset_manifest_hash"
                ],
                "metric_config_hash": slice_result["metric_config_hash"],
                "metric_version": slice_result["metric_version"],
                "reference": slice_result["reference"],
                "method": comparison["method"],
                "metric": analysis["analysis_config"]["metric"],
                "second_judge_metric": ranking["second_judge_metric"],
                "kendall_tau_b": ranking["kendall_tau_b"],
            }
            rows.append(
                {
                    **base,
                    "scope": "overall",
                    "status": "ok",
                    "n": overall["n"],
                    "mean_gap": overall["mean_gap"],
                    "ci_lower": overall["ci95"][0],
                    "ci_upper": overall["ci95"][1],
                    "bootstrap_resamples": overall["resamples"],
                    "permutation_mode": permutation["mode"],
                    "permutations": permutation["permutations"],
                    "p_value": permutation["p_value"],
                }
            )
            for stratum in _PANEL_STRATA:
                stats = comparison["panel_strata"][stratum]
                ci = stats["ci95"]
                rows.append(
                    {
                        **base,
                        "scope": stratum,
                        "status": stats["status"],
                        "n": stats["n"],
                        "mean_gap": stats["mean_gap"],
                        "ci_lower": ci[0] if ci is not None else None,
                        "ci_upper": ci[1] if ci is not None else None,
                        "bootstrap_resamples": stats["resamples"],
                        "permutation_mode": None,
                        "permutations": None,
                        "p_value": None,
                    }
                )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise StatisticsError("Analysis produced no CSV rows")
    columns = list(rows[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(stream.getvalue())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_analysis_outputs(
    analysis: Mapping[str, Any],
    out: Path,
) -> tuple[Path, Path]:
    out = out.expanduser().resolve()
    if out.suffix.lower() == ".json":
        json_path = out
        csv_path = out.with_suffix(".csv")
    else:
        json_path = out / "analysis.json"
        csv_path = out / "analysis.csv"
    write_json_atomic(json_path, analysis)
    _write_csv(csv_path, _analysis_csv_rows(analysis))
    return json_path, csv_path
