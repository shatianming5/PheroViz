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

from .aggregate import (
    AggregationError,
    _verified_frozen_manifest_case,
    assert_paired_ready,
)
from .models import (
    SCHEMA_VERSION,
    ProvenanceError,
    RunRecord,
    canonical_json,
    sha256_json,
    utc_now,
    verify_artifacts,
    write_json_atomic,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PANEL_STRATA = ("P=1", "P=2", "P=3-4", "P=5+")
_PANEL_STRATUM_SCORES = {
    stratum: float(index)
    for index, stratum in enumerate(_PANEL_STRATA, 1)
}
_ANALYSIS_VERSION = "3.2"
_SUPPORTED_ANALYSIS_VERSIONS = {"3.1", "3.2"}
_HOLM_FAMILY_VERSION = "1.0"
_PANEL_SCOPES = {"all", "single_panel", "multi_panel"}
_C5_POINT_THRESHOLD = 0.5
_C5_CI_LOWER_THRESHOLD = 0.0
_C5_METRICS = {
    "metric.visual_form.claude-sonnet-4.6": "visual-form-primary-v1",
    "metric.visual_form.gemini-3.5-flash": "visual-form-secondary-v1",
}


class StatisticsError(ProvenanceError):
    """Raised when a production analysis cannot be validated."""


@dataclass(frozen=True)
class ValidatedSummary:
    path: Path
    summary_hash: str
    source_root: Optional[Path]
    rows: tuple[Dict[str, Any], ...]
    payload: Dict[str, Any]


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


def _normalized_doi(row: Mapping[str, Any]) -> Optional[str]:
    raw = row.get("doi")
    if raw in {"", None}:
        return None
    if not isinstance(raw, str):
        raise StatisticsError("Summary row DOI must be a string when present")
    doi = raw.strip().casefold()
    if (
        doi != raw
        or not doi.startswith("10.")
        or "/" not in doi
        or any(character.isspace() for character in doi)
    ):
        raise StatisticsError(
            f"Summary row has an invalid normalized DOI: {row.get('run_name')!r}"
        )
    return doi


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
    source_root_raw = data.get("source_root")
    source_root: Optional[Path] = None
    if source_root_raw is not None:
        if not isinstance(source_root_raw, str) or not source_root_raw.strip():
            raise StatisticsError("summary.source_root must be a non-empty path")
        source_root = Path(source_root_raw).expanduser()
        if not source_root.is_absolute():
            source_root = resolved.parent / source_root
        source_root = source_root.resolve()

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
        status = row.get("status")
        if status not in {"completed", "failed"}:
            raise StatisticsError(f"Non-terminal summary row: {run_name}")
        if row.get("test_only") is not False:
            raise StatisticsError(
                f"test_only or unlabelled row is forbidden: {run_name}"
            )
        execution_success = _finite_number(
            row.get("execution_success"),
            f"{run_name}.execution_success",
        )
        failure_attribution = row.get("failure_attribution")
        if status == "completed":
            if execution_success != 1.0 or failure_attribution not in {"", None}:
                raise StatisticsError(
                    f"Completed row has inconsistent execution status: {run_name}"
                )
        elif execution_success != 0.0 or failure_attribution != "method":
            raise StatisticsError(
                "Failed rows require execution_success=0 and explicit "
                f"method attribution: {run_name}"
            )

        method = _required_string(row, "method")
        backbone = _required_string(row, "backbone")
        case_id = _required_string(row, "case_id")
        git_commit = _required_string(row, "git_commit")
        if not re.fullmatch(r"[0-9a-f]{7,64}", git_commit):
            raise StatisticsError(f"{run_name}.git_commit is invalid")
        doi = _normalized_doi(row)
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
        metric_execution_success = _finite_number(
            row.get("metric.execution_success"),
            f"{run_name}.metric.execution_success",
        )
        if metric_execution_success != execution_success:
            raise StatisticsError(
                f"Execution-success metric disagrees with status: {run_name}"
            )
        for name, value in row.items():
            if name.startswith("metric."):
                _finite_number(value, f"{run_name}.{name}")

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
        row["doi"] = doi or ""
        validated.append(row)

    if set(record_hashes) != run_names:
        raise StatisticsError(
            "input_record_hashes does not exactly cover summary rows"
        )
    return ValidatedSummary(
        path=resolved,
        summary_hash=recorded_hash,
        source_root=source_root,
        rows=tuple(validated),
        payload=dict(data),
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


def holm_adjust(p_values: Mapping[str, float]) -> Dict[str, float]:
    """Return Holm step-down adjusted p-values keyed by comparison name."""

    validated = {
        name: _finite_number(value, f"p_value[{name}]")
        for name, value in p_values.items()
    }
    if any(not 0.0 <= value <= 1.0 for value in validated.values()):
        raise StatisticsError("Holm adjustment requires p-values in [0,1]")
    ordered = sorted(validated.items(), key=lambda item: (item[1], item[0]))
    adjusted: Dict[str, float] = {}
    running_max = 0.0
    family_size = len(ordered)
    for index, (name, value) in enumerate(ordered):
        candidate = min(1.0, (family_size - index) * value)
        running_max = max(running_max, candidate)
        adjusted[name] = running_max
    return adjusted


def paired_bootstrap(
    gaps: Sequence[float],
    *,
    seed: int,
    resamples: int = 10_000,
    sampling_unit: str = "case_id",
) -> Dict[str, Any]:
    values = [
        _finite_number(value, f"gap[{index}]")
        for index, value in enumerate(gaps)
    ]
    if not values:
        raise StatisticsError("Paired bootstrap requires at least one case")
    if resamples < 1:
        raise StatisticsError("bootstrap resamples must be positive")
    if not sampling_unit:
        raise StatisticsError("sampling_unit must be non-empty")
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
        "sampling_unit": sampling_unit,
    }


def paired_sign_flip_permutation(
    gaps: Sequence[float],
    *,
    seed: int,
    monte_carlo_permutations: int = 100_000,
    exact_max_n: int = 16,
    sampling_unit: str = "case_id",
) -> Dict[str, Any]:
    values = [
        _finite_number(value, f"gap[{index}]")
        for index, value in enumerate(gaps)
    ]
    n = len(values)
    if n < 1:
        raise StatisticsError("Sign-flip test requires at least one case")
    if not sampling_unit:
        raise StatisticsError("sampling_unit must be non-empty")
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
        "sampling_unit": sampling_unit,
    }


def panel_stratum(panel_count: int) -> str:
    if panel_count == 1:
        return "P=1"
    if panel_count == 2:
        return "P=2"
    if panel_count <= 4:
        return "P=3-4"
    return "P=5+"


def _cluster_means(
    values: Mapping[str, float],
    cluster_ids: Mapping[str, str],
) -> Dict[str, float]:
    grouped: Dict[str, list[float]] = {}
    for task_id, value in values.items():
        cluster_id = cluster_ids.get(task_id)
        if not isinstance(cluster_id, str) or not cluster_id:
            raise StatisticsError(f"Task {task_id!r} has no DOI cluster")
        grouped.setdefault(cluster_id, []).append(
            _finite_number(value, f"task[{task_id}]")
        )
    return {
        cluster_id: sum(cluster_values) / len(cluster_values)
        for cluster_id, cluster_values in sorted(grouped.items())
    }


def stratified_bootstrap(
    case_gaps: Mapping[str, float],
    panel_counts: Mapping[str, int],
    *,
    seed: int,
    resamples: int,
    doi_by_case: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, float]] = {
        stratum: {} for stratum in _PANEL_STRATA
    }
    for case_id, gap in case_gaps.items():
        if case_id not in panel_counts:
            raise StatisticsError(
                f"Missing panel_count for paired case {case_id!r}"
            )
        grouped[panel_stratum(panel_counts[case_id])][case_id] = gap
    result: Dict[str, Any] = {}
    for stratum in _PANEL_STRATA:
        task_values = grouped[stratum]
        if not task_values:
            result[stratum] = {
                "status": "NA",
                "n": 0,
                "mean_gap": None,
                "ci95": None,
                "resamples": resamples,
                "sampling_unit": "doi" if doi_by_case is not None else "case_id",
            }
            continue
        values = (
            list(_cluster_means(task_values, doi_by_case).values())
            if doi_by_case is not None
            else list(task_values.values())
        )
        result[stratum] = {
            "status": "ok",
            **paired_bootstrap(
                values,
                seed=seed,
                resamples=resamples,
                sampling_unit=(
                    "doi" if doi_by_case is not None else "case_id"
                ),
            ),
        }
    return result


def ordinal_panel_trend(
    case_gaps: Mapping[str, float],
    panel_counts: Mapping[str, int],
    doi_by_case: Mapping[str, str],
    *,
    seed: int,
    monte_carlo_permutations: int,
    exact_max_n: int,
) -> Dict[str, Any]:
    """Test an increasing panel-stratum trend with DOI-cluster sign flips."""

    grouped: Dict[str, Dict[str, list[float]]] = {}
    for case_id, raw_gap in case_gaps.items():
        if case_id not in panel_counts:
            raise StatisticsError(
                f"Missing panel_count for trend task {case_id!r}"
            )
        doi = doi_by_case.get(case_id)
        if not isinstance(doi, str) or not doi:
            raise StatisticsError(f"Missing DOI for trend task {case_id!r}")
        stratum = panel_stratum(panel_counts[case_id])
        grouped.setdefault(doi, {}).setdefault(stratum, []).append(
            _finite_number(raw_gap, f"trend_gap[{case_id}]")
        )

    doi_stratum_means = {
        doi: {
            stratum: sum(values) / len(values)
            for stratum, values in strata.items()
        }
        for doi, strata in grouped.items()
    }
    present = {
        stratum
        for strata in doi_stratum_means.values()
        for stratum in strata
    }
    missing = [stratum for stratum in _PANEL_STRATA if stratum not in present]
    base: Dict[str, Any] = {
        "sampling_unit": "doi",
        "alternative": "increasing",
        "ordinal_scores": dict(_PANEL_STRATUM_SCORES),
        "doi_count": len(doi_stratum_means),
        "task_count": len(case_gaps),
        "missing_strata": missing,
        "p_value_holm": None,
    }
    if missing:
        return {
            **base,
            "status": "blocked",
            "reason": "missing_panel_strata",
            "statistic": None,
            "stratum_means": None,
            "stratum_doi_counts": {
                stratum: sum(
                    stratum in strata
                    for strata in doi_stratum_means.values()
                )
                for stratum in _PANEL_STRATA
            },
            "p_value": None,
            "mode": None,
            "permutations": 0,
            "seed": seed,
        }

    centered_scores = {
        stratum: score
        - sum(_PANEL_STRATUM_SCORES.values())
        / len(_PANEL_STRATUM_SCORES)
        for stratum, score in _PANEL_STRATUM_SCORES.items()
    }
    denominator = sum(value * value for value in centered_scores.values())

    def statistic(signs: Mapping[str, float]) -> tuple[float, Dict[str, float]]:
        by_stratum: Dict[str, list[float]] = {
            stratum: [] for stratum in _PANEL_STRATA
        }
        for doi, strata in doi_stratum_means.items():
            sign = signs[doi]
            for stratum, value in strata.items():
                by_stratum[stratum].append(sign * value)
        stratum_means = {
            stratum: sum(values) / len(values)
            for stratum, values in by_stratum.items()
        }
        slope = sum(
            centered_scores[stratum] * stratum_means[stratum]
            for stratum in _PANEL_STRATA
        ) / denominator
        return slope, stratum_means

    doi_ids = sorted(doi_stratum_means)
    observed, stratum_means = statistic({doi: 1.0 for doi in doi_ids})
    epsilon = 1e-15
    if len(doi_ids) <= exact_max_n:
        permutations = 1 << len(doi_ids)
        extreme = 0
        for mask in range(permutations):
            permuted, _ = statistic(
                {
                    doi: 1.0 if mask & (1 << index) else -1.0
                    for index, doi in enumerate(doi_ids)
                }
            )
            if permuted + epsilon >= observed:
                extreme += 1
        p_value = extreme / permutations
        mode = "exact"
    else:
        if monte_carlo_permutations < 1:
            raise StatisticsError(
                "Panel-trend Monte Carlo permutations must be positive"
            )
        rng = random.Random(seed)
        permutations = monte_carlo_permutations
        extreme = 0
        for _ in range(permutations):
            permuted, _ = statistic(
                {
                    doi: 1.0 if rng.getrandbits(1) else -1.0
                    for doi in doi_ids
                }
            )
            if permuted + epsilon >= observed:
                extreme += 1
        p_value = (extreme + 1) / (permutations + 1)
        mode = "monte_carlo"

    return {
        **base,
        "status": "ok",
        "reason": None,
        "statistic": observed,
        "stratum_means": stratum_means,
        "stratum_doi_counts": {
            stratum: sum(
                stratum in strata for strata in doi_stratum_means.values()
            )
            for stratum in _PANEL_STRATA
        },
        "p_value": p_value,
        "mode": mode,
        "permutations": permutations,
        "seed": seed,
    }


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


def _method_doi_scores(
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    metric: str,
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    expected_dois: Optional[set[str]] = None
    for method in methods:
        values_by_case: Dict[str, list[float]] = {}
        doi_by_case: Dict[str, str] = {}
        for row in rows:
            if row["method"] != method:
                continue
            case_id = str(row["case_id"])
            doi = str(row["doi"])
            previous = doi_by_case.setdefault(case_id, doi)
            if previous != doi:
                raise StatisticsError(
                    f"Case {case_id!r} maps to multiple DOI clusters"
                )
            values_by_case.setdefault(case_id, []).append(
                _metric_value(row, metric)
            )
        task_means = {
            case_id: sum(values) / len(values)
            for case_id, values in values_by_case.items()
        }
        doi_scores = _cluster_means(task_means, doi_by_case)
        if expected_dois is None:
            expected_dois = set(doi_scores)
        elif set(doi_scores) != expected_dois:
            raise StatisticsError(
                f"Judge ranking DOI coverage differs for method {method!r}"
            )
        result[method] = doi_scores
    return result


def _kendall_doi_bootstrap(
    primary: Mapping[str, Mapping[str, float]],
    second: Mapping[str, Mapping[str, float]],
    methods: Sequence[str],
    *,
    seed: int,
    resamples: int,
) -> Dict[str, Any]:
    doi_sets = [
        set(scores)
        for scores in [*primary.values(), *second.values()]
    ]
    if not doi_sets or any(dois != doi_sets[0] for dois in doi_sets[1:]):
        raise StatisticsError("Kendall bootstrap requires identical DOI coverage")
    doi_ids = sorted(doi_sets[0])
    if not doi_ids:
        raise StatisticsError("Kendall bootstrap requires at least one DOI")

    def tau(sampled_dois: Sequence[str]) -> Optional[float]:
        primary_means = {
            method: sum(primary[method][doi] for doi in sampled_dois)
            / len(sampled_dois)
            for method in methods
        }
        second_means = {
            method: sum(second[method][doi] for doi in sampled_dois)
            / len(sampled_dois)
            for method in methods
        }
        return kendall_tau_b(
            [primary_means[method] for method in methods],
            [second_means[method] for method in methods],
        )

    estimate = tau(doi_ids)
    rng = random.Random(seed)
    bootstrap_values: list[float] = []
    undefined = 0
    for _ in range(resamples):
        sampled = [doi_ids[rng.randrange(len(doi_ids))] for _ in doi_ids]
        value = tau(sampled)
        if value is None:
            undefined += 1
        else:
            bootstrap_values.append(value)
    bootstrap_values.sort()
    estimable = (
        estimate is not None
        and len(bootstrap_values) == resamples
    )
    return {
        "status": "ok" if estimable else "not_estimable",
        "reason": (
            None
            if estimable
            else "kendall_tau_undefined_in_point_or_bootstrap"
        ),
        "estimate": estimate,
        "ci95": (
            [
                _percentile(bootstrap_values, 0.025),
                _percentile(bootstrap_values, 0.975),
            ]
            if estimable
            else None
        ),
        "resamples": resamples,
        "valid_resamples": len(bootstrap_values),
        "undefined_resamples": undefined,
        "seed": seed,
        "sampling_unit": "doi",
        "doi_count": len(doi_ids),
    }


def _c5_summary_provenance(
    summary: ValidatedSummary,
    primary_metric: str,
    second_metric: Optional[str],
) -> Optional[Dict[str, Any]]:
    requested = {primary_metric}
    if second_metric is not None:
        requested.add(second_metric)
    touches_c5 = any(metric.startswith("metric.visual_form.") for metric in requested)
    if not touches_c5:
        return None
    if (
        primary_metric != "metric.visual_form.claude-sonnet-4.6"
        or second_metric != "metric.visual_form.gemini-3.5-flash"
    ):
        raise StatisticsError(
            "C5 analysis requires Claude primary and Gemini secondary metrics"
        )
    c5 = summary.payload.get("c5_rejudge")
    if not isinstance(c5, Mapping) or c5.get("schema_version") != "2.0":
        raise StatisticsError("C5 summary lacks unambiguous dual-judge provenance")
    if c5.get("original_summary_hash") != summary.payload.get(
        "original_summary_hash"
    ):
        raise StatisticsError("C5 original summary lineage is inconsistent")
    input_manifest_hash = c5.get("input_manifest_hash")
    if (
        not isinstance(input_manifest_hash, str)
        or not _SHA256_RE.fullmatch(input_manifest_hash)
    ):
        raise StatisticsError("C5 input_manifest_hash is invalid")
    judges = c5.get("judges")
    order = c5.get("judge_order")
    if (
        not isinstance(judges, Mapping)
        or set(judges) != set(_C5_METRICS.values())
        or not isinstance(order, list)
        or len(order) != 2
        or set(order) != set(judges)
    ):
        raise StatisticsError("C5 summary must contain exactly two distinct judges")

    batch_hashes: Dict[str, str] = {}
    summary_source_hashes: set[str] = set()
    registry_hashes: set[str] = set()
    code_commits: set[str] = set()
    render_binding_hashes: set[str] = set()
    rubric_hashes: set[str] = set()
    prompt_hashes: set[str] = set()
    completed_names = {
        str(row["run_name"])
        for row in summary.rows
        if row.get("status") == "completed"
    }
    for metric, judge_id in _C5_METRICS.items():
        judge = judges.get(judge_id)
        if not isinstance(judge, Mapping):
            raise StatisticsError(f"C5 judge provenance is missing: {judge_id}")
        batch_hash = judge.get("batch_hash")
        if not isinstance(batch_hash, str) or not _SHA256_RE.fullmatch(batch_hash):
            raise StatisticsError(f"C5 judge batch_hash is invalid: {judge_id}")
        if (
            judge.get("judge_id") != judge_id
            or judge.get("judge_request_model")
            != metric.removeprefix("metric.visual_form.")
            or judge.get("judge_expected_served_model")
            != metric.removeprefix("metric.visual_form.")
            or judge.get("judge_served_models")
            != (
                [metric.removeprefix("metric.visual_form.")]
                if completed_names
                else []
            )
            or judge.get("metric") != metric
            or judge.get("input_manifest_hash") != input_manifest_hash
            or judge.get("code_git_dirty") is not False
            or judge.get("judge_max_tokens") != 1024
        ):
            raise StatisticsError(f"C5 judge provenance mismatch: {judge_id}")
        for name in (
            "rubric_hash",
            "prompt_hash",
            "source_summary_hash",
            "source_summary_sha256",
            "model_registry_sha256",
            "judge_config_hash",
            "metric_values_hash",
        ):
            value = judge.get(name)
            if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                raise StatisticsError(f"C5 {judge_id}.{name} is invalid")
        sidecars = judge.get("sidecar_hashes")
        renders = judge.get("selected_render_hashes")
        if not isinstance(sidecars, Mapping) or not isinstance(renders, Mapping):
            raise StatisticsError(f"C5 {judge_id} lacks sidecar/render hashes")
        if set(sidecars) != completed_names:
            raise StatisticsError(f"C5 {judge_id} sidecar coverage is not exact")
        if set(renders) != completed_names:
            raise StatisticsError(f"C5 {judge_id} render coverage is not exact")
        if any(
            not isinstance(value, str) or not _SHA256_RE.fullmatch(value)
            for value in [*sidecars.values(), *renders.values()]
        ):
            raise StatisticsError(f"C5 {judge_id} has invalid sidecar/render hashes")
        render_binding_hashes.add(sha256_json(dict(renders)))
        metric_values = {
            str(row["run_name"]): row.get(metric)
            for row in sorted(summary.rows, key=lambda item: str(item["run_name"]))
        }
        if sha256_json(metric_values) != judge["metric_values_hash"]:
            raise StatisticsError(f"C5 {judge_id} metric values are stale")
        batch_hashes[judge_id] = batch_hash
        summary_source_hashes.add(str(judge["source_summary_hash"]))
        registry_hashes.add(str(judge["model_registry_sha256"]))
        code_commits.add(str(judge.get("code_git_commit")))
        rubric_hashes.add(str(judge["rubric_hash"]))
        prompt_hashes.add(str(judge["prompt_hash"]))
    if (
        len(summary_source_hashes) != 1
        or next(iter(summary_source_hashes)) != c5.get("original_summary_hash")
        or len(registry_hashes) != 1
        or len(code_commits) != 1
        or len(render_binding_hashes) != 1
        or len(rubric_hashes) != 1
        or len(prompt_hashes) != 1
        or next(iter(rubric_hashes)) != c5.get("rubric_hash")
        or next(iter(prompt_hashes)) != c5.get("prompt_hash")
    ):
        raise StatisticsError("C5 judges do not share source/registry/code provenance")
    return {
        "status": "verified",
        "input_summary_hash": summary.summary_hash,
        "original_summary_hash": c5["original_summary_hash"],
        "input_manifest_hash": input_manifest_hash,
        "batch_hashes": batch_hashes,
        "model_registry_sha256": next(iter(registry_hashes)),
        "rejudge_code_git_commit": next(iter(code_commits)),
        "programmatic_correctness_privileged": True,
        "scope": (
            "Visual-form agreement is supporting evidence only; programmatic "
            "fidelity and cohesion remain authoritative for correctness."
        ),
    }


def _c5_kendall_decision(uncertainty: Mapping[str, Any]) -> Dict[str, Any]:
    estimate = uncertainty.get("estimate")
    ci = uncertainty.get("ci95")
    estimable = (
        uncertainty.get("status") == "ok"
        and isinstance(estimate, (int, float))
        and not isinstance(estimate, bool)
        and isinstance(ci, list)
        and len(ci) == 2
    )
    passed = bool(
        estimable
        and float(estimate) >= _C5_POINT_THRESHOLD
        and float(ci[0]) > _C5_CI_LOWER_THRESHOLD
    )
    return {
        "status": (
            "pass"
            if passed
            else ("fail" if estimable else "not_estimable")
        ),
        "point_threshold": _C5_POINT_THRESHOLD,
        "point_comparison": ">=",
        "ci_lower_threshold": _C5_CI_LOWER_THRESHOLD,
        "ci_lower_comparison": ">",
        "estimate": estimate,
        "ci95": ci,
        "concordance_claim_permitted": passed,
        "programmatic_correctness_privileged": True,
    }


def _ranking_analysis(
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    *,
    primary_metric: str,
    second_judge_metric: Optional[str],
    seed: int,
    bootstrap_resamples: int,
) -> Dict[str, Any]:
    primary_doi_scores = _method_doi_scores(
        rows,
        methods,
        primary_metric,
    )
    primary_scores = {
        method: sum(scores.values()) / len(scores)
        for method, scores in primary_doi_scores.items()
    }
    result: Dict[str, Any] = {
        "primary_metric": primary_metric,
        "ranking_direction": "higher_is_better",
        "aggregation_order": ["seed_mean", "task_mean", "doi_mean"],
        "primary_ranking": method_ranking(primary_scores),
    }
    if second_judge_metric is None:
        result.update(
            {
                "second_judge_metric": None,
                "second_judge_ranking": None,
                "kendall_tau_b": None,
                "kendall_tau_b_uncertainty": {
                    "status": "not_requested",
                    "reason": "second_judge_metric_not_requested",
                    "estimate": None,
                    "ci95": None,
                    "resamples": 0,
                    "valid_resamples": 0,
                    "undefined_resamples": 0,
                    "seed": seed,
                    "sampling_unit": "doi",
                    "doi_count": len(next(iter(primary_doi_scores.values()))),
                },
                "status": "not_requested",
            }
        )
        return result
    second_doi_scores = _method_doi_scores(
        rows,
        methods,
        second_judge_metric,
    )
    second_scores = {
        method: sum(scores.values()) / len(scores)
        for method, scores in second_doi_scores.items()
    }
    uncertainty = _kendall_doi_bootstrap(
        primary_doi_scores,
        second_doi_scores,
        methods,
        seed=seed,
        resamples=bootstrap_resamples,
    )
    result.update(
        {
            "second_judge_metric": second_judge_metric,
            "second_judge_ranking": method_ranking(second_scores),
            "kendall_tau_b": uncertainty["estimate"],
            "kendall_tau_b_uncertainty": uncertainty,
            "status": (
                "ok"
                if uncertainty["status"] == "ok"
                else "not_estimable"
            ),
            "c5_decision": (
                _c5_kendall_decision(uncertainty)
                if {
                    primary_metric,
                    second_judge_metric,
                }
                == set(_C5_METRICS)
                else None
            ),
        }
    )
    return result


def normalize_trajectory_threshold_config(
    raw: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate an explicitly frozen joint-quality threshold configuration."""

    if not isinstance(raw, Mapping):
        raise StatisticsError("trajectory threshold config must be an object")
    expected_keys = {
        "schema_version",
        "fidelity",
        "cohesion",
        "restriction",
    }
    if set(raw) != expected_keys:
        raise StatisticsError(
            "trajectory threshold config must contain exactly "
            + ", ".join(sorted(expected_keys))
        )
    if raw.get("schema_version") != "1.0":
        raise StatisticsError("trajectory threshold schema_version must be 1.0")

    metrics: Dict[str, Dict[str, Any]] = {}
    for label, expected_metric in (
        ("fidelity", "data_fidelity"),
        ("cohesion", "series_cohesion"),
    ):
        block = raw.get(label)
        if not isinstance(block, Mapping) or set(block) != {
            "metric",
            "threshold",
        }:
            raise StatisticsError(
                f"trajectory {label} must contain metric and threshold"
            )
        if block.get("metric") != expected_metric:
            raise StatisticsError(
                f"trajectory {label}.metric must be {expected_metric!r}"
            )
        threshold = _finite_number(
            block.get("threshold"),
            f"trajectory.{label}.threshold",
        )
        if not 0.0 <= threshold <= 1.0:
            raise StatisticsError(
                f"trajectory {label}.threshold must be in [0,1]"
            )
        metrics[label] = {
            "metric": expected_metric,
            "threshold": threshold,
        }

    restriction = raw.get("restriction")
    if not isinstance(restriction, Mapping) or set(restriction) != {
        "renders",
        "wall_clock_seconds",
    }:
        raise StatisticsError(
            "trajectory restriction must contain renders and wall_clock_seconds"
        )
    renders = restriction.get("renders")
    if isinstance(renders, bool) or not isinstance(renders, int) or renders < 1:
        raise StatisticsError("trajectory restriction.renders must be positive")
    wall_clock_seconds = _finite_number(
        restriction.get("wall_clock_seconds"),
        "trajectory.restriction.wall_clock_seconds",
    )
    if wall_clock_seconds <= 0:
        raise StatisticsError(
            "trajectory restriction.wall_clock_seconds must be positive"
        )
    return {
        "schema_version": "1.0",
        **metrics,
        "restriction": {
            "renders": renders,
            "wall_clock_seconds": wall_clock_seconds,
        },
    }


def load_trajectory_threshold_config(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        raw = json.loads(
            resolved.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except OSError as exc:
        raise StatisticsError(
            f"Cannot read trajectory threshold config {resolved}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise StatisticsError(
            f"Invalid trajectory threshold config: {exc}"
        ) from exc
    return normalize_trajectory_threshold_config(raw)


def _trajectory_json(path: Path, label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except OSError as exc:
        raise StatisticsError(f"Cannot read {label} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise StatisticsError(f"Invalid {label} JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StatisticsError(f"{label} must be a JSON object: {path}")
    return payload


def _trajectory_run_dir(summary: ValidatedSummary, run_name: str) -> Path:
    if summary.source_root is None:
        raise StatisticsError(
            "C3 trajectory analysis requires summary.source_root"
        )
    root = summary.source_root
    if not root.is_dir():
        raise StatisticsError(
            f"C3 trajectory source_root does not exist: {root}"
        )
    relative = Path(run_name)
    if (
        relative.is_absolute()
        or len(relative.parts) != 1
        or relative.name != run_name
    ):
        raise StatisticsError(f"Unsafe trajectory run_name: {run_name!r}")
    run_dir = root / relative
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise StatisticsError(
            f"Trajectory run directory is missing or a symlink: {run_name}"
        )
    try:
        resolved = run_dir.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise StatisticsError(
            f"Trajectory run directory escaped source_root: {run_name}"
        ) from exc
    return resolved


def _trajectory_record(
    summary: ValidatedSummary,
    row: Mapping[str, Any],
) -> tuple[RunRecord, Path]:
    run_name = str(row["run_name"])
    run_dir = _trajectory_run_dir(summary, run_name)
    record_path = run_dir / "run_record.json"
    try:
        record = RunRecord.read(record_path)
        record.validate_provenance(
            require_completed=row["status"] == "completed"
        )
        verify_artifacts(record, run_dir)
        _, selected_case = _verified_frozen_manifest_case(record, run_dir)
    except (OSError, ProvenanceError, TypeError) as exc:
        raise StatisticsError(
            f"Invalid trajectory provenance for {run_name}: {exc}"
        ) from exc
    mirrored = {
        "run_name": record.run_name,
        "method": record.method,
        "backbone": record.backbone,
        "case_id": record.case_id,
        "panel_count": record.panel_count,
        "split": record.split,
        "seed": record.seed,
        "budget_type": record.budget_type,
        "budget_value": float(record.budget_value),
        "dataset_manifest_hash": record.dataset_manifest_hash,
        "git_commit": record.git_commit,
        "status": record.status,
        "metric_config_hash": record.metric_config_hash,
        "spec_hash": record.spec_hash,
        "record_hash": record.record_hash,
    }
    for name, value in mirrored.items():
        expected = (
            float(row[name]) if name == "budget_value" else row.get(name)
        )
        if value != expected:
            raise StatisticsError(
                f"Trajectory record disagrees with summary for "
                f"{run_name}.{name}"
            )
    if bool(record.experiment_spec.get("git_dirty")):
        raise StatisticsError(
            f"Trajectory record used a dirty worktree: {run_name}"
        )
    if record.test_only:
        raise StatisticsError(
            f"Trajectory record is test-only: {run_name}"
        )
    manifest_doi = _normalized_doi(selected_case.payload)
    if not manifest_doi or manifest_doi != row.get("doi"):
        raise StatisticsError(
            f"Trajectory DOI disagrees with frozen manifest: {run_name}"
        )
    row_metrics = {
        name.removeprefix("metric."): _finite_number(
            value,
            f"{run_name}.{name}",
        )
        for name, value in row.items()
        if name.startswith("metric.")
    }
    if record.status == "completed":
        expected_metrics = {
            name: float(value) for name, value in record.metrics.items()
        }
        expected_metrics["execution_success"] = 1.0
        if set(row_metrics) != set(expected_metrics):
            raise StatisticsError(
                f"Trajectory summary metric set disagrees with record: "
                f"{run_name}"
            )
        for name, value in expected_metrics.items():
            if not math.isclose(
                row_metrics[name],
                value,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise StatisticsError(
                    f"Trajectory summary metric disagrees with record: "
                    f"{run_name}/metric.{name}"
                )
    elif (
        row_metrics.get("execution_success") != 0.0
        or any(value != 0.0 for value in row_metrics.values())
    ):
        raise StatisticsError(
            f"Trajectory failed-run summary metrics must be zero: {run_name}"
        )
    return record, run_dir


def _candidate_sidecar(
    record: RunRecord,
    run_dir: Path,
    candidate: Mapping[str, Any],
) -> None:
    candidate_id = str(candidate.get("candidate_id") or "")
    key = f"{candidate_id}.metadata"
    relative_text = record.artifact_paths.get(key)
    if not relative_text:
        raise StatisticsError(
            f"Trajectory candidate lacks metadata artifact: "
            f"{record.run_name}/{candidate_id}"
        )
    sidecar = _trajectory_json(
        run_dir / relative_text,
        "trajectory candidate metadata",
    )
    if sidecar != candidate:
        raise StatisticsError(
            f"Trajectory candidate metadata disagrees with run record: "
            f"{record.run_name}/{candidate_id}"
        )
    artifact_paths = candidate.get("artifact_paths")
    artifact_hashes = candidate.get("artifact_hashes")
    if not isinstance(artifact_paths, Mapping) or not isinstance(
        artifact_hashes,
        Mapping,
    ):
        raise StatisticsError(
            f"Trajectory candidate artifact maps are invalid: "
            f"{record.run_name}/{candidate_id}"
        )
    if set(artifact_paths) != set(artifact_hashes):
        raise StatisticsError(
            f"Trajectory candidate artifact maps disagree: "
            f"{record.run_name}/{candidate_id}"
        )
    for label, relative in artifact_paths.items():
        record_key = f"{candidate_id}.{label}"
        if (
            record.artifact_paths.get(record_key) != relative
            or record.artifact_hashes.get(record_key)
            != artifact_hashes.get(label)
        ):
            raise StatisticsError(
                f"Trajectory candidate artifact binding disagrees: "
                f"{record.run_name}/{record_key}"
            )


def _programmatic_quality(
    record: RunRecord,
    run_dir: Path,
    candidate: Mapping[str, Any],
) -> tuple[float, float]:
    candidate_id = str(candidate["candidate_id"])
    artifact_paths = candidate["artifact_paths"]
    relative = artifact_paths.get("programmatic_evaluation")
    if not isinstance(relative, str) or not relative:
        raise StatisticsError(
            f"Trajectory candidate has no programmatic evaluation: "
            f"{record.run_name}/{candidate_id}"
        )
    payload = _trajectory_json(
        run_dir / relative,
        "trajectory programmatic evaluation",
    )
    panel_fidelity = payload.get("panel_fidelity")
    if isinstance(panel_fidelity, Mapping):
        fidelity_parts = [
            item for item in panel_fidelity.values() if isinstance(item, Mapping)
        ]
        if len(fidelity_parts) != len(panel_fidelity):
            raise StatisticsError(
                f"Trajectory panel fidelity is malformed: "
                f"{record.run_name}/{candidate_id}"
            )
        numerator = sum(
            _finite_number(
                item.get("numerator"),
                f"{record.run_name}/{candidate_id}.fidelity.numerator",
            )
            for item in fidelity_parts
        )
        denominator = sum(
            _finite_number(
                item.get("denominator"),
                f"{record.run_name}/{candidate_id}.fidelity.denominator",
            )
            for item in fidelity_parts
        )
        if denominator <= 0:
            raise StatisticsError(
                f"Trajectory fidelity is unavailable: "
                f"{record.run_name}/{candidate_id}"
            )
        fidelity = numerator / denominator
    else:
        fidelity_block = payload.get("fidelity")
        if not isinstance(fidelity_block, Mapping):
            raise StatisticsError(
                f"Trajectory fidelity is unavailable: "
                f"{record.run_name}/{candidate_id}"
            )
        fidelity = _finite_number(
            fidelity_block.get("ratio"),
            f"{record.run_name}/{candidate_id}.fidelity",
        )
    cohesion_block = payload.get("cohesion")
    if (
        not isinstance(cohesion_block, Mapping)
        or cohesion_block.get("applicable") is False
        or cohesion_block.get("ratio") is None
    ):
        raise StatisticsError(
            f"Trajectory cohesion is unavailable: "
            f"{record.run_name}/{candidate_id}"
        )
    cohesion = _finite_number(
        cohesion_block.get("ratio"),
        f"{record.run_name}/{candidate_id}.cohesion",
    )
    for value, label in ((fidelity, "fidelity"), (cohesion, "cohesion")):
        if not 0.0 <= value <= 1.0:
            raise StatisticsError(
                f"Trajectory {label} is outside [0,1]: "
                f"{record.run_name}/{candidate_id}"
            )
    metrics = candidate.get("metrics")
    if not isinstance(metrics, Mapping):
        raise StatisticsError(
            f"Trajectory candidate metrics are unavailable: "
            f"{record.run_name}/{candidate_id}"
        )
    for name, actual in (
        ("data_fidelity", fidelity),
        ("series_cohesion", cohesion),
    ):
        recorded = _finite_number(
            metrics.get(name),
            f"{record.run_name}/{candidate_id}.metrics.{name}",
        )
        if not math.isclose(recorded, actual, rel_tol=0.0, abs_tol=1e-12):
            raise StatisticsError(
                f"Trajectory metric disagrees with immutable evaluation: "
                f"{record.run_name}/{candidate_id}/{name}"
            )
    return fidelity, cohesion


def _trajectory_observation(
    summary: ValidatedSummary,
    row: Mapping[str, Any],
    threshold: Mapping[str, Any],
) -> Dict[str, Any]:
    record, run_dir = _trajectory_record(summary, row)
    render_horizon = int(threshold["restriction"]["renders"])
    time_horizon = float(
        threshold["restriction"]["wall_clock_seconds"]
    )
    if record.budget_type != "renders":
        raise StatisticsError(
            f"C3 trajectory requires a render budget: {record.run_name}"
        )
    if (
        not float(record.budget_value).is_integer()
        or int(record.budget_value) != render_horizon
    ):
        raise StatisticsError(
            f"Trajectory restriction does not match run budget: "
            f"{record.run_name}"
        )
    if int(record.panel_count or 0) <= 1:
        raise StatisticsError(
            f"C3 trajectory is inapplicable to single-panel run: "
            f"{record.run_name}"
        )
    panel_count = int(record.panel_count)
    if render_horizon % panel_count:
        raise StatisticsError(
            f"Trajectory restriction is not panel-exact: {record.run_name}"
        )
    method_failed = record.status == "failed"
    if method_failed and (record.error or {}).get("attribution") != "method":
        raise StatisticsError(
            f"Trajectory refuses non-method failure: {record.run_name}"
        )
    maximum_candidates = render_horizon // panel_count
    if method_failed:
        if (
            len(record.candidates) > maximum_candidates
            or record.render_count != len(record.candidates) * panel_count
        ):
            raise StatisticsError(
                f"Method-failure trajectory accounting is incomplete: "
                f"{record.run_name}"
            )
    elif (
        record.render_count != render_horizon
        or len(record.candidates) != maximum_candidates
    ):
        raise StatisticsError(
            f"Trajectory run does not fill the common render restriction: "
            f"{record.run_name}"
        )
    schedule = str(record.experiment_spec.get("schedule") or "")
    if schedule not in {"iterative", "best_of_n"}:
        raise StatisticsError(
            f"Unsupported trajectory schedule: {record.run_name}/{schedule}"
        )

    global_render = 0
    global_time = 0.0
    crossing: Optional[tuple[str, int, float]] = None
    fidelity_threshold = float(threshold["fidelity"]["threshold"])
    cohesion_threshold = float(threshold["cohesion"]["threshold"])
    points: list[Dict[str, Any]] = []
    for index, candidate in enumerate(record.candidates, 1):
        expected_id = f"candidate_{index:04d}"
        if candidate.get("candidate_id") != expected_id:
            raise StatisticsError(
                f"Trajectory candidates are not contiguous: {record.run_name}"
            )
        _candidate_sidecar(record, run_dir, candidate)
        candidate_render_count = candidate.get("render_count")
        if candidate_render_count != panel_count:
            raise StatisticsError(
                f"Trajectory candidate is not a complete panel round: "
                f"{record.run_name}/{expected_id}"
            )
        call_index = candidate.get("call_index")
        if isinstance(call_index, bool) or not isinstance(call_index, int):
            raise StatisticsError(
                f"Trajectory candidate call_index is invalid: "
                f"{record.run_name}/{expected_id}"
            )
        provider_metadata = candidate.get("provider_metadata")
        if not isinstance(provider_metadata, Mapping):
            raise StatisticsError(
                f"Trajectory candidate provider metadata is missing: "
                f"{record.run_name}/{expected_id}"
            )
        cumulative_render = provider_metadata.get("cumulative_render_count")
        if (
            isinstance(cumulative_render, bool)
            or not isinstance(cumulative_render, int)
            or cumulative_render < 1
        ):
            raise StatisticsError(
                f"Trajectory cumulative renders must be positive integers: "
                f"{record.run_name}/{expected_id}"
            )
        cumulative_time = _finite_number(
            provider_metadata.get("cumulative_wall_clock_seconds"),
            f"{record.run_name}/{expected_id}.cumulative_wall_clock_seconds",
        )
        if cumulative_time <= 0:
            raise StatisticsError(
                f"Trajectory cumulative time must be positive: "
                f"{record.run_name}/{expected_id}"
            )
        if schedule == "iterative":
            if (
                call_index != 1
                or cumulative_render != global_render + panel_count
                or cumulative_time <= global_time
            ):
                raise StatisticsError(
                    f"Nonmonotonic iterative trajectory metadata: "
                    f"{record.run_name}/{expected_id}"
                )
            global_render = cumulative_render
            global_time = cumulative_time
        else:
            if (
                call_index != index
                or cumulative_render != global_render + panel_count
                or cumulative_time <= global_time
            ):
                raise StatisticsError(
                    f"Nonmonotonic best-of-N trajectory metadata: "
                    f"{record.run_name}/{expected_id}"
                )
            global_render = cumulative_render
            global_time = cumulative_time
        if global_render > render_horizon or global_time > record.wall_clock_seconds:
            raise StatisticsError(
                f"Trajectory cumulative metadata exceeds terminal accounting: "
                f"{record.run_name}/{expected_id}"
            )
        fidelity, cohesion = _programmatic_quality(
            record,
            run_dir,
            candidate,
        )
        points.append(
            {
                "candidate_id": expected_id,
                "cumulative_render_count": global_render,
                "cumulative_wall_clock_seconds": global_time,
                "data_fidelity": fidelity,
                "series_cohesion": cohesion,
            }
        )
        if (
            crossing is None
            and fidelity >= fidelity_threshold
            and cohesion >= cohesion_threshold
        ):
            crossing = (expected_id, global_render, global_time)
    expected_terminal_render = (
        record.render_count if method_failed else render_horizon
    )
    if global_render != expected_terminal_render:
        raise StatisticsError(
            f"Trajectory does not match terminal render accounting: "
            f"{record.run_name}"
        )

    crossing_id = crossing[0] if crossing else None
    crossing_render = crossing[1] if crossing else None
    crossing_time = crossing[2] if crossing else None
    render_observed = (
        crossing_render is not None and crossing_render <= render_horizon
    )
    time_observed = (
        crossing_time is not None and crossing_time <= time_horizon
    )
    joint = render_observed and time_observed
    if method_failed and crossing is None:
        trajectory_status = "method_failure_censored"
        censor_reason = "explicit_method_failure_before_threshold"
    elif method_failed and joint:
        trajectory_status = "observed_before_method_failure"
        censor_reason = None
    elif joint:
        trajectory_status = "observed"
        censor_reason = None
    else:
        trajectory_status = "right_censored"
        censor_reason = (
            "quality_threshold_not_attained"
            if crossing is None
            else "wall_clock_restriction_exceeded"
        )
    return {
        "run_name": record.run_name,
        "method": record.method,
        "case_id": record.case_id,
        "doi": str(row["doi"]),
        "seed": record.seed,
        "panel_count": record.panel_count,
        "status": trajectory_status,
        "threshold_crossing_candidate": crossing_id,
        "threshold_crossing_render": crossing_render,
        "threshold_crossing_wall_clock_seconds": crossing_time,
        "render_event_observed": render_observed,
        "time_event_observed": time_observed,
        "joint_attainment": joint,
        "restricted_renders_to_threshold": float(
            crossing_render if render_observed else render_horizon
        ),
        "restricted_wall_clock_seconds_to_threshold": float(
            crossing_time if time_observed else time_horizon
        ),
        "censor_reason": censor_reason,
        "record_hash": record.record_hash,
        "trajectory_points": points,
    }


def _trajectory_means(
    observations: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    if not observations:
        raise StatisticsError("Cannot aggregate an empty trajectory group")
    n = len(observations)
    return {
        "attainment_rate": sum(
            bool(item["joint_attainment"]) for item in observations
        )
        / n,
        "render_attainment_rate": sum(
            bool(item["render_event_observed"]) for item in observations
        )
        / n,
        "time_attainment_rate": sum(
            bool(item["time_event_observed"]) for item in observations
        )
        / n,
        "restricted_mean_renders_to_threshold": sum(
            float(item["restricted_renders_to_threshold"])
            for item in observations
        )
        / n,
        "restricted_mean_wall_clock_seconds_to_threshold": sum(
            float(item["restricted_wall_clock_seconds_to_threshold"])
            for item in observations
        )
        / n,
    }


def _trajectory_summary_means(
    summaries: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    if not summaries:
        raise StatisticsError("Cannot aggregate empty trajectory summaries")
    fields = (
        "attainment_rate",
        "render_attainment_rate",
        "time_attainment_rate",
        "restricted_mean_renders_to_threshold",
        "restricted_mean_wall_clock_seconds_to_threshold",
    )
    return {
        field: sum(float(item[field]) for item in summaries) / len(summaries)
        for field in fields
    }


def _trajectory_analysis(
    summary: ValidatedSummary,
    rows: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str],
    threshold: Mapping[str, Any],
) -> Dict[str, Any]:
    by_slice: Dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        by_slice.setdefault(_slice_key(row), []).append(row)
    slices: list[Dict[str, Any]] = []
    for key in sorted(by_slice, key=canonical_json):
        backbone, budget_type, budget_value, split = key
        slice_rows = by_slice[key]
        if budget_type != "renders":
            raise StatisticsError(
                "C3 trajectory currently requires render-budget slices"
            )
        if int(threshold["restriction"]["renders"]) != int(budget_value):
            raise StatisticsError(
                "C3 trajectory restriction does not match slice render budget"
            )
        seeds = sorted({int(row["seed"]) for row in slice_rows})
        case_ids: Optional[set[str]] = None
        for seed in seeds:
            try:
                seed_cases = assert_paired_ready(
                    [
                        dict(row)
                        for row in slice_rows
                        if int(row["seed"]) == seed
                    ],
                    methods=methods,
                    backbone=str(backbone),
                    seed=seed,
                    budget_type=str(budget_type),
                    budget_value=float(budget_value),
                )
            except AggregationError as exc:
                raise StatisticsError(str(exc)) from exc
            if case_ids is None:
                case_ids = seed_cases
            elif seed_cases != case_ids:
                raise StatisticsError(
                    "C3 trajectory task sets differ across seeds"
                )
        if not case_ids:
            raise StatisticsError("C3 trajectory slice contains no paired tasks")
        if any(int(row["panel_count"]) <= 1 for row in slice_rows):
            raise StatisticsError(
                "C3 trajectory is inapplicable to single-panel rows"
            )

        observations = [
            _trajectory_observation(summary, row, threshold)
            for row in sorted(
                slice_rows,
                key=lambda item: (
                    str(item["method"]),
                    str(item["case_id"]),
                    int(item["seed"]),
                ),
            )
        ]
        method_results = []
        for method in methods:
            method_observations = [
                item for item in observations if item["method"] == method
            ]
            task_results = []
            for case_id in sorted(case_ids):
                task_observations = [
                    item
                    for item in method_observations
                    if item["case_id"] == case_id
                ]
                task_seeds = {int(item["seed"]) for item in task_observations}
                if task_seeds != set(seeds):
                    raise StatisticsError(
                        f"C3 trajectory has incomplete seeds for "
                        f"{method}/{case_id}"
                    )
                dois = {str(item["doi"]) for item in task_observations}
                if len(dois) != 1:
                    raise StatisticsError(
                        f"C3 trajectory task DOI changed: {case_id}"
                    )
                task_results.append(
                    {
                        "case_id": case_id,
                        "doi": next(iter(dois)),
                        "seed_count": len(task_observations),
                        **_trajectory_means(task_observations),
                    }
                )
            doi_results = []
            for doi in sorted({item["doi"] for item in task_results}):
                doi_tasks = [
                    item for item in task_results if item["doi"] == doi
                ]
                doi_results.append(
                    {
                        "doi": doi,
                        "task_count": len(doi_tasks),
                        **_trajectory_summary_means(doi_tasks),
                    }
                )
            method_results.append(
                {
                    "method": method,
                    "run_count": len(method_observations),
                    "task_count": len(task_results),
                    "doi_count": len(doi_results),
                    "tasks": task_results,
                    "doi_clusters": doi_results,
                    **_trajectory_summary_means(doi_results),
                }
            )
        slices.append(
            {
                "backbone": backbone,
                "budget_type": budget_type,
                "budget_value": budget_value,
                "split": split,
                "seed_count": len(seeds),
                "task_count": len(case_ids),
                "doi_count": len(
                    {str(item["doi"]) for item in observations}
                ),
                "methods": method_results,
                "run_observations": observations,
            }
        )
    return {
        "status": "ok",
        "estimand": "joint_fidelity_cohesion_threshold_attainment",
        "threshold_config": dict(threshold),
        "threshold_config_hash": sha256_json(threshold),
        "aggregation_order": ["seed", "task", "doi_cluster"],
        "sampling_unit": "doi",
        "trajectory_contract": {
            "quality_artifact": (
                "candidate.artifact_paths.programmatic_evaluation"
            ),
            "cumulative_render_metadata": (
                "candidate.provider_metadata.cumulative_render_count"
            ),
            "cumulative_time_metadata": (
                "candidate.provider_metadata.cumulative_wall_clock_seconds"
            ),
            "candidate_binding": "candidate_NNNN.metadata",
        },
        "slices": slices,
    }


def analyze_summary(
    summary: ValidatedSummary,
    *,
    reference: str,
    methods: Sequence[str],
    metric: str,
    panel_scope: str = "all",
    second_judge_metric: Optional[str] = None,
    trajectory_threshold: Optional[Mapping[str, Any]] = None,
    seed: int = 17_029,
    bootstrap_resamples: int = 10_000,
    monte_carlo_permutations: int = 100_000,
    exact_max_n: int = 16,
) -> Dict[str, Any]:
    """Build analysis schema v3.2.

    Version 3.1 adds a provenance-recorded panel scope so structurally
    inapplicable single-panel cohesion rows can be excluded before pairing.
    When and only when an explicit trajectory threshold is supplied, the same
    schema also emits the provenance-checked C3 right-censored extension.
    Version 3.2 adds provenance-bound dual-judge C5 decisions.
    No production quality threshold is defined in code.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise StatisticsError("Analysis seed must be an integer")
    if bootstrap_resamples < 1:
        raise StatisticsError("bootstrap resamples must be positive")
    if monte_carlo_permutations < 1:
        raise StatisticsError("permutation count must be positive")
    if exact_max_n < 1:
        raise StatisticsError("exact_max_n must be positive")
    if panel_scope not in _PANEL_SCOPES:
        raise StatisticsError(
            f"panel_scope must be one of {sorted(_PANEL_SCOPES)}"
        )
    c5_provenance = _c5_summary_provenance(
        summary,
        metric,
        second_judge_metric,
    )
    normalized_trajectory: Optional[Dict[str, Any]] = None
    if trajectory_threshold is not None:
        normalized_trajectory = normalize_trajectory_threshold_config(
            trajectory_threshold
        )
        if panel_scope != "multi_panel":
            raise StatisticsError(
                "C3 trajectory thresholds require panel_scope='multi_panel'; "
                "single-panel trajectories are inapplicable"
            )
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
    if panel_scope == "single_panel":
        selected_rows = [
            row for row in selected_rows if int(row["panel_count"]) == 1
        ]
    elif panel_scope == "multi_panel":
        selected_rows = [
            row for row in selected_rows if int(row["panel_count"]) > 1
        ]
    if not selected_rows:
        raise StatisticsError("No rows match the requested methods")
    available_methods = {row["method"] for row in selected_rows}
    missing_methods = set(selected_methods) - available_methods
    if missing_methods:
        raise StatisticsError(
            f"Requested methods are absent: {sorted(missing_methods)}"
        )
    missing_doi = [
        str(row["run_name"]) for row in selected_rows if not row.get("doi")
    ]
    if missing_doi:
        raise StatisticsError(
            "Confirmatory analysis requires a sealed DOI cluster for every "
            f"selected run; missing={sorted(missing_doi)}"
        )

    by_slice: Dict[tuple[Any, ...], list[Dict[str, Any]]] = {}
    for row in selected_rows:
        by_slice.setdefault(_slice_key(row), []).append(row)

    slice_results = []
    for key in sorted(by_slice, key=lambda item: canonical_json(item)):
        backbone, budget_type, budget_value, split = key
        rows = by_slice[key]
        seeds = sorted({int(row["seed"]) for row in rows})
        case_ids: Optional[set[str]] = None
        for slice_seed in seeds:
            seed_rows = [row for row in rows if row["seed"] == slice_seed]
            try:
                seed_case_ids = assert_paired_ready(
                    seed_rows,
                    methods=selected_methods,
                    backbone=backbone,
                    seed=slice_seed,
                    budget_type=budget_type,
                    budget_value=budget_value,
                )
            except AggregationError as exc:
                raise StatisticsError(str(exc)) from exc
            if case_ids is None:
                case_ids = seed_case_ids
            elif seed_case_ids != case_ids:
                raise StatisticsError(
                    f"Slice {key!r} has different task sets across seeds"
                )
        if not case_ids:
            raise StatisticsError(f"Slice {key!r} contains no paired tasks")
        manifest_hashes = {row["dataset_manifest_hash"] for row in rows}
        config_hashes = {row["metric_config_hash"] for row in rows}
        metric_versions = {row["metric_version"] for row in rows}
        git_commits = {row["git_commit"] for row in rows}
        if (
            len(manifest_hashes) != 1
            or len(config_hashes) != 1
            or len(metric_versions) != 1
            or len(git_commits) != 1
        ):
            raise StatisticsError(
                f"Slice {key!r} mixes code, manifest, or metric provenance"
            )

        panel_counts: Dict[str, int] = {}
        doi_by_case: Dict[str, str] = {}
        for case_id in case_ids:
            task_rows = [row for row in rows if row["case_id"] == case_id]
            task_panel_counts = {
                int(row["panel_count"]) for row in task_rows
            }
            task_dois = {str(row["doi"]) for row in task_rows}
            if len(task_panel_counts) != 1 or len(task_dois) != 1:
                raise StatisticsError(
                    f"Task metadata changes across methods or seeds: {case_id!r}"
                )
            panel_counts[case_id] = next(iter(task_panel_counts))
            doi_by_case[case_id] = next(iter(task_dois))
        doi_ids = sorted(set(doi_by_case.values()))
        if len(doi_ids) < 2:
            raise StatisticsError(
                f"Slice {key!r} has fewer than two DOI clusters"
            )

        def task_metric_means(method: str, metric_name: str) -> Dict[str, float]:
            means: Dict[str, float] = {}
            for case_id in case_ids:
                task_rows = [
                    row
                    for row in rows
                    if row["method"] == method and row["case_id"] == case_id
                ]
                task_seeds = {int(row["seed"]) for row in task_rows}
                if task_seeds != set(seeds):
                    raise StatisticsError(
                        f"Task {case_id!r} has an incomplete seed set for "
                        f"method {method!r}"
                    )
                values = [
                    _metric_value(row, metric_name) for row in task_rows
                ]
                means[case_id] = sum(values) / len(values)
            return means

        reference_task_means = task_metric_means(reference, metric)
        comparisons = []
        for method in comparison_methods:
            method_task_means = task_metric_means(method, metric)
            case_gaps = {
                case_id: method_task_means[case_id]
                - reference_task_means[case_id]
                for case_id in sorted(case_ids)
            }
            doi_gaps = _cluster_means(case_gaps, doi_by_case)
            gaps = list(doi_gaps.values())
            comparisons.append(
                {
                    "method": method,
                    "gap_definition": f"{method} - {reference}",
                    "overall": paired_bootstrap(
                        gaps,
                        seed=seed,
                        resamples=bootstrap_resamples,
                        sampling_unit="doi",
                    ),
                    "permutation": paired_sign_flip_permutation(
                        gaps,
                        seed=seed,
                        monte_carlo_permutations=monte_carlo_permutations,
                        exact_max_n=exact_max_n,
                        sampling_unit="doi",
                    ),
                    "panel_strata": stratified_bootstrap(
                        case_gaps,
                        panel_counts,
                        seed=seed,
                        resamples=bootstrap_resamples,
                        doi_by_case=doi_by_case,
                    ),
                    "panel_trend": ordinal_panel_trend(
                        case_gaps,
                        panel_counts,
                        doi_by_case,
                        seed=seed,
                        monte_carlo_permutations=monte_carlo_permutations,
                        exact_max_n=exact_max_n,
                    ),
                }
            )

        overall_raw = {
            comparison["method"]: comparison["permutation"]["p_value"]
            for comparison in comparisons
        }
        overall_adjusted = holm_adjust(overall_raw)
        for comparison in comparisons:
            comparison["permutation"]["p_value_holm"] = overall_adjusted[
                comparison["method"]
            ]
            comparison["permutation"]["holm_family"] = "overall"
        overall_family = {
            "status": "ok",
            "adjustment": "holm",
            "scope": "within_slice",
            "estimand": "overall_method_minus_reference",
            "members": list(comparison_methods),
            "raw_p_values": overall_raw,
            "adjusted_p_values": overall_adjusted,
        }

        trend_blockers = [
            comparison["method"]
            for comparison in comparisons
            if comparison["panel_trend"]["status"] != "ok"
        ]
        if trend_blockers:
            for comparison in comparisons:
                comparison["panel_trend"]["holm_family"] = "panel_trend"
            trend_family = {
                "status": "blocked",
                "adjustment": "holm",
                "scope": "within_slice",
                "estimand": "ordinal_panel_stratum_trend",
                "members": list(comparison_methods),
                "blocked_members": trend_blockers,
                "raw_p_values": None,
                "adjusted_p_values": None,
            }
        else:
            trend_raw = {
                comparison["method"]: comparison["panel_trend"]["p_value"]
                for comparison in comparisons
            }
            trend_adjusted = holm_adjust(trend_raw)
            for comparison in comparisons:
                comparison["panel_trend"]["p_value_holm"] = trend_adjusted[
                    comparison["method"]
                ]
                comparison["panel_trend"]["holm_family"] = "panel_trend"
            trend_family = {
                "status": "ok",
                "adjustment": "holm",
                "scope": "within_slice",
                "estimand": "ordinal_panel_stratum_trend",
                "members": list(comparison_methods),
                "raw_p_values": trend_raw,
                "adjusted_p_values": trend_adjusted,
            }

        slice_results.append(
            {
                "backbone": backbone,
                "seeds": seeds,
                "seed_count": len(seeds),
                "budget_type": budget_type,
                "budget_value": budget_value,
                "split": split,
                "dataset_manifest_hash": next(iter(manifest_hashes)),
                "metric_config_hash": next(iter(config_hashes)),
                "metric_version": next(iter(metric_versions)),
                "experiment_git_commit": next(iter(git_commits)),
                "reference": reference,
                "case_count": len(case_ids),
                "case_ids": sorted(case_ids),
                "doi_count": len(doi_ids),
                "dois": doi_ids,
                "comparison_families": {
                    "overall": overall_family,
                    "panel_trend": trend_family,
                },
                "comparisons": comparisons,
                "rankings": _ranking_analysis(
                    rows,
                    selected_methods,
                    primary_metric=metric,
                    second_judge_metric=second_judge_metric,
                    seed=seed,
                    bootstrap_resamples=bootstrap_resamples,
                ),
            }
        )

    config = {
        "reference": reference,
        "methods": comparison_methods,
        "metric": metric,
        "panel_scope": panel_scope,
        "second_judge_metric": second_judge_metric,
        "seed": seed,
        "bootstrap_resamples": bootstrap_resamples,
        "monte_carlo_permutations": monte_carlo_permutations,
        "exact_max_n": exact_max_n,
        "sampling_unit": "doi",
        "aggregation_order": ["seed_mean", "task_mean", "doi_cluster"],
        "panel_strata": list(_PANEL_STRATA),
        "panel_trend": {
            "statistic": "ordinal_stratum_slope",
            "alternative": "increasing",
            "permutation": "doi_cluster_sign_flip",
            "missing_strata": "blocked",
        },
        "multiple_comparisons": {
            "adjustment": "holm",
            "scope": "within_slice",
            "families": ["overall", "panel_trend"],
        },
        "kendall_uncertainty": {
            "method": "doi_cluster_bootstrap",
            "resamples": bootstrap_resamples,
        },
    }
    if c5_provenance is not None:
        config["c5_frozen_decision"] = {
            "point_threshold": _C5_POINT_THRESHOLD,
            "point_comparison": ">=",
            "ci_lower_threshold": _C5_CI_LOWER_THRESHOLD,
            "ci_lower_comparison": ">",
            "required_scope": "every_backbone_slice",
            "programmatic_correctness_privileged": True,
        }
    if normalized_trajectory is not None:
        config["trajectory_threshold"] = normalized_trajectory
    commit, dirty = _git_provenance()
    if (normalized_trajectory is not None or c5_provenance is not None) and dirty:
        raise StatisticsError(
            "C3/C5 confirmatory analysis refuses a dirty analysis worktree"
        )
    c5_slice_decisions = [
        {
            "backbone": item["backbone"],
            **item["rankings"]["c5_decision"],
        }
        for item in slice_results
        if item["rankings"].get("c5_decision") is not None
    ]
    c5_decision = None
    if c5_provenance is not None:
        all_pass = bool(c5_slice_decisions) and all(
            item["status"] == "pass" for item in c5_slice_decisions
        )
        any_not_estimable = any(
            item["status"] == "not_estimable" for item in c5_slice_decisions
        )
        c5_decision = {
            "status": (
                "pass"
                if all_pass
                else ("not_estimable" if any_not_estimable else "fail")
            ),
            "concordance_claim_permitted": all_pass,
            "required_scope": "every_backbone_slice",
            "slice_decisions": c5_slice_decisions,
            "programmatic_correctness_privileged": True,
            "failure_behavior": (
                "Report judge rankings separately; do not make a concordance "
                "or correctness claim from visual judges."
            ),
        }
    right_censored_rmst = (
        _trajectory_analysis(
            summary,
            selected_rows,
            methods=selected_methods,
            threshold=normalized_trajectory,
        )
        if normalized_trajectory is not None
        else {
            "status": "not_implemented",
            "reason": (
                "Run summaries do not contain per-render threshold-crossing "
                "events and censoring times required for provenance-safe RMST."
            ),
            "required_schema": [
                "threshold",
                "event_observed",
                "event_render_or_time",
                "censor_render_or_time",
            ],
        }
    )
    output = {
        "analysis_version": _ANALYSIS_VERSION,
        "generated_at": utc_now(),
        "input_summary_path": str(summary.path),
        "input_summary_hash": summary.summary_hash,
        "analysis_config": config,
        "analysis_config_hash": sha256_json(config),
        "code_git_commit": commit,
        "code_git_dirty": dirty,
        "c5_provenance": c5_provenance,
        "c5_decision": c5_decision,
        "right_censored_rmst": right_censored_rmst,
        "slices": slice_results,
    }
    output["analysis_hash"] = sha256_json(output)
    return output


def _load_analysis_artifact(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        analysis = json.loads(
            resolved.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except OSError as exc:
        raise StatisticsError(
            f"Cannot read analysis artifact {resolved}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise StatisticsError(f"Invalid analysis JSON: {exc}") from exc
    if not isinstance(analysis, dict):
        raise StatisticsError("Analysis artifact must be an object")
    if analysis.get("analysis_version") not in _SUPPORTED_ANALYSIS_VERSIONS:
        raise StatisticsError(
            "Holm family requires a supported analysis_version: "
            f"{sorted(_SUPPORTED_ANALYSIS_VERSIONS)}"
        )
    recorded_hash = analysis.get("analysis_hash")
    if not isinstance(recorded_hash, str) or not _SHA256_RE.fullmatch(
        recorded_hash
    ):
        raise StatisticsError("Analysis artifact has no valid analysis_hash")
    unhashed = dict(analysis)
    unhashed.pop("analysis_hash", None)
    if sha256_json(unhashed) != recorded_hash:
        raise StatisticsError("Analysis artifact failed its integrity hash")
    if analysis.get("code_git_dirty") is not False:
        raise StatisticsError(
            "Holm family refuses analysis produced from a dirty worktree"
        )
    analysis_commit = _required_string(analysis, "code_git_commit")
    if not re.fullmatch(r"[0-9a-f]{7,64}", analysis_commit):
        raise StatisticsError("Analysis code_git_commit is invalid")
    config = analysis.get("analysis_config")
    if not isinstance(config, Mapping):
        raise StatisticsError("Analysis artifact lacks analysis_config")
    config_hash = analysis.get("analysis_config_hash")
    if (
        not isinstance(config_hash, str)
        or not _SHA256_RE.fullmatch(config_hash)
        or sha256_json(config) != config_hash
    ):
        raise StatisticsError("Analysis artifact has invalid analysis_config_hash")
    if not isinstance(analysis.get("slices"), list):
        raise StatisticsError("Analysis artifact lacks slices")
    summary_path_raw = _required_string(analysis, "input_summary_path")
    summary_path = Path(summary_path_raw).expanduser()
    if not summary_path.is_absolute():
        summary_path = resolved.parent / summary_path
    summary = load_provenance_summary(summary_path)
    if analysis.get("input_summary_hash") != summary.summary_hash:
        raise StatisticsError(
            "Analysis input_summary_hash disagrees with source summary"
        )
    try:
        recomputed = analyze_summary(
            summary,
            reference=_required_string(config, "reference"),
            methods=config.get("methods"),
            metric=_required_string(config, "metric"),
            panel_scope=str(config.get("panel_scope", "all")),
            second_judge_metric=config.get("second_judge_metric"),
            trajectory_threshold=config.get("trajectory_threshold"),
            seed=config.get("seed"),
            bootstrap_resamples=config.get("bootstrap_resamples"),
            monte_carlo_permutations=config.get(
                "monte_carlo_permutations"
            ),
            exact_max_n=config.get("exact_max_n"),
        )
    except (TypeError, ValueError) as exc:
        raise StatisticsError(
            f"Analysis config cannot be recomputed: {exc}"
        ) from exc
    if recomputed["analysis_config"] != config:
        raise StatisticsError(
            "Analysis config does not match canonical recomputation"
        )
    if recomputed["slices"] != analysis["slices"]:
        raise StatisticsError(
            "Analysis slices do not match source-summary recomputation"
        )
    if recomputed["right_censored_rmst"] != analysis.get(
        "right_censored_rmst"
    ):
        raise StatisticsError(
            "Analysis RMST declaration does not match recomputation"
        )
    return analysis


def _family_member_slice(
    analysis: Mapping[str, Any],
    member: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    config = analysis["analysis_config"]
    expected_metric = _required_string(member, "metric")
    expected_method = _required_string(member, "method")
    expected_reference = _required_string(member, "reference")
    expected_panel_scope = _required_string(member, "panel_scope")
    if config.get("metric") != expected_metric:
        raise StatisticsError(
            f"Family member metric disagrees with analysis: {expected_metric}"
        )
    methods = config.get("methods")
    if not isinstance(methods, list) or expected_method not in methods:
        raise StatisticsError(
            f"Family method is absent from analysis: {expected_method}"
        )
    if config.get("reference") != expected_reference:
        raise StatisticsError(
            f"Family reference disagrees with analysis: {expected_reference}"
        )
    if config.get("panel_scope", "all") != expected_panel_scope:
        raise StatisticsError(
            "Family panel_scope disagrees with analysis: "
            f"{expected_panel_scope}"
        )
    backbone = _required_string(member, "backbone")
    budget_type = _required_string(member, "budget_type")
    budget_value = _finite_number(
        member.get("budget_value"),
        f"{member.get('name')}.budget_value",
    )
    split = _required_string(member, "split")
    matches = []
    for index, slice_result in enumerate(analysis["slices"]):
        if not isinstance(slice_result, Mapping):
            raise StatisticsError(
                f"Analysis slice {index} must be an object"
            )
        slice_budget = _finite_number(
            slice_result.get("budget_value"),
            f"analysis.slices[{index}].budget_value",
        )
        if (
            slice_result.get("backbone") == backbone
            and slice_result.get("budget_type") == budget_type
            and slice_budget == budget_value
            and slice_result.get("split") == split
        ):
            matches.append(slice_result)
    if len(matches) != 1:
        raise StatisticsError(
            f"Family member must select exactly one analysis slice; "
            f"found={len(matches)}"
        )
    comparisons = matches[0].get("comparisons")
    if not isinstance(comparisons, list):
        raise StatisticsError("Selected analysis slice lacks comparisons")
    comparison_matches = [
        comparison
        for comparison in comparisons
        if isinstance(comparison, Mapping)
        and comparison.get("method") == expected_method
    ]
    if len(comparison_matches) != 1:
        raise StatisticsError(
            "Family member must select exactly one method comparison"
        )
    return matches[0], comparison_matches[0]


def build_holm_family(manifest_path: Path) -> Dict[str, Any]:
    resolved_manifest = manifest_path.expanduser().resolve()
    try:
        manifest = json.loads(
            resolved_manifest.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except OSError as exc:
        raise StatisticsError(
            f"Cannot read Holm family manifest {resolved_manifest}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise StatisticsError(
            f"Invalid Holm family manifest JSON: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise StatisticsError("Holm family manifest must be an object")
    if manifest.get("schema_version") != _HOLM_FAMILY_VERSION:
        raise StatisticsError(
            f"Unsupported Holm family schema_version: "
            f"{manifest.get('schema_version')!r}"
        )
    family_name = _required_string(manifest, "family_name")
    raw_members = manifest.get("members")
    if not isinstance(raw_members, list) or len(raw_members) < 2:
        raise StatisticsError("Holm family requires at least two members")
    code_commit, code_dirty = _git_provenance()
    if code_dirty:
        raise StatisticsError(
            "Holm family must be produced from a clean worktree"
        )

    names: set[str] = set()
    selectors: set[tuple[Any, ...]] = set()
    materialized: list[Dict[str, Any]] = []
    common_provenance: Optional[Dict[str, Any]] = None
    analysis_code_commits: set[str] = set()
    summary_hashes: set[str] = set()
    raw_p_values: Dict[str, float] = {}
    analysis_cache: Dict[Path, Dict[str, Any]] = {}
    for raw_member in raw_members:
        if not isinstance(raw_member, Mapping):
            raise StatisticsError("Every Holm family member must be an object")
        member = dict(raw_member)
        name = _required_string(member, "name")
        if name in names:
            raise StatisticsError(f"Duplicate Holm family member: {name}")
        names.add(name)
        raw_path = _required_string(member, "analysis_path")
        analysis_path = Path(raw_path).expanduser()
        if not analysis_path.is_absolute():
            analysis_path = resolved_manifest.parent / analysis_path
        analysis_path = analysis_path.resolve()
        analysis = analysis_cache.get(analysis_path)
        if analysis is None:
            analysis = _load_analysis_artifact(analysis_path)
            analysis_cache[analysis_path] = analysis
        expected_analysis_hash = _required_string(
            member,
            "analysis_hash",
        )
        if analysis["analysis_hash"] != expected_analysis_hash:
            raise StatisticsError(
                f"Family member analysis_hash mismatch: {name}"
            )
        selected_slice, comparison = _family_member_slice(analysis, member)
        selector = (
            analysis["input_summary_hash"],
            member["metric"],
            member["panel_scope"],
            member["backbone"],
            member["budget_type"],
            float(member["budget_value"]),
            member["split"],
            member["reference"],
            member["method"],
        )
        if selector in selectors:
            raise StatisticsError(
                f"Duplicate Holm hypothesis selector: {name}"
            )
        selectors.add(selector)
        permutation = comparison.get("permutation")
        if not isinstance(permutation, Mapping):
            raise StatisticsError(
                f"Family comparison lacks permutation result: {name}"
            )
        p_value = _finite_number(
            permutation.get("p_value"),
            f"{name}.p_value",
        )
        if not 0.0 <= p_value <= 1.0:
            raise StatisticsError(f"{name}.p_value must be in [0,1]")
        provenance = {
            "dataset_manifest_hash": selected_slice.get(
                "dataset_manifest_hash"
            ),
            "metric_config_hash": selected_slice.get("metric_config_hash"),
            "metric_version": selected_slice.get("metric_version"),
            "experiment_git_commit": selected_slice.get(
                "experiment_git_commit"
            ),
            "budget_type": selected_slice.get("budget_type"),
            "budget_value": selected_slice.get("budget_value"),
            "split": selected_slice.get("split"),
        }
        if common_provenance is None:
            common_provenance = provenance
        elif provenance != common_provenance:
            raise StatisticsError(
                "Holm family members mix experiment provenance"
            )
        analysis_code_commit = _required_string(
            analysis,
            "code_git_commit",
        )
        analysis_code_commits.add(analysis_code_commit)
        summary_hash = _required_string(analysis, "input_summary_hash")
        if not _SHA256_RE.fullmatch(summary_hash):
            raise StatisticsError("Analysis input_summary_hash is invalid")
        summary_hashes.add(summary_hash)
        raw_p_values[name] = p_value
        materialized.append(
            {
                "name": name,
                "analysis_path": str(analysis_path.resolve()),
                "analysis_hash": analysis["analysis_hash"],
                "metric": member["metric"],
                "panel_scope": member["panel_scope"],
                "backbone": member["backbone"],
                "method": member["method"],
                "reference": member["reference"],
                "raw_p_value": p_value,
            }
        )
    if len(analysis_code_commits) != 1:
        raise StatisticsError("Holm family mixes analysis code commits")
    if len(summary_hashes) != 1:
        raise StatisticsError("Holm family mixes input summaries")

    adjusted = holm_adjust(raw_p_values)
    for member in materialized:
        member["adjusted_p_value"] = adjusted[member["name"]]
    config = {
        "family_name": family_name,
        "adjustment": "holm",
        "members": [
            {
                key: member[key]
                for key in (
                    "name",
                    "analysis_hash",
                    "metric",
                    "panel_scope",
                    "backbone",
                    "method",
                    "reference",
                )
            }
            for member in materialized
        ],
    }
    output = {
        "family_version": _HOLM_FAMILY_VERSION,
        "generated_at": utc_now(),
        "family_name": family_name,
        "adjustment": "holm",
        "scope": "declared_cross_analysis_family",
        "manifest_path": str(resolved_manifest),
        "manifest_hash": sha256_json(manifest),
        "family_config": config,
        "family_config_hash": sha256_json(config),
        "input_summary_hash": next(iter(summary_hashes)),
        "common_experiment_provenance": common_provenance,
        "analysis_code_git_commit": next(iter(analysis_code_commits)),
        "code_git_commit": code_commit,
        "code_git_dirty": False,
        "members": materialized,
    }
    output["family_hash"] = sha256_json(output)
    return output


def write_holm_family_output(
    family: Mapping[str, Any],
    out: Path,
) -> Path:
    path = out.expanduser().resolve()
    if path.suffix.lower() != ".json":
        path = path / "holm_family.json"
    write_json_atomic(path, family)
    return path


def _analysis_csv_rows(analysis: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for slice_result in analysis["slices"]:
        ranking = slice_result["rankings"]
        for comparison in slice_result["comparisons"]:
            overall = comparison["overall"]
            permutation = comparison["permutation"]
            trend = comparison["panel_trend"]
            tau_uncertainty = ranking["kendall_tau_b_uncertainty"]
            tau_ci = tau_uncertainty.get("ci95")
            base = {
                "backbone": slice_result["backbone"],
                "seeds": ",".join(
                    str(seed) for seed in slice_result["seeds"]
                ),
                "seed_count": slice_result["seed_count"],
                "task_count": slice_result["case_count"],
                "doi_count": slice_result["doi_count"],
                "budget_type": slice_result["budget_type"],
                "budget_value": slice_result["budget_value"],
                "split": slice_result["split"],
                "dataset_manifest_hash": slice_result[
                    "dataset_manifest_hash"
                ],
                "metric_config_hash": slice_result["metric_config_hash"],
                "metric_version": slice_result["metric_version"],
                "experiment_git_commit": slice_result[
                    "experiment_git_commit"
                ],
                "reference": slice_result["reference"],
                "method": comparison["method"],
                "metric": analysis["analysis_config"]["metric"],
                "panel_scope": analysis["analysis_config"].get(
                    "panel_scope",
                    "all",
                ),
                "second_judge_metric": ranking["second_judge_metric"],
                "kendall_tau_b": ranking["kendall_tau_b"],
                "kendall_tau_b_status": tau_uncertainty["status"],
                "kendall_tau_b_ci_lower": (
                    tau_ci[0] if tau_ci is not None else None
                ),
                "kendall_tau_b_ci_upper": (
                    tau_ci[1] if tau_ci is not None else None
                ),
                "kendall_tau_b_resamples": tau_uncertainty["resamples"],
                "c5_decision_status": (
                    ranking.get("c5_decision") or {}
                ).get("status"),
                "c5_concordance_claim_permitted": (
                    ranking.get("c5_decision") or {}
                ).get("concordance_claim_permitted"),
                "panel_trend_status": trend["status"],
                "panel_trend_statistic": trend["statistic"],
                "panel_trend_p_value": trend["p_value"],
                "panel_trend_p_value_holm": trend["p_value_holm"],
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
                    "p_value_holm": permutation["p_value_holm"],
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
                        "p_value_holm": None,
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
