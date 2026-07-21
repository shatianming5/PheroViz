#!/usr/bin/env python3
"""Recompute the provisional C2 P5+ DOI-cluster power plan.

The script intentionally distinguishes raw VLM score variance from the variance
of the *paired DOI-level method difference*.  Only the latter belongs in the
paired C2 power calculation.  Current P5+ C5 data contain one DOI, so they do
not estimate that quantity; the all-renderable C5 calculation is a provisional
proxy, not evidence that P5+ has the same variance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

from scipy import __version__ as scipy_version
from scipy.stats import nct, norm, t
from statsmodels import __version__ as statsmodels_version
from statsmodels.stats.power import TTestPower


ROOT = Path(__file__).resolve().parents[1]
MERGED_ROOT = ROOT / "agent/experiments/preflight/provenance_inputs/c5/merged"
METRICS = (
    "metric.visual_form.claude-sonnet-4.6",
    "metric.visual_form.gemini-3.5-flash",
)
REFERENCE = "flat_iterative"
CONTRASTS = ("best_of_n", "pheroviz_full")
DELTA = 0.10
TARGET_POWER = 0.80
PLANNING_PAIRED_SD = 0.11


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def two_sided_paired_t_power(
    n_doi: int,
    *,
    delta: float,
    paired_sd: float,
    alpha: float,
) -> float:
    """Noncentral-t power for a paired DOI-cluster mean-difference test."""

    if n_doi < 2:
        raise ValueError("n_doi must be at least 2")
    df = n_doi - 1
    noncentrality = delta * math.sqrt(n_doi) / paired_sd
    critical = float(t.ppf(1.0 - alpha / 2.0, df))
    return float(
        nct.sf(critical, df, noncentrality)
        + nct.cdf(-critical, df, noncentrality)
    )


def required_n(
    *,
    delta: float,
    paired_sd: float,
    alpha: float,
    target_power: float,
) -> tuple[int, float]:
    for n_doi in range(2, 10_000):
        observed_power = two_sided_paired_t_power(
            n_doi,
            delta=delta,
            paired_sd=paired_sd,
            alpha=alpha,
        )
        if observed_power >= target_power:
            return n_doi, observed_power
    raise RuntimeError("No sample size found")


def normal_approximation_n(
    *,
    delta: float,
    paired_sd: float,
    alpha: float,
    target_power: float,
) -> float:
    return float(
        (
            (
                norm.ppf(1.0 - alpha / 2.0)
                + norm.ppf(target_power)
            )
            * paired_sd
            / delta
        )
        ** 2
    )


def _load_rows() -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    rows_by_tier: dict[str, list[dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    for tier in ("frontier", "mid", "open"):
        path = MERGED_ROOT / f"{tier}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("runs")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"Invalid C5 merged rows: {path}")
        rows_by_tier[tier] = rows
        hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    return rows_by_tier, hashes


def _paired_doi_sds(
    rows_by_tier: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for tier, rows in rows_by_tier.items():
        for metric in METRICS:
            values: dict[tuple[str, str], list[float]] = {}
            for row in rows:
                doi = row.get("doi")
                method = row.get("method")
                score = row.get(metric)
                if not isinstance(doi, str) or not isinstance(method, str):
                    raise ValueError(f"Missing DOI or method in {tier}")
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    raise ValueError(f"Missing {metric} in {tier}")
                values.setdefault((doi, method), []).append(float(score))
            dois = sorted({doi for doi, _ in values})
            for contrast in CONTRASTS:
                differences = [
                    statistics.fmean(values[(doi, contrast)])
                    - statistics.fmean(values[(doi, REFERENCE)])
                    for doi in dois
                ]
                records.append(
                    {
                        "tier": tier,
                        "metric": metric,
                        "contrast": contrast,
                        "doi_count": len(differences),
                        "paired_doi_sd": statistics.stdev(differences),
                        "mean_gap": statistics.fmean(differences),
                    }
                )
    return records


def _raw_score_sds(
    rows_by_tier: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    all_rows = [
        row
        for rows in rows_by_tier.values()
        for row in rows
    ]
    for metric in METRICS:
        values = [float(row[metric]) for row in all_rows]
        result[metric] = {
            "n_render_scores": len(values),
            "mean": statistics.fmean(values),
            "sample_sd": statistics.stdev(values),
            "population_sd": statistics.pstdev(values),
        }
    return result


def _extreme_coverage(
    rows_by_tier: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    p5_rows = [
        row
        for rows in rows_by_tier.values()
        for row in rows
        if isinstance(row.get("panel_count"), int) and row["panel_count"] >= 5
    ]
    dois = sorted({str(row["doi"]).casefold() for row in p5_rows})
    return {
        "p5plus_render_rows": len(p5_rows),
        "p5plus_unique_dois": len(dois),
        "p5plus_dois": dois,
        "status": (
            "not_estimable"
            if len(dois) < 2
            else "enough_dois_for_a_sample_sd_but_not_necessarily_stable"
        ),
        "reason": (
            "A DOI-cluster paired-difference variance requires at least two "
            "independent DOI clusters; the current C5 P5+ slice has one."
        ),
    }


def _sample_size_record(paired_sd: float, alpha: float) -> dict[str, float | int]:
    n_doi, exact_power = required_n(
        delta=DELTA,
        paired_sd=paired_sd,
        alpha=alpha,
        target_power=TARGET_POWER,
    )
    solved = float(
        TTestPower().solve_power(
            effect_size=DELTA / paired_sd,
            alpha=alpha,
            power=TARGET_POWER,
            alternative="two-sided",
        )
    )
    if math.ceil(solved) != n_doi:
        raise RuntimeError(
            f"SciPy/statmodels cross-check disagrees: scipy={n_doi}, "
            f"statsmodels={solved}"
        )
    return {
        "paired_sd": paired_sd,
        "alpha_two_sided": alpha,
        "normal_approximation_n": normal_approximation_n(
            delta=DELTA,
            paired_sd=paired_sd,
            alpha=alpha,
            target_power=TARGET_POWER,
        ),
        "noncentral_t_required_n_doi": n_doi,
        "noncentral_t_power_at_required_n": exact_power,
        "statsmodels_continuous_n": solved,
        "statsmodels_ceiling_n": math.ceil(solved),
    }


def build_result() -> dict[str, Any]:
    rows_by_tier, input_hashes = _load_rows()
    raw_sds = _raw_score_sds(rows_by_tier)
    paired_records = _paired_doi_sds(rows_by_tier)
    paired_sds = [float(record["paired_doi_sd"]) for record in paired_records]
    paired_median = statistics.median(paired_sds)
    paired_rms = math.sqrt(statistics.fmean(value * value for value in paired_sds))
    extreme = _extreme_coverage(rows_by_tier)

    unadjusted = _sample_size_record(PLANNING_PAIRED_SD, 0.05)
    holm_first = _sample_size_record(PLANNING_PAIRED_SD, 0.025)
    raw_sd_025 = _sample_size_record(0.25, 0.05)
    raw_sd_025_holm = _sample_size_record(0.25, 0.025)
    favorable_window = [
        _sample_size_record(0.10, 0.05),
        _sample_size_record(0.11, 0.05),
        _sample_size_record(0.12, 0.05),
    ]
    rms_proxy = {
        "unadjusted_alpha_0.05": _sample_size_record(paired_rms, 0.05),
        "two_contrast_holm_alpha_0.025": _sample_size_record(paired_rms, 0.025),
    }
    largest_proxy_sd = max(paired_sds)
    largest_proxy = {
        "unadjusted_alpha_0.05": _sample_size_record(largest_proxy_sd, 0.05),
        "two_contrast_holm_alpha_0.025": _sample_size_record(largest_proxy_sd, 0.025),
    }

    return {
        "schema_version": "1.0",
        "scope": (
            "Prospective, provisional C2 P5+ power plan. The independent "
            "statistical unit is a DOI cluster, not a panel, seed, render, or "
            "reviewer call."
        ),
        "input_c5_merged_sha256": input_hashes,
        "extreme_p5plus_variance_status": extreme,
        "c5_crosscheck": {
            "provided_context_reference_raw_visual_form_sd": {
                "claude-sonnet-4.6": 0.25,
                "gemini-3.5-flash": 0.25,
            },
            "current_tracked_pooled_raw_score_sd": raw_sds,
            "interpretation": (
                "The tracked pooled values use all 513 C5 render scores and are "
                "not necessarily the same filtering/aggregation as the supplied "
                "approximately-0.25 context. Raw score SD is not a DOI-level "
                "paired method-difference SD."
            ),
            "renderable_all_strata_paired_doi_proxy": {
                "records": paired_records,
                "sd_min": min(paired_sds),
                "sd_max": max(paired_sds),
                "sd_median": paired_median,
                "sd_mean": statistics.fmean(paired_sds),
                "sd_rms": paired_rms,
                "planning_sd": PLANNING_PAIRED_SD,
                "warning": (
                    "Only six renderable DOI clusters and mixed P=1/2/3/6 strata "
                    "underlie this proxy. It is not an estimate for the P5+ "
                    "population."
                ),
            },
        },
        "assumptions": {
            "effect_delta": DELTA,
            "target_power": TARGET_POWER,
            "test": (
                "Two-sided paired-t noncentrality planning approximation on one "
                "per-DOI mean method-minus-flat_iterative gap. Production uses "
                "the repository's DOI-cluster paired sign-flip permutation plus "
                "Holm; freeze/simulate that exact analysis if a final guarantee "
                "rather than this planning approximation is required."
            ),
            "alpha_unadjusted": 0.05,
            "holm_family": {
                "planned_contrasts_vs_flat_iterative": 2,
                "worst_case_first_step_alpha": 0.025,
                "note": (
                    "Holm is less conservative after a smaller p-value passes, "
                    "but prospective assurance must cover the first step."
                ),
            },
            "programmatic_metric_status": {
                "c5_programmatic_visual_form": (
                    "Saturated at 1.0 in the supplied C5 context, so it does not "
                    "supply a usable variance or detectable delta for power."
                ),
                "c2_extreme_data_fidelity_and_series_cohesion": (
                    "Described as unsaturated for P5+ in the task context, but no "
                    "sealed P5+ execution data yet exist here to estimate their "
                    "DOI-level paired-difference variance."
                ),
            },
        },
        "formula": {
            "doi_gap": (
                "D_d = mean_{case,seed in DOI d}(score_method - "
                "score_flat_iterative)"
            ),
            "noncentrality": "lambda = Delta * sqrt(K) / sd(D_d)",
            "critical_value": "c = t.ppf(1 - alpha/2, K - 1)",
            "power": (
                "P(T_{K-1}(lambda) > c) + P(T_{K-1}(lambda) < -c)"
            ),
            "normal_approximation": (
                "K_approx = ((z_(1-alpha/2) + z_power) * sd(D_d) / Delta)^2"
            ),
        },
        "results": {
            "requested_k_approximately_8_to_12_interpretation": {
                "normal_approximation_for_sd_D_0.10_to_0.12": [
                    item["normal_approximation_n"] for item in favorable_window
                ],
                "finite_sample_noncentral_t_required_n": [
                    item["noncentral_t_required_n_doi"] for item in favorable_window
                ],
                "conclusion": (
                    "The often-quoted K≈8–12 window is only a normal-approximation "
                    "planning window. Exact finite-sample paired-t calculation is "
                    "K=10–14 for sd(D)=0.10–0.12; K=8 has only about 0.681 power "
                    "at sd(D)=0.10."
                ),
            },
            "single_prespecified_contrast_alpha_0.05": unadjusted,
            "two_c2_contrasts_holm_worst_case_alpha_0.025": holm_first,
            "if_only_raw_sd_0.25_were_justifiably_available": {
                "unadjusted_alpha_0.05": raw_sd_025,
                "two_contrast_holm_alpha_0.025": raw_sd_025_holm,
            },
            "c5_proxy_sensitivity_not_an_extreme_estimate": {
                "rms_paired_sd_across_12_renderable_proxy_records": rms_proxy,
                "largest_paired_sd_across_12_renderable_proxy_records": largest_proxy,
                "interpretation": (
                    "These show why K=12/15 is conditional rather than guaranteed: "
                    "higher P5+ variance implies a substantially larger DOI count."
                ),
            },
        },
        "final_recommendation": {
            "minimum_for_one_prespecified_unadjusted_contrast": 12,
            "headline_target_for_two_C2_contrasts_with_Holm": 15,
            "unit": "verified independent multi-panel DOI clusters at P>=5",
            "condition_for_12": (
                "Only if a pre-execution P5+ pilot supports sd(D_d)<=0.11. At "
                "K=12, alpha=0.05, delta=0.10, sd(D)=0.11, exact power is "
                f"{two_sided_paired_t_power(12, delta=DELTA, paired_sd=0.11, alpha=0.05):.6f}."
            ),
            "condition_for_15": (
                "For the two-comparison Holm first-step alpha=0.025 under the "
                "same sd(D)=0.11 assumption. At K=15 exact power is "
                f"{two_sided_paired_t_power(15, delta=DELTA, paired_sd=0.11, alpha=0.025):.6f}."
            ),
            "recalibration_rule": (
                "Do not claim this target is met from the current one-DOI P5+ "
                "slice. If a prospectively frozen P5+ pilot estimates sd(D_d)>0.11, "
                "recompute K before a confirmatory claim; P5+ complexity may make "
                "the needed K materially larger."
            ),
        },
        "software_crosscheck": {
            "python": sys.version.split()[0],
            "scipy": scipy_version,
            "statsmodels": statsmodels_version,
            "method": (
                "Required integer K is found with scipy.stats.nct and independently "
                "checked against statsmodels.stats.power.TTestPower.solve_power."
            ),
        },
        "reproducible_script": "files/c2_power_K_final.py",
        "reproduce": (
            "/opt/homebrew/Caskroom/miniforge/base/bin/python3 "
            "files/c2_power_K_final.py --out files/c2_power_K_final.json"
        ),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = build_result()
    result["reproducible_script_sha256"] = sha256_file(Path(__file__).resolve())
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(encoded, end="")
    else:
        args.out.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
