from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from jsonschema.exceptions import ValidationError

from app.evaluation import (
    EXPECTATION_SCHEMA_VERSION,
    FIGURE_MANIFEST_SCHEMA_VERSION,
    MetricConfig,
    NumericTolerance,
    evaluate_cohesion,
    evaluate_fidelity,
    evaluate_figure,
    extract_figure_manifest,
    validate_expectation,
    validate_figure_manifest,
    validate_metric_config,
)


def _line_expectation(
    panel_id: str = "panel-a",
    *,
    label: str = "signal",
    y_unit: str | None = "mg",
    y_scale: str | None = "linear",
) -> dict:
    panel = {
        "panel_id": panel_id,
        "axis_index": 0,
        "xlabel": "Time (s)",
        "ylabel": f"Value ({y_unit})" if y_unit else "Value",
        "x_unit": "s",
        "series": [
            {
                "series_id": label,
                "kind": "line",
                "label": label,
                "x": "x",
                "y": "y",
            }
        ],
    }
    if y_unit is not None:
        panel["y_unit"] = y_unit
    if y_scale is not None:
        panel["y_scale"] = y_scale
    return {
        "schema_version": EXPECTATION_SCHEMA_VERSION,
        "panels": [panel],
        "panel_groups": [],
    }


def _make_line_figure(
    y: list[float],
    *,
    panel_id: str = "panel-a",
    label: str = "signal",
    unit: str = "mg",
    color: str = "#1f77b4",
):
    fig, ax = plt.subplots()
    ax.set_gid(panel_id)
    ax.plot([1, 2, 3], y, label=label, color=color)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Value ({unit})")
    ax.legend()
    return fig


def _multi_panel_expectation() -> dict:
    panels = []
    for index, panel_id in enumerate(("panel-a", "panel-b")):
        panels.append(
            {
                "panel_id": panel_id,
                "axis_index": index,
                "ylabel": "Value (mg)",
                "y_unit": "mg",
                "y_scale": "linear",
                "series": [
                    {
                        "series_id": "signal",
                        "kind": "line",
                        "label": "signal",
                        "x": "x",
                        "y": "y",
                    }
                ],
            }
        )
    return {
        "schema_version": EXPECTATION_SCHEMA_VERSION,
        "panels": panels,
        "panel_groups": [
            {
                "group_id": "shared-measurement",
                "panels": ["panel-a", "panel-b"],
                "series": ["signal"],
                "checks": {
                    "shared_y_scale": True,
                    "shared_y_unit": True,
                    "legend_deduplicated": True,
                    "palette_consistent": True,
                },
            }
        ],
    }


def _make_multi_panel_figure(
    *,
    colors: tuple[str, str] = ("#1f77b4", "#1f77b4"),
    duplicate_legends: bool = False,
):
    fig, axes = plt.subplots(1, 2)
    lines = []
    for axis, panel_id, color in zip(axes, ("panel-a", "panel-b"), colors):
        axis.set_gid(panel_id)
        (line,) = axis.plot([1, 2, 3], [10, 20, 30], label="signal", color=color)
        axis.set_ylabel("Value (mg)")
        lines.append(line)
        if duplicate_legends:
            axis.legend()
    if not duplicate_legends:
        fig.legend([lines[0]], ["signal"])
    return fig


@pytest.fixture
def source_df() -> pd.DataFrame:
    return pd.DataFrame({"x": [1, 2, 3], "y": [10.0, 20.0, 30.0]})


def test_full_match_returns_auditable_counts(source_df):
    fig = _make_line_figure([10.0, 20.0, 30.0])
    try:
        result = evaluate_figure(fig, source_df, _line_expectation())
    finally:
        plt.close(fig)

    assert result.fidelity.status == "pass"
    assert result.fidelity.ratio == 1.0
    numeric = result.fidelity.checks["numeric_match"]
    assert (numeric.numerator, numeric.denominator, numeric.ratio) == (6, 6, 1.0)
    assert numeric.details["tolerance"] == {"absolute": 1e-8, "relative": 1e-6}
    assert result.fidelity.checks["series_coverage"].ratio == 1.0
    assert result.fidelity.checks["label_match"].ratio == 1.0
    assert result.fidelity.checks["unit_match"].ratio == 1.0
    assert result.fidelity.checks["scale_match"].ratio == 1.0
    assert result.cohesion.status == "na"
    payload = result.to_dict()
    assert payload["fidelity"]["aggregation"] == "sum_applicable_check_counts"
    assert payload["fidelity"]["denominator"] == sum(
        check["denominator"]
        for check in payload["fidelity"]["checks"].values()
        if check["applicable"]
    )


def test_numeric_error_is_located_without_image_inference(source_df):
    fig = _make_line_figure([10.0, 999.0, 30.0])
    try:
        result = evaluate_figure(fig, source_df, _line_expectation())
    finally:
        plt.close(fig)

    numeric = result.fidelity.checks["numeric_match"]
    assert (numeric.numerator, numeric.denominator) == (4, 6)
    assert numeric.ratio == pytest.approx(2 / 3)
    missing = [item for item in numeric.mismatches if item.code == "numeric_point_missing"]
    assert len(missing) == 1
    assert missing[0].location == {"expected_point_index": 1}
    assert missing[0].expected == {"x": 2, "y": 20.0}


def test_large_exact_line_uses_scalable_point_matching():
    count = 8_000
    source = pd.DataFrame(
        {
            "x": np.arange(count, dtype=float),
            "y": np.linspace(0.0, 1.0, count),
        }
    )
    fig, ax = plt.subplots()
    ax.set_gid("panel-a")
    ax.plot(source["x"], source["y"], label="signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value (mg)")
    ax.legend()
    try:
        result = evaluate_figure(fig, source, _line_expectation())
    finally:
        plt.close(fig)

    numeric = result.fidelity.checks["numeric_match"]
    assert (numeric.numerator, numeric.denominator) == (
        count * 2,
        count * 2,
    )
    assert numeric.ratio == 1.0


def test_missing_series_reduces_coverage_and_numeric_match():
    source = pd.DataFrame(
        {
            "x": [1, 2, 1, 2],
            "group": ["A", "A", "B", "B"],
            "y": [10.0, 20.0, 30.0, 40.0],
        }
    )
    fig, ax = plt.subplots()
    ax.set_gid("panel-a")
    subset = source[source["group"] == "A"]
    ax.plot(subset["x"], subset["y"], label="A")
    expectation = {
        "schema_version": EXPECTATION_SCHEMA_VERSION,
        "panels": [
            {
                "panel_id": "panel-a",
                "axis_index": 0,
                "series": [
                    {
                        "series_id": group,
                        "kind": "line",
                        "label": group,
                        "x": "x",
                        "y": "y",
                        "where": {"group": group},
                    }
                    for group in ("A", "B")
                ],
            }
        ],
        "panel_groups": [],
    }
    try:
        result = evaluate_figure(fig, source, expectation)
    finally:
        plt.close(fig)

    coverage = result.fidelity.checks["series_coverage"]
    assert (coverage.numerator, coverage.denominator, coverage.ratio) == (1, 2, 0.5)
    assert any(item.code == "series_missing" and item.series_id == "B" for item in coverage.mismatches)
    numeric = result.fidelity.checks["numeric_match"]
    assert (numeric.numerator, numeric.denominator) == (4, 6)


def test_unbound_data_axis_reduces_fidelity_purity(source_df):
    fig = _make_line_figure([10.0, 20.0, 30.0])
    extra = fig.add_axes([0.65, 0.6, 0.25, 0.25])
    extra.set_gid("unexpected-panel")
    extra.plot([1, 2, 3], [99.0, 98.0, 97.0], label="unexpected")
    try:
        result = evaluate_figure(fig, source_df, _line_expectation())
    finally:
        plt.close(fig)

    purity = result.fidelity.checks["series_purity"]
    assert purity.ratio == 0.5
    assert any(
        mismatch.code == "series_unexpected"
        and mismatch.panel_id == "unexpected-panel"
        for mismatch in purity.mismatches
    )
    numeric = result.fidelity.checks["numeric_match"]
    assert (numeric.numerator, numeric.denominator) == (6, 9)


def test_log_linear_scale_mismatch_is_explicit(source_df):
    fig = _make_line_figure([10.0, 20.0, 30.0])
    expectation = _line_expectation(y_scale="log")
    try:
        result = evaluate_figure(fig, source_df, expectation)
    finally:
        plt.close(fig)

    scale = result.fidelity.checks["scale_match"]
    assert (scale.numerator, scale.denominator, scale.ratio) == (0, 1, 0.0)
    assert scale.mismatches[0].code == "axis_scale_mismatch"
    assert scale.mismatches[0].expected == "log"
    assert scale.mismatches[0].observed == "linear"


def test_unit_mismatch_is_explicit(source_df):
    fig = _make_line_figure([10.0, 20.0, 30.0], unit="kg")
    try:
        result = evaluate_figure(fig, source_df, _line_expectation(y_unit="mg"))
    finally:
        plt.close(fig)

    unit = result.fidelity.checks["unit_match"]
    assert (unit.numerator, unit.denominator, unit.ratio) == (1, 2, 0.5)
    mismatch = next(item for item in unit.mismatches if item.location["dimension"] == "y")
    assert mismatch.expected == "mg"
    assert mismatch.observed == "kg"


def test_declared_tolerance_is_applied_and_returned(source_df):
    fig = _make_line_figure([10.005, 20.0, 30.0])
    config = MetricConfig(
        numeric_tolerance=NumericTolerance(absolute=0.01, relative=0.0)
    )
    try:
        result = evaluate_figure(fig, source_df, _line_expectation(), config)
    finally:
        plt.close(fig)

    numeric = result.fidelity.checks["numeric_match"]
    assert numeric.ratio == 1.0
    assert numeric.details["tolerance"] == {"absolute": 0.01, "relative": 0.0}
    validate_metric_config(config)


def test_categorical_bar_matches_source_table():
    source = pd.DataFrame({"category": ["A", "B"], "value": [1.0, 2.0]})
    fig, ax = plt.subplots()
    ax.set_gid("panel-bar")
    ax.bar(source["category"], source["value"], label="bars")
    expectation = {
        "schema_version": EXPECTATION_SCHEMA_VERSION,
        "panels": [
            {
                "panel_id": "panel-bar",
                "axis_index": 0,
                "series": [
                    {
                        "series_id": "bars",
                        "kind": "bar",
                        "label": "bars",
                        "x": "category",
                        "value": "value",
                    }
                ],
            }
        ],
        "panel_groups": [],
    }
    try:
        result = evaluate_figure(fig, source, expectation)
    finally:
        plt.close(fig)

    assert result.fidelity.checks["numeric_match"].ratio == 1.0
    assert result.fidelity.checks["series_coverage"].ratio == 1.0


def test_scatter_matches_source_table(source_df):
    fig, ax = plt.subplots()
    ax.set_gid("panel-scatter")
    ax.scatter(source_df["x"], source_df["y"], label="points")
    expectation = {
        "schema_version": EXPECTATION_SCHEMA_VERSION,
        "panels": [
            {
                "panel_id": "panel-scatter",
                "axis_index": 0,
                "series": [
                    {
                        "series_id": "points",
                        "kind": "scatter",
                        "label": "points",
                        "x": "x",
                        "y": "y",
                    }
                ],
            }
        ],
        "panel_groups": [],
    }
    try:
        result = evaluate_figure(fig, source_df, expectation)
    finally:
        plt.close(fig)

    assert result.fidelity.checks["numeric_match"].ratio == 1.0
    assert result.fidelity.checks["series_coverage"].ratio == 1.0


def test_multi_panel_cohesion_all_checks_pass(source_df):
    fig = _make_multi_panel_figure()
    try:
        result = evaluate_figure(fig, source_df, _multi_panel_expectation())
    finally:
        plt.close(fig)

    assert result.cohesion.status == "pass"
    assert (result.cohesion.numerator, result.cohesion.denominator) == (4, 4)
    assert result.cohesion.checks["shared_scale"].ratio == 1.0
    assert result.cohesion.checks["shared_unit"].ratio == 1.0
    assert result.cohesion.checks["legend_deduplication"].ratio == 1.0
    assert result.cohesion.checks["palette_mapping"].ratio == 1.0


def test_palette_mismatch_is_located(source_df):
    fig = _make_multi_panel_figure(colors=("#1f77b4", "#d62728"))
    try:
        result = evaluate_figure(fig, source_df, _multi_panel_expectation())
    finally:
        plt.close(fig)

    palette = result.cohesion.checks["palette_mapping"]
    assert (palette.numerator, palette.denominator, palette.ratio) == (0, 1, 0.0)
    assert palette.mismatches[0].code == "palette_mapping_mismatch"
    assert palette.mismatches[0].location["group_id"] == "shared-measurement"


def test_shared_axis_limit_mismatch_is_explicit(source_df):
    fig = _make_multi_panel_figure()
    fig.axes[1].set_ylim(0, 100)
    expectation = _multi_panel_expectation()
    expectation["panel_groups"][0]["checks"] = {"shared_y_limits": True}
    try:
        result = evaluate_figure(fig, source_df, expectation)
    finally:
        plt.close(fig)

    scale = result.cohesion.checks["shared_scale"]
    assert (scale.numerator, scale.denominator, scale.ratio) == (0, 1, 0.0)
    assert scale.mismatches[0].code == "shared_limits_mismatch"


def test_shared_unit_mismatch_is_explicit(source_df):
    fig = _make_multi_panel_figure()
    fig.axes[1].set_ylabel("Value (kg)")
    expectation = _multi_panel_expectation()
    expectation["panel_groups"][0]["checks"] = {"shared_y_unit": True}
    try:
        result = evaluate_figure(fig, source_df, expectation)
    finally:
        plt.close(fig)

    unit = result.cohesion.checks["shared_unit"]
    assert (unit.numerator, unit.denominator, unit.ratio) == (0, 1, 0.0)
    assert unit.mismatches[0].code == "shared_unit_mismatch"


def test_duplicate_panel_legends_fail_deduplication(source_df):
    fig = _make_multi_panel_figure(duplicate_legends=True)
    try:
        result = evaluate_figure(fig, source_df, _multi_panel_expectation())
    finally:
        plt.close(fig)

    legend = result.cohesion.checks["legend_deduplication"]
    assert (legend.numerator, legend.denominator, legend.ratio) == (0, 1, 0.0)
    assert legend.mismatches[0].observed == 2


def test_single_panel_cohesion_is_na(source_df):
    fig = _make_line_figure([10.0, 20.0, 30.0])
    try:
        result = evaluate_figure(fig, source_df, _line_expectation())
    finally:
        plt.close(fig)

    assert result.cohesion.applicable is False
    assert result.cohesion.status == "na"
    assert result.cohesion.ratio is None
    assert (result.cohesion.numerator, result.cohesion.denominator) == (0, 0)
    assert all(check.status == "na" for check in result.cohesion.checks.values())


def test_missing_rendered_panel_fails_declared_cohesion(source_df):
    fig = _make_line_figure([10.0, 20.0, 30.0], panel_id="panel-a")
    try:
        manifest = extract_figure_manifest(fig)
        result = evaluate_cohesion(manifest, _multi_panel_expectation())
    finally:
        plt.close(fig)

    assert result.applicable is True
    assert result.status == "fail"
    assert result.ratio is not None
    assert result.ratio < 1.0


def test_positional_fallback_does_not_reuse_exactly_bound_axis(source_df):
    fig = _make_line_figure(
        [10.0, 20.0, 30.0],
        panel_id="panel-b",
    )
    try:
        result = evaluate_cohesion(
            extract_figure_manifest(fig),
            _multi_panel_expectation(),
        )
    finally:
        plt.close(fig)

    assert result.applicable is True
    assert result.status == "fail"
    assert result.ratio is not None
    assert result.ratio < 1.0


def test_manifest_extracts_bar_scatter_and_image():
    fig, axes = plt.subplots(1, 3)
    for index, axis in enumerate(axes):
        axis.set_gid(f"panel-{index}")
    axes[0].bar(["A", "B"], [1.0, 2.0], label="bars", color="#1f77b4")
    axes[1].scatter([1.0, 2.0], [3.0, 4.0], label="points", color="#d62728")
    image = axes[2].imshow(np.asarray([[1.0, 2.0], [3.0, 4.0]]), cmap="viridis")
    image.set_gid("heat")
    try:
        manifest = extract_figure_manifest(fig)
        validate_figure_manifest(manifest)
    finally:
        plt.close(fig)

    by_axis = {axis.axis_id: axis for axis in manifest.axes}
    bar = next(series for series in by_axis["panel-0"].series if series.kind == "bar")
    scatter = next(series for series in by_axis["panel-1"].series if series.kind == "scatter")
    heat = next(series for series in by_axis["panel-2"].series if series.kind == "image")
    assert bar.y == [1.0, 2.0]
    assert bar.x_labels == ["A", "B"]
    assert by_axis["panel-0"].x_axis.scale == "linear"
    assert len(by_axis["panel-0"].x_axis.limits) == 2
    assert scatter.x == [1.0, 2.0]
    assert scatter.y == [3.0, 4.0]
    assert heat.value == [[1.0, 2.0], [3.0, 4.0]]
    assert heat.series_id == "heat"


def test_manifest_serialization_is_deterministic_and_json_safe():
    first = _make_line_figure([10.0, 20.0, 30.0])
    second = _make_line_figure([10.0, 20.0, 30.0])
    try:
        first_json = extract_figure_manifest(first).to_json()
        second_json = extract_figure_manifest(second).to_json()
    finally:
        plt.close(first)
        plt.close(second)

    assert first_json == second_json
    decoded = json.loads(first_json)
    assert decoded["schema_version"] == FIGURE_MANIFEST_SCHEMA_VERSION
    assert decoded["axes"][0]["axis_id"] == "panel-a"


def test_reference_lines_are_annotations_not_data_series(source_df):
    fig = _make_line_figure([10.0, 20.0, 30.0])
    axis = fig.axes[0]
    axis.axhline(15.0, color="black")
    axis.axvline(2.0, color="black")
    try:
        result = evaluate_figure(fig, source_df, _line_expectation())
    finally:
        plt.close(fig)

    panel = result.manifest.axes[0]
    assert [series.series_id for series in panel.series] == ["signal"]
    assert len(panel.annotations) == 2
    assert result.fidelity.checks["series_purity"].ratio == 1.0


def test_twin_axis_is_one_logical_panel_with_secondary_fidelity():
    source = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "left": [10.0, 20.0, 30.0],
            "right": [0.1, 1.0, 10.0],
        }
    )
    fig, primary = plt.subplots()
    primary.set_gid("panel-dual")
    secondary = primary.twinx()
    primary.plot(source["x"], source["left"], label="left", color="#1f77b4")
    secondary.plot(source["x"], source["right"], label="right", color="#d62728")
    primary.set_ylabel("Mass (mg)")
    secondary.set_ylabel("Rate (%)")
    secondary.set_yscale("log")
    expectation = {
        "schema_version": EXPECTATION_SCHEMA_VERSION,
        "panels": [
            {
                "panel_id": "panel-dual",
                "axis_index": 0,
                "y_unit": "mg",
                "y_scale": "linear",
                "secondary_y": {"y_unit": "%", "y_scale": "log"},
                "series": [
                    {
                        "series_id": "left",
                        "kind": "line",
                        "label": "left",
                        "x": "x",
                        "y": "left",
                        "y_axis": "primary",
                    },
                    {
                        "series_id": "right",
                        "kind": "line",
                        "label": "right",
                        "x": "x",
                        "y": "right",
                        "y_axis": "secondary",
                    },
                ],
            }
        ],
        "panel_groups": [],
    }
    try:
        result = evaluate_figure(fig, source, expectation)
    finally:
        plt.close(fig)

    assert [axis.role for axis in result.manifest.axes] == ["panel", "secondary"]
    assert {axis.panel_id for axis in result.manifest.axes} == {"panel-dual"}
    assert result.fidelity.status == "pass"
    assert result.cohesion.status == "na"


def test_unknown_metric_config_key_fails_closed(source_df):
    fig = _make_line_figure([10.0, 20.0, 30.0])
    try:
        with pytest.raises(ValidationError):
            evaluate_figure(
                fig,
                source_df,
                _line_expectation(),
                {"metric_version": "1.0.0", "typo_tolerance": 0.1},
            )
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["panels"].append(dict(value["panels"][0])),
        lambda value: value["panels"][0]["series"].append(
            dict(value["panels"][0]["series"][0])
        ),
        lambda value: value.update(
            {
                "panel_groups": [
                    {
                        "group_id": "bad",
                        "panels": ["panel-a", "missing"],
                        "checks": {"shared_y_scale": True},
                    }
                ]
            }
        ),
        lambda value: value["panels"][0].update(
            {"series": [{"series_id": "empty", "kind": "line"}]}
        ),
    ],
)
def test_invalid_expectation_semantics_fail_closed(mutator):
    expectation = _line_expectation()
    mutator(expectation)
    with pytest.raises(ValidationError):
        validate_expectation(expectation)
