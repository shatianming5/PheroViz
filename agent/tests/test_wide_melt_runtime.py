from __future__ import annotations

import matplotlib
import pandas as pd

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from app.evaluation import evaluate_figure
from app.services.single_chain_runner import _materialize_sealed_wide_melt


def _wide_melt_intent() -> dict[str, object]:
    return {
        "binding_mode": "wide_melt",
        "wide_melt": {
            "group_column": "__wide_group__",
            "value_column": "__wide_value__",
            "source_value_columns": ["WT", "KO"],
            "dropped_index_columns": ["Replicate"],
        },
    }


def test_sealed_wide_melt_materializes_only_declared_source_columns() -> None:
    source = pd.DataFrame(
        {
            "Replicate": [1, 2],
            "WT": [1.0, 1.1],
            "KO": [2.0, 2.1],
        }
    )

    materialized, binding = _materialize_sealed_wide_melt(
        source,
        _wide_melt_intent(),
    )

    assert materialized.to_dict(orient="list") == {
        "__wide_group__": ["WT", "WT", "KO", "KO"],
        "__wide_value__": [1.0, 1.1, 2.0, 2.1],
    }
    assert binding == {
        "mode": "wide_melt",
        "implementation": "sealed-wide-melt-v1",
        "source_rows": 2,
        "source_columns": ["Replicate", "WT", "KO"],
        "source_value_columns": ["WT", "KO"],
        "dropped_index_columns": ["Replicate"],
        "materialized_rows": 4,
        "materialized_columns": ["__wide_group__", "__wide_value__"],
    }
    assert source.columns.tolist() == ["Replicate", "WT", "KO"]


def test_programmatic_scatter_maps_jittered_categorical_positions() -> None:
    source = pd.DataFrame(
        {
            "__wide_group__": ["WT", "WT", "KO", "KO"],
            "__wide_value__": [1.0, 1.1, 2.0, 2.1],
        }
    )
    expectation = {
        "schema_version": "1.1.0",
        "panels": [
            {
                "panel_id": "wide",
                "axis_index": 0,
                "series": [
                    {
                        "series_id": "__wide_value__",
                        "kind": "scatter",
                        "x": "__wide_group__",
                        "y": "__wide_value__",
                    }
                ],
            }
        ],
        "panel_groups": [],
    }
    figure, axis = plt.subplots()
    axis.set_gid("wide")
    axis.set_xticks([0, 1], ["WT", "KO"])
    collection = axis.scatter([-0.1, 0.1, 0.9, 1.1], source["__wide_value__"])
    collection.set_gid("__wide_value__")
    try:
        result = evaluate_figure(figure, source, expectation)
    finally:
        plt.close(figure)

    assert result.fidelity.ratio == 1.0
    assert result.manifest.axes[0].series[0].series_id == "__wide_value__"


def test_wide_melt_virtual_series_id_supports_palette_cohesion() -> None:
    source = pd.DataFrame(
        {
            "__wide_group__": ["WT", "WT", "KO", "KO"],
            "__wide_value__": [1.0, 1.1, 2.0, 2.1],
        }
    )
    expectation = {
        "schema_version": "1.1.0",
        "panels": [
            {
                "panel_id": panel_id,
                "axis_index": index,
                "series": [
                    {
                        "series_id": "__wide_value__",
                        "kind": "scatter",
                        "x": "__wide_group__",
                        "y": "__wide_value__",
                    }
                ],
            }
            for index, panel_id in enumerate(("left", "right"))
        ],
        "panel_groups": [
            {
                "group_id": "wide-melt-shared-series",
                "panels": ["left", "right"],
                "series": ["__wide_value__"],
                "checks": {"palette_consistent": True},
            }
        ],
    }
    figure, axes = plt.subplots(1, 2)
    for panel_id, axis in zip(("left", "right"), axes):
        axis.set_gid(panel_id)
        axis.set_xticks([0, 1], ["WT", "KO"])
        collection = axis.scatter(
            [-0.1, 0.1, 0.9, 1.1],
            source["__wide_value__"],
            color="#1f77b4",
        )
        collection.set_gid("__wide_value__")
    try:
        result = evaluate_figure(figure, source, expectation)
    finally:
        plt.close(figure)

    assert result.fidelity.ratio == 1.0
    assert result.cohesion.ratio == 1.0
