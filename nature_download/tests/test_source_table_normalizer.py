from __future__ import annotations

import pytest

from nature_download.corpus.proposals import analyze_table
from nature_download.corpus.source_table_normalizer import (
    NormalizerRejected,
    normalize_grid,
)

N = None


def test_wide_categorical_replicates_becomes_bar():
    # Title row + spacer col + offset header + ragged per-group replicates.
    grid = [
        ["Figure X", "#metric", N, N, N],
        [N, "CondA", "CondB", "CondC", "CondD"],
        [N, 1.0, 2.0, 3.0, 4.0],
        [N, 1.1, 2.1, 3.1, N],
        [N, 1.2, 2.2, N, N],
    ]
    res = normalize_grid(grid)
    assert res.orientation == "wide-categorical"
    assert res.header_row_index == 1
    assert list(res.frame.columns) == ["category", "value"]
    assert set(res.frame["category"]) == {"CondA", "CondB", "CondC", "CondD"}
    analysis = analyze_table(res.frame)
    assert analysis.chart_family == "bar"
    assert analysis.x == "category"


def test_side_by_side_blocks_keeps_first_only():
    # Two sub-experiments separated by empty columns -> keep leftmost block.
    grid = [
        ["Fig Y", N, N, N, N, N, N],
        [N, "PBS", "Prop", N, N, "WT", "KO"],
        [N, 1.0, 2.0, N, N, 5.0, 6.0],
        [N, 1.5, 2.5, N, N, 5.5, 6.5],
    ]
    res = normalize_grid(grid)
    assert res.orientation == "wide-categorical"
    assert res.dropped_side_blocks == 1
    assert set(res.frame["category"]) == {"PBS", "Prop"}
    assert "WT" not in set(res.frame["category"])


def test_long_passthrough_preserves_categorical_label_column():
    # Index col + categorical label col + numeric value col.
    grid = [
        ["Fig Z", N, "desc", N],
        [N, "Number", "Gene", "Log2FC"],
        [N, 1, "GeneA", -1.0],
        [N, 2, "GeneB", -0.9],
        [N, 3, "GeneC", -0.8],
    ]
    res = normalize_grid(grid)
    assert res.orientation == "long-passthrough"
    assert "Gene" in res.frame.columns
    assert list(res.frame["Gene"]) == ["GeneA", "GeneB", "GeneC"]
    analysis = analyze_table(res.frame)
    assert analysis.chart_family == "bar"
    assert analysis.x == "Gene"


def test_reject_duplicate_headers_multilevel_matrix():
    # Two-level "group / n=1,n=2,n=3" matrix has duplicate sub-headers.
    grid = [
        ["title", N, N, N, N, N, N],
        [N, N, N, "grp", N, N, "grp2"],
        [N, "n1", "n2", "n3", "n1", "n2", "n3"],
        ["A", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ]
    with pytest.raises(NormalizerRejected) as exc:
        normalize_grid(grid)
    assert exc.value.reason == "normalizer-duplicate-headers"


def test_reject_no_numeric_header_row():
    # Sankey / edge-list style: all-categorical, no numeric data anywhere.
    grid = [
        ["x", "node", "next_x", "next_node"],
        ["ann", "Macro", "pre", "Macro"],
        ["pre", "Macro", N, N],
    ]
    with pytest.raises(NormalizerRejected) as exc:
        normalize_grid(grid)
    assert exc.value.reason == "normalizer-no-header"


def test_reject_empty_grid():
    with pytest.raises(NormalizerRejected):
        normalize_grid([[N, N], [N, N]])
