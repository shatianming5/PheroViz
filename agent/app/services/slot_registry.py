from __future__ import annotations

from typing import Dict, List

SLOT_KEYS: List[str] = [
    # L1
    "spec.compose",
    "spec.theme_defaults",
    # L2
    "data.prepare",
    "data.aggregate",
    "data.encode",
    # L3 marks
    "marks.dispatch",
    "marks.bar.grouped",
    "marks.bar.stacked",
    "marks.bar.percent",
    "marks.bar.waterfall",
    "marks.lollipop.main",
    "marks.line.main",
    "marks.line.ci",
    "marks.line.error",
    "marks.area.fill",
    "marks.scatter.main",
    "marks.scatter.error",
    "marks.hist.main",
    "marks.hexbin.main",
    "marks.boxplot.main",
    "marks.violin.main",
    "marks.heatmap.main",
    # L3 scales
    "scales.x.kind",
    "scales.x.range",
    "scales.x.breaks",
    "scales.y_left.kind",
    "scales.y_left.range",
    "scales.y_left.breaks",
    "scales.y_right.kind",
    "scales.y_right.range",
    # colorbar
    "colorbar.apply",
    # L4 axes & layout
    "axes.ticks",
    "axes.formatter",
    "axes.labels",
    "axes.spines",
    "axes.aspect",
    "axes.tick_rotate",
    # L4 others
    "legend.apply",
    "grid.apply",
    "annot.reference_lines",
    "annot.bands",
    "annot.peak_labels",
    "annot.text_boxes",
    "annot.inset",
    "theme.fonts",
    "theme.palette",
    "theme.misc",
]

EXEC_DAG: List[str] = [
    "spec.*",
    "data.*",
    "marks.*",
    "scales.*",
    "colorbar.apply",
    "axes.*",
    "grid.apply",
    "legend.apply",
    "annot.*",
    "theme.*",
]

ALLOWED_BY_LAYER: Dict[str, List[str]] = {
    "L1": ["spec.*"],
    "L2": ["data.*"],
    "L3": ["marks.*", "scales.*", "colorbar.apply"],
    "L4": ["axes.*", "legend.apply", "grid.apply", "annot.*", "theme.*"],
}


def slot_to_jinja_var(slot_key: str) -> str:
    """Convert a dotted slot key into the scaffold's Jinja variable name."""

    return "S_" + slot_key.replace(".", "_")
