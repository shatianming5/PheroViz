from __future__ import annotations

from typing import Any, Dict


_DEF_BAR_WIDTH = 0.8


def _choose_first(columns: Dict[str, str], predicate) -> str | None:
    for name, ctype in columns.items():
        if predicate(name, ctype):
            return name
    return None


def _categorical(columns: Dict[str, str]) -> str | None:
    return _choose_first(columns, lambda _n, ct: ct not in {"numeric"})


def _numeric(columns: Dict[str, str]) -> str | None:
    return _choose_first(columns, lambda _n, ct: ct == "numeric")


def derive_spec(intent: Dict[str, Any], data_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Derive a minimal spec from intent and data profile suitable for validation."""

    chart_family = (intent or {}).get("chart_family", "line").lower()
    intent_x = intent.get("x")
    intent_y = intent.get("y")
    intent_group = intent.get("group")

    column_types = (data_profile or {}).get("columns") or {}
    fallback_x = intent_x or _categorical(column_types) or next(iter(column_types), None)
    fallback_y = intent_y or _numeric(column_types) or next(iter(column_types), None)

    if intent_group:
        group_field = intent_group
    else:
        group_field = None
        for name, ctype in column_types.items():
            if name not in {fallback_x, fallback_y} and ctype != "numeric":
                group_field = name
                break

    overlays = [
        {
            "mark": chart_family,
            "variant": "grouped" if chart_family == "bar" else "main",
            "x": fallback_x,
            "y": fallback_y,
            "group": group_field,
            "yaxis": "left",
            "style": {"alpha": 0.9, "width": _DEF_BAR_WIDTH},
        }
    ]

    aesthetics = (intent or {}).get("aesthetics") or {}

    legend_policy = aesthetics.get("legend_policy", "best")
    if group_field and aesthetics.get("legend_policy") in (None, "auto"):
        legend_policy = "outside"

    title_text = intent.get("title") or (intent.get("user_goal")) or "Visualization"

    spec: Dict[str, Any] = {
        "canvas": {"width": 960, "height": 576, "dpi": 300},
        "overlays": overlays,
        "scales": {
            "x": {"kind": "categorical", "range": None, "breaks": None},
            "y_left": {"kind": "linear", "range": [0, None], "breaks": None},
            "y_right": {"kind": "linear", "range": None},
        },
        "layout": {
            "titles": {"top": title_text, "left": None, "right": None, "bottom": None},
            "title_align": aesthetics.get("title_align", "left"),
            "legend": {"loc": legend_policy, "ncol": 1, "frame": False},
            "grid": {"x": False, "y": True, "minor": True},
        },
        "theme": {
            "font": aesthetics.get("font_pref", "Arial"),
            "fontsize": 9,
            "axis_linewidth": 1.0,
            "tick_len": 3.0,
            "tick_width": 0.8,
            "palette_global": aesthetics.get("palette", "tab10"),
            "line_width": 1.5,
            "marker_size": 36,
        },
        "flags": {
            "inherit_palette": True,
            "legend_outside": "auto",
            "safe_log_y": True,
            "max_overlays": 3,
            "tick_density": "normal",
        },
    }
    return spec
