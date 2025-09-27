from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable

_REQUIRED_TOP = ("canvas", "overlays", "scales", "layout", "theme")
_REQUIRED_OVERLAY = ("mark", "x", "y")

_DEFAULTS: Dict[str, Any] = {
    "canvas": {"width": 960, "height": 576, "dpi": 300, "aspect": 1.667},
    "scales": {
        "x": {"kind": "categorical", "range": None, "breaks": None},
        "y_left": {"kind": "linear", "range": [0, None], "breaks": None},
        "y_right": {"kind": "linear", "range": None},
    },
    "layout": {
        "titles": {"top": None, "left": None, "right": None, "bottom": None},
        "title_align": "left",
        "legend": {"loc": "best", "ncol": 1, "frame": False},
        "grid": {"x": False, "y": True, "minor": True},
        "panel_labels": [],
    },
    "theme": {
        "font": "Arial",
        "fontsize": 9,
        "axis_linewidth": 1.0,
        "tick_len": 3.0,
        "tick_width": 0.8,
        "palette_global": "ColorBlindSafe",
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


def _deep_merge(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (upd or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_overlays(overlays: Iterable[Dict[str, Any]]) -> None:
    for idx, overlay in enumerate(overlays):
        _ensure(isinstance(overlay, dict), f"overlay[{idx}] must be object")
        for field in _REQUIRED_OVERLAY:
            _ensure(field in overlay, f"overlay[{idx}] missing '{field}'")
        mark = overlay.get('mark')
        allowed_marks = {"bar", "line", "scatter", "area", "hist", "hexbin", "boxplot", "violin", "heatmap", "lollipop"}
        _ensure(mark in allowed_marks, f"overlay[{idx}].mark '{mark}' unsupported")


def validate_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    _ensure(isinstance(spec, dict), "spec must be a mapping")
    for field in _REQUIRED_TOP:
        _ensure(field in spec, f"missing top-level field '{field}'")

    overlays = spec.get("overlays")
    _ensure(isinstance(overlays, list) and overlays, "spec.overlays must be non-empty list")
    _validate_overlays(overlays)

    _ensure(isinstance(spec.get("scales"), dict), "spec.scales must be object")
    _ensure(isinstance(spec.get("layout"), dict), "spec.layout must be object")
    _ensure(isinstance(spec.get("theme"), dict), "spec.theme must be object")

    merged = _deep_merge(_DEFAULTS, spec)
    return merged
