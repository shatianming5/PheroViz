from __future__ import annotations

import math
import numbers
import unicodedata
from collections import defaultdict, deque
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .manifest import extract_figure_manifest, json_value
from .models import (
    CheckResult,
    EvaluationResult,
    FigureManifest,
    MetricConfig,
    MetricResult,
    Mismatch,
)
from .schema import (
    coerce_metric_config,
    validate_evaluation_result,
    validate_expectation,
    validate_figure_manifest,
)


_COHESION_CHECKS = (
    "shared_scale",
    "shared_unit",
    "legend_deduplication",
    "palette_mapping",
)


def _normal_text(value: Any, config: MetricConfig) -> Optional[str]:
    if value is None:
        return None
    text = " ".join(unicodedata.normalize("NFKC", str(value)).split())
    if not config.case_sensitive_labels:
        text = text.casefold()
    return text


def _normal_unit(value: Any, config: MetricConfig) -> Optional[str]:
    text = _normal_text(value, config)
    if not text:
        return None
    compact = text.replace(" ", "")
    aliases = {
        _normal_text(key, config): _normal_text(alias, config)
        for key, alias in config.unit_aliases.items()
    }
    return aliases.get(text, aliases.get(compact, compact))


def _as_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, numbers.Real):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    if isinstance(value, str):
        try:
            converted = float(value.strip())
        except ValueError:
            return None
        return converted if math.isfinite(converted) else None
    return None


def _value_matches(expected: Any, observed: Any, config: MetricConfig) -> bool:
    expected_number = _as_number(expected)
    observed_number = _as_number(observed)
    if expected_number is not None and observed_number is not None:
        return abs(observed_number - expected_number) <= config.numeric_tolerance.bound(expected_number)
    if expected is None or observed is None:
        return expected is observed
    return _normal_text(expected, config) == _normal_text(observed, config)


def _value_distance(expected: Any, observed: Any, config: MetricConfig) -> float:
    expected_number = _as_number(expected)
    observed_number = _as_number(observed)
    if expected_number is not None and observed_number is not None:
        bound = max(config.numeric_tolerance.bound(expected_number), 1e-15)
        return abs(observed_number - expected_number) / bound
    return 0.0 if _value_matches(expected, observed, config) else math.inf


def _point_matches(expected: Dict[str, Any], observed: Dict[str, Any], config: MetricConfig) -> bool:
    keys = [key for key in ("x", "y", "value") if key in expected]
    return bool(keys) and all(_value_matches(expected[key], observed.get(key), config) for key in keys)


def _point_distance(expected: Dict[str, Any], observed: Dict[str, Any], config: MetricConfig) -> float:
    distances = [
        _value_distance(expected[key], observed.get(key), config)
        for key in ("x", "y", "value")
        if key in expected
    ]
    return sum(distances) if distances and all(math.isfinite(value) for value in distances) else math.inf


def _exact_point_key(
    point: Mapping[str, Any],
    fields: Sequence[str],
    config: MetricConfig,
) -> tuple[Any, ...]:
    values: list[Any] = []
    for field in fields:
        value = point.get(field)
        number = _as_number(value)
        if number is not None:
            rounded = round(number, config.float_precision)
            values.append(("number", 0.0 if rounded == 0 else rounded))
        elif value is None:
            values.append(("none", None))
        else:
            values.append(("text", _normal_text(value, config)))
    return tuple(values)


def _flatten_image_values(values: Any) -> List[Dict[str, Any]]:
    if not isinstance(values, list):
        return [{"value": values}]
    if values and isinstance(values[0], list):
        return [
            {"row": row_index, "column": column_index, "value": value}
            for row_index, row in enumerate(values)
            for column_index, value in enumerate(row)
        ]
    return [{"column": index, "value": value} for index, value in enumerate(values)]


def _series_points(series: Mapping[str, Any], expected: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    if series.get("kind") == "image":
        return _flatten_image_values(series.get("value"))

    x_values = list(series.get("x") or [])
    y_values = list(series.get("y") or [])
    value_values = series.get("value")
    if isinstance(value_values, list) and value_values and isinstance(value_values[0], list):
        return _flatten_image_values(value_values)
    value_values = list(value_values or []) if isinstance(value_values, list) else []

    if expected is not None:
        expected_x = list(expected.get("x") or [])
        expected_has_non_numeric_x = any(_as_number(value) is None for value in expected_x if value is not None)
        x_labels = list(series.get("x_labels") or [])
        if expected_has_non_numeric_x and len(x_labels) == len(x_values) and any(label is not None for label in x_labels):
            x_values = [label if label is not None else raw for label, raw in zip(x_labels, x_values)]

    length = max(len(x_values), len(y_values), len(value_values), 0)
    points: List[Dict[str, Any]] = []
    for index in range(length):
        point: Dict[str, Any] = {}
        if index < len(x_values):
            point["x"] = x_values[index]
        if index < len(y_values):
            point["y"] = y_values[index]
        if index < len(value_values):
            point["value"] = value_values[index]
        if point and not (
            point.get("y") is None
            and point.get("value") is None
            and ("y" in point or "value" in point)
        ):
            points.append(point)
    return points


def _match_points(
    expected_points: Sequence[Dict[str, Any]],
    observed_points: Sequence[Dict[str, Any]],
    config: MetricConfig,
) -> Tuple[int, List[int], List[int]]:
    unmatched_observed = set(range(len(observed_points)))
    missing_expected: List[int] = []
    matched = 0
    field_sets = {
        tuple(key for key in ("x", "y", "value") if key in expected)
        for expected in expected_points
    }
    exact_buckets: dict[
        tuple[tuple[str, ...], tuple[Any, ...]],
        deque[int],
    ] = defaultdict(deque)
    for observed_index, observed in enumerate(observed_points):
        for fields in field_sets:
            if fields:
                exact_buckets[
                    (fields, _exact_point_key(observed, fields, config))
                ].append(observed_index)

    residual_expected: list[tuple[int, Dict[str, Any]]] = []
    for expected_index, expected in enumerate(expected_points):
        fields = tuple(
            key for key in ("x", "y", "value") if key in expected
        )
        bucket = exact_buckets.get(
            (fields, _exact_point_key(expected, fields, config))
        )
        while bucket and bucket[0] not in unmatched_observed:
            bucket.popleft()
        observed_index = bucket.popleft() if bucket else None
        if observed_index is not None and observed_index in unmatched_observed:
            unmatched_observed.remove(observed_index)
            matched += 1
        else:
            residual_expected.append((expected_index, expected))

    for expected_index, expected in residual_expected:
        candidates = [
            (
                _point_distance(expected, observed_points[observed_index], config),
                observed_index,
            )
            for observed_index in unmatched_observed
            if _point_matches(expected, observed_points[observed_index], config)
        ]
        if not candidates:
            missing_expected.append(expected_index)
            continue
        _, observed_index = min(candidates)
        unmatched_observed.remove(observed_index)
        matched += 1
    return matched, missing_expected, sorted(unmatched_observed)


def _frame_for_panel(source_df: Any, panel: Mapping[str, Any]) -> pd.DataFrame:
    if isinstance(source_df, pd.DataFrame):
        return source_df
    if isinstance(source_df, Mapping):
        source_key = panel.get("source_key") or panel["panel_id"]
        frame = source_df.get(source_key)
        if not isinstance(frame, pd.DataFrame):
            raise KeyError(f"No pandas DataFrame found for source key '{source_key}'")
        return frame
    raise TypeError("source_df must be a pandas DataFrame or a mapping of keys to DataFrames")


def _filter_frame(frame: pd.DataFrame, series: Mapping[str, Any]) -> pd.DataFrame:
    filtered = frame
    for column, expected in (series.get("where") or {}).items():
        if column not in filtered:
            raise KeyError(f"Filter column '{column}' is absent from source_df")
        if isinstance(expected, list):
            filtered = filtered[filtered[column].isin(expected)]
        else:
            filtered = filtered[filtered[column] == expected]
    sort_by = list(series.get("sort_by") or [])
    if sort_by:
        missing = [column for column in sort_by if column not in filtered]
        if missing:
            raise KeyError(f"Sort columns absent from source_df: {missing}")
        filtered = filtered.sort_values(sort_by, kind="mergesort")
    return filtered


def _resolve_data_ref(ref: Any, frame: pd.DataFrame, config: MetricConfig) -> Any:
    if ref is None:
        return []
    if isinstance(ref, str):
        if ref not in frame:
            raise KeyError(f"Expected source column '{ref}' is absent from source_df")
        return json_value(frame[ref].tolist(), config.float_precision)
    if isinstance(ref, Mapping):
        if "column" in ref:
            return _resolve_data_ref(ref["column"], frame, config)
        return json_value(ref.get("values") or [], config.float_precision)
    return json_value(ref, config.float_precision)


def _resolve_expectation(
    source_df: Any,
    expectation: Mapping[str, Any],
    config: MetricConfig,
) -> List[Dict[str, Any]]:
    resolved_panels: List[Dict[str, Any]] = []
    for panel in expectation["panels"]:
        frame = _frame_for_panel(source_df, panel)
        resolved_series = []
        for series in panel.get("series") or []:
            filtered = _filter_frame(frame, series)
            resolved = dict(series)
            for field in ("x", "y", "value"):
                if field in series:
                    resolved[field] = _resolve_data_ref(series[field], filtered, config)
            resolved_series.append(resolved)
        resolved_panel = dict(panel)
        resolved_panel["series"] = resolved_series
        resolved_panels.append(resolved_panel)
    return resolved_panels


def _manifest_axes(manifest: FigureManifest) -> List[Dict[str, Any]]:
    return [axis.to_dict() for axis in manifest.axes if axis.role == "panel"]


def _bind_panels(
    manifest: FigureManifest,
    resolved_panels: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Optional[Dict[str, Any]]]]:
    primary_axes = _manifest_axes(manifest)
    all_axes = [axis.to_dict() for axis in manifest.axes]
    by_id = {axis["axis_id"]: axis for axis in primary_axes}
    by_panel_id = {axis["panel_id"]: axis for axis in primary_axes}
    secondary_by_panel: Dict[str, List[Dict[str, Any]]] = {}
    for axis in all_axes:
        if axis["role"] == "secondary" and axis["panel_id"] is not None:
            secondary_by_panel.setdefault(axis["panel_id"], []).append(axis)
    bindings: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = {}
    used_primary_ids: set[str] = set()
    exact_axes: Dict[str, Optional[Dict[str, Any]]] = {}
    for panel in resolved_panels:
        panel_id = panel["panel_id"]
        axis = by_id.get(panel_id) or by_panel_id.get(panel_id)
        if axis is not None and axis["axis_id"] in used_primary_ids:
            axis = None
        exact_axes[panel_id] = axis
        if axis is not None:
            used_primary_ids.add(axis["axis_id"])

    for panel in resolved_panels:
        panel_id = panel["panel_id"]
        axis = exact_axes[panel_id]
        if axis is None and "axis_index" in panel:
            index = panel["axis_index"]
            candidate = primary_axes[index] if index < len(primary_axes) else None
            if (
                candidate is not None
                and candidate["axis_id"] not in used_primary_ids
            ):
                axis = candidate
                used_primary_ids.add(axis["axis_id"])
        logical_panel_id = axis["panel_id"] if axis else panel_id
        secondary_axes = secondary_by_panel.get(logical_panel_id, [])
        bindings[panel_id] = {
            "primary": axis,
            "secondary": secondary_axes[0] if secondary_axes else None,
        }
    return bindings


def _kind_compatible(expected: str, observed: str) -> bool:
    if expected == observed:
        return True
    return {expected, observed} <= {"line", "errorbar"}


def _pair_series(
    expected_series: Sequence[Mapping[str, Any]],
    observed_series: Sequence[Mapping[str, Any]],
    config: MetricConfig,
) -> Tuple[Dict[int, int], Dict[Tuple[int, int], Tuple[int, List[int], List[int]]]]:
    candidates = []
    point_matches: Dict[Tuple[int, int], Tuple[int, List[int], List[int]]] = {}
    for expected_index, expected in enumerate(expected_series):
        expected_points = _series_points(expected)
        expected_id = _normal_text(expected.get("series_id"), config)
        expected_label = _normal_text(expected.get("label"), config)
        for observed_index, observed in enumerate(observed_series):
            expected_axis = str(expected.get("y_axis") or "primary")
            observed_axis = str(observed.get("axis_slot") or "primary")
            if expected_axis != observed_axis:
                continue
            if not _kind_compatible(str(expected.get("kind")), str(observed.get("kind"))):
                continue
            observed_points = _series_points(observed, expected)
            match = _match_points(expected_points, observed_points, config)
            point_matches[(expected_index, observed_index)] = match
            observed_id = _normal_text(observed.get("series_id"), config)
            observed_label = _normal_text(observed.get("label"), config)
            id_match = bool(expected_id and expected_id == observed_id)
            label_match = bool(expected_label and expected_label == observed_label)
            eligible = id_match or label_match or match[0] > 0 or not (expected_id or expected_label)
            if not eligible:
                continue
            score = (
                int(id_match),
                int(label_match),
                match[0],
                -abs(len(expected_points) - len(observed_points)),
                -expected_index,
                -observed_index,
            )
            candidates.append((score, expected_index, observed_index))

    pairs: Dict[int, int] = {}
    used_observed: set[int] = set()
    for _, expected_index, observed_index in sorted(candidates, reverse=True):
        if expected_index in pairs or observed_index in used_observed:
            continue
        pairs[expected_index] = observed_index
        used_observed.add(observed_index)
    return pairs, point_matches


def _mismatch(
    code: str,
    message: str,
    panel_id: Optional[str],
    series_id: Optional[str] = None,
    location: Optional[Dict[str, Any]] = None,
    expected: Any = None,
    observed: Any = None,
) -> Mismatch:
    return Mismatch(
        code=code,
        message=message,
        panel_id=panel_id,
        series_id=series_id,
        location=location or {},
        expected=expected,
        observed=observed,
    )


def _simple_check(
    name: str,
    items: List[Dict[str, Any]],
    mismatches: List[Mismatch],
    details: Optional[Dict[str, Any]] = None,
) -> CheckResult:
    numerator = sum(1 for item in items if item["passed"])
    denominator = len(items)
    return CheckResult(
        name=name,
        applicable=denominator > 0,
        numerator=numerator,
        denominator=denominator,
        items=items,
        mismatches=mismatches,
        details=details or {},
    )


def _limits_match(
    reference: Sequence[Any],
    candidate: Sequence[Any],
    config: MetricConfig,
) -> bool:
    if len(reference) != 2 or len(candidate) != 2:
        return False
    for expected, observed in zip(reference, candidate):
        expected_number = _as_number(expected)
        observed_number = _as_number(observed)
        if expected_number is None or observed_number is None:
            return False
        if abs(observed_number - expected_number) > config.limit_tolerance.bound(expected_number):
            return False
    return True


def evaluate_fidelity(
    manifest: FigureManifest,
    source_df: Any,
    expectation: Mapping[str, Any],
    config: Optional[MetricConfig | Mapping[str, Any]] = None,
) -> MetricResult:
    validate_expectation(expectation)
    validate_figure_manifest(manifest)
    cfg = coerce_metric_config(config)
    resolved_panels = _resolve_expectation(source_df, expectation, cfg)
    bindings = _bind_panels(manifest, resolved_panels)
    bound_axis_ids = {
        axis["axis_id"]
        for binding in bindings.values()
        for axis in (binding.get("primary"), binding.get("secondary"))
        if axis is not None
    }

    numeric_expected = 0
    numeric_observed = 0
    numeric_matched = 0
    numeric_items: List[Dict[str, Any]] = []
    numeric_mismatches: List[Mismatch] = []
    coverage_items: List[Dict[str, Any]] = []
    coverage_mismatches: List[Mismatch] = []
    purity_items: List[Dict[str, Any]] = []
    purity_mismatches: List[Mismatch] = []
    label_items: List[Dict[str, Any]] = []
    label_mismatches: List[Mismatch] = []
    unit_items: List[Dict[str, Any]] = []
    unit_mismatches: List[Mismatch] = []
    scale_items: List[Dict[str, Any]] = []
    scale_mismatches: List[Mismatch] = []

    for panel in resolved_panels:
        panel_id = panel["panel_id"]
        axes = bindings[panel_id]
        primary_axis = axes["primary"]
        secondary_axis = axes["secondary"]
        observed_series = list(primary_axis["series"] if primary_axis else [])
        observed_series.extend(secondary_axis["series"] if secondary_axis else [])
        expected_series = list(panel.get("series") or [])
        pairs, point_matches = (
            _pair_series(expected_series, observed_series, cfg)
            if primary_axis or secondary_axis
            else ({}, {})
        )
        paired_observed = set(pairs.values())

        for expected_index, expected in enumerate(expected_series):
            series_id = str(expected["series_id"])
            expected_points = _series_points(expected)
            observed_index = pairs.get(expected_index)
            observed = observed_series[observed_index] if observed_index is not None else None
            observed_points = _series_points(observed, expected) if observed is not None else []
            if observed is not None:
                matched, missing_indices, extra_indices = point_matches[(expected_index, observed_index)]
            else:
                matched = 0
                missing_indices = list(range(len(expected_points)))
                extra_indices = []

            numeric_expected += len(expected_points)
            numeric_observed += len(observed_points)
            numeric_matched += matched
            numeric_items.append(
                {
                    "panel_id": panel_id,
                    "series_id": series_id,
                    "matched_points": matched,
                    "expected_points": len(expected_points),
                    "observed_points": len(observed_points),
                }
            )
            for index in missing_indices:
                nearest = None
                if observed_points:
                    nearest = min(
                        observed_points,
                        key=lambda point: _point_distance(expected_points[index], point, cfg),
                    )
                numeric_mismatches.append(
                    _mismatch(
                        "numeric_point_missing",
                        "No rendered data point matched the source-table expectation.",
                        panel_id,
                        series_id,
                        {"expected_point_index": index},
                        expected_points[index],
                        nearest,
                    )
                )
            for index in extra_indices:
                numeric_mismatches.append(
                    _mismatch(
                        "numeric_point_unexpected",
                        "Rendered series contains a data point absent from the source-table expectation.",
                        panel_id,
                        series_id,
                        {"observed_point_index": index},
                        None,
                        observed_points[index],
                    )
                )

            covered = observed is not None
            coverage_items.append(
                {"panel_id": panel_id, "series_id": series_id, "passed": covered}
            )
            if not covered:
                coverage_mismatches.append(
                    _mismatch(
                        "series_missing",
                        "Expected source-table series is absent from the figure.",
                        panel_id,
                        series_id,
                        expected={"kind": expected.get("kind"), "label": expected.get("label")},
                        observed=None,
                    )
                )

            if "label" in expected and expected.get("label") is not None:
                observed_label = observed.get("label") if observed else None
                passed = _normal_text(expected["label"], cfg) == _normal_text(observed_label, cfg)
                label_items.append(
                    {
                        "panel_id": panel_id,
                        "series_id": series_id,
                        "field": "series_label",
                        "passed": passed,
                    }
                )
                if not passed:
                    label_mismatches.append(
                        _mismatch(
                            "series_label_mismatch",
                            "Rendered series label differs from the declared expectation.",
                            panel_id,
                            series_id,
                            {"field": "label"},
                            expected["label"],
                            observed_label,
                        )
                    )

        for observed_index, observed in enumerate(observed_series):
            paired = observed_index in paired_observed
            purity_items.append(
                {
                    "panel_id": panel_id,
                    "series_id": observed["series_id"],
                    "passed": paired,
                }
            )
            if not paired:
                observed_points = _series_points(observed)
                numeric_observed += len(observed_points)
                numeric_mismatches.append(
                    _mismatch(
                        "series_unexpected",
                        "Rendered data-bearing series has no source-table expectation.",
                        panel_id,
                        observed["series_id"],
                        observed={
                            "kind": observed.get("kind"),
                            "label": observed.get("label"),
                            "point_count": len(observed_points),
                        },
                    )
                )
                purity_mismatches.append(
                    _mismatch(
                        "series_unexpected",
                        "Rendered data-bearing series has no declared expectation.",
                        panel_id,
                        observed["series_id"],
                        observed={"kind": observed.get("kind"), "label": observed.get("label")},
                    )
                )

        for dimension in ("x", "y"):
            axis_key = f"{dimension}_axis"
            if f"{dimension}label" in panel and panel.get(f"{dimension}label") is not None:
                expected_label = panel[f"{dimension}label"]
                observed_label = primary_axis[axis_key]["label"] if primary_axis else None
                passed = _normal_text(expected_label, cfg) == _normal_text(observed_label, cfg)
                label_items.append(
                    {
                        "panel_id": panel_id,
                        "field": f"{dimension}label",
                        "passed": passed,
                    }
                )
                if not passed:
                    label_mismatches.append(
                        _mismatch(
                            "axis_label_mismatch",
                            f"Rendered {dimension}-axis label differs from the declared expectation.",
                            panel_id,
                            location={"dimension": dimension},
                            expected=expected_label,
                            observed=observed_label,
                        )
                    )

            if f"{dimension}_unit" in panel and panel.get(f"{dimension}_unit") is not None:
                expected_unit = _normal_unit(panel[f"{dimension}_unit"], cfg)
                observed_unit = (
                    _normal_unit(primary_axis[axis_key]["unit"], cfg)
                    if primary_axis
                    else None
                )
                passed = expected_unit == observed_unit
                unit_items.append(
                    {
                        "panel_id": panel_id,
                        "dimension": dimension,
                        "passed": passed,
                    }
                )
                if not passed:
                    unit_mismatches.append(
                        _mismatch(
                            "axis_unit_mismatch",
                            f"Rendered {dimension}-axis unit differs from the declared expectation.",
                            panel_id,
                            location={"dimension": dimension},
                            expected=panel[f"{dimension}_unit"],
                            observed=primary_axis[axis_key]["unit"] if primary_axis else None,
                        )
                    )

            if f"{dimension}_scale" in panel and panel.get(f"{dimension}_scale") is not None:
                expected_scale = str(panel[f"{dimension}_scale"])
                observed_scale = primary_axis[axis_key]["scale"] if primary_axis else None
                passed = expected_scale == observed_scale
                scale_items.append(
                    {
                        "panel_id": panel_id,
                        "dimension": dimension,
                        "passed": passed,
                    }
                )
                if not passed:
                    scale_mismatches.append(
                        _mismatch(
                            "axis_scale_mismatch",
                            f"Rendered {dimension}-axis scale differs from the declared expectation.",
                            panel_id,
                            location={"dimension": dimension},
                            expected=expected_scale,
                            observed=observed_scale,
                        )
                    )

        secondary_expectation = panel.get("secondary_y") or {}
        if secondary_expectation.get("ylabel") is not None:
            expected_label = secondary_expectation["ylabel"]
            observed_label = secondary_axis["y_axis"]["label"] if secondary_axis else None
            passed = _normal_text(expected_label, cfg) == _normal_text(observed_label, cfg)
            label_items.append(
                {
                    "panel_id": panel_id,
                    "field": "secondary_ylabel",
                    "axis_slot": "secondary",
                    "passed": passed,
                }
            )
            if not passed:
                label_mismatches.append(
                    _mismatch(
                        "axis_label_mismatch",
                        "Rendered secondary y-axis label differs from the declared expectation.",
                        panel_id,
                        location={"dimension": "y", "axis_slot": "secondary"},
                        expected=expected_label,
                        observed=observed_label,
                    )
                )

        if secondary_expectation.get("y_unit") is not None:
            expected_unit = _normal_unit(secondary_expectation["y_unit"], cfg)
            observed_raw = secondary_axis["y_axis"]["unit"] if secondary_axis else None
            observed_unit = _normal_unit(observed_raw, cfg)
            passed = expected_unit == observed_unit
            unit_items.append(
                {
                    "panel_id": panel_id,
                    "dimension": "y",
                    "axis_slot": "secondary",
                    "passed": passed,
                }
            )
            if not passed:
                unit_mismatches.append(
                    _mismatch(
                        "axis_unit_mismatch",
                        "Rendered secondary y-axis unit differs from the declared expectation.",
                        panel_id,
                        location={"dimension": "y", "axis_slot": "secondary"},
                        expected=secondary_expectation["y_unit"],
                        observed=observed_raw,
                    )
                )

        if secondary_expectation.get("y_scale") is not None:
            expected_scale = str(secondary_expectation["y_scale"])
            observed_scale = secondary_axis["y_axis"]["scale"] if secondary_axis else None
            passed = expected_scale == observed_scale
            scale_items.append(
                {
                    "panel_id": panel_id,
                    "dimension": "y",
                    "axis_slot": "secondary",
                    "passed": passed,
                }
            )
            if not passed:
                scale_mismatches.append(
                    _mismatch(
                        "axis_scale_mismatch",
                        "Rendered secondary y-axis scale differs from the declared expectation.",
                        panel_id,
                        location={"dimension": "y", "axis_slot": "secondary"},
                        expected=expected_scale,
                        observed=observed_scale,
                    )
                )

    for axis in (
        item.to_dict()
        for item in manifest.axes
        if item.role in {"panel", "secondary"}
        and item.axis_id not in bound_axis_ids
    ):
        panel_id = str(axis.get("panel_id") or axis["axis_id"])
        for observed in axis.get("series") or []:
            observed_points = _series_points(observed)
            numeric_observed += len(observed_points)
            purity_items.append(
                {
                    "panel_id": panel_id,
                    "series_id": observed["series_id"],
                    "passed": False,
                }
            )
            numeric_mismatches.append(
                _mismatch(
                    "series_unexpected",
                    "Data-bearing series appears on an unbound axis.",
                    panel_id,
                    observed["series_id"],
                    observed={
                        "axis_id": axis["axis_id"],
                        "kind": observed.get("kind"),
                        "label": observed.get("label"),
                        "point_count": len(observed_points),
                    },
                )
            )
            purity_mismatches.append(
                _mismatch(
                    "series_unexpected",
                    "Data-bearing series appears on an unbound axis.",
                    panel_id,
                    observed["series_id"],
                    observed={"axis_id": axis["axis_id"]},
                )
            )

    numeric_denominator = numeric_expected + numeric_observed
    numeric_check = CheckResult(
        name="numeric_match",
        applicable=numeric_denominator > 0,
        numerator=2 * numeric_matched,
        denominator=numeric_denominator,
        items=numeric_items,
        mismatches=numeric_mismatches,
        details={
            "matched_points": numeric_matched,
            "expected_points": numeric_expected,
            "observed_points": numeric_observed,
            "tolerance": cfg.numeric_tolerance.to_dict(),
            "definition": "2 * matched / (expected + observed)",
        },
    )
    checks = {
        "numeric_match": numeric_check,
        "series_coverage": _simple_check(
            "series_coverage", coverage_items, coverage_mismatches
        ),
        "series_purity": _simple_check(
            "series_purity", purity_items, purity_mismatches
        ),
        "label_match": _simple_check("label_match", label_items, label_mismatches),
        "unit_match": _simple_check("unit_match", unit_items, unit_mismatches),
        "scale_match": _simple_check("scale_match", scale_items, scale_mismatches),
    }
    return MetricResult(name="fidelity", checks=checks)


def _expected_series_label(
    panel: Mapping[str, Any],
    series_id: str,
) -> str:
    for series in panel.get("series") or []:
        if series.get("series_id") == series_id:
            return str(series.get("label") or series_id)
    return series_id


def _find_observed_series(
    axis: Mapping[str, Any],
    panel: Mapping[str, Any],
    series_id: str,
    config: MetricConfig,
) -> Optional[Mapping[str, Any]]:
    expected_label = _expected_series_label(panel, series_id)
    wanted = {_normal_text(series_id, config), _normal_text(expected_label, config)}
    for series in axis.get("series") or []:
        observed = {
            _normal_text(series.get("series_id"), config),
            _normal_text(series.get("label"), config),
        }
        if any(value is not None and value in observed for value in wanted):
            return series
    return None


def _shared_series_ids(
    group: Mapping[str, Any],
    panels_by_id: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    declared = list(group.get("series") or [])
    if declared:
        return declared
    sets = [
        {str(series["series_id"]) for series in panels_by_id[panel_id].get("series") or []}
        for panel_id in group["panels"]
        if panel_id in panels_by_id
    ]
    return sorted(set.intersection(*sets)) if sets else []


def _empty_cohesion(reason: str) -> MetricResult:
    return MetricResult(
        name="cohesion",
        checks={
            name: CheckResult(
                name=name,
                applicable=False,
                numerator=0,
                denominator=0,
                details={"reason": reason},
            )
            for name in _COHESION_CHECKS
        },
    )


def evaluate_cohesion(
    manifest: FigureManifest,
    expectation: Mapping[str, Any],
    config: Optional[MetricConfig | Mapping[str, Any]] = None,
) -> MetricResult:
    validate_expectation(expectation)
    validate_figure_manifest(manifest)
    cfg = coerce_metric_config(config)
    if len(expectation.get("panels") or []) <= 1:
        return _empty_cohesion("single_panel")
    groups = list(expectation.get("panel_groups") or [])
    if not groups:
        return _empty_cohesion("no_declared_panel_groups")

    resolved_panels = [dict(panel) for panel in expectation["panels"]]
    bindings = _bind_panels(manifest, resolved_panels)
    panels_by_id = {panel["panel_id"]: panel for panel in resolved_panels}
    accumulators = {
        name: {"items": [], "mismatches": []}
        for name in _COHESION_CHECKS
    }

    def add_item(
        check_name: str,
        passed: bool,
        item: Dict[str, Any],
        mismatch: Optional[Mismatch] = None,
    ) -> None:
        accumulators[check_name]["items"].append({**item, "passed": passed})
        if mismatch is not None:
            accumulators[check_name]["mismatches"].append(mismatch)

    for group in groups:
        group_id = group["group_id"]
        panel_ids = list(group["panels"])
        axes = [
            bindings[panel_id]["primary"] if panel_id in bindings else None
            for panel_id in panel_ids
        ]
        checks = group.get("checks") or {}

        for dimension in ("x", "y"):
            if checks.get(f"shared_{dimension}_scale"):
                values = {
                    panel_id: axis[f"{dimension}_axis"]["scale"] if axis else None
                    for panel_id, axis in zip(panel_ids, axes)
                }
                present = list(values.values())
                passed = all(value is not None for value in present) and len(set(present)) == 1
                add_item(
                    "shared_scale",
                    passed,
                    {"group_id": group_id, "dimension": dimension, "observed": values},
                    None
                    if passed
                    else _mismatch(
                        "shared_scale_mismatch",
                        f"Panels in group '{group_id}' do not share one {dimension}-axis scale.",
                        None,
                        location={"group_id": group_id, "dimension": dimension},
                        expected="one shared scale",
                        observed=values,
                    ),
                )

            if checks.get(f"shared_{dimension}_limits"):
                values = {
                    panel_id: axis[f"{dimension}_axis"]["limits"] if axis else None
                    for panel_id, axis in zip(panel_ids, axes)
                }
                present = list(values.values())
                reference = present[0] if present else None
                passed = (
                    reference is not None
                    and all(value is not None for value in present)
                    and all(_limits_match(reference, value, cfg) for value in present[1:])
                )
                add_item(
                    "shared_scale",
                    passed,
                    {
                        "group_id": group_id,
                        "dimension": f"{dimension}_limits",
                        "observed": values,
                    },
                    None
                    if passed
                    else _mismatch(
                        "shared_limits_mismatch",
                        f"Panels in group '{group_id}' do not share one {dimension}-axis range.",
                        None,
                        location={"group_id": group_id, "dimension": dimension},
                        expected="one shared axis range",
                        observed=values,
                    ),
                )

            if checks.get(f"shared_{dimension}_unit"):
                raw_values = {
                    panel_id: axis[f"{dimension}_axis"]["unit"] if axis else None
                    for panel_id, axis in zip(panel_ids, axes)
                }
                values = {
                    panel_id: _normal_unit(value, cfg)
                    for panel_id, value in raw_values.items()
                }
                present = list(values.values())
                missing_allowed = cfg.allow_missing_shared_units and all(value is None for value in present)
                passed = missing_allowed or (
                    all(value is not None for value in present) and len(set(present)) == 1
                )
                add_item(
                    "shared_unit",
                    passed,
                    {"group_id": group_id, "dimension": dimension, "observed": raw_values},
                    None
                    if passed
                    else _mismatch(
                        "shared_unit_mismatch",
                        f"Panels in group '{group_id}' do not share one declared {dimension}-axis unit.",
                        None,
                        location={"group_id": group_id, "dimension": dimension},
                        expected="one non-missing normalized unit",
                        observed=raw_values,
                    ),
                )

        shared_series = _shared_series_ids(group, panels_by_id)
        if checks.get("legend_deduplicated"):
            if not shared_series:
                add_item(
                    "legend_deduplication",
                    False,
                    {"group_id": group_id, "series_id": None, "occurrences": 0},
                    _mismatch(
                        "shared_series_undeclared",
                        f"Group '{group_id}' requests legend checking but declares no shared series.",
                        None,
                        location={"group_id": group_id},
                    ),
                )
            for series_id in shared_series:
                labels = {
                    _normal_text(
                        _expected_series_label(panels_by_id[panel_id], series_id),
                        cfg,
                    )
                    for panel_id in panel_ids
                    if panel_id in panels_by_id
                }
                labels.discard(None)
                occurrences = 0
                for axis in axes:
                    if axis:
                        occurrences += sum(
                            1
                            for entry in axis["legend"]["entries"]
                            if _normal_text(entry["label"], cfg) in labels
                        )
                occurrences += sum(
                    1
                    for legend in manifest.figure_legends
                    for entry in legend.entries
                    if _normal_text(entry.label, cfg) in labels
                )
                passed = occurrences == 1
                add_item(
                    "legend_deduplication",
                    passed,
                    {
                        "group_id": group_id,
                        "series_id": series_id,
                        "occurrences": occurrences,
                    },
                    None
                    if passed
                    else _mismatch(
                        "legend_not_deduplicated",
                        f"Shared series '{series_id}' must appear exactly once across the group legend scope.",
                        None,
                        series_id,
                        {"group_id": group_id},
                        1,
                        occurrences,
                    ),
                )

        if checks.get("palette_consistent"):
            if not shared_series:
                add_item(
                    "palette_mapping",
                    False,
                    {"group_id": group_id, "series_id": None, "colors": {}},
                    _mismatch(
                        "shared_series_undeclared",
                        f"Group '{group_id}' requests palette checking but declares no shared series.",
                        None,
                        location={"group_id": group_id},
                    ),
                )
            for series_id in shared_series:
                colors: Dict[str, Optional[str]] = {}
                for panel_id, axis in zip(panel_ids, axes):
                    panel = panels_by_id.get(panel_id)
                    observed = (
                        _find_observed_series(axis, panel, series_id, cfg)
                        if axis is not None and panel is not None
                        else None
                    )
                    color = observed.get("color") if observed else None
                    if color is None and observed and len(observed.get("colors") or []) == 1:
                        color = observed["colors"][0]
                    colors[panel_id] = color
                values = list(colors.values())
                passed = all(value is not None for value in values) and len(set(values)) == 1
                add_item(
                    "palette_mapping",
                    passed,
                    {"group_id": group_id, "series_id": series_id, "colors": colors},
                    None
                    if passed
                    else _mismatch(
                        "palette_mapping_mismatch",
                        f"Shared series '{series_id}' does not retain one color across panels.",
                        None,
                        series_id,
                        {"group_id": group_id},
                        "one shared color",
                        colors,
                    ),
                )

    checks = {
        name: _simple_check(
            name,
            accumulators[name]["items"],
            accumulators[name]["mismatches"],
        )
        for name in _COHESION_CHECKS
    }
    return MetricResult(name="cohesion", checks=checks)


def evaluate_figure(
    fig: Any,
    source_df: Any,
    expectation: Mapping[str, Any],
    config: Optional[MetricConfig | Mapping[str, Any]] = None,
) -> EvaluationResult:
    """Evaluate a live Matplotlib figure without rendered pixels or a VLM.

    Args:
        fig: A live ``matplotlib.figure.Figure``. Call this API before
            ``plt.close(fig)`` so its Axes and Artist objects remain available.
        source_df: A pandas DataFrame, or a mapping from ``source_key``/panel ID
            to DataFrames, containing the source-table values.
        expectation: A JSON-compatible object conforming to
            ``schemas/expectation.schema.json``.
        config: A versioned ``MetricConfig`` or its JSON-compatible dictionary.

    Returns:
        ``EvaluationResult`` with a JSON-safe figure manifest, fidelity checks,
        cohesion checks, raw numerators/denominators, and localized mismatches.
    """

    validate_expectation(expectation)
    cfg = coerce_metric_config(config)
    manifest = extract_figure_manifest(fig, cfg)
    fidelity = evaluate_fidelity(manifest, source_df, expectation, cfg)
    cohesion = evaluate_cohesion(manifest, expectation, cfg)
    result = EvaluationResult(
        manifest=manifest,
        fidelity=fidelity,
        cohesion=cohesion,
        config=cfg,
    )
    validate_figure_manifest(manifest)
    validate_evaluation_result(result)
    return result
