from __future__ import annotations

import hashlib
import math
import re
from dataclasses import replace
from datetime import date, datetime
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.container import BarContainer, ErrorbarContainer
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

from .models import (
    FIGURE_MANIFEST_SCHEMA_VERSION,
    AxisManifest,
    AxisProperties,
    FigureManifest,
    LegendEntry,
    LegendManifest,
    MetricConfig,
    SeriesManifest,
    stable_json_dumps,
)
from .schema import coerce_metric_config


_UNIT_PATTERNS = (
    re.compile(r"[\(\[\uFF08]\s*([^\)\]\uFF09]+?)\s*[\)\]\uFF09]\s*$"),
    re.compile(r"\s[/|]\s*([^/|]+?)\s*$"),
)


def _round_float(value: float, precision: int) -> Optional[float]:
    if not math.isfinite(value):
        return None
    rounded = round(float(value), precision)
    return 0.0 if rounded == 0 else rounded


def json_value(value: Any, precision: int = 12) -> Any:
    if value is np.ma.masked:
        return None
    if isinstance(value, np.ma.MaskedArray):
        return json_value(value.filled(np.nan), precision)
    if isinstance(value, np.ndarray):
        return [json_value(item, precision) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_value(value.item(), precision)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        return _round_float(value, precision)
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [json_value(item, precision) for item in value]
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return str(value)
    return _round_float(converted, precision)


def extract_unit(label: str, axis_units: Any = None) -> Optional[str]:
    if isinstance(axis_units, str) and axis_units.strip():
        return axis_units.strip()
    text = (label or "").strip()
    for pattern in _UNIT_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None


def _visible_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    label = str(value).strip()
    if not label or label.startswith("_"):
        return None
    return label


def _color_hex(value: Any) -> Optional[str]:
    try:
        return mcolors.to_hex(mcolors.to_rgba(value), keep_alpha=True).lower()
    except (TypeError, ValueError):
        return None


def _colors_from_array(values: Any) -> List[str]:
    if values is None:
        return []
    array = np.asarray(values)
    if array.size == 0:
        return []
    if array.ndim == 1:
        color = _color_hex(array)
        return [color] if color else []
    colors = {_color_hex(row) for row in array}
    return sorted(color for color in colors if color is not None)


def _artist_colors(artist: Any) -> List[str]:
    for accessor in ("get_facecolors", "get_edgecolors"):
        if hasattr(artist, accessor):
            colors = _colors_from_array(getattr(artist, accessor)())
            if colors:
                return colors
    if hasattr(artist, "get_color"):
        color = _color_hex(artist.get_color())
        if color:
            return [color]
    if hasattr(artist, "get_facecolor"):
        color = _color_hex(artist.get_facecolor())
        if color:
            return [color]
    return []


def _primary_color(colors: Sequence[str]) -> Optional[str]:
    return colors[0] if len(colors) == 1 else None


def _series_signature(payload: dict) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()[:12]


def _positions_to_tick_labels(ax: Any, positions: Sequence[Any], precision: int) -> List[Optional[str]]:
    ticks = np.asarray(ax.get_xticks(), dtype=float)
    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    finite_ticks = np.sort(ticks[np.isfinite(ticks)])
    spacings = np.diff(finite_ticks)
    positive_spacings = spacings[spacings > 0]
    nearest_tolerance = (
        0.49 * float(np.min(positive_spacings))
        if positive_spacings.size
        else 10 ** (-min(precision, 8))
    )
    result: List[Optional[str]] = []
    for position in positions:
        try:
            numeric = float(position)
        except (TypeError, ValueError):
            result.append(str(position))
            continue
        if ticks.size == 0:
            result.append(None)
            continue
        index = int(np.argmin(np.abs(ticks - numeric)))
        if abs(float(ticks[index]) - numeric) <= nearest_tolerance and index < len(labels):
            result.append(labels[index] or None)
        else:
            result.append(None)
    return result


def _line_payload(line: Line2D, config: MetricConfig, kind: str = "line") -> dict:
    x = json_value(np.asarray(line.get_xdata(orig=True)), config.float_precision)
    y = json_value(np.asarray(line.get_ydata(orig=True)), config.float_precision)
    colors = _artist_colors(line)
    return {
        "explicit_id": _visible_label(line.get_gid()),
        "kind": kind,
        "label": _visible_label(line.get_label()),
        "x": x,
        "y": y,
        "value": y,
        "color": _primary_color(colors),
        "colors": colors,
        "style": {
            "linestyle": str(line.get_linestyle()),
            "linewidth": json_value(line.get_linewidth(), config.float_precision),
            "marker": str(line.get_marker()),
            "markersize": json_value(line.get_markersize(), config.float_precision),
        },
        "metadata": {
            "artist_class": type(line).__name__,
            "point_count": len(x),
            "zorder": json_value(line.get_zorder(), config.float_precision),
        },
        "x_labels": [],
    }


def _is_explicit_annotation(line: Line2D) -> bool:
    for value in (line.get_gid(), line.get_label()):
        if value is None:
            continue
        normalized = str(value).strip().casefold().lstrip("_")
        if normalized.startswith(("annotation:", "reference:")):
            return True
    return False


def _is_data_line(ax: Any, line: Line2D) -> bool:
    if _is_explicit_annotation(line):
        return False
    transform = line.get_transform()
    return transform is ax.transData or transform == ax.transData


def _annotation_payload(line: Line2D, config: MetricConfig) -> dict:
    x = json_value(np.asarray(line.get_xdata(orig=True)), config.float_precision)
    y = json_value(np.asarray(line.get_ydata(orig=True)), config.float_precision)
    if _is_explicit_annotation(line):
        classification = "explicit_reference"
    elif len(x) == 2 and x == [0, 1] and len(set(y)) == 1:
        classification = "horizontal_reference_line"
    elif len(y) == 2 and y == [0, 1] and len(set(x)) == 1:
        classification = "vertical_reference_line"
    else:
        classification = "non_data_transform"
    payload = {
        "kind": "line",
        "classification": classification,
        "label": _visible_label(line.get_label()),
        "x": x,
        "y": y,
        "metadata": {
            "artist_class": type(line).__name__,
            "transform_class": type(line.get_transform()).__name__,
            "zorder": json_value(line.get_zorder(), config.float_precision),
        },
    }
    explicit_id = _visible_label(line.get_gid())
    signature = _series_signature(payload)
    payload["annotation_id"] = explicit_id or f"annotation-{signature}"
    return payload


def _errorbar_payload(container: ErrorbarContainer, config: MetricConfig) -> Optional[dict]:
    data_line = container.lines[0] if container.lines else None
    if not isinstance(data_line, Line2D):
        return None
    payload = _line_payload(data_line, config, kind="errorbar")
    label = _visible_label(container.get_label())
    if label:
        payload["label"] = label
    if hasattr(container, "get_gid"):
        payload["explicit_id"] = _visible_label(container.get_gid()) or payload["explicit_id"]
    payload["metadata"]["artist_class"] = type(container).__name__
    return payload


def _bar_payload(container: BarContainer, ax: Any, config: MetricConfig) -> dict:
    patches = list(container.patches)
    x = [patch.get_x() + patch.get_width() / 2 for patch in patches]
    y = [patch.get_height() for patch in patches]
    bottoms = [patch.get_y() for patch in patches]
    widths = [patch.get_width() for patch in patches]
    colors = sorted(
        {
            color
            for patch in patches
            for color in _artist_colors(patch)
            if color is not None
        }
    )
    explicit_id = _visible_label(container.get_gid()) if hasattr(container, "get_gid") else None
    x_json = json_value(x, config.float_precision)
    y_json = json_value(y, config.float_precision)
    return {
        "explicit_id": explicit_id,
        "kind": "bar",
        "label": _visible_label(container.get_label()),
        "x": x_json,
        "y": y_json,
        "value": y_json,
        "color": _primary_color(colors),
        "colors": colors,
        "style": {
            "width": json_value(widths, config.float_precision),
        },
        "metadata": {
            "artist_class": type(container).__name__,
            "point_count": len(patches),
            "bottom": json_value(bottoms, config.float_precision),
        },
        "x_labels": _positions_to_tick_labels(ax, x, config.float_precision),
    }


def _scatter_payload(collection: PathCollection, config: MetricConfig) -> dict:
    offsets = np.ma.asarray(collection.get_offsets())
    if offsets.ndim == 2 and offsets.shape[1] >= 2:
        x = json_value(offsets[:, 0], config.float_precision)
        y = json_value(offsets[:, 1], config.float_precision)
    else:
        x, y = [], []
    values = collection.get_array()
    value = json_value(values, config.float_precision) if values is not None else y
    colors = _artist_colors(collection)
    return {
        "explicit_id": _visible_label(collection.get_gid()),
        "kind": "scatter",
        "label": _visible_label(collection.get_label()),
        "x": x,
        "y": y,
        "value": value,
        "color": _primary_color(colors),
        "colors": colors,
        "style": {
            "sizes": json_value(collection.get_sizes(), config.float_precision),
            "linewidths": json_value(collection.get_linewidths(), config.float_precision),
        },
        "metadata": {
            "artist_class": type(collection).__name__,
            "point_count": len(x),
            "zorder": json_value(collection.get_zorder(), config.float_precision),
        },
        "x_labels": [],
    }


def _poly_payload(collection: PolyCollection, config: MetricConfig) -> dict:
    vertices: List[List[float]] = []
    for path in collection.get_paths():
        vertices.extend(path.vertices.tolist())
    x = json_value([point[0] for point in vertices], config.float_precision)
    y = json_value([point[1] for point in vertices], config.float_precision)
    colors = _artist_colors(collection)
    return {
        "explicit_id": _visible_label(collection.get_gid()),
        "kind": "poly",
        "label": _visible_label(collection.get_label()),
        "x": x,
        "y": y,
        "value": y,
        "color": _primary_color(colors),
        "colors": colors,
        "style": {},
        "metadata": {
            "artist_class": type(collection).__name__,
            "vertex_count": len(vertices),
            "zorder": json_value(collection.get_zorder(), config.float_precision),
        },
        "x_labels": [],
    }


def _image_payload(image: AxesImage, config: MetricConfig) -> dict:
    values = np.ma.asarray(image.get_array())
    extent = [float(value) for value in image.get_extent()]
    rows = values.shape[0] if values.ndim >= 1 else 0
    columns = values.shape[1] if values.ndim >= 2 else 0
    x = (
        np.linspace(extent[0], extent[1], columns, endpoint=False)
        + (extent[1] - extent[0]) / (2 * columns)
        if columns
        else np.asarray([])
    )
    y = (
        np.linspace(extent[2], extent[3], rows, endpoint=False)
        + (extent[3] - extent[2]) / (2 * rows)
        if rows
        else np.asarray([])
    )
    cmap = image.get_cmap()
    return {
        "explicit_id": _visible_label(image.get_gid()),
        "kind": "image",
        "label": _visible_label(image.get_label()),
        "x": json_value(x, config.float_precision),
        "y": json_value(y, config.float_precision),
        "value": json_value(values, config.float_precision),
        "color": cmap.name if cmap is not None else None,
        "colors": [cmap.name] if cmap is not None else [],
        "style": {
            "interpolation": str(image.get_interpolation()),
            "origin": str(image.origin),
        },
        "metadata": {
            "artist_class": type(image).__name__,
            "shape": list(values.shape),
            "extent": json_value(extent, config.float_precision),
            "clim": json_value(image.get_clim(), config.float_precision),
        },
        "x_labels": [],
    }


def _finalize_series(
    payloads: Iterable[dict],
    axis_slot: Optional[str],
) -> List[SeriesManifest]:
    sortable = []
    for payload in payloads:
        signature_payload = {key: value for key, value in payload.items() if key != "explicit_id"}
        signature = _series_signature(signature_payload)
        sortable.append((payload["kind"], payload.get("label") or "", signature, payload))
    sortable.sort(key=lambda item: item[:3])

    used_ids: dict[str, int] = {}
    result: List[SeriesManifest] = []
    for _, _, signature, payload in sortable:
        base_id = payload.pop("explicit_id", None) or payload.get("label") or f"{payload['kind']}-{signature}"
        occurrence = used_ids.get(base_id, 0) + 1
        used_ids[base_id] = occurrence
        series_id = base_id if occurrence == 1 else f"{base_id}#{occurrence}"
        result.append(SeriesManifest(series_id=series_id, axis_slot=axis_slot, **payload))
    return result


def _extract_series(
    ax: Any,
    config: MetricConfig,
    axis_slot: Optional[str],
) -> Tuple[List[SeriesManifest], List[dict]]:
    payloads: List[dict] = []
    annotations: List[dict] = []
    skipped_lines: set[int] = set()
    skipped_collections: set[int] = set()

    for container in ax.containers:
        if isinstance(container, BarContainer):
            payloads.append(_bar_payload(container, ax, config))
        elif isinstance(container, ErrorbarContainer):
            payload = _errorbar_payload(container, config)
            if payload is not None:
                payloads.append(payload)
            data_line, caplines, barlinecols = container.lines
            if data_line is not None:
                skipped_lines.add(id(data_line))
            skipped_lines.update(id(line) for line in caplines)
            skipped_collections.update(id(collection) for collection in barlinecols)

    for line in ax.lines:
        if id(line) not in skipped_lines:
            if _is_data_line(ax, line):
                payloads.append(_line_payload(line, config))
            else:
                annotations.append(_annotation_payload(line, config))

    for collection in ax.collections:
        if id(collection) in skipped_collections:
            continue
        if isinstance(collection, PathCollection):
            payloads.append(_scatter_payload(collection, config))
        elif isinstance(collection, PolyCollection):
            payloads.append(_poly_payload(collection, config))

    for image in ax.images:
        payloads.append(_image_payload(image, config))

    annotations.sort(key=stable_json_dumps)
    return _finalize_series(payloads, axis_slot), annotations


def _legend_entry(handle: Any, label: str) -> LegendEntry:
    colors = _artist_colors(handle)
    linestyle = str(handle.get_linestyle()) if hasattr(handle, "get_linestyle") else None
    marker = str(handle.get_marker()) if hasattr(handle, "get_marker") else None
    return LegendEntry(
        label=str(label),
        color=_primary_color(colors) or (colors[0] if colors else None),
        linestyle=linestyle,
        marker=marker,
    )


def _extract_legend(legend: Any) -> LegendManifest:
    if legend is None:
        return LegendManifest(present=False, location=None, entries=[])
    handles = getattr(legend, "legend_handles", None)
    if handles is None:
        handles = getattr(legend, "legendHandles", [])
    labels = [text.get_text() for text in legend.get_texts()]
    entries = [_legend_entry(handle, label) for handle, label in zip(handles, labels)]
    entries.sort(key=lambda entry: (entry.label, entry.color or "", entry.linestyle or "", entry.marker or ""))
    return LegendManifest(
        present=True,
        location=json_value(getattr(legend, "_loc", None)),
        entries=entries,
    )


def _axis_properties(ax: Any, dimension: str, config: MetricConfig) -> AxisProperties:
    if dimension == "x":
        axis = ax.xaxis
        label = ax.get_xlabel()
        scale = ax.get_xscale()
        limits = ax.get_xlim()
        ticks = ax.get_xticklabels()
    else:
        axis = ax.yaxis
        label = ax.get_ylabel()
        scale = ax.get_yscale()
        limits = ax.get_ylim()
        ticks = ax.get_yticklabels()
    return AxisProperties(
        label=str(label or ""),
        unit=extract_unit(str(label or ""), axis.get_units()),
        scale=str(scale),
        limits=json_value(limits, config.float_precision),
        tick_labels=[tick.get_text() for tick in ticks],
        offset_text=axis.get_offset_text().get_text(),
    )


def _axis_role(ax: Any) -> str:
    label = str(ax.get_label() or "")
    if label == "<colorbar>" or getattr(ax, "_colorbar", None) is not None:
        return "colorbar"
    return "panel"


def _same_bbox(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    return all(
        abs(float(a) - float(b)) <= tolerance
        for a, b in zip(left.get_position().bounds, right.get_position().bounds)
    )


def _axes_are_twins(left: Any, right: Any) -> bool:
    if not _same_bbox(left, right):
        return False
    shared_x = left.get_shared_x_axes().joined(left, right)
    shared_y = left.get_shared_y_axes().joined(left, right)
    return bool(shared_x or shared_y)


def _twin_roots(axes: Sequence[Any]) -> dict[int, Optional[int]]:
    roots: dict[int, Optional[int]] = {}
    for source_index, axis in enumerate(axes):
        if _axis_role(axis) == "colorbar":
            roots[source_index] = None
            continue
        root = source_index
        for previous_index in range(source_index):
            previous_root = roots.get(previous_index)
            if previous_root is not None and _axes_are_twins(axis, axes[previous_index]):
                root = previous_root
                break
        roots[source_index] = root
    return roots


def _unique_identifier(base: str, used: dict[str, int]) -> str:
    occurrence = used.get(base, 0) + 1
    used[base] = occurrence
    return base if occurrence == 1 else f"{base}#{occurrence}"


def _axis_sort_key(item: Tuple[int, Any]) -> Tuple[Any, ...]:
    source_index, ax = item
    x0, y0, width, height = ax.get_position().bounds
    role_order = 1 if _axis_role(ax) == "colorbar" else 0
    return (
        role_order,
        -round(float(y0), 9),
        round(float(x0), 9),
        -round(float(height), 9),
        -round(float(width), 9),
        source_index,
    )


def _axis_identifier(ax: Any, role: str, generated_index: int) -> str:
    gid = _visible_label(ax.get_gid())
    if gid:
        return gid
    label = _visible_label(ax.get_label())
    if label and label != "<colorbar>":
        return label
    prefix = "panel" if role == "panel" else "axis"
    return f"{prefix}-{generated_index}"


def extract_figure_manifest(
    fig: Any,
    config: Optional[MetricConfig | dict] = None,
) -> FigureManifest:
    cfg = coerce_metric_config(config)
    fig.canvas.draw()

    sorted_axes = sorted(enumerate(fig.axes), key=_axis_sort_key)
    twin_roots = _twin_roots(fig.axes)
    primary_axes = [
        (source_index, axis)
        for source_index, axis in enumerate(fig.axes)
        if twin_roots[source_index] == source_index
    ]
    primary_axes.sort(key=_axis_sort_key)

    used_axis_ids: dict[str, int] = {}
    panel_id_by_root: dict[int, str] = {}
    for panel_index, (source_index, axis) in enumerate(primary_axes):
        base_id = _axis_identifier(axis, "panel", panel_index)
        panel_id_by_root[source_index] = _unique_identifier(base_id, used_axis_ids)

    axes: List[AxisManifest] = []
    auxiliary_index = 0
    secondary_counts: dict[int, int] = {}

    for stable_index, (source_index, ax) in enumerate(sorted_axes):
        root = twin_roots[source_index]
        if root is None:
            role = "colorbar"
            panel_id = None
            parent_axis_id = None
            axis_slot = None
            base_id = _axis_identifier(ax, role, auxiliary_index)
            axis_id = _unique_identifier(base_id, used_axis_ids)
            auxiliary_index += 1
        elif root == source_index:
            role = "panel"
            panel_id = panel_id_by_root[root]
            parent_axis_id = None
            axis_slot = "primary"
            axis_id = panel_id
        else:
            role = "secondary"
            panel_id = panel_id_by_root[root]
            parent_axis_id = panel_id
            axis_slot = "secondary"
            secondary_index = secondary_counts.get(root, 0) + 1
            secondary_counts[root] = secondary_index
            explicit_id = _visible_label(ax.get_gid())
            base_id = explicit_id or f"{panel_id}:secondary-{secondary_index}"
            axis_id = _unique_identifier(base_id, used_axis_ids)

        bbox = json_value(ax.get_position().bounds, cfg.float_precision)
        series, annotations = _extract_series(ax, cfg, axis_slot)
        axes.append(
            AxisManifest(
                axis_id=axis_id,
                panel_id=panel_id,
                parent_axis_id=parent_axis_id,
                axis_slot=axis_slot,
                index=stable_index,
                source_index=source_index,
                role=role,
                bbox=bbox,
                title=str(ax.get_title() or ""),
                x_axis=_axis_properties(ax, "x", cfg),
                y_axis=_axis_properties(ax, "y", cfg),
                legend=_extract_legend(ax.get_legend()),
                series=series,
                annotations=annotations,
            )
        )

    figure_legends = [_extract_legend(legend) for legend in fig.legends]
    figure_legends.sort(key=lambda legend: stable_json_dumps(legend.to_dict()))
    size_inches = json_value(fig.get_size_inches(), cfg.float_precision)
    return FigureManifest(
        schema_version=FIGURE_MANIFEST_SCHEMA_VERSION,
        figure={
            "size_inches": size_inches,
            "dpi": json_value(fig.dpi, cfg.float_precision),
            "facecolor": _color_hex(fig.get_facecolor()),
        },
        axes=axes,
        figure_legends=figure_legends,
    )


def combine_figure_manifests(
    manifests: Mapping[str, FigureManifest | Mapping[str, Any]],
) -> FigureManifest:
    if not manifests:
        raise ValueError("At least one panel manifest is required")
    combined_axes: List[AxisManifest] = []
    combined_legends: List[LegendManifest] = []
    first_figure: dict[str, Any] | None = None
    for panel_id, raw_manifest in manifests.items():
        manifest = (
            raw_manifest
            if isinstance(raw_manifest, FigureManifest)
            else FigureManifest.from_dict(raw_manifest)
        )
        if first_figure is None:
            first_figure = dict(manifest.figure)
        primary_count = sum(axis.role == "panel" for axis in manifest.axes)
        if primary_count != 1:
            raise ValueError(
                f"Panel {panel_id!r} must contain exactly one primary axis, "
                f"found {primary_count}"
            )
        for axis in manifest.axes:
            if axis.role == "panel":
                axis_id = str(panel_id)
                logical_panel = str(panel_id)
                parent_axis_id = None
            elif axis.role == "secondary":
                suffix = axis.axis_id.split(":")[-1]
                axis_id = f"{panel_id}:{suffix}"
                logical_panel = str(panel_id)
                parent_axis_id = str(panel_id)
            else:
                axis_id = f"{panel_id}:{axis.axis_id}"
                logical_panel = axis.panel_id
                parent_axis_id = axis.parent_axis_id
            combined_axes.append(
                replace(
                    axis,
                    axis_id=axis_id,
                    panel_id=logical_panel,
                    parent_axis_id=parent_axis_id,
                    index=len(combined_axes),
                )
            )
        combined_legends.extend(manifest.figure_legends)
    return FigureManifest(
        schema_version=FIGURE_MANIFEST_SCHEMA_VERSION,
        figure=first_figure or {},
        axes=combined_axes,
        figure_legends=combined_legends,
    )
