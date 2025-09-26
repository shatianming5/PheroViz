from __future__ import annotations

import json
from textwrap import indent
from typing import Any, Dict, Iterable, List

SUPPORTED_CHARTS = {
    "line",
    "bar",
    "stacked_bar",
    "area",
    "pie",
    "scatter",
    "hist",
    "box",
    "heatmap",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _indent(lines: Iterable[str], spaces: int = 4) -> List[str]:
    prefix = " " * spaces
    return [prefix + line if line else "" for line in lines]


def _render_filter(expr: Dict[str, Any]) -> List[str]:
    col = _json(expr.get("column"))
    op = (expr.get("op") or "").lower()
    value = expr.get("value")
    if op in {"==", "!=", ">", "<", ">=", "<="}:
        return [f"df = df[df[{col}] {op} {_json(value)}]"]
    if op == "in" and isinstance(value, list):
        return [f"df = df[df[{col}].isin({_json(value)})]"]
    if op == "not in" and isinstance(value, list):
        return [f"df = df[~df[{col}].isin({_json(value)})]"]
    if op == "between" and isinstance(value, list) and len(value) >= 2:
        return [
            f"df = df[df[{col}].between({_json(value[0])}, {_json(value[1])})]",
        ]
    if op == "contains":
        return [f"df = df[df[{col}].astype(str).str.contains({_json(value)}, na=False)]"]
    return []


def _render_transforms(transforms: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for transform in transforms:
        op = (transform.get("op") or "").lower()
        if op == "filter" and isinstance(transform.get("expr"), dict):
            lines.extend(_render_filter(transform["expr"]))
        elif op == "derive":
            target = _json(transform.get("as"))
            expr = transform.get("expr") or ""
            if isinstance(expr, str) and expr.startswith("to_period"):
                parts = expr.replace("to_period", "").strip("()")
                args = [p.strip().strip("'\"") for p in parts.split(",") if p.strip()]
                if args:
                    source = _json(args[0])
                    freq = _json(args[1]) if len(args) > 1 else _json("M")
                    lines.append(
                        f"df[{target}] = pd.to_datetime(df[{source}], errors='coerce').dt.to_period({freq}).astype(str)"
                    )
            else:
                # fallback: copy source column
                source = transform.get("expr")
                if isinstance(source, str):
                    lines.append(f"df[{target}] = df.get({_json(source)}, None)")
        elif op == "topk":
            group = transform.get("group_by") or []
            order = transform.get("order_by") or {}
            order_col = _json(order.get("col")) if order else _json(transform.get("by"))
            descending = order.get("desc", True) if order else True
            k = int(transform.get("k") or 5)
            group_expr = _json(group)
            lines.append("df = df.sort_values(by=" + order_col + f", ascending={not descending})")
            if group:
                lines.append(
                    "df = df.groupby(" + group_expr + ", dropna=False).head(" + str(max(1, k)) + ")"
                )
            else:
                lines.append("df = df.head(" + str(max(1, k)) + ")")
        elif op == "aggregate":
            group_by = transform.get("group_by") or []
            measures = transform.get("measures") or []
            if measures:
                measure = measures[0]
                col = _json(measure.get("col"))
                agg = measure.get("agg") or "sum"
                alias = _json(measure.get("as") or measure.get("col"))
                group_expr = _json(group_by)
                lines.append(
                    "df = df.groupby(" + group_expr + ", dropna=False)[" + col + "].agg('" + agg + "').reset_index()"
                )
                lines.append(f"df = df.rename(columns={{ {col}: {alias} }})")
        elif op == "sort":
            by = transform.get("by") or []
            if by:
                cols = [_json(item.get("col")) for item in by]
                ascending = [not item.get("desc", False) for item in by]
                lines.append(
                    "df = df.sort_values(by=[" + ", ".join(cols) + "], ascending=" + str(ascending) + ")"
                )
    return lines


def _render_constraints(constraints: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    filters = constraints.get("filters") or []
    for expr in filters:
        if isinstance(expr, dict):
            lines.extend(_render_filter(expr))
    time_range = constraints.get("time_range") or []
    if len(time_range) == 2 and constraints.get("time_field"):
        col = _json(constraints.get("time_field"))
        start = _json(time_range[0])
        end = _json(time_range[1])
        lines.append(
            f"df = df[pd.to_datetime(df[{col}], errors='coerce').between(pd.to_datetime({start}), pd.to_datetime({end}))]"
        )
    return lines


def _chart_lines(spec: Dict[str, Any]) -> List[str]:
    chart_type = (spec.get("chart_type") or "bar").lower()
    if chart_type not in SUPPORTED_CHARTS:
        chart_type = "bar"

    encoding = spec.get("encoding") or {}
    dimensions = spec.get("dimensions") or []
    measures = spec.get("measures") or []

    x_field = encoding.get("x") or (dimensions[0] if dimensions else None)
    y_field = encoding.get("y") or (measures[0] if measures else None)
    series_field = encoding.get("series")

    lines: List[str] = ["fig, ax = plt.subplots(figsize=(8, 5))"]
    if chart_type != "heatmap":
        lines.append("ax.grid(True, alpha=0.25)")
    if y_field:
        lines.append(f"df[{_json(y_field)}] = pd.to_numeric(df[{_json(y_field)}], errors='coerce')")

    if chart_type in {"line", "area"}:
        if not x_field or not y_field:
            lines.append("raise ValueError('Line chart requires x and y fields.')")
        else:
            x_expr = _json(x_field)
            y_expr = _json(y_field)
            if series_field:
                series_expr = _json(series_field)
                lines.extend(
                    [
                        f"for key, group in df.groupby({series_expr}, dropna=False):",
                        "    group = group.sort_values(by=" + x_expr + ")",
                        "    if group.empty:",
                        "        continue",
                        ("    ax.fill_between" if chart_type == "area" else "    ax.plot")
                        + "(group[" + x_expr + "], group[" + y_expr + "], label=str(key))",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"df = df.sort_values(by={x_expr})",
                        ("ax.fill_between" if chart_type == "area" else "ax.plot")
                        + "(df[" + x_expr + "], df[" + y_expr + "], label=str('Series'))",
                    ]
                )
    elif chart_type in {"bar", "stacked_bar"}:
        if not x_field or not y_field:
            lines.append("raise ValueError('Bar chart requires x and y fields.')")
        else:
            x_expr = _json(x_field)
            y_expr = _json(y_field)
            lines.append("df = df.dropna(subset=[" + y_expr + "]).copy()")
            if series_field:
                series_expr = _json(series_field)
                lines.extend(
                    [
                        "pivot = df.pivot_table(index="
                        + x_expr
                        + ", columns="
                        + series_expr
                        + ", values="
                        + y_expr
                        + ", aggfunc='sum', fill_value=0)",
                        "positions = range(len(pivot.index))",
                    ]
                )
                if chart_type == "stacked_bar":
                    lines.extend(
                        [
                            "bottom = None",
                            "for label in pivot.columns:",
                            "    values = pivot[label].values",
                            "    ax.bar(pivot.index, values, bottom=bottom, label=str(label))",
                            "    bottom = values if bottom is None else bottom + values",
                        ]
                    )
                else:
                    lines.extend(
                        [
                            "n = len(pivot.columns)",
                            "width = 0.8 / max(n, 1)",
                            "for offset, label in enumerate(pivot.columns):",
                            "    values = pivot[label].values",
                            "    ax.bar([p + offset * width for p in positions], values, width=width, label=str(label))",
                            "ax.set_xticks([p + (n-1) * width / 2 for p in positions])",
                            "ax.set_xticklabels(pivot.index)",
                        ]
                    )
            else:
                lines.append(
                    "summary = df.groupby(" + x_expr + ", dropna=False)[" + y_expr + "].sum().reset_index()"
                )
                lines.append("ax.bar(summary[" + x_expr + "], summary[" + y_expr + "])" )
    elif chart_type == "scatter":
        x_expr = _json(x_field or dimensions[0] if dimensions else None)
        y_expr = _json(y_field or measures[0] if measures else None)
        if not x_expr or not y_expr:
            lines.append("raise ValueError('Scatter chart requires x and y fields.')")
        else:
            lines.append("ax.scatter(df[" + x_expr + "], df[" + y_expr + "], alpha=0.7)")
    elif chart_type == "hist":
        target = _json(y_field or measures[0] if measures else None)
        if not target:
            lines.append("raise ValueError('Histogram requires a numeric target field.')")
        else:
            bins = spec.get("bins") or 20
            lines.append("ax.hist(df[" + target + "].dropna(), bins=" + str(bins) + ", color='#4e79a7', alpha=0.9)")
    elif chart_type == "box":
        target = _json(y_field or measures[0] if measures else None)
        if not target:
            lines.append("raise ValueError('Box plot requires a target field.')")
        else:
            if x_field:
                x_expr = _json(x_field)
                lines.extend(
                    [
                        "groups = [g[" + target + "].dropna().values for _, g in df.groupby(" + x_expr + ", dropna=False)]",
                        "labels = [str(k) for k in df.groupby(" + x_expr + ", dropna=False).groups.keys()]",
                        "ax.boxplot(groups, labels=labels, vert=True)",
                    ]
                )
            else:
                lines.append("ax.boxplot(df[" + target + "].dropna())")
    elif chart_type == "pie":
        label_field = x_field or (dimensions[0] if dimensions else None)
        value_field = y_field or (measures[0] if measures else None)
        if not label_field or not value_field:
            lines.append("raise ValueError('Pie chart requires label and value fields.')")
        else:
            lines.extend(
                [
                    "summary = df.groupby(" + _json(label_field) + ", dropna=False)[" + _json(value_field) + "].sum()",
                    "summary = summary[summary > 0]",
                    "ax.pie(summary.values, labels=summary.index.astype(str), autopct='%1.1f%%', startangle=90)",
                    "ax.axis('equal')",
                ]
            )
    elif chart_type == "heatmap":
        dim_rows = dimensions[0] if dimensions else None
        dim_cols = dimensions[1] if len(dimensions) > 1 else series_field or None
        value_field = y_field or (measures[0] if measures else None)
        if not dim_rows or not dim_cols or not value_field:
            lines.append("raise ValueError('Heatmap requires two dimensions and one value field.')")
        else:
            lines.extend(
                [
                    "pivot = df.pivot_table(index="
                    + _json(dim_rows)
                    + ", columns="
                    + _json(dim_cols)
                    + ", values="
                    + _json(value_field)
                    + ", aggfunc='mean', fill_value=0)",
                    "im = ax.imshow(pivot.values, aspect='auto', cmap='viridis')",
                    "ax.set_xticks(range(len(pivot.columns)))",
                    "ax.set_xticklabels(pivot.columns, rotation=45, ha='right')",
                    "ax.set_yticks(range(len(pivot.index)))",
                    "ax.set_yticklabels(pivot.index)",
                    "fig.colorbar(im, ax=ax)",
                ]
            )

    legend = spec.get("legend", True)
    title = spec.get("title")
    xlabel = spec.get("xlabel") or (x_field or "")
    ylabel = spec.get("ylabel") or (y_field or "")

    lines.extend(
        [
            "ax.set_title(" + _json(title) + ")" if title else "",
            "ax.set_xlabel(" + _json(xlabel) + ")" if xlabel else "",
            "ax.set_ylabel(" + _json(ylabel) + ")" if ylabel else "",
            "if " + str(bool(legend)) + ":",
            "    ax.legend()",
        ]
    )

    return [line for line in lines if line]


def render_code_from_spec(spec: Dict[str, Any]) -> str:
    lines: List[str] = [
        "def generate_chart(dataset):",
        "    df = dataset.copy()",
        "    if df.empty:",
        "        raise ValueError('Dataset is empty.')",
    ]

    lines += _indent(_render_transforms(spec.get("transforms") or []))
    lines += _indent(_render_constraints(spec.get("constraints") or {}))
    lines.append("    if df.empty:")
    lines.append("        raise ValueError('Dataset is empty after preprocessing.')")

    lines += _indent(_chart_lines(spec))

    lines += _indent(
        [
            "buf = BytesIO()",
            "fig.tight_layout()",
            "fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)",
            "plt.close(fig)",
            "buf.seek(0)",
            "return buf.getvalue()",
        ]
    )

    return "\n".join(lines) + "\n"
