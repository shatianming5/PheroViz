from __future__ import annotations

import textwrap


def _slot(body: str) -> str:
    """Normalize multiline slot code."""

    return textwrap.dedent(body).strip()


AREA_MARK_BODY = """
arts = []
ov = (spec.get('overlays') or [{}])[0]
x = ov.get('x')
y = ov.get('y')
grp = ov.get('group')
style = ov.get('style') or {}
alpha = style.get('alpha', 0.6)
meta = ctx.setdefault('_v2_meta', {})
if not x or x not in df.columns or not y or y not in df.columns:
    return arts
is_time = pd.api.types.is_datetime64_any_dtype(df[x])
is_numeric = np.issubdtype(df[x].dtype, np.number)
line_width = (spec.get('theme') or {}).get('line_width', 1.5)
if grp and grp in df.columns:
    for g_value, subset in df.groupby(grp, dropna=False):
        ordered = subset[[x, y]].dropna().sort_values(x)
        if ordered.empty:
            continue
        xv = ordered[x]
        if is_time:
            xa = pd.to_datetime(xv, errors='coerce').to_numpy()
        elif is_numeric:
            xa = ordered[x].to_numpy()
        else:
            cats = meta.get('x_categories') or list(pd.Index(df[x].astype(str).unique()))
            xa = pd.Categorical(ordered[x].astype(str), categories=cats, ordered=True).codes
        ya = pd.to_numeric(ordered[y], errors='coerce').to_numpy()
        if len(ya) == 0:
            continue
        patch = ax.fill_between(xa, ya, alpha=alpha, label=str(g_value), linewidth=line_width)
        arts.append(patch)
else:
    ordered = df[[x, y]].dropna().sort_values(x)
    if ordered.empty:
        return arts
    xv = ordered[x]
    if is_time:
        xa = pd.to_datetime(xv, errors='coerce').to_numpy()
    elif is_numeric:
        xa = ordered[x].to_numpy()
    else:
        cats = meta.get('x_categories') or list(pd.Index(df[x].astype(str).unique()))
        xa = pd.Categorical(ordered[x].astype(str), categories=cats, ordered=True).codes
    ya = pd.to_numeric(ordered[y], errors='coerce').to_numpy()
    if len(ya):
        patch = ax.fill_between(xa, ya, alpha=alpha, linewidth=line_width)
        arts.append(patch)
return arts
"""


DEFAULT_STAGE_SLOTS_V2 = {
    "L1": {
        "slots": {
            "spec.compose": _slot(
                """
                cols = [str(c) for c in (ctx.get('df_columns') or list(((ctx.get('data_profile') or {}).get('columns') or {}).keys()))]
                if not cols:
                    return spec or {}
                profile_columns = ((ctx.get('data_profile') or {}).get('columns') or {})
                dtypes = ctx.get('df_dtypes') or {}
                unique_counts = ctx.get('df_unique_counts') or {}
                row_count = int(ctx.get('row_count', 0))
                intent = intent or {}
                time_tokens = ('date', 'time', 'year', 'month', 'week', 'day', 'quarter', 'season', '年', '月', '日', '季度', '周', '季', '时', '小时')
                ratio_tokens = ('rate', 'ratio', 'share', 'percent', 'pct', '%', '率', '占比', '份额', '渗透', '占用', '百分比')

                def _dtype(name: str) -> str:
                    source = str(profile_columns.get(name, "")) or str(dtypes.get(name, ""))
                    return source.lower()

                def _is_numeric(name: str) -> bool:
                    dtype = _dtype(name)
                    return dtype.startswith('num') or 'float' in dtype or 'int' in dtype

                def _unique(name: str) -> int:
                    try:
                        return int(unique_counts.get(name, row_count))
                    except Exception:
                        return row_count

                def _is_time_like(name: str) -> bool:
                    dtype = _dtype(name)
                    lower = str(name).lower()
                    if dtype.startswith('date') or 'datetime' in dtype:
                        return True
                    if any(token in lower for token in time_tokens):
                        return True
                    if _is_numeric(name) and _unique(name) <= max(12, row_count // 4 + 1) and any(token in lower for token in ('year', 'quarter', 'week', '月', '周', '季')):
                        return True
                    return False

                def _is_ratio_like(name: str) -> bool:
                    lower = str(name).lower()
                    return any(token in lower for token in ratio_tokens)

                num_cols = [c for c in cols if _is_numeric(c)]
                raw_cat_cols = [c for c in cols if c not in num_cols]
                cat_cols = list(raw_cat_cols)
                for candidate in num_cols:
                    if candidate not in cat_cols and (_is_time_like(candidate) or _unique(candidate) <= max(12, row_count // 2 + 1)):
                        cat_cols.append(candidate)

                x = intent.get('x')
                if x not in cols:
                    time_candidates = [c for c in cols if _is_time_like(c)]
                    if time_candidates:
                        x = time_candidates[0]
                    elif cat_cols:
                        x = cat_cols[0]
                    elif cols:
                        x = cols[0]
                    else:
                        x = None

                y = intent.get('y')
                if y not in cols or y == x:
                    numeric_candidates = [c for c in num_cols if c != x]
                    if not numeric_candidates and x in num_cols:
                        numeric_candidates = [x]
                    ratio_candidates = [c for c in numeric_candidates if _is_ratio_like(c)]
                    if ratio_candidates:
                        y = ratio_candidates[0]
                    elif numeric_candidates:
                        numeric_candidates.sort(key=lambda c: (0 if not _is_time_like(c) else 1, -_unique(c)))
                        y = numeric_candidates[0]
                    elif cols:
                        y = cols[-1]
                    else:
                        y = None

                group = intent.get('group')
                if group not in cols:
                    limit = max(3, min(12, int(row_count / 10) + 1)) if row_count else 12
                    group_candidates = [c for c in cat_cols if c != x and _unique(c) <= limit]
                    if not group_candidates:
                        group_candidates = [c for c in raw_cat_cols if c != x]
                    group = group_candidates[0] if group_candidates else None

                family = (intent.get('chart_family') or '').lower()
                if family in {'line', 'bar', 'scatter', 'area'}:
                    mark = 'area' if family == 'area' else family
                else:
                    if group and y and _is_ratio_like(y):
                        mark = 'area'
                    elif x and _is_time_like(x):
                        mark = 'line'
                    elif group:
                        mark = 'bar'
                    else:
                        mark = 'scatter' if y and _is_numeric(y) else 'line'

                variant = 'fill' if mark == 'area' else ('grouped' if (mark == 'bar' and group) else 'main')

                layout_titles = {
                    'top': f"{y} vs {x}" if x and y else (ctx.get('user_goal') or ''),
                    'left': None,
                    'right': None,
                    'bottom': None,
                }

                spec_out = {
                    'canvas': {'width': 960, 'height': 576, 'dpi': 300},
                    'overlays': [
                        {
                            'mark': mark,
                            'variant': variant,
                            'x': x,
                            'y': y,
                            'group': group,
                            'yaxis': 'left',
                            'style': {'alpha': 0.9},
                        }
                    ],
                    'scales': {
                        'x': {
                            'kind': 'time' if (x and (_dtype(x).startswith('date') or 'datetime' in _dtype(x))) else ('linear' if (x in num_cols and x not in cat_cols) else 'categorical'),
                            'range': None,
                            'breaks': None,
                        },
                        'y_left': {'kind': 'linear', 'range': [0, None], 'breaks': None},
                        'y_right': {'kind': 'linear', 'range': None},
                    },
                    'layout': {
                        'titles': layout_titles,
                        'title_align': 'left',
                        'legend': {'loc': 'best', 'ncol': 1, 'frame': False},
                        'grid': {'x': False, 'y': True, 'minor': True},
                        'panel_labels': [],
                    },
                    'theme': {
                        'font': 'Arial',
                        'fontsize': 9,
                        'axis_linewidth': 1.0,
                        'tick_len': 3.0,
                        'tick_width': 0.8,
                        'palette_global': 'ColorBlindSafe',
                        'line_width': 1.5,
                        'marker_size': 36,
                    },
                    'flags': {
                        'inherit_palette': True,
                        'legend_outside': 'auto',
                        'safe_log_y': True,
                        'max_overlays': 3,
                        'tick_density': 'normal',
                    },
                }
                return spec_out

            """),
            "spec.theme_defaults": "return spec",
        },
        "notes": "???? x/y/group/mark;????????",
    },
    "L2": {
        "slots": {
            "data.prepare": _slot(
                """
                ov = (spec.get('overlays') or [{}])[0]
                x = ov.get('x')
                y = ov.get('y')
                grp = ov.get('group')
                data = df[[c for c in [x, y, grp] if c and c in df.columns]].copy()
                meta = ctx.setdefault('_v2_meta', {})
                meta['x_name'] = x
                meta['y_name'] = y
                meta['group_name'] = grp
                meta['row_count_prepare'] = int(len(data))
                is_time = False
                if x and x in data.columns:
                    try:
                        converted = pd.to_datetime(data[x], errors='coerce', infer_datetime_format=True)
                        if converted.notna().mean() >= 0.8:
                            data[x] = converted
                            is_time = True
                    except Exception:
                        pass
                meta['x_is_time'] = bool(is_time)
                meta['x_is_numeric'] = bool(x in data.columns and np.issubdtype(data[x].dtype, np.number))
                required = [col for col in [x, y] if col in data.columns]
                if required:
                    data = data.dropna(subset=required)
                if x and x in data.columns and not is_time and not np.issubdtype(data[x].dtype, np.number):
                    labels = data[x].astype(str)
                    nunq = labels.nunique(dropna=False)
                    is_ratio = False
                    if y and y in data.columns:
                        numeric_y = pd.to_numeric(data[y], errors='coerce')
                        if numeric_y.notna().any():
                            is_ratio = (
                                str(y).lower().endswith('%')
                                or ('?' in str(y))
                                or ('ratio' in str(y).lower())
                                or ('rate' in str(y).lower())
                                or ('share' in str(y).lower())
                                or ('percent' in str(y).lower())
                                or ('pct' in str(y).lower())
                                or (numeric_y.between(0, 1).mean() > 0.6)
                            )
                    meta['y_is_ratio'] = bool(is_ratio)
                    if nunq <= 30 and y and y in data.columns:
                        agg = 'mean' if is_ratio else 'sum'
                        ordered = (
                            data.groupby(labels, dropna=False)[y]
                            .agg(agg)
                            .sort_values(ascending=True)
                            .index.astype(str)
                        )
                        cat = pd.Categorical(labels, categories=list(ordered), ordered=True)
                        data = data.assign(__xcat=cat).sort_values('__xcat').drop(columns='__xcat')
                        meta['x_categories'] = list(ordered)
                    else:
                        sequence = []
                        seen = {}
                        for val in labels:
                            sval = str(val)
                            if sval not in seen:
                                seen[sval] = len(seen)
                            sequence.append(seen[sval])
                        order_idx = np.argsort(sequence)
                        data = data.iloc[order_idx]
                        meta.setdefault('x_categories', list(dict.fromkeys(labels.tolist())))
                else:
                    if x and x in data.columns:
                        data = data.sort_values(by=x)
                        if not np.issubdtype(data[x].dtype, np.number) and not meta.get('x_is_time'):
                            meta['x_categories'] = list(dict.fromkeys(data[x].astype(str).tolist()))
                return data
                """
            ),
            "data.aggregate": _slot(
                """
                ov = (spec.get('overlays') or [{}])[0]
                x = ov.get('x')
                y = ov.get('y')
                grp = ov.get('group')
                meta = ctx.setdefault('_v2_meta', {})
                meta.setdefault('y_is_ratio', bool(meta.get('y_is_ratio')))
                keys = [col for col in [x, grp] if col and col in df.columns]
                data = df.copy()
                if y and y in data.columns and keys and data.duplicated(keys).any():
                    is_ratio = bool(meta.get('y_is_ratio'))
                    agg_func = 'mean' if is_ratio else 'sum'
                    data = data.groupby(keys, dropna=False)[y].agg(agg_func).reset_index()
                meta['row_count_aggregate'] = int(len(data))
                return data
                """
            ),
            "data.encode": _slot(
                """
                meta = ctx.setdefault('_v2_meta', {})
                meta['row_count_encode'] = int(len(df))
                meta['encoded_sample'] = df.head(50).to_dict(orient='list')
                return df
                """
            ),
        },
        "notes": "?????;??/???? mean,??? sum;?????",
    },
    "L3": {
        "slots": {
            "marks.area.fill": _slot(AREA_MARK_BODY),
            "marks.area.main": _slot(AREA_MARK_BODY),
            "marks.line.main": _slot(
                """
                arts = []
                ov = (spec.get('overlays') or [{}])[0]
                x = ov.get('x')
                y = ov.get('y')
                grp = ov.get('group')
                meta = ctx.setdefault('_v2_meta', {})
                if not x or x not in df.columns or not y or y not in df.columns:
                    return arts
                is_time = pd.api.types.is_datetime64_any_dtype(df[x])
                is_numeric = np.issubdtype(df[x].dtype, np.number)
                meta['x_is_time'] = meta.get('x_is_time') or bool(is_time)
                meta['x_is_numeric'] = meta.get('x_is_numeric') or bool(is_numeric)
                line_width = (spec.get('theme') or {}).get('line_width', 1.5)
                linestyles = ['-', '--', '-.', ':']
                if grp and grp in df.columns:
                    ordered_groups = list(pd.Index(df[grp].astype(str).unique()))
                    for idx, (g_value, subset) in enumerate(df.groupby(grp, dropna=False)):
                        xv = subset[x]
                        if is_time:
                            xa = pd.to_datetime(xv, errors='coerce')
                        elif is_numeric:
                            xa = xv.to_numpy()
                        else:
                            cats = meta.get('x_categories') or list(pd.Index(df[x].astype(str).unique()))
                            xa = pd.Categorical(xv.astype(str), categories=cats, ordered=True).codes
                        ya = subset[y].to_numpy()
                        linestyle = linestyles[idx % len(linestyles)] if len(ordered_groups) > 6 else '-'
                        (ln,) = ax.plot(xa, ya, alpha=0.95, linewidth=line_width, linestyle=linestyle, label=str(g_value))
                        arts.append(ln)
                else:
                    xv = df[x]
                    if is_time:
                        xa = pd.to_datetime(xv, errors='coerce')
                    elif is_numeric:
                        xa = xv.to_numpy()
                    else:
                        cats = meta.get('x_categories') or list(pd.Index(df[x].astype(str).unique()))
                        xa = pd.Categorical(xv.astype(str), categories=cats, ordered=True).codes
                    ya = df[y].to_numpy()
                    (ln,) = ax.plot(xa, ya, alpha=0.95, linewidth=line_width)
                    arts.append(ln)
                return arts
                """
            ),
            "marks.bar.grouped": _slot(
                """
                arts = []
                ov = (spec.get('overlays') or [{}])[0]
                x = ov.get('x')
                y = ov.get('y')
                grp = ov.get('group')
                if not x or x not in df.columns or not y or y not in df.columns:
                    return arts
                cats = pd.Index(df[x].astype(str).unique())
                base = np.arange(len(cats))
                style = ov.get('style') or {}
                alpha = style.get('alpha', 0.9)
                if grp and grp in df.columns:
                    groups = list(pd.Index(df[grp].astype(str).unique()))
                    width = 0.8 / max(1, len(groups))
                    for idx, g_value in enumerate(groups):
                        subset = df[df[grp].astype(str) == g_value]
                        values = []
                        for cat in cats:
                            match = subset[subset[x].astype(str) == cat]
                            num = pd.to_numeric(match[y], errors='coerce').sum()
                            values.append(float(num))
                        bars = ax.bar(base + idx * width, values, width=width, alpha=alpha, label=str(g_value))
                        arts.extend(list(bars))
                    ax.set_xticks(base + 0.4)
                else:
                    values = [float(v) for v in pd.to_numeric(df[y], errors='coerce').fillna(0).tolist()]
                    bars = ax.bar(base, values, width=0.8, alpha=alpha)
                    arts.extend(list(bars))
                    ax.set_xticks(base)
                ax.set_xticklabels(list(cats))
                ctx.setdefault('_v2_meta', {}).setdefault('x_categories', list(cats))
                return arts
                """
            ),
            "marks.scatter.main": _slot(
                """
                ov = (spec.get('overlays') or [{}])[0]
                x = ov.get('x')
                y = ov.get('y')
                if not x or x not in df.columns or not y or y not in df.columns:
                    return []
                is_numeric = np.issubdtype(df[x].dtype, np.number)
                is_time = pd.api.types.is_datetime64_any_dtype(df[x])
                meta = ctx.setdefault('_v2_meta', {})
                meta['x_is_time'] = meta.get('x_is_time') or bool(is_time)
                meta['x_is_numeric'] = meta.get('x_is_numeric') or bool(is_numeric)
                if is_numeric or is_time:
                    xa = df[x].to_numpy()
                else:
                    cats = meta.get('x_categories') or list(pd.Index(df[x].astype(str).unique()))
                    n = len(cats)
                    jitter = min(0.15, 1.0 / max(2, n))
                    codes = pd.Categorical(df[x].astype(str), categories=cats, ordered=True).codes
                    xa = codes + ((np.random.rand(len(codes)) - 0.5) * 2 * jitter)
                values = pd.to_numeric(df[y], errors='coerce').to_numpy()
                marker_size = (spec.get('theme') or {}).get('marker_size', 36)
                count = len(df)
                alpha = 0.85 if count < 1000 else (0.6 if count < 5000 else 0.4)
                sc = ax.scatter(xa, values, s=marker_size, alpha=alpha, marker=(ov.get('style') or {}).get('marker', 'o'))
                return [sc]
                """
            ),
            "scales.x.kind": _slot(
                """
                ov = (spec.get('overlays') or [{}])[0]
                x = ov.get('x')
                if not x or x not in df.columns:
                    return
                meta = ctx.setdefault('_v2_meta', {})
                is_numeric = np.issubdtype(df[x].dtype, np.number)
                is_time = pd.api.types.is_datetime64_any_dtype(df[x])
                meta['x_is_numeric'] = meta.get('x_is_numeric') or bool(is_numeric)
                meta['x_is_time'] = meta.get('x_is_time') or bool(is_time)
                if not (is_numeric or is_time):
                    cats = meta.get('x_categories')
                    if cats is None:
                        cats = list(dict.fromkeys(df[x].astype(str).tolist()))
                        meta['x_categories'] = cats
                    if cats:
                        ax_left.set_xlim(-0.5, len(cats) - 0.5)
                        if ax_right:
                            ax_right.set_xlim(-0.5, len(cats) - 0.5)
                """
            ),
            "scales.y_left.kind": _slot(
                """
                ov = (spec.get('overlays') or [{}])[0]
                y = ov.get('y')
                meta = ctx.setdefault('_v2_meta', {})
                if y and y in df.columns:
                    series = pd.to_numeric(df[y], errors='coerce').dropna()
                    if meta.get('y_is_ratio') is None and not series.empty:
                        name = str(y)
                        ratio_flag = (
                            name.lower().endswith('%')
                            or ('?' in name)
                            or ('ratio' in name.lower())
                            or ('rate' in name.lower())
                            or (series.between(0, 1).mean() > 0.6)
                        )
                        meta['y_is_ratio'] = bool(ratio_flag)
                    scale_requested = ((spec.get('scales') or {}).get('y_left') or {}).get('kind')
                    if scale_requested == 'log':
                        positive = series.gt(0).mean() if not series.empty else 0
                        if positive >= 0.2:
                            ax_left.set_yscale('log')
                            meta['y_scale'] = 'log'
                        else:
                            ax_left.set_yscale('linear')
                            meta['y_scale'] = 'linear'
                    else:
                        ax_left.set_yscale('linear')
                        meta['y_scale'] = 'linear'
                else:
                    ax_left.set_yscale('linear')
                    meta['y_scale'] = 'linear'
                """
            ),
            "scales.y_left.range": _slot(
                """
                rng = ((spec.get('scales') or {}).get('y_left') or {}).get('range')
                if isinstance(rng, list) and len(rng) == 2:
                    ax_left.set_ylim(bottom=rng[0], top=rng[1])
                else:
                    ov = (spec.get('overlays') or [{}])[0]
                    y = ov.get('y')
                    if y and y in df.columns:
                        series = pd.to_numeric(df[y], errors='coerce')
                        if series.notna().any():
                            lower = min(0.0, float(series.min()))
                            upper = float(series.max())
                            upper = upper * 1.1 if upper > 0 else (upper * 0.9 if upper < 0 else upper + 1)
                            ax_left.set_ylim(bottom=lower, top=upper)
                            ctx.setdefault('_v2_meta', {})['y_range'] = (lower, upper)
                """
            ),
        },
        "notes": "?????;????;??????;x ???????",
    },
    "L4": {
        "slots": {
            "axes.labels": _slot(
                """
                layout = spec.get('layout') or {}
                titles = layout.get('titles') or {}
                ov = (spec.get('overlays') or [{}])[0]
                x = ov.get('x')
                y = ov.get('y')
                title = titles.get('top') or ctx.get('user_goal') or ''
                ax_left.set_title(title, loc=layout.get('title_align', 'left'))
                ax_left.set_xlabel(str(x) if x else '')
                ax_left.set_ylabel(str(y) if y else '')
                """
            ),
            "axes.ticks": _slot(
                """
meta = ctx.get('_v2_meta', {}) or {}
mode = (spec.get('flags') or {}).get('tick_density', 'normal')
target = 4 if mode == 'compact' else (6 if mode == 'normal' else 8)
if meta.get('x_is_time') or meta.get('x_is_numeric'):
    ax_left.xaxis.set_major_locator(MaxNLocator(target))
else:
    cats = meta.get('x_categories')
    if cats:
        count = len(cats)
        idx = list(range(count))
        if count > 50:
            step = max(1, (count + 49) // 50)
            idx = list(range(0, count, step))
        labels = [str(cats[i]) for i in idx]
        ax_left.set_xticks(idx)
        ax_left.set_xticklabels(labels)
        if count > 8 or (labels and max(len(label) for label in labels) > 12):
            for tick in ax_left.get_xticklabels():
                tick.set_rotation(30)
                tick.set_ha('right')
            for tick in ax_left.get_xticklabels():
                text = str(tick.get_text())
                if len(text) > 16:
                    chunks = [text[i:i + 16] for i in range(0, len(text), 16)]
                    tick.set_text(chr(10).join(chunks))
ax_left.yaxis.set_major_locator(MaxNLocator(target))
if ax_right:
    ax_right.yaxis.set_major_locator(MaxNLocator(target))
"""
            ),
            "axes.formatter": _slot(
                """
                meta = ctx.get('_v2_meta', {}) or {}

                def _fmt_si(value: float) -> str:
                    abs_value = abs(value)
                    if abs_value >= 1e12:
                        return f"{value / 1e12:.1f}T"
                    if abs_value >= 1e9:
                        return f"{value / 1e9:.1f}B"
                    if abs_value >= 1e6:
                        return f"{value / 1e6:.1f}M"
                    if abs_value >= 1e3:
                        return f"{value / 1e3:.1f}K"
                    if value == int(value):
                        return f"{int(value)}"
                    return f"{value:.2f}"

                if meta.get('y_is_ratio'):
                    ax_left.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1%}"))
                    if ax_right:
                        ax_right.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1%}"))
                else:
                    ax_left.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt_si(v)))
                    if ax_right:
                        ax_right.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt_si(v)))
                """
            ),
            "grid.apply": _slot(
                """
                ax_left.grid(True, axis='y', which='major', alpha=0.3, linestyle='-')
                ax_left.grid(True, axis='y', which='minor', alpha=0.12, linestyle='--')
                if (spec.get('layout') or {}).get('grid', {}).get('x'):
                    ax_left.grid(True, axis='x', which='major', alpha=0.2, linestyle='--')
                """
            ),
            "legend.apply": _slot(
                """
                handles_left, labels_left = ax_left.get_legend_handles_labels()
                handles_right, labels_right = ax_right.get_legend_handles_labels() if ax_right else ([], [])
                labels_left = [str(label) for label in labels_left]
                labels_right = [str(label) for label in labels_right]
                if not handles_left and not handles_right:
                    return
                overlap = set(labels_left) & set(labels_right)
                if overlap:
                    labels_left = [f"{lbl} (L)" if lbl in overlap else lbl for lbl in labels_left]
                    labels_right = [f"{lbl} (R)" if lbl in overlap else lbl for lbl in labels_right]
                combined = []
                seen = set()
                for handle, label in list(zip(handles_left, labels_left)) + list(zip(handles_right, labels_right)):
                    if label in seen:
                        continue
                    combined.append((handle, label))
                    seen.add(label)
                if not combined:
                    return
                desired = ((spec.get('layout') or {}).get('legend') or {}).get('loc', 'best')
                if len(combined) >= 8:
                    ax_left.legend(
                        [h for h, _ in combined],
                        [lbl for _, lbl in combined],
                        loc='lower center',
                        bbox_to_anchor=(0.5, -0.2),
                        frameon=False,
                        ncol=2,
                    )
                else:
                    ax_left.legend(
                        [h for h, _ in combined],
                        [lbl for _, lbl in combined],
                        loc=desired,
                        frameon=False,
                    )
                """
            ),
            "annot.text_boxes": _slot(
                """
                breaks = ((spec.get('scales') or {}).get('y_left') or {}).get('breaks')
                if breaks:
                    fontsize = max(6, ((spec.get('theme') or {}).get('fontsize', 9)) - 1)
                    ax_left.text(0.01, 0.98, 'Note: broken y-axis', transform=ax_left.transAxes, va='top', ha='left', fontsize=fontsize)
                """
            ),
        },
        "notes": "??/??/??;????;SI/???;?????",
    },
}






