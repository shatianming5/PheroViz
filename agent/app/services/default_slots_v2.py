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
alpha = float(style.get('alpha', 0.55))
meta = ctx.setdefault('_v2_meta', {})
if not x or x not in df.columns or not y or y not in df.columns:
    return arts
area_meta = meta.get('area_style') or {}
base_z = float(area_meta.get('z_level', float(style.get('zorder', 1))))
theme = spec.get('theme') or {}
palette_name = theme.get('palette_global', 'tab10')
palette_cache = meta.setdefault('palette_cache', {})
if grp and grp in df.columns:
    need = max(1, int(df[grp].astype(str).nunique()))
else:
    need = 1
colors = palette_cache.get(palette_name)
if not colors or len(colors) < need:
    cmap = plt.get_cmap(palette_name)
    colors = [cmap(i) for i in np.linspace(0, 1, max(need, 1))]
    palette_cache[palette_name] = colors
is_time = pd.api.types.is_datetime64_any_dtype(df[x])
is_numeric = np.issubdtype(df[x].dtype, np.number)
fill_alpha_base = float(area_meta.get('alpha_fill', alpha))
line_alpha_base = float(area_meta.get('alpha_line', min(0.95, fill_alpha_base + 0.2)))
line_width = float(area_meta.get('line_width', theme.get('line_width', 1.5)))
def _compose(color_tuple, mix_ratio, alpha_value):
    rgb = np.array(color_tuple[:3], dtype=float)
    if mix_ratio > 0:
        rgb = rgb * (1.0 - mix_ratio) + np.ones_like(rgb) * mix_ratio
    rgb = np.clip(rgb, 0.0, 1.0)
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), float(alpha_value))
if grp and grp in df.columns:
    groups = list(df[grp].astype(str).unique())
    attenuate = 0.92 if len(groups) > 1 else 1.0
    soften_step = 0.1 if len(groups) > 1 else 0.0
    for idx, (g_value, subset) in enumerate(df.groupby(grp, dropna=False)):
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
        base_color = colors[idx % len(colors)]
        fill_alpha = max(0.25, min(0.95, fill_alpha_base * (attenuate ** idx)))
        line_alpha = max(fill_alpha, min(0.95, line_alpha_base * (attenuate ** idx)))
        mix_ratio = min(0.35, soften_step * idx)
        face = _compose(base_color, mix_ratio, fill_alpha)
        edge = _compose(base_color, mix_ratio * 0.5, 1.0)
        z_fill = base_z + idx * 0.6
        patch = ax.fill_between(
            xa,
            ya,
            alpha=face[3],
            label=str(g_value),
            color=face,
            zorder=z_fill,
        )
        arts.append(patch)
        (line_obj,) = ax.plot(
            xa,
            ya,
            color=edge,
            linewidth=line_width,
            alpha=line_alpha,
            zorder=z_fill + 0.8,
            label='_nolegend_',
        )
        arts.append(line_obj)
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
    base_color = colors[0]
    face = _compose(base_color, 0.1, fill_alpha_base)
    edge = _compose(base_color, 0.05, 1.0)
    patch = ax.fill_between(xa, ya, alpha=face[3], color=face, zorder=base_z)
    arts.append(patch)
    (line_obj,) = ax.plot(
        xa,
        ya,
        color=edge,
        linewidth=line_width,
        alpha=line_alpha_base,
        zorder=base_z + 0.8,
        label='_nolegend_',
    )
    arts.append(line_obj)
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
                meta = ctx.setdefault('_v2_meta', {})
                meta.clear()
                meta['row_count'] = row_count
                time_tokens = ('date', 'time', 'year', 'month', 'week', 'day', 'quarter', 'season', 'å¹´', 'æœˆ', 'æ—¥', 'å­£åº¦', 'å‘¨', 'å­£', 'æ—¶', 'å°æ—¶')
                ratio_tokens = ('rate', 'ratio', 'share', 'percent', 'pct', '%', 'çŽ‡', 'å æ¯”', 'ä»½é¢', 'æ¸—é€', 'å ç”¨', 'ç™¾åˆ†æ¯”')

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
                    if _is_numeric(name) and _unique(name) <= max(12, row_count // 4 + 1) and any(token in lower for token in ('year', 'quarter', 'week', 'æœˆ', 'å‘¨', 'å­£')):
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

                family_raw = (intent.get('chart_family') or '').lower()
                family = family_raw.strip()
                mark = 'line'
                variant = 'main'
                style = {'alpha': 0.9}
                if family in {'area', 'stacked_area'}:
                    mark = 'area'
                    variant = 'stacked' if family == 'stacked_area' else 'fill'
                    style.update({'alpha': 0.55, 'zorder': 1})
                elif family in {'scatter', 'bubble'}:
                    mark = 'scatter'
                    variant = 'main'
                    style.update({'alpha': 0.75, 'marker': 'o'})
                elif family in {'stacked_bar', 'stacked'}:
                    mark = 'bar'
                    variant = 'stacked'
                    style.update({'alpha': 0.85})
                elif family in {'bar', 'column'}:
                    mark = 'bar'
                    variant = 'grouped' if group else 'main'
                    style.update({'alpha': 0.85})
                elif family == 'line':
                    mark = 'line'
                else:
                    if group and y and _is_ratio_like(y):
                        mark = 'area'
                        variant = 'fill'
                        style.update({'alpha': 0.55, 'zorder': 1})
                    elif x and _is_time_like(x):
                        mark = 'line'
                    elif group:
                        mark = 'bar'
                        variant = 'grouped'
                        style.update({'alpha': 0.85})
                    else:
                        mark = 'scatter' if y and _is_numeric(y) else 'line'
                        if mark == 'scatter':
                            variant = 'main'
                            style.update({'alpha': 0.75, 'marker': 'o'})
                if mark == 'bar':
                    style.setdefault('width', 0.8)
                    style.setdefault('alpha', 0.85)
                elif mark == 'area':
                    style.setdefault('alpha', 0.55)
                elif mark == 'scatter':
                    style.setdefault('alpha', 0.75)
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
                            'style': style,
                        }
                    ],
                    'scales': {
                        'x': {
                            'kind': 'time' if (x and (_dtype(x).startswith('date') or 'datetime' in _dtype(x))) else ('linear' if (x in num_cols and x not in cat_cols) else 'categorical'),
                            'range': None,
                            'breaks': None,
                        },
                        'y_left': {'kind': 'linear', 'range': [None, None], 'breaks': None},
                        'y_right': {'kind': 'linear', 'range': None},
                    },
                    'layout': {
                        'titles': layout_titles,
                        'title_align': 'left',
                        'legend': {'loc': 'outside right', 'ncol': 1, 'frame': False},
                        'grid': {'x': False, 'y': True, 'minor': True},
                        'panel_labels': [],
                    },
                    'theme': {
                        'font': 'Arial',
                        'fontsize': 9,
                        'axis_linewidth': 1.0,
                        'tick_len': 3.0,
                        'tick_width': 0.8,
                        'palette_global': 'tab10',
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
                meta.update({
                    'x_name': x,
                    'y_name': y,
                    'group_name': group,
                    'x_is_time_hint': bool(x and _is_time_like(x)),
                    'x_is_numeric_hint': bool(x and _is_numeric(x)),
                    'y_is_ratio_hint': bool(y and _is_ratio_like(y)),
                    'preferred_mark': mark,
                })
                legend_cfg = (spec_out.get('layout') or {}).get('legend', {}) or {}
                overlays_now = spec_out.get('overlays') or []
                meta['overlay_order'] = [ov.get('mark') or 'line' for ov in overlays_now]
                overlay_z = {}
                for idx, ov in enumerate(overlays_now):
                    key = ov.get('mark') or ('mark_%d' % idx)
                    overlay_z[str(key)] = float(idx)
                meta['overlay_z'] = overlay_z
                meta['legend_loc'] = legend_cfg.get('loc')
                meta['legend_policy'] = {
                    'mode': legend_cfg.get('loc'),
                    'ncol': legend_cfg.get('ncol', 1),
                    'outside': bool(legend_cfg.get('loc') and 'outside' in str(legend_cfg.get('loc'))),
                }
                meta['tick_density'] = (spec_out.get('flags') or {}).get('tick_density')
                meta['x_scale_kind'] = ((spec_out.get('scales') or {}).get('x') or {}).get('kind')
                area_style = None
                for ov_idx, ov in enumerate(overlays_now):
                    if ov.get('mark') == 'area':
                        style_now = ov.get('style') or {}
                        base_alpha = float(style_now.get('alpha', 0.55))
                        line_alpha = min(0.95, base_alpha + 0.25)
                        theme_now = spec_out.get('theme') or {}
                        area_style = {
                            'alpha_fill': base_alpha,
                            'alpha_line': line_alpha,
                            'line_width': float(theme_now.get('line_width', 1.5)),
                            'z_level': overlay_z.get(str('area'), float(ov_idx)) * 10.0,
                        }
                        break
                if area_style:
                    meta['area_style'] = area_style
                base_font = float((spec_out.get('theme') or {}).get('fontsize', 9) or 9)
                overlays_count = max(1, len(overlays_now))
                meta['font_scaling'] = {
                    'base': base_font,
                    'title': min(1.6, max(1.1, 1.05 + overlays_count * 0.05)),
                    'label': 1.05,
                    'legend': 0.95,
                    'tick': 0.95,
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
                keep = [col for col in [x, y, grp] if col and col in df.columns]
                data = df[keep].copy()
                meta = ctx.setdefault('_v2_meta', {})
                meta['x_name'] = x
                meta['y_name'] = y
                meta['group_name'] = grp
                meta['row_count_prepare'] = int(len(data))
                scale_cfg = ((spec.get('scales') or {}).get('x') or {})
                x_kind = (scale_cfg.get('kind') or '').lower()
                meta['x_scale_kind'] = x_kind
                series_x = data[x] if x and x in data.columns else None
                is_datetime = bool(series_x is not None and pd.api.types.is_datetime64_any_dtype(series_x))
                is_numeric_x = bool(series_x is not None and np.issubdtype(series_x.dtype, np.number))
                should_parse_time = (x_kind in {'time', 'datetime'} or bool(meta.get('x_is_time_hint'))) and not is_datetime and not is_numeric_x
                is_time = False
                if should_parse_time and series_x is not None:
                    converted = pd.to_datetime(series_x, errors='coerce', infer_datetime_format=True)
                    if converted.notna().mean() >= 0.6:
                        data[x] = converted
                        is_time = True
                elif is_datetime:
                    is_time = True
                meta['x_is_time'] = bool(is_time)
                meta['x_is_numeric'] = bool(series_x is not None and np.issubdtype(data[x].dtype, np.number))
                if x and x in data.columns and y and y in data.columns:
                    data = data.dropna(subset=[x, y])
                elif x and x in data.columns:
                    data = data.dropna(subset=[x])
                elif y and y in data.columns:
                    data = data.dropna(subset=[y])
                ratio_hint = bool(meta.get('y_is_ratio_hint'))
                ratio_value = False
                if y and y in data.columns:
                    numeric_y = pd.to_numeric(data[y], errors='coerce')
                    valid = numeric_y.dropna()
                    if not valid.empty:
                        share_like = valid.between(0, 1).mean() >= 0.7
                        percent_like = valid.between(0, 100).mean() >= 0.7 and valid.max() <= 120
                        ratio_value = bool(share_like or percent_like)
                meta['y_is_ratio'] = bool(ratio_hint or ratio_value)
                if x and x in data.columns and not meta['x_is_time'] and not np.issubdtype(data[x].dtype, np.number):
                    labels = data[x].astype(str)
                    nunq = labels.nunique(dropna=False)
                    if nunq <= 30 and y and y in data.columns:
                        numeric_y = pd.to_numeric(data[y], errors='coerce')
                        agg = 'mean' if meta['y_is_ratio'] else 'sum'
                        ordered = (
                            data.assign(__value=numeric_y)
                            .groupby(labels, dropna=False)['__value']
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
                if 'y_is_ratio' not in meta:
                    ratio_hint = bool(meta.get('y_is_ratio_hint'))
                    ratio_value = False
                    if y and y in df.columns:
                        numeric = pd.to_numeric(df[y], errors='coerce')
                        valid = numeric.dropna()
                        if not valid.empty:
                            share_like = valid.between(0, 1).mean() >= 0.7
                            percent_like = valid.between(0, 100).mean() >= 0.7 and valid.max() <= 120
                            ratio_value = bool(share_like or percent_like)
                    meta['y_is_ratio'] = bool(ratio_hint or ratio_value)
                keys = [col for col in [x, grp] if col and col in df.columns]
                data = df.copy()
                if y and y in data.columns and keys and data.duplicated(keys).any():
                    agg_func = 'mean' if meta.get('y_is_ratio') else 'sum'
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
                theme = spec.get('theme') or {}
                line_width = float(theme.get('line_width', 1.5))
                style_cfg = ov.get('style') or {}
                line_alpha = float(style_cfg.get('alpha', 0.95))
                overlay_z = meta.get('overlay_z') or {}
                base_z = float(overlay_z.get('line', overlay_z.get('line.main', 1.0))) * 10.0
                palette_name = theme.get('palette_global', 'tab10')
                palette_cache = meta.setdefault('palette_cache', {})
                need = 1
                if grp and grp in df.columns:
                    need = max(1, int(df[grp].astype(str).nunique()))
                colors = palette_cache.get(palette_name)
                if not colors or len(colors) < need:
                    cmap = plt.get_cmap(palette_name)
                    colors = [cmap(i) for i in np.linspace(0, 1, max(need, 1))]
                    palette_cache[palette_name] = colors
                linestyles = ['-', '--', '-.', ':']
                if grp and grp in df.columns:
                    groups = list(pd.Index(df[grp].astype(str).unique()))
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
                        linestyle = linestyles[idx % len(linestyles)] if len(groups) > 6 else '-'
                        color = colors[idx % len(colors)]
                        (ln,) = ax.plot(
                            xa,
                            ya,
                            alpha=line_alpha,
                            linewidth=line_width,
                            linestyle=linestyle,
                            label=str(g_value),
                            color=color,
                            zorder=base_z + idx * 0.6,
                        )
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
                    (ln,) = ax.plot(
                        xa,
                        ya,
                        alpha=line_alpha,
                        linewidth=line_width,
                        color=colors[0],
                        zorder=base_z,
                    )
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
                theme = spec.get('theme') or {}
                marker_size = theme.get('marker_size', 36)
                count = len(df)
                alpha = 0.85 if count < 1000 else (0.6 if count < 5000 else 0.4)
                overlay_z = meta.get('overlay_z') or {}
                base_z = float(overlay_z.get('scatter', overlay_z.get('scatter.main', 2.0))) * 10.0
                style_cfg = ov.get('style') or {}
                palette_name = theme.get('palette_global', 'tab10')
                palette_cache = meta.setdefault('palette_cache', {})
                colors = palette_cache.get(palette_name)
                if not colors:
                    cmap = plt.get_cmap(palette_name)
                    colors = [cmap(0.35)]
                    palette_cache[palette_name] = colors
                color = style_cfg.get('color') or colors[0]
                sc = ax.scatter(
                    xa,
                    values,
                    s=marker_size,
                    alpha=alpha,
                    marker=style_cfg.get('marker', 'o'),
                    color=color,
                    edgecolor=style_cfg.get('edgecolor', 'white'),
                    linewidths=style_cfg.get('edgewidth', 0.4),
                    zorder=base_z,
                )
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
if isinstance(rng, list) and len(rng) == 2 and any(val is not None for val in rng):
    bottom, top = rng
    ax_left.set_ylim(bottom=bottom, top=top)
else:
    ov = (spec.get('overlays') or [{}])[0]
    y = ov.get('y')
    if y and y in df.columns:
        series = pd.to_numeric(df[y], errors='coerce').dropna()
        if not series.empty:
            meta = ctx.setdefault('_v2_meta', {})
            y_is_ratio = bool(meta.get('y_is_ratio'))
            smin = float(series.min())
            smax = float(series.max())
            span = smax - smin
            if y_is_ratio and smax <= 1.5:
                pad = max(span * 0.2, 0.02 if smax <= 0.5 else 0.05)
                lower = max(0.0, smin - pad)
                upper = min(1.2, smax + pad)
            else:
                lower = min(0.0, smin)
                upper = smax
                if lower == upper:
                    upper = lower + (abs(lower) * 0.1 if lower else 1.0)
            ax_left.set_ylim(bottom=lower, top=upper)
            meta['y_range'] = (lower, upper)
"""
            ),
        },
        "notes": "L3 defaults: marks & scales",
    },
    "L4": {
        "slots": {
            "axes.labels": _slot(
                """
                layout = spec.get('layout') or {}
                titles = layout.get('titles') or {}
                theme = spec.get('theme') or {}
                meta = ctx.get('_v2_meta', {}) or {}
                font_scale = meta.get('font_scaling') or {}
                base_font_theme = float(theme.get('fontsize', 9))
                base_font = float(font_scale.get('base', base_font_theme))
                title_default = base_font * float(font_scale.get('title', 1.25))
                label_default = base_font * float(font_scale.get('label', 1.05))
                title_size = float(theme.get('title_fontsize', title_default))
                label_size = float(theme.get('label_fontsize', label_default))
                ov = (spec.get('overlays') or [{}])[0]
                x = ov.get('x')
                y = ov.get('y')
                title = titles.get('top') or ctx.get('user_goal') or ''
                ax_left.set_title(title, loc=layout.get('title_align', 'left'), fontsize=title_size)
                ax_left.set_xlabel(str(x) if x else '', fontsize=label_size)
                ax_left.set_ylabel(str(y) if y else '', fontsize=label_size)
                """
            ),
            "axes.ticks": _slot(
                """
theme = spec.get('theme') or {}
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
font_scale = meta.get('font_scaling') or {}
base_font = float(font_scale.get('base', theme.get('fontsize', 9)))
tick_scale = float(font_scale.get('tick', 0.9))
tick_size = max(6, base_font * tick_scale)
ax_left.tick_params(axis='both', labelsize=tick_size)
if ax_right:
    ax_right.tick_params(axis='both', labelsize=tick_size)
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
                meta = ctx.get('_v2_meta', {}) or {}
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
                legend_cfg = (spec.get('layout') or {}).get('legend') or {}
                policy = meta.get('legend_policy') or {}
                desired = policy.get('mode') or legend_cfg.get('loc', 'best')
                ncol_target = int(policy.get('ncol', legend_cfg.get('ncol', 1)))
                theme = spec.get('theme') or {}
                font_scale = meta.get('font_scaling') or {}
                base_font = float(font_scale.get('base', theme.get('fontsize', 9)))
                legend_font = max(6, base_font * float(font_scale.get('legend', 0.95)))
                handles = [h for h, _ in combined]
                labels = [lbl for _, lbl in combined]
                place_outside = policy.get('outside') or (desired in {'outside right', 'right outside', 'outside'})
                if place_outside:
                    ax_left.legend(
                        handles,
                        labels,
                        loc='center left',
                        bbox_to_anchor=(1.02, 0.5),
                        frameon=False,
                        borderaxespad=0.0,
                        fontsize=legend_font,
                        ncol=max(1, ncol_target),
                    )
                elif len(combined) >= 8:
                    ax_left.legend(
                        handles,
                        labels,
                        loc='lower center',
                        bbox_to_anchor=(0.5, -0.2),
                        frameon=False,
                        ncol=max(2, ncol_target),
                        fontsize=legend_font,
                    )
                else:
                    ax_left.legend(
                        handles,
                        labels,
                        loc=desired,
                        frameon=False,
                        fontsize=legend_font,
                        ncol=max(1, ncol_target),
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
        "notes": "L4 defaults: axes, legend, theme",
    },
}







