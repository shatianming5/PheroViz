from __future__ import annotations

import textwrap


def _slot(body: str) -> str:
    """Normalize multiline slot code."""

    return textwrap.dedent(body).strip()


AREA_MARK_BODY = """
arts = []
ov = overlay or (spec.get('overlays') or [{}])[0]
ov_id = ov.get('id') or 'area'
x = ov.get('x')
y = ov.get('y')
grp = ov.get('group')
variant = (ov.get('variant') or 'main').lower()
style = ov.get('style') or {}
alpha = float(style.get('alpha', 0.55))
meta = ctx.setdefault('_v2_meta', {})
if not x or x not in df.columns or not y or y not in df.columns:
    return arts
area_meta = meta.get('area_style') or {}
palette_name = (spec.get('theme') or {}).get('palette_global', 'tab10')
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
is_numeric = pd.api.types.is_numeric_dtype(df[x])
ratio_flags = meta.get('ratio_flags') or {}
is_ratio = bool(ratio_flags.get(ov_id))
stack_mode = area_meta.get('mode') or ('normalize' if is_ratio else 'additive')

def _as_numeric(values):
    if is_time:
        return pd.to_datetime(values, errors='coerce').to_numpy()
    if is_numeric:
        return pd.to_numeric(values, errors='coerce').to_numpy()
    cats = meta.get('x_categories') or list(pd.Index(df[x].astype(str).unique()))
    return pd.Categorical(pd.Index(values).astype(str), categories=cats, ordered=True).codes

if variant == 'stacked' and grp and grp in df.columns:
    subset = df[[c for c in [x, grp, y] if c in df.columns]].dropna(subset=[x, y])
    if subset.empty:
        return arts
    agg_method = 'mean' if is_ratio else 'sum'
    aggregated = subset.groupby([x, grp], dropna=False)[y].agg(agg_method).reset_index()
    if aggregated.empty:
        return arts
    pivot = aggregated.pivot(index=x, columns=grp, values=y).fillna(0.0)
    if stack_mode == 'normalize':
        totals = pivot.sum(axis=1)
        totals.replace(0, 1, inplace=True)
        pivot = pivot.div(totals, axis=0)
    ordered_index = pivot.index
    if meta.get('x_categories'):
        try:
            ordered_index = pd.Index(meta['x_categories'])
            pivot = pivot.reindex(ordered_index, fill_value=0.0)
        except Exception:
            ordered_index = pivot.index
    xa = _as_numeric(pivot.index)
    base = np.zeros(len(pivot.index))
    for idx, col in enumerate(pivot.columns):
        values = pivot[col].to_numpy()
        top = base + values
        color = colors[idx % len(colors)]
        face = (float(color[0]), float(color[1]), float(color[2]), float(alpha))
        edge = (float(color[0]), float(color[1]), float(color[2]), min(0.95, float(alpha) + 0.2))
        patch = ax.fill_between(xa, base, top, color=face, zorder=meta.get('overlay_z', {}).get(ov_id, idx) * 10.0 + idx, label=str(col))
        arts.append(patch)
        ax.plot(xa, top, color=edge, linewidth=area_meta.get('line_width', (spec.get('theme') or {}).get('line_width', 1.5)), alpha=edge[3], zorder=meta.get('overlay_z', {}).get(ov_id, idx) * 10.0 + idx + 0.5)
        base = top
    return arts

subset = df[[c for c in [x, y, grp] if c and c in df.columns]].dropna(subset=[x, y])
if subset.empty:
    return arts
if grp and grp in subset.columns:
    group_order = list(pd.Index(df[grp].astype(str).unique()))
    for idx, (g_value, g_df) in enumerate(subset.groupby(grp, dropna=False)):
        ordered = g_df[[x, y]].sort_values(x)
        xa = _as_numeric(ordered[x])
        ya = pd.to_numeric(ordered[y], errors='coerce').to_numpy()
        color = colors[idx % len(colors)]
        face = (float(color[0]), float(color[1]), float(color[2]), float(alpha))
        edge_alpha = min(0.95, float(alpha) + 0.2)
        patch = ax.fill_between(xa, ya, alpha=face[3], label=str(g_value), color=face, zorder=meta.get('overlay_z', {}).get(ov_id, idx) * 10.0 + idx)
        arts.append(patch)
        ax.plot(xa, ya, color=(float(color[0]), float(color[1]), float(color[2]), edge_alpha), linewidth=(spec.get('theme') or {}).get('line_width', 1.5), zorder=meta.get('overlay_z', {}).get(ov_id, idx) * 10.0 + idx + 0.5, label='_nolegend_')
else:
    ordered = subset[[x, y]].sort_values(x)
    xa = _as_numeric(ordered[x])
    ya = pd.to_numeric(ordered[y], errors='coerce').to_numpy()
    base_color = colors[0]
    face = (float(base_color[0]), float(base_color[1]), float(base_color[2]), float(alpha))
    edge_alpha = min(0.95, float(alpha) + 0.2)
    patch = ax.fill_between(xa, ya, alpha=face[3], color=face, zorder=meta.get('overlay_z', {}).get(ov_id, 0) * 10.0)
    arts.append(patch)
    ax.plot(xa, ya, color=(float(base_color[0]), float(base_color[1]), float(base_color[2]), edge_alpha), linewidth=(spec.get('theme') or {}).get('line_width', 1.5), zorder=meta.get('overlay_z', {}).get(ov_id, 0) * 10.0 + 0.5, label='_nolegend_')
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
time_tokens = ('date', 'time', 'year', 'month', 'week', 'day', 'quarter', 'season', '年', '月', '日', '季度', '周', '季', '时', '小时')
ratio_tokens = ('rate', 'ratio', 'share', 'percent', 'pct', '%', '率', '占比', '份额', '渗透', '占用', '百分比')

def _dtype(name: str) -> str:
    source = str(profile_columns.get(name, "")) or str(dtypes.get(name, ""))
    return source.lower()

def _is_numeric(name: str) -> bool:
    dtype = _dtype(name)
    return dtype.startswith('num') or 'float' in dtype or 'int' in dtype

def _nunique(name: str) -> int:
    try:
        value = unique_counts.get(name, row_count)
        if isinstance(value, list):
            meta.setdefault('debug_unique_lists', {})[name] = len(value)
            return len(value)
        if isinstance(value, dict):
            meta.setdefault('debug_unique_lists', {})[name] = len(value)
            return len(value)
        return int(value)
    except Exception:
        meta.setdefault('debug_unique_lists', {})[name] = 'fallback'
        return row_count

def _is_time_like(name: str) -> bool:
    dtype = _dtype(name)
    lower = str(name).lower()
    if dtype.startswith('date') or 'datetime' in dtype:
        return True
    if any(token in lower for token in time_tokens):
        return True
    if _is_numeric(name):
        unique_val = _nunique(name)
        if isinstance(unique_val, (list, tuple)):
            meta.setdefault('debug_unique_lists', {})[name] = len(unique_val)
            unique_val = len(unique_val)
        try:
            unique_val = int(unique_val)
        except Exception:
            meta.setdefault('debug_unique_lists', {})[name] = 'fallback_is_time'
            unique_val = row_count
        if unique_val <= max(12, row_count // 4 + 1) and any(token in lower for token in ('year', 'quarter', 'week', '月', '周', '季')):
            return True
    return False

def _is_ratio_like(name: str) -> bool:
    lower = str(name).lower()
    return any(token in lower for token in ratio_tokens)

num_cols = [c for c in cols if _is_numeric(c)]
raw_cat_cols = [c for c in cols if c not in num_cols]
cat_cols = list(raw_cat_cols)
for candidate in num_cols:
    if candidate not in cat_cols and (_is_time_like(candidate) or _nunique(candidate) <= max(12, row_count // 2 + 1)):
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
        numeric_candidates.sort(key=lambda c: (0 if not _is_time_like(c) else 1, -_nunique(c)))
        y = numeric_candidates[0]
    elif cols:
        y = cols[-1]
    else:
        y = None

group = intent.get('group')
if group not in cols:
    limit = max(3, min(12, int(row_count / 10) + 1)) if row_count else 12
    group_candidates = [c for c in cat_cols if c != x and _nunique(c) <= limit]
    if not group_candidates:
        group_candidates = [c for c in raw_cat_cols if c != x]
    group = group_candidates[0] if group_candidates else None

is_ratio_y = bool(y and _is_ratio_like(y))

family_raw = (intent.get('chart_family') or '').lower()
family = family_raw.strip()
mark = 'line'
variant = 'main'
style = {'alpha': 0.9}

if family in {'area', 'stacked_area'}:
    mark = 'area'
    variant = 'stacked' if (family == 'stacked_area' or (group and is_ratio_y)) else 'fill'
    style = {'alpha': 0.55, 'zorder': 1}
elif family in {'scatter', 'bubble'}:
    mark = 'scatter'
    variant = 'main'
    style = {'alpha': 0.75, 'marker': 'o'}
elif family in {'stacked_bar', 'stacked'}:
    mark = 'bar'
    variant = 'stacked'
    style = {'alpha': 0.85}
elif family in {'bar', 'column'}:
    mark = 'bar'
    variant = 'grouped' if group else 'main'
    style = {'alpha': 0.85}
elif family == 'line':
    mark = 'line'
else:
    if group and y and _is_ratio_like(y):
        mark = 'area'
        variant = 'stacked'
        style = {'alpha': 0.55, 'zorder': 1}
    elif x and _is_time_like(x):
        mark = 'line'
    elif group:
        mark = 'bar'
        variant = 'grouped'
        style = {'alpha': 0.85}
    else:
        mark = 'scatter' if y and _is_numeric(y) else 'line'
        if mark == 'scatter':
            variant = 'main'
            style = {'alpha': 0.75, 'marker': 'o'}
        else:
            style = {'alpha': 0.9}

if mark == 'bar':
    style.setdefault('width', 0.8)
elif mark == 'area':
    style.setdefault('zorder', 1)

layout_titles = {
    'top': f"{y} vs {x}" if x and y else (ctx.get('user_goal') or ''),
    'left': None,
    'right': None,
    'bottom': None,
}


overlays = []
overlay_roles = []
overlay_ids = {}

role_registry = {}
used_columns = set()
used_pairs = set()
right_counter = {'count': 0}
applied_columns = {}

def _alloc_id(base: str) -> str:
    base = (base or 'overlay').replace(' ', '_')
    count = overlay_ids.get(base, 0)
    overlay_ids[base] = count + 1
    return f"{base}_{count}" if count else base

def _make_overlay(mark_kind: str, variant_kind: str, x_key: str, y_key: str, group_key: str | None, yaxis: str, style_dict: dict, role: str) -> dict:
    ov_id = _alloc_id(mark_kind or 'overlay')
    axis = yaxis or 'left'
    if axis == 'right' and right_counter.get('count', 0) >= 2:
        axis = 'left'
    existing = None
    if y_key and (y_key, axis) in used_pairs:
        for candidate in overlays:
            if candidate.get('y') == y_key and (candidate.get('yaxis') or 'left') == axis:
                existing = candidate
                break
        if existing:
            role_registry.setdefault(role, []).append({'id': existing.get('id'), 'y': y_key, 'axis': axis, 'mark': mark_kind})
            if y_key:
                bucket = applied_columns.setdefault(role, [])
                if y_key not in bucket:
                    bucket.append(y_key)
            return existing
    overlay = {
        'id': ov_id,
        'mark': mark_kind,
        'variant': variant_kind or 'main',
        'x': x_key,
        'y': y_key,
        'group': group_key,
        'yaxis': axis,
        'style': dict(style_dict or {}),
    }
    overlays.append(overlay)
    overlay_roles.append({
        'id': ov_id,
        'mark': mark_kind,
        'variant': variant_kind or 'main',
        'y': y_key,
        'group': group_key,
        'yaxis': axis,
        'role': role,
    })
    if y_key:
        used_columns.add(y_key)
        used_pairs.add((y_key, axis))
        bucket = applied_columns.setdefault(role, [])
        if y_key not in bucket:
            bucket.append(y_key)
    if axis == 'right':
        right_counter['count'] = right_counter.get('count', 0) + 1
    role_registry.setdefault(role, []).append({'id': ov_id, 'y': y_key, 'axis': axis, 'mark': mark_kind})
    return overlay

primary_overlay = _make_overlay(mark, variant, x, y, group, 'left', style, 'primary')


extra_numeric = [c for c in num_cols if c not in {y}]
extra_ratio = [c for c in extra_numeric if _is_ratio_like(c)]
extra_abs = [c for c in extra_numeric if not _is_ratio_like(c)]

target_tokens = ('target', 'goal', 'quota', 'kpi', 'objective', '目标', '指标', '达成')
forecast_tokens = ('plan', 'budget', 'forecast', 'proj', 'projection', 'estimate', 'expected', '预估', '预计', '计划', '预算')
baseline_tokens = ('baseline', 'bench', 'benchmark', 'avg', 'average', 'mean', 'median', 'industry', 'reference', 'ref', '对标', '基准', '均值', '平均', '去年', '上年', '往年', 'prev', 'previous', 'prior', 'last', '同比')
trend_tokens = ('trend', 'rolling', 'ma', 'moving', 'smooth', 'ema', 'sma', '移动', '趋势', '滑动', '平滑')

def _has_token(name, tokens):
    lower = str(name).lower()
    return any(tok in lower for tok in tokens)

def _unique_ordered(seq):
    seen = set()
    ordered = []
    for value in seq:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered

target_cols = _unique_ordered([c for c in extra_numeric if _has_token(c, target_tokens)])
forecast_cols = _unique_ordered([c for c in extra_numeric if _has_token(c, forecast_tokens) and c not in target_cols])
baseline_cols = _unique_ordered([c for c in extra_numeric if _has_token(c, baseline_tokens) and c not in target_cols and c not in forecast_cols])
trend_cols = _unique_ordered([c for c in extra_numeric if _has_token(c, trend_tokens) and c not in target_cols and c not in forecast_cols])

if mark == 'area' and group:
    meta['area_stack_mode'] = 'normalize' if is_ratio_y else 'additive'

def _choose_axis(candidate):
    cand_ratio = _is_ratio_like(candidate)
    if cand_ratio == is_ratio_y:
        return 'left'
    if cand_ratio:
        return 'right'
    if is_ratio_y:
        return 'right'
    return 'right'

for col in target_cols[:2]:
    if col in used_columns:
        continue
    axis = _choose_axis(col)
    style_target = {'alpha': 0.8, 'linestyle': '--', 'width': 1.4, 'color': '#444444'}
    _make_overlay('line', 'main', x, col, group if group else None, axis, style_target, 'target_metric')

for col in forecast_cols[:2]:
    if col in used_columns:
        continue
    axis = _choose_axis(col)
    style_forecast = {'alpha': 0.75, 'linestyle': ':', 'width': 1.3, 'color': '#666666'}
    _make_overlay('line', 'main', x, col, group if group else None, axis, style_forecast, 'forecast_metric')

for col in baseline_cols[:1]:
    if col in used_columns:
        continue
    axis = _choose_axis(col)
    style_base = {'alpha': 0.7, 'linestyle': '-.', 'width': 1.2, 'color': '#2f4f4f'}
    _make_overlay('line', 'main', x, col, None, axis, style_base, 'baseline_metric')

for col in trend_cols[:1]:
    if col in used_columns:
        continue
    axis = 'left'
    style_trend = {'alpha': 0.85, 'linestyle': '-', 'width': 1.6}
    _make_overlay('line', 'main', x, col, None, axis, style_trend, 'trend_overlay')

extra_ratio = [c for c in extra_ratio if c not in used_columns]
extra_abs = [c for c in extra_abs if c not in used_columns]

if is_ratio_y:
    remaining_abs = [c for c in extra_abs if c not in used_columns]
    if remaining_abs:
        y_right = remaining_abs[0]
        mark_right = 'line' if _is_time_like(x) else ('bar' if group else 'line')
        variant_right = 'grouped' if (mark_right == 'bar' and group) else 'main'
        style_right = {'alpha': 0.75}
        if mark_right == 'bar':
            style_right['width'] = 0.45 if group else 0.6
        _make_overlay(mark_right, variant_right, x, y_right, group if mark_right == 'bar' else None, 'right', style_right, 'secondary_metric')
    peer_ratios = [c for c in extra_ratio if c not in used_columns]
    if peer_ratios:
        peer_col = peer_ratios[0]
        mark_peer = 'area' if group else ('line' if _is_time_like(x) else 'scatter')
        variant_peer = 'stacked' if (mark_peer == 'area' and group) else 'main'
        style_peer = {'alpha': 0.55} if mark_peer == 'area' else {'alpha': 0.75}
        _make_overlay(mark_peer, variant_peer, x, peer_col, group if mark_peer == 'area' else None, 'left', style_peer, 'peer_ratio_metric')
else:
    remaining_ratio = [c for c in extra_ratio if c not in used_columns]
    for ratio_col in remaining_ratio[:2]:
        mark_ratio = 'area' if group else ('line' if _is_time_like(x) else 'scatter')
        variant_ratio = 'stacked' if (mark_ratio == 'area' and group) else 'main'
        style_ratio = {'alpha': 0.55} if mark_ratio == 'area' else {'alpha': 0.75}
        _make_overlay(mark_ratio, variant_ratio, x, ratio_col, group if mark_ratio == 'area' else None, 'right', style_ratio, 'secondary_ratio')
    remaining_abs = [c for c in extra_abs if c not in used_columns]
    for abs_col in remaining_abs[:1]:
        mark_abs = 'line' if _is_time_like(x) else 'bar'
        variant_abs = 'grouped' if (mark_abs == 'bar' and group) else 'main'
        style_abs = {'alpha': 0.8}
        if mark_abs == 'bar':
            style_abs['width'] = 0.5 if group else 0.7
        _make_overlay(mark_abs, variant_abs, x, abs_col, group if mark_abs == 'bar' else None, 'left', style_abs, 'supporting_metric')

extra_ratio = [c for c in extra_ratio if c not in used_columns]
extra_abs = [c for c in extra_abs if c not in used_columns]

if len(overlays) == 1 and group and not _is_time_like(x):
    remaining_abs = [c for c in extra_abs if c not in used_columns]
    if remaining_abs:
        fallback_col = remaining_abs[0]
        style_line = {'alpha': 0.85}
        _make_overlay('line', 'main', x, fallback_col, None, 'left', style_line, 'trend_overlay')

if len(overlays) == 1 and extra_ratio:
    ratio_candidate = next((c for c in extra_ratio if c not in used_columns), None)
    if ratio_candidate:
        mark_ratio = 'area' if group else ('line' if _is_time_like(x) else 'scatter')
        variant_ratio = 'stacked' if (mark_ratio == 'area' and group) else 'main'
        style_ratio = {'alpha': 0.55} if mark_ratio == 'area' else {'alpha': 0.75}
        axis_ratio = 'right' if not is_ratio_y else 'left'
        _make_overlay(mark_ratio, variant_ratio, x, ratio_candidate, group if mark_ratio == 'area' else None, axis_ratio, style_ratio, 'secondary_ratio')

if len(overlays) == 1 and group and not _is_time_like(x) and extra_abs:
    y_line = extra_abs[0]
    if y_line != y:
        style_line = {'alpha': 0.85}
        _make_overlay('line', 'main', x, y_line, None, 'left', style_line, 'trend_overlay')

spec_out = {
    'canvas': {'width': 960, 'height': 576, 'dpi': 300},
    'overlays': overlays,
    'scales': {
        'x': {
            'kind': 'time' if (x and (_dtype(x).startswith('date') or 'datetime' in _dtype(x))) else ('linear' if (x in num_cols and x not in cat_cols) else 'categorical'),
            'range': None,
            'breaks': None,
        },
        'y_left': {'kind': 'linear', 'range': [None, None], 'breaks': None},
        'y_right': {'kind': 'linear', 'range': [None, None], 'breaks': None},
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
        'max_overlays': max(3, len(overlays)),
        'tick_density': 'normal',
    },
}

if any(ov.get('yaxis') == 'right' for ov in overlays):
    (spec_out.get('layout') or {}).get('legend', {})['loc'] = 'outside right'
    meta['dual_axis'] = True
else:
    meta['dual_axis'] = False

role_columns_map = {role: _unique_ordered([item['y'] for item in items if item.get('y')]) for role, items in role_registry.items()}
secondary_ratio_columns = _unique_ordered(role_columns_map.get('secondary_ratio', []) + role_columns_map.get('peer_ratio_metric', []))
secondary_metric_columns = _unique_ordered(role_columns_map.get('secondary_metric', []) + role_columns_map.get('supporting_metric', []))
meta.update({
    'x_name': x,
    'y_name': y,
    'group_name': group,
    'x_is_time_hint': bool(x and _is_time_like(x)),
    'x_is_numeric_hint': bool(x and _is_numeric(x)),
    'y_is_ratio_hint': is_ratio_y,
    'preferred_mark': mark,
    'overlay_roles': overlay_roles,
    'overlay_role_map': role_registry,
    'role_columns': role_columns_map,
    'applied_columns': applied_columns,
    'target_columns': role_columns_map.get('target_metric', []),
    'forecast_columns': role_columns_map.get('forecast_metric', []),
    'baseline_columns': role_columns_map.get('baseline_metric', []),
    'trend_columns': role_columns_map.get('trend_overlay', []),
    'secondary_columns': secondary_metric_columns,
    'secondary_ratio_columns': secondary_ratio_columns,
    'right_axis_count': right_counter.get('count', 0),
})

meta['overlay_order'] = [role['id'] for role in overlay_roles]
meta['overlay_z'] = {role['id']: float(idx) for idx, role in enumerate(overlay_roles)}
legend_cfg = spec_out.get('layout', {}).get('legend', {})
if len(overlays) > 4 and legend_cfg.get('ncol', 1) == 1:
    legend_cfg['ncol'] = 2
meta['legend_loc'] = legend_cfg.get('loc')
meta['legend_policy'] = {
    'mode': legend_cfg.get('loc'),
    'ncol': legend_cfg.get('ncol', 1),
    'outside': bool(legend_cfg.get('loc') and 'outside' in str(legend_cfg.get('loc'))),
}
meta['tick_density'] = (spec_out.get('flags') or {}).get('tick_density')
meta['x_scale_kind'] = ((spec_out.get('scales') or {}).get('x') or {}).get('kind')

area_style_details = {}
for role in overlay_roles:
    if role.get('mark') == 'area':
        match = next((ov for ov in overlays if ov.get('id') == role.get('id')), None)
        if not match:
            continue
        base_alpha = float((match.get('style') or {}).get('alpha', 0.55))
        area_style_details = {
            'alpha_fill': base_alpha,
            'alpha_line': min(0.95, base_alpha + 0.25),
            'line_width': float((spec_out.get('theme') or {}).get('line_width', 1.5)),
            'z_level': meta['overlay_z'][role['id']] * 10.0,
            'mode': 'normalize' if is_ratio_y or role.get('role') == 'secondary_ratio' else 'additive',
        }
        break
if area_style_details:
    meta['area_style'] = area_style_details

meta['font_scaling'] = {
    'base': float((spec_out.get('theme') or {}).get('fontsize', 9) or 9),
    'title': min(1.6, max(1.1, 1.05 + len(overlays) * 0.05)),
    'label': 1.05,
    'legend': 0.95,
    'tick': 0.95,
}
return spec_out
                """
            ),
            "spec.theme_defaults": _slot(
                """
                theme = spec.get("theme") or {}

                theme.setdefault("fontsize", 9)
                theme.setdefault("axis_linewidth", 1.0)
                theme.setdefault("tick_len", 3.0)
                theme.setdefault("tick_width", 0.8)
                theme.setdefault("palette_global", "tab10")
                theme.setdefault("line_width", 1.5)
                theme.setdefault("marker_size", 36)
                spec["theme"] = theme
                layout = spec.get("layout") or {}
                titles = layout.get("titles") or {}
                if not titles.get("top"):
                    titles["top"] = ctx.get("user_goal") or titles.get("top")
                layout["titles"] = titles
                spec["layout"] = layout
                ctx.setdefault("_v2_meta", {}).setdefault("font_scaling", {
                    "base": float(theme.get("fontsize", 9)),
                    "title": 1.2,
                    "label": 1.05,
                    "legend": 0.95,
                    "tick": 0.95,
                })
                return spec
                """
            ),
        },
        "notes": "L1 defaults: spec/theme",
    },
    "L2": {
        "slots": {
            "data.prepare": _slot(
                """
                data = df.copy()
                overlays = spec.get('overlays') or []
                meta = ctx.setdefault('_v2_meta', {})
                ratio_tokens = ('rate', 'ratio', 'share', 'percent', 'pct', '%', '率', '占比', '份额', '渗透', '占用', '百分比')
                ratio_flags = {}
                overlay_roles = {role.get('id'): role for role in (meta.get('overlay_roles') or [])}
                if overlays and not meta.get('x_categories'):
                    primary = overlays[0]
                    x_hint = primary.get('x')
                    if x_hint and x_hint in data.columns and not pd.api.types.is_numeric_dtype(data[x_hint]):
                        if not pd.api.types.is_datetime64_any_dtype(data[x_hint]):
                            cats = list(dict.fromkeys(data[x_hint].astype(str).tolist()))
                            if cats:
                                meta['x_categories'] = cats
                group_categories = meta.get('group_categories') or {}
                for idx, ov in enumerate(overlays):
                    ov_id = ov.get('id') or f"overlay_{idx}"
                    x_key = ov.get('x')
                    y_key = ov.get('y')
                    grp_key = ov.get('group')
                    if x_key and x_key in data.columns:
                        series_x = data[x_key]
                        if pd.api.types.is_datetime64_any_dtype(series_x):
                            pass
                        elif (meta.get('x_is_time_hint') or (((spec.get('scales') or {}).get('x') or {}).get('kind') == 'time')):
                            converted = pd.to_datetime(series_x, errors='coerce')
                            if converted.notna().sum() >= max(3, int(len(converted) * 0.3)):
                                data[x_key] = converted
                                series_x = data[x_key]
                        if not pd.api.types.is_numeric_dtype(series_x) and not pd.api.types.is_datetime64_any_dtype(series_x):
                            cats = list(dict.fromkeys(series_x.astype(str).tolist()))
                            if cats and not meta.get('x_categories'):
                                meta['x_categories'] = cats
                    if grp_key and grp_key in data.columns:
                        grp_series = data[grp_key]
                        if not pd.api.types.is_numeric_dtype(grp_series):
                            cats = list(dict.fromkeys(grp_series.astype(str).tolist()))
                            if cats:
                                group_categories[grp_key] = cats
                    if y_key and y_key in data.columns:
                        series_y = pd.to_numeric(data[y_key], errors='coerce')
                        data[y_key] = series_y
                        name_lower = str(y_key).lower()
                        role_tag = (overlay_roles.get(ov_id) or {}).get('role', '')
                        ratio_flag = role_tag in {'secondary_ratio', 'peer_ratio_metric'}
                        if not ratio_flag and role_tag == 'primary' and meta.get('y_is_ratio_hint'):
                            ratio_flag = True
                        if not ratio_flag:
                            ratio_flag = any(token in name_lower for token in ratio_tokens)
                        if not ratio_flag:
                            cleaned = series_y.dropna()
                            if not cleaned.empty:
                                within_unit = cleaned.between(0, 1).mean() >= 0.6 and cleaned.abs().max() <= 1.5
                                near_percent = ('%' in name_lower) and cleaned.abs().max() <= 100 and cleaned.abs().mean() <= 60
                                ratio_flag = within_unit or near_percent
                        ratio_flags[ov_id] = bool(ratio_flag)
                if group_categories:
                    meta['group_categories'] = group_categories
                if overlays:
                    primary = overlays[0]
                    x_col = primary.get('x')
                    if x_col and x_col in data.columns:
                        if pd.api.types.is_datetime64_any_dtype(data[x_col]):
                            data = data.sort_values(x_col)
                        elif meta.get('x_categories'):
                            ordered = pd.Categorical(data[x_col].astype(str), categories=meta['x_categories'], ordered=True)
                            data = data.assign(__order=ordered.codes).sort_values('__order').drop(columns='__order')
                if overlays:
                    valid_mask = pd.Series(True, index=data.index, dtype=bool)
                    for ov in overlays:
                        y_key = ov.get('y')
                        if y_key and y_key in data.columns:
                            valid_mask &= data[y_key].notna()
                    data = data.loc[valid_mask].copy()
                    data.reset_index(drop=True, inplace=True)
                meta['ratio_flags'] = ratio_flags
                meta['row_count_prepare'] = int(len(data))
                return data
                """
            ),
            "data.aggregate": _slot(
                """
                overlays = spec.get('overlays') or [{}]
                meta = ctx.setdefault('_v2_meta', {})
                ratio_flags = meta.get('ratio_flags') or {}
                data = df.copy()
                aggregate_info = {}
                for idx, ov in enumerate(overlays):
                    ov_id = ov.get('id') or f"overlay_{idx}"
                    x = ov.get('x')
                    y = ov.get('y')
                    grp = ov.get('group')
                    if not y or y not in data.columns:
                        continue
                    keys = [col for col in [x, grp] if col and col in data.columns]
                    if not keys:
                        continue
                    needs_agg = data.duplicated(keys).any()
                    agg_func = 'mean' if ratio_flags.get(ov_id) else 'sum'
                    if needs_agg:
                        aggregated = data.groupby(keys, dropna=False)[y].transform(agg_func)
                        data[y] = aggregated
                    aggregate_info[ov_id] = {'keys': keys, 'agg': agg_func}
                meta['aggregate_info'] = aggregate_info
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
            "marks.area.stacked": _slot(AREA_MARK_BODY),
            "marks.line.main": _slot(
                """
                arts = []
                ov = overlay or (spec.get('overlays') or [{}])[0]
                ov_id = ov.get('id') or 'line'
                x = ov.get('x')
                y = ov.get('y')
                grp = ov.get('group')
                meta = ctx.setdefault('_v2_meta', {})
                if not x or x not in df.columns or not y or y not in df.columns:
                    return arts
                is_time = pd.api.types.is_datetime64_any_dtype(df[x])
                is_numeric = pd.api.types.is_numeric_dtype(df[x])
                meta['x_is_time'] = meta.get('x_is_time') or bool(is_time)
                meta['x_is_numeric'] = meta.get('x_is_numeric') or bool(is_numeric)
                theme = spec.get('theme') or {}
                style_cfg = ov.get('style') or {}
                line_width = float(style_cfg.get('width', theme.get('line_width', 1.5)))
                line_alpha = float(style_cfg.get('alpha', 0.95))
                overlay_z = meta.get('overlay_z') or {}
                base_z = overlay_z.get(ov_id, overlay_z.get('line', 1.0)) * 10.0
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
                        ya = pd.to_numeric(subset[y], errors='coerce').to_numpy()
                        linestyle = linestyles[idx % len(linestyles)] if len(groups) > 6 else style_cfg.get('linestyle', '-')
                        color = style_cfg.get('color') or colors[idx % len(colors)]
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
                    ya = pd.to_numeric(df[y], errors='coerce').to_numpy()
                    color = style_cfg.get('color') or colors[0]
                    (ln,) = ax.plot(
                        xa,
                        ya,
                        alpha=line_alpha,
                        linewidth=line_width,
                        color=color,
                        zorder=base_z,
                    )
                    arts.append(ln)
                return arts
"""
            ),
            "marks.bar.grouped": _slot(
                """
                arts = []
                ov = overlay or (spec.get('overlays') or [{}])[0]
                ov_id = ov.get('id') or 'bar'
                x = ov.get('x')
                y = ov.get('y')
                grp = ov.get('group')
                if not x or x not in df.columns or not y or y not in df.columns:
                    return arts
                meta = ctx.setdefault('_v2_meta', {})
                style = ov.get('style') or {}
                alpha = float(style.get('alpha', 0.9))
                width = float(style.get('width', 0.8))
                cats = meta.get('x_categories') or list(pd.Index(df[x].astype(str).unique()))
                base = np.arange(len(cats))
                overlay_z = meta.get('overlay_z') or {}
                base_z = overlay_z.get(ov_id, overlay_z.get('bar', 1.2)) * 10.0
                palette_name = (spec.get('theme') or {}).get('palette_global', 'tab10')
                palette_cache = meta.setdefault('palette_cache', {})
                if grp and grp in df.columns:
                    groups = list(pd.Index(df[grp].astype(str).unique()))
                    colors = palette_cache.get(palette_name)
                    if not colors or len(colors) < len(groups):
                        cmap = plt.get_cmap(palette_name)
                        colors = [cmap(i) for i in np.linspace(0, 1, max(len(groups), 1))]
                        palette_cache[palette_name] = colors
                    group_width = width / max(1, len(groups))
                    for idx, g_value in enumerate(groups):
                        subset = df[df[grp].astype(str) == g_value]
                        values = []
                        for cat in cats:
                            match = subset[subset[x].astype(str) == cat]
                            num = pd.to_numeric(match[y], errors='coerce').sum()
                            values.append(float(num))
                        bars = ax.bar(base + idx * group_width, values, width=group_width, alpha=alpha, label=str(g_value), color=colors[idx % len(colors)], zorder=base_z + idx * 0.2)
                        arts.extend(list(bars))
                    ax.set_xticks(base + width / 2)
                else:
                    values = [float(v) for v in pd.to_numeric(df[y], errors='coerce').fillna(0).tolist()]
                    bars = ax.bar(base, values, width=width, alpha=alpha, zorder=base_z)
                    arts.extend(list(bars))
                    ax.set_xticks(base)
                ax.set_xticklabels(list(cats))
                meta.setdefault('x_categories', list(cats))
                return arts
"""
            ),
            "marks.bar.stacked": _slot(
                """
                arts = []
                ov = overlay or (spec.get('overlays') or [{}])[0]
                ov_id = ov.get('id') or 'bar_stacked'
                x = ov.get('x')
                y = ov.get('y')
                grp = ov.get('group')
                if not x or x not in df.columns or not y or y not in df.columns or not grp or grp not in df.columns:
                    return arts
                meta = ctx.setdefault('_v2_meta', {})
                subset = df[[c for c in [x, grp, y] if c in df.columns]].dropna(subset=[x, y])
                if subset.empty:
                    return arts
                pivot = subset.groupby([x, grp], dropna=False)[y].sum().reset_index().pivot(index=x, columns=grp, values=y).fillna(0.0)
                cats = meta.get('x_categories') or list(pd.Index(df[x].astype(str).unique()))
                pivot.index = pivot.index.astype(str)
                pivot = pivot.reindex([str(c) for c in cats], fill_value=0.0)
                base = np.arange(len(pivot.index))
                style = ov.get('style') or {}
                width = float(style.get('width', 0.8))
                alpha = float(style.get('alpha', 0.9))
                palette_name = (spec.get('theme') or {}).get('palette_global', 'tab10')
                palette_cache = meta.setdefault('palette_cache', {})
                colors = palette_cache.get(palette_name)
                if not colors or len(colors) < len(pivot.columns):
                    cmap = plt.get_cmap(palette_name)
                    colors = [cmap(i) for i in np.linspace(0, 1, max(len(pivot.columns), 1))]
                    palette_cache[palette_name] = colors
                overlay_z = meta.get('overlay_z') or {}
                base_z = overlay_z.get(ov_id, overlay_z.get('bar', 1.4)) * 10.0
                bottoms = np.zeros(len(pivot.index))
                for idx, col in enumerate(pivot.columns):
                    heights = pivot[col].to_numpy().astype(float)
                    bars = ax.bar(base, heights, width=width, alpha=alpha, label=str(col), bottom=bottoms, color=colors[idx % len(colors)], zorder=base_z + idx * 0.2)
                    arts.extend(list(bars))
                    bottoms = bottoms + heights
                ax.set_xticks(base)
                ax.set_xticklabels([str(c) for c in cats])
                meta.setdefault('x_categories', [str(c) for c in cats])
                return arts
"""
            ),
            "marks.scatter.main": _slot(
                """
                ov = overlay or (spec.get('overlays') or [{}])[0]
                ov_id = ov.get('id') or 'scatter'
                x = ov.get('x')
                y = ov.get('y')
                if not x or x not in df.columns or not y or y not in df.columns:
                    return []
                is_numeric = pd.api.types.is_numeric_dtype(df[x])
                is_time = pd.api.types.is_datetime64_any_dtype(df[x])
                meta = ctx.setdefault('_v2_meta', {})
                meta['x_is_time'] = meta.get('x_is_time') or bool(is_time)
                meta['x_is_numeric'] = meta.get('x_is_numeric') or bool(is_numeric)
                style_cfg = ov.get('style') or {}
                theme = spec.get('theme') or {}
                marker_size = style_cfg.get('size', theme.get('marker_size', 36))
                count = len(df)
                alpha = style_cfg.get('alpha')
                if alpha is None:
                    alpha = 0.85 if count < 1000 else (0.6 if count < 5000 else 0.4)
                overlay_z = meta.get('overlay_z') or {}
                base_z = overlay_z.get(ov_id, overlay_z.get('scatter', 2.0)) * 10.0
                if is_numeric or is_time:
                    xa = df[x].to_numpy()
                else:
                    cats = meta.get('x_categories') or list(pd.Index(df[x].astype(str).unique()))
                    n = len(cats)
                    jitter = min(0.15, 1.0 / max(2, n))
                    codes = pd.Categorical(df[x].astype(str), categories=cats, ordered=True).codes
                    xa = codes + ((np.random.rand(len(codes)) - 0.5) * 2 * jitter)
                values = pd.to_numeric(df[y], errors='coerce').to_numpy()
                palette_name = theme.get('palette_global', 'tab10')
                palette_cache = meta.setdefault('palette_cache', {})
                colors = palette_cache.get(palette_name)
                if not colors:
                    cmap = plt.get_cmap(palette_name)
                    colors = [cmap(0.3)]
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
                is_numeric = pd.api.types.is_numeric_dtype(df[x])
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
            "scales.y_left.breaks": _slot(
                """
meta = ctx.setdefault('_v2_meta', {})
scales_cfg = (spec.get('scales') or {}).get('y_left') or {}
if str(scales_cfg.get('kind', 'linear')).lower() != 'linear':
    return
left_overlays = [ov for ov in (spec.get('overlays') or []) if (ov.get('yaxis') or 'left') == 'left']
if not left_overlays:
    return
y_series = []
for ov in left_overlays:
    ycol = ov.get('y')
    if ycol and ycol in df.columns:
        y_series.append(pd.to_numeric(df[ycol], errors='coerce'))
if not y_series:
    return
series = pd.concat(y_series, axis=0).dropna()
if series.empty:
    return
max_val = float(series.max())
min_val = float(series.min())
if not np.isfinite(max_val) or not np.isfinite(min_val):
    return
if max_val <= 0:
    return
values = series.to_numpy()
quantile_level = 0.9 if values.size >= 6 else 0.75
q_high = float(series.quantile(quantile_level))
if not np.isfinite(q_high):
    q_high = float(series.quantile(0.75))
if not np.isfinite(q_high):
    q_high = float(series.median())
if not np.isfinite(q_high):
    return
if q_high >= max_val:
    unique_sorted = np.unique(np.sort(values))
    if unique_sorted.size >= 2:
        q_high = float(unique_sorted[-2])
    else:
        return
spread = q_high - min_val
if spread <= 0:
    spread = max(1.0, abs(min_val) * 0.1)
if (max_val - q_high) <= max(1.0, 0.25 * spread):
    return
gap_start = q_high
gap_end = max_val - 0.35 * (max_val - q_high)
if gap_end <= gap_start:
    gap_end = q_high + 0.2 * max(max_val - q_high, max_val * 0.1)
if gap_end <= gap_start:
    return
width = gap_end - gap_start
compress = 0.12
compressed_width = width * compress
shift = width - compressed_width
if compressed_width <= 0 or shift <= 0:
    return
display_bottom = gap_start
display_top = gap_start + compressed_width
def _forward(values):
    base = np.asarray(values, dtype=float)
    if not base.size:
        return base
    arr = base.copy()
    mask_gap = (base > gap_start) & (base < gap_end)
    if mask_gap.any():
        arr[mask_gap] = gap_start + (base[mask_gap] - gap_start) * (compressed_width / width)
    mask_high = base >= gap_end
    if mask_high.any():
        arr[mask_high] = base[mask_high] - shift
    return arr
def _inverse(values):
    base = np.asarray(values, dtype=float)
    if not base.size:
        return base
    arr = base.copy()
    threshold = display_top
    mask_high = base >= threshold
    if mask_high.any():
        arr[mask_high] = base[mask_high] + shift
    mask_gap = (base > gap_start) & (base < threshold)
    if mask_gap.any():
        arr[mask_gap] = gap_start + (base[mask_gap] - gap_start) * (width / compressed_width)
    return arr
ax_left.set_yscale('function', functions=(_forward, _inverse))
visual_bottom = min(min_val, gap_start - 0.05 * max(1.0, gap_start - min_val))
visual_top = max_val - shift + 0.05 * max(1.0, max_val - gap_end)
ax_left.set_ylim(visual_bottom, visual_top)
scales = spec.setdefault('scales', {})
y_left = scales.setdefault('y_left', {})
y_left['breaks'] = [[float(gap_start), float(gap_end)]]
meta.setdefault('y_breaks', []).append((float(gap_start), float(gap_end)))
meta['y_break_info'] = {
    'start': float(gap_start),
    'end': float(gap_end),
    'display_top': float(display_top),
    'shift': float(shift),
    'compressed_width': float(compressed_width),
    'visual_bottom': float(visual_bottom),
    'visual_top': float(visual_top)
}
trans = matplotlib.transforms.blended_transform_factory(ax_left.transAxes, ax_left.transData)
face_color = ax_left.get_facecolor()
ax_left.add_patch(Rectangle((0.0, display_bottom), 1.0, display_top - display_bottom, transform=trans, facecolor=face_color, edgecolor='none', zorder=50))
ax_left.plot((-0.02, 0.02), (gap_start, gap_start), transform=trans, color='#333333', linewidth=1.0, clip_on=False, zorder=51)
ax_left.plot((-0.02, 0.02), (display_top, display_top), transform=trans, color='#333333', linewidth=1.0, clip_on=False, zorder=51)
return
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
                overlays = spec.get('overlays') or [{}]
                primary = overlays[0]
                x = primary.get('x')
                y = primary.get('y')
                title = titles.get('top') or ctx.get('user_goal') or ''
                ax_left.set_title(title, loc=layout.get('title_align', 'left'), fontsize=title_size)
                ax_left.set_xlabel(str(x) if x else '', fontsize=label_size)
                ax_left.set_ylabel(str(y) if y else '', fontsize=label_size)
                if ax_right:
                    right_overlay = next((ov for ov in overlays if ov.get('yaxis') == 'right'), None)
                    if right_overlay:
                        ax_right.set_ylabel(str(right_overlay.get('y') or ''), fontsize=label_size)
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
break_info = meta.get('y_break_info')
if break_info:
    start = break_info.get('start')
    display_top = break_info.get('display_top')
    lower_lim, upper_lim = ax_left.get_ylim()
    n_lower = max(2, target // 2)
    n_upper = max(2, target - n_lower + 1)
    lower_ticks = np.linspace(lower_lim, start, n_lower)
    upper_ticks = np.linspace(display_top, upper_lim, n_upper)
    ticks = []
    for val in lower_ticks:
        if val <= start + 1e-9:
            ticks.append(float(val))
    for val in upper_ticks:
        if val >= display_top - 1e-9:
            ticks.append(float(val))
    ticks = sorted(set(ticks))
    ax_left.set_yticks(ticks)
    ax_left.yaxis.set_minor_locator(AutoMinorLocator())
    meta['y_break_tick_positions'] = ticks
else:
    ax_left.yaxis.set_major_locator(MaxNLocator(target))
    ax_left.yaxis.set_minor_locator(AutoMinorLocator())
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
                overlay_roles = meta.get('overlay_roles') or []
                ratio_flags = meta.get('ratio_flags') or {}
                left_ids = [role.get('id') for role in overlay_roles if role.get('yaxis') != 'right']
                right_ids = [role.get('id') for role in overlay_roles if role.get('yaxis') == 'right']

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

                break_info = meta.get('y_break_info')
                shift = float(break_info.get('shift', 0.0)) if break_info else 0.0
                display_top = float(break_info.get('display_top', 0.0)) if break_info else None

                def _restore(value: float) -> float:
                    if break_info and display_top is not None and value >= display_top - 1e-9:
                        return value + shift
                    return value

                left_ratio = bool(meta.get('y_is_ratio')) or any(ratio_flags.get(i) for i in left_ids)
                right_ratio = any(ratio_flags.get(i) for i in right_ids)

                if left_ratio:
                    ax_left.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{(_restore(v)):.1%}"))
                else:
                    ax_left.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt_si(_restore(v))))

                if ax_right:
                    if right_ratio:
                        ax_right.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1%}"))
                    else:
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
















