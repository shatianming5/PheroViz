from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _parse_rules(text: str) -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = {}
    current: Dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith('#'):
            continue
        if not line.startswith('  '):
            key = line.rstrip(':')
            data[key] = {}
            current = data[key]
        elif current is not None:
            sub_key, _, raw_value = line.strip().partition(':')
            value_str = raw_value.strip()
            if value_str.lower() in {'true', 'false'}:
                value = value_str.lower() == 'true'
            else:
                try:
                    value = float(value_str)
                except ValueError:
                    value = value_str
            current[sub_key] = value
    return data


def _parse_diagnostics(text: str) -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        key, _, payload = line.partition(':')
        key = key.strip()
        body = payload.strip().strip('{}')
        entry: Dict[str, Any] = {}
        for part in body.split(','):
            sub_key, _, value = part.partition(':')
            entry[sub_key.strip()] = value.strip().strip("'\"")
        data[key] = entry
    return data


MAP = _parse_diagnostics((PROJECT_ROOT / 'configs' / 'diagnostics_map.yml').read_text(encoding='utf-8'))
RULES = _parse_rules((PROJECT_ROOT / 'configs' / 'judge_rules.yml').read_text(encoding='utf-8'))


def _image_nonempty_score(png_path: str) -> float:
    try:
        im = Image.open(png_path).convert('RGB')
        pixels = list(im.getdata())
        step = max(1, len(pixels) // 5000)
        score = 0
        for idx in range(0, len(pixels) - 100, step):
            r1, g1, b1 = pixels[idx]
            r2, g2, b2 = pixels[idx + min(50, len(pixels) - idx - 1)]
            score += abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)
        if score > 50000:
            return 1.0
        if score > 10000:
            return 0.4
        return 0.1
    except Exception:
        return 0.0


def _diagnose(spec: Dict[str, Any], df_cols: List[str], overlays_n: int, png_path: str) -> List[Dict[str, Any]]:
    diagnostics: List[Dict[str, Any]] = []
    layout = (spec.get('layout') or {})
    titles = (layout.get('titles') or {})
    if RULES.get('visual_checks', {}).get('need_title') and not any(titles.values()):
        item = MAP['title.missing'].copy()
        item.update({'key': 'title.missing', 'sev': 2})
        diagnostics.append(item)

    if RULES.get('visual_checks', {}).get('need_legend_if_multi') and overlays_n > 1:
        if (layout.get('legend') or {}).get('loc', 'best') == 'none':
            item = MAP['legend.overlap'].copy()
            item.update({'key': 'legend.overlap', 'sev': 1})
            diagnostics.append(item)

    if RULES.get('visual_checks', {}).get('broken_axis_mark'):
        breaks = ((spec.get('scales') or {}).get('y_left') or {}).get('breaks')
        if breaks and not isinstance(breaks, list):
            item = MAP['broken.axis.symbol.missing'].copy()
            item.update({'key': 'broken.axis.symbol.missing', 'sev': 1})
            diagnostics.append(item)

    if ((spec.get('scales') or {}).get('y_right') or {}).get('kind') == 'log':
        item = MAP['bad.scale.y_right.log'].copy()
        item.update({'key': 'bad.scale.y_right.log', 'sev': 1})
        diagnostics.append(item)

    if overlays_n >= 2:
        item = MAP['low.contrast.series.2'].copy()
        item.update({'key': 'low.contrast.series.2', 'sev': 1})
        diagnostics.append(item)

    if _image_nonempty_score(png_path) < 0.2:
        item = MAP['empty.plot'].copy()
        item.update({'key': 'empty.plot', 'sev': 2})
        diagnostics.append(item)

    return diagnostics


def judge(png_path: str, exec_log: str, df, spec: Dict[str, Any]) -> Dict[str, Any]:
    overlays = spec.get('overlays') or []
    overlays_n = len(overlays)

    vf = _image_nonempty_score(png_path)
    layout = spec.get('layout') or {}
    grid_cfg = layout.get('grid') or {}
    if grid_cfg.get('y'):
        vf += 0.05
    if (layout.get('legend') or {}).get('loc', 'best') != 'none' and overlays_n > 1:
        vf += 0.05
    vf = max(0.0, min(1.0, vf))

    df_cols = list(getattr(df, 'columns', []))
    good_cols = 0
    for overlay in overlays:
        if overlay.get('x') in df_cols and overlay.get('y') in df_cols:
            good_cols += 1
    fid = 0.5 + 0.25 * (good_cols / max(1, overlays_n))
    fid = max(0.0, min(1.0, fid))

    diagnostics = _diagnose(spec, df_cols, overlays_n, png_path)
    return {'visual_form': vf, 'data_fidelity': fid, 'diagnostics': diagnostics}
