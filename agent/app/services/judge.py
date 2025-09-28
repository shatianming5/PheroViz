from __future__ import annotations

import base64
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import requests

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


def _truncate(value: str, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    half = limit // 2
    return value[:half] + '\n...\n' + value[-half:]


def _call_vlm_judge(spec: Dict[str, Any], df_cols: List[str], png_path: str, exec_log: str) -> Dict[str, Any] | None:
    base_url = os.getenv('LLM_API_BASE')
    api_key = os.getenv('VLM_API_KEY') or os.getenv('LLM_API_KEY')
    model = os.getenv('VLM_MODEL') or os.getenv('LLM_MODEL')
    if not (base_url and api_key and model):
        return None
    png_file = Path(png_path)
    if not png_file.exists():
        return None
    try:
        image_b64 = base64.b64encode(png_file.read_bytes()).decode('ascii')
    except Exception as exc:
        debug = os.getenv('VLM_DEBUG')
        if debug:
            print(f'[judge] VLM call failed: {exc}', file=sys.stderr)
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                try:
                    print(f'[judge] status={exc.response.status_code} body={_truncate(exc.response.text, 1200)}', file=sys.stderr)
                except Exception:
                    pass
        return None

    spec_json = json.dumps(spec, ensure_ascii=False)
    data_columns = ', '.join(df_cols) if df_cols else 'n/a'
    exec_excerpt = _truncate(exec_log or '', 600)

    user_prompt = textwrap.dedent(
        f'''
        请作为数据可视化审阅者，评估附图的质量。
        需要给出 JSON 格式结果，包含：
        - scores.visual_form (0-1 浮点数)
        - scores.data_fidelity (0-1 浮点数)
        - diagnostics (列表，每项含 slot、key、hint、sev)
        - 可选 notes （文字说明）

        图表规格（截断）：
        { _truncate(spec_json, 3000) }

        数据列：{data_columns}
        执行日志片段：
        {exec_excerpt}
        '''
    ).strip()

    payload = {
        'model': model,
        'temperature': 0,
        'messages': [
            {
                'role': 'system',
                'content': [
                    {
                        'type': 'text',
                        'text': 'You review charts and must respond with strict JSON.'
                    }
                ]
            },
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': user_prompt},
                    {'type': 'input_image', 'image_base64': image_b64}
                ]
            }
        ],
        'response_format': {'type': 'json_object'},
        'max_output_tokens': int(os.getenv('LLM_MAX_TOKENS', '700') or 700)
    }

    url = base_url.rstrip('/') + '/chat/completions'
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    timeout = float(os.getenv('LLM_TIMEOUT', '180') or 180)

    debug_enabled = bool(os.getenv('VLM_DEBUG'))

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=(10, timeout))
        resp.raise_for_status()
    except Exception as exc:
        if debug_enabled:
            print(f'[judge] VLM call failed: {exc}', file=sys.stderr)
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                try:
                    print(f'[judge] status={exc.response.status_code} body={_truncate(exc.response.text, 1200)}', file=sys.stderr)
                except Exception:
                    pass
        return None

    try:
        data = resp.json()
    except Exception as exc:
        if debug_enabled:
            print(f'[judge] VLM response JSON parse error: {exc}', file=sys.stderr)
            try:
                print(f'[judge] raw body={_truncate(resp.text, 1200)}', file=sys.stderr)
            except Exception:
                pass
        return None

    choices = data.get('choices') or []
    if not choices:
        if debug_enabled:
            try:
                preview = json.dumps(data, ensure_ascii=False)
            except Exception:
                preview = str(data)
            print(f'[judge] VLM response missing choices: {_truncate(preview, 800)}', file=sys.stderr)
        return None

    message = choices[0].get('message') or {}
    content = message.get('content')
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get('type') == 'text':
                    parts.append(str(part.get('text') or ''))
            else:
                parts.append(str(part))
        content = ''.join(parts)
    if not isinstance(content, str):
        content = str(content or '')
    content_str = content.strip()
    if not content_str:
        if debug_enabled:
            try:
                message_preview = json.dumps(message, ensure_ascii=False)
            except Exception:
                message_preview = str(message)
            print('[judge] VLM content empty', file=sys.stderr)
            print(f'[judge] raw message={_truncate(message_preview, 800)}', file=sys.stderr)
        return None
    if content_str.startswith('```'):
        content_str = re.sub(r'^```(?:json)?', '', content_str, count=1).strip()
        if content_str.endswith('```'):
            content_str = content_str[:-3].strip()
    try:
        parsed = json.loads(content_str)
    except Exception as exc:
        if debug_enabled:
            print(f'[judge] VLM content parse error: {exc}', file=sys.stderr)
            print(f'[judge] content={_truncate(content_str, 1200)}', file=sys.stderr)
        return None

    scores = parsed.get('scores') or {}
    diagnostics_raw = parsed.get('diagnostics') or []
    vf = float(scores.get('visual_form', 0.0))
    df_score = float(scores.get('data_fidelity', 0.0))

    diagnostics: List[Dict[str, Any]] = []
    for item in diagnostics_raw:
        if not isinstance(item, dict):
            continue
        diagnostics.append(
            {
                'slot': str(item.get('slot') or ''),
                'key': str(item.get('key') or item.get('issue') or ''),
                'hint': str(item.get('hint') or item.get('issue') or ''),
                'sev': int(item.get('sev', 1))
            }
        )

    return {
        'visual_form': max(0.0, min(1.0, vf)),
        'data_fidelity': max(0.0, min(1.0, df_score)),
        'diagnostics': diagnostics,
        'notes': parsed.get('notes', '')
    }


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
    df_cols = list(getattr(df, 'columns', []))

    vlm_result = _call_vlm_judge(spec, df_cols, png_path, exec_log)
    if vlm_result:
        diagnostics = vlm_result.get('diagnostics') or []
        if not diagnostics:
            vlm_result['diagnostics'] = _diagnose(spec, df_cols, overlays_n, png_path)
        return vlm_result

    vf = _image_nonempty_score(png_path)
    layout = spec.get('layout') or {}
    grid_cfg = layout.get('grid') or {}
    if grid_cfg.get('y'):
        vf += 0.05
    if (layout.get('legend') or {}).get('loc', 'best') != 'none' and overlays_n > 1:
        vf += 0.05
    vf = max(0.0, min(1.0, vf))

    good_cols = 0
    for overlay in overlays:
        if overlay.get('x') in df_cols and overlay.get('y') in df_cols:
            good_cols += 1
    fid = 0.5 + 0.25 * (good_cols / max(1, overlays_n))
    fid = max(0.0, min(1.0, fid))

    diagnostics = _diagnose(spec, df_cols, overlays_n, png_path)
    return {'visual_form': vf, 'data_fidelity': fid, 'diagnostics': diagnostics}


