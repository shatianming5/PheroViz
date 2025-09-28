from __future__ import annotations

import json
import copy
import os
import re
import time
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd
import requests

from app.services.code_assembler import assemble_with_slots
from app.services.default_slots_v2 import DEFAULT_STAGE_SLOTS_V2
from app.services.feedback_builder import compose_feedback
from app.services.judge import judge
from app.services.sandbox_runner import execute_script
from app.services.slot_registry import ALLOWED_BY_LAYER
from app.services.spec_deriver import derive_spec
from app.services.spec_validator import validate_spec

RUNS_DIR = Path("runs")
_ALLOWED_LIBS = "pandas / numpy / matplotlib.pyplot / matplotlib.ticker / matplotlib.patches / matplotlib.transforms / mpl_toolkits.axes_grid1.inset_locator"

_ENV_LOADED = False


def _load_env_file() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_path = Path(".env")
    if not env_path.exists():
        _ENV_LOADED = True
        return
    try:
        raw = env_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = env_path.read_text(encoding="utf-8-sig")
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    _ENV_LOADED = True

_FORBIDDEN_APIS = "os/sys/subprocess/pathlib/shutil/socket/requests/open/eval/exec/__import__ and any I/O or network access"

OUTPUT_CONTRACT = """Output must be JSON (no extra commentary).
{
  \"slots\": { \"<slot.key>\": \"<Python statements only; no def/import/I-O/network>\" },
  \"notes\": \"<design intent / risks / fallback>\"
}
Rules:
- JSON only; do not add Markdown or plain text.
- Allowed libs: pandas, numpy, matplotlib.pyplot, matplotlib.ticker, matplotlib.patches, matplotlib.transforms, mpl_toolkits.axes_grid1.inset_locator.
- Forbidden APIs: %s.
- Function bodies must end with `return ...` (or equivalent) and avoid def/class/with/try blocks.
""" % _FORBIDDEN_APIS

SYSTEM_PROMPT = (
    "You are a visualization assembly expert who emits Matplotlib slot bodies for stages L1-L4."
    "Always obey the output contract and respond with valid JSON only."
)

_STAGE_NAMES = {
    "L1": "Spec & Theme Designer",
    "L2": "Data Preparation Engineer",
    "L3": "Geometry & Scale Engineer",
    "L4": "Micro-layout Designer",
}
_STAGE_SLOT_HINT = {
    "L1": "Allowed: spec.compose, spec.theme_defaults",
    "L2": "Allowed: data.prepare, data.aggregate, data.encode",
    "L3": "Allowed: marks.*, scales.*, colorbar.apply",
    "L4": "Allowed: axes.*, legend.apply, grid.apply, annot.*, theme.*",
}

_FORBIDDEN_SLOT_PATTERNS = [
    (re.compile(r'^\s*import\s+\w+', re.MULTILINE), 'import statements are forbidden'),
    (re.compile(r'^\s*from\s+\w+', re.MULTILINE), 'import statements are forbidden'),
    (re.compile(r'plt\.'), 'plt.* is unavailable inside scaffold'),
    (re.compile(r'matplotlib\.'), 'matplotlib is unavailable'),
    (re.compile(r'sns\.'), 'seaborn is unavailable'),
    (re.compile(r'__import__'), 'dynamic import is forbidden'),
    (re.compile(r'\beval\s*\('), 'eval is forbidden'),
    (re.compile(r'\bexec\s*\('), 'exec is forbidden'),
    (re.compile(r'\bopen\s*\('), 'file I/O is forbidden'),
    (re.compile(r'os\.'), 'os module is forbidden'),
    (re.compile(r'sys\.'), 'sys module is forbidden'),
    (re.compile(r'requests\.'), 'network access is forbidden'),
]


def _snapshot(data: Any) -> Any:
    try:
        return json.loads(json.dumps(data, ensure_ascii=False))
    except TypeError:
        return json.loads(json.dumps(data, ensure_ascii=False, default=lambda o: str(o)))


class SlotLLMClient:
    """Minimal JSON-only client for the Zhizengzeng Responses API."""

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        base = api_base or os.getenv("LLM_API_BASE") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        self.base = base.rstrip("/")
        self.key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        if not self.key:
            raise RuntimeError("Missing LLM_API_KEY; ??? .env ??? Zhizengzeng/OpenAI Key")
        self.model = model or os.getenv("LLM_MODEL") or "gpt-4.1-mini"
        timeout_env = os.getenv("LLM_TIMEOUT")
        if timeout_env:
            try:
                timeout = float(timeout_env)
            except ValueError:
                pass
        self.timeout = max(timeout, 30.0)
        connect_env = os.getenv("LLM_CONNECT_TIMEOUT")
        if connect_env:
            try:
                self.connect_timeout = max(float(connect_env), 1.0)
            except ValueError:
                self.connect_timeout = 30.0
        else:
            self.connect_timeout = 30.0
        retry_env = os.getenv("LLM_RETRY")
        try:
            self.retries = max(int(retry_env), 0) if retry_env is not None else 2
        except ValueError:
            self.retries = 2
        self._session = requests.Session()

    def chat_json(self, messages: list[dict[str, Any]]) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if os.getenv("LLM_FORCE_JSON", "1") != "0":
            payload["response_format"] = {"type": "json_object"}
        max_tokens_env = os.getenv("LLM_MAX_TOKENS")
        if max_tokens_env:
            try:
                payload["max_output_tokens"] = int(max_tokens_env)
            except ValueError:
                pass
        last_error = None
        for attempt in range(self.retries + 1):
            try:
                response = self._session.post(
                    f"{self.base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=(self.connect_timeout, self.timeout),
                )
                response.raise_for_status()
                data = response.json()
                break
            except requests.RequestException as exc:
                last_error = exc
                if attempt == self.retries:
                    raise
                time.sleep(min(2 ** attempt, 5.0))
        else:
            raise last_error
        if isinstance(data, dict) and data.get("code") not in (None, 0):
            raise RuntimeError(f"LLM ????: {data.get('code')} {data.get('msg')}")
        content = data["choices"][0]["message"].get("content", "{}").strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.S)
            if match:
                return json.loads(match.group(0))
            raise RuntimeError("????????? JSON")


_LLM_CLIENT: Optional[SlotLLMClient] = None


def _profile_df(df: pd.DataFrame) -> Dict[str, Any]:
    columns: Dict[str, str] = {}
    for name in df.columns:
        dtype = str(df[name].dtype).lower()
        if "datetime" in dtype or "date" in dtype:
            columns[name] = "datetime"
        elif any(token in dtype for token in ("float", "int", "number")):
            columns[name] = "numeric"
        else:
            columns[name] = "string"
    return {"columns": columns, "n": int(df.shape[0])}


def _get_llm_client() -> SlotLLMClient:
    _load_env_file()
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        _LLM_CLIENT = SlotLLMClient()
    return _LLM_CLIENT


def _format_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def _format_table(data: Dict[str, Any]) -> str:
    if not data:
        return "(empty)"
    try:
        frame = pd.DataFrame(data)
        return frame.head(8).to_string(index=False)
    except Exception:
        return _format_json(data)


def _build_stage_prompt(stage: str, payload: Dict[str, Any]) -> str:
    stage_name = _STAGE_NAMES.get(stage, stage)
    slot_keys = ", ".join(payload.get("slot_keys") or [])
    feedback = payload.get("feedback") or payload.get("feedback_text") or ""
    header_items = [
        f"Stage: {stage_name} ({stage})",
        f"Allowed slots: {slot_keys}",
        _STAGE_SLOT_HINT.get(stage, ""),
    ]
    forbidden_notes = payload.get("forbidden_notes") or ""
    if forbidden_notes:
        header_items.append(f"Recent forbidden issues: {forbidden_notes}")
    header_text = '\n'.join(item for item in header_items if item)
    global_rules = textwrap.dedent(
        """Global rules:
- Follow the JSON output contract strictly.
- Never emit `import`, `from`, `plt.*`, `matplotlib.*`, `sns.*`, `open`, file I/O, or network calls.
- Access nested dictionaries with `.get(...)` and defaults, e.g. `layout = spec.get('layout') or {}`.
- Guard optional objects (`if ax_right:`) before using them.
- Always define `theme = spec.get('theme') or {}` before using theme[...] inside axes/legend/theme slots.
- Keep the overlay-based spec schema; never remove `spec['overlays']` or add top-level Vega-Lite fields like `mark`/`encoding`/`layer`.
- Preserve ctx metadata via `meta = ctx.setdefault('_v2_meta', {})` and update keys incrementally.
"""
    )

    if stage == "L1":
        data_profile = _format_json(payload.get("data_profile", {}))
        intent = _format_json(payload.get("intent", {}))
        spec_json = _format_json(payload.get("spec", {}))
        tasks = textwrap.dedent(
            """L1 duties:
- Return a full spec dict via `spec.compose`; preserve existing fields unless you intentionally override them.
- Keep the overlays/canvas/scales/layout/theme/flags schema; never introduce Vega-Lite style `mark`/`encoding`/`layer` keys or replace the top-level structure.
- Do not assign `spec = {...}` or delete/clear `spec['overlays']`; adjust overlays in place by editing elements within the existing list.
- Derive overlays/layout/theme defaults that respect the intent (x/y/group, chart_family) and update ctx['_v2_meta'] accordingly.
- `spec.theme_defaults` should be `{}` when no additions are needed.
- Always fetch nested keys safely, e.g. `layout = spec.get('layout') or {}`.
Allowed variables: spec, intent, ctx, meta = ctx.setdefault('_v2_meta', {}), profile data.
"""
        )
        body_lines = [
            header_text,
            "",
            "Data profile:",
            data_profile,
            "",
            "Intent:",
            intent,
            "",
            "Current spec:",
            spec_json,
            "",
            "Previous feedback:",
            feedback,
            "",
            tasks,
            global_rules,
        ]
        body = '\n'.join(body_lines)
    elif stage == "L2":
        df_preview = _format_table(payload.get("df_head", {}))
        spec_json = _format_json(payload.get("spec", {}))
        tasks = textwrap.dedent(
            """L2 duties:
- `data.prepare` cleans/derives columns and must return a DataFrame.
- `data.aggregate` only executes when aggregation/top-k is required; otherwise `return df`.
- `data.encode` exposes plotting columns; return df when no extra encoding is needed.
- Absolutely no plotting, axes manipulation, or forbidden libraries.
- Use safe dictionary access (`cfg = spec.get('layout') or {}`).
Allowed variables: df, spec, ctx, pd, np.
"""
        )
        body_lines = [
            header_text,
            "",
            "Data preview (head):",
            df_preview,
            "",
            "Current spec:",
            spec_json,
            "",
            "Previous feedback:",
            feedback,
            "",
            tasks,
            global_rules,
        ]
        body = '\n'.join(body_lines)
    elif stage == "L3":
        df_preview = _format_table(payload.get("dff_head", {}))
        spec_json = _format_json(payload.get("spec", {}))
        tasks = textwrap.dedent(
            """L3 duties:
- Provide marks.* bodies for overlays: draw geometries, manage color/ordering, and respect ctx['_v2_meta'].
- Use the provided axis argument `ax` to draw (e.g., `ax.bar`, `ax.plot`, `ax.scatter`); do not return raw data structures without plotting.
- Keep the overlays list intact; operate on overlay dictionaries without rewriting spec['overlays'] or introducing new top-level mark/encoding keys.
- Configure scales.* or colorbar.apply when needed (log scale, dual axis limits, palettes).
- Do not touch axes/legend/grid/theme slots.
- Reuse palette via `meta = ctx.setdefault('_v2_meta', {})`; never call plt.get_cmap.
- Access spec/ctx with `.get(...)` and fallbacks.
Allowed variables: df, spec, ctx, meta = ctx.setdefault('_v2_meta', {}), ax_left, ax_right, fig, np, pd, theme.
"""
        )
        l3_example = textwrap.dedent(
            """Example:
meta = ctx.setdefault('_v2_meta', {})
overlays = spec.get('overlays') or []
overlay_cfg = overlays[0] if overlays else {}
style_cfg = overlay_cfg.get('style') or {}
"""
        )
        body_lines = [
            header_text,
            "",
            "Encoded data preview:",
            df_preview,
            "",
            "Current spec:",
            spec_json,
            "",
            "Previous feedback:",
            feedback,
            "",
            tasks,
            l3_example,
            global_rules,
        ]
        body = '\n'.join(body_lines)
    elif stage == "L4":
        spec_json = _format_json(payload.get("spec", {}))
        tasks = textwrap.dedent(
            """L4 duties:
- Manage titles, axis labels, tick density/rotation, legend placement, grids, and spines.
- You may add annotations (reference lines/bands/text) and theme adjustments for readability.
- Do not invoke data.* or marks.* functions.
- Treat missing layout safely: `layout = spec.get('layout') or {}` / `grid_cfg = layout.get('grid') or {}`.
- Start with `theme = spec.get('theme') or {}` when you need theme-driven styling.
- Check axis objects exist before mutating (`if ax_right:`).
Allowed variables: ax_left, ax_right, fig, spec, ctx, meta = ctx.setdefault('_v2_meta', {}), theme.
"""
        )
        example = textwrap.dedent(
            """Sample snippet:
layout = spec.get('layout') or {}
grid_cfg = layout.get('grid') or {}
theme = spec.get('theme') or {}
ax_left.grid(bool(grid_cfg.get('y', True)), which='major', axis='y', linestyle='-', alpha=0.3)
if ax_right:
    ax_right.grid(bool(grid_cfg.get('y', True)), which='major', axis='y', linestyle='-', alpha=0.3)
"""
        )
        body_lines = [
            header_text,
            "",
            "Current spec:",
            spec_json,
            "",
            "Previous feedback:",
            feedback,
            "",
            tasks,
            example,
            global_rules,
        ]
        body = '\n'.join(body_lines)
    else:
        payload_json = _format_json(payload)
        body_lines = [
            header_text,
            "",
            "Context:",
            payload_json,
            "",
            "Previous feedback:",
            feedback,
            "",
            global_rules,
        ]
        body = '\n'.join(body_lines)

    return f"{body}\n\n{OUTPUT_CONTRACT}"

def _filter_forbidden_slot_content(stage: str, slots: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    filtered: Dict[str, str] = {}
    forbidden: Dict[str, str] = {}
    autofix: Dict[str, str] = {}
    for key, body in (slots or {}).items():
        if not isinstance(body, str):
            continue
        reasons = []
        for pattern, message in _FORBIDDEN_SLOT_PATTERNS:
            if pattern.search(body):
                reasons.append(message)
        if stage == "L1" and key == "spec.compose":
            if re.search(r"\bspec\s*=\s*\{", body):
                reasons.append("spec.compose must update the existing spec dict instead of rebuilding it.")
            normalized = re.sub(r"\s+", "", body)
            top_level_tokens = (
                "spec.get('mark'",
                "spec.get(\"mark\"",
                "spec.get('encoding'",
                "spec.get(\"encoding\"",
                "spec.get('layer'",
                "spec.get(\"layer\"",
                "spec['mark']",
                "spec[\"mark\"]",
                "spec['encoding']",
                "spec[\"encoding\"]",
                "spec['layer']",
                "spec[\"layer\"]",
            )
            if any(token in body for token in top_level_tokens):
                reasons.append("spec.compose must not introduce Vega-Lite style top-level keys (mark/encoding/layer).")
            if ("spec.pop('overlays')" in normalized or 'spec.pop("overlays")' in normalized or "delspec['overlays']" in normalized or 'delspec["overlays"]' in normalized):
                reasons.append("spec.compose must keep spec['overlays'] and modify it in place.")
            if ("spec['overlays']=[]" in normalized or "spec[\"overlays\"]=[]" in normalized):
                reasons.append("spec.compose must not assign an empty overlays list.")
        if stage == "L3" and key.startswith("marks."):
            if 'ax.' not in body and 'axis.' not in body:
                reasons.append("marks.* must draw using the provided axis (ax).")
        if reasons:
            forbidden[key] = '; '.join(sorted(set(reasons)))
            continue
        new_body = body
        if stage == "L4" and _needs_theme_guard(body):
            new_body = "theme = spec.get('theme') or {}\n" + body
            autofix[key] = 'theme_guard'
        # Autofix: legend.apply bodies that use `legend` without defining it
        if stage == "L4" and key == "legend.apply":
            uses_legend_obj = bool(re.search(r"\blegend\s*\.\w+|\bif\s+legend\b", new_body))
            has_legend_assign = bool(re.search(r"\blegend\s*=", new_body))
            if uses_legend_obj and not has_legend_assign:
                prefix = (
                    "legend = ax_left.legend() if ax_left else None\n"
                )
                new_body = prefix + new_body
                autofix[key] = (autofix.get(key, '') + (';' if autofix.get(key) else '') + 'legend_guard').strip(';')
            # Normalize API: drop unsupported legend._set_ncol / legend.set_ncol in favor of passing ncol to creation
            # Remove fragile calls that cause AttributeError on some Matplotlib versions
            new_body = re.sub(r"legend\._set_ncol\s*\(.*?\)\s*", "", new_body)
            new_body = re.sub(r"legend\.set_ncol\s*\(.*?\)\s*", "", new_body)
        filtered[key] = new_body
    return filtered, forbidden, autofix


def _needs_theme_guard(body: str) -> bool:
    if not re.search(r"\btheme\b", body):
        return False
    if re.search(r"\btheme\s*=", body):
        return False
    if re.search(r'spec\.get\(\s*["\']theme["\']', body):
        return False
    if re.search(r'ctx\.get\(\s*["\']theme["\']', body):
        return False
    return bool(re.search(r"theme\s*\[|theme\.get", body))


def _llm_generate_slots(stage: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt = _build_stage_prompt(stage, payload)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        response = _get_llm_client().chat_json(messages)
    except Exception as exc:  # noqa: BLE001
        return {"slots": {}, "notes": f"llm_error: {exc}", "prompt": prompt, "response": {"error": str(exc)}}

    raw_response = _snapshot(response)
    response_dict = response if isinstance(response, dict) else {}
    slots = response_dict.get("slots", {})
    if not isinstance(slots, dict):
        slots = {}
    clean_slots: Dict[str, str] = {}
    for key, value in slots.items():
        if isinstance(key, str) and isinstance(value, str) and value.strip():
            clean_slots[key.strip()] = value.strip()
    filtered_slots, forbidden_map, autofix_map = _filter_forbidden_slot_content(stage, clean_slots)
    notes = response_dict.get("notes", "")
    if not isinstance(notes, str):
        notes = ""
    if forbidden_map:
        summary = ", ".join(f"{k}: {reason}" for k, reason in forbidden_map.items())
        extra = f"filtered_forbidden[{summary}]"
        notes = f"{notes} {extra}".strip() if notes else extra
    if autofix_map:
        summary_autofix = ", ".join(f"{k}: {label}" for k, label in autofix_map.items())
        extra_autofix = f"autofix[{summary_autofix}]"
        notes = f"{notes} {extra_autofix}".strip() if notes else extra_autofix
    result = {"slots": filtered_slots, "notes": notes, "prompt": prompt, "response": raw_response}
    if forbidden_map:
        result["forbidden"] = forbidden_map
    if autofix_map:
        result["autofix"] = autofix_map
    return result


def _layer_guard(layer: str, slots: Optional[Dict[str, str]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    allowed_patterns = ALLOWED_BY_LAYER[layer]
    ok: Dict[str, str] = {}
    rejected: Dict[str, str] = {}
    for key, value in (slots or {}).items():
        if any(key.startswith(pattern.replace("*", "")) for pattern in allowed_patterns):
            ok[key] = value
        else:
            rejected[key] = value
    return ok, rejected


def _load_tabular(excel_path: str, sheet: Optional[str]) -> pd.DataFrame:
    path = Path(excel_path)
    if path.suffix.lower() in {".xls", ".xlsx", ".xlsm"}:
        sheet_name = 0 if sheet is None else sheet
        return pd.read_excel(path, sheet_name=sheet_name)
    return pd.read_csv(path)


def run_chain(
    excel_path: str,
    user_goal: str,
    chart_family: str,
    rounds: int = 3,
    sheet: Optional[str] = None,
    intent: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    def emit(event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(event, payload or {})
        except Exception:
            return

    emit(
        "startup",
        {
            "excel_path": excel_path,
            "rounds": rounds,
            "sheet": sheet,
            "chart_family": chart_family,
        },
    )

    RUNS_DIR.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    run_dir = RUNS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    emit("run_directory_ready", {"path": str(run_dir)})

    df = _load_tabular(excel_path, sheet)
    profile = _profile_df(df)
    df_columns = [str(c) for c in df.columns]
    df_dtypes = {str(c): str(df[c].dtype) for c in df.columns}
    df_unique_counts = {str(c): int(df[c].nunique(dropna=True)) for c in df.columns}
    row_count = int(df.shape[0])

    emit(
        "data_loaded",
        {"rows": row_count, "columns": df_columns},
    )

    base_intent: Dict[str, Any] = {"chart_family": chart_family, "user_goal": user_goal}
    if intent:
        merged: Dict[str, Any] = base_intent.copy()
        merged.update(intent)
        base_intent = merged

    draft_spec = derive_spec(base_intent, profile)
    spec = validate_spec(draft_spec)

    emit(
        "spec_ready",
        {"keys": list(spec.keys()), "intent_keys": list(base_intent.keys())},
    )

    last_scores: Dict[str, float] = {"visual_form": 0.0, "data_fidelity": 0.0}
    feedback_text = ""
    selected: Optional[Dict[str, Any]] = None

    ctx: Dict[str, Any] = {
        "excel_path": excel_path,
        "sheet_name": sheet,
        "user_goal": user_goal,
        "chart_family": chart_family,
        "data_profile": profile,
        "feedback_text": feedback_text,
        "run_dir": str(run_dir),
        "spec": spec,
        "df_columns": df_columns,
        "df_dtypes": df_dtypes,
        "df_unique_counts": df_unique_counts,
        "row_count": row_count,
    }
    # Debug toggle propagated into scaffold for richer diagnostics/overlays
    debug_env = os.getenv("DEBUG_RUN", "").strip().lower()
    ctx["debug"] = debug_env not in {"", "0", "false", "no"}
    force_all_rounds_raw = os.getenv("FORCE_ALL_ROUNDS", "")
    force_all_rounds = force_all_rounds_raw.strip().lower() not in {"", "0", "false", "no"}
    ctx["force_all_rounds"] = force_all_rounds

    emit(
        "context_ready",
        {"round": 0, "feedback": feedback_text, "spec_keys": list(spec.keys())},
    )

    for round_idx in range(1, max(1, rounds) + 1):
        emit("round_start", {"round": round_idx, "feedback": feedback_text})
        stage_logs: Dict[str, Any] = {}

        stage_payloads = {
            "L1": {
                "data_profile": profile,
                "intent": base_intent,
                "spec": spec,
                "feedback": feedback_text,
                "slot_keys": ["spec.compose", "spec.theme_defaults"],
            },
            "L2": {
                "df_head": df.head(8).to_dict(orient="list"),
                "spec": spec,
                "feedback": feedback_text,
                "slot_keys": ["data.prepare", "data.aggregate", "data.encode"],
            },
            "L3": {
                "dff_head": df.head(8).to_dict(orient="list"),
                "spec": spec,
                "feedback": feedback_text,
                "slot_keys": ["marks.*", "scales.*", "colorbar.apply"],
            },
            "L4": {
                "spec": spec,
                "feedback": feedback_text,
                "slot_keys": ["axes.*", "legend.apply", "grid.apply", "annot.*", "theme.*"],
            },
        }

        forbidden_history = ctx.get('_forbidden_history', {})
        for _layer_key, _payload in stage_payloads.items():
            history_list = forbidden_history.get(_layer_key, [])
            if history_list:
                recent_entries = history_list[-2:]
                summary = " | ".join(
                    f"round {entry.get('round')}: {entry.get('summary')}"
                    for entry in recent_entries
                    if entry.get('summary')
                )
                if summary:
                    _payload["forbidden_notes"] = summary

        ok_by_layer: Dict[str, Dict[str, str]] = {}
        for layer, payload in stage_payloads.items():
            stage_name = _STAGE_NAMES.get(layer, layer)
            emit(
                "stage_start",
                {
                    "round": round_idx,
                    "stage": layer,
                    "stage_name": stage_name,
                    "hint": _STAGE_SLOT_HINT.get(layer, ""),
                },
            )
            if round_idx == 1 and layer in DEFAULT_STAGE_SLOTS_V2:
                default_bundle = DEFAULT_STAGE_SLOTS_V2[layer]
                raw_slots = dict(default_bundle.get("slots", {}))
                notes_default = default_bundle.get("notes", "")
                out = {
                    "slots": raw_slots,
                    "notes": notes_default,
                    "prompt": "DEFAULT_V2",
                    "response": {"source": "default_v2", "notes": notes_default},
                }
                forbidden_map: Dict[str, str] = {}
                autofix_map: Dict[str, str] = {}
            else:
                out = _llm_generate_slots(layer, payload)
                forbidden_map = (out.get("forbidden") if isinstance(out, dict) else {}) or {}
                autofix_map = (out.get("autofix") if isinstance(out, dict) else {}) or {}
            history_ref = ctx.setdefault('_forbidden_history', {}).setdefault(layer, [])
            summary_text = ""
            if forbidden_map:
                summary_entries = [f"{slot}: {reason}" for slot, reason in forbidden_map.items()]
                summary_text = '; '.join(summary_entries)
                history_ref.append({"round": round_idx, "summary": summary_text})
            elif autofix_map:
                summary_autofix = '; '.join(f"{slot}: {label}" for slot, label in autofix_map.items())
                if summary_autofix:
                    history_ref.append({"round": round_idx, "summary": f"autofix {summary_autofix}"})
            emit(
                "llm_io",
                {
                    "round": round_idx,
                    "stage": layer,
                    "stage_name": stage_name,
                    "prompt": out.get("prompt", "") if isinstance(out, dict) else "",
                    "response": out.get("response") if isinstance(out, dict) else None,
                },
            )
            out_dict = out if isinstance(out, dict) else {"slots": {}, "notes": "", "prompt": None, "response": None}
            llm_slots = out_dict.get("slots", {}) or {}
            ok_layer, rej_layer = _layer_guard(layer, llm_slots)
            fallback_used: Optional[str] = None
            fallback_slots: Dict[str, str] = {}
            notes_text = out_dict.get("notes", "")
            if not ok_layer and forbidden_map and layer in DEFAULT_STAGE_SLOTS_V2:
                default_bundle = DEFAULT_STAGE_SLOTS_V2[layer]
                fallback_slots = dict(default_bundle.get("slots", {}))
                ok_layer, rej_default = _layer_guard(layer, fallback_slots)
                fallback_used = "default_after_forbidden"
                if rej_layer:
                    rej_layer = {**rej_layer, **{f"default::{k}": v for k, v in rej_default.items()}}
                else:
                    rej_layer = rej_default
                extra_note = "fallback: default_v2 (forbidden content removed)"
                notes_text = f"{notes_text} {extra_note}".strip() if notes_text else extra_note
                fallback_summary = f"{summary_text} (fallback: default_v2)" if summary_text else "fallback: default_v2 applied"
                if history_ref:
                    history_ref[-1]["summary"] = fallback_summary
                else:
                    history_ref.append({"round": round_idx, "summary": fallback_summary})
            if layer == "L3" and not any(key.startswith("marks.") for key in ok_layer) and layer in DEFAULT_STAGE_SLOTS_V2:
                default_bundle = DEFAULT_STAGE_SLOTS_V2[layer]
                default_mark_candidates = dict(default_bundle.get("slots", {}))
                default_ok, _ = _layer_guard(layer, default_mark_candidates)
                default_marks = {k: v for k, v in default_ok.items() if k.startswith("marks.")}
                if default_marks:
                    ok_layer = {**default_marks, **ok_layer}
                    fallback_slots.update(default_marks)
                    extra_note = "fallback: default_v2 marks added"
                    notes_text = f"{notes_text} {extra_note}".strip() if notes_text else extra_note
                    fallback_used = f"{fallback_used},default_marks".strip(',') if fallback_used else "default_marks"
                    fallback_summary = "fallback default_marks -> please draw using ax.* functions"
                    if history_ref and history_ref[-1].get("round") == round_idx:
                        summary_prev = history_ref[-1].get("summary") or ""
                        combined = f"{summary_prev}; {fallback_summary}".strip('; ')
                        history_ref[-1]["summary"] = combined
                    else:
                        history_ref.append({"round": round_idx, "summary": fallback_summary})
            stage_logs[layer] = {
                "prompt": out_dict.get("prompt"),
                "response": _snapshot(out_dict.get("response")),
                "payload": _snapshot(payload),
                "notes": notes_text,
                "raw_slots": llm_slots,
                "accepted_slots": ok_layer,
                "rejected_slots": rej_layer,
                "forbidden_slots": forbidden_map,
                "autofix_slots": autofix_map,
                "fallback": fallback_used,
            }
            if fallback_used:
                stage_logs[layer]["fallback_slots"] = fallback_slots
            ok_by_layer[layer] = ok_layer
            emit(
                "stage_complete",
                {
                    "round": round_idx,
                    "stage": layer,
                    "stage_name": stage_name,
                    "accepted": list(ok_layer.keys()),
                    "rejected": list(rej_layer.keys()),
                    "notes": stage_logs[layer]["notes"],
                    "fallback": fallback_used,
                    "forbidden": list(forbidden_map.keys()),
                    "autofix": list(autofix_map.keys()),
                },
            )

        slots: Dict[str, str] = {}
        for layer in ("L1", "L2", "L3", "L4"):
            slots.update(ok_by_layer.get(layer, {}))

        emit("slots_assembled", {"round": round_idx, "slot_count": len(slots)})

        py_code = assemble_with_slots(slots)
        # Persist assembled scaffold for this round to aid debugging
        try:
            (run_dir / f"code_round_{round_idx}.py").write_text(py_code, encoding="utf-8")
            # Also persist accepted slots for this round
            (run_dir / f"slots_round_{round_idx}.json").write_text(
                json.dumps(slots, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass
        out_png = str(run_dir / f"figure_round_{round_idx}.png")

        emit("execution_start", {"round": round_idx, "output": out_png})
        prev_spec = copy.deepcopy(spec)
        exec_result = execute_script(py_code, df, base_intent, ctx, out_png)
        updated_ctx = exec_result.get("ctx")
        if isinstance(updated_ctx, dict):
            ctx.update(updated_ctx)
            new_spec = updated_ctx.get("spec")
            if isinstance(new_spec, dict):
                try:
                    validated_spec = validate_spec(new_spec)
                except Exception as exc:
                    spec = prev_spec
                    ctx["spec"] = spec
                    fallback_note = f"spec_validation_failed: {exc}"
                    stage_log = stage_logs.get("L1") if isinstance(stage_logs, dict) else None
                    if isinstance(stage_log, dict):
                        existing = stage_log.get("notes") or ""
                        stage_log["notes"] = f"{existing} {fallback_note}".strip() if existing else fallback_note
                        stage_log["fallback"] = stage_log.get("fallback") or "spec_validation"
                    history = ctx.setdefault('_forbidden_history', {}).setdefault('L1', [])
                    if not history or history[-1].get("round") != round_idx or history[-1].get("summary") != fallback_note:
                        history.append({"round": round_idx, "summary": fallback_note})
                else:
                    spec = validated_spec
                    ctx["spec"] = spec
            else:
                ctx["spec"] = spec
        stderr_preview = (exec_result.get("stderr") or "").strip()
        emit(
            "execution_end",
            {
                "round": round_idx,
                "stderr": stderr_preview[:200],
                "png_path": exec_result.get("png_path"),
            },
        )

        png_for_judge = exec_result.get("png_path") or out_png
        emit("judging_start", {"round": round_idx, "png_path": png_for_judge})
        judge_result = judge(png_for_judge, exec_result.get("stderr", ""), df, spec)
        last_scores = {
            "visual_form": judge_result.get("visual_form", 0.0),
            "data_fidelity": judge_result.get("data_fidelity", 0.0),
        }
        emit(
            "judging_complete",
            {
                "round": round_idx,
                "scores": last_scores,
                "diagnostics": len(judge_result.get("diagnostics", [])),
            },
        )

        # Extract compact debug info from ctx (if scaffold provided it)
        debug_ctx = {}
        if isinstance(ctx.get("_debug"), dict):
            debug_ctx = ctx.get("_debug")

        selected = {
            "round": round_idx,
            "png_path": exec_result.get("png_path"),
            "scores": last_scores,
            "diagnostics": judge_result.get("diagnostics", []),
            "spec": spec,
            "slots": slots,
            "stderr": exec_result.get("stderr", ""),
            "stages": stage_logs,
            "debug": debug_ctx,
        }

        emit(
            "round_complete",
            {
                "round": round_idx,
                "diagnostics": len(selected["diagnostics"]),
                "has_png": bool(selected["png_path"]),
            },
        )

        artifact_path = run_dir / f"iteration_{round_idx}.json"
        artifact_path.write_text(
            json.dumps(selected, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        emit("artifact_written", {"round": round_idx, "path": str(artifact_path)})

        if (not force_all_rounds) and last_scores["visual_form"] >= 0.75 and last_scores["data_fidelity"] >= 0.75:
            emit("round_success", {"round": round_idx, "scores": last_scores})
            break

        layer_guards = {
            "L2": "allow=data.*; deny=ax/plt/text/legend/grid/theme",
            "L3": "allow=marks.*,scales.*,colorbar.apply; deny=axes.*,legend.*,grid.*,annot.*,theme.*",
            "L4": "allow=axes.*,legend.*,grid.*,annot.*,theme.*; deny=data.*,marks.*",
        }
        feedback_text = compose_feedback(
            round_idx, last_scores, judge_result.get("diagnostics", []), layer_guards
        )
        ctx["feedback_text"] = feedback_text
        emit("feedback_ready", {"round": round_idx, "feedback": feedback_text})

    emit(
        "finished",
        {
            "round": selected["round"] if selected else 0,
            "scores": last_scores,
            "run_dir": str(run_dir),
        },
    )
    return selected or {}






















