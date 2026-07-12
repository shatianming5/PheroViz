from __future__ import annotations

import json
import copy
import os
import re
import time
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Tuple

import pandas as pd

from app.services.code_assembler import assemble_with_slots
from app.services.default_slots_v2 import DEFAULT_STAGE_SLOTS_V2
from app.services.feedback_builder import compose_feedback
from app.services.judge import judge
from app.services.model_client import ModelClient
from app.services.pheromones import (
    ConstraintRecord,
    InvariantDecision,
    PatchTemplate,
    PersistentMemory,
    SafetyPredicate,
    Scope,
    chart_meta_class,
)
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


_MODEL_CLIENT: Optional[ModelClient] = None
_LLM_CLIENT: Optional[ModelClient] = None


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


def _get_model_client() -> ModelClient:
    _load_env_file()
    global _LLM_CLIENT, _MODEL_CLIENT
    if _LLM_CLIENT is None:
        _LLM_CLIENT = ModelClient.from_env()
    _MODEL_CLIENT = _LLM_CLIENT
    return _LLM_CLIENT


def _format_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def _format_table(data: Dict[str, Any]) -> str:
    if not data:
        return "(empty)"
    try:
        frame = pd.DataFrame(data)
        return frame.head(8).to_string(index=False)
    except (TypeError, ValueError):
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

    memory_context = payload.get("memory_context")
    if isinstance(memory_context, dict) and memory_context:
        body = (
            f"{body}\n\nPersistent memory context:\n"
            f"{_format_json(memory_context)}\n"
            "Apply hard constraints exactly. Treat eligible patch templates as guarded "
            "examples only; adapt them to the current anchors and never use a rejected template."
        )
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


def _llm_generate_slots(
    stage: str,
    payload: Dict[str, Any],
    *,
    model_client: ModelClient | None = None,
    seed: int | None = None,
    temperature: float | None = None,
) -> Dict[str, Any]:
    prompt = _build_stage_prompt(stage, payload)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    model_response = (model_client or _get_model_client()).generate_json(
        messages,
        seed=seed,
        temperature=temperature,
    )
    response_dict = model_response.value
    raw_response = _snapshot(response_dict)
    slots = response_dict.get("slots", {})
    if not isinstance(slots, dict):
        slots = {}
    clean_slots: Dict[str, str] = {}
    for key, value in slots.items():
        if isinstance(key, str) and isinstance(value, str) and value.strip():
            clean_slots[key.strip()] = value.strip()
        elif (
            stage == "L1"
            and key == "spec.compose"
            and isinstance(value, dict)
        ):
            clean_slots[key] = f"return {value!r}"
        elif (
            stage == "L1"
            and key == "spec.theme_defaults"
            and isinstance(value, dict)
        ):
            clean_slots[key] = (
                f"theme = spec.get('theme') or {{}}\n"
                f"theme.update({value!r})\n"
                "spec['theme'] = theme\n"
                "return spec"
            )
    if not clean_slots:
        raise ValueError(
            f"{stage} model response contained no non-empty slot bodies"
        )
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
    result = {
        "slots": filtered_slots,
        "notes": notes,
        "prompt": prompt,
        "response": raw_response,
        "model_metadata": {
            "model": model_response.model,
            "request_id": model_response.request_id,
            "usage": _snapshot(model_response.usage),
            "stop_reason": model_response.stop_reason,
            "latency_seconds": model_response.latency_seconds,
            "seed": seed,
            "temperature": temperature,
        },
    }
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


_STAGE_LEVEL = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}
MEMORY_MODES = (
    "none",
    "ephemeral",
    "untyped",
    "constraints",
    "patches",
    "full",
)
_MEMORY_FEATURES = {
    "none": (False, False, False, False),
    "ephemeral": (True, True, True, True),
    "untyped": (False, False, False, False),
    "constraints": (True, False, True, False),
    "patches": (False, True, False, True),
    "full": (True, True, True, True),
}
_SHARED_SPEC_PATHS: tuple[tuple[str, ...], ...] = (
    ("canvas", "width"),
    ("canvas", "height"),
    ("canvas", "dpi"),
    ("theme", "font"),
    ("theme", "palette_global"),
    ("layout", "legend", "loc"),
    ("layout", "legend", "ncol"),
    ("layout", "legend", "frame"),
    ("layout", "grid", "x"),
    ("layout", "grid", "y"),
    ("layout", "grid", "minor"),
)
_PANEL_SPEC_PATHS: tuple[tuple[str, ...], ...] = (
    ("scales", "x", "kind"),
    ("scales", "y_left", "kind"),
    ("scales", "y_right", "kind"),
)


def _normalize_initial_generation(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"default", "defaults"}:
        return "defaults"
    if normalized == "model":
        return "model"
    raise ValueError("initial_generation must be 'defaults' or 'model'")


def _normalize_memory_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in MEMORY_MODES:
        raise ValueError(
            f"memory_mode must be one of {', '.join(MEMORY_MODES)}, got {value!r}"
        )
    return normalized


def _path_slot(path: tuple[str, ...]) -> str:
    return ".".join(path)


def _read_path(data: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return copy.deepcopy(current)


def _write_path(data: Dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = data
    for key in path[:-1]:
        nested = current.get(key)
        if not isinstance(nested, dict):
            nested = {}
            current[key] = nested
        current = nested
    current[path[-1]] = copy.deepcopy(value)


def _intent_overrides_path(intent: Mapping[str, Any], path: tuple[str, ...]) -> bool:
    aesthetics = intent.get("aesthetics")
    if not isinstance(aesthetics, Mapping):
        aesthetics = {}
    explicit = {
        ("theme", "font"): "font_pref",
        ("theme", "palette_global"): "palette",
        ("layout", "legend", "loc"): "legend_policy",
        ("layout", "title_align"): "title_align",
    }
    intent_key = explicit.get(path)
    return bool(intent_key and intent_key in aesthetics)


def _apply_shared_memory_constraints(
    spec: Dict[str, Any],
    *,
    memory: PersistentMemory,
    panel_id: str,
    panel_group: str,
    intent: Mapping[str, Any],
) -> tuple[Dict[str, Any], list[str]]:
    resolution = memory.constraint_projection(
        panel_id=panel_id,
        panel_group=panel_group,
    )
    updated = copy.deepcopy(spec)
    reused_ids: list[str] = []
    supported = {_path_slot(path): path for path in _SHARED_SPEC_PATHS}
    for slot, record in sorted(resolution.selected.items()):
        path = supported.get(slot)
        if path is None or _intent_overrides_path(intent, path):
            continue
        _write_path(updated, path, record.value)
        reused_ids.append(record.id)
    return updated, reused_ids


def _memory_context_for_stage(
    *,
    memory: PersistentMemory,
    stage: str,
    chart_class: str,
    slot_keys: list[str],
    df_columns: list[str],
    panel_id: str,
    panel_group: str,
    include_constraints: bool,
    include_patches: bool,
    untyped_log: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if untyped_log is not None:
        return {"untyped_log": copy.deepcopy(untyped_log[-12:])}
    if not include_constraints and not include_patches:
        return {}
    context = memory.reuse_context(
        chart_meta_class=chart_class,
        target_level=_STAGE_LEVEL[stage],
        available_anchors=[*slot_keys, *df_columns],
        writable_slots=slot_keys,
        panel_id=panel_id,
        panel_group=panel_group,
        include_constraints=include_constraints,
        include_patches=include_patches,
    )
    return context.to_prompt_dict()


def _constraint_candidates_from_render(
    *,
    spec: Mapping[str, Any],
    profile: Mapping[str, Any],
    panel_id: str,
    panel_group: str,
    chart_class: str,
    round_idx: int,
    png_path: str,
) -> list[ConstraintRecord]:
    provenance_base = {
        "source": "validated_render",
        "grounding": "executable",
        "panel_id": panel_id,
        "round": round_idx,
        "render_path": png_path,
    }
    records: list[ConstraintRecord] = []
    for path in _SHARED_SPEC_PATHS:
        value = _read_path(spec, path)
        if value is None:
            continue
        records.append(
            ConstraintRecord(
                scope=Scope.PANEL_GROUP,
                scope_key=panel_group,
                level=1 if path[0] in {"canvas", "theme"} else 2,
                slot=_path_slot(path),
                value=value,
                hard=False,
                provenance={**provenance_base, "spec_path": list(path)},
                support=1,
                validated_executable=True,
                chart_meta_class=chart_class,
            )
        )
    for path in _PANEL_SPEC_PATHS:
        value = _read_path(spec, path)
        if value is None:
            continue
        records.append(
            ConstraintRecord(
                scope=Scope.PANEL,
                scope_key=panel_id,
                level=3,
                slot=_path_slot(path),
                value=value,
                hard=False,
                provenance={**provenance_base, "spec_path": list(path)},
                support=1,
                validated_executable=True,
                chart_meta_class=chart_class,
            )
        )

    overlays = spec.get("overlays")
    first_overlay = overlays[0] if isinstance(overlays, list) and overlays else {}
    if isinstance(first_overlay, Mapping):
        for role in ("x", "y", "group"):
            value = first_overlay.get(role)
            if value is None:
                continue
            records.append(
                ConstraintRecord(
                    scope=Scope.PANEL,
                    scope_key=panel_id,
                    level=2,
                    slot=f"encoding.{role}",
                    value=value,
                    hard=True,
                    provenance={
                        **provenance_base,
                        "source_checked": str(value) in (profile.get("columns") or {}),
                        "grounding": "source",
                    },
                    support=1,
                    validated_executable=True,
                    chart_meta_class=chart_class,
                )
            )
    for column, dtype in sorted(dict(profile.get("columns") or {}).items()):
        records.append(
            ConstraintRecord(
                scope=Scope.PANEL,
                scope_key=panel_id,
                level=2,
                slot=f"data.dtype.{column}",
                value=dtype,
                hard=True,
                provenance={
                    **provenance_base,
                    "source_checked": True,
                    "grounding": "source",
                },
                support=1,
                validated_executable=True,
                chart_meta_class=chart_class,
            )
        )
    return records


def _patch_templates_from_render(
    *,
    ok_by_layer: Mapping[str, Mapping[str, str]],
    df_columns: list[str],
    panel_id: str,
    panel_group: str,
    chart_class: str,
    round_idx: int,
    png_path: str,
    constraint_ids_by_level: Mapping[int, list[str]],
) -> list[PatchTemplate]:
    templates: list[PatchTemplate] = []
    for layer in ("L1", "L2", "L3", "L4"):
        level = _STAGE_LEVEL[layer]
        scope = Scope.PANEL_GROUP if level <= 2 else Scope.PANEL
        scope_key = panel_group if level <= 2 else panel_id
        for slot, body in sorted(dict(ok_by_layer.get(layer) or {}).items()):
            column_anchors = [column for column in df_columns if column in body]
            anchors = tuple([slot, *column_anchors])
            templates.append(
                PatchTemplate(
                    scope=scope,
                    scope_key=scope_key,
                    level=level,
                    slot=slot,
                    patch={"slots": {slot: body}},
                    chart_meta_class=chart_class,
                    required_anchors=anchors,
                    anchor_coverage_threshold=1.0,
                    safety=SafetyPredicate(allowed_slots=(slot,)),
                    provenance={
                        "source": "validated_render",
                        "grounding": "executable",
                        "panel_id": panel_id,
                        "round": round_idx,
                        "render_path": png_path,
                    },
                    touched_slots=(slot,),
                    constraint_ids=tuple(constraint_ids_by_level.get(level, [])),
                    support=1,
                    validated_executable=True,
                )
            )
    return templates


def _write_memory_from_render(
    *,
    memory: PersistentMemory,
    spec: Mapping[str, Any],
    profile: Mapping[str, Any],
    ok_by_layer: Mapping[str, Mapping[str, str]],
    df_columns: list[str],
    panel_id: str,
    panel_group: str,
    chart_class: str,
    round_idx: int,
    png_path: str,
    score_deltas: Mapping[str, float],
    write_constraints: bool,
    write_patches: bool,
) -> dict[str, Any]:
    candidates = (
        _constraint_candidates_from_render(
            spec=spec,
            profile=profile,
            panel_id=panel_id,
            panel_group=panel_group,
            chart_class=chart_class,
            round_idx=round_idx,
            png_path=png_path,
        )
        if write_constraints
        else []
    )
    accepted_constraints: list[ConstraintRecord] = []
    rejected: list[dict[str, Any]] = []
    raise_scope_slots: list[str] = []
    for record in candidates:
        metric = "data_fidelity" if record.level in {2, 3} else "visual_form"
        decision = memory.check_oscillation(
            slot=record.slot,
            proposed_value=record.value,
            improvement=float(score_deltas.get(metric, 0.0)),
            scope_key=record.scope_key,
        )
        if not decision.allowed:
            rejected.append(
                {
                    "slot": record.slot,
                    "record_id": record.id,
                    "decision": decision.to_dict(),
                }
            )
            if decision.raise_scope:
                raise_scope_slots.append(record.slot)
            continue
        stored = memory.add_constraint(record)
        memory.record_slot_commit(
            slot=record.slot,
            value=record.value,
            scope_key=record.scope_key,
        )
        accepted_constraints.append(stored)

    constraint_ids_by_level: dict[int, list[str]] = {}
    for record in accepted_constraints:
        constraint_ids_by_level.setdefault(record.level, []).append(record.id)
    accepted_patches: list[PatchTemplate] = []
    if write_patches and not rejected:
        for template in _patch_templates_from_render(
            ok_by_layer=ok_by_layer,
            df_columns=df_columns,
            panel_id=panel_id,
            panel_group=panel_group,
            chart_class=chart_class,
            round_idx=round_idx,
            png_path=png_path,
            constraint_ids_by_level=constraint_ids_by_level,
        ):
            accepted_patches.append(memory.add_patch(template))

    return {
        "constraint_ids": [record.id for record in accepted_constraints],
        "patch_ids": [record.id for record in accepted_patches],
        "rejected": rejected,
        "raise_scope_slots": sorted(set(raise_scope_slots)),
    }


def _write_memory_artifacts(
    *,
    memory: PersistentMemory,
    run_dir: Path,
    round_idx: int,
) -> dict[str, str]:
    round_snapshot = memory.write_snapshot(
        run_dir / f"memory_snapshot_round_{round_idx}.json"
    )
    latest_snapshot = memory.write_snapshot(run_dir / "memory_snapshot.json")
    trace_path = memory.write_trace(run_dir / "memory_trace.json")
    compatibility_path = memory.write_compatibility(run_dir / "pheromones.json")
    return {
        "round_snapshot_path": str(round_snapshot),
        "snapshot_path": str(latest_snapshot),
        "trace_path": str(trace_path),
        "compatibility_path": str(compatibility_path),
    }


def _append_untyped_memory(
    log: list[dict[str, Any]],
    *,
    panel_id: str,
    round_idx: int,
    stage_logs: Mapping[str, Mapping[str, Any]],
    png_path: str,
) -> None:
    for stage in ("L1", "L2", "L3", "L4"):
        stage_log = stage_logs.get(stage) or {}
        slots = sorted(dict(stage_log.get("accepted_slots") or {}))
        notes = str(stage_log.get("notes") or "").strip()
        text = (
            f"validated render {png_path}; stage={stage}; "
            f"accepted_slots={', '.join(slots) or 'none'}"
        )
        if notes:
            text = f"{text}; notes={notes}"
        log.append(
            {
                "panel_id": panel_id,
                "round": round_idx,
                "text": text,
            }
        )


def _write_untyped_memory(
    log: list[dict[str, Any]], path: str | Path
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(log, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def iter_chain(
    excel_path: str,
    user_goal: str,
    chart_family: str,
    rounds: int = 3,
    sheet: Optional[str] = None,
    intent: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    *,
    memory: PersistentMemory | None = None,
    model_client: ModelClient | None = None,
    panel_id: str = "panel-0",
    panel_group: str = "default",
    run_dir: str | Path | None = None,
    initial_generation: str = "defaults",
    seed: int | None = None,
    temperature: float | None = None,
    memory_mode: str = "full",
    untyped_memory: list[dict[str, Any]] | None = None,
    manage_ephemeral_reset: bool = True,
    evaluation_expectation: Mapping[str, Any] | None = None,
    metric_config: Mapping[str, Any] | None = None,
) -> Iterator[Dict[str, Any]]:
    def emit(event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if progress_callback is not None:
            progress_callback(event, payload or {})

    generation_mode = _normalize_initial_generation(initial_generation)
    normalized_memory_mode = _normalize_memory_mode(memory_mode)
    (
        read_constraints,
        read_patches,
        write_constraints,
        write_patches,
    ) = _MEMORY_FEATURES[normalized_memory_mode]
    typed_memory_active = normalized_memory_mode in {
        "ephemeral",
        "constraints",
        "patches",
        "full",
    }
    if normalized_memory_mode in {"none", "untyped"}:
        memory_store = PersistentMemory()
    elif normalized_memory_mode == "ephemeral":
        memory_store = memory or PersistentMemory()
    else:
        memory_store = memory or PersistentMemory()
    untyped_log = (
        untyped_memory if untyped_memory is not None else []
    ) if normalized_memory_mode == "untyped" else []
    chart_class = chart_meta_class(chart_family)

    emit(
        "startup",
        {
            "excel_path": excel_path,
            "rounds": rounds,
            "sheet": sheet,
            "chart_family": chart_family,
            "chart_meta_class": chart_class,
            "panel_id": panel_id,
            "panel_group": panel_group,
            "initial_generation": generation_mode,
            "seed": seed,
            "temperature": temperature,
            "memory_mode": normalized_memory_mode,
        },
    )

    if run_dir is None:
        RUNS_DIR.mkdir(exist_ok=True)
        timestamp = f"{time.strftime('%Y%m%dT%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
        active_run_dir = RUNS_DIR / timestamp
    else:
        active_run_dir = Path(run_dir)
    active_run_dir.mkdir(parents=True, exist_ok=True)

    emit("run_directory_ready", {"path": str(active_run_dir), "panel_id": panel_id})

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
    if read_constraints:
        inherited_spec, inherited_constraint_ids = _apply_shared_memory_constraints(
            draft_spec,
            memory=memory_store,
            panel_id=panel_id,
            panel_group=panel_group,
            intent=base_intent,
        )
    else:
        inherited_spec, inherited_constraint_ids = draft_spec, []
    spec = validate_spec(inherited_spec)

    emit(
        "spec_ready",
        {
            "keys": list(spec.keys()),
            "intent_keys": list(base_intent.keys()),
            "inherited_constraint_ids": inherited_constraint_ids,
            "panel_id": panel_id,
        },
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
        "run_dir": str(active_run_dir),
        "panel_id": panel_id,
        "panel_group": panel_group,
        "spec": spec,
        "df_columns": df_columns,
        "df_dtypes": df_dtypes,
        "df_unique_counts": df_unique_counts,
        "row_count": row_count,
    }
    if evaluation_expectation is not None:
        from app.evaluation import validate_expectation

        expectation_payload = dict(evaluation_expectation)
        validate_expectation(expectation_payload)
        ctx["_evaluation_request"] = {
            "expectation": expectation_payload,
            "metric_config": dict(metric_config or {}),
        }
    # Debug toggle propagated into scaffold for richer diagnostics/overlays
    debug_env = os.getenv("DEBUG_RUN", "").strip().lower()
    ctx["debug"] = debug_env not in {"", "0", "false", "no"}
    force_all_rounds_raw = os.getenv("FORCE_ALL_ROUNDS", "")
    force_all_rounds = force_all_rounds_raw.strip().lower() not in {"", "0", "false", "no"}
    ctx["force_all_rounds"] = force_all_rounds
    try:
        render_timeout = max(1, int(os.getenv("PHEROVIZ_RENDER_TIMEOUT", "30")))
    except ValueError as exc:
        raise ValueError("PHEROVIZ_RENDER_TIMEOUT must be an integer") from exc

    emit(
        "context_ready",
        {"round": 0, "feedback": feedback_text, "spec_keys": list(spec.keys())},
    )

    for round_idx in range(1, max(1, rounds) + 1):
        if (
            normalized_memory_mode == "ephemeral"
            and manage_ephemeral_reset
            and round_idx > 1
        ):
            memory_store.clear_records(reason=f"single_panel_round_{round_idx}")
        if read_constraints:
            spec, round_inherited_ids = _apply_shared_memory_constraints(
                spec,
                memory=memory_store,
                panel_id=panel_id,
                panel_group=panel_group,
                intent=base_intent,
            )
        else:
            round_inherited_ids = []
        spec = validate_spec(spec)
        ctx["spec"] = spec
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
        round_reuse_ids = set(round_inherited_ids)
        for layer, payload in stage_payloads.items():
            memory_context = _memory_context_for_stage(
                memory=memory_store,
                stage=layer,
                chart_class=chart_class,
                slot_keys=list(payload["slot_keys"]),
                df_columns=df_columns,
                panel_id=panel_id,
                panel_group=panel_group,
                include_constraints=read_constraints,
                include_patches=read_patches,
                untyped_log=(
                    untyped_log
                    if normalized_memory_mode == "untyped"
                    else None
                ),
            )
            payload["memory_context"] = memory_context

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
        allow_default_fallback = not (
            round_idx == 1 and generation_mode == "model"
        )
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
            if (
                round_idx == 1
                and generation_mode == "defaults"
                and layer in DEFAULT_STAGE_SLOTS_V2
            ):
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
                out = _llm_generate_slots(
                    layer,
                    payload,
                    model_client=model_client,
                    seed=seed,
                    temperature=temperature,
                )
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
            if (
                allow_default_fallback
                and not ok_layer
                and forbidden_map
                and layer in DEFAULT_STAGE_SLOTS_V2
            ):
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
            if (
                allow_default_fallback
                and layer == "L3"
                and not any(key.startswith("marks.") for key in ok_layer)
                and layer in DEFAULT_STAGE_SLOTS_V2
            ):
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
            if (
                round_idx == 1
                and generation_mode == "model"
                and not ok_layer
            ):
                raise ValueError(
                    f"{layer} model initial generation produced no admissible slots"
                )
            if read_constraints:
                freeze_decision = memory_store.check_invariant_freeze(
                    target_level=_STAGE_LEVEL[layer],
                    touched_slots=ok_layer,
                    panel_id=panel_id,
                    panel_group=panel_group,
                )
            else:
                freeze_decision = InvariantDecision(
                    allowed=True,
                    frozen={},
                    violations=(),
                )
            for frozen_slot in freeze_decision.violations:
                rejected_body = ok_layer.pop(frozen_slot, None)
                if rejected_body is not None:
                    rej_layer[frozen_slot] = (
                        "higher_scope_invariant_frozen"
                    )
            stage_logs[layer] = {
                "prompt": out_dict.get("prompt"),
                "response": _snapshot(out_dict.get("response")),
                "model_metadata": _snapshot(out_dict.get("model_metadata")),
                "payload": _snapshot(payload),
                "notes": notes_text,
                "raw_slots": llm_slots,
                "accepted_slots": ok_layer,
                "rejected_slots": rej_layer,
                "forbidden_slots": forbidden_map,
                "autofix_slots": autofix_map,
                "fallback": fallback_used,
                "invariant_freeze": freeze_decision.to_dict(),
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
        (active_run_dir / f"code_round_{round_idx}.py").write_text(
            py_code, encoding="utf-8"
        )
        (active_run_dir / f"slots_round_{round_idx}.json").write_text(
            json.dumps(slots, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        out_png = str(active_run_dir / f"figure_round_{round_idx}.png")

        emit("execution_start", {"round": round_idx, "output": out_png})
        prev_spec = copy.deepcopy(spec)
        previous_scores = dict(last_scores)
        ctx.pop("_programmatic_evaluation", None)
        ctx.pop("_programmatic_evaluation_round_token", None)
        evaluation_request = ctx.get("_evaluation_request")
        if isinstance(evaluation_request, dict):
            evaluation_request["round_token"] = round_idx
        exec_result = execute_script(
            py_code,
            df,
            base_intent,
            ctx,
            out_png,
            timeout_s=render_timeout,
        )
        if typed_memory_active:
            memory_store.advance_step()
            for record_id in sorted(round_reuse_ids):
                memory_store.mark_reuse(
                    record_id,
                    successful=bool(exec_result.get("ok")),
                )
        updated_ctx = exec_result.get("ctx")
        if isinstance(updated_ctx, dict):
            ctx.update(updated_ctx)
            new_spec = updated_ctx.get("spec")
            if isinstance(new_spec, dict):
                try:
                    validated_spec = validate_spec(new_spec)
                except (TypeError, ValueError) as exc:
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
        current_programmatic = ctx.get("_programmatic_evaluation")
        current_round_token = ctx.get("_programmatic_evaluation_round_token")
        if evaluation_expectation is not None and (
            not exec_result.get("ok")
            or not isinstance(current_programmatic, dict)
            or current_round_token != round_idx
        ):
            raise RuntimeError(
                "Programmatic evaluation was requested but the sandbox did not "
                f"produce a result: {stderr_preview[:500]}"
            )

        png_for_judge = exec_result.get("png_path") or out_png
        emit("judging_start", {"round": round_idx, "png_path": png_for_judge})
        judge_result = judge(png_for_judge, exec_result.get("stderr", ""), df, spec)
        programmatic_result = ctx.get("_programmatic_evaluation")
        programmatic_path: Path | None = None
        fidelity_ratio: float | None = None
        cohesion_ratio: float | None = None
        if isinstance(programmatic_result, dict):
            programmatic_path = (
                active_run_dir / f"programmatic_evaluation_round_{round_idx}.json"
            )
            programmatic_path.write_text(
                json.dumps(
                    programmatic_result,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            raw_fidelity = (programmatic_result.get("fidelity") or {}).get("ratio")
            raw_cohesion = (programmatic_result.get("cohesion") or {}).get("ratio")
            if isinstance(raw_fidelity, (int, float)):
                fidelity_ratio = float(raw_fidelity)
            if isinstance(raw_cohesion, (int, float)):
                cohesion_ratio = float(raw_cohesion)
        last_scores = {
            "visual_form": judge_result.get("visual_form", 0.0),
            "data_fidelity": (
                fidelity_ratio
                if fidelity_ratio is not None
                else judge_result.get("data_fidelity", 0.0)
            ),
        }
        if cohesion_ratio is not None:
            last_scores["series_cohesion"] = cohesion_ratio
        score_deltas = {
            key: float(last_scores[key]) - float(previous_scores.get(key, 0.0))
            for key in last_scores
        }
        if (
            exec_result.get("ok")
            and exec_result.get("png_path")
            and (write_constraints or write_patches)
        ):
            memory_write = _write_memory_from_render(
                memory=memory_store,
                spec=spec,
                profile=profile,
                ok_by_layer=ok_by_layer,
                df_columns=df_columns,
                panel_id=panel_id,
                panel_group=panel_group,
                chart_class=chart_class,
                round_idx=round_idx,
                png_path=str(exec_result["png_path"]),
                score_deltas=score_deltas,
                write_constraints=write_constraints,
                write_patches=write_patches,
            )
        elif (
            exec_result.get("ok")
            and exec_result.get("png_path")
            and normalized_memory_mode == "untyped"
        ):
            before = len(untyped_log)
            _append_untyped_memory(
                untyped_log,
                panel_id=panel_id,
                round_idx=round_idx,
                stage_logs=stage_logs,
                png_path=str(exec_result["png_path"]),
            )
            memory_write = {
                "constraint_ids": [],
                "patch_ids": [],
                "rejected": [],
                "raise_scope_slots": [],
                "untyped_entries_written": len(untyped_log) - before,
            }
        else:
            memory_write = {
                "constraint_ids": [],
                "patch_ids": [],
                "rejected": [],
                "raise_scope_slots": [],
                "reason": (
                    "render_failed"
                    if not exec_result.get("ok")
                    else "memory_writes_disabled"
                ),
            }
        memory_paths = _write_memory_artifacts(
            memory=memory_store,
            run_dir=active_run_dir,
            round_idx=round_idx,
        )
        if normalized_memory_mode == "untyped":
            memory_paths["untyped_path"] = str(
                _write_untyped_memory(
                    untyped_log,
                    active_run_dir / "untyped_memory.json",
                )
            )
        emit(
            "judging_complete",
            {
                "round": round_idx,
                "scores": last_scores,
                "judge_scores": {
                    "visual_form": judge_result.get("visual_form", 0.0),
                    "data_fidelity": judge_result.get("data_fidelity", 0.0),
                },
                "programmatic_fidelity": fidelity_ratio,
                "programmatic_cohesion": cohesion_ratio,
                "programmatic_evaluation_path": (
                    str(programmatic_path)
                    if programmatic_path is not None
                    else None
                ),
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
            "judge_scores": {
                "visual_form": judge_result.get("visual_form", 0.0),
                "data_fidelity": judge_result.get("data_fidelity", 0.0),
            },
            "programmatic_evaluation": programmatic_result,
            "programmatic_evaluation_path": (
                str(programmatic_path) if programmatic_path is not None else None
            ),
            "diagnostics": judge_result.get("diagnostics", []),
            "spec": spec,
            "slots": slots,
            "stderr": exec_result.get("stderr", ""),
            "stages": stage_logs,
            "debug": debug_ctx,
            "panel_id": panel_id,
            "panel_group": panel_group,
            "chart_meta_class": chart_class,
            "render_count": round_idx,
            "run_config": {
                "initial_generation": generation_mode,
                "seed": seed,
                "temperature": temperature,
                "memory_mode": normalized_memory_mode,
            },
            "memory": {
                **memory_paths,
                "mode": normalized_memory_mode,
                "write": memory_write,
                "reused_record_ids": sorted(round_reuse_ids),
            },
        }

        emit(
            "round_complete",
            {
                "round": round_idx,
                "diagnostics": len(selected["diagnostics"]),
                "has_png": bool(selected["png_path"]),
            },
        )

        artifact_path = active_run_dir / f"iteration_{round_idx}.json"
        selected["artifact_path"] = str(artifact_path)
        artifact_path.write_text(
            json.dumps(selected, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        emit("artifact_written", {"round": round_idx, "path": str(artifact_path)})

        yield copy.deepcopy(selected)

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
        if memory_write["raise_scope_slots"]:
            feedback_text = (
                f"{feedback_text}\nMemory oscillation guard requests a higher scope for: "
                f"{', '.join(memory_write['raise_scope_slots'])}"
            )
        ctx["feedback_text"] = feedback_text
        emit("feedback_ready", {"round": round_idx, "feedback": feedback_text})

    emit(
        "finished",
        {
            "round": selected["round"] if selected else 0,
            "scores": last_scores,
            "run_dir": str(active_run_dir),
            "panel_id": panel_id,
        },
    )
    _write_memory_artifacts(
        memory=memory_store,
        run_dir=active_run_dir,
        round_idx=selected["round"] if selected else 0,
    )


def run_chain(
    excel_path: str,
    user_goal: str,
    chart_family: str,
    rounds: int = 3,
    sheet: Optional[str] = None,
    intent: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    *,
    memory: PersistentMemory | None = None,
    model_client: ModelClient | None = None,
    panel_id: str = "panel-0",
    panel_group: str = "default",
    run_dir: str | Path | None = None,
    initial_generation: str = "defaults",
    seed: int | None = None,
    temperature: float | None = None,
    memory_mode: str = "full",
    untyped_memory: list[dict[str, Any]] | None = None,
    evaluation_expectation: Mapping[str, Any] | None = None,
    metric_config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    selected: Dict[str, Any] = {}
    for result in iter_chain(
        excel_path,
        user_goal,
        chart_family,
        rounds=rounds,
        sheet=sheet,
        intent=intent,
        progress_callback=progress_callback,
        memory=memory,
        model_client=model_client,
        panel_id=panel_id,
        panel_group=panel_group,
        run_dir=run_dir,
        initial_generation=initial_generation,
        seed=seed,
        temperature=temperature,
        memory_mode=memory_mode,
        untyped_memory=untyped_memory,
        evaluation_expectation=evaluation_expectation,
        metric_config=metric_config,
    ):
        selected = result
    return selected
