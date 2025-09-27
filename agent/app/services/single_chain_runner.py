from __future__ import annotations

import json
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

_FORBIDDEN_APIS = "os、sys、subprocess、pathlib、shutil、socket、requests、open、eval、exec、__import__ 及任何 I/O/网络 操作"

OUTPUT_CONTRACT = """【输出格式（必须 JSON，严禁多余文本）】\n{\n  \"slots\": { \"<slot.key>\": \"<仅函数体代码，不含 def/导入/I/O/网络>\" },\n  \"notes\": \"<设计要点/风险/回退方案>\"\n}\n【强约束】\n- 仅输出 JSON 字符串，不得附加解释或 Markdown。\n- 允许使用的库：pandas、numpy、matplotlib.pyplot、matplotlib.ticker、matplotlib.patches、matplotlib.transforms、mpl_toolkits.axes_grid1.inset_locator。\n- 禁止导入或调用 %s。\n- 函数体必须是可直接放入模板的 Python 语句，需以 `return ...` 或等价语句结束，不得包含 def/class/with/try 等顶层结构。\n""" % _FORBIDDEN_APIS

SYSTEM_PROMPT = (
    "你是资深可视化编排工程师，负责根据不同层级（L1-L4）生成 Matplotlib 插槽代码。"
    "所有回复必须严格遵守输出契约，只能返回合法 JSON；任何违规键、文本或解释都将被视为错误。"
)

_STAGE_NAMES = {
    "L1": "规格与主题设计师",
    "L2": "数据编排工程师",
    "L3": "几何与标度工程师",
    "L4": "微排版设计师",
}

_STAGE_SLOT_HINT = {
    "L1": "只允许 slots: spec.compose, spec.theme_defaults",
    "L2": "只允许 slots: data.prepare, data.aggregate, data.encode",
    "L3": "只允许 slots: marks.*, scales.*, colorbar.apply",
    "L4": "只允许 slots: axes.*, legend.apply, grid.apply, annot.*, theme.*",
}


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
            raise RuntimeError("Missing LLM_API_KEY; 请先在 .env 中配置 Zhizengzeng/OpenAI Key")
        self.model = model or os.getenv("LLM_MODEL") or "gpt-4.1-mini"
        timeout_env = os.getenv("LLM_TIMEOUT")
        if timeout_env:
            try:
                timeout = float(timeout_env)
            except ValueError:
                pass
        self.timeout = max(timeout, 30.0)
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
        response = self._session.post(
            f"{self.base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=(10, self.timeout),
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and data.get("code") not in (None, 0):
            raise RuntimeError(f"LLM 调用失败: {data.get('code')} {data.get('msg')}")
        content = data["choices"][0]["message"].get("content", "{}").strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.S)
            if match:
                return json.loads(match.group(0))
            raise RuntimeError("模型未返回可解析的 JSON")


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
    feedback = payload.get("feedback") or payload.get("feedback_text") or "无"
    intro = f"【角色】{_STAGE_NAMES.get(stage, stage)}（{stage}）。{_STAGE_SLOT_HINT.get(stage, '')}\n"

    if stage == "L1":
        data_profile = _format_json(payload.get("data_profile", {}))
        intent = _format_json(payload.get("intent", {}))
        spec = _format_json(payload.get("spec", {}))
        task = (
            "- 基于数据画像与业务意图完善 canvas/overlays/scales/layout/theme/flags，并可组合多 overlay（如 bar+scatter/line）。\n"
            "- 支持右轴（overlays[i].yaxis='right'）与断轴/范围控制（scales.y_left.breaks / range），必要时写明理由。\n"
            "- `spec.compose` 必须返回完整 dict；`spec.theme_defaults` 提供字体/字号/色板/线宽/刻度策略等默认值（可 `{}`）。\n"
            "- 优先增量调整，保持兼容性。\n"
        )
        body = (
            f"【数据画像】\n{data_profile}\n\n"
            f"【意图】\n{intent}\n\n"
            f"【当前 spec】\n{spec}\n\n"
            f"【上一轮反馈】\n{feedback}\n\n"
            f"【任务】\n{task}"
        )
    elif stage == "L2":
        df_preview = _format_table(payload.get("df_head", {}))
        spec = _format_json(payload.get("spec", {}))
        task = (
            "- `data.prepare` 负责清洗/缺失处理/类型转换（如 to_datetime）、派生列与过滤，必须返回 DataFrame。\n"
            "- `data.aggregate` 仅在需要聚合或 topK 时使用；否则直接 `return df`。\n"
            "- `data.encode` 输出绘图编码所需列（颜色/尺寸等），若无需额外处理可 `return df`。\n"
            "- 禁止出现轴/图例/注释/主题或 Matplotlib 绘图指令。\n"
        )
        body = (
            f"【数据预览（head）】\n{df_preview}\n\n"
            f"【当前 spec】\n{spec}\n\n"
            f"【上一轮反馈】\n{feedback}\n\n"
            f"【任务】\n{task}"
        )
    elif stage == "L3":
        df_preview = _format_table(payload.get("dff_head", {}))
        spec = _format_json(payload.get("spec", {}))
        task = (
            "- 为 overlays 生成 marks.*，负责几何绘制/调色/图例标签，可按需输出误差棒、CI、回归或密度等增强。\n"
            "- 合理设置 scales.* 与 colorbar.apply，正确处理对数轴、安全断轴、双轴范围与色条。\n"
            "- 禁止调用轴/图例/注释/主题相关函数。\n"
        )
        body = (
            f"【数据预览（encoded head）】\n{df_preview}\n\n"
            f"【当前 spec】\n{spec}\n\n"
            f"【上一轮反馈】\n{feedback}\n\n"
            f"【任务】\n{task}"
        )
    elif stage == "L4":
        spec = _format_json(payload.get("spec", {}))
        task = (
            "- 设置标题四向、轴标签/单位、刻度密度与旋转，整理 legend/grid/spines。\n"
            "- 可添加 annot.reference_lines/bands/text/inset 等注释，及 theme.* 字体/配色/背景微调。\n"
            "- 禁止操作 data.* 或 marks.*。\n"
        )
        body = (
            f"【当前 spec】\n{spec}\n\n"
            f"【上一轮反馈】\n{feedback}\n\n"
            f"【任务】\n{task}"
        )
    else:
        body = f"【上下文】\n{_format_json(payload)}\n\n【上一轮反馈】\n{feedback}\n"

    return f"{intro}{body}\n\n{OUTPUT_CONTRACT}"

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
    notes = response_dict.get("notes", "")
    if not isinstance(notes, str):
        notes = ""
    return {"slots": clean_slots, "notes": notes, "prompt": prompt, "response": raw_response}


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
            else:
                out = _llm_generate_slots(layer, payload)
            emit(
                "llm_io",
                {
                    "round": round_idx,
                    "stage": layer,
                    "stage_name": stage_name,
                    "prompt": out.get("prompt", ""),
                    "response": out.get("response"),
                },
            )
            ok_layer, rej_layer = _layer_guard(layer, out.get("slots"))
            stage_logs[layer] = {
                "prompt": out.get("prompt"),
                "response": _snapshot(out.get("response")),
                "payload": _snapshot(payload),
                "notes": out.get("notes", ""),
                "raw_slots": out.get("slots", {}),
                "accepted_slots": ok_layer,
                "rejected_slots": rej_layer,
            }
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
                },
            )

        slots: Dict[str, str] = {}
        for layer in ("L1", "L2", "L3", "L4"):
            slots.update(ok_by_layer.get(layer, {}))

        emit("slots_assembled", {"round": round_idx, "slot_count": len(slots)})

        py_code = assemble_with_slots(slots)
        out_png = str(run_dir / f"figure_round_{round_idx}.png")

        emit("execution_start", {"round": round_idx, "output": out_png})
        exec_result = execute_script(py_code, df, base_intent, ctx, out_png)
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

        selected = {
            "round": round_idx,
            "png_path": exec_result.get("png_path"),
            "scores": last_scores,
            "diagnostics": judge_result.get("diagnostics", []),
            "spec": spec,
            "slots": slots,
            "stderr": exec_result.get("stderr", ""),
            "stages": stage_logs,
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

        if last_scores["visual_form"] >= 0.75 and last_scores["data_fidelity"] >= 0.75:
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










