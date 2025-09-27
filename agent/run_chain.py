import argparse
import json
import sys
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console, Group
from rich.json import JSON
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from app.services.single_chain_runner import run_chain

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

console = Console(soft_wrap=False, force_terminal=True, color_system='auto')

EVENT_LABELS: Dict[str, str] = {
    "startup": "初始化",
    "run_directory_ready": "准备运行目录",
    "data_loaded": "加载数据",
    "spec_ready": "生成初始 Spec",
    "context_ready": "上下文就绪",
    "round_start": "开始新轮",
    "stage_start": "阶段执行",
    "stage_complete": "阶段完成",
    "llm_io": "LLM 交互",
    "slots_assembled": "合成 Slot",
    "execution_start": "执行绘图脚本",
    "execution_end": "脚本执行结束",
    "judging_start": "开始评估",
    "judging_complete": "评估完成",
    "round_complete": "轮次完成",
    "artifact_written": "写入产物",
    "round_success": "分数达标",
    "feedback_ready": "生成反馈",
    "finished": "告一段落",
}


def _trim(text: str, limit: int = 70) -> str:
    text = text.strip()
    if not text:
        return ""
    return text if len(text) <= limit else f"{text[: limit - 1]}..."



def _truncate_block(text: str, limit: int = 1200) -> str:
    trimmed = text.strip()
    if len(trimmed) <= limit:
        return trimmed
    return f"{trimmed[: limit - 1]}..."



def _stringify_response(data: Any, limit: int = 1200) -> str:
    if data is None:
        return ""
    if isinstance(data, str):
        raw = data
    else:
        try:
            raw = json.dumps(data, ensure_ascii=False, indent=2)
        except TypeError:
            raw = str(data)
    return _truncate_block(raw, limit)


def _detail_for_event(event: str, payload: Dict[str, Any]) -> str:
    if event == "startup":
        return _trim(payload.get("excel_path", ""))
    if event == "data_loaded":
        return f"{payload.get('rows', 0)} 行 / {len(payload.get('columns', []))} 列"
    if event == "spec_ready":
        keys = ", ".join(payload.get("keys", [])[:4])
        return f"Spec 字段: {keys}" if keys else "Spec 已生成"
    if event == "round_start":
        return f"第 {payload.get('round')} 轮"
    if event == "stage_start":
        return _trim(payload.get("hint", "")) or "准备生成代码"
    if event == "stage_complete":
        accepted = len(payload.get("accepted", []))
        rejected = len(payload.get("rejected", []))
        return f"接受 {accepted} / 拒绝 {rejected}"
    if event == "slots_assembled":
        return f"共 {payload.get('slot_count', 0)} 段"
    if event == "execution_end":
        stderr = payload.get("stderr")
        return "存在警告" if stderr else "执行无异常"
    if event == "judging_complete":
        scores = payload.get("scores", {})
        vf = scores.get("visual_form")
        df = scores.get("data_fidelity")
        return f"VF {vf:.2f} / DF {df:.2f}" if vf is not None and df is not None else ""
    if event == "round_complete":
        return f"诊断 {payload.get('diagnostics', 0)} 条"
    if event == "artifact_written":
        return _trim(payload.get("path", ""), 50)
    if event == "feedback_ready":
        return _trim(payload.get("feedback", ""))
    if event == "round_success":
        scores = payload.get("scores", {})
        return f"VF {scores.get('visual_form', 0):.2f} / DF {scores.get('data_fidelity', 0):.2f}"
    if event == "llm_io":
        response = payload.get("response")
        if isinstance(response, dict):
            slots = response.get("slots")
            if isinstance(slots, dict):
                return f"slots {len(slots)} 段"
        prompt = payload.get("prompt") or ""
        return f"prompt {len(str(prompt))} 字符" if prompt else ""
    if event == "finished":
        return _trim(payload.get("run_dir", ""), 50)
    return ""


def _format_event(event: str, payload: Dict[str, Any]) -> str:
    label = EVENT_LABELS.get(event, event)
    if event == "stage_start":
        stage = payload.get("stage_name") or payload.get("stage")
        return f"[cyan]▶ {stage}[/] {label}"
    if event == "stage_complete":
        stage = payload.get("stage_name") or payload.get("stage")
        rejected = len(payload.get("rejected", []))
        color = "green" if rejected == 0 else "yellow"
        return f"[{color}]✔ {stage}[/] 接受 {len(payload.get('accepted', []))} 拒绝 {rejected}"
    if event == "round_start":
        return f"[bold cyan]Round {payload.get('round')}[/] 开始"
    if event == "round_success":
        scores = payload.get("scores", {})
        return (
            f"[bold green]达到阈值[/] VF={scores.get('visual_form', 0):.2f} "
            f"DF={scores.get('data_fidelity', 0):.2f}"
        )
    if event == "llm_io":
        stage = payload.get("stage_name") or payload.get("stage")
        return f"[blue]LLM[/] {stage}"
    if event == "execution_end":
        return "[green]执行完成[/]" if not payload.get("stderr") else "[yellow]执行完成 (含 stderr)"
    if event == "judging_complete":
        scores = payload.get("scores", {})
        return f"[bold white]评分[/] VF={scores.get('visual_form', 0):.2f} DF={scores.get('data_fidelity', 0):.2f}"
    if event == "feedback_ready":
        return "[magenta]反馈已生成[/]"
    if event == "artifact_written":
        return f"[dim]保存 {payload.get('path', '')}[/]"
    if event == "finished":
        return "[bold green]流程结束[/]"
    return f"[dim]{label}[/]"


def _build_progress_panel(state: Dict[str, Any]) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="cyan", no_wrap=True)
    table.add_column(justify="left", style="bold white")
    table.add_row("阶段", state.get("phase", "-"))
    table.add_row("轮次", state.get("round", "-"))
    table.add_row("步骤", state.get("stage", "-"))
    detail = state.get("detail") or "-"
    table.add_row("详情", detail)
    return Panel(table, title="运行进度", border_style="bright_magenta")



def _build_llm_panel(state: Dict[str, Any]) -> Panel:
    history: List[Dict[str, str]] = state.get("llm_history", [])  # type: ignore[assignment]
    if not history:
        return Panel(Text("等待 LLM 交互...", style="dim"), title="LLM 交互", border_style="blue")
    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    table.add_column("#", style="cyan", no_wrap=True)
    table.add_column("阶段", style="bold white", no_wrap=True)
    table.add_column("Prompt", style="white", overflow="fold")
    table.add_column("Response", style="white", overflow="fold")
    for idx, item in enumerate(history, 1):
        table.add_row(str(idx), item.get("stage", "-"), item.get("prompt", ""), item.get("response", ""))
    return Panel(table, title=f"LLM 交互 ({len(history)})", border_style="blue")



def _build_log_panel(logs: Deque[str]) -> Panel:
    if not logs:
        body = Text("等待执行...", style="dim")
    else:
        body = Text()
        for idx, raw in enumerate(logs):
            if idx:
                body.append("\n")
            body.append_text(Text.from_markup(raw))
    return Panel(body, title="事件日志", border_style="grey50")


def _render(state: Dict[str, Any], logs: Deque[str]) -> Group:
    return Group(_build_progress_panel(state), _build_llm_panel(state), _build_log_panel(logs))


def _parse_intent(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:  # noqa: PERF203
        raise SystemExit(f"Invalid JSON for --intent: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("excel_path")
    parser.add_argument("user_goal")
    parser.add_argument("chart_family")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--intent", default=None, help="JSON string of intent")
    args = parser.parse_args()

    intent_payload = _parse_intent(args.intent)

    state: Dict[str, Any] = {
        "phase": "等待启动",
        "round": "-",
        "stage": "-",
        "detail": "",
        "llm_stage": "-",
        "llm_prompt": "",
        "llm_response": "",
        "llm_history": [],
    }
    event_log: Deque[str] = deque(maxlen=8)

    result: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None

    with Live(_render(state, event_log), refresh_per_second=8, console=console) as live:
        def handle_progress(event: str, payload: Dict[str, Any]) -> None:
            state["phase"] = EVENT_LABELS.get(event, event)
            if "round" in payload:
                state["round"] = str(payload.get("round"))
            stage_name = payload.get("stage_name") or payload.get("stage") or "-"
            if event in {"stage_start", "stage_complete", "llm_io"}:
                state["stage"] = stage_name
            elif event in {"round_start", "feedback_ready", "round_success", "finished"}:
                state.setdefault("stage", "-")
            if event == "llm_io":
                state["llm_stage"] = stage_name
                prompt = payload.get("prompt") or ""
                prompt_text = _truncate_block(str(prompt), 1200)
                response_text = _stringify_response(payload.get("response"))
                state["llm_prompt"] = prompt_text
                state["llm_response"] = response_text
                history: List[Dict[str, str]] = state.setdefault("llm_history", [])  # type: ignore[assignment]
                history.append(
                    {
                        "stage": stage_name,
                        "prompt": _truncate_block(str(prompt), 400),
                        "response": _stringify_response(payload.get("response"), 400),
                    }
                )
            detail = _detail_for_event(event, payload)
            if detail:
                state["detail"] = detail
            event_log.appendleft(_format_event(event, payload))
            live.update(_render(state, event_log))

        try:
            result = run_chain(
                args.excel_path,
                args.user_goal,
                args.chart_family,
                rounds=args.rounds,
                sheet=args.sheet,
                intent=intent_payload,
                progress_callback=handle_progress,
            )
        except Exception as exc:  # pragma: no cover - propagate but prettify
            error = exc
            state["phase"] = "执行失败"
            state["detail"] = _trim(str(exc))
            event_log.appendleft(f"[bold red]错误[/] {_trim(str(exc), 60)}")
            live.update(_render(state, event_log))

    if error is not None:
        console.print_exception(show_locals=False)
        raise SystemExit(1)

    console.rule("执行完成")

    if not result:
        console.print(Panel("链路执行未产生输出", title="提示", border_style="red"))
        return

    scores = result.get("scores") or {}
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="cyan", justify="right", no_wrap=True)
    summary.add_column(style="bold white")
    summary.add_row("轮次", str(result.get("round", "-")))
    summary.add_row("视觉表现", f"{scores.get('visual_form', 0.0):.2f}")
    summary.add_row("数据保真", f"{scores.get('data_fidelity', 0.0):.2f}")
    if result.get("png_path"):
        summary.add_row("图像输出", _trim(result["png_path"], 60))
    console.print(Panel(summary, title="链路结果", border_style="green"))

    diagnostics = result.get("diagnostics") or []
    if diagnostics:
        diag_table = Table(show_header=True, header_style="bold cyan")
        diag_table.add_column("槽位", style="magenta")
        diag_table.add_column("描述", style="white", overflow="fold")
        diag_table.add_column("严重度", justify="center", style="yellow")
        for diag in diagnostics:
            slot = diag.get("slot") or diag.get("key") or "-"
            hint = diag.get("hint") or diag.get("key") or ""
            diag_table.add_row(slot, hint, str(diag.get("sev", "-")))
        console.print(Panel(diag_table, title="诊断信息", border_style="yellow"))

    stderr_text = _trim(result.get("stderr", ""), 400)
    if stderr_text:
        console.print(Panel(Text(stderr_text), title="stderr", border_style="red"))

    console.print(Panel(JSON.from_data(result, indent=2), title="原始 JSON", border_style="cyan", expand=False))


if __name__ == "__main__":
    main()
