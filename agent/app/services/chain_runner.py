from __future__ import annotations

import base64
import json
from typing import Any, Callable, Dict, Mapping, Optional

import pandas as pd
from jinja2 import Template

from .excel_loader import LoadedTable
from .judge import simple_judge
from .pheromones import EvidenceType, PheroStore, PheromoneLink
from .prompts_chain import (
    FEEDBACK_TEMPLATE,
    SCHEMA_L1,
    SCHEMA_L2,
    SCHEMA_L3,
    SCHEMA_L4,
    SYSTEM_COMMON,
    USER_L1,
    USER_L2,
    USER_L3,
    USER_L4,
)


class LLMClient:
    """Lightweight client around the Chat Completions endpoint returning JSON."""

    def __init__(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: Optional[str],
        timeout: int = 60,
    ) -> None:
        import os
        import requests

        self.base = (api_base or os.getenv("LLM_API_BASE") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        self.model = model or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        self.timeout = timeout
        self._requests = requests

    def chat_json(self, messages: list[dict[str, Any]]) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        response = self._requests.post(
            f"{self.base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)


def render_template(template: str, **kwargs: Any) -> str:
    return Template(template).render(**kwargs)


def merge_final_spec(l1: Dict[str, Any], l2: Dict[str, Any], l3: Dict[str, Any], l4: Dict[str, Any]) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "table_name": l1.get("table"),
        "chart_type": l1.get("chart_family"),
        "dimensions": l1.get("dimensions") or [],
        "measures": l1.get("measures") or [],
        "time_granularity": l1.get("time_granularity"),
        "encoding": l2.get("encoding_plan", {}),
        "transforms": l2.get("transforms", []),
        "title": l4.get("title"),
        "xlabel": l4.get("xlabel"),
        "ylabel": l4.get("ylabel"),
        "labels": l4.get("labels"),
        "tick_density": l4.get("tick_density"),
        "theme": l4.get("theme"),
        "legend": l3.get("legend", True),
        "bins": l3.get("bins"),
        "stack": l3.get("stack"),
        "scales": l3.get("scales"),
        "order": l3.get("order"),
        "units": l3.get("units"),
        "constraints": l1.get("constraints") or {},
        "agg": None,
    }

    for transform in l2.get("transforms", []):
        if transform.get("op") == "aggregate":
            measures = transform.get("measures") or []
            if measures:
                agg = measures[0].get("agg")
                if agg:
                    spec["agg"] = agg
    return spec


class ChainRunner:
    """Implements the linear Sense?Plan?Code/Patch?Render?Judge loop."""

    def __init__(
        self,
        llm: LLMClient,
        code_renderer: Callable[[Dict[str, Any]], str],
        sandbox_runner: Callable[[str, pd.DataFrame], bytes],
        form_threshold: float = 70.0,
        fidelity_threshold: float = 70.0,
    ) -> None:
        self.llm = llm
        self.code_renderer = code_renderer
        self.sandbox_runner = sandbox_runner
        self.form_threshold = form_threshold
        self.fidelity_threshold = fidelity_threshold
        self.store = PheroStore()

    def reset(self) -> None:
        self.store = PheroStore()

    def run_once(
        self,
        profile_json: str,
        user_goal: str,
        chart_family: str,
        feedback: str = "",
    ) -> Dict[str, Any]:
        feedback_block = ""
        if feedback:
            feedback_block = render_template(FEEDBACK_TEMPLATE, feedback=feedback)

        # L1
        msg = render_template(
            USER_L1,
            profile_json=profile_json,
            user_goal=user_goal,
            chart_family=chart_family,
            schema=SCHEMA_L1,
            feedback=feedback_block,
        )
        l1 = self.llm.chat_json(
            [
                {"role": "system", "content": SYSTEM_COMMON},
                {"role": "user", "content": msg},
            ]
        )
        self.store.append(
            PheromoneLink(
                level=1,
                etype=EvidenceType.constraint,
                delta={"form": 0.0, "fidelity": 0.0, "cohesion": 0.0},
                patch=l1,
            )
        )

        # L2
        msg = render_template(
            USER_L2,
            chart_plan_json=json.dumps(l1, ensure_ascii=False),
            schema=SCHEMA_L2,
        )
        l2 = self.llm.chat_json(
            [
                {"role": "system", "content": SYSTEM_COMMON},
                {"role": "user", "content": msg},
            ]
        )
        self.store.append(
            PheromoneLink(
                level=2,
                etype=EvidenceType.geom,
                delta={"form": 0.0, "fidelity": 0.0, "cohesion": 0.0},
                patch=l2,
            )
        )

        # L3
        msg = render_template(
            USER_L3,
            chart_plan_json=json.dumps(l1, ensure_ascii=False),
            orchestration_json=json.dumps(l2, ensure_ascii=False),
            schema=SCHEMA_L3,
        )
        l3 = self.llm.chat_json(
            [
                {"role": "system", "content": SYSTEM_COMMON},
                {"role": "user", "content": msg},
            ]
        )
        self.store.append(
            PheromoneLink(
                level=3,
                etype=EvidenceType.style,
                delta={"form": 0.0, "fidelity": 0.0, "cohesion": 0.0},
                patch=l3,
            )
        )

        # L4
        msg = render_template(
            USER_L4,
            chart_plan_json=json.dumps(l1, ensure_ascii=False),
            orchestration_json=json.dumps(l2, ensure_ascii=False),
            calibration_json=json.dumps(l3, ensure_ascii=False),
            schema=SCHEMA_L4,
        )
        l4 = self.llm.chat_json(
            [
                {"role": "system", "content": SYSTEM_COMMON},
                {"role": "user", "content": msg},
            ]
        )
        self.store.append(
            PheromoneLink(
                level=4,
                etype=EvidenceType.layout,
                delta={"form": 0.0, "fidelity": 0.0, "cohesion": 0.0},
                patch=l4,
            )
        )

        final_spec = merge_final_spec(l1, l2, l3, l4)
        return {"l1": l1, "l2": l2, "l3": l3, "l4": l4, "final_spec": final_spec}

    def emit_and_run(self, table: LoadedTable, final_spec: Dict[str, Any]) -> Dict[str, Any]:
        code = self.code_renderer(final_spec)
        ok = True
        png_bytes = b""
        diagnosis = ""
        try:
            png_bytes = self.sandbox_runner(code, table.dataframe.copy())
        except Exception as exc:  # noqa: BLE001
            ok = False
            diagnosis = str(exc)
        judge = simple_judge(ok, final_spec, diagnosis)
        self.store.append(
            PheromoneLink(
                level=4,
                etype=EvidenceType.ref,
                delta={
                    "form": float(judge["VisualForm"]) - 50.0,
                    "fidelity": float(judge["DataFidelity"]) - 50.0,
                    "cohesion": 0.0,
                },
                patch={"final_spec": final_spec},
                msg=diagnosis,
            )
        )
        encoded = base64.b64encode(png_bytes).decode("utf-8") if png_bytes else ""
        return {"code": code, "png_base64": encoded, "judge": judge, "ok": ok}

    def run_chain(
        self,
        profile_json: str,
        user_goal: str,
        chart_family: str,
        tables: Mapping[str, LoadedTable],
        rounds: int,
    ) -> Dict[str, Any]:
        self.reset()
        iterations: list[Dict[str, Any]] = []
        feedback_text = ""
        selected_output: Dict[str, Any] | None = None

        capped_rounds = max(1, rounds)

        for round_idx in range(capped_rounds):
            plan = self.run_once(profile_json, user_goal, chart_family, feedback_text)
            spec = plan["final_spec"]
            table_name = spec.get("table_name")
            if table_name not in tables:
                raise ValueError(f"Table '{table_name}' not found in workbook")
            emit_result = self.emit_and_run(tables[table_name], spec)
            judge = emit_result["judge"]

            iteration_record = {
                "round": round_idx + 1,
                "plan": plan,
                "emit": emit_result,
                "feedback": feedback_text,
            }
            iterations.append(iteration_record)

            selected_output = {
                "final_spec": spec,
                "code": emit_result["code"],
                "png_base64": emit_result["png_base64"],
                "judge": judge,
            }

            if self._is_satisfied(judge):
                break

            feedback_text = self._compose_feedback(judge)

        pheromones = self.store.to_json()
        return {
            "output": selected_output,
            "iterations": iterations,
            "pheromones": pheromones,
        }

    def _is_satisfied(self, judge: Dict[str, Any]) -> bool:
        return judge.get("VisualForm", 0) >= self.form_threshold and judge.get("DataFidelity", 0) >= self.fidelity_threshold

    def _compose_feedback(self, judge: Dict[str, Any]) -> str:
        snippets: list[str] = []
        diagnosis = judge.get("diagnosis")
        if diagnosis:
            snippets.append(f"??:{diagnosis}")
        for link in self.store.tail(3):
            summary = f"L{link.level} {link.etype.value}: ?form={link.delta['form']}, ?fid={link.delta['fidelity']}"
            if link.msg:
                summary += f" | {link.msg}"
            snippets.append(summary)
        return "\n".join(snippets)
