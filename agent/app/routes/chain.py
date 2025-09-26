from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..core.settings import get_settings
from ..schemas.chain import PlanEmitRunResponse, ProfileResponse
from ..services import ChainRunner, DataProfiler, ExcelLoader, LLMClient
from ..services.code_templates import render_code_from_spec
from ..services.sandbox import run_render_chart
from ..utils.audit import AuditLogger

router = APIRouter(prefix="/chain", tags=["chain"])

ALLOWED_CHART_FAMILIES = {
    "auto",
    "line",
    "bar",
    "stacked_bar",
    "area",
    "pie",
    "scatter",
    "hist",
    "box",
    "heatmap",
}

_loader = ExcelLoader()
_profiler = DataProfiler()


def _read_file_bytes(upload: UploadFile) -> bytes:
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return data


def _build_runner(settings) -> ChainRunner:
    llm = LLMClient(settings.llm_api_base, settings.llm_api_key, settings.llm_model)
    return ChainRunner(llm=llm, code_renderer=render_code_from_spec, sandbox_runner=run_render_chart)


def _validate_chart_family(chart_family: str) -> str:
    family = (chart_family or "").lower()
    if family not in ALLOWED_CHART_FAMILIES:
        raise HTTPException(status_code=400, detail=f"chart_family must be one of {sorted(ALLOWED_CHART_FAMILIES)}")
    return family


def _validate_spec_columns(spec: Dict[str, Any], whitelist: Dict[str, Any]) -> None:
    table = spec.get("table_name")
    if table not in whitelist:
        raise HTTPException(status_code=400, detail=f"Table '{table}' not found in uploaded workbook.")
    allowed = set(whitelist[table])

    def check_column(col: Any, context: str) -> None:
        if col and col not in allowed:
            raise HTTPException(status_code=400, detail=f"Column '{col}' used in {context} is not present in table '{table}'.")

    for column in spec.get("dimensions", []):
        check_column(column, "dimensions")
    for column in spec.get("measures", []):
        check_column(column, "measures")

    encoding = spec.get("encoding") or {}
    check_column(encoding.get("x"), "encoding.x")
    check_column(encoding.get("y"), "encoding.y")
    check_column(encoding.get("series"), "encoding.series")

    for transform in spec.get("transforms", []):
        if isinstance(transform, dict) and transform.get("op") == "filter" and isinstance(transform.get("expr"), dict):
            check_column(transform["expr"].get("column"), "transforms.filter")

    constraints = spec.get("constraints") or {}
    for expr in constraints.get("filters") or []:
        if isinstance(expr, dict):
            check_column(expr.get("column"), "constraints.filter")


@router.post("/profile_excel", response_model=ProfileResponse)
def profile_excel(file: UploadFile = File(...)) -> ProfileResponse:
    workbook_bytes = _read_file_bytes(file)
    tables = _loader.load_workbook(workbook_bytes)
    if not tables:
        raise HTTPException(status_code=400, detail="Workbook contained no readable sheets.")
    profile = _profiler.build_profile(tables)
    return ProfileResponse.model_validate(profile)


@router.post("/plan_emit_run", response_model=PlanEmitRunResponse)
def plan_emit_run(
    file: UploadFile = File(...),
    user_goal: str = Form(...),
    chart_family: str = Form("auto"),
    rounds: int = Form(None),
    settings=Depends(get_settings),
) -> PlanEmitRunResponse:
    workbook_bytes = _read_file_bytes(file)
    tables = _loader.load_workbook(workbook_bytes)
    if not tables:
        raise HTTPException(status_code=400, detail="Workbook contained no readable sheets.")

    profile = _profiler.build_profile(tables)
    profile_json = json.dumps(profile, ensure_ascii=False, indent=2)

    family = _validate_chart_family(chart_family)
    rounds = rounds or settings.rounds_default
    rounds = max(1, min(rounds, settings.rounds_max))

    runner = _build_runner(settings)
    chain_result = runner.run_chain(profile_json, user_goal, family, tables, rounds)

    output = chain_result.get("output") or {}
    if output:
        _validate_spec_columns(output.get("final_spec", {}), profile.get("column_whitelist", {}))

    audit_logger = AuditLogger(settings.storage_root)
    run_inputs = {
        "user_goal": user_goal,
        "chart_family": family,
        "rounds": rounds,
    }
    audit_path = audit_logger.persist(
        run_inputs=run_inputs,
        profile=profile,
        output=output,
        pheromones=chain_result.get("pheromones"),
        iterations=chain_result.get("iterations"),
    )

    response_payload = {
        "output": output or None,
        "iterations": chain_result.get("iterations", []),
        "pheromones": chain_result.get("pheromones", []),
        "audit_path": str(audit_path),
    }
    return PlanEmitRunResponse.model_validate(response_payload)
