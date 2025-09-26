from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ColumnProfileModel(BaseModel):
    name: str
    dtype: str
    semantic_type: str
    non_nulls: int
    distinct: int
    sample_values: List[Any]
    top_values: List[Dict[str, Any]]
    stats: Dict[str, Any]


class SheetProfileModel(BaseModel):
    sheet_name: str
    row_count: int
    column_count: int
    columns: List[ColumnProfileModel]
    sample_rows: List[Dict[str, Any]]


class ProfileResponse(BaseModel):
    sheets: List[SheetProfileModel]
    table_names: List[str]
    column_whitelist: Dict[str, List[str]]


class JudgeModel(BaseModel):
    VisualForm: float
    DataFidelity: float
    SeriesCohesion: str | float | None
    diagnosis: Optional[str] = None


class RunOutputModel(BaseModel):
    final_spec: Dict[str, Any]
    code: str
    png_base64: str
    judge: JudgeModel


class IterationModel(BaseModel):
    round: int
    plan: Dict[str, Any]
    emit: Dict[str, Any]
    feedback: str


class PlanEmitRunResponse(BaseModel):
    output: Optional[RunOutputModel]
    iterations: List[IterationModel]
    pheromones: List[Dict[str, Any]]
    audit_path: Optional[str] = Field(None, description="Filesystem path to persisted artifacts.")
