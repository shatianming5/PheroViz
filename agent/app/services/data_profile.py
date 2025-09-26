from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from .excel_loader import LoadedTable


def _normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if pd.isna(value):  # type: ignore[arg-type]
        return None
    return value


def _semantic_type(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
        return "temporal"
    if pd.api.types.is_numeric_dtype(series):
        return "quantitative"
    return "categorical"


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    semantic_type: str
    non_nulls: int
    distinct: int
    sample_values: List[Any]
    top_values: List[Dict[str, Any]]
    stats: Dict[str, Any]


class DataProfiler:
    """Produce lightweight JSON-friendly profiles for loaded tables."""

    def __init__(self, topk: int = 5, sample_rows: int = 3) -> None:
        self.topk = topk
        self.sample_rows = sample_rows

    def build_profile(self, tables: Mapping[str, LoadedTable]) -> Dict[str, Any]:
        sheets: List[Dict[str, Any]] = []
        whitelist: Dict[str, List[str]] = {}

        for name, table in tables.items():
            df = table.dataframe
            cols = [self._summarize_column(df[col], col) for col in df.columns]
            whitelist[name] = [col.name for col in cols]
            samples = [self._normalize_record(rec) for rec in df.head(self.sample_rows).to_dict(orient="records")]
            sheets.append(
                {
                    "sheet_name": name,
                    "row_count": int(len(df)),
                    "column_count": int(len(df.columns)),
                    "columns": [self._column_to_dict(col) for col in cols],
                    "sample_rows": samples,
                }
            )

        return {
            "sheets": sheets,
            "table_names": list(tables.keys()),
            "column_whitelist": whitelist,
        }

    def _summarize_column(self, series: pd.Series, name: str) -> ColumnProfile:
        semantic = _semantic_type(series)
        non_nulls = int(series.notna().sum())
        distinct = int(series.nunique(dropna=True))
        sample_values = [_normalize_value(val) for val in series.dropna().head(self.topk).tolist()]
        value_counts = (
            series.value_counts(dropna=False)
            .head(self.topk)
            .rename_axis('value')
            .reset_index(name='count')
        )
        top_values = [
            {"value": _normalize_value(row['value']), "count": int(row['count'])}
            for _, row in value_counts.iterrows()
        ]
        stats: Dict[str, Any] = {}
        if semantic == "quantitative":
            numeric = pd.to_numeric(series, errors="coerce")
            stats = {
                "min": _normalize_value(float(numeric.min())) if numeric.notna().any() else None,
                "max": _normalize_value(float(numeric.max())) if numeric.notna().any() else None,
                "mean": _normalize_value(float(numeric.mean())) if numeric.notna().any() else None,
                "std": _normalize_value(float(numeric.std(ddof=0))) if numeric.notna().any() else None,
            }
        return ColumnProfile(
            name=name,
            dtype=str(series.dtype),
            semantic_type=semantic,
            non_nulls=non_nulls,
            distinct=distinct,
            sample_values=sample_values,
            top_values=top_values,
            stats=stats,
        )

    def _column_to_dict(self, col: ColumnProfile) -> Dict[str, Any]:
        return {
            "name": col.name,
            "dtype": col.dtype,
            "semantic_type": col.semantic_type,
            "non_nulls": col.non_nulls,
            "distinct": col.distinct,
            "sample_values": col.sample_values,
            "top_values": col.top_values,
            "stats": col.stats,
        }

    def _normalize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {key: _normalize_value(value) for key, value in record.items()}
