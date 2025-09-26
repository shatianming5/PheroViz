from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Iterable, List

import pandas as pd


@dataclass
class LoadedTable:
    name: str
    dataframe: pd.DataFrame
    header_row: int


class ExcelLoader:
    """Load Excel workbooks while inferring headers and basic types."""

    def __init__(self, max_sample_rows: int = 200) -> None:
        self.max_sample_rows = max_sample_rows

    def load_workbook(self, data: bytes, sheet_names: Iterable[str] | None = None) -> Dict[str, LoadedTable]:
        buffer = BytesIO(data)
        raw = pd.read_excel(buffer, sheet_name=None, header=None, dtype=object)
        tables: Dict[str, LoadedTable] = {}

        selected_items = raw.items()
        if sheet_names:
            sheet_set = {name.lower() for name in sheet_names}
            selected_items = [(name, df) for name, df in raw.items() if name.lower() in sheet_set]

        for sheet_name, frame in selected_items:
            if frame.empty:
                continue
            header_row = self._detect_header_row(frame)
            table_df = self._build_dataframe(frame, header_row)
            tables[sheet_name] = LoadedTable(name=sheet_name, dataframe=table_df, header_row=header_row)
        return tables

    def _detect_header_row(self, frame: pd.DataFrame) -> int:
        limit = min(len(frame), self.max_sample_rows)
        best_idx = 0
        best_score = float("-inf")
        for idx in range(limit):
            row = frame.iloc[idx]
            non_null = row.notna().sum()
            str_like = sum(self._looks_like_header_value(v) for v in row)
            uniqueness = row.astype(str).nunique(dropna=True)
            score = (non_null * 1.5) + (str_like * 2.5) + uniqueness
            if score > best_score:
                best_idx = idx
                best_score = score
        return best_idx

    def _build_dataframe(self, frame: pd.DataFrame, header_row: int) -> pd.DataFrame:
        header_series = frame.iloc[header_row].fillna("").astype(str).str.strip()
        columns = self._dedupe_columns(header_series.tolist())
        data = frame.iloc[header_row + 1 :].copy()
        data.columns = columns
        data = data.dropna(axis=1, how="all")
        data = data.dropna(how="all").reset_index(drop=True)
        data = self._coerce_types(data)
        return data

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in df.columns:
            series = df[column]
            if series.dropna().empty:
                continue
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().mean() > 0.7:
                df[column] = numeric
                continue
            datetime = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
            if datetime.notna().mean() > 0.7:
                df[column] = datetime
        return df

    def _looks_like_header_value(self, value: object) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return False
            if len(stripped) <= 32:
                return True
        return False

    def _dedupe_columns(self, columns: List[str]) -> List[str]:
        seen: Dict[str, int] = {}
        result: List[str] = []
        for col in columns:
            base = col or "column"
            count = seen.get(base, 0)
            if count:
                new_name = f"{base}_{count+1}"
            else:
                new_name = base
            seen[base] = count + 1
            result.append(new_name)
        return result
