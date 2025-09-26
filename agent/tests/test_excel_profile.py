import io

import pandas as pd

from app.services import DataProfiler, ExcelLoader


def _build_workbook() -> bytes:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "Region": ["East", "West"] * 5,
            "Sales": [100, 120, 130, 90, 150, 160, 170, 180, 140, 155],
        }
    )
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return buffer.getvalue()


def test_excel_loader_and_profile():
    loader = ExcelLoader()
    data = _build_workbook()
    tables = loader.load_workbook(data)
    assert "Sheet1" in tables
    table = tables["Sheet1"].dataframe
    assert set(table.columns) == {"Date", "Region", "Sales"}

    profiler = DataProfiler()
    profile = profiler.build_profile(tables)
    assert profile["sheets"][0]["sheet_name"] == "Sheet1"
    columns = {col["name"] for col in profile["sheets"][0]["columns"]}
    assert {"Date", "Region", "Sales"} <= columns
    whitelist = profile["column_whitelist"]["Sheet1"]
    assert "Sales" in whitelist
