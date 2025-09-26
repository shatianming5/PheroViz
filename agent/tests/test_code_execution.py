import pandas as pd
import pytest

from app.services.code_templates import render_code_from_spec
from app.services.sandbox import run_render_chart, UnsafeCodeError


def test_render_and_run_bar_chart():
    spec = {
        "table_name": "Sheet1",
        "chart_type": "bar",
        "dimensions": ["Region"],
        "measures": ["Sales"],
        "encoding": {"x": "Region", "y": "Sales", "series": None},
        "transforms": [],
        "constraints": {},
        "legend": True,
        "title": "Sales by Region",
        "xlabel": "Region",
        "ylabel": "Sales",
    }
    code = render_code_from_spec(spec)
    assert "def generate_chart" in code

    df = pd.DataFrame({"Region": ["East", "West"], "Sales": [100, 150]})
    png_bytes = run_render_chart(code, df)
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 0


def test_sandbox_rejects_unsafe_code():
    code = "def generate_chart(dataset):\n    open('x.txt', 'w')\n    return b''\n"
    df = pd.DataFrame({"value": [1, 2, 3]})
    with pytest.raises(UnsafeCodeError):
        run_render_chart(code, df)
