from pathlib import Path

import pandas as pd

from app.services.code_assembler import assemble_with_slots
from app.services.sandbox_runner import execute_script
from app.services.spec_deriver import derive_spec
from app.services.spec_validator import validate_spec


def test_spec_roundtrip():
    profile = {"columns": {"日期": "datetime", "销量": "numeric"}}
    intent = {
        "chart_family": "line",
        "x": "日期",
        "y": "销量",
        "aesthetics": {"palette": "ColorBlindSafe"},
    }
    spec = validate_spec(derive_spec(intent, profile))
    assert "overlays" in spec
    assert spec["overlays"][0]["mark"] == "line"


def test_assembler_empty_slots_ok():
    py_code = assemble_with_slots({})
    assert "def run(" in py_code


def test_assembled_scaffold_falls_back_from_invalid_palette(
    tmp_path: Path,
) -> None:
    py_code = assemble_with_slots({})
    result = execute_script(
        py_code,
        pd.DataFrame({"x": [0, 1], "y": [1, 2]}),
        {"chart_family": "line", "x": "x", "y": "y"},
        {
            "spec": {
                "canvas": {"width": 640, "height": 480, "dpi": 100},
                "layout": {},
                "theme": {"palette_global": "ColorBlindSafe"},
                "flags": {},
                "overlays": [
                    {
                        "id": "line",
                        "mark": "line",
                        "variant": "main",
                        "x": "x",
                        "y": "y",
                        "yaxis": "left",
                    }
                ],
                "scales": {},
            }
        },
        str(tmp_path / "figure.png"),
    )

    assert result["ok"] is True
    assert result["ctx"]["spec"]["theme"]["palette_global"] == "tab10"
    assert result["ctx"]["_v2_meta"]["palette_fallbacks"] == {
        "ColorBlindSafe": "tab10"
    }
