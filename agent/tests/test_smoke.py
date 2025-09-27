from app.services.code_assembler import assemble_with_slots
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
