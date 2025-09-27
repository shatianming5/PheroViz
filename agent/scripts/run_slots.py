import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.code_assembler import assemble_with_slots
from app.services.judge import judge
from app.services.sandbox_runner import execute_script


def _load_dataframe(path: str, sheet: str | None) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".xls", ".xlsx", ".xlsm"}:
        return pd.read_excel(p, sheet_name=sheet)
    return pd.read_csv(p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("excel_path")
    parser.add_argument("slots_json")
    parser.add_argument("--sheet", default=None)
    args = parser.parse_args()

    dataframe = _load_dataframe(args.excel_path, args.sheet)
    slots = json.loads(Path(args.slots_json).read_text(encoding="utf-8"))

    py_code = assemble_with_slots(slots)
    out_png = Path("runs/manual_figure.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    exec_result = execute_script(py_code, dataframe, {}, {}, str(out_png))
    png_path = exec_result.get("png_path") or str(out_png)
    scores = judge(png_path, exec_result.get("stderr", ""), dataframe, {"overlays": []})

    print(
        json.dumps(
            {
                "ok": exec_result["ok"],
                "png": exec_result.get("png_path"),
                "scores": scores,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
