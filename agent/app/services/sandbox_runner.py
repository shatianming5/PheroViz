from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict


def execute_script(
    py_code: str,
    df,
    intent: Dict[str, Any],
    ctx: Dict[str, Any],
    out_png: str,
    timeout_s: int = 15,
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        scaffold = tmp / "scaffold.py"
        shim = tmp / "shim.py"
        p_df = tmp / "df.pkl"
        p_intent = tmp / "intent.json"
        p_ctx = tmp / "ctx.json"

        scaffold.write_text(py_code, encoding="utf-8")
        with p_df.open("wb") as handle:
            pickle.dump(df, handle)
        p_intent.write_text(json.dumps(intent or {}, ensure_ascii=False), encoding="utf-8")
        p_ctx.write_text(json.dumps(ctx or {}, ensure_ascii=False), encoding="utf-8")

        shim.write_text(
            textwrap.dedent(
                """
                import json
                import pickle
                import sys
                from pathlib import Path

                import matplotlib

                matplotlib.use("Agg")
                import scaffold

                p_df = Path(sys.argv[1])
                p_int = Path(sys.argv[2])
                p_ctx = Path(sys.argv[3])
                out_png = Path(sys.argv[4])

                df = pickle.loads(p_df.read_bytes())
                intent = json.loads(p_int.read_text(encoding="utf-8"))
                ctx = json.loads(p_ctx.read_text(encoding="utf-8"))
                scaffold.run(df, intent, ctx, str(out_png))
                """
            ).strip(),
            encoding="utf-8",
        )

        out_png_path = Path(out_png)
        out_png_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(shim),
            str(p_df),
            str(p_intent),
            str(p_ctx),
            str(out_png_path),
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True, encoding='utf-8', errors='replace', timeout=timeout_s,
            )
            ok = (
                proc.returncode == 0
                and out_png_path.exists()
                and out_png_path.stat().st_size > 0
            )
            stderr = (proc.stderr or "")
            if proc.stdout:
                stdout = proc.stdout.strip()
                stderr = (stderr + ("\n" if stderr and stdout else "") + stdout).strip()
        except subprocess.TimeoutExpired as exc:
            ok = False
            stderr = f"Execution timed out after {timeout_s}s: {exc}"

        return {
            "ok": ok,
            "png_path": str(out_png_path) if ok else None,
            "stderr": stderr,
        }

