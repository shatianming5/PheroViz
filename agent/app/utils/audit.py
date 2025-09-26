from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class AuditLogger:
    """Persist chain run artifacts for later inspection."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def persist(
        self,
        run_inputs: Dict[str, Any],
        profile: Dict[str, Any],
        output: Dict[str, Any],
        pheromones: Any,
        iterations: Any,
    ) -> Path:
        run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        self._write_json(run_dir / "inputs.json", run_inputs)
        self._write_json(run_dir / "profile.json", profile)
        self._write_json(run_dir / "final_spec.json", output.get("final_spec", {}))
        self._write_json(run_dir / "judge.json", output.get("judge", {}))
        self._write_json(run_dir / "pheromones.json", pheromones)
        self._write_json(run_dir / "iterations.json", iterations)

        code_text = output.get("code") or ""
        (run_dir / "code.py").write_text(code_text, encoding="utf-8")

        png_b64 = output.get("png_base64") or ""
        if png_b64:
            png_bytes = base64.b64decode(png_b64)
            (run_dir / "chart.png").write_bytes(png_bytes)

        return run_dir

    def _write_json(self, path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
