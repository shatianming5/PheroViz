from __future__ import annotations

from typing import Any, Dict


def simple_judge(ok: bool, spec: Dict[str, Any], diagnosis: str = "") -> Dict[str, Any]:
    """Return lightweight scores aligned with the paper terminology."""

    form_score = 50.0
    fidelity_score = 50.0

    if ok:
        form_score += 10 if spec.get("title") else 0
        form_score += 10 if spec.get("xlabel") and spec.get("ylabel") else 0
        form_score += 10 if spec.get("legend", True) else 0
        fidelity_score += 10 if spec.get("y") else 0
        fidelity_score += 10 if spec.get("x") else 0

    return {
        "VisualForm": min(100.0, form_score),
        "DataFidelity": min(100.0, fidelity_score),
        "SeriesCohesion": "NA",
        "diagnosis": diagnosis,
    }
