from __future__ import annotations

from typing import Any, Dict


def alt_text(spec: Dict[str, Any], df_sample_rows: int = 3) -> str:
    overlays = spec.get("overlays") or []
    parts = [f"{len(overlays)}-layer chart:"]
    for idx, overlay in enumerate(overlays, 1):
        parts.append(
            f"{idx}) {overlay.get('mark')} x={overlay.get('x')} y={overlay.get('y')} "
            f"group={overlay.get('group')} axis={overlay.get('yaxis', 'left')}"
        )
    title = ((spec.get("layout") or {}).get("titles") or {}).get("top")
    if title:
        parts.append(f"title: {title}")
    return " ".join(parts)
