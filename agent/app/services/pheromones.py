from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List
import time


class EvidenceType(str, Enum):
    constraint = "constraint"
    style = "style"
    geom = "geom"
    layout = "layout"
    ref = "ref"


@dataclass
class PheromoneLink:
    level: int
    etype: EvidenceType
    delta: Dict[str, float]
    patch: Dict[str, Any]
    msg: str = ""
    ts: float = field(default_factory=lambda: time.time())


class PheroStore:
    """Append-only store for typed, timestamped pheromone links."""

    def __init__(self) -> None:
        self.links: List[PheromoneLink] = []

    def append(self, link: PheromoneLink) -> None:
        self.links.append(link)

    def summary(self) -> Dict[str, Any]:
        return {
            "total": len(self.links),
            "by_type": {t.value: sum(1 for link in self.links if link.etype == t) for t in EvidenceType},
        }

    def to_json(self) -> List[Dict[str, Any]]:
        return [asdict(link) for link in self.links]

    def tail(self, n: int = 3) -> List[PheromoneLink]:
        if n <= 0:
            return []
        return self.links[-n:]
