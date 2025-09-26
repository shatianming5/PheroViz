from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass
class Settings:
    llm_api_base: str | None
    llm_api_key: str | None
    llm_model: str
    rounds_default: int
    rounds_max: int
    storage_root: Path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    root = Path(os.getenv("PHEROVIZ_STORAGE_ROOT", "runs")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return Settings(
        llm_api_base=os.getenv("LLM_API_BASE"),
        llm_api_key=os.getenv("LLM_API_KEY"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
        rounds_default=int(os.getenv("CHAIN_DEFAULT_ROUNDS", "3")),
        rounds_max=int(os.getenv("CHAIN_MAX_ROUNDS", "4")),
        storage_root=root,
    )
