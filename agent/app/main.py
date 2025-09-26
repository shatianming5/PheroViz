from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .core.settings import get_settings
from .routes.chain import router as chain_router

load_dotenv()

app = FastAPI(title="PheroViz Chain API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chain_router)


@app.on_event("startup")
async def ensure_storage() -> None:
    get_settings()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
