"""AgentForge FastAPI entry point.

Run: ``uv run uvicorn backend.app.main:app --reload``

Wires the deterministic gatekeeper, the evolving RQGM evaluator, the LangGraph
orchestrator (compiled with a durable checkpointer), local inference (mock
fallback) and the export generators behind the REST contract. Permissive CORS so
the Next.js dev server (different origin) can call it.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import (
    admin_router,
    catalog_router,
    graph_router,
    sessions_router,
)
from backend.domains.registry import list_domains
from backend.evaluator import versioning
from backend.inference.lemonade_client import get_lemonade_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm/compile the LangGraph so the HITL checkpointer is ready (b4: compiled in).
    try:
        from backend.graph.orchestrator import get_compiled_graph

        get_compiled_graph()
    except Exception:
        # Non-fatal: the contract endpoints work without the orchestrator warmed.
        pass
    yield


app = FastAPI(
    title="AgentForge API",
    version="0.1.0",
    description="Self-evolving AI-agent sizing/education platform on the AMD ROCm stack.",
    lifespan=lifespan,
)

# Permissive CORS for local frontend dev (another origin).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(catalog_router)
app.include_router(sessions_router)
app.include_router(admin_router)
app.include_router(graph_router)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "AgentForge",
        "tagline": "Self-evolving AI-agent sizing/education on the AMD open-source ROCm stack.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    client = get_lemonade_client()
    info: dict[str, Any] = {
        "status": "ok",
        "inference": {"using_mock": client.using_mock, "base_url": client.base_url},
        "epoch_id": versioning.get_epoch(),
        "champion_version": versioning.get_champion_version(),
        "domains": [d.id for d in list_domains()],
    }
    try:
        from backend.memory import get_memory

        info["memory"] = get_memory().stats()
    except Exception as exc:  # pragma: no cover
        info["memory"] = {"error": str(exc)}
    return info
