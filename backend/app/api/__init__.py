"""API routers implementing the REST contract (+ supplementary graph routes)."""

from backend.app.api.admin import router as admin_router  # noqa: F401
from backend.app.api.catalog import router as catalog_router  # noqa: F401
from backend.app.api.graph_api import router as graph_router  # noqa: F401
from backend.app.api.sessions import router as sessions_router  # noqa: F401

__all__ = ["admin_router", "catalog_router", "graph_router", "sessions_router"]
