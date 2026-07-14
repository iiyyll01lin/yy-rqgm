# Backend image for AgentForge (CPU / no-ROCm). The platform runs without ROCm;
# heavy AMD wheels live in the optional [rocm] extra and are NOT installed here.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Install deps first (better layer caching). uv.lock is copied if present.
COPY pyproject.toml ./
COPY uv.lock* ./
COPY .python-version ./

# Source needed to build the editable `backend` package + runtime data.
COPY backend ./backend
COPY data ./data

RUN uv sync --no-dev

EXPOSE 8000

# On Linux, host.docker.internal is provided via extra_hosts in compose.
ENV LEMONADE_BASE_URL=http://host.docker.internal:8020/api/v1
ENV QDRANT_URL=http://qdrant:6333

CMD ["uv", "run", "--no-dev", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
