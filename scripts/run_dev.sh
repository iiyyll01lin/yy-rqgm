#!/usr/bin/env bash
# Dev convenience: start the AgentForge backend with hot reload.
#
#   scripts/run_dev.sh            # runs with the deterministic inference mock
#   LEMONADE_BASE_URL=http://localhost:8020/api/v1 scripts/run_dev.sh
#
# Requires uv (https://docs.astral.sh/uv/). CORS is permissive for the Next.js
# frontend dev server.
set -euo pipefail

cd "$(dirname "$0")/.."

# Fall back to the deterministic mock unless a Lemonade endpoint is configured.
if [[ -z "${LEMONADE_BASE_URL:-}" ]]; then
  export LEMONADE_FORCE_MOCK="${LEMONADE_FORCE_MOCK:-1}"
  echo "[run_dev] No LEMONADE_BASE_URL set -> using deterministic inference mock."
fi

exec uv run uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
