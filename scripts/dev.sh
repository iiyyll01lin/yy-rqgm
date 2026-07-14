#!/usr/bin/env bash
# =============================================================================
# AgentForge — one-command local dev: backend (uv) + frontend (Next.js).
#
#   scripts/dev.sh
#   BACKEND_PORT=8000 FRONTEND_PORT=3000 scripts/dev.sh   # *starting* ports
#   LEMONADE_BASE_URL=http://localhost:8020/api/v1 scripts/dev.sh  # live NPU/ROCm
#
# Ports are auto-selected: the preferred port is used when free, otherwise the
# next free TCP port is picked (so a stale server still holding :3000 no longer
# blocks a one-click start). The frontend is forced to LIVE mode and pointed at
# whichever port the backend actually got, so the wizard talks to the real API.
# Once the frontend is serving, a prominent copy-paste "Open" URL is printed and
# a browser is opened when a launcher exists (best-effort — harmless over SSH).
# Ctrl-C (or the script exiting) tears down both servers cleanly.
#
# Requires: uv (https://docs.astral.sh/uv/) and Node/npm (>=18). python3 is used
# for port probing when present, with `ss` and bash /dev/tcp fallbacks.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Preferred *starting* ports; each is bumped to the next free port if busy.
BIND_HOST="0.0.0.0"
BACKEND_PORT_START="${BACKEND_PORT:-8000}"
FRONTEND_PORT_START="${FRONTEND_PORT:-3000}"

# Deterministic mock inference unless a Lemonade endpoint is provided.
if [[ -z "${LEMONADE_BASE_URL:-}" ]]; then
  export LEMONADE_FORCE_MOCK="${LEMONADE_FORCE_MOCK:-1}"
fi

have() { command -v "$1" >/dev/null 2>&1; }

# --- port helpers ------------------------------------------------------------

# port_free HOST PORT -> exit 0 if a server could bind HOST:PORT right now.
# Prefers a real python3 bind() (most accurate), then `ss`, then /dev/tcp.
port_free() {
  local host="$1" port="$2"
  if have python3; then
    python3 - "$host" "$port" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind((host, port))
except OSError:
    sys.exit(1)   # busy
finally:
    s.close()
sys.exit(0)       # free
PY
    return
  fi
  if have ss; then
    # Busy when something is LISTENing on the port (col 4 = Local Address:Port).
    if ss -ltn 2>/dev/null | awk 'NR>1 {print $4}' | grep -qE "[:.]${port}\$"; then
      return 1
    fi
    return 0
  fi
  # Last resort: a successful bash /dev/tcp connect means the port is busy.
  if (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then
    exec 3>&- 3<&-
    return 1
  fi
  return 0
}

# find_free_port HOST START -> print first free port >= START (scans up to 200).
find_free_port() {
  local host="$1" start="$2" port
  for (( port = start; port < start + 200; port++ )); do
    if port_free "$host" "$port"; then
      printf '%s\n' "$port"
      return 0
    fi
  done
  return 1
}

# tcp_open HOST PORT -> exit 0 if a TCP connect succeeds (something is serving).
tcp_open() {
  local host="$1" port="$2"
  if have python3; then
    python3 - "$host" "$port" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(1.5)
try:
    s.connect((host, port))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
    return
  fi
  if (exec 3<>"/dev/tcp/${host}/${port}") 2>/dev/null; then
    exec 3>&- 3<&-
    return 0
  fi
  return 1
}

# wait_serving URL HOST PORT PID [TRIES] -> 0 once the server responds (or the
# process dies / we run out of tries).
wait_serving() {
  local url="$1" host="$2" port="$3" pid="$4" tries="${5:-90}" i
  for (( i = 1; i <= tries; i++ )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 1   # the server process exited
    fi
    if have curl; then
      curl -fsS -o /dev/null --max-time 3 "$url" 2>/dev/null && return 0
    elif tcp_open "$host" "$port"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

# open_browser URL -> best-effort; never fails the script (may be a headless box).
open_browser() {
  local url="$1" opener
  for opener in xdg-open open; do
    if have "$opener"; then
      ( "$opener" "$url" >/dev/null 2>&1 & ) || true
      return 0
    fi
  done
  return 0
}

# --- process management ------------------------------------------------------

declare -a PGIDS=()

# spawn_pg CMD... -> launch in its own process group so the whole tree (uv ->
# uvicorn, npm -> next, ...) can be killed together on shutdown.
spawn_pg() {
  if have setsid; then
    setsid "$@" &
  else
    "$@" &
  fi
  PGIDS+=("$!")
}

# kill_tree SIG PID -> recursively signal PID and its descendants (fallback path
# used only when setsid is unavailable).
kill_tree() {
  local sig="$1" pid="$2" child
  if have pgrep; then
    while read -r child; do
      [[ -n "$child" ]] && kill_tree "$sig" "$child"
    done < <(pgrep -P "$pid" 2>/dev/null || true)
  fi
  kill "$sig" "$pid" 2>/dev/null || true
}

_cleaned=0
cleanup() {
  [[ "$_cleaned" == 1 ]] && return
  _cleaned=1
  trap - INT TERM EXIT
  echo
  echo "[dev] shutting down…"
  local target
  for target in "${PGIDS[@]:-}"; do
    [[ -n "$target" ]] || continue
    if have setsid; then
      kill -TERM "-$target" 2>/dev/null || true   # negative == process group
    else
      kill_tree -TERM "$target"
    fi
  done
  sleep 1
  for target in "${PGIDS[@]:-}"; do
    [[ -n "$target" ]] || continue
    if have setsid; then
      kill -KILL "-$target" 2>/dev/null || true
    else
      kill_tree -KILL "$target"
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# --- backend -----------------------------------------------------------------

if ! BACKEND_PORT="$(find_free_port "$BIND_HOST" "$BACKEND_PORT_START")"; then
  echo "[dev] ERROR: no free TCP port found near ${BACKEND_PORT_START} for the backend." >&2
  exit 1
fi
if [[ "$BACKEND_PORT" != "$BACKEND_PORT_START" ]]; then
  echo "[dev] port ${BACKEND_PORT_START} busy -> backend using ${BACKEND_PORT} instead."
fi
API_BASE="http://localhost:${BACKEND_PORT}"

echo "[dev] backend  -> ${API_BASE}   (uvicorn, inference mock=${LEMONADE_FORCE_MOCK:-0})"
spawn_pg uv run uvicorn backend.app.main:app --host "$BIND_HOST" --port "$BACKEND_PORT"
BACKEND_PID="${PGIDS[-1]}"

# Wait for the API so the frontend's first (live) calls succeed. Non-fatal: if it
# is slow we still bring up the UI and surface the URL.
if wait_serving "${API_BASE}/health" localhost "$BACKEND_PORT" "$BACKEND_PID" 60; then
  echo "[dev] backend ready at ${API_BASE}"
else
  echo "[dev] WARNING: backend not confirmed healthy yet; continuing to start the UI." >&2
fi

# --- frontend ----------------------------------------------------------------

if [[ ! -d frontend/node_modules ]]; then
  echo "[dev] installing frontend deps (first run)…"
  ( cd frontend && npm install )
fi

# Pick the frontend port *after* the backend is up so the two never race for the
# same number when their ranges overlap.
if ! FRONTEND_PORT="$(find_free_port "$BIND_HOST" "$FRONTEND_PORT_START")"; then
  echo "[dev] ERROR: no free TCP port found near ${FRONTEND_PORT_START} for the frontend." >&2
  exit 1
fi
if [[ "$FRONTEND_PORT" != "$FRONTEND_PORT_START" ]]; then
  echo "[dev] port ${FRONTEND_PORT_START} busy -> frontend using ${FRONTEND_PORT} instead."
fi
FRONTEND_URL="http://localhost:${FRONTEND_PORT}"

echo "[dev] frontend -> ${FRONTEND_URL}  (Next.js, live API at ${API_BASE})"
# Force LIVE mode and point the UI at the backend port we actually got. Run from
# frontend/ via `exec` so the group leader becomes npm (-> next dev) for clean
# teardown, and `next dev -p` binds the pre-selected free port.
spawn_pg bash -c '
  cd frontend
  export NEXT_PUBLIC_USE_MOCK=false
  export NEXT_PUBLIC_API_BASE="$1"
  exec npm run dev -- -p "$2"
' _dev "$API_BASE" "$FRONTEND_PORT"
FRONTEND_PID="${PGIDS[-1]}"

# --- announce ----------------------------------------------------------------

if wait_serving "$FRONTEND_URL" localhost "$FRONTEND_PORT" "$FRONTEND_PID" 120; then
  cat <<BANNER

============================
  ▶ Open:  ${FRONTEND_URL}
  (API: ${API_BASE})
============================

BANNER
  open_browser "$FRONTEND_URL"
else
  echo >&2
  echo "[dev] WARNING: frontend did not report ready in time." >&2
  echo "[dev] Try opening it manually once it finishes compiling:" >&2
  echo "        ${FRONTEND_URL}   (API: ${API_BASE})" >&2
  echo >&2
fi

echo "[dev] both running. Press Ctrl-C to stop both."
wait
