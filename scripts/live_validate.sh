#!/usr/bin/env bash
# =============================================================================
# AgentForge — LIVE model validation runbook (p0-live).
#
# The whole platform runs OFFLINE on a deterministic mock; this script is the
# bridge from "plumbing works" to "evidence works" on a REAL local model served
# on an AMD GPU (vLLM-ROCm or Lemonade). It:
#
#   1. verifies the live PLUMBING offline (fake transport + cassette) — no GPU;
#   2. if a live OpenAI-compatible server is reachable, RECORDS a cassette from
#      the @live tests, then REPLAYS it with no network to prove reproducibility;
#   3. prints the live transparency report (provenance + val/test separation +
#      judge/human κ + hack-ratio) so a human can eyeball real separation quality.
#
# ---------------------------------------------------------------------------
# REQUIRES A REAL AMD GPU BOX for steps 2-3 (be honest — this environment has
# none, so only step 1 runs here). Bring up ONE of:
#
#   # vLLM on ROCm (OpenAI server):
#   docker compose -f infra/docker-compose.rocm.yml up --build
#   #   or: python -m vllm.entrypoints.openai.api_server --model <hf-model> \
#   #         --served-model-name AgentForge-Local  (env VLLM_ROCM_USE_AITER=1)
#   export LEMONADE_BASE_URL=http://localhost:8000/v1
#
#   # Lemonade (Ryzen AI NPU / Radeon):
#   lemonade-server serve           # then:
#   export LEMONADE_BASE_URL=http://localhost:8020/api/v1
#
#   export LEMONADE_MODEL=<served-model-name>      # optional (default AgentForge-Local)
#   export LEMONADE_API_KEY=<token>                # optional (vLLM --api-key)
#
# Usage:
#   scripts/live_validate.sh
#   LEMONADE_BASE_URL=http://localhost:8000/v1 scripts/live_validate.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="${LEMONADE_BASE_URL:-}"
MODEL="${LEMONADE_MODEL:-AgentForge-Local}"
CASSETTE_DIR="${LEMONADE_CASSETTE_DIR:-tests/cassettes}"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
rule() { printf '\033[2m────────────────────────────────────────────────────────────────────\033[0m\n'; }
step() { echo; rule; bold "▶ $*"; rule; }

# --- Step 1: verify the live plumbing OFFLINE (no GPU, no network) -----------
step "Step 1/3 — verify live plumbing OFFLINE (fake transport + cassette)"
echo "These exercise the structured-output contract, seed/response_format threading,"
echo "strict-vs-loose hack-ratio, record→replay reproducibility, and provenance —"
echo "all without a model. This is the part that IS validated in a GPU-less env."
uv run pytest tests/test_live_wiring.py -q

# --- probe for a live server -------------------------------------------------
probe_ok=0
if [[ -n "$BASE" ]]; then
  auth=()
  [[ -n "${LEMONADE_API_KEY:-}" ]] && auth=(-H "Authorization: Bearer ${LEMONADE_API_KEY}")
  if curl -sf -m 3 "${auth[@]}" "${BASE%/}/models" >/dev/null 2>&1; then
    probe_ok=1
  fi
fi

if [[ "$probe_ok" -ne 1 ]]; then
  step "No live server reachable — stopping after the offline checks (EXPECTED without a GPU)"
  cat <<EOF
LEMONADE_BASE_URL is ${BASE:-<unset>} and no OpenAI-compatible /models endpoint
answered. Steps 2-3 REQUIRE a real served model on an AMD GPU. Once one is up
(see the header of this script) and LEMONADE_BASE_URL is exported, re-run:

    LEMONADE_BASE_URL=http://localhost:8000/v1 scripts/live_validate.sh

What still REQUIRES the GPU box (cannot be shown offline):
  * structured JSON / guided decoding actually HONORED by the served model;
  * genuine strict/loose (hack-ratio) SEPARATION quality on real generations;
  * judge/human agreement (Cohen's κ) vs. real human labels;
  * latency / throughput numbers.
EOF
  exit 0
fi

# --- Step 2: record a cassette from the @live tests, then replay it ----------
step "Step 2/3 — RECORD live cassette, then REPLAY offline (reproducibility)"
bold "recording (@live, real model at ${BASE}, model=${MODEL})…"
RQGM_RUN_LIVE=1 \
  LEMONADE_CASSETTE_DIR="$CASSETTE_DIR" LEMONADE_CASSETTE_MODE=record \
  uv run pytest -m live -v

bold "replaying the recording with the mock forbidden path (no new network)…"
LEMONADE_CASSETTE_DIR="$CASSETTE_DIR" LEMONADE_CASSETTE_MODE=replay \
  uv run pytest -m live -v
echo "If replay passed byte-for-byte, the live code path is now reproducible in CI."

# --- Step 3: live transparency report (human eyeballs real separation) -------
step "Step 3/3 — live transparency report (provenance + separation + κ + hack-ratio)"
uv run python - <<'PY'
import json
from backend.evaluator import report
hs = report.health_summary()
print("health_summary:", json.dumps(hs, indent=2))
rep = report.build_report()
print("provenance   :", json.dumps(rep["provenance"], indent=2))
print("separation   :", json.dumps(rep["separation"], indent=2))
print("judge κ (val):", rep["judge_agreement"]["val"]["cohen_kappa"],
      "| κ (test):", rep["judge_agreement"]["test"]["cohen_kappa"])
print("hack_ratio   :", rep["hack_ratio"]["mean_hack_ratio"])
PY

step "Done."
cat <<EOF
Recorded cassette lives in ${CASSETTE_DIR}/ (commit it to make the live path a
reproducible CI check). REMEMBER: judge/human κ above is vs. the PLANTED anchor
labels, not independent human raters — real κ-vs-human still needs a human
labelling pass on live generations.
EOF
