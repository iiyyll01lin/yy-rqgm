#!/usr/bin/env bash
# =============================================================================
# AgentForge — Smart-Manufacturing anchor demo (end-to-end, over the LIVE API).
#
# Drives the full contract with curl against a running backend and narrates a
# concrete factory visual-QC scenario:
#
#   Current hardware : Radeon RX 7900 XTX (24 GB) — a strong consumer card.
#   The ask          : run Llama 3.1 8B at 8k context for 8 concurrent lines.
#   What happens     : it OVERFLOWS 24 GB; the platform explains the gap in
#                      domain language and shows the AMD upgrade path
#                      (W7900 48 GB -> MI300X 192 GB, MI300X badged SIMULATED),
#                      then generates a runnable export (TCO + docker-compose).
#   Plus             : the LangGraph HITL orchestrate/resume flow and the
#                      GEPA epoch propose/approve (RQGM) evolution loop.
#
# Usage:
#   scripts/demo.sh                 # against http://localhost:8000
#   BASE=http://host:8000 scripts/demo.sh
#   scripts/demo.sh --no-evolve     # skip the epoch propose/approve (no state change)
#
# Requires: a running backend (see scripts/dev.sh) + curl + python3.
# The backend runs fully offline (deterministic inference mock); no GPU needed.
# =============================================================================
set -euo pipefail

BASE="${BASE:-http://localhost:8000}"
EVOLVE=1
[[ "${1:-}" == "--no-evolve" ]] && EVOLVE=0

# --- pretty helpers ----------------------------------------------------------
bold() { printf '\033[1m%s\033[0m\n' "$*"; }
rule() { printf '\033[2m%s\033[0m\n' "────────────────────────────────────────────────────────────────────"; }
step() { echo; rule; bold "▶ $*"; rule; }

# POST/GET helpers that pretty-print JSON (python3, no jq dependency).
pp() { python3 -m json.tool 2>/dev/null || cat; }
jget() { python3 -c "import sys,json;print(json.load(sys.stdin)$1)"; }

api() { # api METHOD PATH [JSON_BODY]
  local method="$1" path="$2" body="${3:-}"
  if [[ -n "$body" ]]; then
    curl -fsS -X "$method" "$BASE$path" -H 'Content-Type: application/json' -d "$body"
  else
    curl -fsS -X "$method" "$BASE$path"
  fi
}

# --- 0. health ---------------------------------------------------------------
step "0. Health check ($BASE)"
if ! curl -fsS "$BASE/health" >/tmp/af_health.json 2>/dev/null; then
  echo "!! Backend not reachable at $BASE."
  echo "   Start it first:  cd $(cd "$(dirname "$0")/.." && pwd) && uv run uvicorn backend.app.main:app"
  exit 1
fi
cat /tmp/af_health.json | pp
USING_MOCK=$(cat /tmp/af_health.json | jget "['inference']['using_mock']")
echo "   inference using_mock=$USING_MOCK  (deterministic offline judge when true)"

# --- 1. catalogs -------------------------------------------------------------
step "1. Catalogs — AMD tiers + open-weight models (GET /api/tiers, /api/models)"
api GET /api/tiers | python3 -c "import sys,json;[print(f\"   {t['id']:<18} {t['class']:<10} {t['memory_gb']:>5.0f} GB  {t['bandwidth_tbs']} TB/s\") for t in json.load(sys.stdin)['tiers']]"

# --- 2. session + domain (Step 1) -------------------------------------------
step "2. New session + domain routing (POST /api/session, /domain)"
SID=$(api POST /api/session | jget "['session_id']")
echo "   session_id=$SID"
DOMAIN_BODY='{"domain":"智慧製造 / 工廠品質檢測 (Smart Manufacturing QC)","description":"Visual defect detection / surface scratch inspection on the production line (camera vision QC); auditable decisions; data stays on-prem.","workload_type":"realtime"}'
api POST "/api/session/$SID/domain" "$DOMAIN_BODY" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print('   recommended_template:',d['recommended_template_id']);[print('   -',t['id'],'/',t['name']) for t in d['matched_templates']]"

# --- 3. diagnose the CURRENT hardware (Step 2) — the 24 GB overflow ----------
step "3. Constraint diagnostic on Radeon RX 7900 XTX 24 GB (POST /diagnose)"
DIAG_BODY='{"current_hardware":{"tier_id":"rx_7900_xtx"},"requirements":{"model_id":"llama-3.1-8b","seq_len":8192,"concurrency":8,"dtype":"fp16"}}'
api POST "/api/session/$SID/diagnose" "$DIAG_BODY" > /tmp/af_diag.json
cat /tmp/af_diag.json | python3 -c "
import sys,json
d=json.load(sys.stdin); r=d['report']; b=r['vram_breakdown']
print(f\"   feasible={d['feasible']}  VRAM_total={r['vram_total_gb']:.1f} GB  headroom={r['headroom_gb']:.1f} GB  ~{r['tokens_per_s_est']:.0f} tok/s\")
print(f\"   breakdown: weights={b['weights']:.1f}  kv={b['kv_cache']:.1f}  act={b['activations']:.1f}  overhead={b['overhead']:.1f}\")
for g in d['gaps']:
    print(f\"   GAP [{g['constraint']}] need {g['needed']} vs have {g['have']}\")
    print('        '+g['explanation_domain'])
"

# --- 4. simulate the upgrade path (Step 3) ----------------------------------
step "4. Simulation Lab — same workload across the AMD ladder (POST /simulate)"
SIM_BODY='{"model_id":"llama-3.1-8b","seq_len":8192,"population":16,"dtype":"fp16","prefix_ratio":0.5}'
api POST "/api/session/$SID/simulate" "$SIM_BODY" | python3 -c "
import sys,json
per=json.load(sys.stdin)['per_tier']
print(f\"   {'tier':<18}{'feasible':<10}{'max_pop':>9}{'tok/s':>9}\")
for t in per:
    print(f\"   {t['tier_id']:<18}{str(t['feasible']):<10}{t['max_population']:>9}{t['tokens_per_s_est']:>9.0f}\")
by={t['tier_id']:t for t in per}
if 'mi300x' in by and 'rx_7900_xtx' in by and by['rx_7900_xtx']['max_population']:
    adv=by['mi300x']['max_population']/by['rx_7900_xtx']['max_population']
    print(f\"   -> MI300X carries ~{by['mi300x']['max_population']} branches = {adv:.1f}x the RX 7900 XTX (SIMULATED).\")
"

# --- 5. evaluate an architecture (RQGM judge) -------------------------------
step "5. RQGM Evaluator — deficit-score a weak design (POST /evaluate)"
EVAL_BODY='{"architecture":"A single node applies a static temperature threshold to trigger an automatic shutdown. No state schema, no root-cause, no HITL. Numerical duct-tape.","domain":"smart_manufacturing"}'
api POST "/api/session/$SID/evaluate" "$EVAL_BODY" | python3 -c "
import sys,json;d=json.load(sys.stdin)
print(f\"   deficit_score={d['deficit_score']}  epoch={d['epoch_id']}\")
for f in d['red_flags']: print(f\"   red_flag [{f['severity']}] {f['criterion']}: {f['detail']}\")
"

# --- 6. export the deliverable (Step 4) -------------------------------------
step "6. Export — AMD TCO/ROI proposal + runnable deploy files (POST /export)"
EXPORT_BODY='{"target_tier_id":"mi300x","model_id":"llama-3.1-8b","template_id":"visual_qc","seq_len":8192,"concurrency":8,"dtype":"fp16","prefix_ratio":0.5}'
api POST "/api/session/$SID/export" "$EXPORT_BODY" > /tmp/af_export.json
cat /tmp/af_export.json | python3 -c "
import sys,json;d=json.load(sys.stdin)
print('   deploy_files:', ', '.join(d['deploy_files'].keys()))
print('   tco_markdown (first lines):')
for line in d['tco_markdown'].splitlines()[:6]: print('     '+line)
"

# --- 7. feedback (ground-truth anchor for the cold path) --------------------
step "7. Feedback — the HITL ground-truth anchor (POST /feedback)"
api POST "/api/session/$SID/feedback" '{"rating":5,"correct":true,"notes":"MI300X headroom is exactly what our multi-line population search needs."}' | pp

# --- 8. LangGraph HITL orchestrate + resume ---------------------------------
step "8. LangGraph orchestration — Gatekeeper -> Evaluator -> HITL interrupt"
ORCH_BODY='{"need":"visual defect detection","model_id":"llama-3.1-8b","tier_id":"w7900","seq_len":8192,"concurrency":4,"dtype":"int4"}'
api POST "/api/session/$SID/orchestrate" "$ORCH_BODY" | python3 -c "
import sys,json;d=json.load(sys.stdin)
print(f\"   awaiting_hitl={d['awaiting_hitl']}  next={d['next']}\")
hr=d.get('hitl_request') or {}
if isinstance(hr,dict): print(f\"   HITL asks to approve: deficit={hr.get('deficit_score')} epoch={hr.get('epoch_id')}\")
"
echo "   ...human approves..."
api POST "/api/session/$SID/orchestrate/resume" '{"approved":true,"notes":"approved for PoC"}' | python3 -c "
import sys,json;d=json.load(sys.stdin)
print(f\"   awaiting_hitl={d['awaiting_hitl']}  approved={d['state'].get('approved')}\")
"

# --- 9. GEPA epoch propose + HITL approve (RQGM evolution) -------------------
if [[ "$EVOLVE" == "1" ]]; then
  step "9. RQGM evolution — propose a challenger rubric, then HITL-approve (epoch++)"
  BEFORE=$(curl -fsS "$BASE/health" | jget "['epoch_id']")
  api POST /api/admin/epoch/propose | python3 -c "
import sys,json;d=json.load(sys.stdin)
print('   challenger:',d['challenger_id'])
print('   metrics.separation_delta:',d['metrics'].get('separation_delta'))
print('   rubric_diff (first 6 lines):')
for line in d['rubric_diff'].splitlines()[:6]: print('     '+line)
"
  api POST /api/admin/epoch/approve '{"approve":true}' | pp
  AFTER=$(curl -fsS "$BASE/health" | jget "['epoch_id']")
  echo "   epoch advanced: $BEFORE -> $AFTER  (champion frozen within an epoch; HITL gates the upgrade)"
  echo
  echo "   NOTE: this advanced the persisted champion in data/epoch_state.json."
  echo "   To reset the evolution state back to the epoch-0 seed champion, run:"
  echo "     rm -f data/epoch_state.json data/rubric_history/challenger-*.xml"
else
  step "9. RQGM evolution — skipped (--no-evolve); no persisted state changed."
fi

echo
bold "✔ Demo complete."
echo "  Open the wizard UI (npm run dev in frontend/) to see the same flow visually,"
echo "  with a Live/Mock source pill, VRAM bars and the MI300X upgrade unlock."
