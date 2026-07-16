"""Judge-vs-gold agreement: does the small local judge agree with a STRONGER
"gold" model used as a human-proxy labeler?

RQGM's calibrated anchor metric is judge/human agreement (accuracy + Cohen's κ),
but our κ is currently vs the PLANTED anchor labels, not independent raters. This
tool adds a second, harder reference: have BOTH the small judge (default the local
8B) and a stronger GOLD model score the SAME held-out candidates under the current
champion rubric, then report judge-vs-gold accuracy + Cohen's κ next to the
existing judge-vs-planted κ.

HONEST CAVEAT (surfaced in every output): the gold model is a STRONGER-MODEL
PROXY for a human labeler, NOT a real human. A true κ-vs-human still needs a human
labelling pass — this only removes the "grades its own planted labels" circularity
by introducing an independent, stronger scorer.

Configurable, OpenAI-compatible endpoints (so a cloud gold model can be swapped in
when a key exists; otherwise the local 32B is the default gold):

* judge — ``LEMONADE_BASE_URL`` / ``LEMONADE_MODEL`` (+ ``LEMONADE_API_KEY``)
* gold  — ``GOLD_BASE_URL``     / ``GOLD_MODEL``     (+ ``GOLD_API_KEY``);
          falls back to the ``LEMONADE_*`` judge endpoint if unset (single-server /
          offline-mock wiring test only — a real run points these at a stronger model).

Because both models are served on the SINGLE gfx1151 GPU they cannot run at once,
so scoring is a TWO-PASS, cache-backed flow (serialize serving: judge pass with the
8B up, then the gold pass with the 32B up):

    # 1) judge pass (8B served on :8000)
    LEMONADE_BASE_URL=http://127.0.0.1:8000/v1 LEMONADE_MODEL=llama-3.1-8b \
        uv run python scripts/gold_agreement.py --role judge
    # 2) gold pass (tear down 8B, serve 32B on :8000)
    GOLD_BASE_URL=http://127.0.0.1:8000/v1 GOLD_MODEL=qwen2.5-32b \
        uv run python scripts/gold_agreement.py --role gold        # auto-reports when both present
    # (or force a recompute from the cache)
    uv run python scripts/gold_agreement.py --report-only

Emits the judge-vs-gold result to ``data/metrics/gold_agreement.json`` (read
additively by report.build_report) and prints a ``GOLD_AGREEMENT`` JSON sentinel.
Runs fully offline against the deterministic mock too (for the wiring test /
cassette), where both roles trivially agree.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from backend.evaluator import anchors as anchor_ds
from backend.evaluator import adversarial, panel, versioning
from backend.evaluator.judge import SCORING_LOOSE, _clamp, build_rubric_prompt, judge_response_format
from backend.evaluator.panel import cohen_kappa
from backend.inference.lemonade_client import LemonadeClient
from backend.inference.parsing import extract_json

_REPO_ROOT = Path(__file__).resolve().parents[1]
_METRICS_DIR = _REPO_ROOT / "data" / "metrics"
DEFAULT_CACHE = _METRICS_DIR / "gold_agreement_cache.json"
DEFAULT_RESULT = _METRICS_DIR / "gold_agreement.json"
CAVEAT = "gold model is a STRONGER-MODEL proxy, NOT a human labeler; true κ-vs-human still needs a human pass."


# ---------------------------------------------------------------------------
# Held-out set: TEST anchors (reporting-only split) + a few adversarial samples
# ---------------------------------------------------------------------------
def build_heldout(limit: int) -> list[dict[str, Any]]:
    """Small held-out set: the reporting ``test`` split + a few gamed samples.

    Deterministic. Adversarial samples carry the ground-truth label ``weak`` (they
    are gamed poison-pill designs), so planted-label agreement is well-defined.
    """
    champ = versioning.get_champion_rubric_text()
    items: list[dict[str, Any]] = list(anchor_ds.load_anchors(anchor_ds.TEST))
    adv = adversarial.generate_adversarial_samples(champ, include_out_of_catalog=True)
    # A representative handful of gamed designs (kept small — the 32B gold is slow):
    want = ("adv_kpi_sensor_gaming_0", "adv_concept_drift_blind_0",
            "adv_ooc_audit_log_spoofing", "adv_ooc_prompt_injection_actuation")
    picked = [s for s in adv if s["id"] in want]
    items.extend(picked)
    return items[:limit]


def _client_for(role: str) -> LemonadeClient:
    if role == "gold":
        base = os.getenv("GOLD_BASE_URL") or os.getenv("LEMONADE_BASE_URL")
        model = os.getenv("GOLD_MODEL") or os.getenv("LEMONADE_MODEL")
        api_key = os.getenv("GOLD_API_KEY") or os.getenv("LEMONADE_API_KEY")
        return LemonadeClient(base_url=base, model=model, api_key=api_key)
    return LemonadeClient()  # judge: LEMONADE_* env


def _loose_reading(client: LemonadeClient, cand: str, domain: str | None, epoch: int,
                   rubric: str, seed: int) -> dict[str, Any]:
    """One LOOSE judge call. Returns the holistic scalar AND a penalty-aware reading.

    Some judges (observed on Qwen2.5-14B under a rich rubric) zero the holistic
    ``deficit_loose`` scalar while still correctly filling ``criterion_penalties`` /
    high-severity red flags. The per-criterion signal is the reliable one (see
    judge.py), so the weak/strong label uses ``max(scalar, max criterion_penalty)``.
    """
    msgs = build_rubric_prompt(cand, domain, epoch, rubric, scoring_mode=SCORING_LOOSE)
    rf = judge_response_format() if not client.using_mock else None
    raw = client.chat(msgs, temperature=0.1, max_tokens=900, seed=seed, response_format=rf)
    parsed = extract_json(raw) or {}
    scalar = 0.5
    for k in ("deficit_loose", "deficit_score"):
        if k in parsed:
            try:
                scalar = _clamp(float(parsed[k]))
                break
            except (TypeError, ValueError):
                continue
    max_pen = 0.0
    for v in (parsed.get("criterion_penalties") or {}).values():
        try:
            max_pen = max(max_pen, _clamp(float(v)))
        except (TypeError, ValueError):
            pass
    deficit = max(scalar, max_pen)
    return {"scalar": round(scalar, 4), "max_penalty": round(max_pen, 4), "deficit": round(deficit, 4)}


def _load_cache(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def score_role(role: str, *, limit: int, tau: float, cache_path: Path, base_seed: int = 7000) -> dict[str, Any]:
    """Score the held-out set with ``role``'s endpoint; merge into the cache."""
    epoch = versioning.get_epoch()
    champ = versioning.get_champion_rubric_text()
    champ_ver = versioning.get_champion_version()
    client = _client_for(role)
    items = build_heldout(limit)

    cache = _load_cache(cache_path)
    if cache.get("champion_version") != champ_ver or float(cache.get("tau", tau)) != tau:
        cache = {}  # rubric/tau changed -> stale, start fresh
    cache.setdefault("tau", tau)
    cache.setdefault("champion_version", champ_ver)
    cache.setdefault("items", {})

    for i, item in enumerate(items):
        cand = anchor_ds.anchor_candidate_text(item)
        reading = _loose_reading(client, cand, item.get("domain"), epoch, champ, base_seed + i)
        row = cache["items"].setdefault(item["id"], {"label": item.get("label", "weak"), "domain": item.get("domain")})
        row[role] = {
            "model": client.model,
            "using_mock": client.using_mock,
            "predicted": "weak" if reading["deficit"] >= tau else "strong",
            **reading,
        }
    _save_json(cache_path, cache)
    return cache


def compute_agreement(cache_path: Path, result_path: Path, *, tau: float) -> dict[str, Any] | None:
    """Compute judge-vs-gold (and vs planted) accuracy + κ from a complete cache."""
    cache = _load_cache(cache_path)
    items = cache.get("items", {})
    ready = {k: v for k, v in items.items() if "judge" in v and "gold" in v}
    if not ready:
        return None

    ids = sorted(ready)
    judge_pred = [ready[i]["judge"]["predicted"] for i in ids]
    gold_pred = [ready[i]["gold"]["predicted"] for i in ids]
    planted = [ready[i].get("label", "weak") for i in ids]

    def _acc(a: list[str], b: list[str]) -> float:
        return round(sum(x == y for x, y in zip(a, b)) / len(a), 4) if a else 0.0

    def _confusion(a: list[str], b: list[str]) -> dict[str, int]:
        # rows: judge, cols: gold
        c = {"weak_weak": 0, "weak_strong": 0, "strong_weak": 0, "strong_strong": 0}
        for x, y in zip(a, b):
            c[f"{x}_{y}"] += 1
        return c

    judge_model = ready[ids[0]]["judge"]["model"]
    gold_model = ready[ids[0]]["gold"]["model"]
    result = {
        "generated_at": int(time.time()),
        "n": len(ids),
        "tau": tau,
        "champion_version": cache.get("champion_version"),
        "judge_model": judge_model,
        "gold_model": gold_model,
        "judge_using_mock": ready[ids[0]]["judge"]["using_mock"],
        "gold_using_mock": ready[ids[0]]["gold"]["using_mock"],
        "judge_vs_gold": {
            "accuracy": _acc(judge_pred, gold_pred),
            "cohen_kappa": round(cohen_kappa(judge_pred, gold_pred), 4),
            "confusion": _confusion(judge_pred, gold_pred),
            "n": len(ids),
        },
        "judge_vs_planted": {
            "accuracy": _acc(judge_pred, planted),
            "cohen_kappa": round(cohen_kappa(judge_pred, planted), 4),
        },
        "gold_vs_planted": {
            "accuracy": _acc(gold_pred, planted),
            "cohen_kappa": round(cohen_kappa(gold_pred, planted), 4),
        },
        "caveat": CAVEAT,
        "per_item": [
            {
                "id": i,
                "label": ready[i].get("label"),
                "judge": ready[i]["judge"]["predicted"],
                "gold": ready[i]["gold"]["predicted"],
                "judge_deficit": ready[i]["judge"]["deficit"],
                "gold_deficit": ready[i]["gold"]["deficit"],
            }
            for i in ids
        ],
    }
    _save_json(result_path, result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role", choices=["judge", "gold"], help="score the held-out set with this role's endpoint")
    ap.add_argument("--report-only", action="store_true", help="only (re)compute agreement from the cache")
    ap.add_argument("--limit", type=int, default=20, help="max held-out items (default 20; keep small — 32B gold is slow)")
    ap.add_argument("--tau", type=float, default=panel.DEFAULT_TAU)
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--result", default=str(DEFAULT_RESULT))
    args = ap.parse_args()

    cache_path, result_path = Path(args.cache), Path(args.result)

    if args.role:
        cache = score_role(args.role, limit=args.limit, tau=args.tau, cache_path=cache_path)
        done = sum(1 for v in cache.get("items", {}).values() if "judge" in v and "gold" in v)
        print(f"[gold_agreement] scored role={args.role}; items with BOTH roles: {done}/{len(cache.get('items', {}))}")

    result = compute_agreement(cache_path, result_path, tau=args.tau)
    if result is None:
        print("[gold_agreement] cache incomplete (need BOTH judge and gold passes); run the other --role.")
        return
    jvg, jvp, gvp = result["judge_vs_gold"], result["judge_vs_planted"], result["gold_vs_planted"]
    print(f"[gold_agreement] judge={result['judge_model']}  gold={result['gold_model']}  n={result['n']}  tau={result['tau']}")
    print(f"  judge-vs-GOLD    : accuracy={jvg['accuracy']}  cohen_kappa={jvg['cohen_kappa']}  confusion={jvg['confusion']}")
    print(f"  judge-vs-planted : accuracy={jvp['accuracy']}  cohen_kappa={jvp['cohen_kappa']}")
    print(f"  gold-vs-planted  : accuracy={gvp['accuracy']}  cohen_kappa={gvp['cohen_kappa']}")
    print(f"  CAVEAT: {CAVEAT}")
    print("GOLD_AGREEMENT " + json.dumps({k: result[k] for k in ("n", "tau", "judge_model", "gold_model",
                                                                  "judge_vs_gold", "judge_vs_planted", "gold_vs_planted")}))


if __name__ == "__main__":
    main()
