"""Live model-sweep measurement harness (Part A / Part B re-measure).

Serves nothing itself — point it at an already-running OpenAI-compatible server
(``LEMONADE_BASE_URL`` + ``LEMONADE_MODEL``) and it prints the RQGM headline
metrics for the CURRENT champion under the CURRENT strict/loose mechanism:

* ``hack_ratio``          (mean quality_strict / quality_loose over weak val anchors)
* ``val`` / ``test`` separation  (mean deficit(weak) − mean deficit(strong))
* ``over_acceptance_rate``       (gamed samples the rubric waves through)
* judge κ + accuracy (val/test)  vs PLANTED anchor labels (NOT real humans)
* a representative judge latency (wall-clock seconds/call)

It is deliberately CALL-EFFICIENT: exactly ONE loose judge call per unique anchor
(val ∪ test ∪ adversarial) drives the loose metrics (separation / agreement /
over-acceptance), and the hack ratio goes through the PRODUCTION path
(``RqgmController.assess`` → ``score_candidate`` → ``rqgm`` exploitation detector)
so it tracks whatever strict/loose mechanism is in the code (old or improved).

Emits a single JSON blob under the ``SWEEP_RESULT`` sentinel.

Usage::

    LEMONADE_BASE_URL=http://127.0.0.1:8000/v1 LEMONADE_MODEL=<name> \
        uv run python scripts/live_sweep_measure.py --tag llama-3.1-8b
"""

from __future__ import annotations

import argparse
import json
import os
import time
from statistics import mean

from backend.evaluator import adversarial
from backend.evaluator import anchors as anchor_ds
from backend.evaluator import panel, rqgm_adapter, versioning
from backend.evaluator.judge import (
    SCORING_LOOSE,
    _clamp,
    build_rubric_prompt,
    judge_response_format,
)
from backend.inference.lemonade_client import LemonadeClient
from backend.inference.parsing import extract_json

_TAU = panel.DEFAULT_TAU


def _loose_deficit(client, cand, domain, epoch, rubric, seed):
    """One LOOSE judge call; returns (deficit_loose, saw_deficit_strict_in_payload)."""
    msgs = build_rubric_prompt(cand, domain, epoch, rubric, scoring_mode=SCORING_LOOSE)
    rf = judge_response_format() if not client.using_mock else None
    raw = client.chat(msgs, temperature=0.1, max_tokens=900, seed=seed, response_format=rf)
    parsed = extract_json(raw) or {}
    val = 0.5
    for k in ("deficit_loose", "deficit_score"):
        if k in parsed:
            try:
                val = _clamp(float(parsed[k]))
                break
            except (TypeError, ValueError):
                continue
    return val, ("deficit_strict" in parsed)


def _sep(deficits_by_id, anchors):
    weak = [deficits_by_id[a["id"]] for a in anchor_ds.weak(anchors)]
    strong = [deficits_by_id[a["id"]] for a in anchor_ds.strong(anchors)]
    sep = (mean(weak) if weak else 0.0) - (mean(strong) if strong else 0.0)
    return round(sep, 4), round(mean(weak), 4) if weak else 0.0, round(mean(strong), 4) if strong else 0.0


def _agreement(deficits_by_id, anchors):
    pred = ["weak" if deficits_by_id[a["id"]] >= _TAU else "strong" for a in anchors]
    truth = [a.get("label", "weak") for a in anchors]
    n = len(anchors)
    acc = sum(p == t for p, t in zip(pred, truth)) / n if n else 0.0
    return round(acc, 4), round(panel.cohen_kappa(pred, truth), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=os.getenv("LEMONADE_MODEL", "unknown"))
    ap.add_argument("--latency-reps", type=int, default=3)
    args = ap.parse_args()

    client = LemonadeClient(force_mock=False)
    epoch = versioning.get_epoch()
    champion = versioning.get_champion_rubric_text()

    t_start = time.time()
    val = anchor_ds.load_anchors(anchor_ds.VAL)
    test = anchor_ds.load_anchors(anchor_ds.TEST)
    adv = adversarial.generate_adversarial_samples(champion, include_out_of_catalog=True)

    loose: dict[str, float] = {}
    saw_strict_in_loose = 0
    seed = 4000

    # latency probe (a real weak val anchor, loose mode).
    lat: list[float] = []
    probe = val[0] if val else test[0]
    probe_cand = anchor_ds.anchor_candidate_text(probe)
    for i in range(max(1, args.latency_reps)):
        t0 = time.time()
        _loose_deficit(client, probe_cand, probe.get("domain"), epoch, champion, seed + i)
        lat.append(time.time() - t0)
    mean_latency = sum(lat) / len(lat)
    seed += 100

    # one loose call per unique anchor (drives separation / agreement / over-acceptance).
    for a in list(val) + list(test) + list(adv):
        cand = anchor_ds.anchor_candidate_text(a)
        d, had_strict = _loose_deficit(client, cand, a.get("domain"), epoch, champion, seed)
        loose[a["id"]] = d
        saw_strict_in_loose += int(had_strict)
        seed += 1

    # hack ratio via the PRODUCTION path (score_candidate strict/loose on weak val
    # anchors + rqgm detector) so it reflects the mechanism actually shipped.
    exploit = rqgm_adapter.get_controller().assess(
        champion, anchor_ds.load_anchors(anchor_ds.VAL), epoch=epoch, client=client, persist=False
    )

    val_sep, val_weak, val_strong = _sep(loose, val)
    test_sep, _, _ = _sep(loose, test)
    val_acc, val_kappa = _agreement(loose, val)
    test_acc, test_kappa = _agreement(loose, test)

    adv_deficits = [loose[a["id"]] for a in adv]
    over_acc = sum(1 for d in adv_deficits if d < _TAU) / len(adv_deficits) if adv_deficits else 0.0

    result = {
        "tag": args.tag,
        "judge_model": client.model,
        "using_mock": client.using_mock,
        "rqgm_backend": rqgm_adapter.RQGM_BACKEND,
        "hack_ratio": exploit.mean_hack_ratio,
        "exploitation_detected": exploit.exploitation_detected,
        "n_hack_samples": exploit.n_samples,
        "val_separation": val_sep,
        "val_mean_weak_deficit": val_weak,
        "val_mean_strong_deficit": val_strong,
        "test_separation": test_sep,
        "over_acceptance_rate": round(over_acc, 4),
        "over_acceptance_n": len(adv_deficits),
        "val_kappa": val_kappa,
        "val_accuracy": val_acc,
        "test_kappa": test_kappa,
        "test_accuracy": test_acc,
        "saw_deficit_strict_in_loose_payload": saw_strict_in_loose,
        "mean_latency_s": round(mean_latency, 3),
        "latency_reps": args.latency_reps,
        "wall_clock_s": round(time.time() - t_start, 1),
    }
    print("SWEEP_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
