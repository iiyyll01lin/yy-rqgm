"""Minimal RQGM / GEPA-style self-evolution loop for the evaluator.

Cold path (async, off the hot request path):

    1. PROPOSE  — read accumulated HITL feedback + evaluator traces as *Actionable
       Side Information* and reflectively MUTATE the champion rubric into a
       *challenger* (GEPA: traces are a textual gradient). Score both champion
       and challenger on a held-out ground-truth anchor set (Pareto-ish check).
    2. APPROVE  — a human gates the epoch upgrade (RQGM ground-truth gating; this
       is what prevents reward hacking / evaluator drift). On approval the
       challenger becomes champion, the epoch counter advances, and we trigger
       SELECTIVE ERASURE of the prior epoch's obsolete negative-result memories
       (heuristic_failure), while physics_truth memories are preserved forever.

Everything runs deterministically offline via the inference mock.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import random
import re

from backend.evaluator import adversarial as adversarial_mod
from backend.evaluator import anchors as anchor_ds
from backend.evaluator import frontier as frontier_mod
from backend.evaluator import gate as gate_mod
from backend.evaluator import rqgm_adapter, versioning
from backend.evaluator.frontier import FrontierMember, ParetoFrontier
from backend.evaluator.judge import evaluate_architecture, score_candidate
from backend.evaluator.mutation import mutate_rubric_text
from backend.inference.lemonade_client import LemonadeClient, MockMarker, get_lemonade_client
from backend.inference.parsing import extract_json
from backend.memory.qdrant_store import EvolutionaryMemory, MemoryType, get_memory

_GEPA_CRITERION_ID_RE = re.compile(r'<criterion\b[^>]*?\bid="([^"]+)"[^>]*?origin="gepa"')

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEED_FEEDBACK_PATH = _REPO_ROOT / "data" / "anchor" / "seed_feedback.jsonl"


@dataclass
class ChallengerProposal:
    version: str
    parent_version: str
    epoch_id: int
    reflection: str
    proposed_changes: list[str] = field(default_factory=list)
    new_criteria: list[dict[str, Any]] = field(default_factory=list)
    rubric_text: str = ""
    rubric_diff: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    used_mock: bool = False

    def to_dict(self) -> dict:
        return {
            "challenger_id": self.version,
            "parent_version": self.parent_version,
            "epoch_id": self.epoch_id,
            "reflection": self.reflection,
            "proposed_changes": self.proposed_changes,
            "new_criteria": self.new_criteria,
            "rubric_diff": self.rubric_diff,
            "metrics": self.metrics,
        }


# ---------------------------------------------------------------------------
# Inputs (Actionable Side Information)
# ---------------------------------------------------------------------------
def load_anchors(split: str | None = None) -> list[dict[str, Any]]:
    """Load anchors (optionally a single split). Delegates to :mod:`anchors`."""
    return anchor_ds.load_anchors(split)


def load_seed_feedback() -> list[dict[str, Any]]:
    if not _SEED_FEEDBACK_PATH.exists():
        return []
    out: list[dict[str, Any]] = []
    with _SEED_FEEDBACK_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    return out


def summarize_side_information(
    feedback: list[dict[str, Any]], traces: list[dict[str, Any]]
) -> str:
    lines: list[str] = []
    for f in feedback:
        lines.append(
            f"- HITL feedback: rating={f.get('rating')} correct={f.get('correct')} "
            f"notes={f.get('notes', '')}"
        )
    for t in traces:
        flags = [rf.get("criterion") for rf in t.get("red_flags", [])]
        lines.append(
            f"- Evaluator trace: deficit={t.get('deficit_score')} red_flags={flags}"
        )
    return "\n".join(lines) if lines else "- (no accumulated feedback yet)"


# ---------------------------------------------------------------------------
# PROPOSE
# ---------------------------------------------------------------------------
def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def _challenger_hash(
    side_info: str, new_criteria: list[dict[str, Any]], seed: int | str | None = None
) -> str:
    """Reproducible challenger id suffix.

    Derived from the CONTENT that determines the challenger (its side-information
    textual gradient + the sorted ids of the criteria it adds) plus an optional
    injected ``seed``. This replaces the previous ``time.time()`` hash, which made
    the version id — and therefore the archived rubric filename — nondeterministic
    across otherwise-identical runs (bad for reproducibility / auditing).
    """
    ids = ",".join(sorted(str(c.get("id", "")) for c in new_criteria))
    material = f"{side_info}\n{ids}"
    if seed is not None:
        material += f"\n{seed}"
    return _short_hash(material)


def _mutate_rubric_text(champion_text: str, new_criteria: list[dict], version: str, epoch: int) -> str:
    """Append de-duplicated criteria + bump ONLY the <evaluator> version, validate XML.

    Thin wrapper over :func:`backend.evaluator.mutation.mutate_rubric_text` (the
    version regex used to corrupt the ``<?xml version="1.0">`` prolog; it now
    targets the evaluator tag only, dedups criteria by id, and validates).
    """
    return mutate_rubric_text(champion_text, new_criteria, version, epoch)


def _score_anchors(
    rubric_text: str,
    epoch: int,
    client: LemonadeClient,
    anchors: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    anchors = anchors if anchors is not None else load_anchors(anchor_ds.TRAIN)
    scores: dict[str, float] = {}
    for a in anchors:
        res = score_candidate(
            anchor_ds.anchor_candidate_text(a),
            rubric_text,
            domain_id=a.get("domain"),
            epoch=epoch,
            client=client,
        )
        scores[a["id"]] = res["deficit_loose"]
    return scores


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _separation(scores: dict[str, float], anchors: list[dict]) -> float:
    """Mean deficit(weak) - mean deficit(strong). Higher = better discrimination."""
    weak = [scores[a["id"]] for a in anchors if a.get("label") == "weak" and a["id"] in scores]
    strong = [scores[a["id"]] for a in anchors if a.get("label") == "strong" and a["id"] in scores]
    return _mean(weak) - _mean(strong)


_FALLBACK_CRITERION = {
    "id": "reward_hacking_resistance",
    "text": (
        "Heavily penalize designs that can game their own KPI (e.g. disabling sensors) or rely "
        "on numerical duct-tape instead of physical root cause."
    ),
}


def reflect_and_mutate(
    parent_text: str,
    side_info: str,
    epoch: int,
    client: LemonadeClient,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    """GEPA reflective step: read side-information, propose targeted new criteria.

    Returns ``(reflection, proposed_changes, new_criteria)``. Shared by the
    single-lineage :func:`propose_challenger` and the Pareto ``gepa_evolve`` loop
    so both use the identical mutation primitive.
    """
    system = (
        f"{MockMarker.MUTATE.value}\n"
        "You are a GEPA-style reflective optimizer improving an RQGM evaluator rubric.\n"
        "Read the Actionable Side Information (HITL feedback + evaluator traces) as a\n"
        "textual gradient. Propose targeted rubric improvements that would have caught the\n"
        "documented failures WITHOUT overfitting or collapsing diversity. Respond with STRICT\n"
        'JSON: {"reflection": str, "proposed_changes": [str], '
        '"new_criteria": [{"id": str, "text": str}]}.'
    )
    user = (
        f"Current champion epoch: {epoch}\n\n"
        f"=== CURRENT PARENT RUBRIC ===\n{parent_text}\n\n"
        f"=== ACTIONABLE SIDE INFORMATION ===\n{side_info}\n"
    )
    parsed = extract_json(
        client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.4,
            max_tokens=900,
        )
    ) or {}
    reflection = str(parsed.get("reflection", "")) or "Reflective mutation from accumulated feedback."
    proposed_changes = list(parsed.get("proposed_changes", []) or [])
    new_criteria = list(parsed.get("new_criteria", []) or [])
    if not new_criteria:
        new_criteria = [dict(_FALLBACK_CRITERION)]
    return reflection, proposed_changes, new_criteria


def propose_challenger(
    feedback: list[dict[str, Any]] | None = None,
    traces: list[dict[str, Any]] | None = None,
    domain_id: str | None = "smart_manufacturing",
    client: LemonadeClient | None = None,
    *,
    seed: int | str | None = None,
) -> ChallengerProposal:
    """Reflectively mutate the champion rubric into a scored challenger.

    The challenger's version id is now reproducible: it is derived from the
    side-information + added-criteria content (and an optional injected ``seed``),
    NOT from wall-clock time. Identical inputs therefore yield an identical
    version id and archived rubric filename across runs.
    """
    client = client or get_lemonade_client()
    epoch = versioning.get_epoch()
    champion_text = versioning.get_champion_rubric_text()
    champion_version = versioning.get_champion_version()

    if feedback is None:
        feedback = load_seed_feedback()
    traces = traces or []
    side_info = summarize_side_information(feedback, traces)

    reflection, proposed_changes, new_criteria = reflect_and_mutate(champion_text, side_info, epoch, client)

    version = f"challenger-e{epoch}-{_challenger_hash(side_info, new_criteria, seed)}"
    rubric_text = _mutate_rubric_text(champion_text, new_criteria, version, epoch)
    rubric_diff = "".join(
        difflib.unified_diff(
            champion_text.splitlines(keepends=True),
            rubric_text.splitlines(keepends=True),
            fromfile=f"{champion_version}.xml",
            tofile=f"{version}.xml",
            n=2,
        )
    )

    # Champion-vs-challenger separation on the proposer's TRAIN split only (data
    # isolation: the gate independently re-checks on the held-out VAL split, and
    # TEST is reserved for reporting — so the proposer can never peek at the gate).
    train_anchors = load_anchors(anchor_ds.TRAIN)
    champion_scores = _score_anchors(champion_text, epoch, client, train_anchors)
    challenger_scores = _score_anchors(rubric_text, epoch, client, train_anchors)
    champion_sep = _separation(champion_scores, train_anchors)
    challenger_sep = _separation(challenger_scores, train_anchors)
    metrics = {
        "split": "train",
        "feedback_considered": len(feedback),
        "traces_considered": len(traces),
        "new_criteria_count": len(new_criteria),
        "champion_separation": round(champion_sep, 4),
        "challenger_separation": round(challenger_sep, 4),
        "separation_delta": round(challenger_sep - champion_sep, 4),
        "champion_anchor_scores": {k: round(v, 4) for k, v in champion_scores.items()},
        "challenger_anchor_scores": {k: round(v, 4) for k, v in challenger_scores.items()},
        "note": (
            "separation = mean deficit(weak) - mean deficit(strong) on TRAIN; higher is better. "
            "The code gate re-evaluates on the held-out VAL split (see evaluate_promotion)."
        ),
    }

    versioning.register_challenger(
        version=version, rubric_text=rubric_text, metrics=metrics, parent_version=champion_version
    )

    return ChallengerProposal(
        version=version,
        parent_version=champion_version,
        epoch_id=epoch,
        reflection=reflection,
        proposed_changes=proposed_changes,
        new_criteria=new_criteria,
        rubric_text=rubric_text,
        rubric_diff=rubric_diff,
        metrics=metrics,
        used_mock=client.using_mock,
    )


# ---------------------------------------------------------------------------
# PROPOSE via Pareto frontier (GEPA population search)
# ---------------------------------------------------------------------------
def generate_adversarial_pool(
    domain_id: str | None = "smart_manufacturing",
) -> list[dict[str, Any]]:
    """Self-play red-team samples targeting the CURRENT champion's blind spots."""
    champion_text = versioning.get_champion_rubric_text()
    return adversarial_mod.generate_adversarial_samples(champion_text, domain_id)


def _gepa_added_criteria(rubric_text: str) -> list[str]:
    return _GEPA_CRITERION_ID_RE.findall(rubric_text)


def _select_failure_trace(
    parent_text: str,
    train_weak: list[dict[str, Any]],
    epoch: int,
    client: LemonadeClient,
) -> tuple[str, dict[str, Any] | None]:
    """Pick the train weak anchor the parent most under-penalises (largest
    strict−loose gap = biggest uncaught poison-pill blind spot) and turn its
    planted flaws into GEPA side-information."""
    best_gap = 0.0
    worst: dict[str, Any] | None = None
    for a in train_weak:
        res = score_candidate(
            anchor_ds.anchor_candidate_text(a), parent_text,
            domain_id=a.get("domain"), epoch=epoch, client=client,
        )
        gap = res["deficit_strict"] - res["deficit_loose"]
        if gap > best_gap:
            best_gap = gap
            worst = a
    if worst is None:
        return "- (parent already covers the train failure modes; sharpen an existing criterion)", None
    flaws = ", ".join(worst.get("flaws", []))
    side = (
        f"- Evaluator trace: the champion UNDER-penalised weak anchor '{worst['id']}' "
        f"(strict-vs-loose gap {best_gap:.2f}). Missed failure mode(s): {flaws}. "
        f"Architecture: {worst.get('architecture', '')[:200]}"
    )
    return side, worst


def gepa_evolve(
    incumbent_text: str,
    *,
    epoch: int,
    domain_id: str | None = "smart_manufacturing",
    budget: int = 6,
    top_k: int = 8,
    client: LemonadeClient | None = None,
    adversarial_samples: list[dict[str, Any]] | None = None,
    seed: int = 1234,
) -> ParetoFrontier:
    """GEPA budget loop over a Pareto frontier (docs/03-evaluator.md §5 skeleton).

    From the frontier, stochastically sample a parent, pick a TRAIN trace the
    parent misjudges, reflect→mutate, score the child on the ``dev`` SELECTION
    split (multi-objective + BBε), and keep it only if non-dominated. Selection
    is on ``dev`` so ``val`` stays untouched for the gate (no winner's curse):
    ``val`` is scored ONLY by :func:`gate.evaluate_promotion`, and ``test`` is
    reporting-only. Only ``train`` and the (self-play) adversarial pool ever
    *drive* mutation.
    """
    client = client or get_lemonade_client()
    dev_anchors = load_anchors(anchor_ds.DEV)
    train_weak = anchor_ds.weak(load_anchors(anchor_ds.TRAIN))
    rng = random.Random(seed)

    frontier = ParetoFrontier(top_k=top_k)
    inc_obj, inc_bbe, inc_def = frontier_mod.compute_objectives(
        incumbent_text, dev_anchors, domain_id=domain_id, epoch=epoch,
        client=client, added_criteria=[], adversarial_samples=adversarial_samples,
    )
    frontier.add(
        FrontierMember(
            version=versioning.get_champion_version(),
            rubric_text=incumbent_text,
            objectives=inc_obj, bbe=inc_bbe, added_criteria=[],
            parent_version="", sel_deficits=inc_def,
        )
    )

    for i in range(budget):
        parent = frontier.sample_stochastic(rng)
        side_info, _trace = _select_failure_trace(parent.rubric_text, train_weak, epoch, client)
        _refl, _changes, new_criteria = reflect_and_mutate(parent.rubric_text, side_info, epoch, client)
        child_version = f"challenger-e{epoch}-{_short_hash(parent.version + side_info + str(i))}"
        try:
            child_text = mutate_rubric_text(parent.rubric_text, new_criteria, child_version, epoch)
        except Exception:
            frontier.record_child_outcome(parent, improved=False)  # malformed => failure
            continue  # reject malformed mutations (Phase 0 XML validation)
        added = _gepa_added_criteria(child_text)
        obj, bbe, defs = frontier_mod.compute_objectives(
            child_text, dev_anchors, domain_id=domain_id, epoch=epoch,
            client=client, added_criteria=added, adversarial_samples=adversarial_samples,
        )
        kept = frontier.add(
            FrontierMember(
                version=child_version, rubric_text=child_text, objectives=obj, bbe=bbe,
                added_criteria=added, parent_version=parent.version, sel_deficits=defs,
            )
        )
        # Thompson-sampling feedback: a "success" for the parent is a child that
        # survived the Pareto filter AND improved on the parent's BBε lower bound
        # (the same separation signal the gate scores) — no gate call needed.
        frontier.record_child_outcome(parent, improved=bool(kept and bbe > parent.bbe))
    return frontier


def propose_via_frontier(
    feedback: list[dict[str, Any]] | None = None,
    traces: list[dict[str, Any]] | None = None,
    domain_id: str | None = "smart_manufacturing",
    budget: int = 6,
    client: LemonadeClient | None = None,
    adversarial_samples: list[dict[str, Any]] | None = None,
    *,
    seed: int = 1234,
) -> tuple[ChallengerProposal, ParetoFrontier]:
    """Run ``gepa_evolve`` and register the frontier's best member as the
    challenger handed to the code gate. Persists the whole frontier for repro.

    ``seed`` controls the frontier's stochastic parent selection (threaded to
    ``gepa_evolve``) and the single-mutation fallback's reproducible id."""
    client = client or get_lemonade_client()
    epoch = versioning.get_epoch()
    champion_text = versioning.get_champion_rubric_text()
    champion_version = versioning.get_champion_version()

    frontier = gepa_evolve(
        champion_text, epoch=epoch, domain_id=domain_id, budget=budget,
        client=client, adversarial_samples=adversarial_samples, seed=seed,
    )
    best = frontier.best()
    # If the frontier never beat the incumbent, fall back to a single mutation so
    # the endpoint always yields a concrete challenger for the gate to judge.
    if best is None or best.version == champion_version:
        proposal = propose_challenger(
            feedback=feedback, traces=traces, domain_id=domain_id, client=client, seed=seed
        )
        return proposal, frontier

    rubric_diff = "".join(
        difflib.unified_diff(
            champion_text.splitlines(keepends=True),
            best.rubric_text.splitlines(keepends=True),
            fromfile=f"{champion_version}.xml", tofile=f"{best.version}.xml", n=2,
        )
    )
    # Report the proposal's separation on the SELECTION (dev) split. The held-out
    # VAL split is reserved for gate.evaluate_promotion (no selection leak), so we
    # never compute the challenger's val separation here.
    dev_anchors = load_anchors(anchor_ds.DEV)
    champion_sep = _separation(_score_anchors(champion_text, epoch, client, dev_anchors), dev_anchors)
    challenger_sep = _separation(best.sel_deficits, dev_anchors)
    metrics = {
        "split": "dev",
        "frontier_size": len(frontier.members),
        "added_criteria": best.added_criteria,
        "champion_separation": round(champion_sep, 4),
        "challenger_separation": round(challenger_sep, 4),
        "separation_delta": round(challenger_sep - champion_sep, 4),
        "bbe_lower_bound": round(best.bbe, 4),
        "objectives": {k: round(v, 4) for k, v in best.objectives.items()},
        "note": "best of the Pareto frontier on the dev selection split by BBε lower bound; the code gate re-checks on the held-out val split.",
    }
    versioning.register_challenger(
        version=best.version, rubric_text=best.rubric_text, metrics=metrics, parent_version=champion_version
    )
    try:
        frontier_mod.persist_frontier(frontier, epoch)
    except Exception:
        pass

    proposal = ChallengerProposal(
        version=best.version,
        parent_version=champion_version,
        epoch_id=epoch,
        reflection="GEPA Pareto frontier search: promoted the non-dominated best by dev-split BBε (val reserved for the gate).",
        proposed_changes=[f"Added criteria: {', '.join(best.added_criteria) or '(none)'}"],
        new_criteria=[{"id": c, "text": ""} for c in best.added_criteria],
        rubric_text=best.rubric_text,
        rubric_diff=rubric_diff,
        metrics=metrics,
        used_mock=client.using_mock,
    )
    return proposal, frontier


# ---------------------------------------------------------------------------
# CODE GATE (RQGM statistical promotion test — runs BEFORE any HITL)
# ---------------------------------------------------------------------------
def evaluate_promotion(
    version: str,
    *,
    domain_id: str | None = "smart_manufacturing",
    client: LemonadeClient | None = None,
    gate_config: gate_mod.GateConfig | None = None,
) -> dict[str, Any]:
    """Run the CODE gate for ``version`` against the current champion.

    Two independent signals, both on held-out anchors the proposer never saw:

    * **Separation gate** (``gate.evaluate_promotion``, held-out VAL split): P1
      non-inferiority (tie favours incumbent) + P2 Bayesian Beta-Binomial
      posterior ``P(Δsep>0) >= threshold`` AND effect ``Δsep >= MDE``. VAL is
      scored only here (the frontier selects on the disjoint ``dev`` split).
    * **RQGM hack-ratio** (``rqgm_adapter``): does the *champion* have a
      poison-pill blind spot (loose passes what strict fails)? If so, tolerances
      tighten and adversarial injection is flagged for the next round. We also
      report whether the *challenger* closes that blind spot.

    Returns a JSON-safe dict. ``passed`` reflects the separation gate only — the
    hack-ratio is advisory metadata that drives strictness, not a veto here.
    """
    challenger = versioning.get_challenger(version)
    if challenger is None:
        raise KeyError(f"unknown challenger: {version}")
    client = client or get_lemonade_client()
    epoch = versioning.get_epoch()
    champion_text = versioning.get_champion_rubric_text()
    challenger_text = versioning.get_challenger_rubric_text(version) or champion_text
    val_anchors = load_anchors(anchor_ds.VAL)

    gate_result = gate_mod.evaluate_promotion(
        champion_text,
        challenger_text,
        domain_id=domain_id,
        epoch=epoch,
        val_anchors=val_anchors,
        config=gate_config,
        client=client,
    )

    controller = rqgm_adapter.get_controller()
    # Champion exploitation drives tolerance tightening (persisted); the
    # challenger's is advisory (does it close the blind spot?). Both are measured
    # over the weak/gamed anchors (the population that can expose a blind spot).
    gamed = anchor_ds.weak(val_anchors) or val_anchors
    champion_exploit = controller.assess(champion_text, val_anchors, epoch=epoch, client=client, persist=True)
    challenger_exploit = rqgm_adapter.detect_exploitation(
        *controller.qualities_for_rubric(challenger_text, gamed, epoch=epoch, client=client),
        controller.current_tolerances(),
        threshold=controller.threshold,
    )

    return {
        "version": version,
        "epoch_id": epoch,
        "gate": gate_result.to_dict(),
        "passed": gate_result.passed,
        "champion_exploitation": champion_exploit.to_dict(),
        "challenger_exploitation": challenger_exploit.to_dict(),
    }


# ---------------------------------------------------------------------------
# APPROVE (two-stage: CODE gate first, then HITL as a final safety veto)
# ---------------------------------------------------------------------------
def approve_challenger(
    version: str,
    approve: bool,
    memory: EvolutionaryMemory | None = None,
    *,
    domain_id: str | None = "smart_manufacturing",
    client: LemonadeClient | None = None,
    gate_config: gate_mod.GateConfig | None = None,
) -> dict[str, Any]:
    """Two-stage promotion.

    Stage 1 — CODE gate (:func:`evaluate_promotion`). If it FAILS, the challenger
    is rejected regardless of ``approve`` — HITL cannot override a failed gate.

    Stage 2 — HITL is consulted ONLY after the code gate passes, and acts purely
    as a final safety veto: ``approve=False`` rejects an otherwise-passing
    challenger; ``approve=True`` promotes it (epoch++), then runs selective
    erasure (soft-delete + reconfirm).
    """
    promo = evaluate_promotion(version, domain_id=domain_id, client=client, gate_config=gate_config)
    gate_dict = promo["gate"]

    if not promo["passed"]:
        # CODE gate failed: HITL is NOT consulted; the anti-reward-hacking core.
        return {
            "epoch_id": versioning.get_epoch(),
            "applied": False,
            "champion_version": versioning.get_champion_version(),
            "gate": gate_dict,
            "champion_exploitation": promo["champion_exploitation"],
            "challenger_exploitation": promo["challenger_exploitation"],
            "hitl": {"consulted": False, "approved": None, "vetoed": False},
            "erased_memories": 0,
            "reconfirmed_memories": 0,
            "reason": "CODE GATE FAILED; HITL not consulted (cannot override a failed gate).",
        }

    if not approve:
        # Passed the code gate but the human exercised the final safety veto.
        return {
            "epoch_id": versioning.get_epoch(),
            "applied": False,
            "champion_version": versioning.get_champion_version(),
            "gate": gate_dict,
            "champion_exploitation": promo["champion_exploitation"],
            "challenger_exploitation": promo["challenger_exploitation"],
            "hitl": {"consulted": True, "approved": False, "vetoed": True},
            "erased_memories": 0,
            "reconfirmed_memories": 0,
            "reason": "challenger PASSED the code gate but was vetoed by HITL; champion frozen.",
        }

    result = versioning.promote_challenger(version)  # {epoch_id, champion_version, prior_epoch}

    # P3: when the cross-epoch anytime-valid correction is active, spend one look of
    # the family-wise budget now that a promotion is actually APPLIED (so the
    # e-process wealth accumulates per promotion, not per evaluation).
    if gate_dict.get("sequential_correction"):
        gate_mod.commit_sequential_look(
            gate_dict.get("sequential_e_value", 1.0),
            epoch=result["epoch_id"],
            version=version,
        )

    # Selective erasure: soft-delete + reconfirm against the NEW champion.
    mem = memory or get_memory()
    new_champion_text = versioning.get_champion_rubric_text()
    erasure = selective_erasure(
        mem,
        up_to_epoch=result["prior_epoch"],
        new_champion_text=new_champion_text,
        domain_id=domain_id,
        epoch=result["epoch_id"],
        client=client,
    )

    # P1-ledger: append the new champion's held-out metrics to the cross-epoch
    # time-series (best-effort; must never break a promotion). This is the data
    # the regression guard (report.regression_violations) audits.
    try:
        from backend.evaluator import report as _report

        _report.record_metrics_snapshot(client=client)
    except Exception:
        pass

    return {
        "epoch_id": result["epoch_id"],
        "applied": True,
        "champion_version": result["champion_version"],
        "prior_epoch": result["prior_epoch"],
        "gate": gate_dict,
        "champion_exploitation": promo["champion_exploitation"],
        "challenger_exploitation": promo["challenger_exploitation"],
        "hitl": {"consulted": True, "approved": True, "vetoed": False},
        "erased_memories": erasure["soft_deleted"],
        "reconfirmed_memories": erasure["reconfirmed"],
        "note": "physics_truth memories are preserved forever; obsolete heuristic_failure soft-deleted.",
    }


# ---------------------------------------------------------------------------
# SELECTIVE ERASURE (soft-delete + reconfirm; physics_truth preserved forever)
# ---------------------------------------------------------------------------
def _new_champion_reconfirms(
    hit: Any,
    new_champion_text: str,
    rubric_ids: set[str],
    *,
    domain_id: str | None,
    epoch: int,
    client: LemonadeClient | None,
) -> bool:
    """Does the NEW champion still consider this heuristic_failure a failure?

    Preference order:
      1. ``reconfirm_flaws`` payload tag → reconfirmed iff the new rubric contains
         a criterion that catches at least one of those flaws (deterministic).
      2. Otherwise re-judge the memory text with the new champion; reconfirmed iff
         it still scores a non-trivial deficit.
    """
    from backend.inference.mock_scoring import FLAW_CATALOG

    payload = getattr(hit, "payload", {}) or {}
    flaws = payload.get("reconfirm_flaws") or []
    if flaws:
        return any(
            any(cid in rubric_ids for cid in FLAW_CATALOG.get(f, ((), 0.0, False))[0])
            for f in flaws
        )
    text = payload.get("text", "") or getattr(hit, "text", "")
    if not text:
        return True  # nothing to judge; conservatively keep
    ev = evaluate_architecture(
        text, domain_id=domain_id, client=client, rubric_text=new_champion_text, epoch=epoch
    )
    return ev.deficit_score >= 0.5


def selective_erasure(
    memory: EvolutionaryMemory,
    up_to_epoch: int,
    new_champion_text: str,
    *,
    domain_id: str | None = None,
    epoch: int = 0,
    client: LemonadeClient | None = None,
) -> dict[str, Any]:
    """RQGM selective erasure via SOFT-DELETE + reconfirm (never hard purge here).

    For each ``heuristic_failure`` memory created at/below ``up_to_epoch`` that
    ``depends_on_evaluator_judgement``, re-check it against the new champion. Ones
    the new champion no longer endorses are **soft-deleted** (``active=False``,
    reversible/auditable); still-valid ones are stamped ``reconfirmed_epoch``.
    ``physics_truth`` is never scanned, so physical facts are preserved forever.
    Physical hard purge of long-stale soft-deleted rows is deferred to a separate
    janitor (:func:`purge_soft_deleted`).
    """
    from backend.inference.mock_scoring import rubric_criteria_ids

    rubric_ids = rubric_criteria_ids(new_champion_text)
    try:
        candidates = memory.fetch(
            memory_type=MemoryType.HEURISTIC_FAILURE, max_epoch=up_to_epoch, active_only=True
        )
    except Exception:
        candidates = []

    to_soft_delete: list[str] = []
    reconfirmed_ids: list[str] = []
    for hit in candidates:
        payload = getattr(hit, "payload", {}) or {}
        if not payload.get("depends_on_evaluator_judgement", True):
            reconfirmed_ids.append(hit.id)  # a fact, not a taste call; keep
            continue
        if _new_champion_reconfirms(
            hit, new_champion_text, rubric_ids, domain_id=domain_id, epoch=epoch, client=client
        ):
            reconfirmed_ids.append(hit.id)
        else:
            to_soft_delete.append(hit.id)

    soft_deleted = 0
    try:
        soft_deleted = memory.soft_delete(to_soft_delete)
        memory.mark_reconfirmed(reconfirmed_ids, epoch)
    except Exception:
        pass

    return {
        "soft_deleted": soft_deleted,
        "reconfirmed": len(reconfirmed_ids),
        "candidates": len(candidates),
        "physics_truth_preserved": True,
    }


def purge_soft_deleted(
    memory: EvolutionaryMemory,
    up_to_epoch: int,
) -> int:
    """Deferred janitor: physically reclaim long-stale soft-deleted heuristics.

    Decoupled from the epoch transition so the promotion moment stays cheap and
    the audit trail persists until a scheduled job runs. ``physics_truth`` is
    never purged.
    """
    try:
        return memory.purge_epoch(up_to_epoch=up_to_epoch, memory_type=MemoryType.HEURISTIC_FAILURE)
    except Exception:
        return 0
