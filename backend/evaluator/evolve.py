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
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.evaluator import versioning
from backend.evaluator.judge import evaluate_architecture
from backend.inference.lemonade_client import LemonadeClient, MockMarker, get_lemonade_client
from backend.inference.parsing import extract_json
from backend.memory.qdrant_store import EvolutionaryMemory, MemoryType, get_memory

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ANCHOR_PATH = _REPO_ROOT / "data" / "anchor" / "anchor_architectures.json"
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
def load_anchors() -> list[dict[str, Any]]:
    if not _ANCHOR_PATH.exists():
        return []
    with _ANCHOR_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh).get("anchors", [])


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


def _mutate_rubric_text(champion_text: str, new_criteria: list[dict], version: str, epoch: int) -> str:
    crit_xml = ""
    for c in new_criteria:
        cid = str(c.get("id", "evolved_criterion"))
        text = str(c.get("text", "")).strip()
        crit_xml += (
            f'    <criterion id="{cid}" weight="0.10" origin="gepa" epoch_added="{epoch}">\n'
            f"      {text}\n"
            f"    </criterion>\n"
        )
    out = champion_text
    if "</rubric>" in out:
        out = out.replace("</rubric>", crit_xml + "  </rubric>", 1)
    else:
        out = out + "\n" + crit_xml
    # bump the evaluator version attribute (first occurrence)
    out = re.sub(r'version="[^"]*"', f'version="{version}"', out, count=1)
    return out


def _score_anchors(rubric_text: str, epoch: int, client: LemonadeClient) -> dict[str, float]:
    scores: dict[str, float] = {}
    for a in load_anchors():
        ev = evaluate_architecture(
            a["architecture"],
            domain_id=a.get("domain"),
            client=client,
            rubric_text=rubric_text,
            epoch=epoch,
        )
        scores[a["id"]] = ev.deficit_score
    return scores


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _separation(scores: dict[str, float], anchors: list[dict]) -> float:
    """Mean deficit(weak) - mean deficit(strong). Higher = better discrimination."""
    weak = [scores[a["id"]] for a in anchors if a.get("label") == "weak" and a["id"] in scores]
    strong = [scores[a["id"]] for a in anchors if a.get("label") == "strong" and a["id"] in scores]
    return _mean(weak) - _mean(strong)


def propose_challenger(
    feedback: list[dict[str, Any]] | None = None,
    traces: list[dict[str, Any]] | None = None,
    domain_id: str | None = "smart_manufacturing",
    client: LemonadeClient | None = None,
) -> ChallengerProposal:
    """Reflectively mutate the champion rubric into a scored challenger."""
    client = client or get_lemonade_client()
    epoch = versioning.get_epoch()
    champion_text = versioning.get_champion_rubric_text()
    champion_version = versioning.get_champion_version()

    if feedback is None:
        feedback = load_seed_feedback()
    traces = traces or []
    side_info = summarize_side_information(feedback, traces)

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
        f"Domain: {domain_id or 'general'}\nCurrent champion epoch: {epoch}\n\n"
        f"=== CURRENT CHAMPION RUBRIC ===\n{champion_text}\n\n"
        f"=== ACTIONABLE SIDE INFORMATION ===\n{side_info}\n"
    )
    raw = client.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.4,
        max_tokens=900,
    )
    parsed = extract_json(raw) or {}

    reflection = str(parsed.get("reflection", "")) or "Reflective mutation from accumulated feedback."
    proposed_changes = list(parsed.get("proposed_changes", []) or [])
    new_criteria = list(parsed.get("new_criteria", []) or [])
    if not new_criteria:
        # robust fallback so evolution always yields a concrete change
        new_criteria = [
            {
                "id": "reward_hacking_resistance",
                "text": "Heavily penalize designs that can game their own KPI (e.g. disabling sensors) or rely on numerical duct-tape instead of physical root cause.",
            }
        ]

    version = f"challenger-e{epoch}-{_short_hash(side_info + str(time.time()))}"
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

    # Champion-vs-challenger separation on the held-out anchor set.
    anchors = load_anchors()
    champion_scores = _score_anchors(champion_text, epoch, client)
    challenger_scores = _score_anchors(rubric_text, epoch, client)
    champion_sep = _separation(champion_scores, anchors)
    challenger_sep = _separation(challenger_scores, anchors)
    metrics = {
        "feedback_considered": len(feedback),
        "traces_considered": len(traces),
        "new_criteria_count": len(new_criteria),
        "champion_separation": round(champion_sep, 4),
        "challenger_separation": round(challenger_sep, 4),
        "separation_delta": round(challenger_sep - champion_sep, 4),
        "champion_anchor_scores": {k: round(v, 4) for k, v in champion_scores.items()},
        "challenger_anchor_scores": {k: round(v, 4) for k, v in challenger_scores.items()},
        "note": "separation = mean deficit(weak) - mean deficit(strong); higher is better.",
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
# APPROVE (RQGM ground-truth-gated epoch upgrade + selective erasure)
# ---------------------------------------------------------------------------
def approve_challenger(
    version: str,
    approve: bool,
    memory: EvolutionaryMemory | None = None,
) -> dict[str, Any]:
    """Human-gated promotion. On approval: advance epoch + selective erasure."""
    challenger = versioning.get_challenger(version)
    if challenger is None:
        raise KeyError(f"unknown challenger: {version}")

    if not approve:
        return {
            "epoch_id": versioning.get_epoch(),
            "applied": False,
            "champion_version": versioning.get_champion_version(),
            "erased_memories": 0,
            "reason": "challenger rejected by HITL; champion frozen.",
        }

    result = versioning.promote_challenger(version)  # {epoch_id, champion_version, prior_epoch}

    # Selective erasure: obsolete heuristic_failure memories from the prior epoch.
    mem = memory or get_memory()
    try:
        erased = mem.purge_epoch(
            up_to_epoch=result["prior_epoch"], memory_type=MemoryType.HEURISTIC_FAILURE
        )
    except Exception:
        erased = 0

    return {
        "epoch_id": result["epoch_id"],
        "applied": True,
        "champion_version": result["champion_version"],
        "prior_epoch": result["prior_epoch"],
        "erased_memories": erased,
        "note": "physics_truth memories are preserved; only obsolete heuristic_failure erased.",
    }
