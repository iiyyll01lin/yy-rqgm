"""Epoch + program versioning for the RQGM AGENT half.

A direct mirror of :mod:`backend.evaluator.versioning` (the evaluator half). The
agent is the SECOND evolving half of the platform; to keep its evolution
*controlled* (RQGM) we track:

    * an integer **agent epoch** counter (frozen within an agent epoch),
    * the **champion** program (what the hot task-agent path uses right now),
    * a set of **challenger** programs produced by GEPA-style mutation.

``backend/agent/agent_program_seed.json`` is the immutable epoch-0 seed champion.
Promoted challengers are archived under ``data/agent_history/`` and the active
champion is tracked in ``data/agent_state.json`` — so runtime evolution never
mutates source files and the whole history is auditable.

RQGM ASYMMETRY: the agent is scored/gated by the epoch-FROZEN champion EVALUATOR
(``backend/evaluator/versioning.get_champion_rubric_text`` →
``judge.score_candidate``), never by the held-out anchors. Each challenger/champion
entry therefore stamps the ``evaluator_epoch`` under which its utility was
measured, so an evaluator promotion can trigger a selective RE-SCORE (utility
erasure) of the whole archive (see :mod:`backend.agent.coevolve`).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from backend.agent.agent_program import AgentProgram, load_seed_program

_AGENT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _AGENT_DIR.parents[1]
_DATA_DIR = _REPO_ROOT / "data"
_HISTORY_DIR = _DATA_DIR / "agent_history"
STATE_PATH = _DATA_DIR / "agent_state.json"

SEED_VERSION = "agent-champion-0"


def _now() -> int:
    return int(time.time())


def _default_state() -> dict[str, Any]:
    seed = load_seed_program()
    return {
        "agent_epoch_id": 0,
        "champion_version": SEED_VERSION,
        "champion_program": seed.to_dict(),
        # utility of the champion program + the evaluator epoch it was scored under
        # (None until first measured; drives selective-erasure staleness detection).
        "champion_utility": None,
        "champion_evaluator_epoch": None,
        "challengers": {},  # version -> {path, program, metrics, parent, created_at, ...}
        "history": [
            {"agent_epoch_id": 0, "champion_version": SEED_VERSION, "at": _now()}
        ],
    }


def _ensure_dirs() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return _default_state()
    try:
        with STATE_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return _default_state()


def save_state(state: dict[str, Any]) -> None:
    _ensure_dirs()
    with STATE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def get_agent_epoch() -> int:
    return int(load_state().get("agent_epoch_id", 0))


def get_champion_version() -> str:
    return str(load_state().get("champion_version", SEED_VERSION))


def get_champion_program() -> AgentProgram:
    """The currently-active champion program (seed at agent epoch 0)."""
    state = load_state()
    prog = state.get("champion_program")
    if not prog:
        return load_seed_program()
    try:
        return AgentProgram.from_dict(prog)
    except Exception:
        return load_seed_program()


def get_champion_utility() -> tuple[float | None, int | None]:
    """``(utility, evaluator_epoch)`` the champion was last scored under."""
    state = load_state()
    util = state.get("champion_utility")
    ev_epoch = state.get("champion_evaluator_epoch")
    return (float(util) if util is not None else None,
            int(ev_epoch) if ev_epoch is not None else None)


def record_champion_utility(utility: float, evaluator_epoch: int) -> None:
    """Persist the champion's measured utility + the evaluator epoch it used."""
    state = load_state()
    state["champion_utility"] = round(float(utility), 6)
    state["champion_evaluator_epoch"] = int(evaluator_epoch)
    save_state(state)


def register_challenger(
    version: str,
    program: AgentProgram,
    metrics: dict[str, Any] | None = None,
    parent_version: str | None = None,
    *,
    evaluator_epoch: int | None = None,
) -> dict[str, Any]:
    """Persist a challenger program under data/agent_history and record it."""
    _ensure_dirs()
    path = _HISTORY_DIR / f"{version}.json"
    path.write_text(json.dumps(program.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    state = load_state()
    entry = {
        "path": str(path),
        "program": program.to_dict(),
        "metrics": metrics or {},
        "parent": parent_version or state.get("champion_version"),
        "created_at": _now(),
        "created_agent_epoch": state.get("agent_epoch_id", 0),
        "evaluator_epoch": evaluator_epoch,
        "promoted": False,
    }
    state.setdefault("challengers", {})[version] = entry
    save_state(state)
    return {"version": version, **entry}


def get_challenger(version: str) -> dict[str, Any] | None:
    entry = load_state().get("challengers", {}).get(version)
    if entry is None:
        return None
    return {"version": version, **entry}


def get_challenger_program(version: str) -> AgentProgram | None:
    entry = get_challenger(version)
    if not entry:
        return None
    prog = entry.get("program")
    if prog:
        return AgentProgram.from_dict(prog)
    path = Path(entry["path"])
    if path.exists():
        return AgentProgram.from_dict(json.loads(path.read_text(encoding="utf-8")))
    return None


def list_challengers() -> list[dict[str, Any]]:
    state = load_state()
    return [{"version": v, **e} for v, e in state.get("challengers", {}).items()]


def update_challenger_utility(version: str, utility: float, evaluator_epoch: int) -> None:
    """Re-stamp a challenger's utility + evaluator epoch (selective erasure)."""
    state = load_state()
    entry = state.get("challengers", {}).get(version)
    if entry is None:
        return
    entry.setdefault("metrics", {})["utility"] = round(float(utility), 6)
    entry["evaluator_epoch"] = int(evaluator_epoch)
    save_state(state)


def promote_champion_program(version: str) -> dict[str, Any]:
    """Promote a challenger to champion and advance the AGENT epoch.

    Returns ``{"agent_epoch_id", "champion_version", "prior_epoch"}``. Raises
    KeyError if the challenger is unknown.
    """
    state = load_state()
    challengers = state.get("challengers", {})
    if version not in challengers:
        raise KeyError(f"unknown agent challenger: {version}")
    prior_epoch = int(state.get("agent_epoch_id", 0))
    new_epoch = prior_epoch + 1
    entry = challengers[version]
    state["agent_epoch_id"] = new_epoch
    state["champion_version"] = version
    state["champion_program"] = entry["program"]
    metrics = entry.get("metrics", {}) or {}
    state["champion_utility"] = metrics.get("utility")
    state["champion_evaluator_epoch"] = entry.get("evaluator_epoch")
    entry["promoted"] = True
    state.setdefault("history", []).append(
        {"agent_epoch_id": new_epoch, "champion_version": version, "at": _now()}
    )
    save_state(state)
    return {
        "agent_epoch_id": new_epoch,
        "champion_version": version,
        "prior_epoch": prior_epoch,
    }


def reset() -> None:
    """Reset agent epoch state (test helper). Removes the state file."""
    if STATE_PATH.exists():
        STATE_PATH.unlink()


def reset_to_champion0(*, purge_history: bool = True) -> dict[str, Any]:
    """Reset runtime agent-evolution state back to a clean seed baseline.

    Wipes the runtime ``data/agent_state.json`` and (optionally) the archived
    challenger programs so a fresh gated loop regenerates honest state from the
    immutable ``agent_program_seed.json``. The seed is never touched.
    """
    removed_history: list[str] = []
    if STATE_PATH.exists():
        STATE_PATH.unlink()
    if purge_history and _HISTORY_DIR.exists():
        for path in _HISTORY_DIR.glob("*.json"):
            removed_history.append(path.name)
            path.unlink()
    return {
        "champion_version": SEED_VERSION,
        "agent_epoch_id": 0,
        "removed_state_file": True,
        "removed_history": removed_history,
    }
