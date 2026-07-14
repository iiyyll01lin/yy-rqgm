"""Epoch + rubric versioning for the RQGM evaluator.

The evaluator is the FUZZY, evolving half of the platform. To make evolution
*controlled* (RQGM), we track:

    * an integer **epoch** counter (frozen within an epoch),
    * the **champion** rubric (what the judge uses right now),
    * a set of **challenger** rubrics produced by GEPA-style mutation.

``backend/evaluator/rubric.xml`` is the immutable epoch-0 seed champion. Promoted
challengers are archived under ``data/rubric_history/`` and the active champion
is tracked in ``data/epoch_state.json`` — so runtime evolution never mutates
source files, and the whole history is auditable.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_EVAL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EVAL_DIR.parents[1]
_DATA_DIR = _REPO_ROOT / "data"
_HISTORY_DIR = _DATA_DIR / "rubric_history"
STATE_PATH = _DATA_DIR / "epoch_state.json"
SEED_RUBRIC_PATH = _EVAL_DIR / "rubric.xml"


def _default_state() -> dict[str, Any]:
    return {
        "epoch_id": 0,
        "champion_version": "champion-0",
        "champion_path": str(SEED_RUBRIC_PATH),
        "challengers": {},  # version -> {path, metrics, parent, created_at}
        "history": [
            {"epoch_id": 0, "champion_version": "champion-0", "at": _now()}
        ],
    }


def _now() -> int:
    return int(time.time())


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


def get_epoch() -> int:
    return int(load_state().get("epoch_id", 0))


def get_champion_version() -> str:
    return str(load_state().get("champion_version", "champion-0"))


def get_champion_rubric_text() -> str:
    """Text of the currently-active champion rubric (seed at epoch 0)."""
    state = load_state()
    path = Path(state.get("champion_path", str(SEED_RUBRIC_PATH)))
    if not path.exists():
        path = SEED_RUBRIC_PATH
    return path.read_text(encoding="utf-8")


def register_challenger(
    version: str,
    rubric_text: str,
    metrics: dict[str, Any] | None = None,
    parent_version: str | None = None,
) -> dict[str, Any]:
    """Persist a challenger rubric under data/rubric_history and record it."""
    _ensure_dirs()
    path = _HISTORY_DIR / f"{version}.xml"
    path.write_text(rubric_text, encoding="utf-8")
    state = load_state()
    entry = {
        "path": str(path),
        "metrics": metrics or {},
        "parent": parent_version or state.get("champion_version"),
        "created_at": _now(),
        "created_epoch": state.get("epoch_id", 0),
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


def get_challenger_rubric_text(version: str) -> str | None:
    entry = get_challenger(version)
    if not entry:
        return None
    path = Path(entry["path"])
    return path.read_text(encoding="utf-8") if path.exists() else None


def list_challengers() -> list[dict[str, Any]]:
    state = load_state()
    return [{"version": v, **e} for v, e in state.get("challengers", {}).items()]


def promote_challenger(version: str) -> dict[str, Any]:
    """Promote a challenger to champion and advance the epoch (RQGM upgrade).

    Returns ``{"epoch_id", "champion_version", "prior_epoch"}``. Raises KeyError
    if the challenger is unknown.
    """
    state = load_state()
    challengers = state.get("challengers", {})
    if version not in challengers:
        raise KeyError(f"unknown challenger: {version}")
    prior_epoch = int(state.get("epoch_id", 0))
    new_epoch = prior_epoch + 1
    state["epoch_id"] = new_epoch
    state["champion_version"] = version
    state["champion_path"] = challengers[version]["path"]
    challengers[version]["promoted"] = True
    state.setdefault("history", []).append(
        {"epoch_id": new_epoch, "champion_version": version, "at": _now()}
    )
    save_state(state)
    return {
        "epoch_id": new_epoch,
        "champion_version": version,
        "prior_epoch": prior_epoch,
    }


def reset() -> None:
    """Reset epoch state (test helper). Removes the state file."""
    if STATE_PATH.exists():
        STATE_PATH.unlink()
