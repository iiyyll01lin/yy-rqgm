"""In-process session storage for the PoC (a dict, as specified).

Also keeps process-wide accumulators of HITL feedback + evaluator traces, which
the evolution loop reads as Actionable Side Information.
"""

from __future__ import annotations

import time
import uuid
from typing import Any


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        # cross-session accumulators used by the RQGM evolution loop
        self.feedback_log: list[dict[str, Any]] = []
        self.trace_log: list[dict[str, Any]] = []

    def create(self) -> str:
        sid = uuid.uuid4().hex
        self._sessions[sid] = {
            "id": sid,
            "created_at": int(time.time()),
            "domain": None,
            "description": None,
            "workload_type": None,
            "matched_templates": [],
            "recommended_template_id": None,
            "evaluations": [],
            "feedback": [],
        }
        return sid

    def exists(self, sid: str) -> bool:
        return sid in self._sessions

    def get(self, sid: str) -> dict[str, Any] | None:
        return self._sessions.get(sid)

    def update(self, sid: str, **fields: Any) -> None:
        if sid in self._sessions:
            self._sessions[sid].update(fields)

    def record_evaluation(self, sid: str, evaluation: dict[str, Any]) -> None:
        if sid in self._sessions:
            self._sessions[sid]["evaluations"].append(evaluation)
        self.trace_log.append(evaluation)

    def record_feedback(self, sid: str, feedback: dict[str, Any]) -> None:
        if sid in self._sessions:
            self._sessions[sid]["feedback"].append(feedback)
        self.feedback_log.append(feedback)

    def all(self) -> list[dict[str, Any]]:
        return list(self._sessions.values())


_store: SessionStore | None = None


def get_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store
