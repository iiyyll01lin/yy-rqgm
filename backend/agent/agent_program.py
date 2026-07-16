"""The agent "gene": a serializable task-agent PROGRAM (EvoSkill representation).

Mirrors the way the evaluator half represents its evolving unit as a rubric
(``backend/evaluator/rubric.xml``). The agent's evolving unit is a *program*:

    * ``system_prompt``     — the task-agent persona / guidance (was hard-coded in
      ``backend/graph/nodes/task_agent.py``; extracted here so it can evolve);
    * ``skills``            — a list of capability tags the agent is instructed to
      apply. Each skill maps (offline, deterministically) to a failure mode it
      covers, so ADDING a genuinely-useful skill produces an architecture the
      frozen champion evaluator scores higher (see
      ``backend/inference/mock_scoring.py::AGENT_SKILL_COVERAGE``);
    * ``domain_overrides``  — per-domain extra guidance appended for a given domain.

The seed champion program lives in ``backend/agent/agent_program_seed.json`` (the
immutable epoch-0 seed, analogous to ``rubric.xml``). Runtime evolution never
mutates that file — promoted challengers are archived under ``data/agent_history``
and the active champion is tracked in ``data/agent_state.json`` (see
:mod:`backend.agent.agent_versioning`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_AGENT_DIR = Path(__file__).resolve().parent
SEED_PROGRAM_PATH = _AGENT_DIR / "agent_program_seed.json"

# Machine-readable sentinels the offline mock reads from a generation prompt (the
# LIVE model ignores them). Kept here so the generator and the mock agree.
SKILL_SENTINEL_PREFIX = "[[agent_skills:"
NEED_FLAWS_SENTINEL_PREFIX = "[[need_flaws:"


@dataclass
class AgentProgram:
    """A task-agent program (the unit the agent half evolves)."""

    system_prompt: str
    skills: list[str] = field(default_factory=list)
    domain_overrides: dict[str, str] = field(default_factory=dict)

    # ---- (de)serialize --------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "skills": list(self.skills),
            "domain_overrides": dict(self.domain_overrides),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentProgram":
        return cls(
            system_prompt=str(d.get("system_prompt", "")),
            # normalise: de-dup while preserving order, drop blanks.
            skills=_dedup([str(s).strip() for s in (d.get("skills") or []) if str(s).strip()]),
            domain_overrides={str(k): str(v) for k, v in (d.get("domain_overrides") or {}).items()},
        )

    def genome_text(self) -> str:
        """Canonical, stable serialization (used for hashing + frontier genome).

        Skills are sorted so a re-ordering is NOT treated as a different program
        (a no-op change must not spuriously win the agent gate).
        """
        canonical = {
            "system_prompt": self.system_prompt.strip(),
            "skills": sorted(set(self.skills)),
            "domain_overrides": {k: self.domain_overrides[k] for k in sorted(self.domain_overrides)},
        }
        return json.dumps(canonical, ensure_ascii=False, sort_keys=True)

    def program_id(self) -> str:
        """Short content hash of the canonical genome (reproducible id)."""
        return hashlib.sha256(self.genome_text().encode("utf-8")).hexdigest()[:8]

    # ---- rendering ------------------------------------------------------
    def render_system_prompt(self, domain_id: str | None = None) -> str:
        """The persona/guidance block for the task-agent (skills spelled out)."""
        parts = [self.system_prompt.strip()]
        if self.skills:
            parts.append(
                "Apply these engineering skills explicitly in the design: "
                + ", ".join(sorted(set(self.skills)))
                + "."
            )
        if domain_id and domain_id in self.domain_overrides:
            parts.append(self.domain_overrides[domain_id].strip())
        return "\n".join(p for p in parts if p)

    def skill_sentinel(self) -> str:
        """``[[agent_skills:...]]`` sentinel for the offline deterministic mock."""
        return f"{SKILL_SENTINEL_PREFIX}{','.join(sorted(set(self.skills)))}]]"

    # ---- mutation helpers (used by agent_evolve) ------------------------
    def with_skill(self, skill: str) -> "AgentProgram":
        """Return a copy with ``skill`` added (no-op if already present)."""
        if skill in self.skills:
            return AgentProgram(self.system_prompt, list(self.skills), dict(self.domain_overrides))
        return AgentProgram(self.system_prompt, [*self.skills, skill], dict(self.domain_overrides))

    def with_system_prompt(self, prompt: str) -> "AgentProgram":
        return AgentProgram(prompt, list(self.skills), dict(self.domain_overrides))


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def load_seed_program() -> AgentProgram:
    """Load the immutable epoch-0 seed champion program."""
    with SEED_PROGRAM_PATH.open("r", encoding="utf-8") as fh:
        return AgentProgram.from_dict(json.load(fh))
