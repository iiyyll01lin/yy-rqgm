"""Concrete-architecture archive (the DGM half of the agent evolution).

Where :mod:`backend.agent.agent_evolve` evolves the agent PROGRAM (EvoSkill: the
reusable persona/skills), this module keeps, per NEED, an open-ended ARCHIVE of
concrete architectures (Darwin-Gödel-Machine style): variants that mitigate
different subsets of the need's latent failure modes, each scored by the FROZEN
champion evaluator. Two couplings back into the platform:

* **strong → anchor candidate** — an architecture the champion scores as strong on
  BOTH the loose read and the strict reward-hacking audit is a genuine positive;
  it is emitted in the anchor schema so it can seed a new held-out *strong* anchor
  for the evaluator (broadening ground truth).
* **gamed → adversarial pool** — an architecture that looks strong under the
  champion's LOOSE read but hides a poison pill the champion MISSES (high strict
  deficit / low hack ratio) is a self-play attack: it is emitted as an adversarial
  sample that drives the evaluator's frontier to close that blind spot
  (:func:`backend.evaluator.evolve.propose_via_frontier`'s ``adversarial_samples``).

The archive is deterministic + offline; runtime persistence lives under
``data/agent_archive/`` (gitignored).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.agent.agent_program import AgentProgram
from backend.evaluator import versioning
from backend.evaluator.judge import score_candidate
from backend.inference.lemonade_client import LemonadeClient, get_lemonade_client
from backend.inference.mock_scoring import (
    FLAW_TO_AGENT_SKILL,
    agent_architecture_text,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARCHIVE_DIR = _REPO_ROOT / "data" / "agent_archive"

# An architecture is "strong" when the champion scores it below this deficit on
# BOTH loose and strict; "gamed" when it is strong under loose but its hack ratio
# (quality_strict / quality_loose) drops below the hack threshold (a hidden pill).
DEFAULT_STRONG_TAU = 0.2
DEFAULT_HACK_THRESHOLD = 0.85


@dataclass
class ArchCandidate:
    need_id: str
    domain: str | None
    skills: list[str]
    architecture: str
    mitigations: list[str]
    residual_flaws: list[str]
    deficit_loose: float
    deficit_strict: float
    hack_ratio: float | None
    utility: float

    def public_dict(self) -> dict[str, Any]:
        return {
            "need_id": self.need_id,
            "domain": self.domain,
            "skills": list(self.skills),
            "mitigations": list(self.mitigations),
            "residual_flaws": list(self.residual_flaws),
            "deficit_loose": round(self.deficit_loose, 4),
            "deficit_strict": round(self.deficit_strict, 4),
            "hack_ratio": round(self.hack_ratio, 4) if self.hack_ratio is not None else None,
            "utility": round(self.utility, 4),
        }


@dataclass
class NeedArchive:
    need_id: str
    domain: str | None
    latent_flaws: list[str]
    population: list[ArchCandidate] = field(default_factory=list)

    def best(self) -> ArchCandidate | None:
        """The architecture the champion scores best on the loose read (hot pick)."""
        return min(self.population, key=lambda a: a.deficit_loose) if self.population else None

    def strong(self, tau: float = DEFAULT_STRONG_TAU) -> list[ArchCandidate]:
        return [a for a in self.population if a.deficit_loose < tau and a.deficit_strict < tau]

    def gamed(
        self, *, tau: float = DEFAULT_STRONG_TAU, hack_threshold: float = DEFAULT_HACK_THRESHOLD
    ) -> list[ArchCandidate]:
        return [
            a
            for a in self.population
            if a.deficit_loose < tau
            and a.hack_ratio is not None
            and a.hack_ratio < hack_threshold
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "need_id": self.need_id,
            "domain": self.domain,
            "latent_flaws": list(self.latent_flaws),
            "size": len(self.population),
            "best_deficit_loose": round(self.best().deficit_loose, 4) if self.best() else None,
            "population": [a.public_dict() for a in self.population],
        }


def _score_arch(
    skills: list[str],
    need: dict[str, Any],
    *,
    rubric_text: str,
    evaluator_epoch: int,
    client: LemonadeClient,
) -> ArchCandidate:
    latent = list(need.get("latent_flaws", []) or [])
    architecture = agent_architecture_text(skills, latent)
    scored = score_candidate(
        architecture, rubric_text, domain_id=need.get("domain"),
        epoch=evaluator_epoch, client=client,
    )
    # residual flaws are read back off the generated sentinel line.
    from backend.inference.mock_scoring import parse_sentinel

    residual, _strengths, _had = parse_sentinel(architecture)
    mitigations = [f for f in latent if f not in residual]
    return ArchCandidate(
        need_id=str(need.get("id", "")),
        domain=need.get("domain"),
        skills=list(skills),
        architecture=architecture,
        mitigations=mitigations,
        residual_flaws=residual,
        deficit_loose=float(scored["deficit_loose"]),
        deficit_strict=float(scored["deficit_strict"]),
        hack_ratio=scored.get("hack_ratio"),
        utility=1.0 - float(scored["deficit_loose"]),
    )


def build_need_archive(
    need: dict[str, Any],
    base_program: AgentProgram,
    *,
    rubric_text: str | None = None,
    evaluator_epoch: int | None = None,
    client: LemonadeClient | None = None,
) -> NeedArchive:
    """Open-ended DGM population for one need, scored by the FROZEN champion evaluator.

    Explores: the base program's architecture; the base + each single missing skill;
    and the fully-covered variant. Deduplicated by the resulting architecture text.
    Covering a champion BLIND-SPOT flaw (kpi/drift) does not change the loose
    deficit — so the base variant on such a need stays a GAMED architecture (low
    loose, hidden poison), which is exactly what feeds the evaluator adversarial pool.
    """
    client = client or get_lemonade_client()
    if rubric_text is None:
        rubric_text = versioning.get_champion_rubric_text()  # FROZEN champion evaluator
    if evaluator_epoch is None:
        evaluator_epoch = versioning.get_epoch()

    latent = list(need.get("latent_flaws", []) or [])
    base_skills = list(base_program.skills)
    missing_skills = [
        FLAW_TO_AGENT_SKILL[f]
        for f in latent
        if f in FLAW_TO_AGENT_SKILL and FLAW_TO_AGENT_SKILL[f] not in base_skills
    ]

    skill_sets: list[list[str]] = [list(base_skills)]
    for sk in missing_skills:
        skill_sets.append([*base_skills, sk])
    if missing_skills:
        skill_sets.append([*base_skills, *missing_skills])  # fully covered

    archive = NeedArchive(need_id=str(need.get("id", "")), domain=need.get("domain"), latent_flaws=latent)
    seen: set[str] = set()
    for skills in skill_sets:
        cand = _score_arch(
            skills, need, rubric_text=rubric_text, evaluator_epoch=evaluator_epoch, client=client
        )
        if cand.architecture in seen:
            continue
        seen.add(cand.architecture)
        archive.population.append(cand)
    return archive


def build_archive(
    needs: list[dict[str, Any]],
    base_program: AgentProgram,
    *,
    rubric_text: str | None = None,
    evaluator_epoch: int | None = None,
    client: LemonadeClient | None = None,
) -> list[NeedArchive]:
    """Build a per-need archive over ``needs`` (one :class:`NeedArchive` each)."""
    return [
        build_need_archive(
            n, base_program, rubric_text=rubric_text,
            evaluator_epoch=evaluator_epoch, client=client,
        )
        for n in needs
    ]


def anchor_candidates(
    archives: list[NeedArchive], *, tau: float = DEFAULT_STRONG_TAU, split: str = "test"
) -> list[dict[str, Any]]:
    """Strong architectures re-emitted in the ANCHOR schema (evaluator anchor seeds).

    ``split`` defaults to ``test`` so an agent-mined anchor is reporting-only until a
    human promotes it — it never silently leaks into the gate.
    """
    out: list[dict[str, Any]] = []
    for arch in archives:
        for a in arch.strong(tau):
            out.append(
                {
                    "id": f"agentmined_strong_{a.need_id}",
                    "label": "strong",
                    "split": split,
                    "domain": a.domain,
                    "architecture": a.architecture,
                    "flaws": [],
                    "strengths": [],
                    "provenance": "agent_concrete_archive",
                }
            )
    return out


def adversarial_samples(
    archives: list[NeedArchive],
    *,
    tau: float = DEFAULT_STRONG_TAU,
    hack_threshold: float = DEFAULT_HACK_THRESHOLD,
) -> list[dict[str, Any]]:
    """Gamed architectures re-emitted as evaluator adversarial samples (self-play).

    Each targets the champion blind spot it hides (its residual poison flaw), so
    feeding these to ``evolve.propose_via_frontier(adversarial_samples=...)`` rewards
    evaluator rubrics that stay stringent on the agent's disguised gaming.
    """
    out: list[dict[str, Any]] = []
    for arch in archives:
        for a in arch.gamed(tau=tau, hack_threshold=hack_threshold):
            targets = a.residual_flaws[0] if a.residual_flaws else "unknown"
            out.append(
                {
                    "id": f"agentmined_gamed_{a.need_id}",
                    "label": "weak",
                    "split": "adversarial",
                    "domain": a.domain,
                    "architecture": a.architecture,
                    "flaws": list(a.residual_flaws),
                    "strengths": [],
                    "targets": targets,
                    "provenance": "agent_concrete_archive",
                }
            )
    return out


def persist_archive(
    archives: list[NeedArchive], agent_epoch: int, *, directory: Path | None = None
) -> str:
    directory = directory or _ARCHIVE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"epoch-{agent_epoch}.json"
    payload = {
        "agent_epoch": agent_epoch,
        "created_at": int(time.time()),
        "archives": [a.to_dict() for a in archives],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)
