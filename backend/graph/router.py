"""Lightweight need -> workflow_template router.

Maps a natural-language domain need to the best-fit workflow template using a
hybrid of curated-keyword overlap and a cheap hashing-embedding cosine (numpy
only). This deliberately avoids an extra LLM call on the hot path. If the
optional ``semantic-router`` library were installed we could swap it in behind
the same interface, but the keyword+embedding fallback needs no heavy deps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.domains.base import WorkflowTemplate
from backend.domains.registry import get_domain, list_domains
from backend.memory.qdrant_store import embed


@dataclass
class RouteMatch:
    template: WorkflowTemplate
    domain_id: str
    score: float

    def to_public_dict(self) -> dict:
        d = self.template.to_public_dict()
        d["score"] = round(self.score, 4)
        return d


def _tokenize(text: str) -> set[str]:
    return {
        t
        for t in "".join(c.lower() if c.isalnum() else " " for c in text).split()
        if len(t) > 1
    }


def _keyword_score(need: str, template: WorkflowTemplate) -> float:
    need_l = need.lower()
    need_tokens = _tokenize(need)
    if not template.needs:
        return 0.0
    hits = 0.0
    for kw in template.needs:
        kw_l = kw.lower()
        if kw_l in need_l:  # multi-word / substring match
            hits += 1.0
        elif _tokenize(kw) & need_tokens:  # token overlap
            hits += 0.5
    return hits / len(template.needs)


def _semantic_score(need: str, template: WorkflowTemplate) -> float:
    repr_text = f"{template.name}. {template.description}. {' '.join(template.needs)}"
    a = np.asarray(embed(need), dtype=np.float32)
    b = np.asarray(embed(repr_text), dtype=np.float32)
    # embeddings are L2-normalized already, so dot == cosine.
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _candidates(domain_id: str | None) -> list[tuple[str, WorkflowTemplate]]:
    out: list[tuple[str, WorkflowTemplate]] = []
    if domain_id:
        pack = get_domain(domain_id)
        # ``domain_id`` may be a registered pack id (scoped routing) OR a
        # free-form industry label typed by a user (the wizard sends the latter).
        # When it is not a known pack, fall back to a cross-domain search so the
        # router still returns useful templates instead of nothing.
        packs = [pack] if pack is not None else list_domains()
    else:
        packs = list_domains()
    for pack in packs:
        if pack is None:
            continue
        for tmpl in pack.workflow_templates():
            out.append((pack.id, tmpl))
    return out


def route_need(
    need: str, domain_id: str | None = None, top_k: int | None = None
) -> list[RouteMatch]:
    """Rank workflow templates for ``need`` (best first)."""
    matches: list[RouteMatch] = []
    for dom_id, tmpl in _candidates(domain_id):
        kw = _keyword_score(need, tmpl)
        sem = _semantic_score(need, tmpl)
        score = 0.7 * kw + 0.3 * sem  # keyword-dominant, semantic tiebreak
        matches.append(RouteMatch(template=tmpl, domain_id=dom_id, score=score))
    matches.sort(key=lambda m: m.score, reverse=True)
    if top_k is not None:
        matches = matches[:top_k]
    return matches


def best_template(need: str, domain_id: str | None = None) -> str | None:
    matches = route_need(need, domain_id, top_k=1)
    return matches[0].template.id if matches else None
