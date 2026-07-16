"""OpenAI-compatible client for a local Lemonade server, with a MOCK fallback.

Lemonade (https://github.com/lemonade-sdk/lemonade) exposes an OpenAI-compatible
API (``/api/v1/chat/completions`` etc.) for running LLMs locally on AMD Ryzen AI
(NPU) and Radeon (ROCm) hardware.

CRITICAL design property: if the endpoint is unreachable (no server, no GPU, no
network), :meth:`LemonadeClient.chat` transparently returns a *deterministic
mock* completion. This lets the whole AgentForge platform — LangGraph
orchestration, RQGM evaluation, evolution — run end-to-end on a normal machine.

Nodes that need structured output embed a :class:`MockMarker` sentinel in their
prompt; the mock recognises it and returns schema-valid JSON so downstream
parsing always succeeds offline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://localhost:8020/api/v1"
DEFAULT_MODEL = "AgentForge-Local"

# Lemonade's default port moved to 13305 in recent releases; the task pins the
# default to 8020. Override either via LEMONADE_BASE_URL.
_CONNECT_TIMEOUT_S = 1.5
_REQUEST_TIMEOUT_S = 60.0


class MockMarker(str, Enum):
    """Sentinels a caller embeds in a prompt to steer the deterministic mock.

    Using an explicit marker keeps the inference layer decoupled from any single
    node's schema while still producing parseable offline output.
    """

    TASK_AGENT = "<<AGENTFORGE:TASK_AGENT>>"
    EVALUATOR = "<<AGENTFORGE:RQGM_EVALUATOR>>"
    MUTATE = "<<AGENTFORGE:GEPA_MUTATE>>"
    ROUTER = "<<AGENTFORGE:ROUTER>>"


Message = dict[str, str]


def _normalize_messages(messages: Any) -> list[Message]:
    """Accept a str, a single message dict, or a list of message dicts."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    if isinstance(messages, dict):
        return [messages]
    out: list[Message] = []
    for m in messages:
        if isinstance(m, str):
            out.append({"role": "user", "content": m})
        else:
            out.append({"role": str(m.get("role", "user")), "content": str(m.get("content", ""))})
    return out


@dataclass
class LemonadeClient:
    """Thin OpenAI-compatible chat client with a deterministic mock fallback."""

    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    force_mock: bool | None = None

    def __post_init__(self) -> None:
        self.base_url = (self.base_url or os.getenv("LEMONADE_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.model = self.model or os.getenv("LEMONADE_MODEL", DEFAULT_MODEL)
        self.api_key = self.api_key or os.getenv("LEMONADE_API_KEY")
        if self.force_mock is None:
            self.force_mock = os.getenv("LEMONADE_FORCE_MOCK", "").lower() in ("1", "true", "yes")
        self._live: bool | None = None

    # -- liveness ----------------------------------------------------------
    def is_live(self) -> bool:
        """One-shot probe (cached). Never raises."""
        if self.force_mock:
            return False
        if self._live is not None:
            return self._live
        try:
            headers = self._headers()
            resp = httpx.get(f"{self.base_url}/models", headers=headers, timeout=_CONNECT_TIMEOUT_S)
            self._live = resp.status_code < 500
        except Exception:
            self._live = False
        return self._live

    @property
    def using_mock(self) -> bool:
        return not self.is_live()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # -- chat --------------------------------------------------------------
    def chat(
        self,
        messages: Any,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 768,
        **kwargs: Any,
    ) -> str:
        """Return the assistant text for ``messages``. Falls back to mock."""
        msgs = _normalize_messages(messages)
        use_model = model or self.model

        if not self.is_live():
            return _mock_chat(msgs, use_model)

        payload: dict[str, Any] = {
            "model": use_model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        payload.update(kwargs)
        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=_REQUEST_TIMEOUT_S,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception:
            # Any live failure -> deterministic mock so the platform never stalls.
            self._live = False
            return _mock_chat(msgs, use_model)


# ---------------------------------------------------------------------------
# Deterministic mock
# ---------------------------------------------------------------------------
def _joined_prompt(messages: list[Message]) -> str:
    return "\n".join(f"{m.get('role','')}: {m.get('content','')}" for m in messages)


def _mock_task_agent(prompt: str) -> str:
    return json.dumps(
        {
            "architecture": (
                "LangGraph StateGraph: sensor_ingest -> anomaly_detector -> "
                "root_cause_analyzer -> action_recommender -> hitl_review"
            ),
            "nodes": [
                "sensor_ingest",
                "anomaly_detector",
                "root_cause_analyzer",
                "action_recommender",
                "hitl_review",
            ],
            "state_schema": {
                "sensor_window": "list[float]",
                "anomaly": "bool",
                "root_cause": "str",
                "recommended_action": "str",
                "confidence": "float",
            },
            "tools": ["timeseries_stats", "physics_surrogate_model", "knowledge_base_lookup"],
            "rationale": (
                "Separate detection from root-cause so numerical anomalies are explained "
                "by physical mechanisms before any actuator command is proposed; HITL gates "
                "irreversible actions."
            ),
            "_mock": True,
        }
    )


def _mock_evaluator(prompt: str) -> str:
    """Rubric-aware, deterministic evaluator mock.

    Delegates to :mod:`backend.inference.mock_scoring`, which reads planted flaw
    tags from the candidate and criterion ids from the active rubric so that a
    genuinely better rubric produces a genuinely higher weak-vs-strong
    separation (see that module's docstring). Emits both loose and strict
    deficits so the RQGM hack-ratio can be computed offline.
    """
    from backend.inference import mock_scoring

    return json.dumps(mock_scoring.evaluator_mock_json(prompt))


# Flaw-aware GEPA mutation targets: (prompt hints) -> (criterion id, criterion text).
# The id must match a catching criterion in backend/inference/mock_scoring so the
# added criterion actually closes a blind spot (raises weak-anchor deficit). The
# proposer picks the target from the failure trace / side-information in the
# prompt, so different traces yield different criteria -> a diverse frontier.
_MUTATE_TARGETS: list[tuple[tuple[str, ...], str, str]] = [
    (
        ("reward", "kpi", "disabl", "gaming", "hack", "pass-rate", "yield", "uptime"),
        "reward_hacking_resistance",
        "Heavily penalize designs that game their own KPI (e.g. disabling or muting the sensor "
        "that would catch the defect) instead of fixing the physical root cause.",
    ),
    (
        ("drift", "supplier", "concept", "recalibrat", "material change"),
        "drift_monitoring",
        "Penalize the absence of concept-drift monitoring: designs that never recalibrate after a "
        "supplier/material change and silently degrade.",
    ),
    (
        ("noise", "correlated", "cross-valid", "false positive", "cascading"),
        "noise_resilience",
        "Penalize designs that treat correlated sensor noise as ground-truth signal and act on it "
        "without cross-validating physically-correlated sensors.",
    ),
]


def _mock_mutate(prompt: str) -> str:
    # GEPA drives mutation from the *textual gradient* (the failure trace / side
    # information), NOT from the parent rubric — otherwise the parent's own
    # criteria text would shadow the signal. Scan the side-information section
    # first, then fall back to the whole prompt.
    marker = "ACTIONABLE SIDE INFORMATION"
    gradient = prompt.split(marker, 1)[1].lower() if marker in prompt else prompt.lower()
    chosen: tuple[str, str] | None = None
    for keys, cid, text in _MUTATE_TARGETS:
        if any(k in gradient for k in keys):
            chosen = (cid, text)
            break
    if chosen is None:
        for keys, cid, text in _MUTATE_TARGETS:
            if any(k in prompt.lower() for k in keys):
                chosen = (cid, text)
                break
    if chosen is None:
        chosen = (_MUTATE_TARGETS[0][1], _MUTATE_TARGETS[0][2])
    cid, text = chosen
    return json.dumps(
        {
            "reflection": (
                "[MOCK GEPA] Actionable side-information indicates the champion under-penalises a "
                f"specific failure mode; adding criterion '{cid}' closes that blind spot without "
                "collapsing diversity."
            ),
            "proposed_changes": [
                f"Add criterion '{cid}' to catch the documented failure mode.",
                "Sharpen the affected criterion to require a named physical mechanism.",
            ],
            "new_criteria": [{"id": cid, "text": text}],
            "_mock": True,
        }
    )


def _mock_router(prompt: str) -> str:
    return json.dumps({"template_id": None, "confidence": 0.0, "_mock": True})


def _mock_chat(messages: list[Message], model: str) -> str:
    prompt = _joined_prompt(messages)
    if MockMarker.TASK_AGENT.value in prompt:
        return _mock_task_agent(prompt)
    if MockMarker.EVALUATOR.value in prompt:
        return _mock_evaluator(prompt)
    if MockMarker.MUTATE.value in prompt:
        return _mock_mutate(prompt)
    if MockMarker.ROUTER.value in prompt:
        return _mock_router(prompt)
    # Generic deterministic fallback.
    last_user = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    return (
        f"[MOCK:{model}] Received {len(messages)} message(s). "
        f"No live Lemonade server; returning a deterministic stub for: "
        f"{last_user[:160]}"
    )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_client: LemonadeClient | None = None


def get_lemonade_client() -> LemonadeClient:
    """Return a process-wide :class:`LemonadeClient` singleton."""
    global _client
    if _client is None:
        _client = LemonadeClient()
    return _client
