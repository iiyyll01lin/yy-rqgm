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

import hashlib
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


def _stable_unit_float(seed: str) -> float:
    """Deterministic float in [0, 1) derived from a string (for stable mocks)."""
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


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
    # Deterministic but input-dependent so different architectures score differently.
    score = round(0.15 + 0.6 * _stable_unit_float(prompt), 3)
    red_flags = []
    if "duct-tape" in prompt.lower() or "threshold" in prompt.lower():
        red_flags.append(
            {
                "criterion": "physics_common_sense",
                "severity": "high",
                "detail": "Relies on a static numerical threshold rather than a physical root-cause model.",
            }
        )
    if score > 0.5:
        red_flags.append(
            {
                "criterion": "diagnostic_resilience",
                "severity": "medium",
                "detail": "No explicit handling for correlated sensor noise / cascading false positives.",
            }
        )
    if "state" not in prompt.lower():
        red_flags.append(
            {
                "criterion": "modularity_drift",
                "severity": "low",
                "detail": "State schema under-specified; risk of LangGraph node coupling.",
            }
        )
    return json.dumps(
        {
            "deficit_score": score,
            "red_flags": red_flags,
            "reasoning": (
                "[MOCK judge] Scored on physics_common_sense, diagnostic_resilience and "
                "modularity_drift. Deficit reflects reliance on numerical shortcuts vs. "
                "mechanism-level reasoning. Lower is better."
            ),
            "_mock": True,
        }
    )


def _mock_mutate(prompt: str) -> str:
    return json.dumps(
        {
            "reflection": (
                "[MOCK GEPA] Recurring HITL corrections indicate the champion under-penalises "
                "solutions that ignore correlated sensor noise. Actionable side-information: add "
                "an explicit resilience criterion and a poison-pill for cooling-valve stiction."
            ),
            "proposed_changes": [
                "Add criterion 'noise_resilience' weighting survival under correlated noise.",
                "Sharpen 'physics_common_sense' to require a named physical mechanism.",
            ],
            "new_criteria": [
                {
                    "id": "noise_resilience",
                    "text": "Does the design survive correlated sensor noise and cascading false positives without unsafe actuation?",
                }
            ],
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
