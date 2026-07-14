"""Local AMD inference client (Lemonade, OpenAI-compatible).

Targets a Lemonade server (Ryzen AI NPU / Radeon ROCm) when one is reachable,
and otherwise falls back to a deterministic MOCK so the *entire* platform runs
with no live model, no GPU, and no network.
"""

from backend.inference.lemonade_client import (  # noqa: F401
    DEFAULT_BASE_URL,
    LemonadeClient,
    MockMarker,
    get_lemonade_client,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "LemonadeClient",
    "MockMarker",
    "get_lemonade_client",
]
