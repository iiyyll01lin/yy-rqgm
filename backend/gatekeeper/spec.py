"""Hardware / model specs + JSON loaders for the Static Gatekeeper.

Pure data + loading only. No LLM calls, no mutable global state. These specs are
consumed by the deterministic math in ``vram.py`` / ``bandwidth.py`` /
``feasibility.py`` and surfaced by the ``/api/tiers`` and ``/api/models``
endpoints.

Unit convention (documented once, used everywhere):
    * memory / VRAM is in **GB decimal** (1 GB = 1e9 bytes)
    * bandwidth is in **TB/s** (1 TB/s = 1e12 bytes/s)

Using decimal GB keeps the hand-computed test cases clean (e.g. a 7B int4 model
weighs exactly 3.5 GB = 7e9 * 0.5 B / 1e9).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_GATEKEEPER_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _GATEKEEPER_DIR.parent

TIERS_PATH = _GATEKEEPER_DIR / "tiers.json"
MODELS_PATH = _BACKEND_DIR / "app" / "models_catalog.json"


@dataclass(frozen=True)
class ModelSpec:
    """A single LLM's architecture, enough to size weights + KV cache."""

    id: str
    name: str
    params_b: float
    n_layers: int
    n_kv_heads: int
    head_dim: int
    hidden: int
    context_len: int
    dtype_default: str

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        return cls(
            id=d["id"],
            name=d["name"],
            params_b=float(d["params_b"]),
            n_layers=int(d["n_layers"]),
            n_kv_heads=int(d["n_kv_heads"]),
            head_dim=int(d["head_dim"]),
            hidden=int(d["hidden"]),
            context_len=int(d["context_len"]),
            dtype_default=str(d["dtype_default"]),
        )

    def to_public_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "params_b": self.params_b,
            "n_layers": self.n_layers,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "hidden": self.hidden,
            "context_len": self.context_len,
            "dtype_default": self.dtype_default,
        }


@dataclass(frozen=True)
class TierSpec:
    """A single AMD hardware tier's capacity + bandwidth."""

    id: str
    name: str
    cls: str  # ryzen_ai | radeon | radeon_pro | instinct  ("class" is reserved)
    memory_gb: float
    bandwidth_tbs: float
    form_factor: str
    has_npu: bool
    tops_npu: float | None
    price_usd_est: float | None
    notes: str

    @classmethod
    def from_dict(cls, d: dict) -> "TierSpec":
        return cls(
            id=d["id"],
            name=d["name"],
            cls=d["class"],
            memory_gb=float(d["memory_gb"]),
            bandwidth_tbs=float(d["bandwidth_tbs"]),
            form_factor=d.get("form_factor", ""),
            has_npu=bool(d.get("has_npu", False)),
            tops_npu=(None if d.get("tops_npu") is None else float(d["tops_npu"])),
            price_usd_est=(
                None if d.get("price_usd_est") is None else float(d["price_usd_est"])
            ),
            notes=d.get("notes", ""),
        )

    def to_public_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "class": self.cls,
            "memory_gb": self.memory_gb,
            "bandwidth_tbs": self.bandwidth_tbs,
            "form_factor": self.form_factor,
            "has_npu": self.has_npu,
            "tops_npu": self.tops_npu,
            "price_usd_est": self.price_usd_est,
            "notes": self.notes,
        }


@lru_cache(maxsize=1)
def _load_tiers() -> tuple[TierSpec, ...]:
    with TIERS_PATH.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return tuple(TierSpec.from_dict(t) for t in raw["tiers"])


@lru_cache(maxsize=1)
def _load_models() -> tuple[ModelSpec, ...]:
    with MODELS_PATH.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return tuple(ModelSpec.from_dict(m) for m in raw["models"])


def list_tiers() -> list[TierSpec]:
    return list(_load_tiers())


def list_models() -> list[ModelSpec]:
    return list(_load_models())


def get_tier(tier_id: str) -> TierSpec | None:
    for t in _load_tiers():
        if t.id == tier_id:
            return t
    return None


def get_model(model_id: str) -> ModelSpec | None:
    for m in _load_models():
        if m.id == model_id:
            return m
    return None
