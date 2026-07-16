"""Record/replay cassette for the OpenAI-compatible chat client.

Why this exists
---------------
The default test suite runs against a *deterministic mock* (see
:mod:`backend.inference.lemonade_client`) so CI is fully offline and reproducible
with no GPU. But we also want a *nightly / opt-in* pass against a REAL local model
(Lemonade or vLLM-ROCm) to catch drift between the mock and the live path. A raw
live run is not reproducible (sampling, model updates), so this module records
each request → completion pair once and lets later runs REPLAY them byte-for-byte
with no network. The recorded cassette is the CI determinism guarantee for the
live code path.

Design
------
* Each chat request is canonicalised (model + messages + sampling params, sorted
  keys) and hashed with SHA-256 into a stable key; the completion string is
  stored as ``<key>.json`` under the cassette directory.
* The layer is transport-agnostic and has **no** heavy deps — it is pure stdlib
  so it unit-tests offline without a model.

Modes (``LEMONADE_CASSETTE_MODE``)
----------------------------------
* ``off``    — cassette disabled (default when unset).
* ``record`` — call the live model, persist every (request → completion).
* ``replay`` — serve completions from the cassette; never touch the network. A
  miss deterministically falls back to the mock so CI never stalls.
* ``auto``   — replay on a hit, otherwise record (needs a live server on a miss).

Wire it up via env ``LEMONADE_CASSETTE_DIR`` + ``LEMONADE_CASSETTE_MODE`` (see
:func:`Cassette.from_env`) or by constructing :class:`Cassette` explicitly.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

OFF = "off"
RECORD = "record"
REPLAY = "replay"
AUTO = "auto"
_VALID_MODES = {OFF, RECORD, REPLAY, AUTO}

ENV_DIR = "LEMONADE_CASSETTE_DIR"
ENV_MODE = "LEMONADE_CASSETTE_MODE"


def request_key(payload: dict[str, Any]) -> str:
    """Stable SHA-256 key for a chat request payload.

    Canonicalises with sorted keys + compact separators so logically-identical
    requests (dict ordering aside) map to the same cassette entry.
    """
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


class Cassette:
    """A directory of recorded ``request -> completion`` pairs."""

    def __init__(self, directory: str | Path, mode: str = REPLAY) -> None:
        mode = (mode or REPLAY).lower()
        if mode not in _VALID_MODES:
            raise ValueError(f"unknown cassette mode {mode!r}; expected one of {sorted(_VALID_MODES)}")
        self.directory = Path(directory)
        self.mode = mode

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Cassette | None":
        """Build a cassette from ``LEMONADE_CASSETTE_DIR`` / ``LEMONADE_CASSETTE_MODE``.

        Returns ``None`` when no directory is configured or the mode is ``off`` —
        so the default (unset) environment keeps the client on its normal
        live/mock path with zero cassette overhead.
        """
        env = env if env is not None else dict(os.environ)
        directory = env.get(ENV_DIR)
        if not directory:
            return None
        mode = env.get(ENV_MODE, REPLAY).lower()
        if mode == OFF:
            return None
        return cls(directory, mode)

    @property
    def can_replay(self) -> bool:
        return self.mode in (REPLAY, AUTO)

    @property
    def can_record(self) -> bool:
        return self.mode in (RECORD, AUTO)

    def path_for(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, payload: dict[str, Any]) -> str | None:
        """Return the recorded completion for ``payload``, or ``None`` on a miss."""
        path = self.path_for(request_key(payload))
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        completion = data.get("completion")
        return completion if isinstance(completion, str) else None

    def put(self, payload: dict[str, Any], completion: str) -> str:
        """Persist ``completion`` for ``payload``; returns the request key."""
        self.directory.mkdir(parents=True, exist_ok=True)
        key = request_key(payload)
        record = {"key": key, "request": payload, "completion": completion}
        self.path_for(key).write_text(
            json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
        )
        return key
