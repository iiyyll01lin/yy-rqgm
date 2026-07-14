"""Robust JSON extraction from (possibly messy) LLM output.

LLMs wrap JSON in prose or ```json fences. These helpers recover the first
JSON object so nodes never crash on formatting noise. Used by the evaluator
judge and the GEPA mutation loop.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort parse of the first JSON object in ``text``. None on failure."""
    if not text:
        return None

    # 1) whole string is JSON
    candidate = text.strip()
    obj = _try_load(candidate)
    if obj is not None:
        return obj

    # 2) inside a ```json ... ``` fence
    m = _FENCE_RE.search(text)
    if m:
        obj = _try_load(m.group(1).strip())
        if obj is not None:
            return obj

    # 3) first balanced {...} block
    block = _first_balanced_object(text)
    if block is not None:
        obj = _try_load(block)
        if obj is not None:
            return obj
    return None


def _try_load(s: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(s)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
