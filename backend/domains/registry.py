"""Auto-discovering registry for domain packs.

Discovery rules (a domain is "drop-in"):

    1. Any subpackage under ``backend/domains/`` exposing a module-level ``PACK``
       (a :class:`DomainPack`) is registered as-is.
    2. Any subdirectory containing a ``domain.yaml`` is wrapped in a
       :class:`FileBackedDomainPack` automatically — no python code required.

The registry is lazy + cached. Call :func:`refresh` after adding a folder at
runtime (mainly for tests).
"""

from __future__ import annotations

import importlib
from pathlib import Path

from backend.domains.base import DomainPack, FileBackedDomainPack

_DOMAINS_DIR = Path(__file__).resolve().parent
_registry: dict[str, DomainPack] | None = None


def _discover() -> dict[str, DomainPack]:
    found: dict[str, DomainPack] = {}
    for child in sorted(_DOMAINS_DIR.iterdir()):
        if not child.is_dir() or child.name.startswith((".", "_")):
            continue

        # Rule 1: a subpackage exposing PACK.
        pack: DomainPack | None = None
        module_name = f"{__package__}.{child.name}"
        if (child / "__init__.py").exists() or (child / "pack.py").exists():
            for mod_suffix in ("", ".pack"):
                try:
                    mod = importlib.import_module(module_name + mod_suffix)
                except Exception:
                    continue
                candidate = getattr(mod, "PACK", None)
                if candidate is not None:
                    pack = candidate
                    break

        # Rule 2: data-only folder with domain.yaml.
        if pack is None and (child / "domain.yaml").exists():
            pack = FileBackedDomainPack(child)

        if pack is not None:
            found[pack.id] = pack
    return found


def get_registry() -> dict[str, DomainPack]:
    global _registry
    if _registry is None:
        _registry = _discover()
    return _registry


def refresh() -> dict[str, DomainPack]:
    """Force re-discovery (e.g. after dropping in a new domain folder)."""
    global _registry
    _registry = _discover()
    return _registry


def list_domains() -> list[DomainPack]:
    return list(get_registry().values())


def get_domain(domain_id: str) -> DomainPack | None:
    return get_registry().get(domain_id)
