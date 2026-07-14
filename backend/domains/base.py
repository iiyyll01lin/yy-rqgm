"""DomainPack plugin interface + a file-backed default implementation.

The ``DomainPack`` Protocol defines what every domain must provide:

    * ``id`` / ``name`` / ``description`` — identity
    * ``poison_pills()``      — adversarial scenarios the evaluator injects
    * ``workflow_templates()``— candidate agent architectures + routing keywords
    * ``rubric_fragment()``   — XML criteria merged into the RQGM evaluator rubric

``FileBackedDomainPack`` reads those from a directory so a whole new domain is
just data files — no code required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml


@dataclass(frozen=True)
class WorkflowTemplate:
    """A candidate agent-workflow blueprint for a domain need."""

    id: str
    name: str
    description: str
    needs: list[str] = field(default_factory=list)  # routing keywords
    nodes: list[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "WorkflowTemplate":
        return cls(
            id=d["id"],
            name=d.get("name", d["id"]),
            description=d.get("description", ""),
            needs=list(d.get("needs", []) or []),
            nodes=list(d.get("nodes", []) or []),
            notes=d.get("notes", ""),
        )

    def to_public_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "needs": self.needs,
        }


@runtime_checkable
class DomainPack(Protocol):
    """Structural interface every domain pack satisfies."""

    id: str
    name: str
    description: str

    def poison_pills(self) -> list[str]:
        ...

    def workflow_templates(self) -> list[WorkflowTemplate]:
        ...

    def rubric_fragment(self) -> str:
        ...


class FileBackedDomainPack:
    """A :class:`DomainPack` backed by a directory of data files.

    Expected layout::

        <dir>/domain.yaml            # {id, name, description}
        <dir>/poison_pills.yaml      # {poison_pills: [...]}
        <dir>/workflow_templates/*.yaml
        <dir>/rubric_fragment.xml
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        meta = self._read_yaml(self.directory / "domain.yaml") or {}
        self.id: str = meta.get("id", self.directory.name)
        self.name: str = meta.get("name", self.id.replace("_", " ").title())
        self.description: str = meta.get("description", "")

    @staticmethod
    def _read_yaml(path: Path):
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def poison_pills(self) -> list[str]:
        data = self._read_yaml(self.directory / "poison_pills.yaml") or {}
        pills = data.get("poison_pills", []) if isinstance(data, dict) else data
        return [str(p) for p in (pills or [])]

    def workflow_templates(self) -> list[WorkflowTemplate]:
        tdir = self.directory / "workflow_templates"
        if not tdir.is_dir():
            return []
        templates: list[WorkflowTemplate] = []
        for path in sorted(tdir.glob("*.y*ml")):
            data = self._read_yaml(path)
            if isinstance(data, dict):
                data.setdefault("id", path.stem)
                templates.append(WorkflowTemplate.from_dict(data))
        return templates

    def rubric_fragment(self) -> str:
        path = self.directory / "rubric_fragment.xml"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"FileBackedDomainPack(id={self.id!r}, dir={self.directory})"
