"""Domain packs — drop-in plugins that specialise the platform for a vertical.

A new domain goes live by dropping a folder under ``backend/domains/`` with the
standard files (``domain.yaml``, ``poison_pills.yaml``, ``workflow_templates/``,
``rubric_fragment.xml``); :mod:`backend.domains.registry` auto-discovers it. The
anchor domain is ``smart_manufacturing``.
"""

from backend.domains.base import (  # noqa: F401
    DomainPack,
    FileBackedDomainPack,
    WorkflowTemplate,
)
from backend.domains.registry import (  # noqa: F401
    get_domain,
    get_registry,
    list_domains,
)

__all__ = [
    "DomainPack",
    "FileBackedDomainPack",
    "WorkflowTemplate",
    "get_domain",
    "get_registry",
    "list_domains",
]
