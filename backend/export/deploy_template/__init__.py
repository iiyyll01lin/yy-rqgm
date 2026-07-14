"""Assemble the set of runnable deploy files for an export."""

from __future__ import annotations

from backend.domains.base import WorkflowTemplate
from backend.export.deploy_template.renderers import (
    render_app_py,
    render_docker_compose,
    render_dockerfile,
    render_env_example,
    render_readme,
    render_requirements,
)
from backend.gatekeeper.feasibility import Feasibility
from backend.gatekeeper.spec import ModelSpec, TierSpec


def build_deploy_files(
    model: ModelSpec,
    tier: TierSpec,
    feasibility: Feasibility,
    template: WorkflowTemplate | None = None,
    domain_id: str | None = None,
    max_len: int = 8192,
    dtype: str | None = None,
) -> dict[str, str]:
    """Return ``{filename: content}`` for a runnable ROCm deployment starter."""
    report = feasibility.to_dict()
    return {
        "docker-compose.yml": render_docker_compose(model, tier, max_len=max_len),
        "Dockerfile": render_dockerfile(),
        "requirements.txt": render_requirements(),
        ".env.example": render_env_example(model),
        "app.py": render_app_py(model, tier, template, domain_id),
        "README.md": render_readme(
            model,
            tier,
            template,
            domain_id,
            vram_total_gb=report["vram_total_gb"],
            tokens_per_s=report["tokens_per_s_est"],
            feasible=feasibility.feasible,
            dtype=dtype,
        ),
    }


__all__ = ["build_deploy_files"]
