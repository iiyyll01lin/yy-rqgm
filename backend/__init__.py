"""AgentForge backend.

A self-evolving AI-agent sizing/education platform on the AMD open-source ROCm
stack. The package is split so the *architecture's separation of concerns* is
visible in the folder layout:

- ``backend.gatekeeper`` — DETERMINISTIC physics (VRAM / bandwidth). Never evolves.
- ``backend.evaluator``  — FUZZY, evolving RQGM domain judge. Evolves via HITL-gated epochs.

Everything here is importable without ROCm/torch/quark installed.
"""

__version__ = "0.1.0"
