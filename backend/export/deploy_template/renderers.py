"""Runnable deployment-template renderers (placeholder substitution, no f-string
brace-escaping headaches).

Emits a ROCm vLLM docker-compose, a LangGraph boilerplate app.py, a README, a
requirements.txt and a .env.example — a self-contained starter a customer can
`docker compose up`.
"""

from __future__ import annotations

from backend.domains.base import WorkflowTemplate
from backend.gatekeeper.spec import ModelSpec, TierSpec

# Best-effort Hugging Face repo hints for the vLLM `serve` target.
_HF_HINTS: dict[str, str] = {
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-7b-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
    "deepseek-r1-distill-qwen-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "llama-3.1-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
}


def _hf_id(model: ModelSpec) -> str:
    return _HF_HINTS.get(model.id, f"<HF_REPO_FOR_{model.id}>")


def _default_nodes(template: WorkflowTemplate | None) -> list[str]:
    if template and template.nodes:
        return list(template.nodes)
    return ["ingest", "reason", "act", "hitl_review"]


# ---------------------------------------------------------------------------
_DOCKER_COMPOSE = """# AgentForge deploy template — AMD ROCm vLLM + Qdrant + your LangGraph app.
# Target tier: __TIER_NAME__ (__TIER_CLASS__)
#
# NOTE: this ROCm vLLM profile targets Radeon / Radeon PRO / Instinct.
# For a Ryzen AI (NPU) box, serve via Lemonade instead (see README).
services:
  vllm:
    image: rocm/vllm-dev:main
    container_name: agentforge-vllm
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
      - render
    ipc: host
    security_opt:
      - seccomp=unconfined
    environment:
      - VLLM_ROCM_USE_AITER=1
      - HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    # __QUANT_COMMENT__
    command: >
      vllm serve __VLLM_MODEL__
      --enable-prefix-caching
      --max-model-len __MAX_LEN__
      --port 8000

  qdrant:
    image: qdrant/qdrant
    container_name: agentforge-qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_storage:/qdrant/storage

  app:
    build: .
    container_name: agentforge-app
    depends_on:
      - vllm
      - qdrant
    environment:
      - OPENAI_BASE_URL=http://vllm:8000/v1
      - QDRANT_URL=http://qdrant:6333
      - MODEL=__VLLM_MODEL__
    command: python app.py
"""


def render_docker_compose(model: ModelSpec, tier: TierSpec, max_len: int = 8192) -> str:
    quant = (
        "# int4/fp8 weights: quantize with AMD Quark and point serve at the quantized repo,"
        " adding --quantization <method>."
        if model.dtype_default in ("int4", "fp8", "int8")
        else "# fp16/bf16 weights: serve the base repo directly."
    )
    return (
        _DOCKER_COMPOSE.replace("__TIER_NAME__", tier.name)
        .replace("__TIER_CLASS__", tier.cls)
        .replace("__VLLM_MODEL__", _hf_id(model))
        .replace("__MAX_LEN__", str(max_len))
        .replace("__QUANT_COMMENT__", quant)
    )


# ---------------------------------------------------------------------------
_DOCKERFILE = """FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
CMD ["python", "app.py"]
"""


def render_dockerfile() -> str:
    return _DOCKERFILE


# ---------------------------------------------------------------------------
_REQUIREMENTS = """langgraph>=0.2.50
httpx>=0.27
qdrant-client>=1.9
pyyaml>=6.0
"""


def render_requirements() -> str:
    return _REQUIREMENTS


_ENV_EXAMPLE = """# Point at your local OpenAI-compatible endpoint.
# vLLM on ROCm (Radeon/Instinct):
OPENAI_BASE_URL=http://localhost:8000/v1
# ...or Lemonade on Ryzen AI:
# OPENAI_BASE_URL=http://localhost:8020/api/v1
MODEL=__VLLM_MODEL__
QDRANT_URL=http://localhost:6333
# HUGGING_FACE_HUB_TOKEN=hf_xxx
"""


def render_env_example(model: ModelSpec) -> str:
    return _ENV_EXAMPLE.replace("__VLLM_MODEL__", _hf_id(model))


# ---------------------------------------------------------------------------
_APP_PY = '''"""AgentForge-exported LangGraph app.

Domain: __DOMAIN__
Template: __TEMPLATE_NAME__

Runnable boilerplate. Talks to a local OpenAI-compatible endpoint (vLLM on ROCm,
or Lemonade on Ryzen AI). Replace the stubbed node bodies with your logic.

IMPORTANT (AgentForge design rule): keep any physical/safety feasibility check
DETERMINISTIC and separate from the fuzzy/LLM nodes. Never let a model override
a hard safety gate.
"""
from __future__ import annotations

import os
from typing import Any, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "__BASE_URL__")
MODEL = os.getenv("MODEL", "__MODEL__")


def chat(prompt: str) -> str:
    """Minimal OpenAI-compatible chat call (falls back to a stub when offline)."""
    try:
        resp = httpx.post(
            OPENAI_BASE_URL.rstrip("/") + "/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        return f"[offline stub] {exc}"


class State(TypedDict, total=False):
    input: str
    output: str
    trace: list


__NODE_FUNCS__
def build():
    graph = StateGraph(State)
__ADD_NODES__
__ADD_EDGES__
    return graph.compile()


if __name__ == "__main__":
    app = build()
    result = app.invoke({"input": "hello from AgentForge", "trace": []})
    print(result)
'''


def _render_node_funcs(nodes: list[str]) -> str:
    blocks: list[str] = []
    for n in nodes:
        safe = n.replace("-", "_")
        blocks.append(
            f'def {safe}(state: State) -> dict[str, Any]:\n'
            f'    """Node: {n}. TODO: implement."""\n'
            f'    trace = list(state.get("trace", [])) + ["{n}"]\n'
            f'    _ = chat("[{n}] " + str(state.get("input", "")))  # example LLM call\n'
            f'    return {{"trace": trace}}\n'
        )
    return "\n".join(blocks) + "\n"


def _render_add_nodes(nodes: list[str]) -> str:
    lines = [f'    graph.add_node("{n}", {n.replace("-", "_")})' for n in nodes]
    return "\n".join(lines)


def _render_add_edges(nodes: list[str]) -> str:
    if not nodes:
        return "    graph.add_edge(START, END)"
    lines = [f'    graph.add_edge(START, "{nodes[0]}")']
    for a, b in zip(nodes, nodes[1:]):
        lines.append(f'    graph.add_edge("{a}", "{b}")')
    lines.append(f'    graph.add_edge("{nodes[-1]}", END)')
    return "\n".join(lines)


def render_app_py(
    model: ModelSpec,
    tier: TierSpec,
    template: WorkflowTemplate | None,
    domain_id: str | None,
) -> str:
    nodes = _default_nodes(template)
    base_url = "http://localhost:8020/api/v1" if tier.cls == "ryzen_ai" else "http://localhost:8000/v1"
    return (
        _APP_PY.replace("__DOMAIN__", domain_id or "general")
        .replace("__TEMPLATE_NAME__", template.name if template else "(custom)")
        .replace("__BASE_URL__", base_url)
        .replace("__MODEL__", _hf_id(model))
        .replace("__NODE_FUNCS__", _render_node_funcs(nodes))
        .replace("__ADD_NODES__", _render_add_nodes(nodes))
        .replace("__ADD_EDGES__", _render_add_edges(nodes))
    )


# ---------------------------------------------------------------------------
_README = """# AgentForge deployment starter

Exported for **__DOMAIN__** using template **__TEMPLATE_NAME__**, sized for
**__TIER_NAME__** running **__MODEL_NAME__** (__DTYPE__).

## What's here
- `docker-compose.yml` — AMD ROCm vLLM server + Qdrant + your app.
- `app.py` — runnable LangGraph boilerplate (nodes: __NODES__).
- `Dockerfile`, `requirements.txt`, `.env.example`.

## Run it (Radeon / Radeon PRO / Instinct, ROCm)
```bash
cp .env.example .env         # set HUGGING_FACE_HUB_TOKEN if needed
docker compose up -d vllm qdrant
# wait for vLLM to load, then:
docker compose up app
```
The vLLM server exposes an OpenAI-compatible API at `http://localhost:8000/v1`
with `--enable-prefix-caching` and `VLLM_ROCM_USE_AITER=1`, mapping `/dev/kfd`
and `/dev/dri` into the container.

## Ryzen AI (NPU) alternative
Serve locally with **Lemonade** (OpenAI-compatible) and point `OPENAI_BASE_URL`
at `http://localhost:8020/api/v1`:
```bash
lemonade serve
```

## Sizing (from AgentForge's deterministic gatekeeper)
- Total VRAM: ~__VRAM__ GB on a __MEM__ GB card (__FEASIBLE__).
- Est. decode throughput: ~__TPS__ tokens/s.

Quantize weights with **AMD Quark** for int4/fp8 deployment.
"""


def render_readme(
    model: ModelSpec,
    tier: TierSpec,
    template: WorkflowTemplate | None,
    domain_id: str | None,
    vram_total_gb: float,
    tokens_per_s: float,
    feasible: bool,
    dtype: str | None = None,
) -> str:
    nodes = _default_nodes(template)
    return (
        _README.replace("__DOMAIN__", domain_id or "general")
        .replace("__TEMPLATE_NAME__", template.name if template else "(custom)")
        .replace("__TIER_NAME__", tier.name)
        .replace("__MODEL_NAME__", model.name)
        .replace("__DTYPE__", dtype or model.dtype_default)
        .replace("__NODES__", ", ".join(nodes))
        .replace("__VRAM__", f"{vram_total_gb:.1f}")
        .replace("__MEM__", f"{tier.memory_gb:.0f}")
        .replace("__FEASIBLE__", "fits" if feasible else "DOES NOT FIT — size up")
        .replace("__TPS__", f"{tokens_per_s:.0f}")
    )
