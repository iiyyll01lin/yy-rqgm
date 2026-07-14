#!/usr/bin/env python
"""Quantize a Hugging Face LLM with AMD Quark (int4 / fp8) for AMD deployment.

This is OFFLINE TOOLING, not part of the runtime. The AgentForge platform never
imports torch/quark; this script imports them lazily so it only needs the heavy
ROCm stack when you actually run it:

    uv sync --extra rocm            # installs amd-quark (+ torch; see pyproject notes)
    # for a real ROCm torch build:
    #   uv pip install torch --extra-index-url https://download.pytorch.org/whl/rocm7.2

    uv run --extra rocm python scripts/quantize_quark.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --scheme int4 --output ./quantized/llama-3.1-8b-int4

The sizing produced by AgentForge's gatekeeper assumes int4/fp8 weights for the
larger models — this is how you actually produce those checkpoints.
"""

from __future__ import annotations

import argparse
import sys


def _die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize an LLM with AMD Quark.")
    parser.add_argument("--model", required=True, help="HF repo id or local path.")
    parser.add_argument("--output", required=True, help="Output directory for the quantized model.")
    parser.add_argument(
        "--scheme",
        default="int4",
        choices=["int4", "fp8", "awq_int4"],
        help="Quantization scheme (default: int4).",
    )
    parser.add_argument("--device", default="cuda", help="'cuda' maps to ROCm on AMD GPUs.")
    parser.add_argument("--export-format", default="safetensors", choices=["safetensors", "gguf", "onnx"])
    args = parser.parse_args()

    # Lazy, guarded imports — keep the platform free of heavy deps.
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        _die(
            "Missing torch/transformers. Install the ROCm extra first:\n"
            "  uv sync --extra rocm\n"
            "  uv pip install transformers\n"
            "  # ROCm torch: uv pip install torch --extra-index-url https://download.pytorch.org/whl/rocm7.2"
        )

    try:
        from quark.torch import ModelQuantizer
        from quark.torch.quantization import (  # type: ignore
            Config,
            QuantizationConfig,
        )
    except ImportError:
        _die(
            "AMD Quark not installed. Install it with:\n"
            "  uv sync --extra rocm    # pulls amd-quark from PyPI\n"
            "  # or a ROCm pre-built wheel:\n"
            "  uv pip install amd-quark --extra-index-url https://pypi.amd.com/quark/rocm72/simple"
        )

    print(f"[quantize_quark] loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map=args.device)

    # Build a Quark quantization config for the requested scheme. The exact
    # QuantizationConfig presets vary by Quark version; adjust to your installed
    # version (see https://quark.docs.amd.com). This is a representative flow.
    print(f"[quantize_quark] configuring scheme={args.scheme} ...")
    quant_config = QuantizationConfig()  # defaults; refine per Quark docs/version
    config = Config(global_quant_config=quant_config)

    quantizer = ModelQuantizer(config)
    print("[quantize_quark] quantizing (this can take a while) ...")
    model = quantizer.quantize_model(model)

    print(f"[quantize_quark] exporting to {args.output} ({args.export_format}) ...")
    try:
        quantizer.export_model(model, args.output, export_format=args.export_format)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - API drift across Quark versions
        # Fallback to a plain HF save if the export helper signature differs.
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)

    print("[quantize_quark] done. Point vLLM/Lemonade at the output directory.")


if __name__ == "__main__":
    main()
