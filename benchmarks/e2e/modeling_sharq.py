from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from model_utils import quantize_model_llama, quantize_model_mixtral, quantize_model_qwen  # noqa: E402


def infer_model_family(model_name_or_path: str) -> str:
    model_name = model_name_or_path.rstrip("/").split("/")[-1].lower()
    if "llama" in model_name:
        return "llama"
    if "qwen" in model_name:
        return "qwen"
    if "mixtral" in model_name:
        return "mixtral"
    raise ValueError(f"Unsupported model family for path: {model_name_or_path}")


def get_quantizer(model_name_or_path: str):
    family = infer_model_family(model_name_or_path)
    if family == "llama":
        return quantize_model_llama
    if family == "qwen":
        return quantize_model_qwen
    if family == "mixtral":
        return quantize_model_mixtral
    raise ValueError(f"Unsupported model family for path: {model_name_or_path}")


def load_benchmark_model(
    model_name_or_path: str,
    mode: str,
    device: torch.device,
    kv_cache: bool = False,
    extra_fusion: bool = True,
    trust_remote_code: bool = False,
):
    mode = mode.upper()
    load_kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }

    if mode == "FP16":
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            **load_kwargs,
        )
        model = model.to(device=device, dtype=torch.float16)
    elif mode == "SHARQ":
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            **load_kwargs,
        )
        quantize_model = get_quantizer(model_name_or_path)
        model = quantize_model(
            model,
            device=device,
            kv_cache=kv_cache,
            quant_type="SHARQ",
            extra_fusion=extra_fusion,
        )
        model = model.to(device=device)
    else:
        raise ValueError(f"Unsupported benchmark mode: {mode}")

    model.eval()
    model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True
    return model
