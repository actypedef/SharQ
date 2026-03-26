from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from model_utils import quantize_model_llama, quantize_model_mixtral, quantize_model_qwen  # noqa: E402


def load_model_and_quantizer(model_name_or_path: str):
    model_name = model_name_or_path.rstrip("/").split("/")[-1].lower()
    if "llama" in model_name:
        return quantize_model_llama
    if "qwen" in model_name:
        return quantize_model_qwen
    if "mixtral" in model_name:
        return quantize_model_mixtral
    raise ValueError(f"Unsupported model family for path: {model_name_or_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="End-to-end prefill benchmark for BF16, NVFP4, and SHARQ models.")
    parser.add_argument("--model", type=str, required=True, help="Local model path.")
    parser.add_argument("--quant-type", type=str, default="SHARQ", choices=["BF16", "NVFP4", "SHARQ", "SHARQ_SIM"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--kv-cache", action="store_true", help="Enable fake-int4 KV cache path in wrapped attention.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional text prompt. If omitted, random token ids are used for the requested batch/sequence shape.",
    )
    return parser


def make_inputs(tokenizer, batch_size: int, seqlen: int, prompt: str | None, device: torch.device):
    if prompt is not None:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded.input_ids
        if input_ids.size(1) >= seqlen:
            input_ids = input_ids[:, :seqlen]
        else:
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            pad = torch.full((1, seqlen - input_ids.size(1)), pad_id, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, pad], dim=1)
        input_ids = input_ids.repeat(batch_size, 1)
    else:
        low = 0
        high = min(getattr(tokenizer, "vocab_size", 151936), 151936)
        input_ids = torch.randint(low, high, (batch_size, seqlen), dtype=torch.long)

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "use_cache": True,
    }


def bench_prefill(model, inputs, warmup: int, iters: int):
    def run_once():
        out = model(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return out

    for _ in range(warmup):
        run_once()

    timings = []
    for _ in range(iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_once()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    timings.sort()
    avg_ms = sum(timings) / len(timings)
    med_ms = timings[len(timings) // 2]
    return avg_ms, med_ms


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the prefill benchmark.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This benchmark currently expects a CUDA device.")
    torch.cuda.set_device(device)
    device_index = torch.cuda.current_device() if device.index is None else device.index

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, legacy=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    model.eval()

    quantize_model = load_model_and_quantizer(args.model)

    torch.cuda.reset_peak_memory_stats(device_index)
    t0 = time.perf_counter()
    if args.quant_type == "BF16":
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = quantize_model(model, device=device, kv_cache=args.kv_cache, quant_type=args.quant_type)
        model = model.to(device)
    model.eval()
    model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    quant_ms = (time.perf_counter() - t0) * 1000.0

    inputs = make_inputs(tokenizer, args.batch_size, args.seqlen, args.prompt, device)

    with torch.inference_mode():
        avg_ms, med_ms = bench_prefill(model, inputs, args.warmup, args.iters)

    total_tokens = args.batch_size * args.seqlen
    avg_toks = total_tokens / (avg_ms / 1000.0)
    med_toks = total_tokens / (med_ms / 1000.0)
    peak_gb = torch.cuda.max_memory_allocated(device_index) / (1024 ** 3)

    print(f"model           : {args.model}")
    print(f"quant_type      : {args.quant_type}")
    print(f"batch_size      : {args.batch_size}")
    print(f"seqlen          : {args.seqlen}")
    print(f"warmup          : {args.warmup}")
    print(f"iters           : {args.iters}")
    print(f"quantize_ms     : {quant_ms:.3f}")
    print(f"prefill_avg_ms  : {avg_ms:.3f}")
    print(f"prefill_med_ms  : {med_ms:.3f}")
    print(f"avg_tok_per_s   : {avg_toks:.2f}")
    print(f"med_tok_per_s   : {med_toks:.2f}")
    print(f"peak_mem_gb     : {peak_gb:.3f}")


if __name__ == "__main__":
    main()
