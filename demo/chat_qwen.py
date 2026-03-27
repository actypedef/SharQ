from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from model_utils import quantize_model_qwen  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple local Qwen chatbot with NVFP4 / SharQ quantization.")
    parser.add_argument(
        "--model",
        type=str,
        default=str((REPO_ROOT / ".." / "Qwen2.5-7B-Instruct").resolve()),
        help="Local path to a Qwen model.",
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        default="SHARQ",
        choices=["NVFP4", "SHARQ", "SHARQ_SIM"],
        help="Quantization path to use.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device used for quantization and generation.")
    parser.add_argument("--kv-cache", action="store_true", help="Apply simple int4 fake quantization to KV tensors.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--prompt", type=str, default=None, help="Single-turn prompt. If omitted, starts an interactive chat loop.")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt inserted into the chat template.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def load_quantized_model(model_path: str, device: torch.device, quant_type: str, kv_cache: bool, extra_fusion: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, legacy=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    model.eval()

    start_time = time.time()
    model = quantize_model_qwen(
        model,
        device=device,
        kv_cache=kv_cache,
        quant_type=quant_type,
        extra_fusion=extra_fusion,
    )
    model = model.to(device)
    model.eval()
    model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    elapsed = time.time() - start_time
    print(f"Loaded and quantized model in {elapsed:.2f}s with {quant_type}")
    return tokenizer, model


@torch.no_grad()
def generate_reply(
    tokenizer,
    model,
    messages,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> str:
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    else:
        generation_kwargs["temperature"] = None
        generation_kwargs["top_p"] = None
        generation_kwargs["top_k"] = None

    output_ids = model.generate(**inputs, **generation_kwargs)
    generated_ids = output_ids[0][inputs.input_ids.shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main():
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    tokenizer, model = load_quantized_model(
        model_path=args.model,
        device=device,
        quant_type=args.quant_type,
        kv_cache=args.kv_cache,
        extra_fusion=True,
    )

    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})

    if args.prompt is not None:
        messages.append({"role": "user", "content": args.prompt})
        reply = generate_reply(
            tokenizer=tokenizer,
            model=model,
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        print(reply)
        return

    print("Interactive chat started. Type /exit to quit, /clear to clear history.")
    while True:
        user_text = input("\nUser: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"/exit", "exit", "quit"}:
            break
        if user_text.lower() == "/clear":
            messages = []
            if args.system_prompt:
                messages.append({"role": "system", "content": args.system_prompt})
            print("History cleared.")
            continue

        messages.append({"role": "user", "content": user_text})
        reply = generate_reply(
            tokenizer=tokenizer,
            model=model,
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        print(f"\nAssistant: {reply}")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
