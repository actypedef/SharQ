from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import transformers
from packaging.version import Version
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from qLinearLayer import QLinearLayer  # noqa: E402


MIN_TRANSFORMERS_VERSION = Version("5.2.0")
SUPPORTED_QUANT_TYPES = {"BF16", "NVFP4", "SHARQ"}


def require_supported_transformers() -> None:
    version = Version(transformers.__version__)
    if version < MIN_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"Qwen3.5 demo requires transformers>={MIN_TRANSFORMERS_VERSION}, got {transformers.__version__}. "
            "Please update your environment first."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone Qwen3.5-27B text-only demo for BF16 / NVFP4 / SHARQ.")
    parser.add_argument(
        "--model",
        type=str,
        default="/mnt/d/models/Qwen3.5-27B",
        help="Local path to Qwen3.5 model directory.",
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        default="SHARQ",
        choices=sorted(SUPPORTED_QUANT_TYPES),
        help="Execution path to use. NVFP4 and SHARQ quantize the text model linears only.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device for text model execution.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--prompt", type=str, default=None, help="Single-turn prompt. If omitted, starts interactive chat.")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt inserted into the chat template.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _replace_text_model_linears(module: nn.Module, quant_type: str, prefix: str = "") -> tuple[int, list[str]]:
    replaced = 0
    skipped: list[str] = []
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            try:
                setattr(module, name, QLinearLayer(child, quant_type=quant_type, extra_fusion=True))
                replaced += 1
            except Exception as exc:
                if quant_type == "NVFP4":
                    skipped.append(f"{full_name} ({child.in_features}->{child.out_features}): {exc}")
                else:
                    raise RuntimeError(f"Failed to quantize {full_name} for {quant_type}: {exc}") from exc
        else:
            child_replaced, child_skipped = _replace_text_model_linears(child, quant_type, full_name)
            replaced += child_replaced
            skipped.extend(child_skipped)
    return replaced, skipped


@torch.no_grad()
def prepare_model(model_path: str, quant_type: str, device: torch.device):
    require_supported_transformers()

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, legacy=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Qwen3_5ForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.eval()

    start_time = time.time()
    replaced = 0
    skipped: list[str] = []
    if quant_type != "BF16":
        replaced, skipped = _replace_text_model_linears(model.model, quant_type)

    model.model = model.model.to(device)
    model.lm_head = model.lm_head.to(device=device, dtype=torch.bfloat16)
    model.eval()
    model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    elapsed = time.time() - start_time
    print(f"Loaded Qwen3.5 text model in {elapsed:.2f}s with {quant_type}")
    if quant_type != "BF16":
        print(f"Quantized text-model linear layers: {replaced}")
        if skipped:
            print(f"Skipped {len(skipped)} unsupported linear layers for {quant_type}; they remain BF16.")
            for item in skipped[:8]:
                print(f"  - {item}")
            if len(skipped) > 8:
                print(f"  ... and {len(skipped) - 8} more")
    print("This demo uses the text-only Qwen3.5 model path and does not load the vision tower.")
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


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    tokenizer, model = prepare_model(args.model, args.quant_type, device)

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

    print("Interactive Qwen3.5 chat started. Type /exit to quit, /clear to clear history.")
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
