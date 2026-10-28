"""Produce calibration-free baseline checkpoints with llm-compressor (RTN, data-free).

  * NVFP4  : W4A4 NVFP4 (16-group FP8 scales + per-tensor global scale), dynamic activation quant
  * W4A8   : INT4 weight (group 128, symmetric) for Marlin W4A8 (activations quantized online
             to FP8 per-token via VLLM_MARLIN_INPUT_DTYPE=fp8)
  * FP8    : W8A8 FP8 dynamic (per-channel weight, per-token activation)

Weights use RTN. NVFP4 additionally needs a tiny calibration pass (static activation global
scale required by vLLM's NVFP4 format); everything else is data-free.
lm_head is kept in BF16 for every scheme.
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def build_calib(tok, num_samples=64, seq_len=512):
    from datasets import Dataset, load_dataset

    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("\n\n".join(raw["text"]), return_tensors=None)["input_ids"]
    samples = []
    for i in range(num_samples):
        chunk = ids[i * seq_len:(i + 1) * seq_len]
        samples.append({"input_ids": chunk, "attention_mask": [1] * len(chunk)})
    return Dataset.from_list(samples)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--schemes", nargs="+", default=["NVFP4", "W4A16", "FP8_DYNAMIC"])
    args = p.parse_args()

    name = os.path.basename(args.model.rstrip("/"))
    for scheme in args.schemes:
        save_dir = os.path.join(args.out_root, scheme.lower(), name)
        if os.path.exists(os.path.join(save_dir, "config.json")):
            print(f"[skip] {save_dir} exists")
            continue
        print(f"[quant] {scheme} -> {save_dir}")
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
        tok = AutoTokenizer.from_pretrained(args.model)
        recipe = QuantizationModifier(targets="Linear", scheme=scheme, ignore=["lm_head"])
        if scheme == "NVFP4":
            # vLLM's NVFP4 W4A4 format stores a static per-tensor activation global scale,
            # which has to be calibrated (64 x 512 tokens of wikitext2).
            oneshot(model=model, recipe=recipe, dataset=build_calib(tok), max_seq_length=512,
                    num_calibration_samples=64)
        else:
            oneshot(model=model, recipe=recipe)
        model.save_pretrained(save_dir, save_compressed=True)
        tok.save_pretrained(save_dir)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
