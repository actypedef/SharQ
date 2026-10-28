"""Quick vLLM generation sanity check for a given model / quantization.

    python smoke_generate.py --model <checkpoint>            # one model
    python smoke_generate.py --baselines <quant_root>        # BF16 + FP8 + NVFP4 + W4A8 sweep

`--baselines` expects <quant_root>/{fp8_dynamic,nvfp4,w4a16}/<name> plus a BF16 model via
--base-model; each checkpoint runs in its own process (so GPU memory is released between them)
and the generations are echoed.
"""

import argparse
import json
import os
import subprocess
import sys
import time

from vllm import LLM, SamplingParams

PROMPTS = [
    "The capital of France is",
    "Explain in one sentence what a GPU is:",
    "1, 2, 3, 4, 5, 6,",
    "def fibonacci(n):",
]

# (label, sub-path under --baselines, extra env)
BASELINES = [
    ("FP8", "fp8_dynamic", {}),
    ("NVFP4", "nvfp4", {}),
    ("W4A8", "w4a16", {"VLLM_MARLIN_INPUT_DTYPE": "fp8"}),
]


def resolve_max_len(model: str, requested):
    """4096 unless the model's positional limit is smaller (keeps the 3 GiB smoke KV valid)."""
    if requested:
        return requested
    try:
        with open(os.path.join(model, "config.json")) as f:
            cap = int(json.load(f).get("max_position_embeddings", 4096))
    except Exception:
        cap = 4096
    return min(4096, cap)


def run_baselines(args):
    ckpts = [("BF16", args.base_model, {})] + [
        (label, os.path.join(args.baselines, sub, args.model_name), env) for label, sub, env in BASELINES]
    for label, path, env in ckpts:
        print("=" * 90)
        print(label, path, flush=True)
        cmd = [sys.executable, os.path.abspath(__file__), "--model", path,
               "--kv-cache-gib", str(args.kv_cache_gib), "--max-tokens", str(args.max_tokens)]
        if label == "BF16":
            cmd += ["--dtype", "bfloat16"]
        full_env = dict(os.environ, **env)
        t0 = time.time()
        proc = subprocess.run(cmd, env=full_env, capture_output=True, text=True)
        print(f"  exit={proc.returncode}  {time.time() - t0:.0f}s")
        for line in proc.stdout.splitlines():
            if line.startswith("[load]") or " -> " in line:
                print("  " + line[:150])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--baselines", default=None, help="quantized root dir; sweeps the standard baselines")
    p.add_argument("--base-model", default=None, help="BF16 model dir for the --baselines sweep")
    p.add_argument("--model-name", default="Llama-3.1-8B", help="sub-directory name under --baselines")
    p.add_argument("--quantization", default=None)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--kv-cache-gib", type=float, default=3.0)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-tokens", type=int, default=48)
    args = p.parse_args()
    if args.baselines:
        if not args.base_model:
            raise SystemExit("--baselines needs --base-model <BF16 model dir>")
        run_baselines(args)
        return
    if not args.model:
        raise SystemExit("either --model or --baselines is required")

    t0 = time.time()
    llm = LLM(
        model=args.model,
        quantization=args.quantization,
        dtype=args.dtype,
        max_model_len=resolve_max_len(args.model, args.max_model_len),
        max_num_seqs=64,
        kv_cache_memory_bytes=int(args.kv_cache_gib * (1 << 30)),
        gpu_memory_utilization=0.2,
        enforce_eager=args.enforce_eager,
        seed=0,
    )
    print(f"[load] {time.time() - t0:.1f}s")
    outs = llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
    for o in outs:
        print("=" * 80)
        print(repr(o.prompt), "->", repr(o.outputs[0].text))


if __name__ == "__main__":
    main()
