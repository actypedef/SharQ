"""Quick vLLM generation sanity check for a given model / quantization."""

import argparse
import time

from vllm import LLM, SamplingParams

PROMPTS = [
    "The capital of France is",
    "Explain in one sentence what a GPU is:",
    "1, 2, 3, 4, 5, 6,",
    "def fibonacci(n):",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--quantization", default=None)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--kv-cache-gib", type=float, default=3.0)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-tokens", type=int, default=48)
    args = p.parse_args()

    t0 = time.time()
    llm = LLM(
        model=args.model,
        quantization=args.quantization,
        dtype=args.dtype,
        max_model_len=4096,
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
