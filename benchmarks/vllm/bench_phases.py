"""Measure prefill and decode separately on one vLLM engine.

prefill : one batch of `length`-token prompts with max_tokens=1 -> TTFT (prefill dominates)
decode  : same batch with a 16-token prompt; time(max_tokens=1+S) - time(max_tokens=1)
          = pure decode time for S steps (prefill cancelled out)
"""

import argparse
import json
import time

import numpy as np

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def make_prompts(rng, num, length, low=100, high=100000):
    return [TokensPrompt(prompt_token_ids=rng.integers(low, high, size=length).tolist()) for _ in range(num)]


def sp(max_tokens):
    return SamplingParams(n=1, temperature=1.0, top_p=1.0, ignore_eos=True, max_tokens=max_tokens, detokenize=False)


def timed(llm, prompts, max_tokens, iters=1):
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        llm.generate(prompts, sp(max_tokens), use_tqdm=False)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--quantization", default=None)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--kv-cache-gib", type=float, default=40.0)
    p.add_argument("--gpu-mem-util", type=float, default=0.9)
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--prefill-lens", default="1024,2048,4096,8192")
    p.add_argument("--prefill-batch", type=int, default=1)
    p.add_argument("--prefill-iters", type=int, default=2)
    p.add_argument("--decode-batches", default="1,4,16,64,256")
    p.add_argument("--decode-steps", type=int, default=128)
    p.add_argument("--decode-prompt", type=int, default=16)
    p.add_argument("--targets", default="prefill,decode")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    targets = set(args.targets.split(","))
    lens = [int(v) for v in args.prefill_lens.split(",")]
    batches = [int(v) for v in args.decode_batches.split(",")]
    max_len = max(max(lens), args.decode_prompt + args.decode_steps + 16)

    llm = LLM(model=args.model, quantization=args.quantization, dtype=args.dtype,
              max_model_len=max_len + 64, max_num_seqs=args.max_num_seqs,
              max_num_batched_tokens=args.max_num_batched_tokens,
              gpu_memory_utilization=args.gpu_mem_util,
              kv_cache_memory_bytes=int(args.kv_cache_gib * (1 << 30)),
              enable_prefix_caching=False, seed=0)
    cfg = llm.llm_engine.vllm_config.cache_config
    kv = cfg.num_gpu_blocks * cfg.block_size
    print(f"[{args.label}] engine ready, KV {kv} tokens", flush=True)

    res = {"label": args.label, "model": args.model, "kv_tokens": kv, "prefill": {}, "decode": {}}
    rng = np.random.default_rng(0)

    if "prefill" in targets:
        for L in lens:
            prompts = make_prompts(rng, args.prefill_batch, L)
            timed(llm, prompts, 1)  # warmup / compile
            t = timed(llm, prompts, 1, args.prefill_iters)
            r = {
                "batch": args.prefill_batch,
                "length": L,
                "ttft_s": t,
                "prefill_tok_per_s": args.prefill_batch * L / t,
                "us_per_token_per_seq": t / L * 1e6,
            }
            res["prefill"][str(L)] = r
            print(f"[{args.label}] prefill bs={args.prefill_batch} L={L:5d}: {t*1e3:8.1f} ms  "
                  f"{r['prefill_tok_per_s']:9.0f} tok/s  {r['us_per_token_per_seq']:6.1f} us/token/seq", flush=True)

    if "decode" in targets:
        for B in batches:
            prompts = make_prompts(rng, B, args.decode_prompt)
            timed(llm, prompts, 1)
            t1 = timed(llm, prompts, 1)
            t2 = timed(llm, prompts, 1 + args.decode_steps)
            dec = t2 - t1
            step = dec / args.decode_steps
            r = {
                "batch": B,
                "steps": args.decode_steps,
                "prefill_ref_s": t1,
                "decode_s": dec,
                "step_ms": step * 1e3,
                "decode_tok_per_s": B / step,
                "ms_per_token_per_seq": step * 1e3,
            }
            res["decode"][str(B)] = r
            print(f"[{args.label}] decode  bs={B:4d}: {step*1e3:7.2f} ms/step  {r['decode_tok_per_s']:9.0f} tok/s", flush=True)
            with open(args.out, "w") as f:
                json.dump(res, f, indent=2)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{args.label}] saved {args.out}")


if __name__ == "__main__":
    main()
