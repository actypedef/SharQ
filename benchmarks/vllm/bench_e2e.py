"""End-to-end vLLM benchmark (latency + throughput) for one quantization method.

Replicates the methodology of `vllm bench latency` / `vllm bench throughput`
(random-token prompts, ignore_eos, fixed output length) but keeps a single engine
alive for all sequence-length configs so torch.compile / CUDA-graph capture is
paid once per method.

Paper setting (SharQ, Sec. 4.3): Llama-3.1-8B, single GPU, batch size 4,
input = output length in {256, 512, 1024, 2048}.
"""

import argparse
import json
import os
import time

import numpy as np

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def make_prompts(rng, num, length, low=100, high=100000):
    return [TokensPrompt(prompt_token_ids=rng.integers(low, high, size=length).tolist()) for _ in range(num)]


def sampling(out_len):
    return SamplingParams(n=1, temperature=1.0, top_p=1.0, ignore_eos=True, max_tokens=out_len, detokenize=False)


def run_latency(llm, rng, batch, length, warmup, iters):
    prompts = make_prompts(rng, batch, length)
    sp = sampling(length)
    for _ in range(warmup):
        llm.generate(prompts, sp, use_tqdm=False)
    lat = []
    for _ in range(iters):
        prompts = make_prompts(rng, batch, length)
        t0 = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        lat.append(time.perf_counter() - t0)
    lat = np.array(lat)
    total_tokens = batch * 2 * length
    return {
        "avg_latency_s": float(lat.mean()),
        "p50_latency_s": float(np.percentile(lat, 50)),
        "latency_all_s": lat.tolist(),
        "total_tok_per_s": float(total_tokens / lat.mean()),
        "decode_tok_per_s": float(batch * length / lat.mean()),
    }


def metrics_snapshot(llm):
    """Counters needed to derive the effective batch: (#engine steps, generated tokens, preemptions)."""
    snap = {"steps": 0, "gen_tokens": 0, "preemptions": 0}
    try:
        for m in llm.get_metrics():
            if m.name == "vllm:iteration_tokens_total":
                snap["steps"] += int(m.count)
            elif m.name == "vllm:generation_tokens":
                snap["gen_tokens"] += int(m.value)
            elif m.name == "vllm:num_preemptions":
                snap["preemptions"] += int(m.value)
    except Exception:  # stats disabled
        return None
    return snap


def run_throughput(llm, rng, num_prompts, length):
    prompts = make_prompts(rng, num_prompts, length)
    sp = sampling(length)
    before = metrics_snapshot(llm)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    elapsed = time.perf_counter() - t0
    after = metrics_snapshot(llm)
    out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    in_tokens = sum(len(o.prompt_token_ids) for o in outs)
    r = {
        "num_prompts": num_prompts,
        "elapsed_s": elapsed,
        "req_per_s": num_prompts / elapsed,
        "total_tok_per_s": (in_tokens + out_tokens) / elapsed,
        "out_tok_per_s": out_tokens / elapsed,
    }
    if before is not None and after is not None:
        steps = after["steps"] - before["steps"]
        gen = after["gen_tokens"] - before["gen_tokens"]
        r["engine_steps"] = steps
        # every running request emits one token per step -> avg #sequences per step
        r["avg_seqs_per_step"] = gen / steps if steps else None
        r["preemptions"] = after["preemptions"] - before["preemptions"]
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--quantization", default=None)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--gpu-mem-util", type=float, default=0.26)
    p.add_argument("--kv-cache-gib", type=float, default=None,
                   help="Fixed KV-cache budget (overrides gpu-mem-util based sizing)")
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--lengths", default="256,512,1024,2048")
    p.add_argument("--latency-batch", type=int, default=4)
    p.add_argument("--latency-warmup", type=int, default=1)
    p.add_argument("--latency-iters", type=int, default=3)
    p.add_argument("--tp-prompts", default="256:512,512:256,1024:128,2048:64",
                   help="length:num_prompts for the throughput test")
    p.add_argument("--targets", default="latency,throughput")
    p.add_argument("--log-stats", action="store_true",
                   help="enable vLLM stats to record effective batch size / preemptions")
    p.add_argument("--tp-warmup", type=int, default=0,
                   help="number of 128/128 warmup prompts before the throughput runs")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    tp_prompts = {int(k): int(v) for k, v in (kv.split(":") for kv in args.tp_prompts.split(","))}
    targets = set(args.targets.split(","))

    t0 = time.time()
    llm = LLM(
        model=args.model,
        quantization=args.quantization,
        dtype=args.dtype,
        max_model_len=2 * max(lengths) + 64,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_mem_util,
        kv_cache_memory_bytes=None if args.kv_cache_gib is None else int(args.kv_cache_gib * (1 << 30)),
        enable_prefix_caching=False,
        enforce_eager=args.enforce_eager,
        disable_log_stats=not args.log_stats,
        seed=0,
    )
    load_s = time.time() - t0
    cache_cfg = llm.llm_engine.vllm_config.cache_config
    num_gpu_blocks = getattr(cache_cfg, "num_gpu_blocks", None)
    block_size = getattr(cache_cfg, "block_size", None)
    kv_tokens = num_gpu_blocks * block_size if num_gpu_blocks and block_size else None
    print(f"[{args.label}] engine ready in {load_s:.1f}s, KV capacity = {kv_tokens} tokens", flush=True)

    results = {
        "label": args.label,
        "model": args.model,
        "quantization": args.quantization,
        "env": {k: v for k, v in os.environ.items() if k.startswith(("VLLM_MARLIN", "SHARQ"))},
        "gpu_mem_util": args.gpu_mem_util,
        "kv_cache_gib": args.kv_cache_gib,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "kv_capacity_tokens": kv_tokens,
        "latency": {},
        "throughput": {},
    }
    rng = np.random.default_rng(0)
    if "throughput" in targets and args.tp_warmup > 0:
        llm.generate(make_prompts(rng, args.tp_warmup, 128), sampling(128), use_tqdm=False)
    for length in lengths:
        if "latency" in targets:
            r = run_latency(llm, rng, args.latency_batch, length, args.latency_warmup, args.latency_iters)
            results["latency"][str(length)] = r
            print(f"[{args.label}] latency bs={args.latency_batch} {length}/{length}: "
                  f"{r['avg_latency_s']:.3f}s  total {r['total_tok_per_s']:.1f} tok/s  "
                  f"decode {r['decode_tok_per_s']:.1f} tok/s", flush=True)
        if "throughput" in targets:
            r = run_throughput(llm, rng, tp_prompts[length], length)
            results["throughput"][str(length)] = r
            extra = ""
            if r.get("avg_seqs_per_step") is not None:
                extra = f"  avg_seqs/step {r['avg_seqs_per_step']:.1f}  preemptions {r['preemptions']}"
            print(f"[{args.label}] throughput {length}/{length} x{tp_prompts[length]}: "
                  f"{r['req_per_s']:.2f} req/s  total {r['total_tok_per_s']:.1f} tok/s  "
                  f"out {r['out_tok_per_s']:.1f} tok/s{extra}", flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    print(f"[{args.label}] saved {args.out}")


if __name__ == "__main__":
    main()
