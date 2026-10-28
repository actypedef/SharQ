"""Classify GPU kernels from an nsys cuda_gpu_kern_sum CSV into functional buckets and print shares."""

import csv
import sys

# order matters: first match wins
RULES = [
    ("attention", ("flash::", "flash_fwd", "flash_attn", "flashinfer", "attention", "mla_")),
    ("sharq_prep", ("sharq_prepare", "absmax_bf16", "pair_add")),
    ("act_quant", ("cvt_fp16_to_fp4", "scaled_fp4_quant", "silu_mul_cvt", "dynamic_per_token",
                   "per_token_group_quant", "quant_fp8", "silu_and_mul_scaled")),
    ("gemm", ("cutlass::device_kernel", "cutlass::Kernel2", "nvjet", "gemm", "cublas", "wmma", "marlin::")),
    ("rmsnorm", ("rms_norm", "layer_norm")),
    ("act", ("silu_and_mul", "silu", "gelu", "act_and_mul")),
    ("rope_kvcache", ("rotary", "reshape_and_cache", "concat_and_cache", "kv_cache")),
    ("sampling", ("gumbel", "argmax", "sampling", "_topk", "top_k_top_p", "penalties", "logits")),
    ("input_prep", ("elementwise", "index_", "gather", "copy", "scatter", "cat_index_select",
                    "fill", "arange", "embedding", "memset", "Memcpy", "Memcpy")),
]


def classify(name: str) -> str:
    low = name.lower()
    for cat, keys in RULES:
        if any(k.lower() in low for k in keys):
            return cat
    return "other"


def main(path):
    rows = list(csv.DictReader(open(path)))
    total = sum(float(r["Total Time (ns)"]) for r in rows)
    agg = {}
    for r in rows:
        c = classify(r["Name"])
        t = float(r["Total Time (ns)"])
        n = int(r["Instances"])
        a = agg.setdefault(c, [0.0, 0])
        a[0] += t
        a[1] += n
    print(f"{path}\n  total {total / 1e6:.2f} ms, kernels {sum(v[1] for v in agg.values())}")
    for c, (t, n) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        print(f"  {c:14s} {t / 1e6:9.2f} ms  {t / total * 100:5.1f}%  n={n}")
    return total, agg


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
        print()
