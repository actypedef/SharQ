"""One report generator for all vLLM measurements.

    python summarize.py e2e    [out.md]    latency / throughput / full-load / PPL   (default)
    python summarize.py phases [out.md]    prefill & decode, plus kernel attribution
    python summarize.py kern <csv> [...]   classify a single nsys cuda_gpu_kern_sum CSV

Directories default to the layout used in this worktree and can be overridden with
SHARQ_RESULTS / SHARQ_PROF:

    $SHARQ_RESULTS/r1/*.json          end-to-end (bench_e2e.py via run_all.sh)
    $SHARQ_RESULTS/fullload/*.json    full-load throughput
    $SHARQ_RESULTS/ppl/*.json         WikiText-2 ppl (ppl_vllm.py)
    $SHARQ_RESULTS/ph1/*.json         prefill / decode (bench_phases.py via run_phases.sh)
    $SHARQ_PROF/<METHOD>_<cfg>.kern.csv   nsys kernel CSVs for the phase runs
"""

import csv
import json
import os
import sys

RESULTS = os.environ.get("SHARQ_RESULTS", "/dockerdata/sharq_work/results")
PHASES = os.path.join(RESULTS, "ph1")
PROF = os.environ.get("SHARQ_PROF", "/dockerdata/sharq_work/prof/ph1")

# ==============================================================================
# kernel classification (phases / kern)
# ==============================================================================

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
                    "fill", "arange", "embedding", "memset", "Memcpy")),
]


def classify(name: str) -> str:
    low = name.lower()
    for cat, keys in RULES:
        if any(k.lower() in low for k in keys):
            return cat
    return "other"


def kern_report(paths):
    for path in paths:
        rows = list(csv.DictReader(open(path)))
        total = sum(float(r["Total Time (ns)"]) for r in rows)
        agg = {}
        for r in rows:
            a = agg.setdefault(classify(r["Name"]), [0.0, 0])
            a[0] += float(r["Total Time (ns)"])
            a[1] += int(r["Instances"])
        print(f"{path}\n  total {total / 1e6:.2f} ms, kernels {sum(v[1] for v in agg.values())}")
        for c, (t, n) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
            print(f"  {c:14s} {t / 1e6:9.2f} ms  {t / total * 100:5.1f}%  n={n}")
        print()


# ==============================================================================
# end-to-end tables
# ==============================================================================

# (column label, json path); missing files are rendered as "—"
RUNS = [
    ("BF16", f"{RESULTS}/r1/BF16.json"),
    ("FP8 W8A8", f"{RESULTS}/r1/FP8.json"),
    ("W4A8", f"{RESULTS}/r1/W4A8.json"),
    ("NVFP4", f"{RESULTS}/r1/NVFP4.json"),
    ("SharQ v1", f"{RESULTS}/v0/SHARQ.json"),
    ("SharQ v2", f"{RESULTS}/v2/SHARQ.json"),
    ("SharQ v3-CUTLASS", f"{RESULTS}/r1/SHARQ_CUT.json"),
    ("SharQ v3", f"{RESULTS}/r1/SHARQ.json"),
]
PPL = [("BF16", "BF16"), ("FP8 W8A8", "FP8"), ("W4A8", "W4A8"), ("NVFP4", "NVFP4"),
       ("SharQ v1", "SHARQ_v1"), ("SharQ v2", "SHARQ_v2"), ("SharQ v3", "SHARQ_v3")]
FULLLOAD = [("BF16", "BF16"), ("FP8 W8A8", "FP8"), ("W4A8", "W4A8"), ("NVFP4", "NVFP4"),
            ("SharQ v3-CUTLASS", "SHARQ_CUT"), ("SharQ v3", "SHARQ")]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def table(data, section, key, fmt, ref="NVFP4", lower_better=True):
    labels = [l for l, _ in RUNS]
    lengths = sorted({int(k) for d in data.values() if d for k in d[section]})
    head = "| 长度 | " + " | ".join(labels) + f" | SharQ v3 / {ref} |"
    lines = [head, "|" + "---|" * (len(labels) + 2)]
    for L in lengths:
        vals = {l: (data[l][section].get(str(L), {}).get(key) if data[l] else None) for l in labels}
        cand = [v for v in vals.values() if v is not None]
        best = min(cand) if lower_better else max(cand)
        cells = []
        for l in labels:
            v = vals[l]
            s = "—" if v is None else fmt.format(v)
            cells.append(f"**{s}**" if v == best else s)
        ratio = vals["SharQ v3"] / vals[ref] if vals["SharQ v3"] and vals[ref] else float("nan")
        n = data["NVFP4"][section][str(L)].get("num_prompts") if section == "throughput" else None
        tag = f"{L}" + (f" × {n}" if n else "")
        lines.append(f"| {tag} | " + " | ".join(cells) + f" | {ratio:.3f}x |")
    return "\n".join(lines)


def fullload_table():
    data = {l: load(f"{RESULTS}/fullload/{f}.json") for l, f in FULLLOAD}
    if not data.get("NVFP4"):
        return ""
    labels = [l for l, _ in FULLLOAD]
    lines = ["| 长度 × 请求数 | 实际并发（序列/步） | 抢占次数 | " + " | ".join(labels)
             + " | SharQ v3 / NVFP4 | SharQ v3 / BF16 |",
             "|" + "---|" * (len(labels) + 5)]
    for L, r_nv in sorted(data["NVFP4"]["throughput"].items(), key=lambda kv: int(kv[0])):
        vals = {l: data[l]["throughput"][L]["total_tok_per_s"] if data[l] and L in data[l]["throughput"] else None
                for l in labels}
        best = max(v for v in vals.values() if v is not None)
        cells = ["—" if v is None else (f"**{v:.0f}**" if v == best else f"{v:.0f}") for v in vals.values()]
        pre = sorted({data[l]["throughput"][L].get("preemptions") for l in labels if data[l]})
        pre_s = str(pre[0]) if len(pre) == 1 else f"{pre[0]}～{pre[-1]}"
        lines.append(f"| {L} × {r_nv['num_prompts']} | {r_nv['avg_seqs_per_step']:.1f} | {pre_s} | "
                     + " | ".join(cells)
                     + f" | {vals['SharQ v3'] / vals['NVFP4']:.3f}x | {vals['SharQ v3'] / vals['BF16']:.2f}x |")
    cfg = data["NVFP4"]
    head = (f"KV cache {cfg['kv_cache_gib']} GiB（{cfg['kv_capacity_tokens']:,} tokens），"
            f"max_num_seqs={cfg['max_num_seqs']}，max_num_batched_tokens={cfg['max_num_batched_tokens']}")
    return "\n".join([head, "", *lines])


def e2e_tables():
    data = {l: load(p) for l, p in RUNS}
    out = ["### 延迟（batch=4，输入=输出长度，秒，越低越好）", "",
           table(data, "latency", "avg_latency_s", "{:.3f}"), "",
           "### 吞吐（KV 2.25 GiB，受 KV 容量限制的低并发，总 tok/s，越高越好）", "",
           table(data, "throughput", "total_tok_per_s", "{:.0f}", lower_better=False), "",
           "### 满负载吞吐（总 tok/s，越高越好）", "",
           fullload_table(), "",
           "### WikiText-2 PPL（seqlen 2048，vLLM 内计算，越低越好）", "",
           "| " + " | ".join(l for l, _ in PPL) + " |", "|" + "---|" * len(PPL)]
    cells = []
    for _, f in PPL:
        d = load(f"{RESULTS}/ppl/{f}.json")
        cells.append("—" if d is None else f"{d['ppl']:.4f}")
    out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ==============================================================================
# prefill / decode + kernel attribution
# ==============================================================================

METHODS = ["BF16", "FP8", "W4A8", "NVFP4", "SHARQ_CUT", "SHARQ"]
NAMES = {"BF16": "BF16", "FP8": "FP8 W8A8", "W4A8": "W4A8", "NVFP4": "NVFP4",
         "SHARQ_CUT": "SharQ v3-CUTLASS", "SHARQ": "SharQ v3"}
CATS = ["gemm", "attention", "act_quant", "sharq_prep", "rmsnorm", "act", "rope_kvcache",
        "sampling", "input_prep", "other"]
CONFIGS = [("prefill", "prefill（bs=1, 8192 tokens ×3 次）"),
           ("decode4", "decode（bs=4, 128 steps）"),
           ("decode64", "decode（bs=64, 128 steps）")]


def perf_tables():
    data = {m: load(f"{PHASES}/phases_{m}.json") for m in METHODS
            if os.path.exists(f"{PHASES}/phases_{m}.json")}
    out = []
    lens = sorted({int(k) for d in data.values() for k in d["prefill"]})
    out.append("#### Prefill（单请求，单位 tok/s / 每 token 耗时 µs）\n")
    out.append("| 长度 | " + " | ".join(NAMES[m] for m in METHODS) + " | SharQ/NVFP4 | SharQ/BF16 |")
    out.append("|" + "---|" * (len(METHODS) + 3))
    for L in lens:
        vals = {m: data[m]["prefill"][str(L)]["prefill_tok_per_s"] for m in METHODS if m in data}
        best = max(vals.values())
        cells = [f"**{vals[m]:.0f}**" if vals[m] == best else f"{vals[m]:.0f}" for m in METHODS]
        out.append(f"| {L} | " + " | ".join(cells)
                   + f" | {vals['SHARQ'] / vals['NVFP4']:.3f}x | {vals['SHARQ'] / vals['BF16']:.2f}x |")
    out.append("")
    out.append("#### Decode（ms/步，越低越好；括号内为 tok/s）\n")
    out.append("| batch | " + " | ".join(NAMES[m] for m in METHODS) + " | SharQ/NVFP4 |")
    out.append("|" + "---|" * (len(METHODS) + 2))
    batches = sorted({int(k) for d in data.values() for k in d["decode"]})
    for B in batches:
        vals = {m: data[m]["decode"][str(B)]["step_ms"] for m in METHODS if m in data}
        best = min(vals.values())
        cells = [f"**{vals[m]:.2f}**" if vals[m] == best else f"{vals[m]:.2f}" for m in METHODS]
        cells = [f"{c} ({data[m]['decode'][str(B)]['decode_tok_per_s']:.0f})" for c, m in zip(cells, METHODS)]
        out.append(f"| {B} | " + " | ".join(cells) + f" | {vals['SHARQ'] / vals['NVFP4']:.3f}x |")
    return "\n".join(out)


def _kern_agg(m):
    """per-category totals (ns) and grand total for one method/config"""
    agg, tot = {}, 0.0
    for r in csv.DictReader(open(f"{PROF}/{m}.kern.csv")):
        t = float(r["Total Time (ns)"])
        tot += t
        c = classify(r["Name"])
        agg[c] = agg.get(c, 0.0) + t
    return agg, tot


def share_tables():
    out = []
    for key, title in CONFIGS:
        rows, totals = {}, {}
        for m in METHODS:
            path = f"{PROF}/{m}_{key}.kern.csv"
            if not os.path.exists(path):
                continue
            rows[m], totals[m] = _kern_agg(f"{m}_{key}")
        if not rows:
            continue
        out.append(f"#### {title}\n")
        out.append("| 类别 | " + " | ".join(f"{NAMES[m]}<br>({totals[m] / 1e6:.1f} ms)" for m in rows) + " |")
        out.append("|" + "---|" * (len(rows) + 1))
        for c in CATS:
            cells = []
            for m in rows:
                v = rows[m].get(c, 0.0) / totals[m] * 100
                cells.append(f"{v:.1f}%" if v >= 0.05 else "—")
            if any(x != "—" for x in cells):
                out.append(f"| {c} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def abs_tables(iters=None):
    """Absolute GPU ms per prefill iteration / per decode step, by category."""
    iters = iters or {"prefill": 3, "decode4": 128, "decode64": 128}
    out = []
    for key, title in CONFIGS:
        rows, tot = {}, {}
        for m in METHODS:
            if not os.path.exists(f"{PROF}/{m}_{key}.kern.csv"):
                continue
            agg, _ = _kern_agg(f"{m}_{key}")
            rows[m] = {c: v / iters[key] / 1e6 for c, v in agg.items()}
            tot[m] = sum(rows[m].values())
        if not rows:
            continue
        out.append(f"#### {title} — 每次迭代的 GPU 时间（ms）\n")
        out.append("| 类别 | " + " | ".join(NAMES[m] for m in rows) + " | SharQ−NVFP4 |")
        out.append("|" + "---|" * (len(rows) + 2))
        out.append("| **合计** | " + " | ".join(f"**{tot[m]:.2f}**" for m in rows)
                   + f" | **{tot['SHARQ'] - tot['NVFP4']:+.2f}** |")
        for c in CATS:
            vals = {m: rows[m].get(c, 0.0) for m in rows}
            if max(vals.values()) < 1e-3:
                continue
            cells = [f"{vals[m]:.2f}" if vals[m] >= 1e-3 else "—" for m in rows]
            delta = vals["SHARQ"] - vals["NVFP4"]
            out.append(f"| {c} | " + " | ".join(cells) + f" | {delta:+.2f} |")
        out.append("")
    return "\n".join(out)


def phase_tables():
    return perf_tables() + "\n\n" + abs_tables() + "\n" + share_tables()


# ==============================================================================

def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "e2e"
    if what == "kern":
        kern_report(sys.argv[2:])
        return
    if what not in ("e2e", "phases"):
        raise SystemExit(__doc__)
    text = e2e_tables() if what == "e2e" else phase_tables()
    print(text)
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
