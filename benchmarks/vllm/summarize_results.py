"""Summarize end-to-end vLLM results (latency / throughput JSONs) and WikiText-2 PPL into markdown."""

import json
import os
import sys

R = "/dockerdata/sharq_work/results"
# (column label, json path)
RUNS = [
    ("BF16", f"{R}/r1/BF16.json"),
    ("FP8 W8A8", f"{R}/r1/FP8.json"),
    ("W4A8", f"{R}/r1/W4A8.json"),
    ("NVFP4", f"{R}/r1/NVFP4.json"),
    ("SharQ v1", f"{R}/v0/SHARQ.json"),
    ("SharQ v2", f"{R}/v2/SHARQ.json"),
    ("SharQ v3-CUTLASS", f"{R}/r1/SHARQ_CUT.json"),
    ("SharQ v3", f"{R}/r1/SHARQ.json"),
]
PPL = [("BF16", "BF16"), ("FP8 W8A8", "FP8"), ("W4A8", "W4A8"), ("NVFP4", "NVFP4"),
       ("SharQ v1", "SHARQ_v1"), ("SharQ v2", "SHARQ_v2"), ("SharQ v3", "SHARQ_v3")]


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
        best = [v for v in vals.values() if v is not None]
        best = min(best) if lower_better else max(best)
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


FULLLOAD = [("BF16", "BF16"), ("FP8 W8A8", "FP8"), ("W4A8", "W4A8"), ("NVFP4", "NVFP4"),
            ("SharQ v3-CUTLASS", "SHARQ_CUT"), ("SharQ v3", "SHARQ")]


def fullload_table():
    data = {l: load(f"{R}/fullload/{f}.json") for l, f in FULLLOAD}
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


def main():
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
        d = load(f"{R}/ppl/{f}.json")
        cells.append("—" if d is None else f"{d['ppl']:.4f}")
    out.append("| " + " | ".join(cells) + " |")
    text = "\n".join(out)
    print(text)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
