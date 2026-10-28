"""Build prefill/decode performance tables and kernel-category share tables from ph1 results."""

import csv
import json
import os
import sys

from analyze_phases import classify

R = "/dockerdata/sharq_work/results/ph1"
P = "/dockerdata/sharq_work/prof/ph1"
METHODS = ["BF16", "FP8", "W4A8", "NVFP4", "SHARQ_CUT", "SHARQ"]
NAMES = {"BF16": "BF16", "FP8": "FP8 W8A8", "W4A8": "W4A8", "NVFP4": "NVFP4",
         "SHARQ_CUT": "SharQ v3-CUTLASS", "SHARQ": "SharQ v3"}
CATS = ["gemm", "attention", "act_quant", "sharq_prep", "rmsnorm", "act", "rope_kvcache",
        "sampling", "input_prep", "other"]
CONFIGS = [("prefill", "prefill（bs=1, 8192 tokens ×3 次）"),
           ("decode4", "decode（bs=4, 128 steps）"),
           ("decode64", "decode（bs=64, 128 steps）")]


def load(path):
    with open(path) as f:
        return json.load(f)


def perf_tables():
    data = {m: load(f"{R}/phases_{m}.json") for m in METHODS if os.path.exists(f"{R}/phases_{m}.json")}
    out = []
    lens = sorted({int(k) for d in data.values() for k in d["prefill"]})
    out.append("#### Prefill（单请求，单位 tok/s / 每 token 耗时 µs）\n")
    out.append("| 长度 | " + " | ".join(NAMES[m] for m in METHODS) + " | SharQ/NVFP4 | SharQ/BF16 |")
    out.append("|" + "---|" * (len(METHODS) + 3))
    for L in lens:
        vals = {m: data[m]["prefill"][str(L)]["prefill_tok_per_s"] for m in METHODS if m in data}
        best = max(vals.values())
        cells = [f"**{vals[m]:.0f}**" if vals[m] == best else f"{vals[m]:.0f}" for m in METHODS]
        out.append(f"| {L} | " + " | ".join(cells) + f" | {vals['SHARQ'] / vals['NVFP4']:.3f}x | {vals['SHARQ'] / vals['BF16']:.2f}x |")
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


def share_tables():
    out = []
    for key, title in CONFIGS:
        rows = {}
        totals = {}
        for m in METHODS:
            path = f"{P}/{m}_{key}.kern.csv"
            if not os.path.exists(path):
                continue
            agg = {}
            tot = 0.0
            for r in csv.DictReader(open(path)):
                t = float(r["Total Time (ns)"])
                tot += t
                c = classify(r["Name"])
                agg[c] = agg.get(c, 0.0) + t
            rows[m] = agg
            totals[m] = tot
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
            path = f"{P}/{m}_{key}.kern.csv"
            if not os.path.exists(path):
                continue
            agg = {}
            for r in csv.DictReader(open(path)):
                t = float(r["Total Time (ns)"])
                c = classify(r["Name"])
                agg[c] = agg.get(c, 0.0) + t
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


if __name__ == "__main__":
    text = perf_tables() + "\n\n" + abs_tables() + "\n" + share_tables()
    print(text)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            f.write(text + "\n")
