"""WikiText-2 PPL with quantization applied to activations only / weights only / both.

Weights: 4-bit RTN, group 32 (the SharQ checkpoint format, no GPTQ/AWQ).
Activations: SharQ sparse-32g + residual-16g fake quant (model/quantize.py simulators).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "model"))
from quantize import (  # noqa: E402
    quantize_activation_sharq_sim,
    quantize_nvfp4_tensor,
    quantize_weight_sharq_sim,
)


def q_act(x):
    xs, xr, scale = quantize_activation_sharq_sim(x)
    return ((xs.float() + xr.float()) * scale).to(x.dtype)


def q_act_nvfp4(x):
    from quantize import global_nvfp4_scale
    scale = global_nvfp4_scale(x)
    return (quantize_nvfp4_tensor(x.float() / scale, group_size=16) * scale).to(x.dtype)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/dockerdata/sharq_work/models/Llama-3.1-8B")
    p.add_argument("--ids", default="/dockerdata/sharq_work/results/ppl/wikitext2_test_llama3_ids.json")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--windows", type=int, default=20)
    p.add_argument("--weight-scheme", default="g32", choices=["g32", "g16", "fp8_g32"],
                   help="g32: 4-bit RTN group 32 (SharQ checkpoint); g16: 4-bit group 16; "
                        "fp8_g32: FP8 E4M3 with group-32 scales (upper bound)")
    p.add_argument("--act-scheme", default="sharq", choices=["sharq", "nvfp4"],
                   help="sharq: sparse-32g + residual-16g (7 bits); nvfp4: dense 4-bit g16 (4.5 bits)")
    p.add_argument("--act", action="store_true", help="also apply activation quantization (both)")
    p.add_argument("--skip-act-only", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    def q_w(w):
        w32 = w.float()
        if args.weight_scheme == "g32":
            wq, ws = quantize_weight_sharq_sim(w)
            return (wq.float() * ws).to(w.dtype)
        if args.weight_scheme == "g16":
            from quantize import global_nvfp4_scale
            scale = global_nvfp4_scale(w32)
            return (quantize_nvfp4_tensor(w32 / scale, group_size=16) * scale).to(w.dtype)
        # fp8 e4m3 with group-32 scales
        flat = w32.reshape(-1, 32)
        scale = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-9) / 448.0
        q = (flat / scale).to(torch.float8_e4m3fn).float() * scale
        return q.reshape(w.shape).to(w.dtype)

    dev = "cuda"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev).eval()
    ids = json.load(open(args.ids))
    wins = [torch.tensor([ids[i * args.seqlen:(i + 1) * args.seqlen]], device=dev) for i in range(args.windows)]

    def ppl():
        nll, cnt = 0.0, 0
        with torch.no_grad():
            for w in wins:
                out = model(w, labels=w)
                nll += out.loss.float().item() * (w.shape[1] - 1)
                cnt += w.shape[1] - 1
        return math.exp(nll / cnt)

    results = {}
    t0 = time.time()
    results["bf16"] = ppl()
    print(f"bf16           ppl = {results['bf16']:.4f}   ({time.time() - t0:.0f}s)", flush=True)

    # ---- activations only ----
    def add_act_hooks():
        hs = []
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "Linear" and any(
                    k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")):
                fn = q_act if args.act_scheme == "sharq" else q_act_nvfp4
                hs.append(mod.register_forward_pre_hook(lambda m, inp, f=fn: (f(inp[0]),)))
        return hs

    handles = []
    if not args.skip_act_only:
        handles = add_act_hooks()
        t0 = time.time()
        results["act_only"] = ppl()
        print(f"act only ({args.act_scheme}) ppl = {results['act_only']:.4f}   ({time.time() - t0:.0f}s)", flush=True)
        for h in handles:
            h.remove()

    # ---- weights only, then both ----
    t0 = time.time()
    with torch.no_grad():
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "Linear" and any(k in name for k in ("q_proj", "k_proj", "v_proj",
                                                                             "o_proj", "gate_proj", "up_proj",
                                                                             "down_proj")):
                mod.weight.data = q_w(mod.weight.data)
    print(f"weight dequant done ({time.time() - t0:.0f}s)", flush=True)

    t0 = time.time()
    results["w_only"] = ppl()
    print(f"weight only    ppl = {results['w_only']:.4f}   ({time.time() - t0:.0f}s)", flush=True)

    if args.act:
        handles = add_act_hooks()
        t0 = time.time()
        results["both"] = ppl()
        print(f"both           ppl = {results['both']:.4f}   ({time.time() - t0:.0f}s)", flush=True)

    b = results["bf16"]
    print("\nscheme=%s  delta vs bf16:  w_only %+.4f" % (args.weight_scheme, results["w_only"] - b), end="")
    if "act_only" in results:
        print("   act_only %+.4f" % (results["act_only"] - b), end="")
    if "both" in results:
        print("   both %+.4f" % (results["both"] - b), end="")
    print()
    sq_act = max(results.get("act_only", b) - b, 0.0) ** 2
    sq_w = max(results["w_only"] - b, 0.0) ** 2
    tot = sq_act + sq_w
    if tot > 0 and "act_only" in results:
        print(f"share of squared ppl delta:  weight side {sq_w / tot * 100:.1f}%   activation side {sq_act / tot * 100:.1f}%")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({**results, "windows": args.windows, "seqlen": args.seqlen}, f, indent=2)


if __name__ == "__main__":
    main()
