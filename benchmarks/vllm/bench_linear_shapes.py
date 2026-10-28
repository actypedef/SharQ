"""Per-linear micro benchmark on Llama-3.1-8B shapes (CUDA-graph timed).

Compares: BF16 (torch.mm), vLLM NVFP4 (FlashInfer CUTLASS, the backend vLLM picks),
vLLM FP8 (cutlass scaled_mm, per-token act), and SharQ (sharq_ops_vllm).
Also prints a torch.profiler kernel breakdown for SharQ.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kernels" / "build_vllm_sm120a"))
import sharq_ops  # noqa: E402
import sharq_ops_vllm  # noqa: E402

SHAPES = {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (28672, 4096), "down": (4096, 14336)}
NVFP4_DENOM = 448.0 * 6.0


def graph_time(fn, iters=200):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    g.replay()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters // 10):
        g.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1000.0  # us


def make_sharq(w):
    scale_b = torch.clamp(w.abs().max().float() / NVFP4_DENOM, min=1e-9)
    qw, sfs, sfd = sharq_ops.quantize_w32_shared((w / scale_b).to(torch.bfloat16))
    return qw, sfs, sfd, float(scale_b)


def make_vllm_nvfp4(w):
    from vllm._custom_ops import scaled_fp4_quant
    from flashinfer import mm_fp4

    w_gs = (448.0 * 6.0) / w.abs().max().float()
    wq, wsf = scaled_fp4_quant(w, w_gs.view(1))
    def run(x):
        x_gs = (448.0 * 6.0) / x.abs().max().float()
        xq, xsf = scaled_fp4_quant(x, x_gs.view(1))
        alpha = 1.0 / (x_gs * w_gs)
        return mm_fp4(xq, wq.t(), xsf, wsf.t(), alpha, torch.bfloat16, backend="cutlass")
    return run


def make_vllm_fp8(w):
    from vllm import _custom_ops as ops

    wq, ws = ops.scaled_fp8_quant(w, use_per_token_if_dynamic=True)
    wq_t = wq.t()
    def run(x):
        xq, xs = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        return ops.cutlass_scaled_mm(xq, wq_t, out_dtype=torch.bfloat16, scale_a=xs, scale_b=ws.t().contiguous().t())
    return run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", default="4,64,256,2048,8192")
    p.add_argument("--profile-m", type=int, default=4)
    args = p.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    ms = [int(m) for m in args.ms.split(",")]
    print(f"{'layer':8s} {'M':>6s} | {'BF16':>9s} {'FP8':>9s} {'NVFP4':>9s} {'SharQ':>9s} | SharQ/NVFP4 (us)")
    totals = {}
    for m in ms:
        for name, (n, k) in SHAPES.items():
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            w = (torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16)
            qw, sfs, sfd, ws = make_sharq(w)
            nv = make_vllm_nvfp4(w)
            f8 = make_vllm_fp8(w)
            t = {
                "BF16": graph_time(lambda: torch.mm(x, w.t())),
                "FP8": graph_time(lambda: f8(x)),
                "NVFP4": graph_time(lambda: nv(x)),
                "SharQ": graph_time(lambda: sharq_ops_vllm.sharq_linear_forward(x, qw, sfs, sfd, ws)),
            }
            for kname, v in t.items():
                totals.setdefault(m, {}).setdefault(kname, 0.0)
                totals[m][kname] += v
            print(f"{name:8s} {m:6d} | {t['BF16']:9.1f} {t['FP8']:9.1f} {t['NVFP4']:9.1f} {t['SharQ']:9.1f} | {t['SharQ']/t['NVFP4']:.2f}x")
        tt = totals[m]
        print(f"{'layer-sum':8s} {m:6d} | {tt['BF16']:9.1f} {tt['FP8']:9.1f} {tt['NVFP4']:9.1f} {tt['SharQ']:9.1f} | {tt['SharQ']/tt['NVFP4']:.2f}x")
        print("-" * 80)

    # kernel breakdown for SharQ
    m = args.profile_m
    print(f"\nSharQ kernel breakdown at M={m}")
    from torch.profiler import ProfilerActivity, profile
    for name, (n, k) in SHAPES.items():
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        w = (torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16)
        qw, sfs, sfd, ws = make_sharq(w)
        for _ in range(5):
            sharq_ops_vllm.sharq_linear_forward(x, qw, sfs, sfd, ws)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                sharq_ops_vllm.sharq_linear_forward(x, qw, sfs, sfd, ws)
            torch.cuda.synchronize()
        print(f"--- {name} (N={n}, K={k})")
        for ev in sorted(prof.key_averages(), key=lambda e: -e.device_time_total)[:8]:
            if ev.device_time_total > 0:
                print(f"   {ev.device_time_total / 20:8.1f} us/iter  x{ev.count / 20:.0f}  {ev.key[:110]}")


if __name__ == "__main__":
    main()
