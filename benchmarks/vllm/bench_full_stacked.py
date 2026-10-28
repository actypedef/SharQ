"""Compare execution strategies for the SharQ linear across M (CUDA-graph timed).

  AUTO      : what v3 ships (stacked+FlashInfer for M <= 192/256, else sparse+dense CUTLASS)
  STACKED_FI: always [A_sparse; A_res] (2M rows) in ONE FlashInfer NVFP4 GEMM + pair add
  STACKED_CU: same operand, CUTLASS dense GEMM (128x128x128 / 128x128x256 / stream-K)
  TWOG_FI   : sparse GEMM (CUTLASS 4:8) + dense residual GEMM (FlashInfer, epilogue-free add)
  TWOG_CU   : sparse GEMM + dense residual GEMM, both CUTLASS (the paper's kernel)

Also reports vLLM's NVFP4 linear (static scale + FlashInfer) and FP8 as references.
"""

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
BUILD = Path(os.environ.get("SHARQ_VLLM_KERNEL_BUILD", ROOT / "kernels" / "build_vllm2_sm120a"))
sys.path.insert(0, str(ROOT / "kernels" / "build_vllm_sm120a"))
sys.path.insert(0, str(BUILD))
import sharq_ops  # noqa: E402
import sharq_ops_vllm  # noqa: E402
from flashinfer import mm_fp4  # noqa: E402
from flashinfer.autotuner import autotune  # noqa: E402
from vllm._custom_ops import scaled_fp4_quant  # noqa: E402

SHAPES = {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (28672, 4096), "down": (4096, 14336)}
NV = 448.0 * 6.0


def make_nvfp4(w, x_calib):
    w_gs = (NV / w.abs().max().float()).view(1)
    wq, wsf = scaled_fp4_quant(w, w_gs)
    x_gs = (NV / x_calib.abs().max().float()).view(1)
    alpha = (1.0 / (x_gs * w_gs)).float().view(1)
    b = wq.t()
    bsf = wsf.t()

    def run(x):
        xq, xsf = scaled_fp4_quant(x, x_gs)
        return mm_fp4(xq, b, xsf, bsf, alpha, torch.bfloat16, backend="cutlass")
    return run


def make_fp8(w):
    from vllm import _custom_ops as ops
    wq, ws = ops.scaled_fp8_quant(w, use_per_token_if_dynamic=True)
    b, bsf = wq.t(), ws.t().contiguous().t()

    def run(x):
        xq, xs = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        return ops.cutlass_scaled_mm(xq, b, out_dtype=torch.bfloat16, scale_a=xs, scale_b=bsf)
    return run


def make_sharq(w):
    sb = float(torch.clamp(w.abs().max().float() / NV, min=1e-9))
    qw, sfs, sfd = sharq_ops.quantize_w32_shared((w / sb).to(torch.bfloat16))
    n, k = w.shape
    npad = (n + 127) // 128 * 128
    sfd_fi = sfd[: npad * k // 16].view(torch.float8_e4m3fn).view(npad, k // 16)
    return {"qw": qw, "sfs": sfs, "sfd": sfd, "sb": sb, "sfd_fi": sfd_fi, "n": n, "k": k}


def stacked_fi(x, p):
    q2, sf2, alpha = sharq_ops_vllm.sharq_prepare_stacked_fast(x, p["sb"])
    y2 = mm_fp4(q2, p["qw"].t(), sf2, p["sfd_fi"].t(), alpha, torch.bfloat16, backend="cutlass")
    return sharq_ops_vllm.sharq_pair_add(y2)


def stacked_cu(x, p):
    m = x.shape[0]
    dv = 1 if (p["k"] >= 8192 and 2 * m <= 192) else (2 if 2 * m <= 128 else 0)
    return sharq_ops_vllm.sharq_linear_forward_cfg(x, p["qw"], p["sfs"], p["sfd"], p["sb"], 1, dv, 0, 1)


def twog_cu(x, p):
    m = x.shape[0]
    sk = 1 if (p["k"] >= 8192 and m <= 1024) else 0
    return sharq_ops_vllm.sharq_linear_forward_cfg(x, p["qw"], p["sfs"], p["sfd"], p["sb"], 0, sk, sk, 1)


def twog_fi(x, p):
    """Sparse GEMM (CUTLASS) into y, then residual GEMM via FlashInfer with a following add."""
    m = x.shape[0]
    y = sharq_ops_vllm.sharq_linear_forward_cfg(x, p["qw"], p["sfs"], p["sfd"], p["sb"], 0, 1, 1, 1)
    q2, sf2, alpha = sharq_ops_vllm.sharq_prepare_stacked_fast(x, p["sb"])
    q_res = q2[m:]                       # residual half only
    sf_res = sf2[m:]                     # its 16-group scales (rows are aligned to 128 only when m%128==0)
    if m % 128:
        return y                       # fallback: not comparable, skip
    yr = mm_fp4(q_res, p["qw"].t(), sf_res, p["sfd_fi"].t(), alpha, torch.bfloat16, backend="cutlass")
    return y + yr


def graph_time(fn, iters=30):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1e3  # us


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", default="1,4,16,64,128,192,256,384,512,1024,2048,4096,8192")
    p.add_argument("--shapes", default="qkv,o,gate_up,down")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    torch.manual_seed(0)
    ms = [int(v) for v in args.ms.split(",")]
    shapes = args.shapes.split(",")
    variants = ["NVFP4", "FP8", "AUTO", "STACKED_FI", "STACKED_CU", "TWOG_CU"]
    print(f"{'layer':8s} {'M':>5s} | " + " ".join(f"{v:>10s}" for v in variants) + " | STK_FI/AUTO STK_FI/TWOG_CU STK_FI/NVFP4")
    rows = []
    for m in ms:
        tot = {v: 0.0 for v in variants}
        for name in shapes:
            n, k = SHAPES[name]
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
            sh = make_sharq(w)
            nv = make_nvfp4(w, x)
            f8 = make_fp8(w)
            fns = {
                "NVFP4": lambda: nv(x),
                "FP8": lambda: f8(x),
                "AUTO": lambda: sharq_ops_vllm.sharq_linear_forward(x, sh["qw"], sh["sfs"], sh["sfd"], sh["sb"]),
                "STACKED_FI": lambda: stacked_fi(x, sh),
                "STACKED_CU": lambda: stacked_cu(x, sh),
                "TWOG_CU": lambda: twog_cu(x, sh),
            }
            t = {}
            for v in variants:
                try:
                    t[v] = graph_time(fns[v])
                except Exception as e:  # noqa: BLE001
                    t[v] = float("nan")
                    print(f"  [{v} {name} M={m}] {type(e).__name__}: {str(e)[:90]}")
            for v in variants:
                tot[v] += t[v]
            print(f"{name:8s} {m:5d} | " + " ".join(f"{t[v]:10.1f}" for v in variants)
                  + f" | {t['STACKED_FI'] / t['AUTO']:10.2f} {t['STACKED_FI'] / t['TWOG_CU']:12.2f} {t['STACKED_FI'] / t['NVFP4']:11.2f}")
            del w, sh, nv, f8
            torch.cuda.empty_cache()
        print("-" * 120)
        print(f"{'sum':8s} {m:5d} | " + " ".join(f"{tot[v]:10.1f}" for v in variants)
              + f" | {tot['STACKED_FI'] / tot['AUTO']:10.2f} {tot['STACKED_FI'] / tot['TWOG_CU']:12.2f} {tot['STACKED_FI'] / tot['NVFP4']:11.2f}")
        rows.append({"M": m, **{v: tot[v] for v in variants}})
        if args.out:
            import json
            with open(args.out, "w") as f:
                json.dump(rows, f, indent=2)


if __name__ == "__main__":
    with torch.inference_mode(), autotune(True):
        main()
