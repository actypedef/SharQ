"""Fair, L2-cold decode-step benchmark (L layers x {qkv,o,gate_up,down}, distinct weights,
CUDA-graph captured), with FlashInfer autotuning applied to every FlashInfer GEMM.

  NVFP4_VLLM : vLLM's NVFP4 linear path (static input global scale, scaled_fp4_quant, FlashInfer
               CUTLASS mm_fp4) -- what vLLM actually runs.
  SHARQ_CUT  : SharQ v2 CUTLASS kernels (auto plan).
  SHARQ_FI   : SharQ stacked operand (sharq_prepare_stacked_fast) + FlashInfer mm_fp4 on 2M rows
               + pair add. Same numerics as SharQ.
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


def nvfp4_vllm(w, x_calib):
    w_gs = (NV / w.abs().max().float()).view(1)
    wq, wsf = scaled_fp4_quant(w, w_gs)
    x_gs = (NV / x_calib.abs().max().float()).view(1)  # static (calibrated) input scale
    alpha = (1.0 / (x_gs * w_gs)).float().view(1)

    def run(x):
        xq, xsf = scaled_fp4_quant(x, x_gs)
        return mm_fp4(xq, wq.t(), xsf, wsf.t(), alpha, torch.bfloat16, backend="cutlass")
    return run


def sharq_weights(w):
    sb = float(torch.clamp(w.abs().max().float() / NV, min=1e-9))
    qw, sfs, sfd = sharq_ops.quantize_w32_shared((w / sb).to(torch.bfloat16))
    n, k = w.shape
    npad = (n + 127) // 128 * 128
    sfd_fi = sfd[: npad * k // 16].view(torch.float8_e4m3fn).view(npad, k // 16)
    return qw, sfs, sfd, sb, sfd_fi


def sharq_fi_forward(x, qw, sfd_fi, sb):
    q2, sf2, alpha = sharq_ops_vllm.sharq_prepare_stacked_fast(x, sb)
    y2 = mm_fp4(q2, qw.t(), sf2, sfd_fi.t(), alpha, torch.bfloat16, backend="cutlass")
    return sharq_ops_vllm.sharq_pair_add(y2)


def graph_time(step, iters=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--ms", default="1,4,8,16,32,64,128,256")
    p.add_argument("--methods", default="NVFP4_VLLM,SHARQ_CUT,SHARQ_FI")
    args = p.parse_args()
    torch.manual_seed(0)
    ms = [int(v) for v in args.ms.split(",")]
    methods = args.methods.split(",")
    xs = {m: {n_: torch.randn(m, k, device="cuda", dtype=torch.bfloat16) for n_, (n, k) in SHAPES.items()} for m in ms}
    res = {}
    for meth in methods:
        layers = []
        for _ in range(args.layers):
            layer = {}
            for name, (n, k) in SHAPES.items():
                w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
                layer[name] = nvfp4_vllm(w, xs[ms[0]][name]) if meth == "NVFP4_VLLM" else sharq_weights(w)
                del w
            layers.append(layer)
        torch.cuda.empty_cache()

        def make_step(m):
            fns = []
            for layer in layers:
                for name in SHAPES:
                    x, obj = xs[m][name], layer[name]
                    if meth == "NVFP4_VLLM":
                        fns.append(lambda f=obj, x=x: f(x))
                    elif meth == "SHARQ_CUT":
                        qw, sfs, sfd, sb, _ = obj
                        fns.append(lambda x=x, qw=qw, sfs=sfs, sfd=sfd, sb=sb:
                                   sharq_ops_vllm.sharq_linear_forward(x, qw, sfs, sfd, sb))
                    else:
                        qw, _, _, sb, sfd_fi = obj
                        fns.append(lambda x=x, qw=qw, sfd_fi=sfd_fi, sb=sb: sharq_fi_forward(x, qw, sfd_fi, sb))

            def step():
                for f in fns:
                    f()
            return step

        res[meth] = {}
        for m in ms:
            step = make_step(m)
            if meth != "SHARQ_CUT":
                with torch.inference_mode(), autotune(True):
                    step()
                torch.cuda.synchronize()
            res[meth][m] = graph_time(step)
            print(f"{meth:11s} M={m:5d}: {res[meth][m]:7.3f} ms / {args.layers} layers", flush=True)
        del layers
        torch.cuda.empty_cache()

    print("\nM     " + " ".join(f"{mm:>11s}" for mm in methods) + "  | SHARQ_FI/NVFP4  SHARQ_CUT/NVFP4")
    for m in ms:
        row = " ".join(f"{res[mm][m]:11.3f}" for mm in methods)
        ex = ""
        if "NVFP4_VLLM" in res:
            for mm in ("SHARQ_FI", "SHARQ_CUT"):
                if mm in res:
                    ex += f"   {res[mm][m] / res['NVFP4_VLLM'][m]:6.2f}x"
        print(f"{m:<5d} {row}  |{ex}")


if __name__ == "__main__":
    main()
