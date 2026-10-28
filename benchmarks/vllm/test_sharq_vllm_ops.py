"""Correctness check: sharq_ops_vllm vs the original sharq_ops pipeline.

Run inside the sharq-vllm env:
    SHARQ_VLLM_KERNEL_BUILD=kernels/build_vllm2_sm120a CUDA_VISIBLE_DEVICES=0 \
        python benchmarks/vllm/test_sharq_vllm_ops.py
"""

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
BUILD = Path(os.environ.get("SHARQ_VLLM_KERNEL_BUILD", ROOT / "kernels" / "build_vllm2_sm120a"))
sys.path.insert(0, str(ROOT / "kernels" / "build_vllm_sm120a"))  # original sharq_ops (torch 2.11 build)
sys.path.insert(0, str(BUILD))

import sharq_ops  # noqa: E402
import sharq_ops_vllm  # noqa: E402

NVFP4_DENOM = 448.0 * 6.0


def reference_forward(x, qw, sfw_sparse, sfw_dense, weight_scale):
    """Original fallback path from benchmarks/vllm/sharq.py."""
    scale_a = torch.clamp(x.abs().max().float() / NVFP4_DENOM, min=1e-9).view(1)
    x_scaled = (x / scale_a).to(torch.bfloat16).contiguous()
    n = int(qw.shape[0])
    a_comp, e, sfa, q_res, sf_res = sharq_ops.fused_sparse_residual_quantize_x(x_scaled, n)
    y = sharq_ops.sparse_matmul(a_comp, qw, e, sfa, sfw_sparse, x.shape[0], n, x.shape[1], alpha=weight_scale)
    out = sharq_ops.matmul_accum(q_res, qw, sf_res, sfw_dense, weight_scale, y, 1.0)
    torch.cuda.synchronize()
    return out.float() * scale_a, (a_comp, e, sfa, q_res, sf_res, scale_a, x_scaled)


def quantize_weight(w):
    scale_b = torch.clamp(w.abs().max().float() / NVFP4_DENOM, min=1e-9)
    qw, sfs, sfd = sharq_ops.quantize_w32_shared((w / scale_b).to(torch.bfloat16))
    return qw, sfs, sfd, float(scale_b)


def rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def check_stacked(x, n, ref_bufs):
    """Stacked operand must be the dense expansion of the sparse operand + the residual."""
    m, k = x.shape
    q_st, sf_st, _ = sharq_ops_vllm.sharq_prepare_stacked(x)
    _, _, _, r_q, r_sf, _, x_scaled = ref_bufs
    # residual rows must equal the reference residual payload
    res_ok = torch.equal(q_st[m:], r_q)
    # sparse rows must equal the reference raw (uncompressed) sparse payload
    q_sparse_dense, sfa_sparse, _, _ = sharq_ops.fused_sparse_residual_quantize_x_debug(x_scaled, n)
    sp_ok = torch.equal(q_st[:m], q_sparse_dense)
    # scales: dequantize both halves through the duplicated layout and compare against reference
    sf_dup = sharq_ops.duplicate_sfa32_to_sfa16(sfa_sparse, m, n, k)
    w_eye_ok = True
    return res_ok and sp_ok, sf_dup


def stress_prepare_bitexact():
    """Heavy-tailed / tiny / zero-heavy inputs: prepare buffers must match the original kernel."""
    ok = True
    dev = "cuda"
    gens = {
        "heavy_tail": lambda m, k: torch.randn(m, k, device=dev) * torch.exp(torch.randn(m, k, device=dev) * 2),
        "outlier_ch": lambda m, k: torch.randn(m, k, device=dev) * (1 + 200 * (torch.rand(1, k, device=dev) > 0.995)),
        "tiny": lambda m, k: torch.randn(m, k, device=dev) * 1e-3 * (1 + 1e4 * (torch.rand(m, k, device=dev) > 0.9999)),
        "sparse_zero": lambda m, k: torch.randn(m, k, device=dev) * (torch.rand(m, k, device=dev) > 0.7),
        "ties": lambda m, k: torch.randint(-8, 9, (m, k), device=dev).float() * 0.25,
    }
    for seed in range(3):
        torch.manual_seed(100 + seed)
        for gname, g in gens.items():
            for m, k in ((3, 4096), (64, 14336), (513, 4096)):
                x = g(m, k).to(torch.bfloat16)
                n = 4096
                scale = torch.clamp(x.abs().max().float() / NVFP4_DENOM, min=1e-9).view(1)
                xs = (x / scale).to(torch.bfloat16).contiguous()
                r = sharq_ops.fused_sparse_residual_quantize_x(xs, n)
                v = sharq_ops_vllm.sharq_prepare(x, n)
                same = (torch.equal(v[0][: m * k // 4], r[0][: m * k // 4]) and torch.equal(v[1], r[1])
                        and torch.equal(v[2], r[2]) and torch.equal(v[3], r[3]) and torch.equal(v[4], r[4]))
                q_dbg, sfa_dbg, _, _ = sharq_ops.fused_sparse_residual_quantize_x_debug(xs, n)
                q_st, _, _ = sharq_ops_vllm.sharq_prepare_stacked(x)
                same &= torch.equal(q_st[:m], q_dbg) and torch.equal(q_st[m:], r[3])
                if not same:
                    print(f"  stress FAIL seed={seed} {gname} M={m} K={k}")
                ok &= same
    print(f"prepare bit-exact stress test: {'OK' if ok else 'FAIL'}")
    return ok


def main():
    torch.manual_seed(0)
    dev = "cuda"
    ok = stress_prepare_bitexact()
    shapes = [(1, 6144, 4096), (4, 6144, 4096), (37, 4096, 4096), (64, 28672, 4096),
              (128, 28672, 4096), (300, 4096, 14336), (2048, 6144, 4096)]
    plans = [  # (name, mode, dense_variant, sparse_variant)
        ("two/d0/s0", 0, 0, 0), ("two/d1/s1", 0, 1, 1),
        ("stk/d0", 1, 0, 0), ("stk/d1", 1, 1, 0), ("stk/d2", 1, 2, 0), ("stk/d3", 1, 3, 0),
    ]
    for m, n, k in shapes:
        x = (torch.randn(m, k, device=dev) * torch.rand(1, k, device=dev) * 4).to(torch.bfloat16)
        w = (torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16)
        qw, sfs, sfd, ws = quantize_weight(w)
        ref, ref_bufs = reference_forward(x, qw, sfs, sfd, ws)
        bf16_ref = x.float() @ w.float().t()

        a_comp, e, sfa, q_res, sf_res, act_scale = sharq_ops_vllm.sharq_prepare(x, n)
        r_a, r_e, r_sfa, r_q, r_sf, r_scale, _ = ref_bufs
        same = all([
            torch.equal(a_comp[: m * k // 4], r_a[: m * k // 4]), torch.equal(e, r_e), torch.equal(sfa, r_sfa),
            torch.equal(q_res, r_q), torch.equal(sf_res, r_sf), torch.allclose(act_scale, r_scale),
        ])
        stacked_ok, _ = check_stacked(x, n, ref_bufs)
        line = f"M={m:5d} N={n:6d} K={k:6d} | prep_bitexact={same} stacked_payload_bitexact={stacked_ok} |"
        good = same and stacked_ok
        for name, mode, dv, sv in plans:
            out = sharq_ops_vllm.sharq_linear_forward_cfg(x, qw, sfs, sfd, ws, mode, dv, sv, 1)
            torch.cuda.synchronize()
            err = rel_err(out, ref)
            good &= err < 1e-2
            line += f" {name}:{err:.1e}"
        out = sharq_ops_vllm.sharq_linear_forward(x, qw, sfs, sfd, ws)
        err_auto = rel_err(out, ref)
        good &= err_auto < 1e-2
        line += f" auto:{err_auto:.1e} | rel(auto,bf16)={rel_err(out, bf16_ref):.4f} rel(ref,bf16)={rel_err(ref, bf16_ref):.4f}"
        print(line + (" OK" if good else " FAIL"))
        ok &= good

    # CUDA graph capture (both modes) on a side stream
    for m in (4, 256):
        n, k = 6144, 4096
        x = torch.randn(m, k, device=dev).to(torch.bfloat16)
        w = (torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16)
        qw, sfs, sfd, ws = quantize_weight(w)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                sharq_ops_vllm.sharq_linear_forward(x, qw, sfs, sfd, ws)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y_static = sharq_ops_vllm.sharq_linear_forward(x, qw, sfs, sfd, ws)
        x.copy_(torch.randn(m, k, device=dev).to(torch.bfloat16) * 3)
        g.replay()
        torch.cuda.synchronize()
        y_eager = sharq_ops_vllm.sharq_linear_forward(x, qw, sfs, sfd, ws)
        graph_ok = torch.equal(y_static, y_eager)
        ok &= graph_ok
        print(f"CUDA graph replay (M={m}) matches eager: {graph_ok}")
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
