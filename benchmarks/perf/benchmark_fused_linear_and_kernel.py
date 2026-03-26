from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_sharq_ops():
    build_dir = REPO_ROOT / "kernels" / "build_cmake_sm120a"
    sys.path.insert(0, str(build_dir))
    import sharq_ops as backend  # type: ignore

    return backend


def bench_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def nvfp4_quantize_only(x: torch.Tensor, backend) -> None:
    scale = torch.clamp(x.abs().max().float() / (448.0 * 6.0), min=1e-9)
    backend.quantize_x_nvfp4((x / scale).contiguous())


def sharq_quantize_only(x: torch.Tensor, out_features: int, backend) -> None:
    scale = torch.clamp(x.abs().max().float() / (448.0 * 6.0), min=1e-9)
    backend.fused_sparse_residual_quantize_x((x / scale).contiguous(), out_features)


def nvfp4_linear_e2e(x: torch.Tensor, qweight: torch.Tensor, scale_w: torch.Tensor, weight_scale, backend) -> None:
    scale_x = torch.clamp(x.abs().max().float() / (448.0 * 6.0), min=1e-9)
    qx, sfx = backend.quantize_x_nvfp4((x / scale_x).contiguous())
    backend.matmul(qx, qweight, sfx, scale_w, float(scale_x * weight_scale))


def sharq_linear_e2e(
    x: torch.Tensor,
    out_features: int,
    qweight: torch.Tensor,
    scale_w_sparse: torch.Tensor,
    scale_w_dense: torch.Tensor,
    weight_scale,
    backend,
) -> None:
    scale_x = torch.clamp(x.abs().max().float() / (448.0 * 6.0), min=1e-9)
    a_comp, e, sfa_sparse, q_res, sf_res = backend.fused_sparse_residual_quantize_x((x / scale_x).contiguous(), out_features)
    alpha = float(scale_x * weight_scale)
    y_sparse = backend.sparse_matmul(
        a_comp, qweight, e, sfa_sparse, scale_w_sparse, x.shape[0], out_features, x.shape[1], alpha=alpha
    )
    y_res = backend.matmul(q_res, qweight, sf_res, scale_w_dense, alpha)
    return y_sparse + y_res


def sharq_gemm_only(
    a_comp: torch.Tensor,
    e: torch.Tensor,
    sfa_sparse: torch.Tensor,
    q_res: torch.Tensor,
    sf_res: torch.Tensor,
    qweight: torch.Tensor,
    scale_w_sparse: torch.Tensor,
    scale_w_dense: torch.Tensor,
    alpha: float,
    m: int,
    n: int,
    k: int,
    backend,
) -> None:
    y_sparse = backend.sparse_matmul(a_comp, qweight, e, sfa_sparse, scale_w_sparse, m, n, k, alpha=alpha)
    y_res = backend.matmul(q_res, qweight, sf_res, scale_w_dense, alpha)
    return y_sparse + y_res


def bf16_linear(x: torch.Tensor, weight: torch.Tensor) -> None:
    return torch.nn.functional.linear(x, weight)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark NVFP4 and SHARQ quantize, GEMM, and end-to-end linear latency.")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=5120)
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    backend = load_sharq_ops()

    x2d = torch.randn((args.m, args.k), device=device, dtype=torch.bfloat16)
    w = torch.randn((args.n, args.k), device=device, dtype=torch.bfloat16)

    weight_scale_nvfp4 = torch.clamp(w.abs().max().float() / (448.0 * 6.0), min=1e-9)
    qweight_nvfp4, scale_w_nvfp4 = backend.quantize_w_nvfp4((w / weight_scale_nvfp4).contiguous())

    weight_scale_sharq = torch.clamp(w.abs().max().float() / (448.0 * 6.0), min=1e-9)
    qweight_sharq, scale_w_sharq_sparse, scale_w_sharq_dense = backend.quantize_w32_shared((w / weight_scale_sharq).contiguous())

    scale_x_nvfp4 = torch.clamp(x2d.abs().max().float() / (448.0 * 6.0), min=1e-9)
    qx_nvfp4, sfx_nvfp4 = backend.quantize_x_nvfp4((x2d / scale_x_nvfp4).contiguous())
    alpha_nvfp4 = float(scale_x_nvfp4 * weight_scale_nvfp4)

    scale_x_sharq = torch.clamp(x2d.abs().max().float() / (448.0 * 6.0), min=1e-9)
    a_comp, e, sfa_sparse, q_res, sf_res = backend.fused_sparse_residual_quantize_x((x2d / scale_x_sharq).contiguous(), args.n)
    alpha_sharq = float(scale_x_sharq * weight_scale_sharq)

    nvfp4_quantize_ms = bench_cuda(lambda: nvfp4_quantize_only(x2d, backend), args.warmup, args.iters)
    sharq_quantize_ms = bench_cuda(lambda: sharq_quantize_only(x2d, args.n, backend), args.warmup, args.iters)

    bf16_linear_ms = bench_cuda(lambda: bf16_linear(x2d, w), args.warmup, args.iters)

    nvfp4_gemm_ms = bench_cuda(
        lambda: backend.matmul(qx_nvfp4, qweight_nvfp4, sfx_nvfp4, scale_w_nvfp4, alpha_nvfp4),
        args.warmup,
        args.iters,
    )
    sharq_gemm_ms = bench_cuda(
        lambda: sharq_gemm_only(
            a_comp,
            e,
            sfa_sparse,
            q_res,
            sf_res,
            qweight_sharq,
            scale_w_sharq_sparse,
            scale_w_sharq_dense,
            alpha_sharq,
            args.m,
            args.n,
            args.k,
            backend,
        ),
        args.warmup,
        args.iters,
    )

    nvfp4_linear_ms = bench_cuda(
        lambda: nvfp4_linear_e2e(x2d, qweight_nvfp4, scale_w_nvfp4, weight_scale_nvfp4, backend),
        args.warmup,
        args.iters,
    )
    sharq_linear_ms = bench_cuda(
        lambda: sharq_linear_e2e(
            x2d,
            args.n,
            qweight_sharq,
            scale_w_sharq_sparse,
            scale_w_sharq_dense,
            weight_scale_sharq,
            backend,
        ),
        args.warmup,
        args.iters,
    )

    print(f"problem: M={args.m}, N={args.n}, K={args.k}, seed={args.seed}")
    print()
    print("Quantize only")
    print("  BF16                      : N/A")
    print(f"  NVFP4                     : {nvfp4_quantize_ms:.6f} ms")
    print(f"  SHARQ                     : {sharq_quantize_ms:.6f} ms")
    print(f"  SHARQ/NVFP4 ratio         : {sharq_quantize_ms / max(nvfp4_quantize_ms, 1e-12):.6f}")
    print()
    print("GEMM only")
    print(f"  BF16                      : {bf16_linear_ms:.6f} ms")
    print(f"  NVFP4                     : {nvfp4_gemm_ms:.6f} ms")
    print(f"  SHARQ                     : {sharq_gemm_ms:.6f} ms")
    print(f"  NVFP4/BF16 ratio          : {nvfp4_gemm_ms / max(bf16_linear_ms, 1e-12):.6f}")
    print(f"  SHARQ/BF16 ratio          : {sharq_gemm_ms / max(bf16_linear_ms, 1e-12):.6f}")
    print(f"  SHARQ/NVFP4 ratio         : {sharq_gemm_ms / max(nvfp4_gemm_ms, 1e-12):.6f}")
    print()
    print("Whole linear")
    print(f"  BF16                      : {bf16_linear_ms:.6f} ms")
    print(f"  NVFP4                     : {nvfp4_linear_ms:.6f} ms")
    print(f"  SHARQ                     : {sharq_linear_ms:.6f} ms")
    print(f"  NVFP4/BF16 ratio          : {nvfp4_linear_ms / max(bf16_linear_ms, 1e-12):.6f}")
    print(f"  SHARQ/BF16 ratio          : {sharq_linear_ms / max(bf16_linear_ms, 1e-12):.6f}")
    print(f"  SHARQ/NVFP4 ratio         : {sharq_linear_ms / max(nvfp4_linear_ms, 1e-12):.6f}")


if __name__ == "__main__":
    main()
