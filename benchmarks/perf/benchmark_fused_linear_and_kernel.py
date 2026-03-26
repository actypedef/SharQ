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


def load_qlinear():
    sys.path.insert(0, str(REPO_ROOT / "model"))
    from qLinearLayer import QLinearLayer  # type: ignore

    return QLinearLayer


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


def kernel_dense_prepare(x: torch.Tensor, reorder_index: torch.Tensor, backend) -> None:
    scale = torch.clamp(x.abs().max().float() / (448.0 * 6.0), min=1e-9)
    backend.reorder_quantize_x((x / scale).contiguous(), reorder_index, 0)


def kernel_fused_prepare(x: torch.Tensor, out_features: int, backend) -> None:
    scale = torch.clamp(x.abs().max().float() / (448.0 * 6.0), min=1e-9)
    backend.fused_sparse_residual_quantize_x((x / scale).contiguous(), out_features)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark standalone activation kernels and one QLinearLayer forward.")
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
    QLinearLayer = load_qlinear()

    x2d = torch.randn((args.m, args.k), device=device, dtype=torch.bfloat16)
    x3d = x2d.view(1, args.m, args.k).contiguous()
    w = torch.randn((args.n, args.k), device=device, dtype=torch.bfloat16)
    layer = torch.nn.Linear(args.k, args.n, bias=False, dtype=torch.bfloat16).to(device)
    with torch.no_grad():
        layer.weight.copy_(w)

    reorder_index = torch.arange(args.k, device=device, dtype=torch.int16)
    reorder_index_cpu = torch.arange(args.k, dtype=torch.int16)

    qlinear_dense = QLinearLayer(layer, select_num=0, reorder_index=reorder_index_cpu, quant_type="NVFP4")
    qlinear_fused = QLinearLayer(layer, select_num=0, reorder_index=reorder_index_cpu, quant_type="SHARQ")

    dense_kernel_ms = bench_cuda(lambda: kernel_dense_prepare(x2d, reorder_index, backend), args.warmup, args.iters)
    fused_kernel_ms = bench_cuda(lambda: kernel_fused_prepare(x2d, args.n, backend), args.warmup, args.iters)

    dense_linear_ms = bench_cuda(lambda: qlinear_dense((x3d, 1, args.m)), args.warmup, args.iters)
    fused_linear_ms = bench_cuda(lambda: qlinear_fused((x3d, 1, args.m)), args.warmup, args.iters)

    print(f"problem: M={args.m}, N={args.n}, K={args.k}, seed={args.seed}")
    print()
    print("Kernel only")
    print(f"  reorder_quantize_x        : {dense_kernel_ms:.6f} ms")
    print(f"  fused_sparse_prepare      : {fused_kernel_ms:.6f} ms")
    print(f"  fused/dense ratio         : {fused_kernel_ms / max(dense_kernel_ms, 1e-12):.6f}")
    print()
    print("One Linear")
    print(f"  QLinearLayer NVFP4        : {dense_linear_ms:.6f} ms")
    print(f"  QLinearLayer SHARQ        : {fused_linear_ms:.6f} ms")
    print(f"  fused/dense ratio         : {fused_linear_ms / max(dense_linear_ms, 1e-12):.6f}")


if __name__ == "__main__":
    main()
