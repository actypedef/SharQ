from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def load_agemm():
    repo_root = Path(__file__).resolve().parents[2]
    build_dir = repo_root / "kernels" / "build_cmake_sm120a"
    sys.path.insert(0, str(build_dir))
    try:
        import sharq_ops as backend  # type: ignore
    except ImportError:
        import agemm as backend  # type: ignore

    return backend


def summarize_error(name: str, y: torch.Tensor, y_ref: torch.Tensor) -> None:
    diff = (y.float() - y_ref.float()).contiguous()
    mse = F.mse_loss(y.float(), y_ref.float()).item()
    rmse = mse ** 0.5
    ref_rms = y_ref.float().pow(2).mean().sqrt().item()
    rel_rmse = rmse / max(ref_rms, 1e-12)
    print(name)
    print(f"  mse      : {mse:.8f}")
    print(f"  rmse     : {rmse:.8f}")
    print(f"  rel_rmse : {rel_rmse:.8f}")
    print(f"  max_abs  : {diff.abs().max().item():.8f}")


def global_nvfp4_scale(x: torch.Tensor) -> torch.Tensor:
    return x.abs().max().float() / (448.0 * 6.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fused sparse+residual path with shared W32 payload.")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=5120)
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.k % 128 != 0:
        raise ValueError(f"k must be a multiple of 128, got {args.k}")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    agemm = load_agemm()

    x = torch.randn((args.m, args.k), device=device, dtype=torch.bfloat16)
    w = torch.randn((args.n, args.k), device=device, dtype=torch.bfloat16)
    reorder_index = torch.arange(args.k, device=device, dtype=torch.int16)

    y_ref = torch.matmul(x.float(), w.float().t())

    scale_x = global_nvfp4_scale(x)
    scale_w = global_nvfp4_scale(w)
    x_scaled = (x / scale_x).to(torch.bfloat16)
    w_scaled = (w / scale_w).to(torch.bfloat16)

    qx_dense, sfx_dense = agemm.reorder_quantize_x(x_scaled, reorder_index, 0)
    qw16, sfw16 = agemm.reorder_quantize_w(w_scaled, reorder_index, 0)
    y_dense = agemm.matmul(qx_dense, qw16, sfx_dense, sfw16, float(scale_x * scale_w)).float()

    qw32, sfw_sparse32, sfw_dense16 = agemm.quantize_w32_shared(w_scaled)
    y_dense_shared = agemm.matmul(
        qx_dense, qw32, sfx_dense, sfw_dense16, float(scale_x * scale_w)
    ).float()

    a_comp, e, sfa_sparse, q_res, sf_res = agemm.fused_sparse_residual_quantize_x(x_scaled, args.n)
    y_sparse = agemm.sparse_matmul(
        a_comp, qw32, e, sfa_sparse, sfw_sparse32, args.m, args.n, args.k, float(scale_x * scale_w)
    ).float()
    y_res = agemm.matmul(
        q_res, qw32, sf_res, sfw_dense16, float(scale_x * scale_w)
    ).float()
    y_fused = y_sparse + y_res

    print(f"problem: M={args.m}, N={args.n}, K={args.k}, seed={args.seed}")
    print(f"bf16 checksum [0:16]      : {y_ref.flatten()[:16].sum().item():.8f}")
    print(f"dense w16 checksum [0:16] : {y_dense.flatten()[:16].sum().item():.8f}")
    print(f"dense w32 checksum [0:16] : {y_dense_shared.flatten()[:16].sum().item():.8f}")
    print(f"fused checksum [0:16]     : {y_fused.flatten()[:16].sum().item():.8f}")
    print()

    summarize_error("Dense NVFP4 W16 vs BF16", y_dense, y_ref)
    print()
    summarize_error("Dense shared-payload W32 vs BF16", y_dense_shared, y_ref)
    print()
    summarize_error("Fused shared-payload path vs BF16", y_fused, y_ref)
    print()
    summarize_error("Fused shared-payload path vs dense shared-payload W32", y_fused, y_dense_shared)


if __name__ == "__main__":
    main()
