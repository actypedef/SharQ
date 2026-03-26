from __future__ import annotations

import argparse
import math
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
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()

    print(name)
    print(f"  mse      : {mse:.8f}")
    print(f"  rmse     : {rmse:.8f}")
    print(f"  rel_rmse : {rel_rmse:.8f}")
    print(f"  mean_abs : {mean_abs:.8f}")
    print(f"  max_abs  : {max_abs:.8f}")


def global_nvfp4_scale(x: torch.Tensor) -> torch.Tensor:
    return x.abs().max().float() / (448.0 * 6.0)


def make_outlier_activation(
    m: int,
    k: int,
    device: torch.device,
    outlier_ratio: float,
    outlier_min_scale: float,
    outlier_max_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), device=device, dtype=torch.float32)
    num_outlier_channels = max(1, int(round(k * outlier_ratio)))
    outlier_idx = torch.randperm(k, device=device)[:num_outlier_channels]
    log_scales = torch.empty(num_outlier_channels, device=device, dtype=torch.float32).uniform_(
        math.log10(outlier_min_scale),
        math.log10(outlier_max_scale),
    )
    scales = torch.pow(10.0, log_scales)
    x[:, outlier_idx] *= scales.unsqueeze(0)
    return x.to(torch.bfloat16), outlier_idx, scales


def top2_4(x: torch.Tensor) -> torch.Tensor:
    groups = x.view(*x.shape[:-1], x.shape[-1] // 4, 4)
    idx = groups.abs().topk(k=2, dim=-1).indices
    mask = torch.zeros_like(groups, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return (groups * mask).view_as(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure linear-layer error for fused sparse+residual NVFP4 path.")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=5120)
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outlier-ratio", type=float, default=0.05)
    parser.add_argument("--outlier-min-scale", type=float, default=10.0)
    parser.add_argument("--outlier-max-scale", type=float, default=100.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if args.k % 256 != 0:
        raise ValueError(f"k must be a multiple of 256 for the current fused kernel, got {args.k}")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    agemm = load_agemm()

    x, outlier_idx, outlier_scales = make_outlier_activation(
        args.m,
        args.k,
        device,
        args.outlier_ratio,
        args.outlier_min_scale,
        args.outlier_max_scale,
    )
    w = torch.randn((args.n, args.k), device=device, dtype=torch.bfloat16)
    reorder_index = torch.arange(args.k, device=device, dtype=torch.int16)

    y_ref = torch.matmul(x, w.t()).float()

    qx_dense, sfx_dense = agemm.reorder_quantize_x(x, reorder_index, 0)
    qw, sfw = agemm.reorder_quantize_w(w, reorder_index, 0)
    y_dense_quant = agemm.matmul(qx_dense, qw, sfx_dense, sfw, 1.0).float()

    scale_x = global_nvfp4_scale(x)
    scale_w = global_nvfp4_scale(w)
    x_scaled = (x / scale_x).to(torch.bfloat16)
    w_scaled = (w / scale_w).to(torch.bfloat16)

    qx_dense_scaled, sfx_dense_scaled = agemm.reorder_quantize_x(x_scaled, reorder_index, 0)
    qw_scaled, sfw_scaled = agemm.reorder_quantize_w(w_scaled, reorder_index, 0)
    y_dense_quant_scaled = (
        agemm.matmul(qx_dense_scaled, qw_scaled, sfx_dense_scaled, sfw_scaled, float(scale_x * scale_w)).float()
    )

    a_comp, e, sfa_sparse, q_res, sf_res = agemm.fused_sparse_residual_quantize_x(x, args.n)
    y_sparse_main = agemm.sparse_matmul(a_comp, qw, e, sfa_sparse, sfw, args.m, args.n, args.k).float()
    y_residual = agemm.matmul(q_res, qw, sf_res, sfw, 1.0).float()
    y_fused = y_sparse_main + y_residual

    i_mat = torch.eye(args.k, device=device, dtype=torch.bfloat16)
    qwi, sfi = agemm.reorder_quantize_w(i_mat, reorder_index, 0)
    x_sparse_ref = top2_4(x)
    qx_sparse_ref, sfx_sparse_ref = agemm.reorder_quantize_x(x_sparse_ref, reorder_index, 0)
    a_comp_ref, e_ref = agemm.compress_sparse_a(qx_sparse_ref, args.n)
    x_approx_ref = agemm.matmul(qx_sparse_ref, qwi, sfx_sparse_ref, sfi, 1.0).float()
    x_res_ref = (x.float() - x_approx_ref).to(torch.bfloat16)
    qx_res_ref, sfx_res_ref = agemm.reorder_quantize_x(x_res_ref, reorder_index, 0)
    y_sparse_ref = agemm.sparse_matmul(
        a_comp_ref, qw, e_ref, sfx_sparse_ref, sfw, args.m, args.n, args.k
    ).float()
    y_res_ref = agemm.matmul(qx_res_ref, qw, sfx_res_ref, sfw, 1.0).float()
    y_fused_ref = y_sparse_ref + y_res_ref

    a_comp_scaled, e_scaled, sfa_sparse_scaled, q_res_scaled, sf_res_scaled = agemm.fused_sparse_residual_quantize_x(
        x_scaled, args.n
    )
    y_sparse_main_scaled = agemm.sparse_matmul(
        a_comp_scaled, qw_scaled, e_scaled, sfa_sparse_scaled, sfw_scaled, args.m, args.n, args.k, float(scale_x * scale_w)
    ).float()
    y_residual_scaled = agemm.matmul(
        q_res_scaled, qw_scaled, sf_res_scaled, sfw_scaled, float(scale_x * scale_w)
    ).float()
    y_fused_scaled = y_sparse_main_scaled + y_residual_scaled

    print(f"problem: M={args.m}, N={args.n}, K={args.k}, seed={args.seed}")
    print(
        "outliers: "
        f"channels={outlier_idx.numel()} ({100.0 * outlier_idx.numel() / args.k:.2f}%), "
        f"scale_range=[{outlier_scales.min().item():.3f}, {outlier_scales.max().item():.3f}]"
    )
    print(f"reference checksum[0:16]: {y_ref.flatten()[:16].sum().item():.8f}")
    print(f"dense quant checksum[0:16]: {y_dense_quant.flatten()[:16].sum().item():.8f}")
    print(f"dense scaled checksum[0:16]: {y_dense_quant_scaled.flatten()[:16].sum().item():.8f}")
    print(f"fused path checksum[0:16]: {y_fused.flatten()[:16].sum().item():.8f}")
    print(f"fused scaled checksum[0:16]: {y_fused_scaled.flatten()[:16].sum().item():.8f}")
    print(f"pseudo-sparse ref checksum[0:16]: {y_fused_ref.flatten()[:16].sum().item():.8f}")
    print()

    summarize_error("Dense NVFP4 vs FP32 reference", y_dense_quant, y_ref)
    print()
    summarize_error("Dense NVFP4 with global pre-scale vs FP32 reference", y_dense_quant_scaled, y_ref)
    print()
    summarize_error("Sparse main only vs FP32 reference", y_sparse_main, y_ref)
    print()
    summarize_error("Sparse main + residual vs FP32 reference", y_fused, y_ref)
    print()
    summarize_error("Sparse main + residual with global pre-scale vs FP32 reference", y_fused_scaled, y_ref)
    print()
    summarize_error("Fused kernel vs pseudo-sparse reference", y_fused, y_fused_ref)


if __name__ == "__main__":
    main()
