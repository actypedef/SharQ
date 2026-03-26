from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def load_sharq_ops():
    repo_root = Path(__file__).resolve().parents[2]
    build_dir = repo_root / "kernels" / "build_cmake_sm120a"
    sys.path.insert(0, str(build_dir))
    import sharq_ops as backend  # type: ignore

    return backend


def quantize_e2m1(tensor: torch.Tensor) -> torch.Tensor:
    representable_vals = torch.tensor(
        [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=tensor.device,
        dtype=tensor.dtype,
    )
    diff = torch.abs(tensor.unsqueeze(-1) - representable_vals)
    return representable_vals[torch.argmin(diff, dim=-1)]


def quantize_ue4m3(tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.clamp(tensor, min=2e-3, max=448.0)
    exponent = torch.floor(torch.log2(tensor + 1e-9))
    mantissa_val = tensor / (2 ** exponent) - 1.0
    quantized_mantissa_val = torch.round(mantissa_val * 8) / 8
    return (1 + quantized_mantissa_val) * (2 ** exponent)


def quantize_nvfp4_tensor(tensor: torch.Tensor, group_size: int) -> torch.Tensor:
    original_shape = tensor.shape
    padding = (group_size - tensor.shape[-1] % group_size) % group_size
    if padding != 0:
        tensor = F.pad(tensor, (0, padding))

    reshaped = tensor.view(-1, group_size)
    max_abs = reshaped.abs().max(dim=1, keepdim=True)[0]
    scale = max_abs / 6.0
    scale[scale == 0] = 1e-9
    dq_scale = quantize_ue4m3(scale)
    normalized = reshaped / dq_scale
    q = quantize_e2m1(normalized)
    out = (q * dq_scale).view(tensor.shape)

    if padding != 0:
        out = out[..., :-padding]
    return out.view(original_shape)


def top2_pairs_8_maxabs(x: torch.Tensor) -> torch.Tensor:
    groups = x.view(*x.shape[:-1], x.shape[-1] // 8, 4, 2)
    pair_scores = groups.abs().amax(dim=-1)
    idx = pair_scores.topk(k=2, dim=-1).indices
    pair_mask = torch.zeros_like(pair_scores, dtype=torch.bool)
    pair_mask.scatter_(-1, idx, True)
    mask = pair_mask.unsqueeze(-1).expand_as(groups)
    return (groups * mask).view_as(x)


def global_nvfp4_scale(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.abs().max().float() / (448.0 * 6.0), min=1e-9)


def summarize(name: str, pred: torch.Tensor, ref: torch.Tensor) -> None:
    diff = pred.float() - ref.float()
    mse = F.mse_loss(pred.float(), ref.float()).item()
    rmse = mse ** 0.5
    ref_rms = ref.float().pow(2).mean().sqrt().item()
    rel_rmse = rmse / max(ref_rms, 1e-12)
    print(name)
    print(f"  mse      : {mse:.8f}")
    print(f"  rmse     : {rmse:.8f}")
    print(f"  rel_rmse : {rel_rmse:.8f}")
    print(f"  mean_abs : {diff.abs().mean().item():.8f}")
    print(f"  max_abs  : {diff.abs().max().item():.8f}")


def make_outlier_rich_x(m: int, k: int, device: torch.device) -> torch.Tensor:
    groups = k // 4
    x = torch.randn((m, groups, 4), device=device, dtype=torch.float32) * 0.03

    big_idx = torch.randint(0, 4, (m, groups, 2), device=device)
    same = big_idx[..., 0] == big_idx[..., 1]
    big_idx[..., 1] = torch.where(same, (big_idx[..., 1] + 1) % 4, big_idx[..., 1])

    big_mag_0 = torch.empty((m, groups), device=device).uniform_(8.0, 24.0)
    big_mag_1 = torch.empty((m, groups), device=device).uniform_(6.0, 20.0)
    sign0 = torch.where(torch.rand((m, groups), device=device) > 0.5, 1.0, -1.0)
    sign1 = torch.where(torch.rand((m, groups), device=device) > 0.5, 1.0, -1.0)

    x.scatter_(2, big_idx[..., 0:1], (big_mag_0 * sign0).unsqueeze(-1))
    x.scatter_(2, big_idx[..., 1:2], (big_mag_1 * sign1).unsqueeze(-1))
    x = x.view(m, k)

    num_outlier_channels = int(k * 0.08)
    ch_idx = torch.randperm(k, device=device)[:num_outlier_channels]
    ch_scales = torch.empty(num_outlier_channels, device=device).uniform_(10.0, 40.0)
    x[:, ch_idx] *= ch_scales.unsqueeze(0)
    return x.to(torch.bfloat16)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare current kernel path against a Python pseudo simulation.")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=5120)
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case", choices=("outlier_rich", "random"), default="outlier_rich")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.k % 128 != 0:
        raise ValueError(f"k must be a multiple of 128, got {args.k}")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    backend = load_sharq_ops()

    if args.case == "outlier_rich":
        x = make_outlier_rich_x(args.m, args.k, device)
    else:
        x = torch.randn((args.m, args.k), device=device, dtype=torch.bfloat16)
    w = torch.randn((args.n, args.k), device=device, dtype=torch.bfloat16)
    reorder_index = torch.arange(args.k, device=device, dtype=torch.int16)

    y_ref = torch.matmul(x.float(), w.float().t())

    scale_x = global_nvfp4_scale(x)
    scale_w = global_nvfp4_scale(w)
    x_scaled = (x / scale_x).to(torch.bfloat16)
    w_scaled = (w / scale_w).to(torch.bfloat16)
    output_scale = float(scale_x * scale_w)

    # Python pseudo simulation of the current hardware-compatible idea:
    # sparse main path: pair-top2 on each 8 values + NVFP4 group32
    # residual path: x - Q32(pair_top2(x)), then NVFP4 group16
    # weight path: one W32 quantization shared by both branches
    w32_pseudo = quantize_nvfp4_tensor(w_scaled.float(), group_size=32)
    x_sparse = top2_pairs_8_maxabs(x_scaled.float())
    x_sparse_q32 = quantize_nvfp4_tensor(x_sparse, group_size=32)
    x_res = x_scaled.float() - x_sparse_q32
    x_res_q16 = quantize_nvfp4_tensor(x_res, group_size=16)

    y_pseudo_sparse = F.linear(x_sparse_q32, w32_pseudo)
    y_pseudo_res = F.linear(x_res_q16, w32_pseudo)
    y_pseudo = (y_pseudo_sparse + y_pseudo_res) * output_scale

    # Kernel path
    qw32, sfw_sparse32, sfw_dense16 = backend.quantize_w32_shared(w_scaled)
    a_comp, e, sfa_sparse, q_res, sf_res = backend.fused_sparse_residual_quantize_x(x_scaled, args.n)
    y_kernel_sparse = backend.sparse_matmul(
        a_comp, qw32, e, sfa_sparse, sfw_sparse32, args.m, args.n, args.k, output_scale
    ).float()
    y_kernel_res = backend.matmul(
        q_res, qw32, sf_res, sfw_dense16, output_scale
    ).float()
    y_kernel = y_kernel_sparse + y_kernel_res

    print(f"problem: M={args.m}, N={args.n}, K={args.k}, seed={args.seed}, case={args.case}")
    print(f"ref checksum [0:16]           : {y_ref.flatten()[:16].sum().item():.8f}")
    print(f"pseudo sparse checksum [0:16] : {y_pseudo_sparse.flatten()[:16].sum().item():.8f}")
    print(f"pseudo total checksum [0:16]  : {y_pseudo.flatten()[:16].sum().item():.8f}")
    print(f"kernel sparse checksum [0:16] : {y_kernel_sparse.flatten()[:16].sum().item():.8f}")
    print(f"kernel total checksum [0:16]  : {y_kernel.flatten()[:16].sum().item():.8f}")
    print()

    summarize("Pseudo pair-top2 idea vs BF16", y_pseudo, y_ref)
    print()
    summarize("Kernel current path vs BF16", y_kernel, y_ref)
    print()
    summarize("Kernel vs pseudo pair-top2 idea", y_kernel, y_pseudo)
    print()
    summarize("Pseudo pair-top2 sparse-only vs BF16", y_pseudo_sparse * output_scale, y_ref)
    print()
    summarize("Kernel sparse-only vs BF16", y_kernel_sparse, y_ref)


if __name__ == "__main__":
    main()
