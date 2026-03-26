from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F


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


def top2_4_scalar(x: torch.Tensor) -> torch.Tensor:
    groups = x.view(*x.shape[:-1], x.shape[-1] // 4, 4)
    idx = groups.abs().topk(k=2, dim=-1).indices
    mask = torch.zeros_like(groups, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return (groups * mask).view_as(x)


def _apply_pair_scores(x: torch.Tensor, pair_scores: torch.Tensor) -> torch.Tensor:
    groups = x.view(*x.shape[:-1], x.shape[-1] // 8, 4, 2)
    top_idx = pair_scores.topk(k=2, dim=-1).indices
    pair_mask = torch.zeros_like(pair_scores, dtype=torch.bool)
    pair_mask.scatter_(-1, top_idx, True)
    value_mask = pair_mask.unsqueeze(-1).expand_as(groups)
    return (groups * value_mask).reshape_as(x)


def pair_top2_maxabs(x: torch.Tensor) -> torch.Tensor:
    groups = x.view(*x.shape[:-1], x.shape[-1] // 8, 4, 2)
    pair_scores = groups.abs().amax(dim=-1)
    return _apply_pair_scores(x, pair_scores)


def pair_top2_l1(x: torch.Tensor) -> torch.Tensor:
    groups = x.view(*x.shape[:-1], x.shape[-1] // 8, 4, 2)
    pair_scores = groups.abs().sum(dim=-1)
    return _apply_pair_scores(x, pair_scores)


def pair_top2_l2sq(x: torch.Tensor) -> torch.Tensor:
    groups = x.view(*x.shape[:-1], x.shape[-1] // 8, 4, 2)
    pair_scores = groups.square().sum(dim=-1)
    return _apply_pair_scores(x, pair_scores)


def pair_top2_hybrid(x: torch.Tensor, lam: float = 0.25) -> torch.Tensor:
    groups = x.view(*x.shape[:-1], x.shape[-1] // 8, 4, 2).abs()
    pair_scores = groups.amax(dim=-1) + lam * groups.amin(dim=-1)
    return _apply_pair_scores(x, pair_scores)


def pair_project_from_scalar_top2_4(x: torch.Tensor) -> torch.Tensor:
    groups8 = x.view(*x.shape[:-1], x.shape[-1] // 8, 2, 4)
    half_abs = groups8.abs()
    idx = half_abs.topk(k=2, dim=-1).indices
    scalar_mask = torch.zeros_like(groups8, dtype=torch.bool)
    scalar_mask.scatter_(-1, idx, True)
    selected = groups8.abs() * scalar_mask
    pair_scores = selected.view(*selected.shape[:-2], 4, 2).sum(dim=-1)
    return _apply_pair_scores(x, pair_scores)


def pair_top2_from_global_top4(x: torch.Tensor) -> torch.Tensor:
    groups8 = x.view(*x.shape[:-1], x.shape[-1] // 8, 8)
    idx = groups8.abs().topk(k=4, dim=-1).indices
    scalar_mask = torch.zeros_like(groups8, dtype=torch.bool)
    scalar_mask.scatter_(-1, idx, True)
    selected = groups8.abs() * scalar_mask
    pair_scores = selected.view(*selected.shape[:-1], 4, 2).sum(dim=-1)
    return _apply_pair_scores(x, pair_scores)


def raw_byte_violation_rate(x_sparse: torch.Tensor) -> float:
    pair_nonzero = (x_sparse.view(x_sparse.size(0), -1, 2) != 0).any(dim=-1)
    count = pair_nonzero.view(x_sparse.size(0), -1, 4).sum(dim=-1)
    return (count > 2).float().mean().item()


def global_nvfp4_scale(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.abs().max().float() / (448.0 * 6.0), min=1e-9)


def metrics(pred: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    diff = pred.float() - ref.float()
    mse = diff.square().mean().item()
    rmse = mse ** 0.5
    ref_rms = ref.float().square().mean().sqrt().item()
    return mse, rmse / max(ref_rms, 1e-12)


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
    num_outlier_channels = max(1, int(k * 0.08))
    ch_idx = torch.randperm(k, device=device)[:num_outlier_channels]
    ch_scales = torch.empty(num_outlier_channels, device=device).uniform_(10.0, 40.0)
    x[:, ch_idx] *= ch_scales.unsqueeze(0)
    return x.to(torch.bfloat16)


def evaluate_strategy(name: str, x_scaled: torch.Tensor, w32: torch.Tensor, y_ref: torch.Tensor, output_scale: float, fn):
    x_sparse = fn(x_scaled.float())
    x_sparse_q32 = quantize_nvfp4_tensor(x_sparse, group_size=32)
    x_res_q16 = quantize_nvfp4_tensor(x_scaled.float() - x_sparse_q32, group_size=16)
    y = (F.linear(x_sparse_q32, w32) + F.linear(x_res_q16, w32)) * output_scale
    mse, rel_rmse = metrics(y, y_ref)
    return {
        "name": name,
        "mse": mse,
        "rel_rmse": rel_rmse,
        "viol": raw_byte_violation_rate(x_sparse_q32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pair-level sparse mask strategies under CUTLASS FP4 sparse constraints.")
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

    if args.case == "outlier_rich":
        x = make_outlier_rich_x(args.m, args.k, device)
    else:
        x = torch.randn((args.m, args.k), device=device, dtype=torch.bfloat16)
    w = torch.randn((args.n, args.k), device=device, dtype=torch.bfloat16)

    y_ref = torch.matmul(x.float(), w.float().t())
    scale_x = global_nvfp4_scale(x)
    scale_w = global_nvfp4_scale(w)
    x_scaled = (x / scale_x).to(torch.bfloat16)
    w_scaled = (w / scale_w).to(torch.bfloat16)
    output_scale = float(scale_x * scale_w)
    w32 = quantize_nvfp4_tensor(w_scaled.float(), group_size=32)
    w16 = quantize_nvfp4_tensor(w_scaled.float(), group_size=16)

    dense16 = (F.linear(quantize_nvfp4_tensor(x_scaled.float(), group_size=16), w16) * output_scale)
    scalar = evaluate_strategy("scalar_top2:4", x_scaled, w32, y_ref, output_scale, top2_4_scalar)

    candidates = [
        ("pair_maxabs", pair_top2_maxabs),
        ("pair_l1", pair_top2_l1),
        ("pair_l2sq", pair_top2_l2sq),
        ("pair_max_plus_qmin", pair_top2_hybrid),
        ("pair_project_scalar_top2:4", pair_project_from_scalar_top2_4),
        ("pair_project_global_top4", pair_top2_from_global_top4),
    ]

    results = []
    for name, fn in candidates:
        results.append(evaluate_strategy(name, x_scaled, w32, y_ref, output_scale, fn))

    dense16_mse, dense16_rel = metrics(dense16, y_ref)
    print(f"problem: M={args.m}, N={args.n}, K={args.k}, seed={args.seed}, case={args.case}")
    print(f"{'strategy':30s} {'viol':>8s} {'rel_rmse':>12s} {'mse':>14s}")
    print(f"{'dense_nvfp4_w16':30s} {'-':>8s} {dense16_rel:12.8f} {dense16_mse:14.4f}")
    print(f"{scalar['name']:30s} {scalar['viol']:8.4f} {scalar['rel_rmse']:12.8f} {scalar['mse']:14.4f}")
    for item in sorted(results, key=lambda x: x['rel_rmse']):
        print(f"{item['name']:30s} {item['viol']:8.4f} {item['rel_rmse']:12.8f} {item['mse']:14.4f}")


if __name__ == "__main__":
    main()
