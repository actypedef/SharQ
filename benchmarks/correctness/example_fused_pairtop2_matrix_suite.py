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
    top_idx = pair_scores.topk(k=2, dim=-1).indices
    pair_mask = torch.zeros_like(pair_scores, dtype=torch.bool)
    pair_mask.scatter_(-1, top_idx, True)
    value_mask = pair_mask.unsqueeze(-1).expand_as(groups)
    return (groups * value_mask).reshape_as(x)


def global_nvfp4_scale(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.abs().max().float() / (448.0 * 6.0), min=1e-9)


def raw_byte_violation_rate(q_sparse: torch.Tensor) -> float:
    byte_chunks = q_sparse.view(q_sparse.size(0), -1, 4)
    nonzero_raw_bytes = (byte_chunks != 0).sum(dim=-1)
    return (nonzero_raw_bytes > 2).float().mean().item()


def metric_dict(pred: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    diff = pred.float() - ref.float()
    mse = diff.square().mean().item()
    rmse = mse ** 0.5
    ref_rms = ref.float().square().mean().sqrt().item()
    return {
        "mse": mse,
        "rel_rmse": rmse / max(ref_rms, 1e-12),
        "mean_abs": diff.abs().mean().item(),
        "max_abs": diff.abs().max().item(),
    }


def print_metrics(title: str, pred: torch.Tensor, ref: torch.Tensor) -> None:
    metrics = metric_dict(pred, ref)
    print(title)
    print(f"  mse      : {metrics['mse']:.8f}")
    print(f"  rel_rmse : {metrics['rel_rmse']:.8f}")
    print(f"  mean_abs : {metrics['mean_abs']:.8f}")
    print(f"  max_abs  : {metrics['max_abs']:.8f}")


def make_zero_x(m: int, k: int, device: torch.device) -> torch.Tensor:
    return torch.zeros((m, k), device=device, dtype=torch.bfloat16)


def make_one_x(m: int, k: int, device: torch.device) -> torch.Tensor:
    return torch.ones((m, k), device=device, dtype=torch.bfloat16)


def make_pair_friendly_x(m: int, k: int, device: torch.device) -> torch.Tensor:
    groups = torch.zeros((m, k // 8, 4, 2), device=device, dtype=torch.float32)
    groups[..., 0, 0] = 9.0
    groups[..., 0, 1] = 7.5
    groups[..., 2, 0] = -8.0
    groups[..., 2, 1] = -6.5
    groups += 0.02 * torch.randn_like(groups)
    return groups.view(m, k).to(torch.bfloat16)


def make_alternating_pair_x(m: int, k: int, device: torch.device) -> torch.Tensor:
    groups = torch.zeros((m, k // 8, 4, 2), device=device, dtype=torch.float32)
    groups[..., 1, 0] = 6.0
    groups[..., 1, 1] = -5.5
    groups[..., 3, 0] = -7.0
    groups[..., 3, 1] = 6.5
    groups += 0.01 * torch.randn_like(groups)
    return groups.view(m, k).to(torch.bfloat16)


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


def make_eye_w(n: int, k: int, device: torch.device) -> torch.Tensor:
    return torch.eye(n, k, device=device, dtype=torch.bfloat16)


def make_ones_w(n: int, k: int, device: torch.device) -> torch.Tensor:
    return torch.ones((n, k), device=device, dtype=torch.bfloat16)


def make_random_w(n: int, k: int, device: torch.device) -> torch.Tensor:
    return torch.randn((n, k), device=device, dtype=torch.bfloat16) * 0.1


def make_block_repeat_w(n: int, k: int, device: torch.device) -> torch.Tensor:
    w = torch.randn((n, k), device=device, dtype=torch.bfloat16) * 0.1
    for base in range(0, k, 32):
        if base + 32 <= k:
            w[:, base + 16: base + 32] = w[:, base: base + 16]
    return w


def run_case(name: str, x: torch.Tensor, w: torch.Tensor, agemm) -> None:
    m, k = x.shape
    n = w.shape[0]

    y_ref = torch.matmul(x.float(), w.float().t())
    scale_x = global_nvfp4_scale(x)
    scale_w = global_nvfp4_scale(w)
    x_scaled = (x / scale_x).to(torch.bfloat16)
    w_scaled = (w / scale_w).to(torch.bfloat16)
    output_scale = float(scale_x * scale_w)

    x_sparse_pseudo = top2_pairs_8_maxabs(x_scaled.float())
    x_sparse_q32 = quantize_nvfp4_tensor(x_sparse_pseudo, group_size=32)
    x_res_q16 = quantize_nvfp4_tensor(x_scaled.float() - x_sparse_q32, group_size=16)
    w32_pseudo = quantize_nvfp4_tensor(w_scaled.float(), group_size=32)

    y_pseudo_sparse = F.linear(x_sparse_q32, w32_pseudo) * output_scale
    y_pseudo_res = F.linear(x_res_q16, w32_pseudo) * output_scale
    y_pseudo_total = y_pseudo_sparse + y_pseudo_res

    qw32, sfw_sparse32, sfw_dense16 = agemm.quantize_w32_shared(w_scaled)
    q_sparse_raw, sfa_sparse, q_res, sf_res = agemm.fused_sparse_residual_quantize_x_debug(x_scaled, n)
    a_comp, e = agemm.compress_sparse_a(q_sparse_raw, n)
    sfa_dense16 = agemm.duplicate_sfa32_to_sfa16(sfa_sparse, m, n, k)

    y_sparse_dense_decode = agemm.matmul(q_sparse_raw, qw32, sfa_dense16, sfw_dense16, output_scale).float()
    y_kernel_sparse = agemm.sparse_matmul(a_comp, qw32, e, sfa_sparse, sfw_sparse32, m, n, k, output_scale).float()
    y_kernel_res = agemm.matmul(q_res, qw32, sf_res, sfw_dense16, output_scale).float()
    y_kernel_total = y_kernel_sparse + y_kernel_res

    print(name)
    print(f"  q_sparse raw-byte violation : {raw_byte_violation_rate(q_sparse_raw):.8f}")
    print(f"  ref checksum [0:16]         : {y_ref.flatten()[:16].sum().item():.8f}")
    print(f"  kernel total checksum [0:16]: {y_kernel_total.flatten()[:16].sum().item():.8f}")
    print(f"  pseudo total checksum [0:16]: {y_pseudo_total.flatten()[:16].sum().item():.8f}")
    print_metrics("  sparse_gemm vs dense_decode", y_kernel_sparse, y_sparse_dense_decode)
    print_metrics("  kernel_sparse vs pseudo_sparse", y_kernel_sparse, y_pseudo_sparse)
    print_metrics("  kernel_res vs pseudo_res", y_kernel_res, y_pseudo_res)
    print_metrics("  kernel_total vs pseudo_total", y_kernel_total, y_pseudo_total)
    print_metrics("  kernel_total vs bf16_ref", y_kernel_total, y_ref)


def main() -> None:
    parser = argparse.ArgumentParser(description="Matrix suite for pair-top2 fused sparse+residual correctness checks.")
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.k % 128 != 0 or args.n % 128 != 0:
        raise ValueError(f"n and k must be multiples of 128, got n={args.n}, k={args.k}")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    agemm = load_agemm()

    cases = [
        ("zero x eye", make_zero_x(args.m, args.k, device), make_eye_w(args.n, args.k, device)),
        ("ones x eye", make_one_x(args.m, args.k, device), make_eye_w(args.n, args.k, device)),
        ("ones x ones", make_one_x(args.m, args.k, device), make_ones_w(args.n, args.k, device)),
        ("pair-friendly x eye", make_pair_friendly_x(args.m, args.k, device), make_eye_w(args.n, args.k, device)),
        ("pair-friendly x random_w", make_pair_friendly_x(args.m, args.k, device), make_random_w(args.n, args.k, device)),
        ("alternating-pair x block-repeat_w", make_alternating_pair_x(args.m, args.k, device), make_block_repeat_w(args.n, args.k, device)),
        ("outlier-rich x random_w", make_outlier_rich_x(args.m, args.k, device), make_random_w(args.n, args.k, device)),
    ]

    print(f"problem template: M={args.m}, N={args.n}, K={args.k}, seed={args.seed}")
    print()
    for idx, (name, x, w) in enumerate(cases):
        run_case(name, x, w, agemm)
        if idx != len(cases) - 1:
            print()


if __name__ == "__main__":
    main()
