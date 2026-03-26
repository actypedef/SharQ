import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SHARQ_OPS = None


@torch.no_grad()
def quantize_int_group(tensor: torch.Tensor, nbits: int, group_size: int) -> torch.Tensor:
    saved_shape = tensor.shape
    tensor = tensor.reshape(-1, group_size)

    tensor_max = tensor.amax(dim=-1, keepdim=True)
    tensor_min = tensor.amin(dim=-1, keepdim=True)
    q_max = (2**nbits) - 1
    q_min = 0

    scales = (tensor_max - tensor_min).clamp(min=1e-5) / q_max
    zero_point = torch.round(-tensor_min / scales).clamp_(min=q_min, max=q_max)
    tensor = (torch.clamp(torch.round(tensor / scales) + zero_point, q_min, q_max) - zero_point) * scales
    return tensor.reshape(saved_shape)


def load_sharq_ops():
    global _SHARQ_OPS
    if _SHARQ_OPS is not None:
        return _SHARQ_OPS

    for build_dir in (_REPO_ROOT / "kernels" / "build_cmake_sm120a", _REPO_ROOT / "kernels" / "build"):
        build_dir_str = str(build_dir)
        if build_dir_str not in sys.path:
            sys.path.append(build_dir_str)
        try:
            _SHARQ_OPS = __import__("sharq_ops")
            return _SHARQ_OPS
        except ImportError:
            continue

    raise ImportError("Failed to import sharq_ops from kernels/build_cmake_sm120a or kernels/build")


def global_nvfp4_scale(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.abs().max().float() / (448.0 * 6.0), min=1e-9)


def apply_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    weight_float = weight.float()
    inv_rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_float * inv_rms * weight_float).to(torch.bfloat16)


def get_rmsnorm_weight_eps(norm_module):
    norm = getattr(norm_module, "original_norm", norm_module)
    eps = getattr(norm, "variance_epsilon", None)
    if eps is None:
        eps = getattr(norm, "eps", None)
    if eps is None:
        raise AttributeError(f"Unsupported RMSNorm module: {type(norm)}")
    return norm.weight, float(eps)


def top2_pairs_8_maxabs(x: torch.Tensor) -> torch.Tensor:
    groups = x.view(*x.shape[:-1], x.shape[-1] // 8, 4, 2)
    pair_scores = groups.abs().amax(dim=-1)
    top_idx = pair_scores.topk(k=2, dim=-1).indices
    pair_mask = torch.zeros_like(pair_scores, dtype=torch.bool)
    pair_mask.scatter_(-1, top_idx, True)
    value_mask = pair_mask.unsqueeze(-1).expand_as(groups)
    return (groups * value_mask).reshape_as(x)


def quantize_e2m1(tensor: torch.Tensor) -> torch.Tensor:
    representable_vals = torch.tensor(
        [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=tensor.device,
        dtype=tensor.dtype,
    )
    best = torch.full_like(tensor, representable_vals[0])
    best_diff = torch.abs(tensor - representable_vals[0])
    for value in representable_vals[1:]:
        diff = torch.abs(tensor - value)
        mask = diff < best_diff
        best = torch.where(mask, value, best)
        best_diff = torch.where(mask, diff, best_diff)
    return best


def quantize_ue4m3(tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.clamp(tensor, min=2e-3, max=448.0)
    exponent = torch.floor(torch.log2(tensor + 1e-9))
    mantissa_val = tensor / (2**exponent) - 1.0
    quantized_mantissa_val = torch.round(mantissa_val * 8) / 8
    return (1 + quantized_mantissa_val) * (2**exponent)


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


def to_python_float(x) -> float:
    if torch.is_tensor(x):
        return float(x.detach().float().cpu().item())
    return float(x)


@torch.no_grad()
def quantize_weight_nvfp4(weight: torch.Tensor):
    scale = global_nvfp4_scale(weight)
    sharq_ops = load_sharq_ops()
    q_weight, scale_weight = sharq_ops.quantize_w_nvfp4((weight / scale).to(torch.bfloat16))
    return q_weight, scale_weight, scale


@torch.no_grad()
def quantize_weight_shared_nvfp4(weight: torch.Tensor):
    scale = global_nvfp4_scale(weight)
    sharq_ops = load_sharq_ops()
    q_weight, scale_weight_sparse, scale_weight_dense = sharq_ops.quantize_w32_shared((weight / scale).to(torch.bfloat16))
    return q_weight, scale_weight_sparse, scale_weight_dense, scale


@torch.no_grad()
def quantize_weight_sharq_sim(weight: torch.Tensor):
    weight_tensor = weight.detach().to(dtype=torch.bfloat16)
    weight_scale = global_nvfp4_scale(weight_tensor)
    weight_sim_q32 = quantize_nvfp4_tensor((weight_tensor / weight_scale).float(), group_size=32).to(torch.bfloat16)
    return weight_sim_q32, weight_scale


@torch.no_grad()
def quantize_activation_nvfp4(x: torch.Tensor):
    scale = global_nvfp4_scale(x)
    sharq_ops = load_sharq_ops()
    q_x, scale_x = sharq_ops.quantize_x_nvfp4((x / scale).to(torch.bfloat16))
    return q_x, scale_x, scale


@torch.no_grad()
def quantize_activation_sparse_residual_nvfp4(x: torch.Tensor, out_features: int):
    scale = global_nvfp4_scale(x)
    sharq_ops = load_sharq_ops()
    a_comp, e, sfa_sparse, q_res, sf_res = sharq_ops.fused_sparse_residual_quantize_x((x / scale).to(torch.bfloat16), out_features)
    return a_comp, e, sfa_sparse, q_res, sf_res, scale


@torch.no_grad()
def quantize_activation_rmsnorm_sparse_residual_nvfp4(
    x: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    rmsnorm_eps: float,
    out_features: int,
):
    sharq_ops = load_sharq_ops()
    a_comp, e, sfa_sparse, q_res, sf_res, scale = sharq_ops.fused_rmsnorm_sparse_residual_quantize_x(
        x.to(torch.bfloat16),
        rmsnorm_weight.to(torch.bfloat16),
        float(rmsnorm_eps),
        out_features,
    )
    return a_comp, e, sfa_sparse, q_res, sf_res, scale


@torch.no_grad()
def quantize_activation_sharq_sim(x: torch.Tensor):
    x_float = x.float()
    scale_x = global_nvfp4_scale(x)
    x_scaled = x_float / scale_x
    x_sparse = top2_pairs_8_maxabs(x_scaled)
    x_sparse_q32 = quantize_nvfp4_tensor(x_sparse, group_size=32).to(torch.bfloat16)
    x_res_q16 = quantize_nvfp4_tensor(x_scaled - x_sparse_q32.float(), group_size=16).to(torch.bfloat16)
    return x_sparse_q32, x_res_q16, scale_x


__all__ = [
    "apply_rmsnorm",
    "get_rmsnorm_weight_eps",
    "global_nvfp4_scale",
    "load_sharq_ops",
    "quantize_activation_nvfp4",
    "quantize_activation_rmsnorm_sparse_residual_nvfp4",
    "quantize_activation_sharq_sim",
    "quantize_activation_sparse_residual_nvfp4",
    "quantize_int_group",
    "quantize_nvfp4_tensor",
    "quantize_weight_nvfp4",
    "quantize_weight_sharq_sim",
    "quantize_weight_shared_nvfp4",
    "to_python_float",
    "top2_pairs_8_maxabs",
]
