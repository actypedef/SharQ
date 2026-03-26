import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SHARQ_OPS = None


def _load_sharq_ops():
    global _SHARQ_OPS
    if _SHARQ_OPS is not None:
        return _SHARQ_OPS

    for build_dir in (_REPO_ROOT / "kernels" / "build_cmake_sm120a", _REPO_ROOT / "kernels" / "build"):
        build_dir_str = str(build_dir)
        if build_dir_str not in sys.path:
            sys.path.append(build_dir_str)
        for module_name in ("sharq_ops", "agemm"):
            try:
                module = __import__(module_name)
                _SHARQ_OPS = module
                return _SHARQ_OPS
            except ImportError:
                continue
    raise ImportError("Failed to import sharq_ops or agemm from kernels/build_cmake_sm120a or kernels/build")


def _global_nvfp4_scale(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.abs().max().float() / (448.0 * 6.0), min=1e-9)


def _top2_4(x: torch.Tensor) -> torch.Tensor:
    groups = x.view(-1, x.shape[-1] // 4, 4)
    top2_indices = torch.topk(groups.abs(), k=2, dim=-1).indices
    mask = torch.zeros_like(groups, dtype=torch.bool)
    mask.scatter_(dim=-1, index=top2_indices, src=torch.ones_like(top2_indices, dtype=torch.bool))
    return (groups * mask).view_as(x)


def _top2_pairs_8_maxabs(x: torch.Tensor) -> torch.Tensor:
    groups = x.view(*x.shape[:-1], x.shape[-1] // 8, 4, 2)
    pair_scores = groups.abs().amax(dim=-1)
    top_idx = pair_scores.topk(k=2, dim=-1).indices
    pair_mask = torch.zeros_like(pair_scores, dtype=torch.bool)
    pair_mask.scatter_(-1, top_idx, True)
    value_mask = pair_mask.unsqueeze(-1).expand_as(groups)
    return (groups * value_mask).reshape_as(x)


def _quantize_e2m1(tensor: torch.Tensor) -> torch.Tensor:
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


def _quantize_ue4m3(tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.clamp(tensor, min=2e-3, max=448.0)
    exponent = torch.floor(torch.log2(tensor + 1e-9))
    mantissa_val = tensor / (2**exponent) - 1.0
    quantized_mantissa_val = torch.round(mantissa_val * 8) / 8
    return (1 + quantized_mantissa_val) * (2**exponent)


def _quantize_nvfp4_tensor(tensor: torch.Tensor, group_size: int) -> torch.Tensor:
    original_shape = tensor.shape
    padding = (group_size - tensor.shape[-1] % group_size) % group_size
    if padding != 0:
        tensor = F.pad(tensor, (0, padding))

    reshaped = tensor.view(-1, group_size)
    max_abs = reshaped.abs().max(dim=1, keepdim=True)[0]
    scale = max_abs / 6.0
    scale[scale == 0] = 1e-9
    dq_scale = _quantize_ue4m3(scale)
    normalized = reshaped / dq_scale
    q = _quantize_e2m1(normalized)
    out = (q * dq_scale).view(tensor.shape)

    if padding != 0:
        out = out[..., :-padding]
    return out.view(original_shape)


def _to_python_float(x) -> float:
    if torch.is_tensor(x):
        return float(x.detach().float().cpu().item())
    return float(x)


def find_qlinear_layers(module: nn.Module, name: str = ""):
    if isinstance(module, (QLinearLayer, QLinearLayerFused)):
        if getattr(module, "enable_quant", True):
            return {name: module}
    result = {}
    for child_name, child in module.named_children():
        full_name = child_name if not name else f"{name}.{child_name}"
        result.update(find_qlinear_layers(child, name=full_name))
    return result


def quantize_weight_nvfp4(weight: torch.Tensor):
    scale = _global_nvfp4_scale(weight)
    sharq_ops = _load_sharq_ops()
    q_weight, scale_weight = sharq_ops.quantize_w_nvfp4((weight / scale).to(torch.bfloat16))
    return q_weight, scale_weight, scale


def quantize_weight_shared_nvfp4(weight: torch.Tensor):
    scale = _global_nvfp4_scale(weight)
    sharq_ops = _load_sharq_ops()
    q_weight, scale_weight_sparse, scale_weight_dense = sharq_ops.quantize_w32_shared(
        (weight / scale).to(torch.bfloat16)
    )
    return q_weight, scale_weight_sparse, scale_weight_dense, scale


def quantize_activation_nvfp4(x: torch.Tensor):
    scale = _global_nvfp4_scale(x)
    sharq_ops = _load_sharq_ops()
    q_x, scale_x = sharq_ops.quantize_x_nvfp4((x / scale).to(torch.bfloat16))
    return q_x, scale_x, scale


def quantize_activation_sparse_residual_nvfp4(x: torch.Tensor, out_features: int):
    scale = _global_nvfp4_scale(x)
    sharq_ops = _load_sharq_ops()
    a_comp, e, sfa_sparse, q_res, sf_res = sharq_ops.fused_sparse_residual_quantize_x(
        (x / scale).to(torch.bfloat16), out_features
    )
    return a_comp, e, sfa_sparse, q_res, sf_res, scale


class QLinearLayer(nn.Module):
    def __init__(
        self,
        original_layer: nn.Linear,
        select_num=None,
        reorder_index=None,
        out_reorder_index=None,
        quant_type: str = "NVFP4",
    ):
        super().__init__()

        if quant_type not in {"NVFP4", "NVFP4_FUSED", "SHARQ", "SHARQ_SIM", "NVFP4_SIM"}:
            raise ValueError(f"Unsupported quant_type: {quant_type}")

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        if quant_type in {"SHARQ", "NVFP4_FUSED"}:
            self.quant_type = "SHARQ"
        elif quant_type in {"SHARQ_SIM", "NVFP4_SIM"}:
            self.quant_type = "SHARQ_SIM"
        else:
            self.quant_type = quant_type
        self.enable_quant = True

        if original_layer.bias is not None:
            self.register_buffer("bias", original_layer.bias.detach().clone())
        else:
            self.bias = None

        if self.quant_type == "SHARQ_SIM":
            weight_device = original_layer.weight.device
            weight_tensor = original_layer.weight.detach().to(device=weight_device, dtype=torch.bfloat16)
            self.weight_scale = _global_nvfp4_scale(weight_tensor)
            self.weight_sim_q32 = _quantize_nvfp4_tensor((weight_tensor / self.weight_scale).float(), group_size=32).to(
                torch.bfloat16
            )
            del weight_tensor
        else:
            weight_gpu = original_layer.weight.detach().to(device="cuda", dtype=torch.bfloat16)

            if self.quant_type == "SHARQ":
                self.weight_q, self.scale_w_sparse, self.scale_w_dense, self.weight_scale = quantize_weight_shared_nvfp4(
                    weight_gpu
                )
            else:
                self.weight_q, self.scale_w, self.weight_scale = quantize_weight_nvfp4(weight_gpu)
                identity = torch.eye(self.in_features, dtype=torch.bfloat16, device=weight_gpu.device)
                self.identity_q, self.identity_scale_q, self.identity_scale = quantize_weight_nvfp4(identity)
                del identity

            del weight_gpu
            torch.cuda.empty_cache()

    @torch.no_grad()
    def forward(self, x):
        if isinstance(x, tuple):
            x_orig, bsz, q_len = x
        else:
            x_orig = x
            bsz = None
            q_len = x.shape[0]

        in_features = x_orig.shape[-1]
        x_orig_reshaped = x_orig.reshape(-1, in_features).to(torch.bfloat16)

        if self.quant_type == "SHARQ_SIM":
            x_float = x_orig_reshaped.float()
            scale_x = _global_nvfp4_scale(x_orig_reshaped)
            x_scaled = x_float / scale_x
            x_sparse = _top2_pairs_8_maxabs(x_scaled)
            x_sparse_q32 = _quantize_nvfp4_tensor(x_sparse, group_size=32).to(torch.bfloat16)
            x_res_q16 = _quantize_nvfp4_tensor(x_scaled - x_sparse_q32.float(), group_size=16).to(torch.bfloat16)
            output_scale = scale_x * self.weight_scale
            y_sparse = F.linear(x_sparse_q32, self.weight_sim_q32)
            y_res = F.linear(x_res_q16, self.weight_sim_q32)
            y = (y_sparse + y_res) * output_scale
        elif self.quant_type == "SHARQ":
            a_comp, e, sfa_sparse, qx_res, scale_x_res, scale_x = quantize_activation_sparse_residual_nvfp4(
                x_orig_reshaped, self.out_features
            )
            output_scale = _to_python_float(scale_x * self.weight_scale)
            sharq_ops = _load_sharq_ops()

            y_sparse = sharq_ops.sparse_matmul(
                a_comp,
                self.weight_q,
                e,
                sfa_sparse,
                self.scale_w_sparse,
                x_orig_reshaped.shape[0],
                self.out_features,
                self.in_features,
                alpha=output_scale,
            )
            y_res = sharq_ops.matmul(qx_res, self.weight_q, scale_x_res, self.scale_w_dense, output_scale)
            y = y_sparse + y_res
        else:
            x_sparse = _top2_4(x_orig_reshaped)
            qx_sparse, scale_x_sparse, scale_sparse = quantize_activation_nvfp4(x_sparse)
            sharq_ops = _load_sharq_ops()

            x_approx = sharq_ops.matmul(
                qx_sparse,
                self.identity_q,
                scale_x_sparse,
                self.identity_scale_q,
                _to_python_float(scale_sparse * self.identity_scale),
            )
            residual = x_orig_reshaped.float() - x_approx.float()
            qx_res, scale_x_res, scale_res = quantize_activation_nvfp4(residual.to(torch.bfloat16))

            y_sparse = sharq_ops.matmul(
                qx_sparse,
                self.weight_q,
                scale_x_sparse,
                self.scale_w,
                _to_python_float(scale_sparse * self.weight_scale),
            )
            y_res = sharq_ops.matmul(
                qx_res,
                self.weight_q,
                scale_x_res,
                self.scale_w,
                _to_python_float(scale_res * self.weight_scale),
            )
            y = y_sparse + y_res

        if self.bias is not None:
            bias = self.bias.float() if self.quant_type == "SHARQ_SIM" else self.bias
            y = y + bias

        if self.quant_type == "SHARQ_SIM":
            y = y.to(x_orig_reshaped.dtype)

        if bsz is not None:
            return y.reshape(bsz, q_len, -1)
        return y.reshape(q_len, -1)


class QLinearLayerFused(QLinearLayer):
    def __init__(
        self,
        original_layer: nn.Linear,
        select_num=None,
        reorder_index=None,
        out_reorder_index=None,
        quant_type: str = "SHARQ",
    ):
        super().__init__(
            original_layer=original_layer,
            select_num=select_num,
            reorder_index=reorder_index,
            out_reorder_index=out_reorder_index,
            quant_type=quant_type,
        )
