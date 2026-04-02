# vLLM/vllm/model_executor/layers/quantization/sharq.py

import os
from pathlib import Path
from typing import Any, Tuple

import torch
from torch.nn.parameter import Parameter

import sys
_THIS_DIR = Path(__file__).resolve().parent


def _kernel_build_candidates() -> list[Path]:
    candidates: list[Path] = []

    env_build = os.environ.get("SHARQ_VLLM_KERNEL_BUILD")
    if env_build:
        candidates.append(Path(env_build))

    candidates.append(_THIS_DIR / 'kernels' / 'build_cmake_sm120a')

    for parent in _THIS_DIR.parents:
        candidates.append(parent / 'temp-vllm' / 'kernels' / 'build_cmake_sm120a')

    candidates.append(Path('/root/autodl-tmp/SharQ/kernels/build_cmake_sm120a'))
    return candidates


for _candidate in _kernel_build_candidates():
    if _candidate.exists():
        sys.path.insert(0, os.fspath(_candidate))

try:
    import sharq_ops_vllm as sharq_ops
except ImportError:
    import sharq_ops

_HAS_FUSED_SHARQ_LINEAR = hasattr(sharq_ops, "sharq_linear_forward")

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


NVFP4_DENOM = 448.0 * 6.0


def _to_python_float(scale: torch.Tensor | float) -> float:
    if torch.is_tensor(scale):
        return float(scale.detach().float().cpu().item())
    return float(scale)


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _get_dense_scale_bytes(num_rows: int, k_dim: int) -> int:
    return ((num_rows // 128) + 1) * 128 * k_dim // 16


def _get_weight_sparse_scale_bytes(output_size: int, input_size: int) -> int:
    # bindings.cpp::get_sparse_nvfp4_buffer_sizes returns
    # (A_comp_bytes, E_bytes, SFA_bytes, SFB_bytes).
    _, _, _, sfb_bytes = sharq_ops.get_sparse_nvfp4_buffer_sizes(1, output_size, input_size)
    return int(sfb_bytes)


# ==============================================================================
# Custom Ops Registration
# ==============================================================================

def sharq_fused_sparse_residual_quantize_x(
    x: torch.Tensor,
    out_features: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scale_a = torch.clamp(x.abs().max().float() / NVFP4_DENOM, min=1e-9).view(1)
    a_comp, e, sfa_sparse, q_res, sf_res = sharq_ops.fused_sparse_residual_quantize_x(
        (x / scale_a).to(torch.bfloat16).contiguous(),
        int(out_features),
    )
    return a_comp, e, sfa_sparse, q_res, sf_res, scale_a


def sharq_fused_sparse_residual_quantize_x_fake(
    x: torch.Tensor,
    out_features: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = x.shape[0]
    hidden_dim = x.shape[1]
    # Match kernels/include/sparse_nvfp4.h:
    # - A_comp: M * round_up(K, 256) / 4
    # - E: round_up(M, 128) * round_up(K, 256) / 16
    # - SFA/SF_res: (M // 128 + 1) * 128 * K / 16
    aligned_tokens_e = _round_up(num_tokens, 128)
    aligned_hidden_a = _round_up(hidden_dim, 256)
    aligned_hidden_e = _round_up(hidden_dim, 256)
    padded_tokens_scale = (num_tokens // 128 + 1) * 128
    device = x.device
    return (
        torch.empty((num_tokens * aligned_hidden_a // 4,), dtype=torch.uint8, device=device),
        torch.empty((aligned_tokens_e * aligned_hidden_e // 16,), dtype=torch.uint8, device=device),
        torch.empty((padded_tokens_scale * hidden_dim // 16,), dtype=torch.uint8, device=device),
        torch.empty((num_tokens, hidden_dim // 2), dtype=torch.uint8, device=device),
        torch.empty((padded_tokens_scale * hidden_dim // 16,), dtype=torch.uint8, device=device),
        torch.empty((1,), dtype=torch.float32, device=device),
    )
def sharq_sparse_matmul(
    a_comp: torch.Tensor,
    b: torch.Tensor,
    e: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    m: int,
    n: int,
    k: int,
    scale: float,
) -> torch.Tensor:
    return sharq_ops.sparse_matmul(
        a_comp,
        b,
        e,
        sfa,
        sfb,
        int(m),
        int(n),
        int(k),
        alpha=float(scale),
    )



def sharq_sparse_matmul_fake(
    a_comp: torch.Tensor,
    b: torch.Tensor,
    e: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    m: int,
    n: int,
    k: int,
    scale: float,
) -> torch.Tensor:
    return torch.empty((m, n), dtype=torch.bfloat16, device=a_comp.device)
def sharq_matmul_accum(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    scale: float,
    c_in: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    return sharq_ops.matmul_accum(
        a,
        b,
        sfa,
        sfb,
        float(scale),
        c_in,
        float(beta),
    )



def sharq_matmul_accum_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    scale: float,
    c_in: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    return torch.empty_like(c_in, dtype=torch.bfloat16, device=a.device)


def sharq_linear_forward(
    x: torch.Tensor,
    qw: torch.Tensor,
    sfw_sparse: torch.Tensor,
    sfw_dense: torch.Tensor,
    weight_scale: float,
) -> torch.Tensor:
    if _HAS_FUSED_SHARQ_LINEAR:
        return sharq_ops.sharq_linear_forward(
            x,
            qw,
            sfw_sparse,
            sfw_dense,
            float(weight_scale),
        )

    scale_a = torch.clamp(x.abs().max().float() / NVFP4_DENOM, min=1e-9).view(1)
    x_scaled = (x / scale_a).to(torch.bfloat16).contiguous()
    out_features = int(qw.shape[0])
    a_comp, e, sfa_sparse, q_res, sf_res = sharq_ops.fused_sparse_residual_quantize_x(
        x_scaled,
        out_features,
    )
    y_sparse = sharq_ops.sparse_matmul(
        a_comp,
        qw,
        e,
        sfa_sparse,
        sfw_sparse,
        int(x.shape[0]),
        out_features,
        int(x.shape[1]),
        alpha=float(weight_scale),
    )
    out = sharq_ops.matmul_accum(
        q_res,
        qw,
        sf_res,
        sfw_dense,
        float(weight_scale),
        y_sparse,
        1.0,
    )
    return out * scale_a.to(out.dtype)


def sharq_linear_forward_fake(
    x: torch.Tensor,
    qw: torch.Tensor,
    sfw_sparse: torch.Tensor,
    sfw_dense: torch.Tensor,
    weight_scale: float,
) -> torch.Tensor:
    return torch.empty((x.shape[0], qw.shape[0]), dtype=torch.bfloat16, device=x.device)


direct_register_custom_op(
    op_name='sharq_fused_sparse_residual_quantize_x_v2',
    op_func=sharq_fused_sparse_residual_quantize_x,
    mutates_args=[],
    fake_impl=sharq_fused_sparse_residual_quantize_x_fake,
    dispatch_key=current_platform.dispatch_key,
)


direct_register_custom_op(
    op_name='sharq_sparse_matmul_v2',
    op_func=sharq_sparse_matmul,
    mutates_args=[],
    fake_impl=sharq_sparse_matmul_fake,
    dispatch_key=current_platform.dispatch_key,
)


direct_register_custom_op(
    op_name='sharq_matmul_accum_v2',
    op_func=sharq_matmul_accum,
    mutates_args=[],
    fake_impl=sharq_matmul_accum_fake,
    dispatch_key=current_platform.dispatch_key,
)


direct_register_custom_op(
    op_name='sharq_linear_forward_v1',
    op_func=sharq_linear_forward,
    mutates_args=[],
    fake_impl=sharq_linear_forward_fake,
    dispatch_key=current_platform.dispatch_key,
)


# ==============================================================================
# SHARQ Configuration
# ==============================================================================

class SHARQConfig(QuantizationConfig):
    """Config class for SHARQ quantization."""

    def __init__(self, **kwargs) -> None:
        super().__init__()

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return 'sharq'

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 120

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ['quantize_config.json']

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'SHARQConfig':
        return cls(**config)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> LinearMethodBase | None:
        if isinstance(layer, LinearBase):
            return SHARQLinearMethod(self, prefix=prefix)
        return None


# ==============================================================================
# SHARQ Linear Method
# ==============================================================================

class SHARQLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: SHARQConfig, prefix: str = ''):
        self.quant_config = quant_config
        self.prefix = prefix

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if input_size % 128 != 0:
            raise ValueError(
                f'SHARQ requires input_size to be a multiple of 128, got {input_size} for {self.prefix}'
            )
        if input_size % 32 != 0:
            raise ValueError(
                f'SHARQ requires input_size to be a multiple of 32, got {input_size} for {self.prefix}'
            )

        device = 'cuda'
        layer.input_size = input_size
        layer.output_size = output_size
        layer.scale_b_value = 1.0

        def register_param(name, shape, dtype):
            def weight_loader(param: Parameter, loaded_weight: torch.Tensor, *args):
                if param.numel() == 0:
                    return

                real_weight = loaded_weight.to(device=param.device, dtype=param.dtype)
                if real_weight.dim() == 0 and param.numel() == 1:
                    real_weight = real_weight.view(1)
                if real_weight.dim() == 1 and real_weight.numel() == param.numel():
                    real_weight = real_weight.view(param.shape)
                if real_weight.shape != param.shape:
                    raise ValueError(
                        f'Shape mismatch for {self.prefix}: expected {param.shape}, got {real_weight.shape}'
                    )
                param.data.copy_(real_weight)
                if name == 'scale_b':
                    layer.scale_b_value = float(real_weight.reshape(-1)[0].float().cpu().item())

            param = Parameter(torch.empty(shape, dtype=dtype, device=device), requires_grad=False)
            set_weight_attrs(param, {**extra_weight_attrs, 'weight_loader': weight_loader})
            layer.register_parameter(name, param)

        sfw_sparse_bytes = _get_weight_sparse_scale_bytes(output_size, input_size)
        sfw_dense_bytes = _get_dense_scale_bytes(output_size, input_size)
        register_param('QW', (output_size, input_size // 2), torch.uint8)
        register_param('SFW_sparse', (sfw_sparse_bytes,), torch.uint8)
        register_param('SFW_dense', (sfw_dense_bytes,), torch.uint8)
        register_param('scale_b', (1,), torch.float32)

    def apply(self, layer, x, bias=None):
        x_2d = x.view(-1, x.shape[-1]).contiguous()
        weight_scale = layer.scale_b_value
        out = torch.ops.vllm.sharq_linear_forward_v1(
            x_2d,
            layer.QW,
            layer.SFW_sparse,
            layer.SFW_dense,
            weight_scale,
        )

        if bias is not None:
            out.add_(bias)
        return out.view(*x.shape[:-1], -1)













