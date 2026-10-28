"""SharQ quantization method for vLLM (registered as an out-of-tree plugin).

Ported from benchmarks/vllm/sharq.py (which was written as an in-tree vLLM patch)
and wired to the CUDA-graph safe `sharq_ops_vllm.sharq_linear_forward` op.

Checkpoint format (produced by benchmarks/vllm/sharq_quant.py):
    <prefix>.QW          uint8 [N, K/2]   shared NVFP4 payload (32-group quantized)
    <prefix>.SFW_sparse  uint8           UE4M3 32-group scales, sparse-GEMM SFB layout
    <prefix>.SFW_dense   uint8           same scales duplicated to 16-group, dense SFB layout
    <prefix>.scale_b     fp32 [1]        per-tensor weight scale
QKV and gate/up are stored pre-merged (qkv_proj / gate_up_proj), TP=1 only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn.parameter import Parameter

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _kernel_build_candidates() -> list[Path]:
    candidates: list[Path] = []
    for env_var in ("SHARQ_VLLM_KERNEL_BUILD", "SHARQ_KERNEL_BUILD"):
        env_build = os.environ.get(env_var)
        if env_build:
            candidates.append(Path(env_build))
    candidates.append(_REPO_ROOT / "kernels" / "build_vllm2_sm120a")
    candidates.append(_REPO_ROOT / "kernels" / "build_vllm_sm120a")
    candidates.append(_REPO_ROOT / "kernels" / "build_vllm_sm100a")
    return candidates


for _candidate in reversed(_kernel_build_candidates()):
    if _candidate.exists() and os.fspath(_candidate) not in sys.path:
        sys.path.insert(0, os.fspath(_candidate))

import sharq_ops_vllm  # noqa: E402

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase  # noqa: E402
from vllm.model_executor.layers.quantization import register_quantization_config  # noqa: E402
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig  # noqa: E402
from vllm.model_executor.utils import set_weight_attrs  # noqa: E402
from vllm.platforms import current_platform  # noqa: E402
from vllm.utils.torch_utils import direct_register_custom_op  # noqa: E402


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _dense_scale_bytes(num_rows: int, k_dim: int) -> int:
    return ((num_rows // 128) + 1) * 128 * k_dim // 16


def _sparse_weight_scale_bytes(output_size: int, input_size: int) -> int:
    # Matches sparse_nvfp4::get_sfb_bytes(1, N, K).
    return ((output_size // 128) + 1) * 128 * input_size // 16


# ==============================================================================
# Custom op (opaque to torch.compile, captured inside vLLM CUDA graphs)
#
#   M <= stack limit : SharQ stacked operand [A_sparse; A_res] (2M x K, bit-identical operand
#                      values) -> ONE dense NVFP4 GEMM via FlashInfer mm_fp4 (CUTLASS backend,
#                      tactics autotuned by vLLM's FlashInfer autotune warmup) -> row-pair add.
#                      W is streamed once (decode is weight-bandwidth bound).
#   M >  stack limit : the paper's fused kernel: 4:8 sparse NVFP4 GEMM + dense residual GEMM
#                      accumulating in the epilogue (sharq_ops_vllm CUTLASS kernels).
#
#   SHARQ_VLLM_GEMM=cutlass forces the sharq_ops_vllm CUTLASS kernels for every M.
# ==============================================================================

_GEMM_BACKEND = os.environ.get("SHARQ_VLLM_GEMM", "flashinfer").lower()
_STACK_MAX_M = int(os.environ.get("SHARQ_STACK_MAX_M", "192"))
_STACK_MAX_M_LONG_K = int(os.environ.get("SHARQ_STACK_MAX_M_LONG_K", "256"))


def _stack_limit(k: int) -> int:
    return _STACK_MAX_M_LONG_K if k >= 8192 else _STACK_MAX_M


def _flashinfer_tuning_mode() -> bool:
    try:
        from flashinfer.autotuner import AutoTuner

        return bool(AutoTuner.get().is_tuning_mode)
    except Exception:
        return False


def _stacked_flashinfer(x, qw, sfw_dense_fi, weight_scale):
    from flashinfer import mm_fp4

    q2, sf2, alpha = sharq_ops_vllm.sharq_prepare_stacked_fast(x, weight_scale)
    y2 = mm_fp4(q2, qw.t(), sf2, sfw_dense_fi.t(), alpha, torch.bfloat16, backend="cutlass")
    return sharq_ops_vllm.sharq_pair_add(y2)


def sharq_linear_forward(
    x: torch.Tensor,
    qw: torch.Tensor,
    sfw_sparse: torch.Tensor,
    sfw_dense: torch.Tensor,
    sfw_dense_fi: torch.Tensor,
    weight_scale: float,
) -> torch.Tensor:
    weight_scale = float(weight_scale)
    if _GEMM_BACKEND == "flashinfer":
        m, k = x.shape
        limit = _stack_limit(k)
        if m <= limit:
            return _stacked_flashinfer(x, qw, sfw_dense_fi, weight_scale)
        if _flashinfer_tuning_mode():
            # vLLM's autotune warmup runs with M = max_num_batched_tokens; tune the stacked
            # GEMM tactics for every bucket up to 2 * limit rows for this (N, K).
            _stacked_flashinfer(x[:limit].contiguous(), qw, sfw_dense_fi, weight_scale)
    return sharq_ops_vllm.sharq_linear_forward(x, qw, sfw_sparse, sfw_dense, weight_scale)


def sharq_linear_forward_fake(
    x: torch.Tensor,
    qw: torch.Tensor,
    sfw_sparse: torch.Tensor,
    sfw_dense: torch.Tensor,
    sfw_dense_fi: torch.Tensor,
    weight_scale: float,
) -> torch.Tensor:
    return torch.empty((x.shape[0], qw.shape[0]), dtype=torch.bfloat16, device=x.device)


direct_register_custom_op(
    op_name="sharq_linear",
    op_func=sharq_linear_forward,
    mutates_args=[],
    fake_impl=sharq_linear_forward_fake,
    dispatch_key=current_platform.dispatch_key,
)


# ==============================================================================
# Config
# ==============================================================================


@register_quantization_config("sharq")
class SHARQConfig(QuantizationConfig):
    """Config class for SHARQ quantization."""

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return "SHARQConfig()"

    @classmethod
    def get_name(cls) -> str:
        return "sharq"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SHARQConfig":
        return cls(**{k: v for k, v in config.items() if k != "quant_method"})

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> LinearMethodBase | None:
        if isinstance(layer, LinearBase):
            return SHARQLinearMethod(self, prefix=prefix)
        return None


# ==============================================================================
# Linear method
# ==============================================================================


class SHARQLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: SHARQConfig, prefix: str = ""):
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
        output_size_per_partition = sum(output_partition_sizes)
        if input_size_per_partition != input_size or output_size_per_partition != output_size:
            raise ValueError("SHARQ vLLM integration currently supports tensor_parallel_size=1 only")
        if input_size % 256 != 0:
            raise ValueError(f"SHARQ requires input_size % 256 == 0, got {input_size} for {self.prefix}")

        layer.input_size = input_size
        layer.output_size = output_size
        layer.scale_b_value = 1.0
        device = torch.cuda.current_device()

        def register_param(name, shape, dtype):
            def weight_loader(param: Parameter, loaded_weight: torch.Tensor, *args, **kwargs):
                if args or kwargs.get("loaded_shard_id") is not None:
                    raise ValueError(
                        f"SHARQ checkpoints must store pre-merged qkv_proj/gate_up_proj ({self.prefix})"
                    )
                real_weight = loaded_weight.to(device=param.device, dtype=param.dtype)
                if real_weight.numel() == param.numel():
                    real_weight = real_weight.reshape(param.shape)
                if real_weight.shape != param.shape:
                    raise ValueError(
                        f"Shape mismatch for {self.prefix}.{name}: expected {tuple(param.shape)}, "
                        f"got {tuple(real_weight.shape)}"
                    )
                param.data.copy_(real_weight)
                if name == "scale_b":
                    layer.scale_b_value = float(real_weight.reshape(-1)[0].float().cpu().item())

            param = Parameter(torch.empty(shape, dtype=dtype, device=device), requires_grad=False)
            set_weight_attrs(param, {**extra_weight_attrs, "weight_loader": weight_loader})
            layer.register_parameter(name, param)

        register_param("QW", (output_size, input_size // 2), torch.uint8)
        register_param("SFW_sparse", (_sparse_weight_scale_bytes(output_size, input_size),), torch.uint8)
        register_param("SFW_dense", (_dense_scale_bytes(output_size, input_size),), torch.uint8)
        register_param("scale_b", (1,), torch.float32)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scale_b_value = float(layer.scale_b.detach().float().cpu().item())
        n, k = layer.output_size, layer.input_size
        n_pad = _round_up(n, 128)
        # 16-group weight scales in the 128x4 swizzled layout, viewed as FlashInfer's B scale.
        layer.sfw_dense_fi = layer.SFW_dense.data[: n_pad * k // 16].view(torch.float8_e4m3fn).view(n_pad, k // 16)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1])
        if x_2d.dtype != torch.bfloat16:
            x_2d = x_2d.to(torch.bfloat16)
        out = torch.ops.vllm.sharq_linear(
            x_2d,
            layer.QW,
            layer.SFW_sparse,
            layer.SFW_dense,
            layer.sfw_dense_fi,
            layer.scale_b_value,
        )
        if bias is not None:
            out = out + bias
        return out.view(*x.shape[:-1], -1)
