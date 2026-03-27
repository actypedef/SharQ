import torch
import torch.nn as nn
import torch.nn.functional as F

from quantize import (
    apply_rmsnorm,
    load_sharq_ops,
    quantize_activation_nvfp4,
    quantize_activation_rmsnorm_sparse_residual_nvfp4,
    quantize_activation_sharq_sim,
    quantize_activation_sparse_residual_nvfp4,
    quantize_weight_nvfp4,
    quantize_weight_sharq_sim,
    quantize_weight_shared_nvfp4,
    to_python_float,
)


class QLinearLayer(nn.Module):
    def __init__(
        self,
        original_layer: nn.Linear,
        select_num=None,
        reorder_index=None,
        out_reorder_index=None,
        quant_type: str = "NVFP4",
        extra_fusion: bool = True,
    ):
        super().__init__()

        if quant_type not in {"NVFP4", "SHARQ", "SHARQ_SIM"}:
            raise ValueError(f"Unsupported quant_type: {quant_type}")

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.quant_type = quant_type
        self.enable_quant = True
        self.extra_fusion = extra_fusion

        if original_layer.bias is not None:
            self.register_buffer("bias", original_layer.bias.detach().clone())
        else:
            self.bias = None

        if self.quant_type == "SHARQ_SIM":
            self.weight_sim_q32, self.weight_scale = quantize_weight_sharq_sim(original_layer.weight)
            return

        weight_gpu = original_layer.weight.detach().to(device="cuda", dtype=torch.bfloat16)
        if self.quant_type == "SHARQ":
            self.weight_q, self.scale_w_sparse, self.scale_w_dense, self.weight_scale = quantize_weight_shared_nvfp4(weight_gpu)
        else:
            self.weight_q, self.scale_w, self.weight_scale = quantize_weight_nvfp4(weight_gpu)

        del weight_gpu
        torch.cuda.empty_cache()

    @torch.no_grad()
    def prepare_input(self, x_2d: torch.Tensor, out_features_hint: int | None = None):
        x_2d = x_2d.to(torch.bfloat16)

        if self.quant_type == "SHARQ_SIM":
            x_sparse_q32, x_res_q16, scale_x = quantize_activation_sharq_sim(x_2d)
            return ("SHARQ_SIM", x_sparse_q32, x_res_q16, scale_x)

        if self.quant_type == "SHARQ":
            sparse_out_features = self.out_features if out_features_hint is None else int(out_features_hint)
            a_comp, e, sfa_sparse, qx_res, scale_x_res, scale_x = quantize_activation_sparse_residual_nvfp4(
                x_2d, sparse_out_features
            )
            return ("SHARQ", a_comp, e, sfa_sparse, qx_res, scale_x_res, scale_x)

        qx, scale_x, scale = quantize_activation_nvfp4(x_2d)
        return ("NVFP4", qx, scale_x, scale)

    @torch.no_grad()
    def prepare_input_rmsnorm(
        self,
        x_2d: torch.Tensor,
        rmsnorm_weight: torch.Tensor,
        rmsnorm_eps: float,
        out_features_hint: int | None = None,
    ):
        if self.quant_type == "SHARQ" and self.extra_fusion:
            x_2d = x_2d.to(torch.bfloat16)
            sparse_out_features = self.out_features if out_features_hint is None else int(out_features_hint)
            a_comp, e, sfa_sparse, qx_res, scale_x_res, scale_x = quantize_activation_rmsnorm_sparse_residual_nvfp4(
                x_2d,
                rmsnorm_weight,
                rmsnorm_eps,
                sparse_out_features,
            )
            return ("SHARQ", a_comp, e, sfa_sparse, qx_res, scale_x_res, scale_x)

        x_norm = apply_rmsnorm(x_2d.to(torch.bfloat16), rmsnorm_weight, rmsnorm_eps)
        return self.prepare_input(x_norm, out_features_hint=out_features_hint)

    @torch.no_grad()
    def apply_prepared(self, prepared):
        tag = prepared[0]

        if tag == "SHARQ_SIM":
            _, x_sparse_q32, x_res_q16, scale_x = prepared
            output_scale = scale_x * self.weight_scale
            y_sparse = F.linear(x_sparse_q32, self.weight_sim_q32)
            y_res = F.linear(x_res_q16, self.weight_sim_q32)
            y = (y_sparse + y_res) * output_scale
        elif tag == "SHARQ":
            _, a_comp, e, sfa_sparse, qx_res, scale_x_res, scale_x = prepared
            output_scale = to_python_float(scale_x * self.weight_scale)
            sharq_ops = load_sharq_ops()
            y_sparse = sharq_ops.sparse_matmul(
                a_comp,
                self.weight_q,
                e,
                sfa_sparse,
                self.scale_w_sparse,
                qx_res.shape[0],
                self.out_features,
                self.in_features,
                alpha=output_scale,
            )
            if self.extra_fusion:
                y = sharq_ops.matmul_accum(
                    qx_res,
                    self.weight_q,
                    scale_x_res,
                    self.scale_w_dense,
                    output_scale,
                    y_sparse,
                    1.0,
                )
            else:
                y_res = sharq_ops.matmul(
                    qx_res,
                    self.weight_q,
                    scale_x_res,
                    self.scale_w_dense,
                    output_scale,
                )
                y = y_sparse + y_res
        elif tag == "NVFP4":
            _, qx, scale_x, scale = prepared
            sharq_ops = load_sharq_ops()
            y = sharq_ops.matmul(
                qx,
                self.weight_q,
                scale_x,
                self.scale_w,
                to_python_float(scale * self.weight_scale),
            )
        else:
            raise ValueError(f"Unsupported prepared input tag: {tag}")

        if self.bias is not None:
            bias = self.bias.float() if self.quant_type == "SHARQ_SIM" else self.bias
            y = y + bias

        if self.quant_type == "SHARQ_SIM":
            y = y.to(torch.bfloat16)

        return y

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
        prepared = self.prepare_input(x_orig_reshaped)
        y = self.apply_prepared(prepared)

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
        extra_fusion: bool = True,
    ):
        super().__init__(
            original_layer=original_layer,
            select_num=select_num,
            reorder_index=reorder_index,
            out_reorder_index=out_reorder_index,
            quant_type=quant_type,
            extra_fusion=extra_fusion,
        )
