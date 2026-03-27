from typing import Optional, Tuple

import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaMLP, LlamaRMSNorm

from qLinearLayer import QLinearLayer
from quantize import get_rmsnorm_weight_eps, quantize_int_group


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


class QLlamaRMSNorm(nn.Module):
    def __init__(self, original_norm: LlamaRMSNorm):
        super().__init__()
        self.original_norm = original_norm

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.original_norm(hidden_states)


class QLlamaAttention(nn.Module):
    def __init__(self, original_attn: LlamaAttention, layer_idx: int, kv_cache: bool, quant_type: str, extra_fusion: bool = True):
        super().__init__()
        self.layer_idx = layer_idx
        self.quant_type = quant_type
        self.extra_fusion = extra_fusion
        self.q_kv_cache = kv_cache
        self.config = original_attn.config
        self.hidden_size = original_attn.hidden_size
        self.num_heads = original_attn.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = original_attn.num_key_value_heads
        self.num_key_value_groups = original_attn.num_key_value_groups
        self.max_position_embeddings = original_attn.max_position_embeddings
        self.rope_theta = original_attn.rope_theta
        self.attention_dropout = original_attn.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got hidden_size={self.hidden_size}, num_heads={self.num_heads})"
            )

        self.q_proj = QLinearLayer(original_attn.q_proj, quant_type=quant_type, extra_fusion=extra_fusion)
        self.k_proj = QLinearLayer(original_attn.k_proj, quant_type=quant_type, extra_fusion=extra_fusion)
        self.v_proj = QLinearLayer(original_attn.v_proj, quant_type=quant_type, extra_fusion=extra_fusion)
        self.o_proj = QLinearLayer(original_attn.o_proj, quant_type=quant_type, extra_fusion=extra_fusion)
        self.rotary_emb = original_attn.rotary_emb

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rmsnorm_weight: Optional[torch.Tensor] = None,
        rmsnorm_eps: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        hidden_states_2d = hidden_states.reshape(bsz * q_len, -1).contiguous().detach()
        qkv_out_features = max(self.q_proj.out_features, self.k_proj.out_features, self.v_proj.out_features)
        if self.quant_type == "SHARQ" and self.extra_fusion and rmsnorm_weight is not None and rmsnorm_eps is not None:
            qkv_prepared = self.q_proj.prepare_input_rmsnorm(
                hidden_states_2d,
                rmsnorm_weight,
                rmsnorm_eps,
                out_features_hint=qkv_out_features,
            )
        else:
            qkv_prepared = self.q_proj.prepare_input(hidden_states_2d, out_features_hint=qkv_out_features)

        query_states = self.q_proj.apply_prepared(qkv_prepared).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj.apply_prepared(qkv_prepared).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj.apply_prepared(qkv_prepared).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.q_kv_cache:
            key_states = quantize_int_group(key_states, nbits=4, group_size=64)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if self.q_kv_cache:
            value_states = quantize_int_group(value_states, nbits=4, group_size=64)

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = causal_mask is None and q_len > 1
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output_2d = attn_output.reshape(bsz * q_len, -1).contiguous().detach()
        attn_output = self.o_proj((attn_output_2d, bsz, q_len))

        attn_weights = None
        return attn_output, attn_weights, past_key_value


class QLlamaMLP(nn.Module):
    def __init__(self, original_mlp: LlamaMLP, quant_type: str, extra_fusion: bool = True):
        super().__init__()
        self.gate_proj = QLinearLayer(original_mlp.gate_proj, quant_type=quant_type, extra_fusion=extra_fusion)
        self.up_proj = QLinearLayer(original_mlp.up_proj, quant_type=quant_type, extra_fusion=extra_fusion)
        self.down_proj = QLinearLayer(original_mlp.down_proj, quant_type=quant_type, extra_fusion=extra_fusion)
        self.act_fn = original_mlp.act_fn
        self.quant_type = quant_type
        self.extra_fusion = extra_fusion

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        rmsnorm_weight: Optional[torch.Tensor] = None,
        rmsnorm_eps: Optional[float] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape
        hidden_states_2d = hidden_states.reshape(bsz * q_len, -1).contiguous().detach()
        gateup_out_features = max(self.gate_proj.out_features, self.up_proj.out_features)
        if self.quant_type == "SHARQ" and self.extra_fusion and rmsnorm_weight is not None and rmsnorm_eps is not None:
            gateup_prepared = self.gate_proj.prepare_input_rmsnorm(
                hidden_states_2d,
                rmsnorm_weight,
                rmsnorm_eps,
                out_features_hint=gateup_out_features,
            )
        else:
            gateup_prepared = self.gate_proj.prepare_input(hidden_states_2d, out_features_hint=gateup_out_features)
        mlp_output = self.act_fn(self.gate_proj.apply_prepared(gateup_prepared)) * self.up_proj.apply_prepared(gateup_prepared)
        mlp_output_2d = mlp_output.reshape(bsz * q_len, -1).contiguous().detach()
        return self.down_proj((mlp_output_2d, bsz, q_len))


class QLlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        original_layer: LlamaDecoderLayer,
        kv_cache: bool = False,
        layer_idx: int = 0,
        quant_type: str = "SHARQ",
        extra_fusion: bool = True,
    ):
        super().__init__()
        self.hidden_size = original_layer.hidden_size
        self.quant_type = quant_type
        self.extra_fusion = extra_fusion
        self.self_attn = QLlamaAttention(original_layer.self_attn, layer_idx, kv_cache, quant_type, extra_fusion=extra_fusion)
        self.mlp = QLlamaMLP(original_layer.mlp, quant_type, extra_fusion=extra_fusion)
        self.input_layernorm = QLlamaRMSNorm(original_layer.input_layernorm)
        self.post_attention_layernorm = QLlamaRMSNorm(original_layer.post_attention_layernorm)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        if self.quant_type == "SHARQ" and self.extra_fusion:
            input_rmsnorm_weight, input_rmsnorm_eps = get_rmsnorm_weight_eps(self.input_layernorm)
            attn_input = hidden_states
        else:
            attn_input = self.input_layernorm(hidden_states)
            input_rmsnorm_weight = None
            input_rmsnorm_eps = None

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=attn_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            rmsnorm_weight=input_rmsnorm_weight,
            rmsnorm_eps=input_rmsnorm_eps,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        if self.quant_type == "SHARQ" and self.extra_fusion:
            post_rmsnorm_weight, post_rmsnorm_eps = get_rmsnorm_weight_eps(self.post_attention_layernorm)
            mlp_input = hidden_states
        else:
            mlp_input = self.post_attention_layernorm(hidden_states)
            post_rmsnorm_weight = None
            post_rmsnorm_eps = None
        hidden_states = self.mlp(mlp_input, rmsnorm_weight=post_rmsnorm_weight, rmsnorm_eps=post_rmsnorm_eps)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs
