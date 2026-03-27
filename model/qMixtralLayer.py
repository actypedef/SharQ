from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention,
    MixtralBlockSparseTop2MLP,
    MixtralDecoderLayer,
    MixtralRMSNorm,
    MixtralSparseMoeBlock,
)

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
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


class QMixtralRMSNorm(nn.Module):
    def __init__(self, original_norm: MixtralRMSNorm):
        super().__init__()
        self.original_norm = original_norm

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.original_norm(hidden_states)


class QMixtralAttention(nn.Module):
    def __init__(self, original_attn: MixtralAttention, layer_idx: int, kv_cache: bool, quant_type: str, extra_fusion: bool = True):
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

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        else:
            cos, sin = position_embeddings

        if self.q_kv_cache:
            key_states = quantize_int_group(key_states, nbits=4, group_size=128)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if self.q_kv_cache:
            value_states = quantize_int_group(value_states, nbits=4, group_size=128)

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


class QMixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, original_block: MixtralBlockSparseTop2MLP, quant_type: str, extra_fusion: bool = True):
        super().__init__()
        self.ffn_dim = original_block.ffn_dim
        self.hidden_dim = original_block.hidden_dim
        self.w1 = QLinearLayer(original_block.w1, quant_type=quant_type, extra_fusion=extra_fusion)
        self.w2 = QLinearLayer(original_block.w2, quant_type=quant_type, extra_fusion=extra_fusion)
        self.w3 = QLinearLayer(original_block.w3, quant_type=quant_type, extra_fusion=extra_fusion)
        self.act_fn = original_block.act_fn

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q_len, _ = hidden_states.shape
        hidden_states_2d = hidden_states.contiguous().detach()
        gateup_out_features = max(self.w1.out_features, self.w3.out_features)
        gateup_prepared = self.w1.prepare_input(hidden_states_2d, out_features_hint=gateup_out_features)
        hidden_states = self.act_fn(self.w1.apply_prepared(gateup_prepared)) * self.w3.apply_prepared(gateup_prepared)
        hidden_states_2d = hidden_states.reshape(q_len, -1).contiguous().detach()
        return self.w2((hidden_states_2d, None, q_len))


class QMixtralSparseMoeBlock(nn.Module):
    def __init__(self, original_sparse_moe_block: MixtralSparseMoeBlock, quant_type: str, extra_fusion: bool = True):
        super().__init__()
        self.hidden_dim = original_sparse_moe_block.hidden_dim
        self.ffn_dim = original_sparse_moe_block.ffn_dim
        self.num_experts = original_sparse_moe_block.num_experts
        self.top_k = original_sparse_moe_block.top_k
        self.gate = original_sparse_moe_block.gate
        self.experts = original_sparse_moe_block.experts

        for expert_idx in range(self.num_experts):
            self.experts[expert_idx] = QMixtralBlockSparseTop2MLP(
                original_sparse_moe_block.experts[expert_idx],
                quant_type=quant_type,
                extra_fusion=extra_fusion,
            )

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class QMixtralDecoderLayer(nn.Module):
    def __init__(
        self,
        original_layer: MixtralDecoderLayer,
        kv_cache: bool = False,
        layer_idx: int = 0,
        quant_type: str = "SHARQ",
        extra_fusion: bool = True,
    ):
        super().__init__()
        self.hidden_size = original_layer.hidden_size
        self.quant_type = quant_type
        self.extra_fusion = extra_fusion
        self.self_attn = QMixtralAttention(original_layer.self_attn, layer_idx, kv_cache, quant_type, extra_fusion=extra_fusion)
        self.block_sparse_moe = QMixtralSparseMoeBlock(original_layer.block_sparse_moe, quant_type, extra_fusion=extra_fusion)
        self.input_layernorm = QMixtralRMSNorm(original_layer.input_layernorm)
        self.post_attention_layernorm = QMixtralRMSNorm(original_layer.post_attention_layernorm)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
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
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        if output_router_logits:
            outputs += (router_logits,)
        return outputs
