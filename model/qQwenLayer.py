from typing import Optional, Tuple

import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2DecoderLayer, Qwen2MLP, Qwen2RMSNorm

from qLinearLayer import QLinearLayer
from quantize import quantize_int_group


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


class QQwen2RMSNorm(nn.Module):
    def __init__(self, original_norm: Qwen2RMSNorm):
        super().__init__()
        self.original_norm = original_norm

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.original_norm(hidden_states)


class QQwen2Attention(nn.Module):
    def __init__(self, original_attn: Qwen2Attention, layer_idx: int, kv_cache: bool, quant_type: str):
        super().__init__()
        self.layer_idx = layer_idx
        self.quant_type = quant_type
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

        self.q_proj = QLinearLayer(original_attn.q_proj, quant_type=quant_type)
        self.k_proj = QLinearLayer(original_attn.k_proj, quant_type=quant_type)
        self.v_proj = QLinearLayer(original_attn.v_proj, quant_type=quant_type)
        self.o_proj = QLinearLayer(original_attn.o_proj, quant_type=quant_type)
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        hidden_states_2d = hidden_states.reshape(bsz * q_len, -1).contiguous().detach()
        linear_input = (hidden_states_2d, bsz, q_len)

        query_states = self.q_proj(linear_input).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(linear_input).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(linear_input).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        if self.q_kv_cache:
            key_states = quantize_int_group(key_states, nbits=4, group_size=128)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
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

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output_2d = attn_output.reshape(bsz * q_len, -1).contiguous().detach()
        attn_output = self.o_proj((attn_output_2d, bsz, q_len))

        attn_weights = None
        return attn_output, attn_weights, past_key_value


class QQwen2MLP(nn.Module):
    def __init__(self, original_mlp: Qwen2MLP, quant_type: str):
        super().__init__()
        self.gate_proj = QLinearLayer(original_mlp.gate_proj, quant_type=quant_type)
        self.up_proj = QLinearLayer(original_mlp.up_proj, quant_type=quant_type)
        self.down_proj = QLinearLayer(original_mlp.down_proj, quant_type=quant_type)
        self.act_fn = original_mlp.act_fn

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape
        hidden_states_2d = hidden_states.reshape(bsz * q_len, -1).contiguous().detach()
        linear_input = (hidden_states_2d, bsz, q_len)
        mlp_output = self.act_fn(self.gate_proj(linear_input)) * self.up_proj(linear_input)
        mlp_output_2d = mlp_output.reshape(bsz * q_len, -1).contiguous().detach()
        return self.down_proj((mlp_output_2d, bsz, q_len))


class QQwen2DecoderLayer(nn.Module):
    def __init__(
        self,
        original_layer: Qwen2DecoderLayer,
        kv_cache: bool = False,
        layer_idx: int = 0,
        quant_type: str = "SHARQ",
    ):
        super().__init__()
        self.hidden_size = original_layer.hidden_size
        self.self_attn = QQwen2Attention(original_layer.self_attn, layer_idx, kv_cache, quant_type)
        self.mlp = QQwen2MLP(original_layer.mlp, quant_type)
        self.input_layernorm = QQwen2RMSNorm(original_layer.input_layernorm)
        self.post_attention_layernorm = QQwen2RMSNorm(original_layer.post_attention_layernorm)

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
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs
