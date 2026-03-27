import torch
from tqdm import tqdm
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from qLlamaLayer import QLlamaDecoderLayer
from qMixtralLayer import QMixtralDecoderLayer
from qQwenLayer import QQwen2DecoderLayer


def _quantize_decoder_layers(model, wrapped_cls, source_cls, device, kv_cache: bool, quant_type: str, extra_fusion: bool = True):
    model.config.use_cache = False
    layers = model.model.layers

    for layer_idx in tqdm(range(len(layers))):
        layers[layer_idx] = layers[layer_idx].to(device)
        if isinstance(layers[layer_idx], wrapped_cls):
            quantized_layer = layers[layer_idx]
        elif isinstance(layers[layer_idx], source_cls):
            quantized_layer = wrapped_cls(
                original_layer=layers[layer_idx],
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                quant_type=quant_type,
                extra_fusion=extra_fusion,
            )
        else:
            raise TypeError(f"Unsupported layer type at index {layer_idx}: {type(layers[layer_idx])}")

        layers[layer_idx] = layers[layer_idx].cpu()
        layers[layer_idx] = quantized_layer.cpu()
        del quantized_layer
        torch.cuda.empty_cache()

    return model


def quantize_model_llama(model, device, kv_cache: bool = False, quant_type: str = "SHARQ", extra_fusion: bool = True):
    return _quantize_decoder_layers(
        model=model,
        wrapped_cls=QLlamaDecoderLayer,
        source_cls=LlamaDecoderLayer,
        device=device,
        kv_cache=kv_cache,
        quant_type=quant_type,
        extra_fusion=extra_fusion,
    )


def quantize_model_qwen(model, device, kv_cache: bool = False, quant_type: str = "SHARQ", extra_fusion: bool = True):
    return _quantize_decoder_layers(
        model=model,
        wrapped_cls=QQwen2DecoderLayer,
        source_cls=Qwen2DecoderLayer,
        device=device,
        kv_cache=kv_cache,
        quant_type=quant_type,
        extra_fusion=extra_fusion,
    )


def quantize_model_mixtral(model, device, kv_cache: bool = False, quant_type: str = "SHARQ", extra_fusion: bool = True):
    return _quantize_decoder_layers(
        model=model,
        wrapped_cls=QMixtralDecoderLayer,
        source_cls=MixtralDecoderLayer,
        device=device,
        kv_cache=kv_cache,
        quant_type=quant_type,
        extra_fusion=extra_fusion,
    )
