# vLLM/sharq_quant.py

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_THIS_DIR = Path(__file__).resolve().parent


def _ordered_build_dir_names() -> tuple[str, ...]:
    if not torch.cuda.is_available():
        return ("build_cmake_sm120a", "build_cmake_sm100a", "build")

    major, _minor = torch.cuda.get_device_capability()
    if major >= 12:
        return ("build_cmake_sm120a", "build_cmake_sm100a", "build")
    if major >= 10:
        return ("build_cmake_sm100a", "build_cmake_sm120a", "build")
    return ("build_cmake_sm120a", "build_cmake_sm100a", "build")


def _kernel_build_candidates() -> list[Path]:
    candidates: list[Path] = []
    build_dir_names = _ordered_build_dir_names()

    for env_var in ('SHARQ_VLLM_KERNEL_BUILD', 'SHARQ_KERNEL_BUILD'):
        env_build = os.environ.get(env_var)
        if env_build:
            candidates.append(Path(env_build))

    for build_dir_name in build_dir_names:
        candidates.append(_THIS_DIR / 'kernels' / build_dir_name)
        candidates.append(Path('/root/autodl-tmp/SharQ/kernels') / build_dir_name)

    return candidates


for _candidate in reversed(_kernel_build_candidates()):
    if _candidate.exists():
        sys.path.insert(0, os.fspath(_candidate))

try:
    import sharq_ops_vllm as sharq_ops
    print('Success: sharq_ops module loaded.')
except ImportError:
    try:
        import sharq_ops
        print('Success: fallback sharq_ops module loaded.')
    except ImportError:
        print('Error: sharq_ops module not found. Please compile it first.')
        sys.exit(1)


NVFP4_DENOM = 448.0 * 6.0


def quantize_layer_weights(weight: torch.Tensor):
    weight_gpu = weight.to('cuda', dtype=torch.bfloat16).contiguous()
    scale_b = torch.clamp(weight_gpu.abs().max().float() / NVFP4_DENOM, min=1e-9)
    qw, sfw_sparse, sfw_dense = sharq_ops.quantize_w32_shared((weight_gpu / scale_b).to(torch.bfloat16))

    results = {
        'QW': qw.cpu(),
        'SFW_sparse': sfw_sparse.cpu(),
        'SFW_dense': sfw_dense.cpu(),
        'scale_b': scale_b.cpu(),
    }

    del weight_gpu, qw, sfw_sparse, sfw_dense
    torch.cuda.empty_cache()
    return results


def save_sharq_checkpoint(model, config, tokenizer, save_path: str, src_dir: str | None = None):
    """Quantize the in-memory model's linear layers and write a vLLM-ready SharQ checkpoint.

    Reused by quantize_gptq.py so that GPTQ-optimised weights go through exactly the same
    packing / merging / config-writing path as the plain RTN checkpoints.
    """
    new_state_dict = {}
    original_state_dict = model.state_dict()

    print('Starting SHARQ quantization (merged QKV & GateUp)...')
    layers = model.model.layers
    for i, layer in tqdm(enumerate(layers), total=len(layers), desc='Processing Layers'):
        prefix = f'model.layers.{i}'

        attn = layer.self_attn
        w_qkv = torch.cat(
            [attn.q_proj.weight.data, attn.k_proj.weight.data, attn.v_proj.weight.data],
            dim=0,
        ).contiguous()
        q_res = quantize_layer_weights(w_qkv)
        base_name = f'{prefix}.self_attn.qkv_proj'
        for k, v in q_res.items():
            new_state_dict[f'{base_name}.{k}'] = v
        if attn.q_proj.bias is not None:
            b_qkv = torch.cat([attn.q_proj.bias.data, attn.k_proj.bias.data, attn.v_proj.bias.data], dim=0)
            new_state_dict[f'{base_name}.bias'] = b_qkv
        for sub in ['q_proj', 'k_proj', 'v_proj']:
            original_state_dict.pop(f'{prefix}.self_attn.{sub}.weight', None)
            original_state_dict.pop(f'{prefix}.self_attn.{sub}.bias', None)

        q_res = quantize_layer_weights(attn.o_proj.weight.data)
        base_name = f'{prefix}.self_attn.o_proj'
        for k, v in q_res.items():
            new_state_dict[f'{base_name}.{k}'] = v
        if attn.o_proj.bias is not None:
            new_state_dict[f'{base_name}.bias'] = attn.o_proj.bias.data
        original_state_dict.pop(f'{prefix}.self_attn.o_proj.weight', None)
        original_state_dict.pop(f'{prefix}.self_attn.o_proj.bias', None)

        mlp = layer.mlp
        w_gate_up = torch.cat([mlp.gate_proj.weight.data, mlp.up_proj.weight.data], dim=0).contiguous()
        q_res = quantize_layer_weights(w_gate_up)
        base_name = f'{prefix}.mlp.gate_up_proj'
        for k, v in q_res.items():
            new_state_dict[f'{base_name}.{k}'] = v
        original_state_dict.pop(f'{prefix}.mlp.gate_proj.weight', None)
        original_state_dict.pop(f'{prefix}.mlp.up_proj.weight', None)
        original_state_dict.pop(f'{prefix}.mlp.gate_proj.bias', None)
        original_state_dict.pop(f'{prefix}.mlp.up_proj.bias', None)

        q_res = quantize_layer_weights(mlp.down_proj.weight.data)
        base_name = f'{prefix}.mlp.down_proj'
        for k, v in q_res.items():
            new_state_dict[f'{base_name}.{k}'] = v
        if getattr(mlp.down_proj, 'bias', None) is not None:
            new_state_dict[f'{base_name}.bias'] = mlp.down_proj.bias.data
        original_state_dict.pop(f'{prefix}.mlp.down_proj.weight', None)
        original_state_dict.pop(f'{prefix}.mlp.down_proj.bias', None)

    print('Merging non-quantized weights...')
    for key, value in original_state_dict.items():
        new_state_dict[key] = value

    # safetensors cannot store two names that share storage (tied embeddings), and
    # transformers re-ties them on load when tie_word_embeddings is set.
    if getattr(config, 'tie_word_embeddings', False):
        new_state_dict.pop('lm_head.weight', None)

    os.makedirs(save_path, exist_ok=True)
    print(f'Saving quantized model to {save_path}...')
    save_file(new_state_dict, os.path.join(save_path, 'model.safetensors'))

    config.quantization_config = {
        'quant_method': 'sharq',
    }
    config.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    if src_dir is not None and (Path(src_dir) / 'generation_config.json').exists():
        shutil.copy(Path(src_dir) / 'generation_config.json', Path(save_path) / 'generation_config.json')
    print('Done! Model is ready for vLLM (with merged QKV/GateUp).')


def quantize_model(model_path: str, save_path: str):
    print(f'Loading model from {model_path}...')
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype='auto',
        device_map='cpu',
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    save_sharq_checkpoint(model, config, tokenizer, save_path, src_dir=model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to original HF model')
    parser.add_argument('--save_path', type=str, required=True, help='Directory to save the quantized model')
    args = parser.parse_args()

    quantize_model(args.model, args.save_path)
