# SharQ🦈: Bridging Activation Sparsity and FP4 Quantization for LLM Inference

<h5 align="center">

[![arXiv](https://img.shields.io/badge/SharQ-2606.26587-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2606.26587)
<br>

</h5>

SharQ is a Blackwell-oriented LLM quantization repo: a sparse FP4 main path captures the important
activations, and a dense FP4 residual path recovers the loss from sparsification and quantization.

![SharQ method overview](assets/readme/sharq_method_overview.png)

Three modes are provided:

- `NVFP4` — dense FP4 baseline
- `SHARQ` — fused sparse-residual FP4 kernel path
- `SHARQ_SIM` — pure-PyTorch fake-quantized simulation (accuracy reference only)

## vLLM serving

This tree adds a CUDA-graph safe vLLM integration so SharQ can be served and benchmarked
end-to-end next to other quantization methods, without patching vLLM:

| Component | Location | Notes |
|---|---|---|
| `sharq_ops_vllm` | `kernels/src/vllm_ops.cu` | stream-aware, no host sync, no allocation on the hot path; stacked decode path, Stream-K variants, per-M dispatch |
| Plugin | `benchmarks/vllm/plugin/` | registers the `sharq` quant method via `vllm.general_plugins`; custom op `vllm::sharq_linear` |
| Weight quantization | `benchmarks/vllm/sharq_quant.py` (RTN), `quantize_gptq.py` (GPTQ) | GPTQ solves on the deployed g32 grid, so the gain survives packing |
| Benchmarks / docs | `benchmarks/vllm/` | correctness, latency, throughput, prefill/decode split, nsys attribution |

```bash
cmake -S kernels -B kernels/build_vllm2_sm120a -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DSHARQ_CUDA_ARCH=sm120a \
  -DPython3_EXECUTABLE=$(which python)
cmake --build kernels/build_vllm2_sm120a --target sharq_ops      -j 16   # weight quantization
cmake --build kernels/build_vllm2_sm120a --target sharq_ops_vllm -j 16   # vLLM inference
pip install -e benchmarks/vllm/plugin --no-deps --no-build-isolation

# quantize (RTN, no calibration data) and serve
python benchmarks/vllm/sharq_quant.py --model /path/to/Llama-3.1-8B --save_path /path/to/quantized/sharq/Llama-3.1-8B
python -c "from vllm import LLM; LLM(model='/path/to/quantized/sharq/Llama-3.1-8B')"
```

Measured on Llama-3.1-8B, single RTX PRO 5000 72 GB (`sm_120a`), vLLM 0.24.0, CUDA 13.0,
identical KV budget for every method:

| Method | Latency bs=4, 256/256 | Full-load tok/s, 256×2048 | WikiText-2 ppl | Calibration |
|---|---|---|---|---|
| BF16 | 3.747 s | 9503 | 6.2396 | — |
| FP8 W8A8 | 2.168 s | 14564 | 6.2893 | no |
| W4A8 (Marlin) | **1.520 s** | 12745 | 7.0313 | no |
| NVFP4 W4A4 | 1.562 s | **20328** | 6.9387 | yes |
| SharQ (RTN weights) | 1.742 s | 14486 | 6.7481 | no |
| **SharQ + GPTQ weights** | 1.742 s | 14486 | **6.6134** | yes |

SharQ is the most accurate 4-bit method here, runs about 2.1× faster than BF16 at small batch and
matches FP8 in throughput; the optional GPTQ variant buys another 0.13 ppl at zero runtime cost
(same checkpoint format, same kernels, same 6.1 GB). The remaining gap to NVFP4 is structural
(two GEMMs per layer at large M, no fusion of the activation quantizer into RMSNorm/SiLU).

## Clone

```bash
git clone --recurse-submodules https://github.com/actypedef/SharQ.git
cd SharQ
# already cloned without submodules:
git submodule update --init --recursive
```

## Environment

Two environments are supported:

**A. Repository evaluation path** (`model/main.py`, original benchmarks)

- Python 3.10, PyTorch with CUDA support, CUDA 12.8+ recommended, Blackwell `sm_120a` GPU
- `pip install -r requirements.txt && conda install pybind11 -y`

**B. vLLM serving path** (`benchmarks/vllm/`, `kernels/src/vllm_ops.cu`)

- Python 3.12, PyTorch 2.11.0+cu130, vLLM 0.24.0, FlashInfer 0.6.12, CUDA 13.0

`requirements.txt` pins `transformers==4.44.0`, which is incompatible with vLLM 0.24 — keep the two
environments separate. The extension is ABI-bound to the torch version, so build each environment
into its own directory (e.g. `kernels/build_cmake_sm120a` for torch 2.9, `kernels/build_vllm2_sm120a`
for torch 2.11).

## Build

```bash
cmake -S kernels -B kernels/build_cmake_sm120a -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DSHARQ_CUDA_ARCH=sm120a \
  -DPython3_EXECUTABLE=$(which python)
cmake --build kernels/build_cmake_sm120a --target sharq_ops -j
```

For B200 use `-DSHARQ_CUDA_ARCH=sm100a`. `sharq_ops` is what `model/main.py` and the correctness
benchmarks use; `sharq_ops_vllm` (see above) is the stream-aware module the plugin loads. The
`SHARQ_SIM` mode needs no CUDA extension.

## Evaluation

```bash
# WikiText-2 perplexity through the repository path
python model/main.py /path/to/model --dataset wikitext2 --eval_ppl --quant_type SHARQ
bash evaluate.sh /path/to/model SHARQ

# zero-shot downstream tasks (paper suite, 0-shot)
python model/main.py /path/to/model --tasks piqa,arc_challenge,boolq,hellaswag,winogrande,lambada_openai,arc_easy \
  --lm_eval_num_fewshot 0 --quant_type SHARQ
```

The vLLM path has its own entry points (details in [`benchmarks/vllm/README.md`](benchmarks/vllm/README.md)):

```bash
cd benchmarks/vllm
python ppl_vllm.py --label SHARQ --model /path/to/quantized/sharq/Llama-3.1-8B --tokenizer /path/to/Llama-3.1-8B
python bench_e2e.py --label SHARQ --model /path/to/quantized/sharq/Llama-3.1-8B --lengths 256,512,1024,2048 --targets latency
CUDA_VISIBLE_DEVICES=0 bash run_phases.sh ph1 SHARQ && python summarize.py phases   # prefill/decode + nsys
```

Multi-method sweeps (`BF16`, `FP8`, `W4A8`, `NVFP4`, `SHARQ`, `SHARQ_CUT`) run one GPU per method
with `bash run_all.sh <tag> <METHOD>...`; baselines are produced by `make_baselines.py`
(llm-compressor).

## Benchmarks

- [`benchmarks/correctness`](benchmarks/correctness) — kernel vs. simulation checks
- [`benchmarks/perf`](benchmarks/perf) — per-kernel timing
- [`benchmarks/e2e`](benchmarks/e2e) — prefill end-to-end (`benchmark_prefill_e2e.py`)
- [`benchmarks/ablation`](benchmarks/ablation) — RMSNorm / epilogue fusion ablations
- [`benchmarks/vllm`](benchmarks/vllm) — serving benchmarks (above)

## Supported models

- Llama, Qwen2/Qwen2.5, Mistral (dense decoders)
- The vLLM path is validated on Llama-3.1-8B; linear layers must satisfy `K % 256 == 0`
- MoE architectures (Mixtral and friends) are *not* supported by the vLLM path

## Dependencies

- [CUTLASS](https://github.com/NVIDIA/cutlass), [transformers](https://github.com/huggingface/transformers),
  [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- vLLM path only: [vLLM](https://github.com/vllm-project/vllm),
  [FlashInfer](https://github.com/flashinfer-ai/flashinfer),
  [llm-compressor](https://github.com/vllm-project/llm-compressor) (baseline checkpoints)

## Documents

- [`benchmarks/vllm/README.md`](benchmarks/vllm/README.md) — vLLM setup, weight quantization, benchmarks
- [`benchmarks/vllm/README.zh.md`](benchmarks/vllm/README.zh.md) — the same, in Chinese and more detailed
- [`benchmarks/vllm/PERF_RETROSPECTIVE.zh.md`](benchmarks/vllm/PERF_RETROSPECTIVE.zh.md) — profiling-driven
  optimization history, stage-by-stage attribution (Chinese)
- [`kernels/README.md`](kernels/README.md) — kernel structure and low-level CUDA benchmarks
- [`benchmarks/README.md`](benchmarks/README.md) — correctness / perf / ablation scripts
