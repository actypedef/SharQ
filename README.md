# SharQ🦈: Bridging Activation Sparsity and FP4 Quantization for LLM Inference

<h5 align="center">

[![arXiv](https://img.shields.io/badge/SharQ-2606.26587-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2606.26587)
<br>

</h5>

SharQ is a Blackwell-oriented LLM quantization repo built around this idea: use a sparse FP4 main path to capture important activations, then use a dense FP4 residual path to recover the loss from sparsification and quantization.

![SharQ method overview](assets/readme/sharq_method_overview.png)

The repo currently provides three practical modes:

- `NVFP4`: dense FP4 baseline
- `SHARQ`: fused sparse-residual FP4 kernel path
- `SHARQ_SIM`: pure PyTorch fake-quantized, fake-sparse simulation for accuracy-only reference

## vLLM Serving (added in this worktree)

This tree additionally contains a **CUDA-graph safe vLLM integration** of SharQ so it can be
served and benchmarked end-to-end next to other quantization methods:

| Component | Location | Notes |
|---|---|---|
| `sharq_ops_vllm` | `kernels/src/vllm_ops.cu` | stream-aware, no host sync, no allocation on the hot path; stacked decode path; Stream-K variants; automatic mode selection by M |
| vLLM plugin | `benchmarks/vllm/plugin/` | registers the `sharq` quant method via `vllm.general_plugins`, custom op `vllm::sharq_linear`. No vLLM source changes |
| Tests / benchmarks | `benchmarks/vllm/` | correctness, latency, throughput, prefill/decode split, nsys kernel attribution |
| Docs | [`benchmarks/vllm/README.md`](benchmarks/vllm/README.md) (中文操作指南), [`benchmarks/vllm/PERF_RETROSPECTIVE.md`](benchmarks/vllm/PERF_RETROSPECTIVE.md) (中文性能复盘) | |

Quick start (full details in [`benchmarks/vllm/README.md`](benchmarks/vllm/README.md)):

```bash
cmake -S kernels -B kernels/build_vllm2_sm120a -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DSHARQ_CUDA_ARCH=sm120a \
  -DPython3_EXECUTABLE=$(which python)
cmake --build kernels/build_vllm2_sm120a --target sharq_ops      -j 16   # for weight quantization
cmake --build kernels/build_vllm2_sm120a --target sharq_ops_vllm -j 16   # for vLLM inference
pip install -e benchmarks/vllm/plugin --no-deps --no-build-isolation

python -c "from vllm import LLM; LLM(model='/path/to/quantized/sharq/Llama-3.1-8B')"
```

Measured on Llama-3.1-8B, single RTX PRO 5000 72GB (Blackwell, `sm_120a`), vLLM 0.24.0,
CUDA 13.0, torch 2.11, FlashInfer 0.6.12, identical KV-cache budget for every method:

| Method | Latency, bs=4, 256/256 (s) | Latency, 2048/2048 (s) | Full-load throughput, 256×2048 (tok/s) | WikiText-2 ppl |
|---|---|---|---|---|
| BF16 | 3.747 | 32.537 | 9503 | 6.2396 |
| FP8 W8A8 | 2.168 | 19.891 | 14564 | 6.2893 |
| W4A8 (Marlin) | **1.520** | **14.595** | 12745 | 7.0313 |
| NVFP4 W4A4 (calibrated) | 1.562 | 14.970 | **20328** | 6.9387 |
| **SharQ** | 1.742 | 16.387 | 14486 | **6.7481** |

SharQ keeps the best accuracy among the 4-bit methods **without any calibration data**, runs
about 2.1× faster than BF16 at small batch, and stays on par with FP8 W8A8 in throughput.
Remaining gaps (two-GEMM cost at large M, missing quant fusion at small M) and the full
optimization history are in [`benchmarks/vllm/PERF_RETROSPECTIVE.md`](benchmarks/vllm/PERF_RETROSPECTIVE.md).

## Clone

Clone the repo together with CUTLASS:

```bash
git clone --recurse-submodules https://github.com/actypedef/SharQ.git
cd SharQ
```

If you already cloned the repo without submodules:

```bash
git submodule update --init --recursive
```

## Environment

Create and activate a `conda` environment named `sharq` with Python `3.10`:

```bash
conda create -n sharq python=3.10 -y
conda activate sharq
```

## Repository Layout

```text
model/              model wrappers and evaluation entry point
kernels/            CUDA extension, CUTLASS sparse GEMM integration, low-level benchmarks
  src/vllm_ops.cu   CUDA-graph safe ops for vLLM (module sharq_ops_vllm)
benchmarks/         correctness, perf, and ablation scripts
  vllm/             vLLM plugin, quantization, end-to-end / phase benchmarks, profiling
demo/               simple local chat demo
```

For kernel structure, build details, and low-level CUDA benchmarks, see [`kernels/README.md`](kernels/README.md).
For the vLLM integration, see [`benchmarks/vllm/README.md`](benchmarks/vllm/README.md).

## Requirements

Two environments are supported and have different requirements:

**A. Repository evaluation path (`model/main.py`, `demo/`, original benchmarks)**

- Python `3.10`
- PyTorch with CUDA support
- CUDA `12.8` recommended
- NVIDIA RTX 50 / Blackwell `sm_120a` GPU for the default real `SHARQ` kernel path

**B. vLLM serving path (`benchmarks/vllm/`, `kernels/src/vllm_ops.cu`)**

- Python `3.12`, PyTorch `2.11.0+cu130`, vLLM `0.24.0`, FlashInfer `0.6.12`, CUDA `13.0`
- Verified on RTX PRO 5000 72GB (`sm_120a`); `sm_100a` (B200) is supported via `-DSHARQ_CUDA_ARCH=sm100a`

> `kernels/CMakeLists.txt` hard-codes `/usr/local/cuda/include`, so the toolkit used for the
> build must match the CUDA version of the PyTorch you link against. Build each environment into
> its own directory (the extension is ABI-bound to the torch version), e.g.
> `kernels/build_cmake_sm120a` (torch 2.9.1) and `kernels/build_vllm2_sm120a` (torch 2.11).

Install dependencies (environment A):

```bash
pip install -r requirements.txt
conda install pybind11 -y
```

Environment B does not use `requirements.txt`: install `vllm`, `torch`, and `pybind11` for the
matching Python/torch version; the exact commands are in
[`benchmarks/vllm/README.md`](benchmarks/vllm/README.md) §3. Note that `requirements.txt` pins
`transformers==4.44.0`, which is incompatible with vLLM 0.24 — keep the two environments separate.

## Build

Build the CUDA extension:

```bash
cmake -S kernels -B kernels/build_cmake_sm120a \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DSHARQ_CUDA_ARCH=sm120a \
  -DPython3_EXECUTABLE=$(which python)

cmake --build kernels/build_cmake_sm120a --target sharq_ops -j
```

If you are targeting B200 instead of RTX 50 series, switch the architecture explicitly:

```bash
cmake -S kernels -B kernels/build_cmake_sm100a -DSHARQ_CUDA_ARCH=sm100a
```

If you only want a numerical reference path, `SHARQ_SIM` does not require the CUDA extension.

The build provides two targets. `sharq_ops` is the original extension used by `model/`,
`demo/`, the correctness benchmarks, and weight quantization; `sharq_ops_vllm` is the one the
vLLM plugin loads (both can live in the same build directory):

```bash
cmake --build kernels/build_cmake_sm120a --target sharq_ops      -j   # original kernels (quantization)
cmake --build kernels/build_cmake_sm120a --target sharq_ops_vllm -j   # stream-aware ops for vLLM
```

`sharq_ops_vllm` differs from `sharq_ops` in three ways that matter for serving: every kernel
launches on the current PyTorch stream, workspaces are torch-allocated (no `cudaMalloc` on the
hot path), and the per-tensor activation scale is reduced on device and folded into the GEMM
epilogue, so the whole linear is CUDA-graph capturable.

## Evaluation

Perplexity:

```bash
python model/main.py /path/to/model \
  --dataset wikitext2 \
  --eval_ppl \
  --quant_type SHARQ

# disable RMSNorm fusion for accuracy ablations
python model/main.py /path/to/model \
  --dataset wikitext2 \
  --eval_ppl \
  --quant_type SHARQ \
  --disable_rmsnorm_fusion
```

Simulation-only reference:

```bash
python model/main.py /path/to/model \
  --dataset wikitext2 \
  --eval_ppl \
  --quant_type SHARQ_SIM
```

Zero-shot lm-eval:

```bash
python model/main.py /path/to/model \
  --tasks piqa,arc_challenge,boolq,hellaswag,winogrande,lambada_openai,arc_easy \
  --lm_eval_num_fewshot 0 \
  --quant_type SHARQ
```

Or use:

```bash
bash evaluate.sh /path/to/model SHARQ
```

### vLLM end-to-end

The vLLM path has its own evaluation and benchmarking entry points (details and all knobs in
[`benchmarks/vllm/README.md`](benchmarks/vllm/README.md)):

```bash
cd benchmarks/vllm

# perplexity inside vLLM (same WikiText-2 protocol as above)
python ppl_vllm.py --label SHARQ --model /path/to/quantized/sharq/Llama-3.1-8B \
  --tokenizer /path/to/Llama-3.1-8B

# latency (batch=4) and throughput for one method
python bench_e2e.py --label SHARQ --model /path/to/quantized/sharq/Llama-3.1-8B \
  --kv-cache-gib 40 --gpu-mem-util 0.9 --lengths 256,512,1024,2048 --targets latency \
  --out /path/to/results/latency_SHARQ.json

# prefill / decode measured separately, plus nsys kernel attribution
python bench_phases.py --label SHARQ --model /path/to/quantized/sharq/Llama-3.1-8B \
  --out /path/to/results/phases_SHARQ.json
CUDA_VISIBLE_DEVICES=0 bash run_phases.sh ph1 SHARQ
python3 summarize_phases.py
```

Multi-method sweeps (`SHARQ`, `NVFP4`, `FP8`, `W4A8`, `BF16`, `SHARQ_CUT`) can be run one GPU
per method with `bash run_all.sh <tag> <METHOD>...`; baseline checkpoints are produced by
`make_baselines.py` with llm-compressor (see the guide).

SharQ-specific runtime knobs are environment variables, e.g. `SHARQ_VLLM_GEMM=cutlass` to force
the original sparse+dense kernel pair instead of the stacked single-GEMM decode path, and
`SHARQ_STACK_MAX_M` to move the switch-over batch (see the guide for the full table).

## Chat Demo

Single-turn example:

```bash
python demo/chat_qwen.py \
  --model ../Qwen2.5-7B-Instruct \
  --quant-type SHARQ \
  --prompt "Briefly explain SharQ." \
  --max-new-tokens 128
```

Interactive multi-turn example:

```bash
python demo/chat_qwen.py \
  --model ../Qwen2.5-7B-Instruct \
  --quant-type SHARQ
```

## Benchmarks

The benchmark scripts are grouped under:

- [`benchmarks/correctness`](benchmarks/correctness)
- [`benchmarks/perf`](benchmarks/perf)
- [`benchmarks/e2e`](benchmarks/e2e)
- [`benchmarks/ablation`](benchmarks/ablation)
- [`benchmarks/vllm`](benchmarks/vllm) — vLLM end-to-end latency / throughput / phase benchmarks and profiling

See [`benchmarks/README.md`](benchmarks/README.md) for a short guide, and
[`benchmarks/vllm/README.md`](benchmarks/vllm/README.md) for the serving benchmarks.

End-to-end prefill benchmark example:

```bash
python benchmarks/e2e/benchmark_prefill_e2e.py \
  --model /path/to/model \
  --quant-type SHARQ \
  --batch-size 1 \
  --seqlen 2048

# compare against the unfused RMSNorm path
python benchmarks/e2e/benchmark_prefill_e2e.py \
  --model /path/to/model \
  --quant-type SHARQ \
  --batch-size 1 \
  --seqlen 2048 \
  --disable-rmsnorm-fusion
```

The same script also supports `BF16`, `NVFP4`, and `SHARQ_SIM` for side-by-side prefill comparison.

Model evaluation keeps extra fusion disabled for accuracy measurements, while the benchmark and demo paths enable both RMSNorm fusion and epilogue fusion automatically.

## Supported Models

- Llama
- Qwen2
- Mixtral

The vLLM path has been validated on Llama-3.1-8B (TP=1). Layers must satisfy `K % 256 == 0`,
which holds for the models above.

## Dependencies

- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [transformers](https://github.com/huggingface/transformers)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

Additional dependencies for the vLLM path: [vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer) (used as the NVFP4 GEMM backend for the
decode path), and [llm-compressor](https://github.com/vllm-project/llm-compressor) (only for
generating baseline checkpoints).

## Documents

- [`benchmarks/vllm/README.md`](benchmarks/vllm/README.md) — vLLM 端到端配置、运行与复现指南（中文）
- [`benchmarks/vllm/PERF_RETROSPECTIVE.md`](benchmarks/vllm/PERF_RETROSPECTIVE.md) — 性能问题定位、三轮优化与收益、分阶段归因（中文）
- [`kernels/README.md`](kernels/README.md) — kernel 结构与底层 CUDA benchmark
- [`benchmarks/README.md`](benchmarks/README.md) — 原仓库的 correctness / perf / ablation 脚本说明
