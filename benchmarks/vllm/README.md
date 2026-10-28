# SharQ x vLLM

Everything needed to serve SharQ in vLLM and reproduce the numbers below: a CUDA-graph safe
operator module, an out-of-tree vLLM plugin, weight quantization (RTN and GPTQ), and the
benchmark / profiling harness. No vLLM source changes are required.

| Part | Path | Notes |
|---|---|---|
| Ops (`sharq_ops_vllm`) | `kernels/src/vllm_ops.cu` | stream-aware, no host sync, no allocation on the hot path; stacked decode path, Stream-K variants, per-M dispatch |
| Plugin | `plugin/` | registers the `sharq` quant method via `vllm.general_plugins`; custom op `vllm::sharq_linear` |
| Weight quantization | `sharq_quant.py` (RTN), `quantize_gptq.py` (GPTQ) | both emit the same checkpoint format |
| Benchmarks | `bench_e2e.py`, `bench_phases.py`, `bench_decode_fair.py`, `bench_full_stacked.py`, `profile_phases.py` | latency, throughput, prefill/decode split, nsys attribution |
| Reports | `summarize.py` | `e2e` / `phases` / `kern` sub-commands |

Chinese guides (more detail): [`README.zh.md`](README.zh.md) for setup and reproduction,
[`PERF_RETROSPECTIVE.zh.md`](PERF_RETROSPECTIVE.zh.md) for the optimization history.

## Requirements

Verified on a single RTX PRO 5000 72 GB (Blackwell, `sm_120a`), Ubuntu 22.04:

- CUDA 13.0 (`/usr/local/cuda`), Python 3.12, PyTorch 2.11.0+cu130
- vLLM 0.24.0, FlashInfer 0.6.12 (both pip releases)
- `sm_100a` works via `-DSHARQ_CUDA_ARCH=sm100a`
- `kernels/CMakeLists.txt` hard-codes `/usr/local/cuda/include`, so the toolkit must match the
  CUDA version of the PyTorch you link against; build each environment into its own directory
  (the extension is ABI-bound to the torch version)

## Build and install

```bash
cmake -S kernels -B kernels/build_vllm2_sm120a -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DSHARQ_CUDA_ARCH=sm120a \
  -DPython3_EXECUTABLE=$(which python)
cmake --build kernels/build_vllm2_sm120a --target sharq_ops      -j 16   # weight quantization
cmake --build kernels/build_vllm2_sm120a --target sharq_ops_vllm -j 16   # vLLM inference
pip install -e benchmarks/vllm/plugin --no-deps --no-build-isolation
```

The plugin resolves the ops directory in this order: `SHARQ_VLLM_KERNEL_BUILD` /
`SHARQ_KERNEL_BUILD` -> `kernels/build_vllm2_sm120a` -> `kernels/build_vllm_sm120a` ->
`kernels/build_vllm_sm100a`.

## Weights

Default is round-to-nearest (no calibration data):

```bash
SHARQ_KERNEL_BUILD=/path/to/kernels/build_vllm2_sm120a CUDA_VISIBLE_DEVICES=0 \
python sharq_quant.py --model /path/to/Llama-3.1-8B \
  --save_path /path/to/quantized/sharq/Llama-3.1-8B
```

Optional GPTQ (better ppl, needs calibration, **zero** runtime cost - same format, same 6.1 GB):

```bash
SHARQ_KERNEL_BUILD=/path/to/kernels/build_cmake_sm120a CUDA_VISIBLE_DEVICES=0 \
python quantize_gptq.py --model /path/to/Llama-3.1-8B \
  --calib-ids /path/to/wikitext2_train_ids.json --calib-windows 64 \
  --ids /path/to/wikitext2_test_ids.json --eval-windows 141 \
  --method gptq_opt --damp 0.1 --save-sharq /path/to/quantized/sharq_gptq/Llama-3.1-8B
```

GPTQ runs on the deployed grid on purpose: SharQ's weight scale granularity is pinned to 32
elements by CUTLASS's sparse NVFP4 MMA (`nv_float4_t` -> `SfVectorSize == 32` for sparse
schedules), and re-projecting a 16-group solution onto g32 loses about 85% of the gain. Keep the
calibration split separate from the evaluation split, otherwise the gain is inflated by ~0.06 ppl.

`config.json` gets `quantization_config.quant_method = "sharq"`, so vLLM picks the plugin
automatically. Baselines are produced with llm-compressor (`make_baselines.py`, separate env).

## Smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python test_sharq_vllm_ops.py      # bit-exactness + CUDA graph, expect "ALL OK"
CUDA_VISIBLE_DEVICES=0 python smoke_generate.py --model /path/to/quantized/sharq/Llama-3.1-8B
CUDA_VISIBLE_DEVICES=0 python smoke_generate.py --baselines /path/to/quantized \
  --base-model /path/to/Llama-3.1-8B                      # BF16 + FP8 + NVFP4 + W4A8 sweep
```

## Benchmarks

```bash
# one method: latency (bs=4) and throughput, one engine for all lengths
python bench_e2e.py --label SHARQ --model $Q/sharq/Llama-3.1-8B \
  --kv-cache-gib 40 --gpu-mem-util 0.9 --lengths 256,512,1024,2048 --targets latency,throughput

# several methods, one GPU per method
CUDA_VISIBLE_DEVICES=0 bash run_all.sh r1 BF16 FP8 W4A8 NVFP4 SHARQ SHARQ_CUT

# full-load throughput (records real concurrency and preemptions)
CUDA_VISIBLE_DEVICES=0 EXTRA_ARGS="--kv-cache-gib 40 --gpu-mem-util 0.9 --max-num-seqs 512 \
  --targets throughput --tp-warmup 64 --log-stats \
  --tp-prompts 256:2048,512:1024,1024:512,2048:256" bash run_all.sh fullload SHARQ NVFP4

# prefill (TTFT) and decode (per-step) separately, plus nsys attribution
CUDA_VISIBLE_DEVICES=0 bash run_phases.sh ph1 SHARQ

# reports
python summarize.py e2e    results/summary.md      # latency / throughput / full-load / ppl
python summarize.py phases results/ph_summary.md   # prefill & decode + kernel categories
python summarize.py kern   prof/SHARQ_decode4.kern.csv
```

Notes that matter when comparing numbers:

- **Always report the KV-cache size.** With a small budget the "throughput" test degenerates into
  a small-batch decode test. `--log-stats` records the real concurrency (`avg_seqs_per_step`) and
  the preemption count. Llama-3.1-8B needs ~128 KB of KV per token, so 2.25 GiB ~ 18K tokens and
  40 GiB ~ 328K tokens.
- nsys must run with `--cuda-graph-trace=node` and `VLLM_ENABLE_V1_MULTIPROCESSING=0`, otherwise
  the decode kernels (inside CUDA graphs) are not expanded.
- The per-layer microbenchmarks are L2-hot; use `bench_decode_fair.py` (same static activation
  scale + autotune as vLLM) for cross-method comparisons.

## Results (Llama-3.1-8B, single GPU, same KV budget for every method)

| Method | Latency bs=4, 256/256 | Full-load tok/s, 256x2048 | WikiText-2 ppl | Calibration |
|---|---|---|---|---|
| BF16 | 3.747 s | 9503 | 6.2396 | none |
| FP8 W8A8 | 2.168 s | 14564 | 6.2893 | no |
| W4A8 (Marlin) | **1.520 s** | 12745 | 7.0313 | no |
| NVFP4 W4A4 | 1.562 s | **20328** | 6.9387 | yes (64x512) |
| SharQ (RTN weights) | 1.742 s | 14486 | 6.7481 | no |
| **SharQ + GPTQ weights** | 1.742 s | 14486 | **6.6134** | yes (64x2048) |

SharQ is the most accurate 4-bit format here, runs about 2.1x faster than BF16 at small batch, and
matches FP8 in throughput. The remaining gap to NVFP4 is structural: two GEMMs per layer at large
M (sparse + residual) and no fusion of the activation quantizer into RMSNorm/SiLU. Stage-by-stage
numbers and the attribution are in [`PERF_RETROSPECTIVE.zh.md`](PERF_RETROSPECTIVE.zh.md) 7.3-7.5.

## Runtime knobs

| Env var | Default | Meaning |
|---|---|---|
| `SHARQ_VLLM_GEMM` | `flashinfer` | `cutlass` forces the paper kernels for every M |
| `SHARQ_STACK_MAX_M` | 192 | switch-over batch from stacked to two-GEMM |
| `SHARQ_STACK_MAX_M_LONG_K` | 256 | same, for `K >= 8192` (down_proj) |
| `SHARQ_FORCE_MODE` / `SHARQ_FORCE_DENSE` / `SHARQ_FORCE_SPARSE` | - | debugging overrides |

## Limitations

- `tensor_parallel_size=1` only; linear layers require `K % 256 == 0`; bf16 activations only.
- MoE models are not supported (the sparse path has no grouped/blockscaled kernel in CUTLASS).
- Qwen2-family checkpoints need per-shard loading (vLLM's loader rewrites the pre-merged
  `gate_up_proj` key); Llama-family models work as-is.

## Scripts

`test_sharq_vllm_ops.py` (correctness), `smoke_generate.py` (generation), `sharq_quant.py` /
`quantize_gptq.py` / `make_baselines.py` (weights), `ppl_vllm.py` / `lmeval.py` (accuracy),
`bench_e2e.py` / `bench_phases.py` / `run_all.sh` / `run_phases.sh` (throughput),
`bench_decode_fair.py` / `bench_full_stacked.py` (microbenchmarks), `profile_phases.py` (nsys),
`summarize.py` (reports), `analysis_ppl_decomp.py` (activation-vs-weight error split),
`legacy/` (the old patch-vLLM-source approach, reference only).
