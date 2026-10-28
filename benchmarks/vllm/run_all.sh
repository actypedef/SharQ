#!/usr/bin/env bash
# Run the end-to-end vLLM benchmark for every method.
#   usage: bash run_all.sh <tag> [methods...]
#   env  : CUDA_VISIBLE_DEVICES (default 1), EXTRA_ARGS (passed to bench_e2e.py)
set -u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sharq-vllm
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
TAG=${1:-main}; shift || true
METHODS=${@:-"BF16 FP8 W4A8 NVFP4 SHARQ"}
EXTRA_ARGS=${EXTRA_ARGS:-}

M=/dockerdata/sharq_work/models/Llama-3.1-8B
Q=/dockerdata/sharq_work/quantized
OUT=/dockerdata/sharq_work/results/$TAG
mkdir -p "$OUT"

for method in $METHODS; do
  case $method in
    BF16)  args="--model $M --dtype bfloat16"; envs="" ;;
    FP8)   args="--model $Q/fp8_dynamic/Llama-3.1-8B"; envs="" ;;
    W4A8)  args="--model $Q/w4a16/Llama-3.1-8B"; envs="VLLM_MARLIN_INPUT_DTYPE=fp8" ;;
    W4A16) args="--model $Q/w4a16/Llama-3.1-8B"; envs="" ;;
    NVFP4) args="--model $Q/nvfp4/Llama-3.1-8B"; envs="" ;;
    SHARQ) args="--model $Q/sharq/Llama-3.1-8B"; envs="" ;;
    SHARQ_CUT) args="--model $Q/sharq/Llama-3.1-8B"; envs="SHARQ_VLLM_GEMM=cutlass" ;;
    *) echo "unknown method $method"; continue ;;
  esac
  echo "=========== [$TAG] $method  ($(date +%T))"
  env $envs timeout 7200 python bench_e2e.py --label "$method" $args $EXTRA_ARGS \
      --out "$OUT/$method.json" > "$OUT/$method.log" 2>&1
  echo "exit=$?"
  grep -E "^\[$method\]" "$OUT/$method.log"
done
