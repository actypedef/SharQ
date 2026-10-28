#!/usr/bin/env bash
# Smoke test all baselines in vLLM; logs selected kernels + generations.
set -u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sharq-vllm
cd /root/SharQ/benchmarks/vllm
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
M=/dockerdata/sharq_work/models/Llama-3.1-8B
Q=/dockerdata/sharq_work/quantized

run() {
  local name=$1; shift
  echo "=================== $name"
  timeout 1500 env "$@" > /tmp/smoke_${name}.log 2>&1
  echo "exit=$?"
  grep -E "Using .*(Kernel|kernel|backend)|Selected|Error|Traceback|->" /tmp/smoke_${name}.log | grep -v "Attention" | cut -c1-260 | tail -8
}

run BF16  python smoke_generate.py --model $M --dtype bfloat16
run FP8   python smoke_generate.py --model $Q/fp8_dynamic/Llama-3.1-8B
run NVFP4 python smoke_generate.py --model $Q/nvfp4/Llama-3.1-8B
run W4A8  VLLM_MARLIN_INPUT_DTYPE=fp8 python smoke_generate.py --model $Q/w4a16/Llama-3.1-8B
