#!/usr/bin/env bash
# Prefill/decode phase benchmark + profiling for one method on one GPU.
#   usage: CUDA_VISIBLE_DEVICES=N bash run_phases.sh <tag> <method>
set -u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sharq-vllm
cd "$(dirname "$0")"

TAG=${1:?tag}; METHOD=${2:?method}
M=/dockerdata/sharq_work/models/Llama-3.1-8B
Q=/dockerdata/sharq_work/quantized
OUT=/dockerdata/sharq_work/results/$TAG
PROF=/dockerdata/sharq_work/prof/$TAG
mkdir -p "$OUT" "$PROF"

case $METHOD in
  BF16)      MODEL=$M;   ENVS="" ;;
  FP8)       MODEL=$Q/fp8_dynamic/Llama-3.1-8B; ENVS="" ;;
  W4A8)      MODEL=$Q/w4a16/Llama-3.1-8B; ENVS="VLLM_MARLIN_INPUT_DTYPE=fp8" ;;
  NVFP4)     MODEL=$Q/nvfp4/Llama-3.1-8B; ENVS="" ;;
  SHARQ)     MODEL=$Q/sharq/Llama-3.1-8B; ENVS="" ;;
  SHARQ_CUT) MODEL=$Q/sharq/Llama-3.1-8B; ENVS="SHARQ_VLLM_GEMM=cutlass" ;;
  *) echo "unknown method $METHOD"; exit 1 ;;
esac

echo "=== [$TAG] $METHOD bench_phases ($(date +%T))"
env $ENVS timeout 3600 python bench_phases.py --label "$METHOD" --model "$MODEL" \
  --out "$OUT/phases_$METHOD.json" > "$OUT/phases_$METHOD.log" 2>&1
echo "bench exit=$?"
grep -E "^\[$METHOD\]" "$OUT/phases_$METHOD.log"

export VLLM_ENABLE_V1_MULTIPROCESSING=0
prof() {  # name extra args...
  local name=$1; shift
  echo "--- [$TAG] $METHOD prof $name ($(date +%T))"
  env $ENVS timeout 1800 nsys profile -f true --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-graph-trace=node -t cuda -o "$PROF/${METHOD}_$name" \
    python profile_phases.py --model "$MODEL" "$@" > "$PROF/${METHOD}_$name.log" 2>&1
  echo "prof $name exit=$?"
  nsys stats -q -r cuda_gpu_kern_sum -f csv "$PROF/${METHOD}_$name.nsys-rep" > "$PROF/${METHOD}_$name.kern.csv" 2>/dev/null
}

prof prefill --mode prefill --batch 1 --len 8192 --iters 3
prof decode4 --mode decode --batch 4 --prompt 16 --steps 128
prof decode64 --mode decode --batch 64 --prompt 16 --steps 128
echo "=== [$TAG] $METHOD done ($(date +%T))"
