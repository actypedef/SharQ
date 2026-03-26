#!/bin/bash

MODEL=${1}
QUANT_TYPE=${2:-SHARQ}

if [ -z "${MODEL}" ]; then
  echo "Usage: bash evaluate.sh /path/to/model [NVFP4|SHARQ]"
  exit 1
fi

DIR=$(pwd)
export CUDA_VISIBLE_DEVICES="0"

python "${DIR}/model/main.py" "${MODEL}" \
  --dataset wikitext2 \
  --eval_ppl \
  --quant_type "${QUANT_TYPE}"

python "${DIR}/model/main.py" "${MODEL}" \
  --dataset wikitext2 \
  --tasks piqa,arc_challenge,boolq,hellaswag,winogrande,lambada_openai,arc_easy \
  --lm_eval_num_fewshot 0 \
  --lm_eval_limit -1 \
  --quant_type "${QUANT_TYPE}"

python "${DIR}/model/main.py" "${MODEL}" \
  --dataset wikitext2 \
  --tasks mmlu \
  --lm_eval_num_fewshot 5 \
  --lm_eval_limit -1 \
  --quant_type "${QUANT_TYPE}"
