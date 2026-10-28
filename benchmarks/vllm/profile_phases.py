"""Profile a pure-prefill or pure-decode window for nsys.

  prefill : batch x 8192-token prompts, max_tokens=1, `iters` repetitions inside the capture
  decode  : batch x 16-token prompts, max_tokens=steps, one generate inside the capture

    VLLM_ENABLE_V1_MULTIPROCESSING=0 nsys profile --capture-range=cudaProfilerApi \
        --capture-range-end=stop --cuda-graph-trace=node -t cuda -o out \
        python profile_phases.py --mode prefill --batch 1 --len 8192 ...
"""

import argparse

import numpy as np
import torch

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=["prefill", "decode"], required=True)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--len", type=int, default=8192, help="prefill prompt length")
    p.add_argument("--prompt", type=int, default=16, help="decode prompt length")
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--iters", type=int, default=3, help="prefill iterations inside the capture")
    p.add_argument("--kv-cache-gib", type=float, default=40.0)
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    args = p.parse_args()

    length = args.len if args.mode == "prefill" else args.prompt
    max_tokens = 1 if args.mode == "prefill" else args.steps
    max_len = max(args.len, args.prompt + args.steps + 16)

    llm = LLM(model=args.model, dtype=args.dtype, max_model_len=max_len + 64,
              max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
              gpu_memory_utilization=0.9, kv_cache_memory_bytes=int(args.kv_cache_gib * (1 << 30)),
              enable_prefix_caching=False, seed=0)
    rng = np.random.default_rng(0)

    def prompts():
        return [TokensPrompt(prompt_token_ids=rng.integers(100, 100000, size=length).tolist())
                for _ in range(args.batch)]

    sp = SamplingParams(n=1, temperature=1.0, ignore_eos=True, max_tokens=max_tokens, detokenize=False)
    llm.generate(prompts(), sp, use_tqdm=False)  # warmup / compile / graph capture
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.iters if args.mode == "prefill" else 1):
        llm.generate(prompts(), sp, use_tqdm=False)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
