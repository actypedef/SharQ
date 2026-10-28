"""WikiText-2 perplexity inside vLLM (prompt logprobs), seqlen 2048, non-overlapping windows.

Token ids are pre-tokenized (Llama-3 tokenizer, "\\n\\n".join(test["text"])) to keep the vLLM
env free of `datasets`.
"""

import argparse
import json
import math
import time

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

DEFAULT_IDS = "/dockerdata/sharq_work/results/ppl/wikitext2_test_llama3_ids.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--ids", default=DEFAULT_IDS)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--max-chunks", type=int, default=-1)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    with open(args.ids) as f:
        ids = json.load(f)
    n_chunks = len(ids) // args.seqlen
    if args.max_chunks > 0:
        n_chunks = min(n_chunks, args.max_chunks)
    prompts = [TokensPrompt(prompt_token_ids=ids[i * args.seqlen:(i + 1) * args.seqlen]) for i in range(n_chunks)]

    llm = LLM(model=args.model, tokenizer=args.tokenizer or args.model, dtype=args.dtype,
              max_model_len=args.seqlen + 16, max_num_seqs=8, max_num_batched_tokens=4096,
              gpu_memory_utilization=0.2, kv_cache_memory_bytes=1 << 30, enable_prefix_caching=False, seed=0)
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0), use_tqdm=False)
    nll, count = 0.0, 0
    for o in outs:
        for tid, lp in zip(o.prompt_token_ids[1:], o.prompt_logprobs[1:]):
            nll -= lp[tid].logprob
            count += 1
    ppl = math.exp(nll / count)
    print(f"[{args.label}] wikitext2 ppl = {ppl:.4f}  ({n_chunks} x {args.seqlen} tokens, {time.time() - t0:.1f}s)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"label": args.label, "ppl": ppl, "chunks": n_chunks, "seqlen": args.seqlen}, f)


if __name__ == "__main__":
    main()
