"""Downstream zero-shot evaluation (the SharQ paper suite) and result summary.

    python lmeval.py run <tag> <model_path> [ENV=VAL ...]   # 7 tasks, 0-shot, lm-eval + vLLM
    python lmeval.py summarize [out.md]                     # markdown table

`run` uses lm-eval's in-process vLLM backend so every format (including SharQ's custom
checkpoint) is served by the same engine. Environment: `lm_eval==0.4.8`, `datasets==3.6.x`
in the vLLM env, plus three patches — lm_eval 0.4.8's vLLM backend predates vLLM 0.24's API:

  1. lm_eval/models/__init__.py -- import only the backends you need, otherwise every optional
     dependency (peft / anthropic / ibm-watsonx / nemo ...) must be installed:
         from . import (dummy, huggingface, vllm_causallms)
  2. lm_eval/models/vllm_causallms.py -- move `import ray` into its own try/except, otherwise
     the `from vllm import LLM, SamplingParams` below it is silently skipped; and fall back for
     `get_tokenizer`: vllm.transformers_utils.tokenizer -> vllm.tokenizers.
  3. the same file -- `LLM.generate(prompt_token_ids=...)` no longer exists; wrap the token-id
     lists in vllm.inputs.TokensPrompt and pass them positionally:
         from vllm.inputs import TokensPrompt
         self.model.generate([TokensPrompt(prompt_token_ids=list(r)) for r in requests], ...)

Task yaml dataset paths also need namespacing under datasets>=3.x: hellaswag -> Rowan/hellaswag,
winogrande -> allenai/winogrande, piqa -> ybisk/piqa, super_glue -> aps/super_glue.
Without the patches, run a vLLM OpenAI server and use `lm_eval --model local-completions`
(scores drift 0.2-0.7pp because the prompt is re-tokenised from a string).
"""

import glob
import json
import os
import subprocess
import sys

OUT = os.environ.get("SHARQ_LMEVAL", "/dockerdata/sharq_work/results/lmeval")
TASKS = "piqa,arc_challenge,boolq,hellaswag,winogrande,lambada_openai,arc_easy"

TAGS = [("BF16", "BF16"), ("FP8", "FP8 W8A8"), ("W4A8", "W4A8"),
        ("NVFP4", "NVFP4 RTN"), ("NVFP4_GPTQ", "NVFP4 + GPTQ"),
        ("SHARQ", "SharQ RTN"), ("SHARQ_GPTQ", "SharQ + GPTQ")]
# conventional metric per task (leaderboard style)
METRIC = {"arc_challenge": "acc_norm", "arc_easy": "acc_norm", "hellaswag": "acc_norm",
          "piqa": "acc_norm", "winogrande": "acc", "boolq": "acc", "lambada_openai": "acc"}
ORDER = ["piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande", "boolq", "lambada_openai"]
SHORT = {"piqa": "PIQA", "arc_easy": "ARC-e", "arc_challenge": "ARC-c", "hellaswag": "HSwag",
         "winogrande": "WinoG", "boolq": "BoolQ", "lambada_openai": "LAMBADA"}


def run(argv):
    if len(argv) < 2:
        raise SystemExit("usage: lmeval.py run <tag> <model_path> [ENV=VAL ...]")
    tag, model = argv[0], argv[1]
    env = dict(os.environ)
    env.update(HF_DATASETS_TRUST_REMOTE_CODE="1", TOKENIZERS_PARALLELISM="false",
               HF_ENDPOINT=env.get("HF_ENDPOINT", "https://hf-mirror.com"))
    for kv in argv[2:]:
        k, _, v = kv.partition("=")
        env[k] = v
    out = os.path.join(OUT, tag)
    os.makedirs(OUT, exist_ok=True)
    cmd = [sys.executable, "-m", "lm_eval", "--model", "vllm",
           "--model_args", f"pretrained={model},dtype=auto,gpu_memory_utilization=0.85,max_model_len=4096",
           "--tasks", TASKS, "--num_fewshot", "0", "--batch_size", "256", "--output_path", out]
    log_path = f"{out}.log"
    print(f"[{tag}] {' '.join(cmd)}\n[{tag}] log -> {log_path}", flush=True)
    with open(log_path, "w") as log:
        rc = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"[{tag}] exit={rc}")
    with open(log_path) as f:
        for line in f:
            if line.startswith("|") and any(t in line for t in ORDER):
                print(line.rstrip())


def load(tag):
    fs = sorted(glob.glob(f"{OUT}/{tag}/**/results*.json", recursive=True))
    if not fs:
        return None
    with open(fs[-1]) as f:
        return json.load(f)["results"]


def summarize(argv):
    data = {t: load(t) for t, _ in TAGS}
    lines = ["| 任务 | " + " | ".join(l for _, l in TAGS) + " |",
             "|" + "---|" * (len(TAGS) + 1)]
    avg = {t: [] for t, _ in TAGS}
    for task in ORDER:
        cells = []
        for t, _ in TAGS:
            d = data[t]
            v = d.get(task, {}).get(f"{METRIC[task]},none") if d else None
            if v is None and d:
                v = d.get(task, {}).get("acc,none")
            cells.append("—" if v is None else f"{v * 100:.2f}")
            if v is not None:
                avg[t].append(v * 100)
        lines.append(f"| {SHORT[task]} | " + " | ".join(cells) + " |")
    means = {t: (sum(v) / len(v) if v else float("nan")) for t, v in avg.items()}
    base = means["BF16"]
    lines.append("| **平均** | " + " | ".join(f"**{means[t]:.2f}**" for t, _ in TAGS) + " |")
    lines.append("| Δ vs BF16 | " + " | ".join("—" if t == "BF16" else f"{means[t] - base:+.2f}"
                                               for t, _ in TAGS) + " |")
    text = "\n".join(lines)
    print(text)
    if argv:
        with open(argv[0], "w") as f:
            f.write(text + "\n")


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "summarize"
    if what == "run":
        run(sys.argv[2:])
    elif what == "summarize":
        summarize(sys.argv[2:])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
