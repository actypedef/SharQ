import argparse
import os
import time

import torch
from datautils import DEV, get_loaders
from eval import eval_ppl

from model_utils import quantize_model_llama, quantize_model_mixtral, quantize_model_qwen


def get_llama(model_name_or_path: str):
    from transformers import LlamaForCausalLM

    return LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)


def get_qwen(model_name_or_path: str):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype="auto")


def get_mixtral(model_name_or_path: str):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype="auto")


def load_model_and_quantizer(model_name_or_path: str):
    model_name = model_name_or_path.rstrip("/").split("/")[-1]

    if "llama" in model_name_or_path.lower():
        return model_name, get_llama(model_name_or_path), quantize_model_llama
    if "qwen" in model_name_or_path.lower():
        return model_name, get_qwen(model_name_or_path), quantize_model_qwen
    if "mixtral" in model_name_or_path.lower():
        return model_name, get_mixtral(model_name_or_path), quantize_model_mixtral

    raise ValueError(f"Unsupported model family for path: {model_name_or_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate dense NVFP4 or SharQ quantized LLM inference.")
    parser.add_argument("model", type=str, help="Hugging Face model path or local checkpoint path.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for evaluation loaders.")
    parser.add_argument("--kv_cache", action="store_true", help="Apply simple int4 fake quantization to KV cache tensors.")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated lm-eval tasks.")
    parser.add_argument("--eval_ppl", action="store_true", help="Evaluate perplexity on the selected dataset.")
    parser.add_argument("--lm_eval_num_fewshot", type=int, default=0, help="Number of few-shot examples for lm-eval.")
    parser.add_argument("--lm_eval_limit", type=int, default=-1, help="Limit lm-eval examples; -1 uses the full task.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "ptb", "c4"],
        help="Dataset used for perplexity evaluation.",
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        default="SHARQ",
        choices=["NVFP4", "SHARQ", "SHARQ_SIM"],
        help=(
            "Quantization path to evaluate. SHARQ is the fused sparse-residual FP4 method, "
            "and SHARQ_SIM runs a pure PyTorch fake-quantized fake-sparse simulation."
        ),
    )
    return parser


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model_name, model, quantize_model = load_model_and_quantizer(args.model)
    model.eval()

    torch.cuda.reset_max_memory_allocated()
    start_time = time.time()
    model = quantize_model(model, device=DEV, kv_cache=args.kv_cache, quant_type=args.quant_type)
    end_time = time.time()
    peak_memory = torch.cuda.max_memory_allocated()

    print(model)
    print(f"Quantized Model Size: {peak_memory / (1024 * 1024 * 1024):.2f} GB")
    print(f"Quantized Type: {args.quant_type}")
    print(f"Total quantization time: {end_time - start_time:.2f} seconds")

    if args.eval_ppl:
        _, testloader = get_loaders(args.dataset, seed=args.seed, model=args.model, seqlen=2048)
        print(f"Evaluating {args.dataset} ...")
        ppl = eval_ppl(model, testloader, "cuda")
        print(f"Result,{args.dataset},{ppl:.3f}")

    if args.tasks is not None:
        from lm_eval import evaluator as lm_evaluator
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import TaskManager
        from lm_eval.utils import make_table

        batch_size = "auto"
        if "mmlu" in args.tasks:
            batch_size = 2

        lm = HFLM(model, batch_size=batch_size)
        lm.model.eval()
        for param in lm.model.parameters():
            param.requires_grad = False

        lm._device = DEV
        lm._model = lm._model.to(lm._device)

        task_manager = TaskManager()
        task_names = args.tasks.split(",")
        results = lm_evaluator.simple_evaluate(
            lm,
            tasks=task_names,
            num_fewshot=args.lm_eval_num_fewshot,
            limit=None if args.lm_eval_limit == -1 else args.lm_eval_limit,
            batch_size=batch_size,
            task_manager=task_manager,
        )

        table_results = make_table(results)
        print(table_results)

        os.makedirs("./results", exist_ok=True)
        from datetime import datetime
        import logging

        log_filename = f"./results/log_{model_name.lower()}_{args.tasks}_{datetime.now().strftime('%Y%m%d')}.log"
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.info(f"Results for {model_name.lower()} on {args.tasks}:\n{table_results}")


if __name__ == "__main__":
    main()
