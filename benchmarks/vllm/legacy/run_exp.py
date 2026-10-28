# vLLM/run_exp.py

import argparse
import os
import re
import subprocess
import time

MODELS_ROOT = "/root/autodl-tmp"
STATS_DIR = "/root/autodl-tmp/vllm-micromix/saved"
QUANT_SAVE_ROOT = "/root/autodl-tmp/quantized"
LOG_DIR = "/root/autodl-tmp/bench_logs"

TARGET_MODELS = [
    "Llama-3.1-8B",
    # "Qwen2.5-7B",
    # "Qwen/Qwen2.5-14B",
    # "Qwen/Qwen3-4B",
]

GPU_MEM_UTIL = "0.90"


def run_command(cmd, log_file, description):
    print(f"[*] Running: {description}")
    print(f"    -> log: {os.path.basename(log_file)}")

    full_output = ""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 20} {description} {'=' * 20}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command: {cmd}\n\n")
        f.flush()

        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                f.write(line)
                full_output += line

        f.flush()

        if process.returncode != 0:
            print(f"[!] Failed: {description}. See {log_file}")
            f.write(f"\n[!] FAILED with return code {process.returncode}\n")
            return False, full_output

        print(f"[+] Done: {description}")
        f.write("\n[+] SUCCESS\n")
        return True, full_output


def check_model_exists(save_path):
    return os.path.exists(os.path.join(save_path, "config.json"))


def parse_metrics(output_text, bench_type):
    metrics = {}
    if bench_type == "latency":
        match = re.search(r"Avg latency:\s+([\d\.]+)\s+seconds", output_text)
        metrics["avg_latency"] = float(match.group(1)) if match else -1.0
    elif bench_type == "throughput":
        match = re.search(
            r"Throughput:\s+([\d\.]+)\s+requests/s,\s+([\d\.]+)\s+total tokens/s,\s+([\d\.]+)\s+output tokens/s",
            output_text,
        )
        if match:
            metrics["req_per_s"] = float(match.group(1))
            metrics["total_tok_per_s"] = float(match.group(2))
            metrics["out_tok_per_s"] = float(match.group(3))
        else:
            metrics["req_per_s"] = -1.0
            metrics["total_tok_per_s"] = -1.0
            metrics["out_tok_per_s"] = -1.0
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Auto quantization and benchmark runner")
    parser.add_argument("--clean", action="store_true", help="Re-quantize even if outputs already exist")
    parser.add_argument(
        "--target",
        nargs="+",
        default=["throughput", "latency"],
        help="Targets to run, e.g. --target latency throughput",
    )
    parser.add_argument("--batch-size", type=str, default="8", help="Latency benchmark batch size")
    parser.add_argument("--input-len", type=str, default="1024", help="Prompt length")
    parser.add_argument("--output-len", type=str, default="128", help="Output length")
    parser.add_argument("--num-prompts", type=str, default="500", help="Total prompts for throughput bench")
    parser.add_argument("--max-model-len", type=str, default="2048", help="Max model length")
    parser.add_argument(
        "--gpu-mem-util",
        type=str,
        default=GPU_MEM_UTIL,
        help="Value passed to --gpu-memory-utilization",
    )
    # parser.add_argument(
    #     "--eager-modes",
    #     nargs="*",
    #     default=["SHARQ"],
    #     help="Benchmark modes that should append --enforce-eager to avoid CUDA graph capture",
    # )
    parser.add_argument(
        "--extra-bench-args",
        type=str,
        default="",
        help="Extra engine args appended to every vLLM benchmark command",
    )
    args = parser.parse_args()

    target_tasks = [t.lower() for t in args.target]
    # eager_modes = {mode.upper() for mode in args.eager_modes}

    os.makedirs(QUANT_SAVE_ROOT, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    global_results = {}

    print(f"Start tasks... (clean={args.clean})")
    print(f"Targets: {target_tasks}")
    print(
        f"Config: batch={args.batch_size}, input={args.input_len}, output={args.output_len}, max_len={args.max_model_len}"
    )
    print(f"GPU mem util: {args.gpu_mem_util}")
    # print(f"Eager modes: {sorted(eager_modes)}")
    print(f"Log dir: {LOG_DIR}")

    for relative_path in TARGET_MODELS:
        full_model_path = os.path.join(MODELS_ROOT, relative_path)
        model_name = os.path.basename(relative_path)
        quant_save_path_s = os.path.join(QUANT_SAVE_ROOT, "sharq", model_name)
        quant_save_path_m = os.path.join(QUANT_SAVE_ROOT, "micromix", model_name)

        global_results[model_name] = {}

        print("\n\n" + "#" * 50)
        print(f"Processing model: {model_name}")
        print("#" * 50)

        should_quantize_s = True
        if check_model_exists(quant_save_path_s):
            if args.clean:
                print("[!] Existing SHARQ model found, but --clean is enabled. Re-quantizing...")
            else:
                print("[i] Existing SHARQ model found. Skipping SHARQ quantization.")
                should_quantize_s = False

        if should_quantize_s:
            quant_log_file = os.path.join(LOG_DIR, f"{model_name}_Quant_SHARQ.log")
            cmd_quant = (
                f"python sharq_quant.py "
                f"--model {full_model_path} "
                f"--save_path {quant_save_path_s}"
            )
            success, _ = run_command(cmd_quant, quant_log_file, "Quantization [SHARQ]")
            if not success:
                print(f"[-] SHARQ quantization failed for {model_name}.")

        should_quantize_m = True
        if check_model_exists(quant_save_path_m):
            if args.clean:
                print("[!] Existing Micromix model found, but --clean is enabled. Re-quantizing...")
            else:
                print("[i] Existing Micromix model found. Skipping Micromix quantization.")
                should_quantize_m = False

        if should_quantize_m:
            quant_log_file = os.path.join(LOG_DIR, f"{model_name}_Quant_Micromix.log")
            cmd_quant = (
                f"python micromix_quant.py "
                f"--model {full_model_path} "
                f"--stats_dir {STATS_DIR} "
                f"--save_path {quant_save_path_m}"
            )
            success, _ = run_command(cmd_quant, quant_log_file, "Quantization [Micromix]")
            if not success:
                print(f"[-] Micromix quantization failed for {model_name}.")

        bench_configs = [
            {
                "name": "SHARQ",
                "model_path": quant_save_path_s,
                "quant_arg": "--quantization sharq",
                "available": check_model_exists(quant_save_path_s),
            },
            {
                "name": "FP16",
                "model_path": full_model_path,
                "quant_arg": "",
                "available": True,
            },
            {
                "name": "FP8",
                "model_path": full_model_path,
                "quant_arg": "--quantization fp8",
                "available": True,
            },
            {
                "name": "Micromix",
                "model_path": quant_save_path_m,
                "quant_arg": "--quantization micromix",
                "available": check_model_exists(quant_save_path_m),
            },
        ]

        for config in bench_configs:
            mode_name = config["name"]
            bench_log_file = os.path.join(LOG_DIR, f"{model_name}_Bench_{mode_name}.log")
            mode_extra_args = args.extra_bench_args
            # if mode_name.upper() in eager_modes:
            #     mode_extra_args = f"{mode_extra_args} --enforce-eager".strip()

            if not config["available"]:
                global_results[model_name][mode_name] = "N/A"
                continue

            print(f"\n--- Running Benchmark: {mode_name} ---")
            global_results[model_name][mode_name] = {}

            if "throughput" in target_tasks:
                cmd_throughput = (
                    f"python -m vllm.entrypoints.cli.main bench throughput "
                    f"--model {config['model_path']} "
                    f"{config['quant_arg']} "
                    f"--gpu-memory-utilization {args.gpu_mem_util} "
                    f"--max-model-len {args.max_model_len} "
                    f"--random-input-len {args.input_len} "
                    f"--random-output-len {args.output_len} "
                    f"--num-prompts {args.num_prompts} "
                    f"{mode_extra_args} "
                )
                success_t, out_t = run_command(cmd_throughput, bench_log_file, f"Bench Throughput [{mode_name}]")
                if success_t:
                    global_results[model_name][mode_name].update(parse_metrics(out_t, "throughput"))
                else:
                    global_results[model_name][mode_name]["error_throughput"] = "OOM/Fail"

            if "latency" in target_tasks:
                cmd_latency = (
                    f"python -m vllm.entrypoints.cli.main bench latency "
                    f"--model {config['model_path']} "
                    f"{config['quant_arg']} "
                    f"--gpu-memory-utilization {args.gpu_mem_util} "
                    f"--max-model-len {args.max_model_len} "
                    f"--input-len {args.input_len} "
                    f"--output-len {args.output_len} "
                    f"--batch-size {args.batch_size} "
                    f"{mode_extra_args} "
                )
                success_l, out_l = run_command(cmd_latency, bench_log_file, f"Bench Latency [{mode_name}]")
                if success_l:
                    global_results[model_name][mode_name].update(parse_metrics(out_l, "latency"))
                else:
                    if "error_throughput" not in global_results[model_name][mode_name]:
                        global_results[model_name][mode_name]["error_latency"] = "Lat Fail"
                    else:
                        global_results[model_name][mode_name]["error_latency"] = "All Fail"

    print("\n\n")
    print("=" * 110)
    print(f"{'BENCHMARK SUMMARY':^110}")
    print(f"Config: BS={args.batch_size} | In={args.input_len} | Out={args.output_len} | Targets={target_tasks}")
    print("=" * 110)

    header = f"{'Model':<25} | {'Mode':<10} | {'Latency (s)':<12} | {'Req/s':<10} | {'Total Tok/s':<12} | {'Out Tok/s':<12}"
    print(header)
    print("-" * 110)

    for model_name, modes in global_results.items():
        for mode_name, metrics in modes.items():
            if isinstance(metrics, str):
                row = f"{model_name:<25} | {mode_name:<10} | {metrics:<12} | {'-':<10} | {'-':<12} | {'-':<12}"
            elif metrics.get("error_throughput") or metrics.get("error_latency"):
                row = f"{model_name:<25} | {mode_name:<10} | {'FAILED':<12} | {'-':<10} | {'-':<12} | {'-':<12}"
            else:
                if "avg_latency" in metrics:
                    lat = f"{metrics['avg_latency']:.4f}"
                elif "latency" not in target_tasks:
                    lat = "-"
                else:
                    lat = "Err"

                if "req_per_s" in metrics:
                    req = f"{metrics['req_per_s']:.2f}"
                    ttok = f"{metrics['total_tok_per_s']:.1f}"
                    otok = f"{metrics['out_tok_per_s']:.1f}"
                elif "throughput" not in target_tasks:
                    req, ttok, otok = "-", "-", "-"
                else:
                    req, ttok, otok = "Err", "Err", "Err"

                prefix = "*" if mode_name in {"SHARQ", "Micromix"} else " "
                row = f"{prefix + model_name[:24]:<25} | {mode_name:<10} | {lat:<12} | {req:<10} | {ttok:<12} | {otok:<12}"
            print(row)
        print("-" * 110)

    print(f"\nFull logs saved to: {LOG_DIR}")


if __name__ == "__main__":
    main()

