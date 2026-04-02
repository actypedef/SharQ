from __future__ import annotations

import argparse
import functools
import gc
import math
import pprint
import time
from dataclasses import dataclass

import torch

from modeling_sharq import load_benchmark_model


DEFAULT_WARMUP_STEPS = 2
DEFAULT_BENCH_STEPS = 4
DEFAULT_NUM_REPEATS = 10


@dataclass
class BenchmarkInputs:
    prefill_input_ids: torch.Tensor
    prefill_attention_mask: torch.Tensor
    decode_input_ids: torch.Tensor | None
    decode_attention_masks: list[torch.Tensor]


@dataclass
class ModeResults:
    prefill_ms: tuple[float, ...]
    prefill_mem: tuple[float, ...]
    decode_ms: tuple[float, ...] | None
    decode_mem: tuple[float, ...] | None
    e2e_ms: tuple[float, ...] | None
    e2e_mem: tuple[float, ...] | None


def repeated_run(num_repeats=DEFAULT_NUM_REPEATS):
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            results = []
            for _ in range(num_repeats):
                results.append(func(*args, **kwargs))
            return tuple(zip(*results))

        return wrapped

    return decorator


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _as_float_list(values) -> list[float]:
    if isinstance(values, (list, tuple)):
        return [float(v) for v in values]
    return [float(values)]


def _mean(values) -> float:
    values = _as_float_list(values)
    return sum(values) / len(values) if values else 0.0


def _std(values) -> float:
    values = _as_float_list(values)
    if not values:
        return 0.0
    mean_value = _mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _scale(values, factor: float) -> list[float]:
    return [value / factor for value in _as_float_list(values)]


def _format_stat(values, unit: str) -> str:
    return f"{_mean(values):.3f} +- {1.96 * _std(values):.3f}{unit}"


@torch.inference_mode()
def module_benchmark(run_fn, warmup_steps, bench_steps, setup_fn=None):
    for _ in range(warmup_steps):
        state = setup_fn() if setup_fn is not None else None
        run_fn(state)
        del state
        _cleanup()

    times_ms = []
    peak_memories = []
    for _ in range(bench_steps):
        state = setup_fn() if setup_fn is not None else None
        _sync()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start_time = time.perf_counter()
        run_fn(state)
        _sync()
        end_time = time.perf_counter()
        times_ms.append((end_time - start_time) * 1000.0)
        if torch.cuda.is_available():
            peak_memories.append(float(torch.cuda.max_memory_allocated()))
        else:
            peak_memories.append(0.0)
        del state
        _cleanup()

    return _mean(times_ms), max(peak_memories) if peak_memories else 0.0


def build_benchmark_inputs(
    vocab_size: int,
    batch_size: int,
    prefill_seq_len: int,
    decode_steps: int | None,
    device: torch.device,
    seed: int,
) -> BenchmarkInputs:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    prefill_input_ids = torch.randint(
        0,
        max(vocab_size, 2),
        (batch_size, prefill_seq_len),
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    prefill_attention_mask = torch.ones_like(prefill_input_ids)

    decode_input_ids = None
    decode_attention_masks: list[torch.Tensor] = []
    if decode_steps is not None and decode_steps > 0:
        decode_input_ids = torch.randint(
            0,
            max(vocab_size, 2),
            (batch_size, decode_steps),
            generator=generator,
            device=device,
            dtype=torch.long,
        )
        decode_attention_masks = [
            torch.ones((batch_size, prefill_seq_len + step + 1), device=device, dtype=torch.long)
            for step in range(decode_steps)
        ]

    return BenchmarkInputs(
        prefill_input_ids=prefill_input_ids,
        prefill_attention_mask=prefill_attention_mask,
        decode_input_ids=decode_input_ids,
        decode_attention_masks=decode_attention_masks,
    )


def run_prefill(model, bench_inputs: BenchmarkInputs, warmup_steps: int, bench_steps: int, num_repeats: int):
    def _prefill(_state=None):
        model(
            input_ids=bench_inputs.prefill_input_ids,
            attention_mask=bench_inputs.prefill_attention_mask,
            use_cache=True,
        )

    @repeated_run(num_repeats)
    def _bench():
        return module_benchmark(_prefill, warmup_steps=warmup_steps, bench_steps=bench_steps)

    return _bench()


def run_decode(
    model,
    bench_inputs: BenchmarkInputs,
    warmup_steps: int,
    bench_steps: int,
    num_repeats: int,
):
    if bench_inputs.decode_input_ids is None:
        raise ValueError("Decode benchmark requested without decode inputs.")

    def _setup_decode():
        outputs = model(
            input_ids=bench_inputs.prefill_input_ids,
            attention_mask=bench_inputs.prefill_attention_mask,
            use_cache=True,
        )
        return outputs.past_key_values

    def _decode_for_multiple_steps(past_key_values):
        past = past_key_values
        for step in range(bench_inputs.decode_input_ids.shape[1]):
            outputs = model(
                input_ids=bench_inputs.decode_input_ids[:, step : step + 1],
                attention_mask=bench_inputs.decode_attention_masks[step],
                past_key_values=past,
                use_cache=True,
            )
            past = outputs.past_key_values

    @repeated_run(num_repeats)
    def _bench():
        return module_benchmark(
            _decode_for_multiple_steps,
            warmup_steps=warmup_steps,
            bench_steps=bench_steps,
            setup_fn=_setup_decode,
        )

    return _bench()


def run_e2e(model, bench_inputs: BenchmarkInputs, warmup_steps: int, bench_steps: int, num_repeats: int):
    if bench_inputs.decode_input_ids is None:
        raise ValueError("E2E benchmark requested without decode inputs.")

    def _prefill_and_decode(_state=None):
        outputs = model(
            input_ids=bench_inputs.prefill_input_ids,
            attention_mask=bench_inputs.prefill_attention_mask,
            use_cache=True,
        )
        past = outputs.past_key_values
        for step in range(bench_inputs.decode_input_ids.shape[1]):
            outputs = model(
                input_ids=bench_inputs.decode_input_ids[:, step : step + 1],
                attention_mask=bench_inputs.decode_attention_masks[step],
                past_key_values=past,
                use_cache=True,
            )
            past = outputs.past_key_values

    @repeated_run(num_repeats)
    def _bench():
        return module_benchmark(_prefill_and_decode, warmup_steps=warmup_steps, bench_steps=bench_steps)

    return _bench()


@torch.inference_mode()
def run_all_for_model(
    model,
    bench_inputs: BenchmarkInputs,
    warmup_steps: int,
    bench_steps: int,
    num_repeats: int,
    decode_steps: int | None,
) -> ModeResults:
    model.eval()
    time_prefill, memory_prefill = run_prefill(
        model,
        bench_inputs,
        warmup_steps=warmup_steps,
        bench_steps=bench_steps,
        num_repeats=num_repeats,
    )

    if decode_steps is not None and decode_steps > 0:
        _cleanup()
        time_decode, memory_decode = run_decode(
            model,
            bench_inputs,
            warmup_steps=warmup_steps,
            bench_steps=bench_steps,
            num_repeats=num_repeats,
        )
        _cleanup()
        time_e2e, memory_e2e = run_e2e(
            model,
            bench_inputs,
            warmup_steps=warmup_steps,
            bench_steps=bench_steps,
            num_repeats=num_repeats,
        )
        _cleanup()
    else:
        time_decode = None
        time_e2e = None
        memory_decode = None
        memory_e2e = None

    return ModeResults(
        prefill_ms=time_prefill,
        prefill_mem=memory_prefill,
        decode_ms=time_decode,
        decode_mem=memory_decode,
        e2e_ms=time_e2e,
        e2e_mem=memory_e2e,
    )


def print_mode_results(mode: str, results: ModeResults):
    mem_scale = 1024 ** 3
    print(f"Prefill {mode} time: {_format_stat(results.prefill_ms, 'ms')}")
    print("--------------")
    if results.decode_ms is not None:
        print(f"Decode {mode} time: {_format_stat(results.decode_ms, 'ms')}")
        print(f"E2E {mode} time: {_format_stat(results.e2e_ms, 'ms')}")
        print("--------------")
    print(f"{mode} prefill memory: {_format_stat(_scale(results.prefill_mem, mem_scale), 'GB')}")
    if results.decode_ms is not None and results.decode_mem is not None and results.e2e_mem is not None:
        print(f"{mode} decode memory: {_format_stat(_scale(results.decode_mem, mem_scale), 'GB')}")
        print(f"{mode} e2e memory: {_format_stat(_scale(results.e2e_mem, mem_scale), 'GB')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="End-to-end benchmark for SHARQ and FP16 baseline models.")
    parser.add_argument("--model", type=str, required=True, help="Local model path or Hugging Face model path.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--prefill_seq_len", type=int, default=2048, help="Prefill sequence length.")
    parser.add_argument("--decode_steps", type=int, default=None, help="Decode steps. Omit to benchmark prefill only.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["SHARQ", "FP16"],
        choices=["SHARQ", "FP16"],
        help="Benchmark modes to run.",
    )
    parser.add_argument("--warmup_steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--bench_steps", type=int, default=DEFAULT_BENCH_STEPS)
    parser.add_argument("--num_repeats", type=int, default=DEFAULT_NUM_REPEATS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kv_cache", action="store_true", help="Enable fake-int4 KV cache path in wrapped attention.")
    parser.add_argument(
        "--no_extra_fusion",
        action="store_true",
        help="Disable SHARQ extra fusion path for debugging.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser


def benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This benchmark expects a CUDA device.")
    torch.cuda.set_device(device)

    all_results: dict[str, ModeResults | dict[str, str]] = {}

    for mode in args.modes:
        print(f"\n=== Benchmarking {mode} ===")
        model = None
        try:
            model = load_benchmark_model(
                args.model,
                mode=mode,
                device=device,
                kv_cache=args.kv_cache,
                extra_fusion=not args.no_extra_fusion,
                trust_remote_code=args.trust_remote_code,
            )
            vocab_size = int(getattr(model.config, "vocab_size", 151936))
            bench_inputs = build_benchmark_inputs(
                vocab_size=vocab_size,
                batch_size=args.batch_size,
                prefill_seq_len=args.prefill_seq_len,
                decode_steps=args.decode_steps,
                device=device,
                seed=args.seed,
            )
            results = run_all_for_model(
                model,
                bench_inputs,
                warmup_steps=args.warmup_steps,
                bench_steps=args.bench_steps,
                num_repeats=args.num_repeats,
                decode_steps=args.decode_steps,
            )
            all_results[mode] = results
            print_mode_results(mode, results)
        except Exception as exc:
            if _is_oom_error(exc):
                all_results[mode] = {"error": f"OOM: {exc}"}
                print(f"{mode} skipped: CUDA OOM during load or benchmark.")
            else:
                raise
        finally:
            del model
            _cleanup()

    if "SHARQ" in all_results and "FP16" in all_results:
        sharq_results = all_results["SHARQ"]
        fp16_results = all_results["FP16"]
        if isinstance(sharq_results, ModeResults) and isinstance(fp16_results, ModeResults):
            sharq_prefill = _mean(sharq_results.prefill_ms)
            fp16_prefill = _mean(fp16_results.prefill_ms)
            print("\n=== Relative Speedup ===")
            print(f"SHARQ vs FP16 prefill speedup: {fp16_prefill / sharq_prefill:.3f}x")
            if args.decode_steps is not None and sharq_results.e2e_ms is not None and fp16_results.e2e_ms is not None:
                sharq_e2e = _mean(sharq_results.e2e_ms)
                fp16_e2e = _mean(fp16_results.e2e_ms)
                print(f"SHARQ vs FP16 E2E speedup: {fp16_e2e / sharq_e2e:.3f}x")


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    pprint.pprint(vars(parsed_args))
    benchmark(parsed_args)
