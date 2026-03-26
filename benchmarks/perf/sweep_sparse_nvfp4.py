from __future__ import annotations

import sys
from pathlib import Path

import torch


def load_sharq_ops():
    repo_root = Path(__file__).resolve().parents[2]
    build_dir = repo_root / "kernels" / "build_cmake_sm120a"
    sys.path.insert(0, str(build_dir))
    import sharq_ops as backend  # type: ignore

    return backend


def make_structured_sparse_input(m: int, k: int, device: torch.device) -> torch.Tensor:
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    groups = x.view(m, k // 4, 4)
    groups[..., 2:] = 0
    return x


def benchmark_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    backend = load_sharq_ops()
    device = torch.device("cuda")

    n = 5120
    k = 5120
    ms = [256, 512, 1024, 2048, 4096, 8192]

    w = torch.randn((n, k), device=device, dtype=torch.bfloat16)
    reorder_index = torch.arange(k, device=device, dtype=torch.int16)
    qw, sfw = backend.reorder_quantize_w(w, reorder_index, 0)

    print(f"{'M':>6} {'warmup':>8} {'iters':>8} {'sparse(ms)':>12} {'dense(ms)':>11} {'speedup':>9}")

    for m in ms:
        x = make_structured_sparse_input(m, k, device)
        qx, sfx = backend.reorder_quantize_x(x, reorder_index, 0)
        a_comp, e = backend.compress_sparse_a(qx, n)

        warmup = 100
        iters = 1000

        sparse_ms = benchmark_cuda(
            lambda: backend.sparse_matmul(a_comp, qw, e, sfx, sfw, m, n, k),
            warmup=warmup,
            iters=iters,
        )
        dense_ms = benchmark_cuda(
            lambda: backend.matmul(qx, qw, sfx, sfw, 1.0),
            warmup=warmup,
            iters=iters,
        )
        speedup = dense_ms / sparse_ms
        print(f"{m:6d} {warmup:8d} {iters:8d} {sparse_ms:12.6f} {dense_ms:11.6f} {speedup:9.4f}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
