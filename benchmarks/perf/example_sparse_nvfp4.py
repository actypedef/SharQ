from __future__ import annotations

import sys
import time
from pathlib import Path

import torch


def load_agemm():
    repo_root = Path(__file__).resolve().parents[2]
    build_dir = repo_root / "kernels" / "build_cmake_sm120a"
    sys.path.insert(0, str(build_dir))
    try:
        import sharq_ops as backend  # type: ignore
    except ImportError:
        import agemm as backend  # type: ignore

    return backend


def make_structured_sparse_input(m: int, k: int, device: torch.device) -> torch.Tensor:
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    groups = x.view(m, k // 4, 4)
    groups[..., 2:] = 0
    return x


def benchmark(fn, iters: int = 100, warmup: int = 20) -> tuple[torch.Tensor, float]:
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / iters
    return out, elapsed_ms


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    agemm = load_agemm()
    device = torch.device("cuda")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    m, n, k = 128, 128, 4096
    x = make_structured_sparse_input(m, k, device)
    w = torch.randn((n, k), device=device, dtype=torch.bfloat16)
    reorder_index = torch.arange(k, device=device, dtype=torch.int16)

    qx, sfx = agemm.reorder_quantize_x(x, reorder_index, 0)
    qw, sfw = agemm.reorder_quantize_w(w, reorder_index, 0)
    a_comp, e = agemm.compress_sparse_a(qx, n)

    out_sparse, sparse_ms = benchmark(
        lambda: agemm.sparse_matmul(a_comp, qw, e, sfx, sfw, m, n, k)
    )
    out_dense, dense_ms = benchmark(
        lambda: agemm.matmul(qx, qw, sfx, sfw, 1.0)
    )

    checksum = out_sparse.flatten()[:16].float().sum().item()
    max_diff = (out_sparse.float() - out_dense.float()).abs().max().item()

    a_comp_bytes, e_bytes, sfa_bytes, sfb_bytes = agemm.get_sparse_nvfp4_buffer_sizes(m, n, k)

    print(f"problem: M={m}, N={n}, K={k}")
    print(
        "buffers(bytes): "
        f"A_comp={a_comp_bytes}, E={e_bytes}, SFA={sfa_bytes}, SFB={sfb_bytes}"
    )
    print(f"sparse checksum[0:16]: {checksum:.6f}")
    print(f"sparse avg runtime: {sparse_ms:.6f} ms")
    print(f"dense  avg runtime: {dense_ms:.6f} ms")
    print(f"max |sparse - dense|: {max_diff:.6f}")


if __name__ == "__main__":
    main()
