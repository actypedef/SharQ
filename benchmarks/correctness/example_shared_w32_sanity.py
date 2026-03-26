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


def make_fixed_top2_4_ones(m: int, k: int, device: torch.device) -> torch.Tensor:
    groups = torch.zeros((m, k // 4, 4), device=device, dtype=torch.bfloat16)
    groups[..., 0] = 1
    groups[..., 1] = 1
    return groups.view(m, k)


def run_case(name: str, x_sparse: torch.Tensor, w: torch.Tensor, backend) -> None:
    m, k = x_sparse.shape
    n = w.shape[0]
    reorder_index = torch.arange(k, device=x_sparse.device, dtype=torch.int16)

    qx_sparse, sfx_sparse = backend.reorder_quantize_x(x_sparse, reorder_index, 0)
    qw32, sfw_sparse32, sfw_dense16 = backend.quantize_w32_shared(w)
    a_comp, e = backend.compress_sparse_a(qx_sparse, n)

    y_dense = backend.matmul(qx_sparse, qw32, sfx_sparse, sfw_dense16, 1.0).float()
    y_sparse = backend.sparse_matmul(a_comp, qw32, e, sfx_sparse, sfw_sparse32, m, n, k).float()
    y_ref = torch.matmul(x_sparse.float(), w.float().t())
    diff = (y_dense - y_sparse).abs()
    diff_ref = (y_dense - y_ref).abs()

    print(name)
    print(f"  dense checksum [0:16]: {y_dense.flatten()[:16].sum().item():.8f}")
    print(f"  sparse checksum[0:16]: {y_sparse.flatten()[:16].sum().item():.8f}")
    print(f"  max |dense-sparse|   : {diff.max().item():.8f}")
    print(f"  mean|dense-sparse|   : {diff.mean().item():.8f}")
    print(f"  max |dense-bf16|     : {diff_ref.max().item():.8f}")
    print(f"  mean|dense-bf16|     : {diff_ref.mean().item():.8f}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = torch.device("cuda")
    backend = load_sharq_ops()

    k = 5120
    m = 4

    x_sparse = make_fixed_top2_4_ones(m, k, device)
    eye = torch.eye(k, device=device, dtype=torch.bfloat16)
    rand_w = torch.randn((k, k), device=device, dtype=torch.bfloat16) * 0.1
    block_scaled_w = torch.randn((k, k), device=device, dtype=torch.bfloat16) * 0.1
    block_scaled_w[:, 16:32] = block_scaled_w[:, 0:16]
    block_scaled_w[:, 48:64] = block_scaled_w[:, 32:48]

    run_case("shared-w32: x_sparse x eye", x_sparse, eye, backend)
    print()
    run_case("shared-w32: x_sparse x rand_w", x_sparse, rand_w, backend)
    print()
    run_case("shared-w32: x_sparse x partially_tied_w", x_sparse, block_scaled_w, backend)


if __name__ == "__main__":
    main()
