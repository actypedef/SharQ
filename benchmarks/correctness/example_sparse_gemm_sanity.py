from __future__ import annotations

import sys
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


def make_top2_4(x: torch.Tensor) -> torch.Tensor:
    groups = x.view(*x.shape[:-1], x.shape[-1] // 4, 4)
    idx = groups.abs().topk(k=2, dim=-1).indices
    mask = torch.zeros_like(groups, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return (groups * mask).view_as(x)


def run_case(name: str, x: torch.Tensor, w: torch.Tensor, agemm) -> None:
    m, k = x.shape
    n = w.shape[0]
    reorder_index = torch.arange(k, device=x.device, dtype=torch.int16)

    qx_sparse, sfx_sparse = agemm.reorder_quantize_x(x, reorder_index, 0)
    qw, sfw = agemm.reorder_quantize_w(w, reorder_index, 0)
    a_comp, e = agemm.compress_sparse_a(qx_sparse, n)

    y_dense = agemm.matmul(qx_sparse, qw, sfx_sparse, sfw, 1.0).float()
    y_sparse = agemm.sparse_matmul(a_comp, qw, e, sfx_sparse, sfw, m, n, k).float()
    diff = (y_dense - y_sparse).abs()

    print(name)
    print(f"  dense checksum [0:16]: {y_dense.flatten()[:16].sum().item():.8f}")
    print(f"  sparse checksum[0:16]: {y_sparse.flatten()[:16].sum().item():.8f}")
    print(f"  max diff            : {diff.max().item():.8f}")
    print(f"  mean diff           : {diff.mean().item():.8f}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = torch.device("cuda")
    agemm = load_agemm()

    k = 5120
    eye = torch.eye(k, device=device, dtype=torch.bfloat16)
    ones = torch.ones((1, k), device=device, dtype=torch.bfloat16)
    zeros = torch.zeros((1, k), device=device, dtype=torch.bfloat16)
    rand_sparse = make_top2_4(torch.randn((1, k), device=device, dtype=torch.bfloat16))

    run_case("zeros x eye", zeros, eye, agemm)
    print()
    run_case("ones x eye", make_top2_4(ones), eye, agemm)
    print()
    run_case("rand_sparse x eye", rand_sparse, eye, agemm)
    print()
    run_case("ones x ones", make_top2_4(ones), torch.ones((k, k), device=device, dtype=torch.bfloat16), agemm)


if __name__ == "__main__":
    main()
