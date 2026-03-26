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


def top2_4(x: torch.Tensor) -> torch.Tensor:
    groups = x.view(*x.shape[:-1], x.shape[-1] // 4, 4)
    idx = groups.abs().topk(k=2, dim=-1).indices
    mask = torch.zeros_like(groups, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return (groups * mask).view_as(x)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda")
    agemm = load_agemm()

    m, n, k = 256, 5120, 5120
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    w = torch.randn((n, k), device=device, dtype=torch.bfloat16)
    reorder_index = torch.arange(k, device=device, dtype=torch.int16)

    a_comp, e, sfa_sparse, q_res, sf_res = agemm.fused_sparse_residual_quantize_x(x, n)
    qw, sfw = agemm.reorder_quantize_w(w, reorder_index, 0)

    y_sparse = agemm.sparse_matmul(a_comp, qw, e, sfa_sparse, sfw, m, n, k)
    y_res = agemm.matmul(q_res, qw, sf_res, sfw, 1.0)
    y_total = y_sparse + y_res

    x_sparse = top2_4(x)
    qx_sparse_ref, sfa_sparse_ref = agemm.reorder_quantize_x(x_sparse, reorder_index, 0)
    a_comp_ref, e_ref = agemm.compress_sparse_a(qx_sparse_ref, n)
    y_sparse_ref = agemm.sparse_matmul(a_comp_ref, qw, e_ref, sfa_sparse_ref, sfw, m, n, k)

    print(f"problem: M={m}, N={n}, K={k}")
    print(f"A_comp exact match: {bool(torch.equal(a_comp, a_comp_ref))}")
    print(f"E      exact match: {bool(torch.equal(e, e_ref))}")
    print(f"SFA    exact match: {bool(torch.equal(sfa_sparse, sfa_sparse_ref))}")
    print(f"sparse main max diff: {(y_sparse.float() - y_sparse_ref.float()).abs().max().item():.6f}")
    print(f"Q_res shape: {tuple(q_res.shape)}")
    print(f"SF_res bytes: {sf_res.numel()}")
    print(f"output checksum[0:16]: {y_total.flatten()[:16].float().sum().item():.6f}")
    print(f"output max abs: {y_total.float().abs().max().item():.6f}")


if __name__ == "__main__":
    main()
