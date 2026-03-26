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


def make_outlier_rich_input(m: int, k: int, seed: int, device: torch.device) -> torch.Tensor:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    x = torch.randn((m, k), dtype=torch.float32, device=device)
    x_groups = x.view(m, -1, 4)
    x_groups[..., 0] *= 12.0
    x_groups[..., 2] *= 8.0
    x = x_groups.view(m, k)

    num_channels = max(1, int(0.08 * k))
    outlier_idx = torch.randperm(k, device=device)[:num_channels]
    outlier_scale = torch.empty((num_channels,), dtype=torch.float32, device=device).uniform_(10.0, 40.0)
    x[:, outlier_idx] *= outlier_scale
    return x.to(dtype=torch.bfloat16)


def raw_fp4_chunk_violation_stats(q_sparse: torch.Tensor) -> tuple[float, dict[int, int]]:
    # CUTLASS sparse FP4 compresses 4 raw bytes -> 2 raw bytes.
    # Each raw byte stores 2 logical FP4 elements, so one chunk spans 8 logical scalars.
    byte_chunks = q_sparse.view(q_sparse.size(0), -1, 4)
    nonzero_raw_bytes = (byte_chunks != 0).sum(dim=-1)
    violation_rate = (nonzero_raw_bytes > 2).float().mean().item()
    values, counts = torch.unique(nonzero_raw_bytes, return_counts=True)
    hist = {int(v): int(c) for v, c in zip(values.cpu(), counts.cpu())}
    return violation_rate, hist


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    agemm = load_agemm()

    m = 256
    n = 5120
    k = 5120
    seed = 0

    x = make_outlier_rich_input(m, k, seed, device)
    q_sparse, sfa_sparse, q_res, sf_res = agemm.fused_sparse_residual_quantize_x_debug(x, n)
    violation_rate, hist = raw_fp4_chunk_violation_stats(q_sparse)

    print(f"problem: M={m}, N={n}, K={k}, seed={seed}")
    print("CUTLASS sparse FP4 expects at most 2 nonzero raw bytes in each 4-byte chunk")
    print(f"raw q_sparse byte-level violation rate: {violation_rate:.6f}")
    print(f"raw-byte nonzero histogram        : {hist}")
    print(f"q_sparse shape                    : {tuple(q_sparse.shape)}")
    print(f"SFA_sparse bytes                  : {sfa_sparse.numel()}")
    print(f"Q_res shape                       : {tuple(q_res.shape)}")
    print(f"SF_res bytes                      : {sf_res.numel()}")


if __name__ == "__main__":
    main()
