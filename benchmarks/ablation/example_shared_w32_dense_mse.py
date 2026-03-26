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


def summarize(tag: str, pred: torch.Tensor, ref: torch.Tensor) -> None:
    diff = (pred.float() - ref.float())
    mse = diff.square().mean().item()
    rmse = mse ** 0.5
    rel_rmse = rmse / (ref.float().square().mean().sqrt().item() + 1e-12)
    print(tag)
    print(f"  mse      : {mse:.8f}")
    print(f"  rmse     : {rmse:.8f}")
    print(f"  rel_rmse : {rel_rmse:.8f}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = torch.device("cuda")
    backend = load_sharq_ops()

    m = 256
    n = 5120
    k = 5120

    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    w = torch.randn((n, k), device=device, dtype=torch.bfloat16) * 0.1
    reorder_index = torch.arange(k, device=device, dtype=torch.int16)

    qx, sfx = backend.reorder_quantize_x(x, reorder_index, 0)
    qw16, sfw16 = backend.reorder_quantize_w(w, reorder_index, 0)
    qw32, _sfw_sparse32, sfw16_dup = backend.quantize_w32_shared(w)

    y_ref = torch.matmul(x.float(), w.float().t())
    y_w16 = backend.matmul(qx, qw16, sfx, sfw16, 1.0)
    y_w32_shared = backend.matmul(qx, qw32, sfx, sfw16_dup, 1.0)

    print(f"problem: M={m}, N={n}, K={k}")
    print()
    summarize("Dense NVFP4 W16 vs BF16 reference", y_w16, y_ref)
    print()
    summarize("Dense shared-payload W32 vs BF16 reference", y_w32_shared, y_ref)
    print()
    summarize("Dense shared-payload W32 vs Dense NVFP4 W16", y_w32_shared, y_w16)


if __name__ == "__main__":
    main()
