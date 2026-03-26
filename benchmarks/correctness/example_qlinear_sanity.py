from __future__ import annotations

import sys
from pathlib import Path

import torch


def load_qlinear():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "model"))
    from qLinearLayer import QLinearLayer  # type: ignore

    return QLinearLayer


def run_case(name: str, x: torch.Tensor, w: torch.Tensor, qlinear_cls) -> None:
    k = w.shape[1]
    reorder_index = torch.arange(k, dtype=torch.int16)
    layer = torch.nn.Linear(k, w.shape[0], bias=False, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        layer.weight.copy_(w)

    q_old = qlinear_cls(layer, select_num=0, reorder_index=reorder_index, quant_type="NVFP4")
    q_new = qlinear_cls(layer, select_num=0, reorder_index=reorder_index, quant_type="SHARQ")

    y_old = q_old((x, x.shape[0], x.shape[1])).float()
    y_new = q_new((x, x.shape[0], x.shape[1])).float()
    diff = (y_old - y_new).abs()

    print(name)
    print(f"  old checksum [0:16]: {y_old.flatten()[:16].sum().item():.8f}")
    print(f"  new checksum [0:16]: {y_new.flatten()[:16].sum().item():.8f}")
    print(f"  max diff          : {diff.max().item():.8f}")
    print(f"  mean diff         : {diff.mean().item():.8f}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    qlinear_cls = load_qlinear()
    k = 5120

    x_zero = torch.zeros((1, 1, k), device="cuda", dtype=torch.bfloat16)
    x_one = torch.ones((1, 1, k), device="cuda", dtype=torch.bfloat16)
    eye = torch.eye(k, device="cuda", dtype=torch.bfloat16)
    ones_w = torch.ones((k, k), device="cuda", dtype=torch.bfloat16)

    run_case("zero x eye", x_zero, eye, qlinear_cls)
    print()
    run_case("ones x eye", x_one, eye, qlinear_cls)
    print()
    run_case("ones x ones", x_one, ones_w, qlinear_cls)


if __name__ == "__main__":
    main()
