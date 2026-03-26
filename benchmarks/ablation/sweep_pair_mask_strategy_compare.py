from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean, pstdev

import torch


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import example_pair_mask_strategy_compare as base  # type: ignore


def make_pair_friendly_x(m: int, k: int, device: torch.device) -> torch.Tensor:
    groups = torch.zeros((m, k // 8, 4, 2), device=device, dtype=torch.float32)
    groups[..., 0, 0] = 9.0
    groups[..., 0, 1] = 7.5
    groups[..., 2, 0] = -8.0
    groups[..., 2, 1] = -6.5
    groups += 0.02 * torch.randn_like(groups)
    return groups.view(m, k).to(torch.bfloat16)


def make_alternating_pair_x(m: int, k: int, device: torch.device) -> torch.Tensor:
    groups = torch.zeros((m, k // 8, 4, 2), device=device, dtype=torch.float32)
    groups[..., 1, 0] = 6.0
    groups[..., 1, 1] = -5.5
    groups[..., 3, 0] = -7.0
    groups[..., 3, 1] = 6.5
    groups += 0.01 * torch.randn_like(groups)
    return groups.view(m, k).to(torch.bfloat16)


def make_channel_outlier_x(m: int, k: int, device: torch.device) -> torch.Tensor:
    x = torch.randn((m, k), device=device, dtype=torch.float32) * 0.08
    num_channels = max(1, int(k * 0.05))
    idx = torch.randperm(k, device=device)[:num_channels]
    scales = torch.empty((num_channels,), device=device, dtype=torch.float32).uniform_(10.0, 100.0)
    x[:, idx] *= scales
    return x.to(torch.bfloat16)


def make_onebig_pair_x(m: int, k: int, device: torch.device) -> torch.Tensor:
    # Stress case: many pairs have one huge element and one small element.
    groups = torch.randn((m, k // 8, 4, 2), device=device, dtype=torch.float32) * 0.02
    groups[..., 0, 0] = 14.0
    groups[..., 0, 1] = 0.15
    groups[..., 2, 0] = -12.0
    groups[..., 2, 1] = -0.10
    groups[..., 1, 0] = 4.0
    groups[..., 1, 1] = 3.5
    return groups.view(m, k).to(torch.bfloat16)


CASE_FACTORIES = {
    "random": lambda m, k, device: torch.randn((m, k), device=device, dtype=torch.bfloat16),
    "outlier_rich": base.make_outlier_rich_x,
    "pair_friendly": make_pair_friendly_x,
    "alternating_pair": make_alternating_pair_x,
    "channel_outlier": make_channel_outlier_x,
    "onebig_pair": make_onebig_pair_x,
}


def evaluate_all_strategies(x: torch.Tensor, w: torch.Tensor):
    y_ref = torch.matmul(x.float(), w.float().t())
    scale_x = base.global_nvfp4_scale(x)
    scale_w = base.global_nvfp4_scale(w)
    x_scaled = (x / scale_x).to(torch.bfloat16)
    w_scaled = (w / scale_w).to(torch.bfloat16)
    output_scale = float(scale_x * scale_w)
    w32 = base.quantize_nvfp4_tensor(w_scaled.float(), group_size=32)
    w16 = base.quantize_nvfp4_tensor(w_scaled.float(), group_size=16)

    dense16 = (torch.nn.functional.linear(base.quantize_nvfp4_tensor(x_scaled.float(), group_size=16), w16) * output_scale)
    dense16_mse, dense16_rel = base.metrics(dense16, y_ref)

    strategies = {
        "scalar_top2:4": base.top2_4_scalar,
        "pair_maxabs": base.pair_top2_maxabs,
        "pair_l1": base.pair_top2_l1,
        "pair_l2sq": base.pair_top2_l2sq,
        "pair_max_plus_qmin": base.pair_top2_hybrid,
        "pair_project_scalar_top2:4": base.pair_project_from_scalar_top2_4,
        "pair_project_global_top4": base.pair_top2_from_global_top4,
    }

    results = {"dense_nvfp4_w16": {"rel_rmse": dense16_rel, "mse": dense16_mse, "viol": None}}
    for name, fn in strategies.items():
        results[name] = base.evaluate_strategy(name, x_scaled, w32, y_ref, output_scale, fn)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep pair mask strategies across multiple seeds and synthetic cases.")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=5120)
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["outlier_rich", "random", "pair_friendly", "alternating_pair", "channel_outlier", "onebig_pair"],
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.k % 128 != 0:
        raise ValueError(f"k must be a multiple of 128, got {args.k}")

    device = torch.device("cuda")

    aggregate: dict[str, list[float]] = {}
    wins: dict[str, int] = {}
    strict_wins: dict[str, int] = {}
    total_runs = 0

    for case in args.cases:
        if case not in CASE_FACTORIES:
            raise ValueError(f"Unknown case: {case}")
        print(f"case={case}")
        print(f"{'seed':>4s} {'best_pair':24s} {'best_rel':>12s} {'pair_l2sq':>12s} {'pair_l1':>12s} {'pair_maxabs':>12s}")
        for seed in args.seeds:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            x = CASE_FACTORIES[case](args.m, args.k, device)
            w = torch.randn((args.n, args.k), device=device, dtype=torch.bfloat16)
            results = evaluate_all_strategies(x, w)

            pair_only = {k: v for k, v in results.items() if k.startswith("pair_")}
            best_name, best_item = min(pair_only.items(), key=lambda kv: kv[1]["rel_rmse"])
            tol = 1e-7
            tied = [
                name for name, item in pair_only.items()
                if abs(item["rel_rmse"] - best_item["rel_rmse"]) <= tol
            ]
            print(
                f"{seed:4d} {best_name:24s} {best_item['rel_rmse']:12.8f} "
                f"{results['pair_l2sq']['rel_rmse']:12.8f} {results['pair_l1']['rel_rmse']:12.8f} "
                f"{results['pair_maxabs']['rel_rmse']:12.8f}"
            )

            for name, item in results.items():
                aggregate.setdefault(name, []).append(item["rel_rmse"])
            for name in tied:
                wins[name] = wins.get(name, 0) + 1
            if len(tied) == 1:
                strict_wins[best_name] = strict_wins.get(best_name, 0) + 1
            total_runs += 1
        print()

    print("aggregate")
    print(f"{'strategy':30s} {'mean_rel':>12s} {'std_rel':>12s} {'wins':>6s} {'strict':>6s}")
    for name, vals in sorted(aggregate.items(), key=lambda kv: mean(kv[1])):
        print(
            f"{name:30s} {mean(vals):12.8f} {pstdev(vals):12.8f} "
            f"{wins.get(name, 0):6d} {strict_wins.get(name, 0):6d}"
        )
    print(f"total_pair_runs: {total_runs}")


if __name__ == "__main__":
    main()
