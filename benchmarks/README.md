# SharQ Benchmark Scripts

The benchmark directory is grouped by purpose.

## correctness/

Low-level correctness and consistency checks for SharQ kernels and model-side paths.

- `example_qlinear_sanity.py`
- `example_sparse_gemm_sanity.py`
- `example_sparse_fp4_chunk_diagnose.py`
- `example_kernel_vs_pseudo_current.py`
- `example_fused_pairtop2_matrix_suite.py`
- `example_fused_sparse_prepare.py`
- `example_shared_w32_sanity.py`

## perf/

Latency and throughput measurements.

- `benchmark_fused_linear_and_kernel.py`
- `example_sparse_nvfp4.py`
- `sweep_sparse_nvfp4.py`

## ablation/

Error analysis and mask-strategy comparison.

- `example_fused_linear_mse.py`
- `example_fused_linear_shared_payload.py`
- `example_shared_w32_dense_mse.py`
- `example_pair_mask_strategy_compare.py`
- `example_pair_top2_pseudo_current.py`
- `sweep_pair_mask_strategy_compare.py`
