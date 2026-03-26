# SharQ Kernels

This directory contains the CUDA extension and low-level kernel code used by SharQ.

## Main Components

- [`src/nvfp4.cu`](src/nvfp4.cu): dense NVFP4 GEMM path
- [`src/reorder.cu`](src/reorder.cu): dense activation/weight quantization kernels used by the dense baseline
- [`src/fused_sparse_prepare.cu`](src/fused_sparse_prepare.cu): fused SharQ activation kernel
- [`src/sparse_nvfp4.cu`](src/sparse_nvfp4.cu): CUTLASS-based sparse NVFP4 GEMM wrapper
- [`src/shared_weight_nvfp4.cu`](src/shared_weight_nvfp4.cu): shared-payload `W32` weight preparation
- [`src/bindings.cpp`](src/bindings.cpp): Python bindings exposed through `sharq_ops`

Headers live in [`include/`](include), and standalone CUDA benchmarks live in [`benchmark/`](benchmark).

## Build

```bash
cmake -S kernels -B kernels/build_cmake_sm120a \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DPython3_EXECUTABLE=$(which python)

cmake --build kernels/build_cmake_sm120a --target sharq_ops -j
```

The built extension is:

```text
kernels/build_cmake_sm120a/sharq_ops.so
```

## Notes

- The real SharQ kernel path requires Blackwell `sm_120a`.
- Python-side loaders prefer `sharq_ops.so` and still fall back to an older `agemm.so` if one exists.
- `SHARQ_SIM` does not use this extension; it is a pure PyTorch reference mode.

## Low-Level Benchmarks

Build the standalone CUDA benchmarks if needed:

```bash
cmake --build kernels/build_cmake_sm120a --target bench_nvfp4 -j
cmake --build kernels/build_cmake_sm120a --target bench_sparse_nvfp4 -j
```

Higher-level Python benchmark scripts live in:

- [`../benchmarks/correctness`](../benchmarks/correctness)
- [`../benchmarks/perf`](../benchmarks/perf)
- [`../benchmarks/ablation`](../benchmarks/ablation)
