# SharQ Server Build Notes

These commands assume:

- CUDA is installed at `/usr/local/cuda`
- you are building from the repository root
- the active Python environment already contains PyTorch and the packages from `requirements.txt`

## Build `sharq_ops`

```bash
conda install pybind11 -y

cmake -S kernels -B kernels/build_cmake_sm120a \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DPython3_EXECUTABLE=$(which python)

cmake --build kernels/build_cmake_sm120a --target sharq_ops -j
```

The extension is generated at:

```text
kernels/build_cmake_sm120a/sharq_ops.so
```

## Recommended Smoke Tests

```bash
python benchmarks/correctness/example_sparse_fp4_chunk_diagnose.py
python benchmarks/correctness/example_kernel_vs_pseudo_current.py
python benchmarks/correctness/example_fused_pairtop2_matrix_suite.py
python benchmarks/perf/benchmark_fused_linear_and_kernel.py
```

## Recommended Model-Side Check

```bash
python model/main.py /path/to/model \
  --dataset wikitext2 \
  --eval_ppl \
  --quant_type SHARQ
```

## Notes

- Python-side loaders prefer the repo-local `kernels/build_cmake_sm120a/sharq_ops.so`.
- `SHARQ` is the fused sparse-residual kernel path used by this repo.
- `SHARQ_SIM` is the pure PyTorch simulation path and can be used as an accuracy-only reference without building the CUDA extension.
