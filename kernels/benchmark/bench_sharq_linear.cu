#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "fused_sparse_prepare.h"
#include "nvfp4.h"
#include "reorder.cuh"
#include "shared_weight_nvfp4.h"
#include "sparse_nvfp4.h"

namespace {

using bf16_t = cutlass::bfloat16_t;
using sf_t = cutlass::float_ue4m3_t;
using fp4_t = cutlass::float_e2m1_t;

struct Options {
  int M = 256;
  int N = 5120;
  int K = 5120;
  int warmup = 50;
  int iters = 200;
  int seed = 0;
};

void parse_args(int argc, char** argv, Options& opts) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--m") == 0 && i + 1 < argc) opts.M = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) opts.N = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--k") == 0 && i + 1 < argc) opts.K = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) opts.warmup = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) opts.iters = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) opts.seed = std::atoi(argv[++i]);
  }
}

size_t dense_scale_bytes(int rows, int k_dim) {
  return static_cast<size_t>((rows / 128 + 1) * 128 * k_dim / 16);
}

size_t packed_fp4_bytes(int rows, int k_dim) {
  return static_cast<size_t>(rows) * static_cast<size_t>(k_dim / 2);
}

void fill_scaled_bf16(std::vector<bf16_t>& values, int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-6.0f, 6.0f);
  for (auto& v : values) {
    v = bf16_t(dist(rng));
  }
}

#define CASE_QX_16(VAL) case VAL: run_reorder_x_bf16_nvfp4<16, VAL>(x_d, M, reorder_idx_d, qx_d, sfx_d, K, 0); break;
#define CASE_QX_32(VAL) case VAL: run_reorder32_x_bf16_nvfp4<32, VAL>(x_d, M, reorder_idx_d, qx_d, sfx_d, K, 0); break;
#define CASE_QW_16(VAL) case VAL: run_reorder_w_bf16_nvfp4<16, VAL>(w_d, N, reorder_idx_d, qw_d, sfw_d, K, 0); break;
#define CASE_QW_32(VAL) case VAL: run_reorder32_w_bf16_nvfp4<32, VAL>(w_d, N, reorder_idx_d, qw_d, sfw_d, K, 0); break;
#define CASE_DW_32(VAL) case VAL: run_down32_w_bf16_nvfp4<32, VAL>(w_d, N, qw_d, sfw_d, K, 0); break;

void launch_dense_quant_x(int M, int K, bf16_t* x_d, int16_t* reorder_idx_d, uint8_t* qx_d, sf_t* sfx_d) {
  switch (K) {
    CASE_QX_16(2048)
    CASE_QX_16(3072)
    CASE_QX_16(4096)
    CASE_QX_16(5120)
    CASE_QX_16(8192)
    CASE_QX_16(11008)
    CASE_QX_16(13824)
    CASE_QX_16(14336)
    CASE_QX_32(3584)
    CASE_QX_32(18944)
    default:
      std::cerr << "Unsupported K for dense activation quant benchmark: " << K << std::endl;
      std::exit(EXIT_FAILURE);
  }
}

void launch_dense_quant_w(int N, int K, bf16_t* w_d, int16_t* reorder_idx_d, uint8_t* qw_d, sf_t* sfw_d) {
  switch (K) {
    CASE_QW_16(2048)
    CASE_QW_16(3072)
    CASE_QW_16(4096)
    CASE_QW_16(5120)
    CASE_QW_16(8192)
    CASE_QW_16(11008)
    CASE_QW_16(13824)
    CASE_QW_16(14336)
    CASE_QW_32(3584)
    CASE_QW_32(18944)
    CASE_DW_32(27648)
    CASE_DW_32(28672)
    default:
      std::cerr << "Unsupported K for dense weight quant benchmark: " << K << std::endl;
      std::exit(EXIT_FAILURE);
  }
}

template <class Fn>
float time_cuda(Fn fn, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) fn();
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) fn();
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return ms / static_cast<float>(iters);
}

}  // namespace

int main(int argc, char** argv) {
  Options opts;
  parse_args(argc, argv, opts);

  if (opts.K % 128 != 0) {
    std::cerr << "K must be a multiple of 128 for SHARQ fused prepare, got " << opts.K << std::endl;
    return EXIT_FAILURE;
  }

  const int M = opts.M;
  const int N = opts.N;
  const int K = opts.K;

  std::vector<bf16_t> x_host(static_cast<size_t>(M) * K);
  std::vector<bf16_t> w_host(static_cast<size_t>(N) * K);
  std::vector<int16_t> reorder_idx(K);
  fill_scaled_bf16(x_host, opts.seed);
  fill_scaled_bf16(w_host, opts.seed + 1);
  for (int i = 0; i < K; ++i) reorder_idx[i] = static_cast<int16_t>(i);

  bf16_t* x_d = nullptr;
  bf16_t* w_d = nullptr;
  int16_t* reorder_idx_d = nullptr;
  CHECK_CUDA(cudaMalloc(&x_d, sizeof(bf16_t) * x_host.size()));
  CHECK_CUDA(cudaMalloc(&w_d, sizeof(bf16_t) * w_host.size()));
  CHECK_CUDA(cudaMalloc(&reorder_idx_d, sizeof(int16_t) * reorder_idx.size()));
  CHECK_CUDA(cudaMemcpy(x_d, x_host.data(), sizeof(bf16_t) * x_host.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(w_d, w_host.data(), sizeof(bf16_t) * w_host.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(reorder_idx_d, reorder_idx.data(), sizeof(int16_t) * reorder_idx.size(), cudaMemcpyHostToDevice));

  uint8_t* qx_nvfp4_d = nullptr;
  sf_t* sfx_nvfp4_d = nullptr;
  uint8_t* qw_nvfp4_d = nullptr;
  sf_t* sfw_nvfp4_d = nullptr;
  CHECK_CUDA(cudaMalloc(&qx_nvfp4_d, packed_fp4_bytes(M, K)));
  CHECK_CUDA(cudaMalloc(&sfx_nvfp4_d, dense_scale_bytes(M, K)));
  CHECK_CUDA(cudaMalloc(&qw_nvfp4_d, packed_fp4_bytes(N, K)));
  CHECK_CUDA(cudaMalloc(&sfw_nvfp4_d, dense_scale_bytes(N, K)));

  uint8_t* qw_sharq_d = nullptr;
  sf_t* sfw_sparse_d = nullptr;
  sf_t* sfw_dense_d = nullptr;
  CHECK_CUDA(cudaMalloc(&qw_sharq_d, packed_fp4_bytes(N, K)));
  CHECK_CUDA(cudaMalloc(&sfw_sparse_d, sparse_nvfp4::get_sfb_bytes(1, N, K)));
  CHECK_CUDA(cudaMalloc(&sfw_dense_d, dense_scale_bytes(N, K)));

  uint8_t* a_comp_d = nullptr;
  uint8_t* e_d = nullptr;
  sf_t* sfa_sparse_d = nullptr;
  uint8_t* q_res_d = nullptr;
  sf_t* sf_res_d = nullptr;
  CHECK_CUDA(cudaMalloc(&a_comp_d, sparse_nvfp4::get_compressed_a_bytes(M, N, K)));
  CHECK_CUDA(cudaMalloc(&e_d, sparse_nvfp4::get_metadata_e_bytes(M, N, K)));
  CHECK_CUDA(cudaMalloc(&sfa_sparse_d, sparse_nvfp4::get_sfa_bytes(M, N, K)));
  CHECK_CUDA(cudaMalloc(&q_res_d, packed_fp4_bytes(M, K)));
  CHECK_CUDA(cudaMalloc(&sf_res_d, dense_scale_bytes(M, K)));

  bf16_t* y_dense_d = nullptr;
  bf16_t* y_sparse_d = nullptr;
  bf16_t* y_out_d = nullptr;
  CHECK_CUDA(cudaMalloc(&y_dense_d, sizeof(bf16_t) * static_cast<size_t>(M) * N));
  CHECK_CUDA(cudaMalloc(&y_sparse_d, sizeof(bf16_t) * static_cast<size_t>(M) * N));
  CHECK_CUDA(cudaMalloc(&y_out_d, sizeof(bf16_t) * static_cast<size_t>(M) * N));
  CHECK_CUDA(cudaMemset(y_dense_d, 0, sizeof(bf16_t) * static_cast<size_t>(M) * N));
  CHECK_CUDA(cudaMemset(y_sparse_d, 0, sizeof(bf16_t) * static_cast<size_t>(M) * N));
  CHECK_CUDA(cudaMemset(y_out_d, 0, sizeof(bf16_t) * static_cast<size_t>(M) * N));

  launch_dense_quant_w(N, K, w_d, reorder_idx_d, qw_nvfp4_d, sfw_nvfp4_d);
  run_quantize_w32_bf16_nvfp4_shared(w_d, N, K, qw_sharq_d, sfw_sparse_d, sfw_dense_d);
  launch_dense_quant_x(M, K, x_d, reorder_idx_d, qx_nvfp4_d, sfx_nvfp4_d);
  run_fused_sparse_residual_x_bf16_nvfp4(x_d, M, N, K, a_comp_d, e_d, nullptr, sfa_sparse_d, q_res_d, sf_res_d);
  CHECK_CUDA(cudaDeviceSynchronize());

  const float alpha = 1.0f;

  float nvfp4_quant_ms = time_cuda([&] {
    launch_dense_quant_x(M, K, x_d, reorder_idx_d, qx_nvfp4_d, sfx_nvfp4_d);
  }, opts.warmup, opts.iters);

  float sharq_quant_ms = time_cuda([&] {
    run_fused_sparse_residual_x_bf16_nvfp4(x_d, M, N, K, a_comp_d, e_d, nullptr, sfa_sparse_d, q_res_d, sf_res_d);
  }, opts.warmup, opts.iters);

  float nvfp4_gemm_ms = time_cuda([&] {
    matmul_host_nvfp4_bf16(
        reinterpret_cast<fp4_t*>(qx_nvfp4_d),
        reinterpret_cast<fp4_t*>(qw_nvfp4_d),
        M, N, K,
        y_dense_d, y_dense_d,
        sfx_nvfp4_d, sfw_nvfp4_d,
        alpha, 0.0f);
  }, opts.warmup, opts.iters);

  float sharq_sparse_main_ms = time_cuda([&] {
    sparse_nvfp4::matmul_host_sparse_nvfp4_bf16(
        reinterpret_cast<fp4_t*>(a_comp_d),
        reinterpret_cast<fp4_t*>(qw_sharq_d),
        e_d,
        M, N, K,
        y_sparse_d, y_sparse_d,
        sfa_sparse_d, sfw_sparse_d,
        alpha, 0.0f);
  }, opts.warmup, opts.iters);

  float sharq_residual_ms = time_cuda([&] {
    matmul_host_nvfp4_bf16(
        reinterpret_cast<fp4_t*>(q_res_d),
        reinterpret_cast<fp4_t*>(qw_sharq_d),
        M, N, K,
        y_dense_d, y_dense_d,
        sf_res_d, sfw_dense_d,
        alpha, 0.0f);
  }, opts.warmup, opts.iters);

  float sharq_gemm_fused_ms = time_cuda([&] {
    sparse_nvfp4::matmul_host_sparse_nvfp4_bf16(
        reinterpret_cast<fp4_t*>(a_comp_d),
        reinterpret_cast<fp4_t*>(qw_sharq_d),
        e_d,
        M, N, K,
        y_sparse_d, y_sparse_d,
        sfa_sparse_d, sfw_sparse_d,
        alpha, 0.0f);
    matmul_host_nvfp4_bf16(
        reinterpret_cast<fp4_t*>(q_res_d),
        reinterpret_cast<fp4_t*>(qw_sharq_d),
        M, N, K,
        y_sparse_d, y_out_d,
        sf_res_d, sfw_dense_d,
        alpha, 1.0f);
  }, opts.warmup, opts.iters);

  float nvfp4_whole_ms = time_cuda([&] {
    launch_dense_quant_x(M, K, x_d, reorder_idx_d, qx_nvfp4_d, sfx_nvfp4_d);
    matmul_host_nvfp4_bf16(
        reinterpret_cast<fp4_t*>(qx_nvfp4_d),
        reinterpret_cast<fp4_t*>(qw_nvfp4_d),
        M, N, K,
        y_dense_d, y_dense_d,
        sfx_nvfp4_d, sfw_nvfp4_d,
        alpha, 0.0f);
  }, opts.warmup, opts.iters);

  float sharq_whole_ms = time_cuda([&] {
    run_fused_sparse_residual_x_bf16_nvfp4(x_d, M, N, K, a_comp_d, e_d, nullptr, sfa_sparse_d, q_res_d, sf_res_d);
    sparse_nvfp4::matmul_host_sparse_nvfp4_bf16(
        reinterpret_cast<fp4_t*>(a_comp_d),
        reinterpret_cast<fp4_t*>(qw_sharq_d),
        e_d,
        M, N, K,
        y_sparse_d, y_sparse_d,
        sfa_sparse_d, sfw_sparse_d,
        alpha, 0.0f);
    matmul_host_nvfp4_bf16(
        reinterpret_cast<fp4_t*>(q_res_d),
        reinterpret_cast<fp4_t*>(qw_sharq_d),
        M, N, K,
        y_sparse_d, y_out_d,
        sf_res_d, sfw_dense_d,
        alpha, 1.0f);
  }, opts.warmup, opts.iters);

  std::cout << "problem: M=" << M << ", N=" << N << ", K=" << K << ", seed=" << opts.seed << std::endl;
  std::cout << std::endl;
  std::cout << "Quantize only" << std::endl;
  std::cout << "  NVFP4                     : " << nvfp4_quant_ms << " ms" << std::endl;
  std::cout << "  SHARQ                     : " << sharq_quant_ms << " ms" << std::endl;
  std::cout << "  SHARQ/NVFP4 ratio         : " << sharq_quant_ms / std::max(nvfp4_quant_ms, 1e-12f) << std::endl;
  std::cout << std::endl;
  std::cout << "GEMM only" << std::endl;
  std::cout << "  NVFP4 dense               : " << nvfp4_gemm_ms << " ms" << std::endl;
  std::cout << "  SHARQ sparse main         : " << sharq_sparse_main_ms << " ms" << std::endl;
  std::cout << "  SHARQ residual            : " << sharq_residual_ms << " ms" << std::endl;
  std::cout << "  SHARQ fused-accum         : " << sharq_gemm_fused_ms << " ms" << std::endl;
  std::cout << "  sparse/dense ratio        : " << sharq_sparse_main_ms / std::max(nvfp4_gemm_ms, 1e-12f) << std::endl;
  std::cout << "  residual/dense ratio      : " << sharq_residual_ms / std::max(nvfp4_gemm_ms, 1e-12f) << std::endl;
  std::cout << "  fused/dense ratio         : " << sharq_gemm_fused_ms / std::max(nvfp4_gemm_ms, 1e-12f) << std::endl;
  std::cout << std::endl;
  std::cout << "Whole linear" << std::endl;
  std::cout << "  NVFP4                     : " << nvfp4_whole_ms << " ms" << std::endl;
  std::cout << "  SHARQ                     : " << sharq_whole_ms << " ms" << std::endl;
  std::cout << "  SHARQ/NVFP4 ratio         : " << sharq_whole_ms / std::max(nvfp4_whole_ms, 1e-12f) << std::endl;

  cudaFree(x_d);
  cudaFree(w_d);
  cudaFree(reorder_idx_d);
  cudaFree(qx_nvfp4_d);
  cudaFree(sfx_nvfp4_d);
  cudaFree(qw_nvfp4_d);
  cudaFree(sfw_nvfp4_d);
  cudaFree(qw_sharq_d);
  cudaFree(sfw_sparse_d);
  cudaFree(sfw_dense_d);
  cudaFree(a_comp_d);
  cudaFree(e_d);
  cudaFree(sfa_sparse_d);
  cudaFree(q_res_d);
  cudaFree(sf_res_d);
  cudaFree(y_dense_d);
  cudaFree(y_sparse_d);
  cudaFree(y_out_d);
  return 0;
}
