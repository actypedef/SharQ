#include "nvfp4.h"
#include "reorder.cuh"
#include "sparse_nvfp4.h"

#include "cutlass/transform/device/transform_universal_adapter.hpp"
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"

#include <vector>
#include <random>
#include <iostream>

namespace {

using bf16_t = cutlass::bfloat16_t;
using sf_t = cutlass::float_ue4m3_t;
using fp4_t = cutlass::float_e2m1_t;

constexpr int M = 128;
constexpr int N = 128;
constexpr int K = 4096;
constexpr int WarmupIters = 10;
constexpr int BenchIters = 50;

using SparseConfig = typename sparse_nvfp4::CollectiveMainloop::SparseConfig;
using DenseStrideA = cutlass::gemm::TagToStrideA_t<sparse_nvfp4::LayoutATag>;
using CompressorUtility = cutlass::transform::kernel::StructuredSparseCompressorUtility<
    cute::Shape<int, int, int, int>,
    fp4_t,
    sparse_nvfp4::LayoutATag,
    SparseConfig>;
using CompressorKernel = cutlass::transform::kernel::StructuredSparseCompressor<
    cute::Shape<int, int, int, int>,
    fp4_t,
    sparse_nvfp4::LayoutATag,
    SparseConfig,
    cutlass::arch::Sm120>;
using Compressor = cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;

template <class Layout>
size_t scale_buffer_size(Layout layout) {
  return static_cast<size_t>(cute::size(cute::filter_zeros(layout)));
}

template <class Element>
size_t storage_bytes(size_t logical_elements) {
  return (logical_elements * cutlass::sizeof_bits<Element>::value + 7) / 8;
}

void fill_sparse_input(std::vector<bf16_t>& values) {
  std::mt19937 rng(2026);
  std::uniform_real_distribution<float> dist(-6.0f, 6.0f);
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; k += 4) {
      values[m * K + k + 0] = bf16_t(dist(rng));
      values[m * K + k + 1] = bf16_t(dist(rng));
      values[m * K + k + 2] = bf16_t(0.0f);
      values[m * K + k + 3] = bf16_t(0.0f);
    }
  }
}

void fill_dense_input(std::vector<bf16_t>& values) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-6.0f, 6.0f);
  for (auto& v : values) {
    v = bf16_t(dist(rng));
  }
}

}  // namespace

int main() {
  std::vector<bf16_t> a_host(M * K);
  std::vector<bf16_t> b_host(N * K);
  fill_sparse_input(a_host);
  fill_dense_input(b_host);

  bf16_t* a_dense_d = nullptr;
  bf16_t* b_dense_d = nullptr;
  int16_t* reorder_idx_d = nullptr;

  CHECK_CUDA(cudaMalloc(&a_dense_d, sizeof(bf16_t) * a_host.size()));
  CHECK_CUDA(cudaMalloc(&b_dense_d, sizeof(bf16_t) * b_host.size()));
  CHECK_CUDA(cudaMalloc(&reorder_idx_d, sizeof(int16_t) * K));

  std::vector<int16_t> reorder_idx(K);
  for (int i = 0; i < K; ++i) {
    reorder_idx[i] = static_cast<int16_t>(i);
  }

  CHECK_CUDA(cudaMemcpy(a_dense_d, a_host.data(), sizeof(bf16_t) * a_host.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(b_dense_d, b_host.data(), sizeof(bf16_t) * b_host.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(reorder_idx_d, reorder_idx.data(), sizeof(int16_t) * K, cudaMemcpyHostToDevice));

  auto dense_layout_sfa = nvfp4::get_layoutSFA(M, K);
  auto dense_layout_sfb = nvfp4::get_layoutSFB(N, K);

  cutlass::HostTensor<fp4_t, cutlass::layout::PackedVectorLayout> a_q_dense;
  cutlass::HostTensor<fp4_t, cutlass::layout::PackedVectorLayout> a_q_compressed;
  cutlass::HostTensor<uint8_t, cutlass::layout::PackedVectorLayout> metadata_e;
  cutlass::HostTensor<sf_t, cutlass::layout::PackedVectorLayout> scale_a;
  cutlass::HostTensor<fp4_t, cutlass::layout::PackedVectorLayout> b_q_dense;
  cutlass::HostTensor<sf_t, cutlass::layout::PackedVectorLayout> scale_b;

  a_q_dense.reset(cutlass::make_Coord(M * K));
  b_q_dense.reset(cutlass::make_Coord(N * K));
  scale_a.reset(cutlass::make_Coord(scale_buffer_size(dense_layout_sfa)));
  scale_b.reset(cutlass::make_Coord(scale_buffer_size(dense_layout_sfb)));

  CHECK_CUDA(cudaMemset(reinterpret_cast<uint8_t*>(a_q_dense.device_data()), 0, storage_bytes<fp4_t>(a_q_dense.capacity())));
  CHECK_CUDA(cudaMemset(reinterpret_cast<uint8_t*>(b_q_dense.device_data()), 0, storage_bytes<fp4_t>(b_q_dense.capacity())));
  CHECK_CUDA(cudaMemset(reinterpret_cast<uint8_t*>(scale_a.device_data()), 0, storage_bytes<sf_t>(scale_a.capacity())));
  CHECK_CUDA(cudaMemset(reinterpret_cast<uint8_t*>(scale_b.device_data()), 0, storage_bytes<sf_t>(scale_b.capacity())));

  run_reorder_x_bf16_nvfp4<16, K>(
      a_dense_d,
      M,
      reorder_idx_d,
      reinterpret_cast<uint8_t*>(a_q_dense.device_data()),
      reinterpret_cast<sf_t*>(scale_a.device_data()),
      K,
      0);
  run_reorder_w_bf16_nvfp4<16, K>(
      b_dense_d,
      N,
      reorder_idx_d,
      reinterpret_cast<uint8_t*>(b_q_dense.device_data()),
      reinterpret_cast<sf_t*>(scale_b.device_data()),
      K,
      0);
  CHECK_CUDA(cudaDeviceSynchronize());

  auto workload = cute::make_shape(M, N, K, 1);
  DenseStrideA stride_a = cutlass::make_cute_packed_stride(DenseStrideA{}, cute::make_shape(M, K, 1));
  CompressorUtility compressor_utility(workload, stride_a);

  int aligned_m_e = compressor_utility.get_metadata_m_physical();
  int aligned_k_e = compressor_utility.get_metadata_k_physical();
  int aligned_m_a = compressor_utility.get_tensorA_m_physical();
  int aligned_k_a = compressor_utility.get_tensorA_k_physical();

  a_q_compressed.reset(cutlass::make_Coord(aligned_m_a * aligned_k_a));
  metadata_e.reset(cutlass::make_Coord(aligned_m_e * aligned_k_e));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  typename Compressor::Arguments compressor_args{
      {M, N, K, 1},
      {a_q_dense.device_data(),
       stride_a,
       a_q_compressed.device_data(),
       metadata_e.device_data()},
      {hw_info}};

  Compressor compressor;
  size_t compressor_workspace_size = Compressor::get_workspace_size(compressor_args);
  cutlass::device_memory::allocation<uint8_t> compressor_workspace(compressor_workspace_size);
  auto status = compressor.can_implement(compressor_args);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Sparse compressor cannot implement workload: "
              << cutlass::cutlassGetStatusString(status) << std::endl;
    return -1;
  }
  status = compressor.initialize(compressor_args, compressor_workspace.get());
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Sparse compressor initialize failed: "
              << cutlass::cutlassGetStatusString(status) << std::endl;
    return -1;
  }
  status = compressor.run();
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Sparse compressor run failed: "
              << cutlass::cutlassGetStatusString(status) << std::endl;
    return -1;
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  bf16_t* c_d = nullptr;
  bf16_t* d_d = nullptr;
  CHECK_CUDA(cudaMalloc(&c_d, sizeof(bf16_t) * M * N));
  CHECK_CUDA(cudaMalloc(&d_d, sizeof(bf16_t) * M * N));
  CHECK_CUDA(cudaMemset(c_d, 0, sizeof(bf16_t) * M * N));
  CHECK_CUDA(cudaMemset(d_d, 0, sizeof(bf16_t) * M * N));

  for (int i = 0; i < WarmupIters; ++i) {
    sparse_nvfp4::matmul_host_sparse_nvfp4_bf16(
        a_q_compressed.device_data(),
        b_q_dense.device_data(),
        metadata_e.device_data(),
        M,
        N,
        K,
        c_d,
        d_d,
        scale_a.device_data(),
        scale_b.device_data(),
        1.0f,
        0.0f);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < BenchIters; ++i) {
    sparse_nvfp4::matmul_host_sparse_nvfp4_bf16(
        a_q_compressed.device_data(),
        b_q_dense.device_data(),
        metadata_e.device_data(),
        M,
        N,
        K,
        c_d,
        d_d,
        scale_a.device_data(),
        scale_b.device_data(),
        1.0f,
        0.0f);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

  std::vector<bf16_t> out_host(M * N);
  CHECK_CUDA(cudaMemcpy(out_host.data(), d_d, sizeof(bf16_t) * out_host.size(), cudaMemcpyDeviceToHost));

  float checksum = 0.0f;
  for (int i = 0; i < 16; ++i) {
    checksum += static_cast<float>(out_host[i]);
  }

  std::cout << "Sparse NVFP4 GEMM ran successfully." << std::endl;
  std::cout << "Problem size: " << M << " x " << N << " x " << K << std::endl;
  std::cout << "Average runtime: " << (elapsed_ms / BenchIters) << " ms" << std::endl;
  std::cout << "Output checksum[0:16]: " << checksum << std::endl;

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(a_dense_d));
  CHECK_CUDA(cudaFree(b_dense_d));
  CHECK_CUDA(cudaFree(reorder_idx_d));
  CHECK_CUDA(cudaFree(c_d));
  CHECK_CUDA(cudaFree(d_d));
  return 0;
}

