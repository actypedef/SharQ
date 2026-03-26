#include <torch/extension.h>

#include <iostream>

#include "cutlass/transform/device/transform_universal_adapter.hpp"
#include "fused_sparse_prepare.h"
#include "nvfp4.h"
#include "reorder.cuh"
#include "shared_weight_nvfp4.h"
#include "sparse_nvfp4.h"

/**************************** Dense NVFP4 Quantization Kernels ****************************/

#define CASE_REORDER_X_16(VAL)                                                                          \
  case VAL:                                                                                             \
    run_reorder_x_bf16_nvfp4<16, VAL>(ptr_X, M, ptr_idx, ptr_QX, ptr_SFX, KQ, KE);                     \
    break;

#define CASE_REORDER_X_32(VAL)                                                                          \
  case VAL:                                                                                             \
    run_reorder32_x_bf16_nvfp4<32, VAL>(ptr_X, M, ptr_idx, ptr_QX, ptr_SFX, KQ, KE);                   \
    break;

#define CASE_DOWN_X_32(VAL)                                                                             \
  case VAL: {                                                                                           \
    auto tmp_X = torch::index_select(X, 1, reorder_index.to(torch::kInt32));                           \
    run_down32_x_bf16_nvfp4<32, VAL>(                                                                   \
        reinterpret_cast<cutlass::bfloat16_t *>(tmp_X.data_ptr<at::BFloat16>()), M, ptr_QX, ptr_SFX,   \
        KQ, KE);                                                                                        \
  } break;

#define CASE_REORDER_W_16(VAL)                                                                          \
  case VAL:                                                                                             \
    run_reorder_w_bf16_nvfp4<16, VAL>(ptr_W, N, ptr_idx, ptr_QW, ptr_SFW, KQ, KE);                     \
    break;

#define CASE_REORDER_W_32(VAL)                                                                          \
  case VAL:                                                                                             \
    run_reorder32_w_bf16_nvfp4<32, VAL>(ptr_W, N, ptr_idx, ptr_QW, ptr_SFW, KQ, KE);                   \
    break;

#define CASE_DOWN_W_32(VAL)                                                                             \
  case VAL: {                                                                                           \
    auto tmp_W = torch::index_select(W, 1, reorder_index.to(torch::kInt32));                           \
    run_down32_w_bf16_nvfp4<32, VAL>(                                                                   \
        reinterpret_cast<cutlass::bfloat16_t *>(tmp_W.data_ptr<at::BFloat16>()), N, ptr_QW, ptr_SFW,   \
        KQ, KE);                                                                                        \
  } break;

inline size_t get_sfa_buffer_size_in_bytes(int num_rows, int k_dim) {
  auto layout = filter_zeros(nvfp4::get_layoutSFA(num_rows, k_dim));
  (void)layout;
  return (num_rows / 128 + 1) * 128 * k_dim / 16;
}

inline size_t get_sfb_buffer_size_in_bytes(int num_rows, int k_dim) {
  auto layout = filter_zeros(nvfp4::get_layoutSFB(num_rows, k_dim));
  (void)layout;
  return (num_rows / 128 + 1) * 128 * k_dim / 16;
}

std::tuple<torch::Tensor, torch::Tensor> reorder_quantize_x(
    const torch::Tensor &X,
    const torch::Tensor &reorder_index,
    const int KE) {
  int M = static_cast<int>(X.size(0));
  int KQ = static_cast<int>(X.size(1));
  int K = KQ + KE;

  auto options = torch::dtype(torch::kUInt8).device(X.device());
  auto QX = torch::empty({M, K / 2}, options);
  auto SFX = torch::empty({static_cast<int64_t>(get_sfa_buffer_size_in_bytes(M, K))}, options);

  auto ptr_X = reinterpret_cast<cutlass::bfloat16_t *>(X.data_ptr<at::BFloat16>());
  auto ptr_idx = reorder_index.data_ptr<int16_t>();
  auto ptr_QX = QX.data_ptr<uint8_t>();
  auto ptr_SFX = reinterpret_cast<cutlass::float_ue4m3_t *>(SFX.data_ptr<uint8_t>());

  switch (KQ) {
    CASE_REORDER_X_16(2048)
    CASE_REORDER_X_16(3072)
    CASE_REORDER_X_16(4096)
    CASE_REORDER_X_16(5120)
    CASE_REORDER_X_16(8192)
    CASE_REORDER_X_16(11008)
    CASE_REORDER_X_16(13824)
    CASE_REORDER_X_16(14336)
    CASE_REORDER_X_32(3584)
    CASE_REORDER_X_32(18944)
    CASE_DOWN_X_32(27648)
    CASE_DOWN_X_32(28672)
    default:
      std::cerr << "Unsupported activation hidden size: " << KQ << std::endl;
      throw std::runtime_error("Unsupported hidden size in reorder_quantize_x");
  }

  return std::make_tuple(QX, SFX);
}

std::tuple<torch::Tensor, torch::Tensor> reorder_quantize_w(
    const torch::Tensor &W,
    const torch::Tensor &reorder_index,
    const int KE) {
  int N = static_cast<int>(W.size(0));
  int KQ = static_cast<int>(W.size(1));
  int K = KQ + KE;

  auto options = torch::dtype(torch::kUInt8).device(W.device());
  auto QW = torch::empty({N, K / 2}, options);
  auto SFW = torch::empty({static_cast<int64_t>(get_sfb_buffer_size_in_bytes(N, K))}, options);

  auto ptr_W = reinterpret_cast<cutlass::bfloat16_t *>(W.data_ptr<at::BFloat16>());
  auto ptr_idx = reorder_index.data_ptr<int16_t>();
  auto ptr_QW = QW.data_ptr<uint8_t>();
  auto ptr_SFW = reinterpret_cast<cutlass::float_ue4m3_t *>(SFW.data_ptr<uint8_t>());

  switch (KQ) {
    CASE_REORDER_W_16(2048)
    CASE_REORDER_W_16(3072)
    CASE_REORDER_W_16(4096)
    CASE_REORDER_W_16(5120)
    CASE_REORDER_W_16(8192)
    CASE_REORDER_W_16(11008)
    CASE_REORDER_W_16(13824)
    CASE_REORDER_W_16(14336)
    CASE_REORDER_W_32(3584)
    CASE_REORDER_W_32(18944)
    CASE_DOWN_W_32(27648)
    CASE_DOWN_W_32(28672)
    default:
      std::cerr << "Unsupported weight hidden size: " << KQ << std::endl;
      throw std::runtime_error("Unsupported hidden size in reorder_quantize_w");
  }

  return std::make_tuple(QW, SFW);
}

std::tuple<torch::Tensor, torch::Tensor> quantize_x_nvfp4(const torch::Tensor &X) {
  auto reorder_index =
      torch::arange(X.size(1), torch::dtype(torch::kInt16).device(X.device()));
  return reorder_quantize_x(X, reorder_index, 0);
}

std::tuple<torch::Tensor, torch::Tensor> quantize_w_nvfp4(const torch::Tensor &W) {
  auto reorder_index =
      torch::arange(W.size(1), torch::dtype(torch::kInt16).device(W.device()));
  return reorder_quantize_w(W, reorder_index, 0);
}

/**************************** Sparse NVFP4 Utilities ****************************/

std::tuple<torch::Tensor, torch::Tensor> compress_sparse_a(
    const torch::Tensor &A_q_dense,
    const int N) {
  TORCH_CHECK(A_q_dense.is_cuda(), "A_q_dense must be a CUDA tensor");
  TORCH_CHECK(A_q_dense.is_contiguous(), "A_q_dense must be contiguous");
  TORCH_CHECK(A_q_dense.scalar_type() == at::ScalarType::Byte, "A_q_dense must be torch.uint8");
  TORCH_CHECK(A_q_dense.dim() == 2, "A_q_dense must be shaped [M, K/2]");

  using CompressorKernel = cutlass::transform::kernel::StructuredSparseCompressor<
      sparse_nvfp4::ProblemShape,
      cutlass::float_e2m1_t,
      sparse_nvfp4::LayoutATag,
      sparse_nvfp4::SparseConfig,
      cutlass::arch::Sm120>;
  using Compressor = cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;

  const int M = static_cast<int>(A_q_dense.size(0));
  const int K = static_cast<int>(A_q_dense.size(1)) * 2;

  auto expected_dense_bytes = static_cast<int64_t>(
      sparse_nvfp4::packed_storage_bytes<cutlass::float_e2m1_t>(
          static_cast<size_t>(M) * static_cast<size_t>(K)));
  TORCH_CHECK(A_q_dense.numel() == expected_dense_bytes,
              "A_q_dense.numel() mismatch: expected ", expected_dense_bytes,
              ", got ", A_q_dense.numel());

  auto options = torch::dtype(torch::kUInt8).device(A_q_dense.device());
  auto A_comp = torch::empty({static_cast<int64_t>(sparse_nvfp4::get_compressed_a_bytes(M, N, K))}, options);
  auto E = torch::empty({static_cast<int64_t>(sparse_nvfp4::get_metadata_e_bytes(M, N, K))}, options);

  sparse_nvfp4::DenseStrideA stride_a =
      cutlass::make_cute_packed_stride(sparse_nvfp4::DenseStrideA{}, cute::make_shape(M, K, 1));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = A_q_dense.get_device();
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename Compressor::Arguments compressor_args{
      {M, N, K, 1},
      {reinterpret_cast<cutlass::float_e2m1_t *>(A_q_dense.data_ptr<uint8_t>()),
       stride_a,
       reinterpret_cast<cutlass::float_e2m1_t *>(A_comp.data_ptr<uint8_t>()),
       E.data_ptr<uint8_t>()},
      {hw_info}};

  Compressor compressor;
  auto status = compressor.can_implement(compressor_args);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "compress_sparse_a can_implement failed: ",
              cutlass::cutlassGetStatusString(status));

  size_t workspace_size = Compressor::get_workspace_size(compressor_args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  status = compressor.initialize(compressor_args, workspace.get());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "compress_sparse_a initialize failed: ",
              cutlass::cutlassGetStatusString(status));

  status = compressor.run();
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "compress_sparse_a run failed: ",
              cutlass::cutlassGetStatusString(status));

  return std::make_tuple(A_comp, E);
}

std::tuple<int64_t, int64_t, int64_t, int64_t> get_sparse_nvfp4_buffer_sizes(
    const int M,
    const int N,
    const int K) {
  return std::make_tuple(
      static_cast<int64_t>(sparse_nvfp4::get_compressed_a_bytes(M, N, K)),
      static_cast<int64_t>(sparse_nvfp4::get_metadata_e_bytes(M, N, K)),
      static_cast<int64_t>(sparse_nvfp4::get_sfa_bytes(M, N, K)),
      static_cast<int64_t>(sparse_nvfp4::get_sfb_bytes(M, N, K)));
}

std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> get_fused_sparse_residual_buffer_sizes(
    const int M,
    const int N,
    const int K) {
  return std::make_tuple(
      static_cast<int64_t>(sparse_nvfp4::get_compressed_a_bytes(M, N, K)),
      static_cast<int64_t>(sparse_nvfp4::get_metadata_e_bytes(M, N, K)),
      static_cast<int64_t>(get_sfa_buffer_size_in_bytes(M, K)),
      static_cast<int64_t>(M) * static_cast<int64_t>(K / 2),
      static_cast<int64_t>(get_sfa_buffer_size_in_bytes(M, K)));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_sparse_residual_quantize_x(const torch::Tensor &X, const int N) {
  TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
  TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
  TORCH_CHECK(X.scalar_type() == at::ScalarType::BFloat16, "X must be torch.bfloat16");
  TORCH_CHECK(X.dim() == 2, "X must be shaped [M, K]");

  const int M = static_cast<int>(X.size(0));
  const int K = static_cast<int>(X.size(1));

  TORCH_CHECK(K % 128 == 0,
              "fused_sparse_residual_quantize_x currently requires K to be a multiple of 128, got ",
              K);

  auto options = torch::dtype(torch::kUInt8).device(X.device());
  auto A_comp = torch::zeros({static_cast<int64_t>(sparse_nvfp4::get_compressed_a_bytes(M, N, K))}, options);
  auto E = torch::zeros({static_cast<int64_t>(sparse_nvfp4::get_metadata_e_bytes(M, N, K))}, options);
  auto SFA_sparse = torch::zeros({static_cast<int64_t>(sparse_nvfp4::get_sfa_bytes(M, N, K))}, options);
  auto Q_res = torch::zeros({static_cast<int64_t>(M), static_cast<int64_t>(K / 2)}, options);
  auto SF_res = torch::zeros({static_cast<int64_t>(get_sfa_buffer_size_in_bytes(M, K))}, options);

  run_fused_sparse_residual_x_bf16_nvfp4(
      reinterpret_cast<cutlass::bfloat16_t *>(X.data_ptr<at::BFloat16>()),
      M,
      N,
      K,
      A_comp.data_ptr<uint8_t>(),
      E.data_ptr<uint8_t>(),
      nullptr,
      reinterpret_cast<cutlass::float_ue4m3_t *>(SFA_sparse.data_ptr<uint8_t>()),
      Q_res.data_ptr<uint8_t>(),
      reinterpret_cast<cutlass::float_ue4m3_t *>(SF_res.data_ptr<uint8_t>()));

  return std::make_tuple(A_comp, E, SFA_sparse, Q_res, SF_res);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_sparse_residual_quantize_x_debug(const torch::Tensor &X, const int N) {
  TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
  TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
  TORCH_CHECK(X.scalar_type() == at::ScalarType::BFloat16, "X must be torch.bfloat16");
  TORCH_CHECK(X.dim() == 2, "X must be shaped [M, K]");

  const int M = static_cast<int>(X.size(0));
  const int K = static_cast<int>(X.size(1));

  TORCH_CHECK(K % 128 == 0,
              "fused_sparse_residual_quantize_x_debug currently requires K to be a multiple of 128, got ",
              K);

  auto options = torch::dtype(torch::kUInt8).device(X.device());
  auto Q_sparse = torch::zeros({static_cast<int64_t>(M), static_cast<int64_t>(K / 2)}, options);
  auto SFA_sparse = torch::zeros({static_cast<int64_t>(sparse_nvfp4::get_sfa_bytes(M, N, K))}, options);
  auto Q_res = torch::zeros({static_cast<int64_t>(M), static_cast<int64_t>(K / 2)}, options);
  auto SF_res = torch::zeros({static_cast<int64_t>(get_sfa_buffer_size_in_bytes(M, K))}, options);

  run_fused_sparse_residual_x_bf16_nvfp4(
      reinterpret_cast<cutlass::bfloat16_t *>(X.data_ptr<at::BFloat16>()),
      M,
      N,
      K,
      nullptr,
      nullptr,
      Q_sparse.data_ptr<uint8_t>(),
      reinterpret_cast<cutlass::float_ue4m3_t *>(SFA_sparse.data_ptr<uint8_t>()),
      Q_res.data_ptr<uint8_t>(),
      reinterpret_cast<cutlass::float_ue4m3_t *>(SF_res.data_ptr<uint8_t>()));

  return std::make_tuple(Q_sparse, SFA_sparse, Q_res, SF_res);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quantize_w32_shared(const torch::Tensor &W) {
  TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");
  TORCH_CHECK(W.is_contiguous(), "W must be contiguous");
  TORCH_CHECK(W.scalar_type() == at::ScalarType::BFloat16, "W must be torch.bfloat16");
  TORCH_CHECK(W.dim() == 2, "W must be shaped [N, K]");

  int N = static_cast<int>(W.size(0));
  int K = static_cast<int>(W.size(1));

  TORCH_CHECK(K % 32 == 0, "quantize_w32_shared requires K to be a multiple of 32, got ", K);

  auto options = torch::dtype(torch::kUInt8).device(W.device());
  auto QW = torch::empty({static_cast<int64_t>(N), static_cast<int64_t>(K / 2)}, options);
  auto SFW_sparse = torch::zeros({static_cast<int64_t>(sparse_nvfp4::get_sfb_bytes(1, N, K))}, options);
  auto SFW_dense = torch::zeros({static_cast<int64_t>(get_sfb_buffer_size_in_bytes(N, K))}, options);

  run_quantize_w32_bf16_nvfp4_shared(
      reinterpret_cast<cutlass::bfloat16_t *>(W.data_ptr<at::BFloat16>()),
      N,
      K,
      QW.data_ptr<uint8_t>(),
      reinterpret_cast<cutlass::float_ue4m3_t *>(SFW_sparse.data_ptr<uint8_t>()),
      reinterpret_cast<cutlass::float_ue4m3_t *>(SFW_dense.data_ptr<uint8_t>()));

  return std::make_tuple(QW, SFW_sparse, SFW_dense);
}

torch::Tensor duplicate_sfa32_to_sfa16(
    const torch::Tensor &SFA_sparse,
    const int M,
    const int N,
    const int K) {
  TORCH_CHECK(SFA_sparse.is_cuda(), "SFA_sparse must be a CUDA tensor");
  TORCH_CHECK(SFA_sparse.is_contiguous(), "SFA_sparse must be contiguous");
  TORCH_CHECK(SFA_sparse.scalar_type() == at::ScalarType::Byte, "SFA_sparse must be torch.uint8");

  auto options = torch::dtype(torch::kUInt8).device(SFA_sparse.device());
  auto SFA_dense = torch::zeros({static_cast<int64_t>(get_sfa_buffer_size_in_bytes(M, K))}, options);
  run_duplicate_sfa32_to_sfa16(
      reinterpret_cast<cutlass::float_ue4m3_t *>(SFA_sparse.data_ptr<uint8_t>()),
      M,
      N,
      K,
      reinterpret_cast<cutlass::float_ue4m3_t *>(SFA_dense.data_ptr<uint8_t>()));
  return SFA_dense;
}

/**************************** GEMM Entry Points ****************************/

torch::Tensor matmul(
    const torch::Tensor &A,
    const torch::Tensor &B,
    const torch::Tensor &SFA,
    const torch::Tensor &SFB,
    const float scale) {
  uint32_t M = static_cast<uint32_t>(A.size(0));
  uint32_t N = static_cast<uint32_t>(B.size(0));
  uint32_t K = static_cast<uint32_t>(A.size(1) * 2);
  auto C = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A.device()));

  matmul_host_nvfp4_bf16(
      reinterpret_cast<cutlass::float_e2m1_t *>(A.data_ptr<uint8_t>()),
      reinterpret_cast<cutlass::float_e2m1_t *>(B.data_ptr<uint8_t>()),
      M,
      N,
      K,
      reinterpret_cast<cutlass::bfloat16_t *>(C.data_ptr<at::BFloat16>()),
      reinterpret_cast<cutlass::bfloat16_t *>(C.data_ptr<at::BFloat16>()),
      reinterpret_cast<cutlass::float_ue4m3_t *>(SFA.data_ptr<uint8_t>()),
      reinterpret_cast<cutlass::float_ue4m3_t *>(SFB.data_ptr<uint8_t>()),
      scale,
      0.0f);
  return C;
}

torch::Tensor matmul_accum(
    const torch::Tensor &A,
    const torch::Tensor &B,
    const torch::Tensor &SFA,
    const torch::Tensor &SFB,
    const float scale,
    const torch::Tensor &C_in,
    const float beta) {
  TORCH_CHECK(C_in.is_cuda(), "C_in must be a CUDA tensor");
  TORCH_CHECK(C_in.is_contiguous(), "C_in must be contiguous");
  TORCH_CHECK(C_in.scalar_type() == at::ScalarType::BFloat16, "C_in must be torch.bfloat16");

  uint32_t M = static_cast<uint32_t>(A.size(0));
  uint32_t N = static_cast<uint32_t>(B.size(0));
  uint32_t K = static_cast<uint32_t>(A.size(1) * 2);
  TORCH_CHECK(C_in.dim() == 2, "C_in must be shaped [M, N]");
  TORCH_CHECK(C_in.size(0) == M && C_in.size(1) == N,
              "C_in shape mismatch: expected [", M, ", ", N, "] got [", C_in.size(0), ", ", C_in.size(1), "]");
  TORCH_CHECK(C_in.device() == A.device(), "C_in must be on the same device as A");

  auto D = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A.device()));

  matmul_host_nvfp4_bf16(
      reinterpret_cast<cutlass::float_e2m1_t *>(A.data_ptr<uint8_t>()),
      reinterpret_cast<cutlass::float_e2m1_t *>(B.data_ptr<uint8_t>()),
      M,
      N,
      K,
      reinterpret_cast<cutlass::bfloat16_t *>(const_cast<at::BFloat16 *>(C_in.data_ptr<at::BFloat16>())),
      reinterpret_cast<cutlass::bfloat16_t *>(D.data_ptr<at::BFloat16>()),
      reinterpret_cast<cutlass::float_ue4m3_t *>(SFA.data_ptr<uint8_t>()),
      reinterpret_cast<cutlass::float_ue4m3_t *>(SFB.data_ptr<uint8_t>()),
      scale,
      beta);
  return D;
}

torch::Tensor sparse_matmul(
    const torch::Tensor &A_comp,
    const torch::Tensor &B,
    const torch::Tensor &E,
    const torch::Tensor &SFA,
    const torch::Tensor &SFB,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta) {
  TORCH_CHECK(M > 0 && N > 0 && K > 0, "M, N, K must be positive");
  TORCH_CHECK(beta == 0.0f, "sparse_matmul currently requires beta == 0");
  TORCH_CHECK(A_comp.is_cuda() && B.is_cuda() && E.is_cuda() && SFA.is_cuda() && SFB.is_cuda(),
              "sparse_matmul inputs must be CUDA tensors");
  TORCH_CHECK(A_comp.is_contiguous() && B.is_contiguous() && E.is_contiguous() &&
                  SFA.is_contiguous() && SFB.is_contiguous(),
              "sparse_matmul inputs must be contiguous");
  TORCH_CHECK(A_comp.scalar_type() == at::ScalarType::Byte, "A_comp must be torch.uint8");
  TORCH_CHECK(B.scalar_type() == at::ScalarType::Byte, "B must be torch.uint8");
  TORCH_CHECK(E.scalar_type() == at::ScalarType::Byte, "E must be torch.uint8");
  TORCH_CHECK(SFA.scalar_type() == at::ScalarType::Byte, "SFA must be torch.uint8");
  TORCH_CHECK(SFB.scalar_type() == at::ScalarType::Byte, "SFB must be torch.uint8");
  TORCH_CHECK(A_comp.device() == B.device() && A_comp.device() == E.device() &&
                  A_comp.device() == SFA.device() && A_comp.device() == SFB.device(),
              "sparse_matmul inputs must be on the same device");

  auto expected_a_bytes = static_cast<int64_t>(sparse_nvfp4::get_compressed_a_bytes(M, N, K));
  auto expected_e_bytes = static_cast<int64_t>(sparse_nvfp4::get_metadata_e_bytes(M, N, K));
  auto expected_sfa_bytes = static_cast<int64_t>(sparse_nvfp4::get_sfa_bytes(M, N, K));
  auto expected_sfb_bytes = static_cast<int64_t>(sparse_nvfp4::get_sfb_bytes(M, N, K));
  auto expected_b_cols = K / 2;

  TORCH_CHECK(A_comp.numel() == expected_a_bytes,
              "A_comp.numel() mismatch: expected ", expected_a_bytes, ", got ", A_comp.numel());
  TORCH_CHECK(E.numel() == expected_e_bytes,
              "E.numel() mismatch: expected ", expected_e_bytes, ", got ", E.numel());
  TORCH_CHECK(SFA.numel() == expected_sfa_bytes,
              "SFA.numel() mismatch: expected ", expected_sfa_bytes, ", got ", SFA.numel());
  TORCH_CHECK(SFB.numel() == expected_sfb_bytes,
              "SFB.numel() mismatch: expected ", expected_sfb_bytes, ", got ", SFB.numel());
  TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor shaped [N, K/2]");
  TORCH_CHECK(B.size(0) == N, "B.size(0) mismatch: expected ", N, ", got ", B.size(0));
  TORCH_CHECK(B.size(1) == expected_b_cols,
              "B.size(1) mismatch: expected ", expected_b_cols, ", got ", B.size(1));

  auto C = torch::zeros({M, N}, torch::dtype(torch::kBFloat16).device(A_comp.device()));

  sparse_nvfp4::matmul_host_sparse_nvfp4_bf16(
      reinterpret_cast<cutlass::float_e2m1_t *>(A_comp.data_ptr<uint8_t>()),
      reinterpret_cast<cutlass::float_e2m1_t *>(B.data_ptr<uint8_t>()),
      E.data_ptr<uint8_t>(),
      M,
      N,
      K,
      reinterpret_cast<cutlass::bfloat16_t *>(C.data_ptr<at::BFloat16>()),
      reinterpret_cast<cutlass::bfloat16_t *>(C.data_ptr<at::BFloat16>()),
      reinterpret_cast<cutlass::float_ue4m3_t *>(SFA.data_ptr<uint8_t>()),
      reinterpret_cast<cutlass::float_ue4m3_t *>(SFB.data_ptr<uint8_t>()),
      alpha,
      beta);

  return C;
}

/**************************** Python Bindings ****************************/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul", &matmul,
        "Dense NVFP4 GEMM returning torch.bfloat16",
        py::arg("A"), py::arg("B"), py::arg("SFA"), py::arg("SFB"), py::arg("scale"));
  m.def("matmul_accum", &matmul_accum,
        "Dense NVFP4 GEMM with epilogue accumulation into an existing bf16 tensor",
        py::arg("A"), py::arg("B"), py::arg("SFA"), py::arg("SFB"), py::arg("scale"), py::arg("C_in"), py::arg("beta") = 1.0f);

  m.def("sparse_matmul", &sparse_matmul,
        "Sparse NVFP4 GEMM for precompressed A_comp/E/SFA inputs",
        py::arg("A_comp"), py::arg("B"), py::arg("E"), py::arg("SFA"), py::arg("SFB"),
        py::arg("M"), py::arg("N"), py::arg("K"), py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f);

  m.def("quantize_x_nvfp4", &quantize_x_nvfp4,
        "Quantize activation into dense NVFP4 without reordering",
        py::arg("X"));
  m.def("quantize_w_nvfp4", &quantize_w_nvfp4,
        "Quantize weight into dense NVFP4 without reordering",
        py::arg("W"));

  m.def("reorder_quantize_x", &reorder_quantize_x,
        "Deprecated compatibility alias for dense NVFP4 activation quantization",
        py::arg("X"), py::arg("reorder_index"), py::arg("KE"));
  m.def("reorder_quantize_w", &reorder_quantize_w,
        "Deprecated compatibility alias for dense NVFP4 weight quantization",
        py::arg("W"), py::arg("reorder_index"), py::arg("KE"));

  m.def("compress_sparse_a", &compress_sparse_a,
        "Compress a dense packed NVFP4 activation tensor into A_comp/E for sparse GEMM",
        py::arg("A_q_dense"), py::arg("N"));
  m.def("get_sparse_nvfp4_buffer_sizes", &get_sparse_nvfp4_buffer_sizes,
        "Return byte sizes for A_comp, E, SFA, SFB buffers",
        py::arg("M"), py::arg("N"), py::arg("K"));
  m.def("get_fused_sparse_residual_buffer_sizes", &get_fused_sparse_residual_buffer_sizes,
        "Return byte sizes for A_comp, E, SFA_sparse, Q_res, SF_res buffers",
        py::arg("M"), py::arg("N"), py::arg("K"));
  m.def("fused_sparse_residual_quantize_x", &fused_sparse_residual_quantize_x,
        "Fused sparse main-path quantization plus dense residual quantization",
        py::arg("X"), py::arg("N"));
  m.def("fused_sparse_residual_quantize_x_debug", &fused_sparse_residual_quantize_x_debug,
        "Debug helper returning raw dense sparse/residual quantized activations before compression",
        py::arg("X"), py::arg("N"));
  m.def("quantize_w32_shared", &quantize_w32_shared,
        "Quantize weight once with 32-element groups and return shared payload plus sparse/dense scale views",
        py::arg("W"));
  m.def("duplicate_sfa32_to_sfa16", &duplicate_sfa32_to_sfa16,
        "Expand sparse-path 32-group activation scales into dense 16-group activation scales",
        py::arg("SFA_sparse"), py::arg("M"), py::arg("N"), py::arg("K"));
}

#undef CASE_REORDER_X_16
#undef CASE_REORDER_X_32
#undef CASE_DOWN_X_32
#undef CASE_REORDER_W_16
#undef CASE_REORDER_W_32
#undef CASE_DOWN_W_32

