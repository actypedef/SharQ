#include "sparse_nvfp4.h"
#include "cutlass/util/device_memory.h"

namespace sparse_nvfp4 {

void matmul_host_sparse_nvfp4_bf16(
    const cutlass::float_e2m1_t* A,
    const cutlass::float_e2m1_t* B,
    const uint8_t* E,
    int M,
    int N,
    int K,
    cutlass::bfloat16_t* C,
    cutlass::bfloat16_t* D,
    const cutlass::float_ue4m3_t* SFA,
    const cutlass::float_ue4m3_t* SFB,
    float alpha,
    float beta) {
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  LayoutA layout_A = get_layoutA(M, N, K);
  LayoutE layout_E = get_layoutE(M, N, K);
  LayoutSFA layout_SFA = get_layoutSFA(M, N, K);
  LayoutSFB layout_SFB = get_layoutSFB(M, N, K);

  Gemm gemm_op;
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {
          A, layout_A,
          B, stride_B,
          E, layout_E,
          SFA, layout_SFA,
          SFB, layout_SFB,
      },
      {
          {alpha, beta},
          C, stride_C,
          D, stride_D,
      }};

  auto status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS sparse NVFP4 GEMM can_implement failed with status: "
              << cutlass::cutlassGetStatusString(status) << " ("
              << static_cast<int>(status) << ")" << std::endl;
  }
  assert(status == cutlass::Status::kSuccess);

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS sparse NVFP4 GEMM initialize failed with status: "
              << cutlass::cutlassGetStatusString(status) << " ("
              << static_cast<int>(status) << ")" << std::endl;
  }
  assert(status == cutlass::Status::kSuccess);

  status = gemm_op.run();
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS sparse NVFP4 GEMM failed with status: "
              << cutlass::cutlassGetStatusString(status) << " ("
              << static_cast<int>(status) << ")" << std::endl;
  }
  assert(status == cutlass::Status::kSuccess);
}

}  // namespace sparse_nvfp4
