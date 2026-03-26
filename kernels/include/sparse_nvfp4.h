#pragma once

#include <cassert>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace sparse_nvfp4 {

using namespace cute;

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 64;

using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementE = uint8_t;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledSparseTensorOp;
using KernelScheduleType = cutlass::gemm::KernelSparseTmaWarpSpecializedNvf4Sm120;

using ThreadBlockShape = Shape<_128, _128, _256>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::SparseTmaWarpSpecializedCooperativeSm120>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelScheduleType>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using SparseConfig = typename CollectiveMainloop::SparseConfig;
using ProblemShape = Shape<int, int, int, int>;
using DenseStrideA = cutlass::gemm::TagToStrideA_t<LayoutATag>;
using CompressorUtility = cutlass::transform::kernel::StructuredSparseCompressorUtility<
    ProblemShape,
    cutlass::float_e2m1_t,
    LayoutATag,
    SparseConfig>;

using LayoutA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutA;
using LayoutE = typename Gemm::GemmKernel::CollectiveMainloop::LayoutE;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

inline LayoutA get_layoutA(int M, int N, int K) {
  return SparseConfig::fill_layoutA(make_shape(M, N, K, 1));
}

inline LayoutE get_layoutE(int M, int N, int K) {
  return SparseConfig::fill_layoutE(make_shape(M, N, K, 1));
}

inline LayoutSFA get_layoutSFA(int M, int N, int K) {
  return Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
}

inline LayoutSFB get_layoutSFB(int M, int N, int K) {
  return Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
}

template <class Element>
inline size_t packed_storage_bytes(size_t logical_elements) {
  return (logical_elements * cutlass::sizeof_bits<Element>::value + 7) / 8;
}

inline CompressorUtility get_compressor_utility(int M, int N, int K) {
  DenseStrideA stride_a = cutlass::make_cute_packed_stride(DenseStrideA{}, make_shape(M, K, 1));
  return CompressorUtility(make_shape(M, N, K, 1), stride_a);
}

inline size_t get_compressed_a_numel(int M, int N, int K) {
  auto utility = get_compressor_utility(M, N, K);
  return static_cast<size_t>(utility.get_tensorA_m_physical()) *
         static_cast<size_t>(utility.get_tensorA_k_physical());
}

inline size_t get_metadata_e_numel(int M, int N, int K) {
  auto utility = get_compressor_utility(M, N, K);
  return static_cast<size_t>(utility.get_metadata_m_physical()) *
         static_cast<size_t>(utility.get_metadata_k_physical());
}

inline size_t get_compressed_a_bytes(int M, int N, int K) {
  return packed_storage_bytes<cutlass::float_e2m1_t>(get_compressed_a_numel(M, N, K));
}

inline size_t get_metadata_e_bytes(int M, int N, int K) {
  return packed_storage_bytes<uint8_t>(get_metadata_e_numel(M, N, K));
}

inline size_t get_sfa_numel(int M, int N, int K) {
  return static_cast<size_t>(cute::size(cute::filter_zeros(get_layoutSFA(M, N, K))));
}

inline size_t get_sfb_numel(int M, int N, int K) {
  return static_cast<size_t>(cute::size(cute::filter_zeros(get_layoutSFB(M, N, K))));
}

inline size_t get_sfa_bytes(int M, int N, int K) {
  return static_cast<size_t>((M / 128 + 1) * 128 * K / 16);
}

inline size_t get_sfb_bytes(int M, int N, int K) {
  return static_cast<size_t>((N / 128 + 1) * 128 * K / 16);
}

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
    float alpha = 1.0f,
    float beta = 0.0f);

}  // namespace sparse_nvfp4
