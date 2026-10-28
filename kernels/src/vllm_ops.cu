// SharQ ops for vLLM integration (python module: sharq_ops_vllm).
//
// Compared with bindings.cpp (sharq_ops) this module is CUDA-graph friendly and tuned
// for serving:
//   * every kernel / CUTLASS GEMM is launched on the current PyTorch CUDA stream, no host
//     synchronisation, no cudaMalloc/cudaFree, no autotuning on the hot path,
//   * the per-tensor activation scale is reduced on device and consumed through pointers
//     (prepare kernel + GEMM epilogue alpha_ptr) -> no host round trip,
//   * prepare kernel v2: one thread per 8-element sparse chunk, group-32 / group-16 scale
//     reductions with warp shuffles, 16B coalesced loads, vectorised stores,
//   * two execution modes for y = alpha * (A_sparse @ W^T + A_res @ W^T):
//       - TWO_GEMM : 4:8 sparse NVFP4 GEMM + dense NVFP4 GEMM accumulating in the epilogue
//                    (the paper's kernel; best when compute bound, i.e. large M),
//       - STACKED  : [A_sparse ; A_res] (2M x K, A_sparse expanded to dense NVFP4 with its
//                    32-group scale duplicated to 16-groups -> bit-identical operand values)
//                    in ONE dense NVFP4 GEMM + row-pair add. W is streamed once instead of
//                    twice, which is what matters in the memory-bound decode regime,
//   * Stream-K GEMM variants for shapes that cannot fill all SMs.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

#include "reorder.cuh"
#include "sparse_nvfp4.h"

namespace sharq_vllm {

using namespace cute;

constexpr float kNvfp4Denom = 448.0f * 6.0f;
constexpr float kMinActScale = 1.0e-9f;
constexpr float kFp4Max = 6.0f;
constexpr float kFp8Max = 448.0f;
constexpr float kScaleEps = 0.001953125f;
constexpr int kMetadataGroupsPerTile = 8;  // 8 x 32 = 256 K-elements per metadata tile
constexpr int kMetadataTileRowBytes = 16;

__host__ __device__ __forceinline__ int div_up(int x, int y) { return (x + y - 1) / y; }

int env_int(const char* name, int fallback) {
  const char* v = std::getenv(name);
  return (v == nullptr || v[0] == '\0') ? fallback : std::atoi(v);
}

int cached_sm_count(int device) {
  static int counts[64] = {0};
  if (device < 0 || device >= 64) {
    return cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);
  }
  if (counts[device] == 0) {
    counts[device] = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);
  }
  return counts[device];
}

// ======================================================================================
// Per-tensor |x| max
// ======================================================================================

template <bool kAtomic>
__global__ void absmax_bf16_kernel(const uint16_t* __restrict__ x, int64_t n, unsigned int* __restrict__ out) {
  float m = 0.0f;
  const int64_t n8 = n / 8;
  const uint4* x8 = reinterpret_cast<const uint4*>(x);
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n8; i += stride) {
    uint4 v = __ldg(x8 + i);
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      m = fmaxf(m, fmaxf(fabsf(__uint_as_float(w[j] << 16)), fabsf(__uint_as_float(w[j] & 0xFFFF0000u))));
    }
  }
  for (int64_t i = n8 * 8 + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    m = fmaxf(m, fabsf(__uint_as_float(static_cast<uint32_t>(x[i]) << 16)));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
  }
  __shared__ float warp_max[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) warp_max[warp] = m;
  __syncthreads();
  if (warp == 0) {
    const int num_warps = (blockDim.x + 31) >> 5;
    m = lane < num_warps ? warp_max[lane] : 0.0f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    }
    if (lane == 0) {
      if constexpr (kAtomic) {
        atomicMax(out, __float_as_uint(m));  // non-negative floats order like their bits
      } else {
        *out = __float_as_uint(m);
      }
    }
  }
}

void launch_absmax(const torch::Tensor& x, unsigned int* out, cudaStream_t stream) {
  const int64_t n = x.numel();
  const auto* ptr = reinterpret_cast<const uint16_t*>(x.data_ptr<at::BFloat16>());
  if (n <= (int64_t{1} << 16)) {
    absmax_bf16_kernel<false><<<1, 1024, 0, stream>>>(ptr, n, out);
  } else {
    C10_CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(unsigned int), stream));
    constexpr int kThreads = 512;
    int64_t blocks = std::min<int64_t>((n / 8 + kThreads - 1) / kThreads, 4 * cached_sm_count(x.get_device()));
    absmax_bf16_kernel<true><<<static_cast<int>(std::max<int64_t>(blocks, 1)), kThreads, 0, stream>>>(ptr, n, out);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ======================================================================================
// Prepare kernel v2
// ======================================================================================

template <typename ScaleTensor>
__device__ __forceinline__ void store_scale(ScaleTensor& scale_tensor, int row_id, int group_id, sf_t scale) {
  auto c0 = make_coord(make_coord(row_id % 32, (row_id / 32) % 4), row_id / 128);
  auto c1 = make_coord(make_coord(0, group_id % 4), group_id / 4);
  scale_tensor(make_coord(c0, c1, make_coord(0, 0))) = scale;
}

// ---- hardware conversions (sm_100a / sm_120a), bit-identical to CUTLASS RNE conversions ----

// 8 floats -> 8 e2m1 nibbles, element i in bits [4i, 4i+4). RNE, satfinite.
__device__ __forceinline__ uint32_t pack_e2m1x8(const float (&v)[8]) {
  uint32_t r;
  asm("{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "}"
      : "=r"(r)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
  return r;
}

// e2m1 nibble -> float (branch-free): {0, .5, 1, 1.5, 2, 3, 4, 6} with sign.
__device__ __forceinline__ float e2m1_to_float(uint32_t nib) {
  const uint32_t e = (nib >> 1) & 3u;
  const uint32_t m = nib & 1u;
  const uint32_t bits = e ? (((126u + e) << 23) | (m << 22)) : (m ? 0x3F000000u : 0u);
  return __uint_as_float(bits | ((nib & 8u) << 28));
}

// positive float -> (u)e4m3 bits, RNE, satfinite (same instruction CUTLASS uses for float_ue4m3_t).
__device__ __forceinline__ uint32_t f32_to_e4m3(float v) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(0.0f), "f"(v));
  return r & 0xFFu;
}

__device__ __forceinline__ float e4m3_to_f32(uint32_t b) {
  uint32_t packed;
  const uint16_t bits = static_cast<uint16_t>(b);
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(packed) : "h"(bits));
  return __half2float(__ushort_as_half(static_cast<unsigned short>(packed & 0xFFFFu)));
}

__device__ __forceinline__ sf_t sf_from_bits(uint32_t b) {
  sf_t s;
  s.storage = static_cast<uint8_t>(b);
  return s;
}

// float(bf16_rn(fp32_rn(x / s))) computed as x * rcp(s); |x*rcp(s) - fp32_rn(x/s)| <= 2.5 ulp,
// so the bf16 rounding can only differ when within a few ulps of a bf16 midpoint -> exact fallback.
__device__ __forceinline__ float scaled_to_bf16(float x, float s, float rs) {
  float q = x * rs;
  const int low = static_cast<int>(__float_as_uint(q) & 0xFFFFu) - 0x8000;
  if (low >= -32 && low <= 32) {
    q = __fdiv_rn(x, s);
  }
  return __bfloat162float(__float2bfloat16_rn(q));
}

constexpr int kPrepThreads = 256;

// One thread = one 8-element chunk (4 pairs) of one row. 4 consecutive lanes = one 32-group.
// kStacked == false : writes A_comp / E / SFA_sparse (sparse GEMM operand) and Q_res / SF_res.
// kStacked == true  : writes a (2M x K) dense NVFP4 operand: rows [0,M) = sparse part (pruned
//                     values = 0, 32-group scale duplicated to both 16-groups), rows [M,2M) =
//                     residual part, plus its 16-group scale layout for 2M rows.
template <bool kStacked, typename SparseScaleTensor, typename DenseScaleTensor>
__global__ void __launch_bounds__(kPrepThreads) sharq_prepare_v2_kernel(
    const uint16_t* __restrict__ input,
    const unsigned int* __restrict__ absmax_bits,
    float weight_scale,
    float* __restrict__ scalars,  // [1] act_scale, [2] alpha
    uint8_t* __restrict__ a_comp,
    uint8_t* __restrict__ e,
    SparseScaleTensor sparse_scale_tensor,
    uint8_t* __restrict__ q_dense,
    DenseScaleTensor dense_scale_tensor,
    int M,
    int K,
    int aligned_M) {
  const int chunks_per_row = K >> 3;
  const int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(M) * chunks_per_row;
  const float act_scale = fmaxf(__uint_as_float(__ldg(absmax_bits)) / kNvfp4Denom, kMinActScale);
  if (gid == 0) {
    scalars[1] = act_scale;
    scalars[2] = act_scale * weight_scale;
  }
  // K % 256 == 0 -> chunks_per_row % 32 == 0 -> a warp never straddles rows and
  // total is a multiple of 32, so whole warps exit together (shuffles stay full-mask).
  if (gid >= total) {
    return;
  }
  const int row = static_cast<int>(gid / chunks_per_row);
  const int chunk = static_cast<int>(gid - static_cast<int64_t>(row) * chunks_per_row);
  const int lane = threadIdx.x & 31;

  // ---- load + per-tensor scale (bit-identical to (x / scale).to(bfloat16)) ----
  const float act_rscale = __frcp_rn(act_scale);
  const uint4 pack = __ldg(reinterpret_cast<const uint4*>(input) + gid);
  const uint32_t w[4] = {pack.x, pack.y, pack.z, pack.w};
  float x[8];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    x[2 * j + 0] = scaled_to_bf16(__uint_as_float(w[j] << 16), act_scale, act_rscale);
    x[2 * j + 1] = scaled_to_bf16(__uint_as_float(w[j] & 0xFFFF0000u), act_scale, act_rscale);
  }

  // ---- pair top-2 of 8 (same tie-breaking as the reference kernel) ----
  float best0 = -1.0f, best1 = -1.0f;
  int idx0 = 0, idx1 = 1;
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    const float s = fmaxf(fabsf(x[2 * p]), fabsf(x[2 * p + 1]));
    if (s > best0) {
      best1 = best0; idx1 = idx0; best0 = s; idx0 = p;
    } else if (s > best1) {
      best1 = s; idx1 = p;
    }
  }
  if (idx1 < idx0) {
    const int t = idx0; idx0 = idx1; idx1 = t;
  }
  const uint32_t keep_mask = (1u << idx0) | (1u << idx1);

  float sparse_max = 0.0f;
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    if (keep_mask & (1u << p)) {
      sparse_max = fmaxf(sparse_max, fmaxf(fabsf(x[2 * p]), fabsf(x[2 * p + 1])));
    }
  }
  sparse_max = fmaxf(sparse_max, __shfl_xor_sync(0xffffffffu, sparse_max, 1));
  sparse_max = fmaxf(sparse_max, __shfl_xor_sync(0xffffffffu, sparse_max, 2));

  const uint32_t sparse_scale_bits = f32_to_e4m3(fminf(fmaxf(__fdiv_rn(sparse_max, kFp4Max), kScaleEps), kFp8Max));
  const sf_t sparse_scale = sf_from_bits(sparse_scale_bits);
  const float sparse_scale_q = e4m3_to_f32(sparse_scale_bits);
  const float sparse_rscale = __frcp_rn(sparse_scale_q);

  float sv[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    sv[i] = (keep_mask & (1u << (i >> 1))) ? fminf(fmaxf(x[i] * sparse_rscale, -kFp4Max), kFp4Max) : 0.0f;
  }
  const uint32_t sparse_word = pack_e2m1x8(sv);  // dense-form sparse nibbles (pruned -> 0)

  float residual[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (keep_mask & (1u << (i >> 1))) {
      residual[i] = x[i] - e2m1_to_float((sparse_word >> (4 * i)) & 0xFu) * sparse_scale_q;
    } else {
      residual[i] = x[i];
    }
  }

  // ---- residual: 16-group scale over lane pairs ----
  float res_max = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) res_max = fmaxf(res_max, fabsf(residual[i]));
  res_max = fmaxf(res_max, __shfl_xor_sync(0xffffffffu, res_max, 1));
  const uint32_t res_scale_bits = f32_to_e4m3(fminf(fmaxf(__fdiv_rn(res_max, kFp4Max), kScaleEps), kFp8Max));
  const sf_t res_scale = sf_from_bits(res_scale_bits);
  const float res_rscale = __frcp_rn(e4m3_to_f32(res_scale_bits));
  float rv[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    rv[i] = fminf(fmaxf(residual[i] * res_rscale, -kFp4Max), kFp4Max);
  }
  const uint32_t res_word = pack_e2m1x8(rv);

  const int group16 = chunk >> 1;
  const int group32 = chunk >> 2;
  const size_t row_bytes = static_cast<size_t>(K) >> 1;

  if constexpr (kStacked) {
    reinterpret_cast<uint32_t*>(q_dense + static_cast<size_t>(row) * row_bytes)[chunk] = sparse_word;
    reinterpret_cast<uint32_t*>(q_dense + static_cast<size_t>(row + M) * row_bytes)[chunk] = res_word;
    if ((lane & 1) == 0) {
      store_scale(dense_scale_tensor, row, group16, sparse_scale);
      store_scale(dense_scale_tensor, row + M, group16, res_scale);
    }
  } else {
    // compressed sparse payload: bytes of the two kept pairs
    const uint32_t b0 = (sparse_word >> (8 * idx0)) & 0xFFu;
    const uint32_t b1 = (sparse_word >> (8 * idx1)) & 0xFFu;
    reinterpret_cast<uint16_t*>(a_comp + static_cast<size_t>(row) * (K >> 2))[chunk] =
        static_cast<uint16_t>(b0 | (b1 << 8));
    // metadata: 4 bits per chunk, 16 bits per 32-group
    uint32_t meta = static_cast<uint32_t>(idx0 | (idx1 << 2)) << (4 * (lane & 3));
    meta |= __shfl_xor_sync(0xffffffffu, meta, 1);
    meta |= __shfl_xor_sync(0xffffffffu, meta, 2);
    if ((lane & 3) == 0) {
      const int tile_k = group32 / kMetadataGroupsPerTile;
      const int g_in_tile = group32 % kMetadataGroupsPerTile;
      *reinterpret_cast<uint16_t*>(e + static_cast<size_t>(tile_k) * aligned_M * kMetadataTileRowBytes +
                                   static_cast<size_t>(row) * kMetadataTileRowBytes + g_in_tile * 2) =
          static_cast<uint16_t>(meta);
      store_scale(sparse_scale_tensor, row, group32, sparse_scale);
    }
    reinterpret_cast<uint32_t*>(q_dense + static_cast<size_t>(row) * row_bytes)[chunk] = res_word;
    if ((lane & 1) == 0) {
      store_scale(dense_scale_tensor, row, group16, res_scale);
    }
  }
}

// out[i] = y[i] + y[i + M*N]   (bf16, row-pair reduction for the stacked mode)
__global__ void pair_add_bf16_kernel(const uint4* __restrict__ y, uint4* __restrict__ out, int64_t n8) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n8) return;
  const uint4 a = __ldg(y + i);
  const uint4 b = __ldg(y + i + n8);
  const __nv_bfloat162* pa = reinterpret_cast<const __nv_bfloat162*>(&a);
  const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&b);
  uint4 r;
  __nv_bfloat162* pr = reinterpret_cast<__nv_bfloat162*>(&r);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 fa = __bfloat1622float2(pa[j]);
    const float2 fb = __bfloat1622float2(pb[j]);
    pr[j] = __floats2bfloat162_rn(fa.x + fb.x, fa.y + fb.y);
  }
  out[i] = r;
}

// ======================================================================================
// GEMM variants
// ======================================================================================

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementCD = cutlass::bfloat16_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutCD = cutlass::layout::RowMajor;
constexpr int kAlignCD = 8;

template <class TileShape, class Scheduler>
struct DenseGemmCfg {
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      TileShape, Shape<_1, _1, _1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      ElementCD, LayoutCD, kAlignCD,
      ElementCD, LayoutCD, kAlignCD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      ElementA, LayoutA, 32,
      ElementB, LayoutB, 32,
      float,
      TileShape, Shape<_1, _1, _1>,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, Scheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  static constexpr bool kStreamK = !std::is_same_v<Scheduler, void>;
};

template <class Scheduler>
struct SparseGemmCfg {
  using TileShape = sharq::blackwell_arch::SparseTileShape;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledSparseTensorOp,
      TileShape, Shape<_1, _1, _1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      ElementCD, LayoutCD, kAlignCD,
      ElementCD, LayoutCD, kAlignCD,
      sharq::blackwell_arch::SparseEpilogueSchedule>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledSparseTensorOp,
      ElementA, LayoutA, 64,
      ElementB, LayoutB, 32,
      float,
      TileShape, Shape<_1, _1, _1>,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      sharq::blackwell_arch::SparseKernelSchedule>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, Scheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  static constexpr bool kStreamK = !std::is_same_v<Scheduler, void>;
};

using Dense128 = DenseGemmCfg<Shape<_128, _128, _128>, void>;
using Dense128SK = DenseGemmCfg<Shape<_128, _128, _128>, cutlass::gemm::StreamKScheduler>;
using Dense256 = DenseGemmCfg<Shape<_128, _128, _256>, void>;
using Dense256SK = DenseGemmCfg<Shape<_128, _128, _256>, cutlass::gemm::StreamKScheduler>;
using Sparse = SparseGemmCfg<void>;
using SparseSK = SparseGemmCfg<cutlass::gemm::StreamKScheduler>;

// Layout compatibility: activation / weight scale buffers are produced with the layouts
// of nvfp4::Gemm (dense) and sparse_nvfp4::Gemm (sparse).
static_assert(std::is_same_v<typename Dense128::Gemm::GemmKernel::CollectiveMainloop::LayoutSFA,
                             typename nvfp4::Gemm::GemmKernel::CollectiveMainloop::LayoutSFA>);
static_assert(std::is_same_v<typename Dense256::Gemm::GemmKernel::CollectiveMainloop::LayoutSFA,
                             typename nvfp4::Gemm::GemmKernel::CollectiveMainloop::LayoutSFA>);
static_assert(std::is_same_v<typename Sparse::Gemm::GemmKernel::CollectiveMainloop::LayoutSFA,
                             typename sparse_nvfp4::Gemm::GemmKernel::CollectiveMainloop::LayoutSFA>);

template <class Cfg>
void run_gemm(typename Cfg::Gemm::Arguments& arguments, const torch::Tensor& ref, cudaStream_t stream,
              int splits, const char* name) {
  using Gemm = typename Cfg::Gemm;
  arguments.hw_info.device_id = ref.get_device();
  arguments.hw_info.sm_count = cached_sm_count(ref.get_device());
  if constexpr (Cfg::kStreamK) {
    using DecompositionMode = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
    if (splits > 1) {
      arguments.scheduler.splits = splits;
      arguments.scheduler.decomposition_mode = DecompositionMode::SplitK;
    } else {
      arguments.scheduler.decomposition_mode = DecompositionMode::Heuristic;
    }
  }
  Gemm gemm_op;
  cutlass::Status status = gemm_op.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, name, " can_implement failed: ",
              cutlass::cutlassGetStatusString(status));
  const size_t workspace_size = Gemm::get_workspace_size(arguments);
  torch::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size > 0) {
    workspace = torch::empty({static_cast<int64_t>(workspace_size)}, torch::dtype(torch::kUInt8).device(ref.device()));
    workspace_ptr = workspace.data_ptr();
  }
  status = gemm_op.initialize(arguments, workspace_ptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, name, " initialize failed: ",
              cutlass::cutlassGetStatusString(status));
  status = gemm_op.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, name, " run failed: ",
              cutlass::cutlassGetStatusString(status));
}

template <class Cfg>
void dense_gemm_impl(const uint8_t* a, const uint8_t* b, const uint8_t* sfa, const uint8_t* sfb,
                     int M, int N, int K, at::BFloat16* c, at::BFloat16* d, float beta, const float* alpha_ptr,
                     const torch::Tensor& ref, cudaStream_t stream, int splits) {
  using Gemm = typename Cfg::Gemm;
  using GK = typename Gemm::GemmKernel;
  using Mainloop = typename GK::CollectiveMainloop;
  auto stride_A = cutlass::make_cute_packed_stride(typename GK::StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(typename GK::StrideB{}, {N, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(typename GK::StrideC{}, {M, N, 1});
  auto stride_D = cutlass::make_cute_packed_stride(typename GK::StrideD{}, {M, N, 1});
  auto shape = cute::make_shape(M, N, K, 1);
  auto layout_SFA = Mainloop::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(shape);
  auto layout_SFB = Mainloop::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(shape);
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {reinterpret_cast<const cutlass::float_e2m1_t*>(a), stride_A,
       reinterpret_cast<const cutlass::float_e2m1_t*>(b), stride_B,
       reinterpret_cast<const cutlass::float_ue4m3_t*>(sfa), layout_SFA,
       reinterpret_cast<const cutlass::float_ue4m3_t*>(sfb), layout_SFB},
      {{}, reinterpret_cast<ElementCD*>(c), stride_C, reinterpret_cast<ElementCD*>(d), stride_D}};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = beta;
  arguments.epilogue.thread.alpha_ptr = alpha_ptr;
  run_gemm<Cfg>(arguments, ref, stream, splits, "sharq dense NVFP4 GEMM");
}

template <class Cfg>
void sparse_gemm_impl(const uint8_t* a_comp, const uint8_t* b, const uint8_t* e, const uint8_t* sfa,
                      const uint8_t* sfb, int M, int N, int K, at::BFloat16* d, const float* alpha_ptr,
                      const torch::Tensor& ref, cudaStream_t stream, int splits) {
  using Gemm = typename Cfg::Gemm;
  using GK = typename Gemm::GemmKernel;
  using Mainloop = typename GK::CollectiveMainloop;
  auto shape = cute::make_shape(M, N, K, 1);
  auto stride_B = cutlass::make_cute_packed_stride(typename GK::StrideB{}, {N, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(typename GK::StrideC{}, {M, N, 1});
  auto stride_D = cutlass::make_cute_packed_stride(typename GK::StrideD{}, {M, N, 1});
  auto layout_A = Mainloop::SparseConfig::fill_layoutA(shape);
  auto layout_E = Mainloop::SparseConfig::fill_layoutE(shape);
  auto layout_SFA = Mainloop::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(shape);
  auto layout_SFB = Mainloop::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(shape);
  auto* d_ptr = reinterpret_cast<ElementCD*>(d);
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {reinterpret_cast<const cutlass::float_e2m1_t*>(a_comp), layout_A,
       reinterpret_cast<const cutlass::float_e2m1_t*>(b), stride_B,
       e, layout_E,
       reinterpret_cast<const cutlass::float_ue4m3_t*>(sfa), layout_SFA,
       reinterpret_cast<const cutlass::float_ue4m3_t*>(sfb), layout_SFB},
      {{}, d_ptr, stride_C, d_ptr, stride_D}};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;
  arguments.epilogue.thread.alpha_ptr = alpha_ptr;
  run_gemm<Cfg>(arguments, ref, stream, splits, "sharq sparse NVFP4 GEMM");
}

// variant ids: dense  0=128x128x128  1=128x128x128 stream-K  2=128x128x256  3=128x128x256 stream-K
//              sparse 0=128x128x256  1=128x128x256 stream-K
void dense_gemm(int variant, const uint8_t* a, const uint8_t* b, const uint8_t* sfa, const uint8_t* sfb,
                int M, int N, int K, at::BFloat16* c, at::BFloat16* d, float beta, const float* alpha_ptr,
                const torch::Tensor& ref, cudaStream_t stream, int splits) {
  switch (variant) {
    case 0: return dense_gemm_impl<Dense128>(a, b, sfa, sfb, M, N, K, c, d, beta, alpha_ptr, ref, stream, splits);
    case 1: return dense_gemm_impl<Dense128SK>(a, b, sfa, sfb, M, N, K, c, d, beta, alpha_ptr, ref, stream, splits);
    case 2: return dense_gemm_impl<Dense256>(a, b, sfa, sfb, M, N, K, c, d, beta, alpha_ptr, ref, stream, splits);
    case 3: return dense_gemm_impl<Dense256SK>(a, b, sfa, sfb, M, N, K, c, d, beta, alpha_ptr, ref, stream, splits);
    default: TORCH_CHECK(false, "unknown dense variant ", variant);
  }
}

void sparse_gemm(int variant, const uint8_t* a_comp, const uint8_t* b, const uint8_t* e, const uint8_t* sfa,
                 const uint8_t* sfb, int M, int N, int K, at::BFloat16* d, const float* alpha_ptr,
                 const torch::Tensor& ref, cudaStream_t stream, int splits) {
  switch (variant) {
    case 0: return sparse_gemm_impl<Sparse>(a_comp, b, e, sfa, sfb, M, N, K, d, alpha_ptr, ref, stream, splits);
    case 1: return sparse_gemm_impl<SparseSK>(a_comp, b, e, sfa, sfb, M, N, K, d, alpha_ptr, ref, stream, splits);
    default: TORCH_CHECK(false, "unknown sparse variant ", variant);
  }
}

// ======================================================================================
// Dispatch heuristics (tuned on RTX PRO 5000 Blackwell / sm_120a, Llama-3.1-8B shapes)
// ======================================================================================

struct Plan {
  int mode;          // 0 two-gemm, 1 stacked
  int dense_variant;
  int sparse_variant;
  int splits;
};

// Derived from benchmarks/vllm/tune_sharq_plans.py:
//   * stacked single-GEMM wins while the problem is weight-bandwidth bound (M <= ~192,
//     up to ~256 for the long-K down projection),
//   * Stream-K only pays off for long K (down_proj, K = 14336) when tiles < #SMs,
//   * the 128x128x256 tile is best for the tiny stacked problems (2M <= 128).
Plan auto_plan(int M, int N, int K, int sm_count) {
  Plan p{0, 0, 0, 1};
  const int stack_max_m = env_int("SHARQ_STACK_MAX_M", 192);
  const int stack_max_m_long_k = env_int("SHARQ_STACK_MAX_M_LONG_K", 256);
  const bool long_k = K >= 8192;
  if (M <= stack_max_m || (long_k && M <= stack_max_m_long_k)) {
    p.mode = 1;
    const int tiles = div_up(2 * M, 128) * div_up(N, 128);
    if (long_k && tiles < sm_count) {
      p.dense_variant = 1;
    } else {
      p.dense_variant = (2 * M <= 128) ? 2 : 0;
    }
  } else {
    p.mode = 0;
    const bool sk = long_k && M <= 1024;
    p.sparse_variant = sk ? 1 : 0;
    p.dense_variant = sk ? 1 : 0;
  }
  p.mode = env_int("SHARQ_FORCE_MODE", p.mode);
  p.dense_variant = env_int("SHARQ_FORCE_DENSE", p.dense_variant);
  p.sparse_variant = env_int("SHARQ_FORCE_SPARSE", p.sparse_variant);
  return p;
}

// ======================================================================================
// Ops
// ======================================================================================

inline int64_t dense_sf_bytes(int rows, int k) { return static_cast<int64_t>(rows / 128 + 1) * 128 * k / 16; }

torch::Tensor checked_input(const torch::Tensor& x_in) {
  TORCH_CHECK(x_in.is_cuda() && x_in.dim() == 2, "x must be a 2D CUDA tensor");
  TORCH_CHECK(x_in.scalar_type() == at::ScalarType::BFloat16, "x must be bfloat16");
  TORCH_CHECK(x_in.size(1) % 256 == 0, "SharQ requires K % 256 == 0, got ", x_in.size(1));
  torch::Tensor x = x_in.contiguous();
  if (reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 != 0) x = x.clone();
  return x;
}

struct Buffers {
  torch::Tensor scalars, a_comp, e, sfa_sparse, q_dense, sf_dense;
};

// Runs absmax + prepare. Stacked: q_dense is (2M x K/2) with sf for 2M rows.
Buffers prepare(const torch::Tensor& x, int N, float weight_scale, bool stacked, cudaStream_t stream, bool zero_init) {
  const int M = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  auto u8 = torch::dtype(torch::kUInt8).device(x.device());
  auto alloc = [&](int64_t n) { return zero_init ? torch::zeros({n}, u8) : torch::empty({n}, u8); };
  Buffers b;
  b.scalars = torch::empty({4}, torch::dtype(torch::kFloat32).device(x.device()));
  auto* absmax_bits = reinterpret_cast<unsigned int*>(b.scalars.data_ptr<float>());
  launch_absmax(x, absmax_bits, stream);

  const int dense_rows = stacked ? 2 * M : M;
  b.q_dense = alloc(static_cast<int64_t>(dense_rows) * (K / 2));
  b.sf_dense = alloc(dense_sf_bytes(dense_rows, K));
  auto dense_sf_tensor = cute::make_tensor(reinterpret_cast<sf_t*>(b.sf_dense.data_ptr<uint8_t>()),
                                           filter_zeros(nvfp4::get_layoutSFA(dense_rows, K)));
  const int64_t total = static_cast<int64_t>(M) * (K / 8);
  const int blocks = static_cast<int>((total + kPrepThreads - 1) / kPrepThreads);
  const int aligned_M = div_up(M, 128) * 128;

  if (stacked) {
    auto sparse_sf_tensor = dense_sf_tensor;  // unused
    sharq_prepare_v2_kernel<true><<<blocks, kPrepThreads, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(x.data_ptr<at::BFloat16>()), absmax_bits, weight_scale,
        b.scalars.data_ptr<float>(), nullptr, nullptr, sparse_sf_tensor,
        b.q_dense.data_ptr<uint8_t>(), dense_sf_tensor, M, K, aligned_M);
  } else {
    b.a_comp = alloc(static_cast<int64_t>(sparse_nvfp4::get_compressed_a_bytes(M, N, K)));
    b.e = torch::zeros({static_cast<int64_t>(sparse_nvfp4::get_metadata_e_bytes(M, N, K))}, u8);
    b.sfa_sparse = alloc(static_cast<int64_t>(sparse_nvfp4::get_sfa_bytes(M, N, K)));
    auto sparse_sf_tensor = cute::make_tensor(reinterpret_cast<sf_t*>(b.sfa_sparse.data_ptr<uint8_t>()),
                                              filter_zeros(sparse_nvfp4::get_layoutSFA(M, N, K)));
    sharq_prepare_v2_kernel<false><<<blocks, kPrepThreads, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(x.data_ptr<at::BFloat16>()), absmax_bits, weight_scale,
        b.scalars.data_ptr<float>(), b.a_comp.data_ptr<uint8_t>(), b.e.data_ptr<uint8_t>(), sparse_sf_tensor,
        b.q_dense.data_ptr<uint8_t>(), dense_sf_tensor, M, K, aligned_M);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return b;
}

torch::Tensor linear_impl(const torch::Tensor& x_in, const torch::Tensor& qw, const torch::Tensor& sfw_sparse,
                          const torch::Tensor& sfw_dense, double weight_scale, Plan plan, bool use_auto) {
  torch::Tensor x = checked_input(x_in);
  TORCH_CHECK(qw.is_cuda() && qw.is_contiguous() && qw.scalar_type() == at::ScalarType::Byte,
              "QW must be a contiguous uint8 CUDA tensor");
  TORCH_CHECK(qw.size(1) * 2 == x.size(1), "QW / x K mismatch");
  const at::cuda::CUDAGuard device_guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const int M = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(qw.size(0));
  auto bf16 = torch::dtype(torch::kBFloat16).device(x.device());
  if (M == 0) return torch::empty({0, N}, bf16);
  if (use_auto) plan = auto_plan(M, N, K, cached_sm_count(x.get_device()));

  if (plan.mode == 1) {
    Buffers b = prepare(x, N, static_cast<float>(weight_scale), true, stream, false);
    const float* alpha = b.scalars.data_ptr<float>() + 2;
    auto y2 = torch::empty({2 * M, N}, bf16);
    dense_gemm(plan.dense_variant, b.q_dense.data_ptr<uint8_t>(), qw.data_ptr<uint8_t>(),
               b.sf_dense.data_ptr<uint8_t>(), sfw_dense.data_ptr<uint8_t>(), 2 * M, N, K,
               y2.data_ptr<at::BFloat16>(), y2.data_ptr<at::BFloat16>(), 0.0f, alpha, x, stream, plan.splits);
    auto out = torch::empty({M, N}, bf16);
    const int64_t n8 = static_cast<int64_t>(M) * N / 8;
    pair_add_bf16_kernel<<<static_cast<int>((n8 + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<const uint4*>(y2.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint4*>(out.data_ptr<at::BFloat16>()), n8);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
  }

  Buffers b = prepare(x, N, static_cast<float>(weight_scale), false, stream, false);
  const float* alpha = b.scalars.data_ptr<float>() + 2;
  auto out = torch::empty({M, N}, bf16);
  sparse_gemm(plan.sparse_variant, b.a_comp.data_ptr<uint8_t>(), qw.data_ptr<uint8_t>(), b.e.data_ptr<uint8_t>(),
              b.sfa_sparse.data_ptr<uint8_t>(), sfw_sparse.data_ptr<uint8_t>(), M, N, K,
              out.data_ptr<at::BFloat16>(), alpha, x, stream, plan.splits);
  dense_gemm(plan.dense_variant, b.q_dense.data_ptr<uint8_t>(), qw.data_ptr<uint8_t>(),
             b.sf_dense.data_ptr<uint8_t>(), sfw_dense.data_ptr<uint8_t>(), M, N, K,
             out.data_ptr<at::BFloat16>(), out.data_ptr<at::BFloat16>(), 1.0f, alpha, x, stream, plan.splits);
  return out;
}

// y = act_scale * weight_scale * (sparse(Xs) @ W^T + residual(Xs) @ W^T), automatic plan
torch::Tensor sharq_linear_forward(const torch::Tensor& x, const torch::Tensor& qw, const torch::Tensor& sfw_sparse,
                                   const torch::Tensor& sfw_dense, double weight_scale) {
  return linear_impl(x, qw, sfw_sparse, sfw_dense, weight_scale, Plan{}, true);
}

// explicit plan (for tuning / ablation)
torch::Tensor sharq_linear_forward_cfg(const torch::Tensor& x, const torch::Tensor& qw,
                                       const torch::Tensor& sfw_sparse, const torch::Tensor& sfw_dense,
                                       double weight_scale, int64_t mode, int64_t dense_variant,
                                       int64_t sparse_variant, int64_t splits) {
  Plan p{static_cast<int>(mode), static_cast<int>(dense_variant), static_cast<int>(sparse_variant),
         static_cast<int>(splits)};
  return linear_impl(x, qw, sfw_sparse, sfw_dense, weight_scale, p, false);
}

// Debug: two-gemm buffers (A_comp, E, SFA_sparse, Q_res, SF_res, act_scale)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sharq_prepare(const torch::Tensor& x_in, int64_t N) {
  torch::Tensor x = checked_input(x_in);
  const at::cuda::CUDAGuard device_guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  Buffers b = prepare(x, static_cast<int>(N), 1.0f, false, stream, true);
  return std::make_tuple(b.a_comp, b.e, b.sfa_sparse, b.q_dense.view({x.size(0), x.size(1) / 2}), b.sf_dense,
                         b.scalars.slice(0, 1, 2).clone());
}

// Debug: stacked buffers (Q_stacked [2M, K/2], SF_stacked, act_scale)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sharq_prepare_stacked(const torch::Tensor& x_in) {
  torch::Tensor x = checked_input(x_in);
  const at::cuda::CUDAGuard device_guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  Buffers b = prepare(x, 128, 1.0f, true, stream, true);
  return std::make_tuple(b.q_dense.view({2 * x.size(0), x.size(1) / 2}), b.sf_dense, b.scalars.slice(0, 1, 2).clone());
}

// Production stacked prepare for an external (e.g. FlashInfer, autotuned) dense NVFP4 GEMM.
// Returns (Q_stacked uint8 [2M, K/2], SF_stacked fp8e4m3 [round_up(2M,128), K/16] in the
// 128x4 swizzled layout, alpha fp32 [1] = act_scale * weight_scale). No zero-init.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sharq_prepare_stacked_fast(const torch::Tensor& x_in,
                                                                                   double weight_scale) {
  torch::Tensor x = checked_input(x_in);
  const at::cuda::CUDAGuard device_guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const int64_t M = x.size(0);
  const int64_t K = x.size(1);
  Buffers b = prepare(x, 128, static_cast<float>(weight_scale), true, stream, false);
  const int64_t rows_pad = div_up(static_cast<int>(2 * M), 128) * 128;
  auto sf = b.sf_dense.slice(0, 0, rows_pad * (K / 16)).view(torch::kFloat8_e4m3fn).view({rows_pad, K / 16});
  return std::make_tuple(b.q_dense.view({2 * M, K / 2}), sf, b.scalars.slice(0, 2, 3));
}

// out = y2[:M] + y2[M:]
torch::Tensor sharq_pair_add(const torch::Tensor& y2) {
  TORCH_CHECK(y2.is_cuda() && y2.is_contiguous() && y2.dim() == 2 && y2.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(y2.size(0) % 2 == 0 && y2.size(1) % 8 == 0);
  const at::cuda::CUDAGuard device_guard(y2.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const int64_t M = y2.size(0) / 2;
  const int64_t N = y2.size(1);
  auto out = torch::empty({M, N}, y2.options());
  const int64_t n8 = M * N / 8;
  if (n8 > 0) {
    pair_add_bf16_kernel<<<static_cast<int>((n8 + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<const uint4*>(y2.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint4*>(out.data_ptr<at::BFloat16>()), n8);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

}  // namespace sharq_vllm

PYBIND11_MODULE(sharq_ops_vllm, m) {
  m.def("sharq_linear_forward", &sharq_vllm::sharq_linear_forward,
        "Fused SharQ linear (device-side act scale, stream-aware, CUDA-graph safe, auto plan)",
        py::arg("x"), py::arg("QW"), py::arg("SFW_sparse"), py::arg("SFW_dense"), py::arg("weight_scale"));
  m.def("sharq_linear_forward_cfg", &sharq_vllm::sharq_linear_forward_cfg,
        "SharQ linear with an explicit plan: mode 0=two-gemm 1=stacked; dense variant 0..3; sparse variant 0..1",
        py::arg("x"), py::arg("QW"), py::arg("SFW_sparse"), py::arg("SFW_dense"), py::arg("weight_scale"),
        py::arg("mode"), py::arg("dense_variant"), py::arg("sparse_variant"), py::arg("splits") = 1);
  m.def("sharq_prepare", &sharq_vllm::sharq_prepare, "Debug: two-gemm prepare buffers", py::arg("x"), py::arg("N"));
  m.def("sharq_prepare_stacked", &sharq_vllm::sharq_prepare_stacked, "Debug: stacked prepare buffers", py::arg("x"));
  m.def("sharq_prepare_stacked_fast", &sharq_vllm::sharq_prepare_stacked_fast,
        "Stacked prepare for an external dense NVFP4 GEMM -> (Q[2M,K/2], SF fp8 [pad(2M),K/16], alpha[1])",
        py::arg("x"), py::arg("weight_scale"));
  m.def("sharq_pair_add", &sharq_vllm::sharq_pair_add, "out = y2[:M] + y2[M:]", py::arg("y2"));
}
