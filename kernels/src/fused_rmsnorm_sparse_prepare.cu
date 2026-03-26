#include "fused_sparse_prepare.h"

#include <cuda_runtime.h>

namespace fused_rmsnorm_sparse_prepare {

constexpr int kSparseElementsPerThread = 32;
constexpr int kResidualElementsPerGroup = 16;
constexpr int kCompressedSparseBytesPerThread = kSparseElementsPerThread / 4;
constexpr int kMetadataBytesPerThread = kSparseElementsPerThread / 16;
constexpr int kMetadataGroupsPerTile = 8;
constexpr int kMetadataTileRowBytes = 16;
constexpr int kMaxGroupsPerRow = 1024;

constexpr float kFp4Max = 6.0f;
constexpr float kFp8Max = 448.0f;
constexpr float kScaleEps = 0.001953125f;

struct PackFp4 {
  int8_t low : 4;
  int8_t high : 4;
};

__device__ __forceinline__ float fp_abs(float x) {
  return x < 0.0f ? -x : x;
}

__device__ __forceinline__ float fp_max(float a, float b) {
  return a > b ? a : b;
}

__device__ __forceinline__ float fp_min(float a, float b) {
  return a < b ? a : b;
}

__device__ __forceinline__ float clamp_val(float x, float lo, float hi) {
  return fp_max(lo, fp_min(x, hi));
}

__host__ __device__ __forceinline__ int div_up_int(int x, int y) {
  return (x + y - 1) / y;
}

template <typename ScaleTensor>
__device__ __forceinline__ void store_scale(
    ScaleTensor scale_tensor,
    int row_id,
    int group_id,
    sf_t scale) {
  auto logical_coord0 = make_coord(make_coord(row_id % 32, (row_id / 32) % 4), row_id / 128);
  auto logical_coord1 = make_coord(make_coord(0, group_id % 4), group_id / 4);
  auto logical_coord2 = make_coord(0, 0);
  scale_tensor(make_coord(logical_coord0, logical_coord1, logical_coord2)) = scale;
}

__device__ __forceinline__ void top2_pair_indices_8(
    const float *vals,
    int &idx0,
    int &idx1) {
  float best0 = -1.0f;
  float best1 = -1.0f;
  idx0 = 0;
  idx1 = 1;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    float score = fp_max(fp_abs(vals[2 * i]), fp_abs(vals[2 * i + 1]));
    if (score > best0) {
      best1 = best0;
      idx1 = idx0;
      best0 = score;
      idx0 = i;
    }
    else if (score > best1) {
      best1 = score;
      idx1 = i;
    }
  }
  if (idx1 < idx0) {
    int tmp = idx0;
    idx0 = idx1;
    idx1 = tmp;
  }
}

__global__ void fused_rmsnorm_row_stats_kernel(
    const bf16_t *input,
    const bf16_t *rmsnorm_weight,
    float rmsnorm_eps,
    int seq_len,
    int hidden_dim,
    float *inv_rms,
    float *row_absmax) {
  __shared__ float shared_sum_sq[kMaxGroupsPerRow];
  __shared__ float shared_max_abs[kMaxGroupsPerRow];
  __shared__ float shared_inv_rms;

  int row_id = blockIdx.x;
  int group_id = threadIdx.x;
  int groups_per_row = hidden_dim / kSparseElementsPerThread;

  if (row_id >= seq_len || group_id >= groups_per_row) {
    return;
  }

  cutlass::NumericConverter<bf16_t, float, cutlass::FloatRoundStyle::round_to_nearest> float_to_bf16;

  int group_base_k = group_id * kSparseElementsPerThread;
  const bf16_t *row_ptr = input + static_cast<size_t>(row_id) * hidden_dim + group_base_k;
  const bf16_t *weight_ptr = rmsnorm_weight + group_base_k;

  float sum_sq = 0.0f;
  #pragma unroll
  for (int i = 0; i < kSparseElementsPerThread; ++i) {
    float v = static_cast<float>(row_ptr[i]);
    sum_sq += v * v;
  }
  shared_sum_sq[group_id] = sum_sq;
  __syncthreads();

  if (group_id == 0) {
    float total_sum_sq = 0.0f;
    for (int i = 0; i < groups_per_row; ++i) {
      total_sum_sq += shared_sum_sq[i];
    }
    shared_inv_rms = rsqrtf(total_sum_sq / static_cast<float>(hidden_dim) + rmsnorm_eps);
    inv_rms[row_id] = shared_inv_rms;
  }
  __syncthreads();

  float local_max = 0.0f;
  float row_inv_rms = shared_inv_rms;
  #pragma unroll
  for (int i = 0; i < kSparseElementsPerThread; ++i) {
    float y = __fmul_rn(static_cast<float>(row_ptr[i]), row_inv_rms);
    y = __fmul_rn(y, static_cast<float>(weight_ptr[i]));
    bf16_t y_bf16 = float_to_bf16(y);
    local_max = fp_max(local_max, fp_abs(static_cast<float>(y_bf16)));
  }
  shared_max_abs[group_id] = local_max;
  __syncthreads();

  if (group_id == 0) {
    float row_max = 0.0f;
    for (int i = 0; i < groups_per_row; ++i) {
      row_max = fp_max(row_max, shared_max_abs[i]);
    }
    row_absmax[row_id] = row_max;
  }
}

template <typename SparseScaleTensor, typename ResidualScaleTensor>
__global__ void fused_rmsnorm_sparse_residual_x_kernel(
    const bf16_t *input,
    const bf16_t *rmsnorm_weight,
    const float *inv_rms_ptr,
    float input_scale,
    uint8_t *a_comp,
    uint8_t *e,
    uint8_t *q_sparse_dense,
    SparseScaleTensor sparse_scale_tensor,
    uint8_t *q_residual,
    ResidualScaleTensor residual_scale_tensor,
    int seq_len,
    int hidden_dim,
    int dense_row_bytes,
    int compressed_row_bytes,
    int aligned_seq_len) {
  int row_id = blockIdx.x;
  int group_id = threadIdx.x;
  int groups_per_row = hidden_dim / kSparseElementsPerThread;

  if (row_id >= seq_len || group_id >= groups_per_row) {
    return;
  }

  int group_base_k = group_id * kSparseElementsPerThread;
  const bf16_t *row_ptr = input + static_cast<size_t>(row_id) * hidden_dim + group_base_k;
  const bf16_t *weight_ptr = rmsnorm_weight + group_base_k;

  cutlass::NumericConverter<fp4_t, float, cutlass::FloatRoundStyle::round_to_nearest> float_to_fp4;
  cutlass::NumericConverter<float, fp4_t, cutlass::FloatRoundStyle::round_to_nearest> fp4_to_float;
  cutlass::NumericConverter<sf_t, float, cutlass::FloatRoundStyle::round_to_nearest> float_to_sf;
  cutlass::NumericConverter<float, sf_t, cutlass::FloatRoundStyle::round_to_nearest> sf_to_float;
  cutlass::NumericConverter<bf16_t, float, cutlass::FloatRoundStyle::round_to_nearest> float_to_bf16;

  float x[kSparseElementsPerThread];
  bool keep[kSparseElementsPerThread];
  float residual[kSparseElementsPerThread];
  uint8_t sparse_nibbles[kSparseElementsPerThread];
  uint8_t residual_packed[kSparseElementsPerThread / 2];
  int selected_pair_idx0[kSparseElementsPerThread / 8];
  int selected_pair_idx1[kSparseElementsPerThread / 8];

  float inv_rms = inv_rms_ptr[row_id];
  float input_scale_clamped = fp_max(input_scale, 1.0e-9f);

  #pragma unroll
  for (int i = 0; i < kSparseElementsPerThread; ++i) {
    float y = __fmul_rn(static_cast<float>(row_ptr[i]), inv_rms);
    y = __fmul_rn(y, static_cast<float>(weight_ptr[i]));
    bf16_t y_bf16 = float_to_bf16(y);
    float scaled = __fdiv_rn(static_cast<float>(y_bf16), input_scale_clamped);
    bf16_t scaled_bf16 = float_to_bf16(scaled);
    x[i] = static_cast<float>(scaled_bf16);
    keep[i] = false;
    residual[i] = x[i];
    sparse_nibbles[i] = 0;
  }

  float sparse_max = 0.0f;
  #pragma unroll
  for (int chunk = 0; chunk < kSparseElementsPerThread / 8; ++chunk) {
    int base = chunk * 8;
    int idx0 = 0;
    int idx1 = 1;
    top2_pair_indices_8(x + base, idx0, idx1);
    selected_pair_idx0[chunk] = idx0;
    selected_pair_idx1[chunk] = idx1;
    int pair0_base = base + 2 * idx0;
    int pair1_base = base + 2 * idx1;
    keep[pair0_base + 0] = true;
    keep[pair0_base + 1] = true;
    keep[pair1_base + 0] = true;
    keep[pair1_base + 1] = true;
    sparse_max = fp_max(sparse_max, fp_abs(x[pair0_base + 0]));
    sparse_max = fp_max(sparse_max, fp_abs(x[pair0_base + 1]));
    sparse_max = fp_max(sparse_max, fp_abs(x[pair1_base + 0]));
    sparse_max = fp_max(sparse_max, fp_abs(x[pair1_base + 1]));
  }

  float sparse_scale_f = clamp_val(sparse_max / kFp4Max, kScaleEps, kFp8Max);
  sf_t sparse_scale = float_to_sf(sparse_scale_f);
  float sparse_scale_quant = sf_to_float(sparse_scale);
  float sparse_rscale = 1.0f / sparse_scale_quant;

  #pragma unroll
  for (int i = 0; i < kSparseElementsPerThread; ++i) {
    if (!keep[i]) {
      continue;
    }
    float q = clamp_val(x[i] * sparse_rscale, -kFp4Max, kFp4Max);
    fp4_t q_fp4 = float_to_fp4(q);
    sparse_nibbles[i] = static_cast<uint8_t>(q_fp4.storage & 0xF);
    residual[i] = x[i] - fp4_to_float(q_fp4) * sparse_scale_quant;
  }

  PackFp4 *residual_pack = reinterpret_cast<PackFp4 *>(residual_packed);
  sf_t residual_scales[2];

  #pragma unroll
  for (int group16 = 0; group16 < 2; ++group16) {
    int base16 = group16 * kResidualElementsPerGroup;
    float residual_max = 0.0f;
    #pragma unroll
    for (int i = 0; i < kResidualElementsPerGroup; ++i) {
      residual_max = fp_max(residual_max, fp_abs(residual[base16 + i]));
    }

    float residual_scale_f = clamp_val(residual_max / kFp4Max, kScaleEps, kFp8Max);
    residual_scales[group16] = float_to_sf(residual_scale_f);
    float residual_scale_quant = sf_to_float(residual_scales[group16]);
    float residual_rscale = 1.0f / residual_scale_quant;

    #pragma unroll
    for (int i = 0; i < kResidualElementsPerGroup; i += 2) {
      float q0 = clamp_val(residual[base16 + i + 0] * residual_rscale, -kFp4Max, kFp4Max);
      float q1 = clamp_val(residual[base16 + i + 1] * residual_rscale, -kFp4Max, kFp4Max);
      fp4_t fp4_0 = float_to_fp4(q0);
      fp4_t fp4_1 = float_to_fp4(q1);
      residual_pack[(base16 + i) / 2].low = static_cast<int8_t>(fp4_0.storage & 0xF);
      residual_pack[(base16 + i) / 2].high = static_cast<int8_t>(fp4_1.storage & 0xF);
    }
  }

  store_scale(sparse_scale_tensor, row_id, group_id, sparse_scale);
  store_scale(residual_scale_tensor, row_id, 2 * group_id, residual_scales[0]);
  store_scale(residual_scale_tensor, row_id, 2 * group_id + 1, residual_scales[1]);

  if (q_sparse_dense != nullptr) {
    uint8_t *q_sparse_ptr = q_sparse_dense + static_cast<size_t>(row_id) * dense_row_bytes +
                            group_id * (kSparseElementsPerThread / 2);
    #pragma unroll
    for (int i = 0; i < kSparseElementsPerThread / 2; ++i) {
      q_sparse_ptr[i] = sparse_nibbles[2 * i] |
                        static_cast<uint8_t>(sparse_nibbles[2 * i + 1] << 4);
    }
  }

  if (a_comp != nullptr && e != nullptr) {
    uint8_t *a_comp_ptr = a_comp + static_cast<size_t>(row_id) * compressed_row_bytes +
                          group_id * kCompressedSparseBytesPerThread;
    int metadata_tile_k = group_id / kMetadataGroupsPerTile;
    int metadata_group_in_tile = group_id % kMetadataGroupsPerTile;
    uint8_t *e_ptr = e + static_cast<size_t>(metadata_tile_k) * aligned_seq_len * kMetadataTileRowBytes +
                     static_cast<size_t>(row_id) * kMetadataTileRowBytes +
                     metadata_group_in_tile * kMetadataBytesPerThread;
    uint8_t metadata_nibbles[kSparseElementsPerThread / 8];

    #pragma unroll
    for (int chunk = 0; chunk < kSparseElementsPerThread / 8; ++chunk) {
      int base = chunk * 8;
      uint8_t dense_bytes[4];
      #pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        dense_bytes[pair] = sparse_nibbles[base + 2 * pair] |
                            static_cast<uint8_t>(sparse_nibbles[base + 2 * pair + 1] << 4);
      }
      int idx0 = selected_pair_idx0[chunk];
      int idx1 = selected_pair_idx1[chunk];
      a_comp_ptr[2 * chunk + 0] = dense_bytes[idx0];
      a_comp_ptr[2 * chunk + 1] = dense_bytes[idx1];
      metadata_nibbles[chunk] = static_cast<uint8_t>(idx0 | (idx1 << 2));
    }

    e_ptr[0] = static_cast<uint8_t>(metadata_nibbles[0] | (metadata_nibbles[1] << 4));
    e_ptr[1] = static_cast<uint8_t>(metadata_nibbles[2] | (metadata_nibbles[3] << 4));
  }

  uint8_t *q_res_ptr = q_residual + static_cast<size_t>(row_id) * dense_row_bytes +
                       group_id * (kSparseElementsPerThread / 2);
  #pragma unroll
  for (int i = 0; i < kSparseElementsPerThread / 2; ++i) {
    q_res_ptr[i] = residual_packed[i];
  }
}

}  // namespace fused_rmsnorm_sparse_prepare

void run_fused_rmsnorm_row_stats_bf16(
    bf16_t *hidden_states,
    bf16_t *rmsnorm_weight,
    float rmsnorm_eps,
    int seq_len,
    int hidden_dim,
    float *inv_rms,
    float *row_absmax) {
  int groups_per_row = hidden_dim / fused_rmsnorm_sparse_prepare::kSparseElementsPerThread;
  dim3 grid(seq_len);
  dim3 block(groups_per_row);
  fused_rmsnorm_sparse_prepare::fused_rmsnorm_row_stats_kernel<<<grid, block>>>(
      hidden_states,
      rmsnorm_weight,
      rmsnorm_eps,
      seq_len,
      hidden_dim,
      inv_rms,
      row_absmax);
}

void run_fused_rmsnorm_sparse_residual_x_bf16_nvfp4(
    bf16_t *hidden_states,
    bf16_t *rmsnorm_weight,
    float *inv_rms,
    float input_scale,
    int seq_len,
    int out_features,
    int hidden_dim,
    uint8_t *a_comp,
    uint8_t *e,
    uint8_t *q_sparse_dense,
    sf_t *sf_sparse,
    uint8_t *q_residual,
    sf_t *sf_residual) {
  int groups_per_row = hidden_dim / fused_rmsnorm_sparse_prepare::kSparseElementsPerThread;
  int dense_row_bytes = hidden_dim / 2;
  int compressed_row_bytes = hidden_dim / 4;
  int aligned_seq_len = fused_rmsnorm_sparse_prepare::div_up_int(seq_len, 128) * 128;

  dim3 grid(seq_len);
  dim3 block(groups_per_row);

  auto sparse_scale_tensor =
      cute::make_tensor(sf_sparse, filter_zeros(sparse_nvfp4::get_layoutSFA(seq_len, out_features, hidden_dim)));
  auto residual_scale_tensor =
      cute::make_tensor(sf_residual, filter_zeros(nvfp4::get_layoutSFA(seq_len, hidden_dim)));

  fused_rmsnorm_sparse_prepare::fused_rmsnorm_sparse_residual_x_kernel<<<grid, block>>>(
      hidden_states,
      rmsnorm_weight,
      inv_rms,
      input_scale,
      a_comp,
      e,
      q_sparse_dense,
      sparse_scale_tensor,
      q_residual,
      residual_scale_tensor,
      seq_len,
      hidden_dim,
      dense_row_bytes,
      compressed_row_bytes,
      aligned_seq_len);
}
