#include "shared_weight_nvfp4.h"

#include "sparse_nvfp4.h"

using namespace cute;

struct PackFp4 {
  uint8_t low : 4, high : 4;
};

constexpr float kFp4Max = 6.0f;
constexpr float kFp8Max = 448.0f;
constexpr float kScaleEps = 0.001953125f;

__device__ __forceinline__ float clamp_val(float x, float lo, float hi) {
  return fminf(fmaxf(x, lo), hi);
}

template <class SparseScaleTensor>
__global__ void quantize_w32_shared_kernel(
    bf16_t* input,
    int hidden_dim,
    uint8_t* q_out,
    SparseScaleTensor sparse_scale_tensor) {
  constexpr int kGroupSize = 32;

  int row_id = blockIdx.x;
  int group_id = threadIdx.x;
  int groups_per_row = hidden_dim / kGroupSize;
  if (group_id >= groups_per_row) {
    return;
  }

  input += row_id * hidden_dim;
  q_out += row_id * (hidden_dim / 2) + group_id * (kGroupSize / 2);

  bf16_t input_frag[kGroupSize];
  uint8_t output_frag_packed[kGroupSize / 2];

  #pragma unroll
  for (int i = 0; i < kGroupSize; ++i) {
    input_frag[i] = input[group_id * kGroupSize + i];
  }

  cutlass::NumericConverter<fp4_t, float, cutlass::FloatRoundStyle::round_to_nearest> float_to_e2m1;
  cutlass::NumericConverter<sf_t, float, cutlass::FloatRoundStyle::round_to_nearest> float_to_ue4m3;
  cutlass::NumericConverter<float, sf_t, cutlass::FloatRoundStyle::round_to_nearest> ue4m3_to_float;

  float maxv = 0.0f;
  #pragma unroll
  for (int i = 0; i < kGroupSize; ++i) {
    maxv = fmaxf(maxv, fabsf(static_cast<float>(input_frag[i])));
  }

  float scale = clamp_val(maxv / kFp4Max, kScaleEps, kFp8Max);
  float inverse_scale = 1.0f / ue4m3_to_float(float_to_ue4m3(scale));

  auto logical_coord0 = make_coord(make_coord(row_id % 32, (row_id / 32) % 4), row_id / 128);
  auto logical_coord1 = make_coord(make_coord(0, group_id % 4), group_id / 4);
  auto logical_coord2 = make_coord(0, 0);
  sparse_scale_tensor(make_coord(logical_coord0, logical_coord1, logical_coord2)) = float_to_ue4m3(scale);

  PackFp4* output_frag_fp4 = reinterpret_cast<PackFp4*>(output_frag_packed);
  constexpr float lower_bound = -kFp4Max;
  constexpr float upper_bound = kFp4Max;

  #pragma unroll
  for (int i = 0; i < kGroupSize; i += 4) {
    float r0 = clamp_val(static_cast<float>(input_frag[i + 0]) * inverse_scale, lower_bound, upper_bound);
    float r1 = clamp_val(static_cast<float>(input_frag[i + 1]) * inverse_scale, lower_bound, upper_bound);
    float r2 = clamp_val(static_cast<float>(input_frag[i + 2]) * inverse_scale, lower_bound, upper_bound);
    float r3 = clamp_val(static_cast<float>(input_frag[i + 3]) * inverse_scale, lower_bound, upper_bound);
    output_frag_fp4[i / 2 + 0].low = float_to_e2m1(r0).storage;
    output_frag_fp4[i / 2 + 0].high = float_to_e2m1(r1).storage;
    output_frag_fp4[i / 2 + 1].low = float_to_e2m1(r2).storage;
    output_frag_fp4[i / 2 + 1].high = float_to_e2m1(r3).storage;
  }

  *(reinterpret_cast<ulonglong2*>(q_out)) = *(reinterpret_cast<ulonglong2*>(output_frag_packed));
}

template <class SparseScaleTensor, class DenseScaleTensor>
__global__ void duplicate_sfb32_to_dense_sfb16_kernel(
    int out_features,
    int groups32_per_row,
    SparseScaleTensor sparse_scale_tensor,
    DenseScaleTensor dense_scale_tensor) {
  int row_id = blockIdx.x;
  int group32_id = threadIdx.x;
  if (row_id >= out_features || group32_id >= groups32_per_row) {
    return;
  }

  auto logical_coord0 = make_coord(make_coord(row_id % 32, (row_id / 32) % 4), row_id / 128);
  auto logical_coord2 = make_coord(0, 0);

  auto sparse_coord1 = make_coord(make_coord(0, group32_id % 4), group32_id / 4);
  sf_t scale = sparse_scale_tensor(make_coord(logical_coord0, sparse_coord1, logical_coord2));

  int dense_group0 = 2 * group32_id;
  int dense_group1 = dense_group0 + 1;

  auto dense_coord1_0 = make_coord(make_coord(0, dense_group0 % 4), dense_group0 / 4);
  auto dense_coord1_1 = make_coord(make_coord(0, dense_group1 % 4), dense_group1 / 4);
  dense_scale_tensor(make_coord(logical_coord0, dense_coord1_0, logical_coord2)) = scale;
  dense_scale_tensor(make_coord(logical_coord0, dense_coord1_1, logical_coord2)) = scale;
}

void run_quantize_w32_bf16_nvfp4_shared(
    bf16_t* weights,
    int out_features,
    int hidden_dim,
    uint8_t* q_out,
    sf_t* sparse_scale,
    sf_t* dense_scale_dup) {
  constexpr int kGroupSize = 32;
  if (hidden_dim % kGroupSize != 0) {
    throw std::runtime_error("run_quantize_w32_bf16_nvfp4_shared requires hidden_dim to be a multiple of 32");
  }

  int groups32_per_row = hidden_dim / kGroupSize;
  if (groups32_per_row > 1024) {
    throw std::runtime_error("run_quantize_w32_bf16_nvfp4_shared requires hidden_dim / 32 <= 1024");
  }

  auto sparse_scale_tensor =
      make_tensor(sparse_scale, filter_zeros(sparse_nvfp4::get_layoutSFB(1, out_features, hidden_dim)));
  auto dense_scale_tensor =
      make_tensor(dense_scale_dup, filter_zeros(nvfp4::get_layoutSFB(out_features, hidden_dim)));

  dim3 grids(out_features);
  dim3 blocks(groups32_per_row);
  quantize_w32_shared_kernel<<<grids, blocks>>>(weights, hidden_dim, q_out, sparse_scale_tensor);
  duplicate_sfb32_to_dense_sfb16_kernel<<<grids, blocks>>>(
      out_features, groups32_per_row, sparse_scale_tensor, dense_scale_tensor);
}

template <class SparseScaleTensor, class DenseScaleTensor>
__global__ void duplicate_sfa32_to_dense_sfa16_kernel(
    int seq_len,
    int groups32_per_row,
    SparseScaleTensor sparse_scale_tensor,
    DenseScaleTensor dense_scale_tensor) {
  int row_id = blockIdx.x;
  int group32_id = threadIdx.x;
  if (row_id >= seq_len || group32_id >= groups32_per_row) {
    return;
  }

  auto logical_coord0 = make_coord(make_coord(row_id % 32, (row_id / 32) % 4), row_id / 128);
  auto logical_coord2 = make_coord(0, 0);

  auto sparse_coord1 = make_coord(make_coord(0, group32_id % 4), group32_id / 4);
  sf_t scale = sparse_scale_tensor(make_coord(logical_coord0, sparse_coord1, logical_coord2));

  int dense_group0 = 2 * group32_id;
  int dense_group1 = dense_group0 + 1;
  auto dense_coord1_0 = make_coord(make_coord(0, dense_group0 % 4), dense_group0 / 4);
  auto dense_coord1_1 = make_coord(make_coord(0, dense_group1 % 4), dense_group1 / 4);
  dense_scale_tensor(make_coord(logical_coord0, dense_coord1_0, logical_coord2)) = scale;
  dense_scale_tensor(make_coord(logical_coord0, dense_coord1_1, logical_coord2)) = scale;
}

void run_duplicate_sfa32_to_sfa16(
    sf_t* sparse_scale,
    int seq_len,
    int out_features,
    int hidden_dim,
    sf_t* dense_scale_dup) {
  constexpr int kGroupSize = 32;
  if (hidden_dim % kGroupSize != 0) {
    throw std::runtime_error("run_duplicate_sfa32_to_sfa16 requires hidden_dim to be a multiple of 32");
  }

  int groups32_per_row = hidden_dim / kGroupSize;
  if (groups32_per_row > 1024) {
    throw std::runtime_error("run_duplicate_sfa32_to_sfa16 requires hidden_dim / 32 <= 1024");
  }

  auto sparse_scale_tensor =
      make_tensor(sparse_scale, filter_zeros(sparse_nvfp4::get_layoutSFA(seq_len, out_features, hidden_dim)));
  auto dense_scale_tensor =
      make_tensor(dense_scale_dup, filter_zeros(nvfp4::get_layoutSFA(seq_len, hidden_dim)));

  dim3 grids(seq_len);
  dim3 blocks(groups32_per_row);
  duplicate_sfa32_to_dense_sfa16_kernel<<<grids, blocks>>>(
      seq_len, groups32_per_row, sparse_scale_tensor, dense_scale_tensor);
}
