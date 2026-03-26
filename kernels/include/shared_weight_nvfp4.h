#pragma once

#include "reorder.cuh"

void run_quantize_w32_bf16_nvfp4_shared(
    bf16_t* weights,
    int out_features,
    int hidden_dim,
    uint8_t* q_out,
    sf_t* sparse_scale,
    sf_t* dense_scale_dup);

void run_duplicate_sfa32_to_sfa16(
    sf_t* sparse_scale,
    int seq_len,
    int out_features,
    int hidden_dim,
    sf_t* dense_scale_dup);
