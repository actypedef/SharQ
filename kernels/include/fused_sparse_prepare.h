#pragma once

#include "reorder.cuh"
#include "sparse_nvfp4.h"

void run_fused_sparse_residual_x_bf16_nvfp4(
    bf16_t *hidden_states,
    int seq_len,
    int out_features,
    int hidden_dim,
    uint8_t *a_comp,
    uint8_t *e,
    uint8_t *q_sparse_dense,
    sf_t *sf_sparse,
    uint8_t *q_residual,
    sf_t *sf_residual);
