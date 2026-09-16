#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

namespace xinfer::targets::qwen3_8 {

// Causal 1D Depthwise Convolution with kernel size 4 and SiLU activation
// in_qkv:  [seq_len, 10240] (row-major)
// conv_w:  [10240, 4] (weight per channel, k=0..3)
// out_qkv: [seq_len, 10240]
void causal_conv1d_silu(sycl::queue& q,
                        float* out_qkv,
                        const float* in_qkv,
                        const float* conv_w,
                        int64_t seq_len);

// Recurrent Gated Delta Rule + RMSNormGated:
// qkv: [seq_len, 10240] (channels: q=0..2047, k=2048..4095, v=4096..10239)
// z:   [seq_len, 6144]
// b:   [seq_len, 48]
// a:   [seq_len, 48]
// A_log:   [48]
// dt_bias: [48]
// norm_weight: [128]
// out: [seq_len, 6144]
// state_buffer: [48, 128, 128] scratchpad
void recurrent_gated_delta_net(sycl::queue& q,
                               float* out,
                               const float* qkv,
                               const float* z,
                               const float* b,
                               const float* a,
                               const float* A_log,
                               const float* dt_bias,
                               const float* norm_weight,
                               float* state_buffer,
                               int64_t seq_len);

} // namespace xinfer::targets::qwen3_8
