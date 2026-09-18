#pragma once

#include "targets/qwen3_8_27b/weights.h"
#include <sycl/sycl.hpp>
#include <cstdint>

namespace xinfer::targets::qwen3_8 {

// Causal 1D Depthwise Convolution with kernel size 4 and SiLU activation
// in_qkv:  [seq_len, num_channels] (row-major)
// conv_w:  [num_channels, 4] (weight per channel, k=0..3)
// out_qkv: [seq_len, num_channels]
// conv_state: optional [3, num_channels] past timesteps buffer (if nullptr, assumes zeros for past)
// num_channels: number of channels (defaults to ModelConfig::kDefaultLinearConvChannels = 10240)
sycl::event causal_conv1d_silu(sycl::queue& q,
                               sycl::half* out_qkv,
                               const sycl::half* in_qkv,
                               const float* conv_w,
                               int64_t seq_len,
                               float* conv_state = nullptr,
                               int64_t num_channels = qwen3_8_27b::ModelConfig::kDefaultLinearConvChannels);

sycl::event causal_conv1d_silu(sycl::queue& q,
                               float* out_qkv,
                               const float* in_qkv,
                               const float* conv_w,
                               int64_t seq_len,
                               float* conv_state = nullptr,
                               int64_t num_channels = qwen3_8_27b::ModelConfig::kDefaultLinearConvChannels);

// Recurrent Gated Delta Rule + RMSNormGated:
// qkv: [seq_len, total_channels] (channels: q=0..2047, k=2048..4095, v=4096..10239)
// z:   [seq_len, value_dim]
// b:   [seq_len, num_v_heads]
// a:   [seq_len, num_v_heads]
// A_log:   [num_v_heads]
// dt_bias: [num_v_heads]
// norm_weight: [head_v_dim]
// out: [seq_len, value_dim]
// state_buffer: [num_v_heads, head_k_dim, head_v_dim] state buffer
// zero_state: if true, zeroes state_buffer before processing (e.g. at start of prompt)
sycl::event recurrent_gated_delta_net(sycl::queue& q,
                                      sycl::half* out,
                                      const sycl::half* qkv,
                                      const sycl::half* z,
                                      const sycl::half* b,
                                      const sycl::half* a,
                                      const float* A_log,
                                      const float* dt_bias,
                                      const float* norm_weight,
                                      float* state_buffer,
                                      int64_t seq_len,
                                      bool zero_state = false,
                                      int64_t num_v_heads = qwen3_8_27b::ModelConfig::kDefaultLinearNumVHeads,
                                      int64_t num_k_heads = qwen3_8_27b::ModelConfig::kDefaultLinearNumKHeads,
                                      int64_t head_k_dim = qwen3_8_27b::ModelConfig::kDefaultLinearHeadKDim,
                                      int64_t head_v_dim = qwen3_8_27b::ModelConfig::kDefaultLinearHeadVDim,
                                      int64_t total_channels = qwen3_8_27b::ModelConfig::kDefaultLinearConvChannels,
                                      int64_t value_dim = qwen3_8_27b::ModelConfig::kDefaultLinearZDim);

sycl::event recurrent_gated_delta_net(sycl::queue& q,
                                      float* out,
                                      const float* qkv,
                                      const float* z,
                                      const float* b,
                                      const float* a,
                                      const float* A_log,
                                      const float* dt_bias,
                                      const float* norm_weight,
                                      float* state_buffer,
                                      int64_t seq_len,
                                      bool zero_state = false,
                                      int64_t num_v_heads = qwen3_8_27b::ModelConfig::kDefaultLinearNumVHeads,
                                      int64_t num_k_heads = qwen3_8_27b::ModelConfig::kDefaultLinearNumKHeads,
                                      int64_t head_k_dim = qwen3_8_27b::ModelConfig::kDefaultLinearHeadKDim,
                                      int64_t head_v_dim = qwen3_8_27b::ModelConfig::kDefaultLinearHeadVDim,
                                      int64_t total_channels = qwen3_8_27b::ModelConfig::kDefaultLinearConvChannels,
                                      int64_t value_dim = qwen3_8_27b::ModelConfig::kDefaultLinearZDim);

} // namespace xinfer::targets::qwen3_8
