#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

namespace xinfer::ops {

// In-place Rotary Position Embedding (RoPE) applied to query and key tensors
// Shapes:
//   Q: [num_tokens, num_q_heads, head_dim]
//   K: [num_tokens, num_kv_heads, head_dim]
//   positions: [num_tokens] (token sequence indices)
sycl::event rope(sycl::queue& q,
                 float* q_ptr,
                 float* k_ptr,
                 int64_t num_tokens,
                 int64_t num_q_heads,
                 int64_t num_kv_heads,
                 int64_t head_dim,
                 const int64_t* positions,
                 float theta = 10000000.0f,
                 int64_t rotary_dim = 64);

} // namespace xinfer::ops
