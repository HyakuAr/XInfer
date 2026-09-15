#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

namespace xinfer::ops {

// Scaled Dot-Product Attention (SDPA) with Grouped-Query Attention (GQA) and Causal Masking
// Q: [seq_len, num_q_heads, head_dim]
// K: [seq_len, num_kv_heads, head_dim]
// V: [seq_len, num_kv_heads, head_dim]
// out: [seq_len, num_q_heads, head_dim]
// scale: default 1.0 / sqrt(head_dim)
void sdpa_causal_naive(sycl::queue& q,
                       float* out,
                       const float* Q,
                       const float* K,
                       const float* V,
                       int64_t seq_len,
                       int64_t num_q_heads,
                       int64_t num_kv_heads,
                       int64_t head_dim,
                       float scale = 0.0f);

} // namespace xinfer::ops
