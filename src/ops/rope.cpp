#include "rope.h"
#include <cmath>

namespace xinfer::ops {

void rope(sycl::queue& q,
          float* q_ptr,
          float* k_ptr,
          int64_t num_tokens,
          int64_t num_q_heads,
          int64_t num_kv_heads,
          int64_t head_dim,
          const int64_t* positions,
          float theta,
          int64_t rotary_dim) {
    if (num_tokens <= 0 || head_dim <= 0) return;
    if (rotary_dim <= 0 || rotary_dim > head_dim) rotary_dim = head_dim;
    int64_t half_rot = rotary_dim / 2;

    // Apply to Q heads
    if (q_ptr && num_q_heads > 0) {
        int64_t total_pairs = num_tokens * num_q_heads * half_rot;
        q.parallel_for(sycl::range<1>(static_cast<size_t>(total_pairs)), [=](sycl::id<1> idx) {
            int64_t global_idx = idx[0];
            int64_t pair_idx = global_idx % half_rot;
            int64_t rem = global_idx / half_rot;
            int64_t head_idx = rem % num_q_heads;
            int64_t token_idx = rem / num_q_heads;

            int64_t pos = positions[token_idx];
            float freq = 1.0f / sycl::pow(theta, static_cast<float>(2 * pair_idx) / static_cast<float>(rotary_dim));
            float angle = static_cast<float>(pos) * freq;
            float cos_val = sycl::cos(angle);
            float sin_val = sycl::sin(angle);

            int64_t base_offset = (token_idx * num_q_heads + head_idx) * head_dim;
            int64_t idx0 = base_offset + pair_idx;
            int64_t idx1 = base_offset + pair_idx + half_rot;

            float v0 = q_ptr[idx0];
            float v1 = q_ptr[idx1];

            q_ptr[idx0] = v0 * cos_val - v1 * sin_val;
            q_ptr[idx1] = v1 * cos_val + v0 * sin_val;
        });
    }

    // Apply to K heads
    if (k_ptr && num_kv_heads > 0) {
        int64_t total_pairs = num_tokens * num_kv_heads * half_rot;
        q.parallel_for(sycl::range<1>(static_cast<size_t>(total_pairs)), [=](sycl::id<1> idx) {
            int64_t global_idx = idx[0];
            int64_t pair_idx = global_idx % half_rot;
            int64_t rem = global_idx / half_rot;
            int64_t head_idx = rem % num_kv_heads;
            int64_t token_idx = rem / num_kv_heads;

            int64_t pos = positions[token_idx];
            float freq = 1.0f / sycl::pow(theta, static_cast<float>(2 * pair_idx) / static_cast<float>(rotary_dim));
            float angle = static_cast<float>(pos) * freq;
            float cos_val = sycl::cos(angle);
            float sin_val = sycl::sin(angle);

            int64_t base_offset = (token_idx * num_kv_heads + head_idx) * head_dim;
            int64_t idx0 = base_offset + pair_idx;
            int64_t idx1 = base_offset + pair_idx + half_rot;

            float v0 = k_ptr[idx0];
            float v1 = k_ptr[idx1];

            k_ptr[idx0] = v0 * cos_val - v1 * sin_val;
            k_ptr[idx1] = v1 * cos_val + v0 * sin_val;
        });
    }
}

} // namespace xinfer::ops
