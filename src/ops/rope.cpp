// Citing vendor documentation per AGENTS.md §5:
// - docs/vendor/xe-gpu-architecture.md (lines 48-50):
//   Vector Engine (VE) ALUs support native FP16 and FP32 operations.

#include "rope.h"
#include <cmath>

namespace xinfer::ops {

namespace {

template <typename T>
sycl::event rope_impl(sycl::queue& q,
                      T* q_ptr,
                      T* k_ptr,
                      int64_t num_tokens,
                      int64_t num_q_heads,
                      int64_t num_kv_heads,
                      int64_t head_dim,
                      const int64_t* positions,
                      float theta,
                      int64_t rotary_dim) {
    if (num_tokens <= 0 || head_dim <= 0) return sycl::event{};
    if (rotary_dim <= 0 || rotary_dim > head_dim) rotary_dim = head_dim;
    int64_t half_rot = rotary_dim / 2;

    float log_theta = sycl::log(theta);
    float inv_rot = 1.0f / static_cast<float>(rotary_dim);

    size_t q_heads = (q_ptr && num_q_heads > 0) ? static_cast<size_t>(num_tokens * num_q_heads) : 0;
    size_t kv_heads = (k_ptr && num_kv_heads > 0) ? static_cast<size_t>(num_tokens * num_kv_heads) : 0;
    size_t total_heads = q_heads + kv_heads;
    if (total_heads == 0) return sycl::event{};

    return q.parallel_for(sycl::range<2>(total_heads, static_cast<size_t>(half_rot)), [=](sycl::id<2> idx) {
        size_t head_linear = idx[0];
        size_t pair_idx = idx[1];

        bool is_q = (head_linear < q_heads);
        T* target_ptr = is_q ? q_ptr : k_ptr;
        size_t eff_heads = is_q ? static_cast<size_t>(num_q_heads) : static_cast<size_t>(num_kv_heads);
        size_t eff_head_idx = is_q ? head_linear : (head_linear - q_heads);

        size_t token_idx = eff_head_idx / eff_heads;
        int64_t pos = positions[token_idx];

        float freq = sycl::exp(-static_cast<float>(2 * pair_idx) * inv_rot * log_theta);
        float angle = static_cast<float>(pos) * freq;
        float cos_val = sycl::cos(angle);
        float sin_val = sycl::sin(angle);

        size_t base_offset = eff_head_idx * static_cast<size_t>(head_dim);
        size_t idx0 = base_offset + pair_idx;
        size_t idx1 = base_offset + pair_idx + static_cast<size_t>(half_rot);

        float v0 = static_cast<float>(target_ptr[idx0]);
        float v1 = static_cast<float>(target_ptr[idx1]);

        target_ptr[idx0] = static_cast<T>(v0 * cos_val - v1 * sin_val);
        target_ptr[idx1] = static_cast<T>(v1 * cos_val + v0 * sin_val);
    });
}

} // anonymous namespace

sycl::event rope(sycl::queue& q,
                 float* q_ptr,
                 float* k_ptr,
                 int64_t num_tokens,
                 int64_t num_q_heads,
                 int64_t num_kv_heads,
                 int64_t head_dim,
                 const int64_t* positions,
                 float theta,
                 int64_t rotary_dim) {
    return rope_impl<float>(q, q_ptr, k_ptr, num_tokens, num_q_heads, num_kv_heads, head_dim, positions, theta, rotary_dim);
}

sycl::event rope(sycl::queue& q,
                 sycl::half* q_ptr,
                 sycl::half* k_ptr,
                 int64_t num_tokens,
                 int64_t num_q_heads,
                 int64_t num_kv_heads,
                 int64_t head_dim,
                 const int64_t* positions,
                 float theta,
                 int64_t rotary_dim) {
    return rope_impl<sycl::half>(q, q_ptr, k_ptr, num_tokens, num_q_heads, num_kv_heads, head_dim, positions, theta, rotary_dim);
}

} // namespace xinfer::ops
