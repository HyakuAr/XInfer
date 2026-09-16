#include "attention.h"
#include <cmath>
#include <limits>

namespace xinfer::ops {

void sdpa_causal_naive(sycl::queue& q,
                       float* out,
                       const float* Q,
                       const float* K,
                       const float* V,
                       int64_t seq_len,
                       int64_t num_q_heads,
                       int64_t num_kv_heads,
                       int64_t head_dim,
                       float scale) {
    if (seq_len <= 0 || num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) return;

    if (scale <= 0.0f) {
        scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
    }

    int64_t gqa_ratio = num_q_heads / num_kv_heads;

    // Launch 1 work-item per (seq_pos, q_head)
    q.parallel_for(sycl::range<2>(static_cast<size_t>(seq_len), static_cast<size_t>(num_q_heads)), [=](sycl::id<2> idx) {
        int64_t i = idx[0]; // Target token position
        int64_t h = idx[1]; // Query head index

        int64_t kv_h = h / gqa_ratio;
        const float* q_vec = Q + (i * num_q_heads + h) * head_dim;
        float* out_vec = out + (i * num_q_heads + h) * head_dim;

        // Numerically stable online softmax (FlashAttention formulation)
        // Eliminates need for O(seq_len) intermediate storage
        float max_score = -std::numeric_limits<float>::infinity();
        float sum_exp = 0.0f;

        // Initialize output accumulator to 0
        for (int64_t d = 0; d < head_dim; ++d) {
            out_vec[d] = 0.0f;
        }

        // Causal attention: attend only to positions j <= i
        for (int64_t j = 0; j <= i; ++j) {
            const float* k_vec = K + (j * num_kv_heads + kv_h) * head_dim;
            const float* v_vec = V + (j * num_kv_heads + kv_h) * head_dim;

            // Dot product Q_i . K_j
            float dot = 0.0f;
            for (int64_t d = 0; d < head_dim; ++d) {
                dot += q_vec[d] * k_vec[d];
            }
            float score = dot * scale;

            // Online update
            float new_max = sycl::max(max_score, score);
            float exp_old = (max_score == -std::numeric_limits<float>::infinity()) ? 0.0f : sycl::exp(max_score - new_max);
            float exp_new = sycl::exp(score - new_max);

            sum_exp = sum_exp * exp_old + exp_new;
            for (int64_t d = 0; d < head_dim; ++d) {
                out_vec[d] = out_vec[d] * exp_old + exp_new * v_vec[d];
            }
            max_score = new_max;
        }

        // Final normalization
        float inv_sum = 1.0f / (sum_exp > 0.0f ? sum_exp : 1.0f);
        for (int64_t d = 0; d < head_dim; ++d) {
            out_vec[d] *= inv_sum;
        }
    });
}

void attention_write_kv_cache(sycl::queue& q,
                              sycl::half* k_cache,
                              sycl::half* v_cache,
                              const float* K_in,
                              const float* V_in,
                              int64_t start_pos,
                              int64_t num_tokens,
                              int64_t num_kv_heads,
                              int64_t head_dim) {
    if (num_tokens <= 0 || num_kv_heads <= 0 || head_dim <= 0) return;
    int64_t kv_stride = num_kv_heads * head_dim;
    size_t total_elements = static_cast<size_t>(num_tokens * kv_stride);

    q.parallel_for(sycl::range<1>(total_elements), [=](sycl::id<1> idx) {
        size_t i = idx[0];
        size_t t = i / kv_stride;
        size_t rem = i % kv_stride;
        size_t cache_idx = (static_cast<size_t>(start_pos) + t) * kv_stride + rem;

        k_cache[cache_idx] = static_cast<sycl::half>(K_in[i]);
        v_cache[cache_idx] = static_cast<sycl::half>(V_in[i]);
    });
}

void sdpa_causal_cached(sycl::queue& q,
                        float* out,
                        const float* Q,
                        const sycl::half* k_cache,
                        const sycl::half* v_cache,
                        int64_t start_pos,
                        int64_t num_q_tokens,
                        int64_t num_q_heads,
                        int64_t num_kv_heads,
                        int64_t head_dim,
                        float scale) {
    if (num_q_tokens <= 0 || num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) return;

    if (scale <= 0.0f) {
        scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
    }

    int64_t gqa_ratio = num_q_heads / num_kv_heads;

    q.parallel_for(sycl::range<2>(static_cast<size_t>(num_q_tokens), static_cast<size_t>(num_q_heads)), [=](sycl::id<2> idx) {
        int64_t i = idx[0]; // Query token index within Q
        int64_t h = idx[1]; // Query head index

        int64_t kv_h = h / gqa_ratio;
        const float* q_vec = Q + (i * num_q_heads + h) * head_dim;
        float* out_vec = out + (i * num_q_heads + h) * head_dim;

        int64_t total_keys = start_pos + i + 1;

        float max_score = -std::numeric_limits<float>::infinity();
        float sum_exp = 0.0f;

        for (int64_t d = 0; d < head_dim; ++d) {
            out_vec[d] = 0.0f;
        }

        for (int64_t j = 0; j < total_keys; ++j) {
            const sycl::half* k_ptr = k_cache + (j * num_kv_heads + kv_h) * head_dim;
            const sycl::half* v_ptr = v_cache + (j * num_kv_heads + kv_h) * head_dim;

            float dot = 0.0f;
            for (int64_t d = 0; d < head_dim; ++d) {
                dot += q_vec[d] * static_cast<float>(k_ptr[d]);
            }
            float score = dot * scale;

            float new_max = sycl::max(max_score, score);
            float exp_old = (max_score == -std::numeric_limits<float>::infinity()) ? 0.0f : sycl::exp(max_score - new_max);
            float exp_new = sycl::exp(score - new_max);

            sum_exp = sum_exp * exp_old + exp_new;
            for (int64_t d = 0; d < head_dim; ++d) {
                out_vec[d] = out_vec[d] * exp_old + exp_new * static_cast<float>(v_ptr[d]);
            }
            max_score = new_max;
        }

        float inv_sum = 1.0f / (sum_exp > 0.0f ? sum_exp : 1.0f);
        for (int64_t d = 0; d < head_dim; ++d) {
            out_vec[d] *= inv_sum;
        }
    });
}

} // namespace xinfer::ops
