// Citing vendor documentation per AGENTS.md §5:
// - docs/vendor/xe-gpu-architecture.md (lines 22-39):
//   Intel Arc Pro B60: 20 Xe-cores, 8 Vector Engines per core, 8 HW threads per VE
//   = 64 HW threads per core (1280 total). Sub-group size: 16, 32.
// - docs/vendor/thread-mapping-occupancy.md (lines 9-15):
//   Sub-group size 16 maps to one Vector Engine hardware thread; work-group to Xe-core.
//   sycl::reqd_sub_group_size(16).
// - docs/vendor/xetla-gemm.md (lines 23-51):
//   Subgroup-level reduction across vector dimensions with cooperative SIMD lanes.

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

    // Reference-only naive kernel (1 work-item per (seq_pos, q_head))
    q.parallel_for(sycl::range<2>(static_cast<size_t>(seq_len), static_cast<size_t>(num_q_heads)), [=](sycl::id<2> idx) {
        int64_t i = idx[0]; // Target token position
        int64_t h = idx[1]; // Query head index

        int64_t kv_h = h / gqa_ratio;
        const float* q_vec = Q + (i * num_q_heads + h) * head_dim;
        float* out_vec = out + (i * num_q_heads + h) * head_dim;

        float max_score = -std::numeric_limits<float>::infinity();
        float sum_exp = 0.0f;

        for (int64_t d = 0; d < head_dim; ++d) {
            out_vec[d] = 0.0f;
        }

        for (int64_t j = 0; j <= i; ++j) {
            const float* k_vec = K + (j * num_kv_heads + kv_h) * head_dim;
            const float* v_vec = V + (j * num_kv_heads + kv_h) * head_dim;

            float dot = 0.0f;
            for (int64_t d = 0; d < head_dim; ++d) {
                dot += q_vec[d] * k_vec[d];
            }
            float score = dot * scale;

            float new_max = sycl::max(max_score, score);
            float exp_old = (max_score == -std::numeric_limits<float>::infinity()) ? 0.0f : sycl::exp(max_score - new_max);
            float exp_new = sycl::exp(score - new_max);

            sum_exp = sum_exp * exp_old + exp_new;
            for (int64_t d = 0; d < head_dim; ++d) {
                out_vec[d] = out_vec[d] * exp_old + exp_new * v_vec[d];
            }
            max_score = new_max;
        }

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

sycl::event attention_write_kv_cache_dynamic(sycl::queue& q,
                                       sycl::half* k_cache,
                                       sycl::half* v_cache,
                                       const float* K_in,
                                       const float* V_in,
                                       const int64_t* d_start_pos,
                                       int64_t num_tokens,
                                       int64_t num_kv_heads,
                                       int64_t head_dim) {
    if (num_tokens <= 0 || num_kv_heads <= 0 || head_dim <= 0) return sycl::event{};
    int64_t kv_stride = num_kv_heads * head_dim;
    size_t total_elements = static_cast<size_t>(num_tokens * kv_stride);

    return q.parallel_for(sycl::range<1>(total_elements), [=](sycl::id<1> idx) {
        size_t i = idx[0];
        size_t t = i / kv_stride;
        size_t rem = i % kv_stride;
        int64_t start_pos = *d_start_pos;
        size_t cache_idx = (static_cast<size_t>(start_pos) + t) * kv_stride + rem;

        k_cache[cache_idx] = static_cast<sycl::half>(K_in[i]);
        v_cache[cache_idx] = static_cast<sycl::half>(V_in[i]);
    });
}

// Milestone 7 Hardware-Accelerated Sub-group Cooperative Attention
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

    constexpr size_t SG_SIZE = 16;
    constexpr size_t WG_SIZE = 64; // 4 sub-groups per work-group

    size_t total_subgroups = static_cast<size_t>(num_q_tokens * num_q_heads);
    size_t global_threads = total_subgroups * SG_SIZE;
    size_t padded_global = ((global_threads + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;

    // For head_dim=256, each lane handles 16 elements
    size_t elems_per_lane = static_cast<size_t>(head_dim) / SG_SIZE;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(padded_global, WG_SIZE),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                sycl::sub_group sg = item.get_sub_group();
                size_t global_sg_id = item.get_global_linear_id() / SG_SIZE;
                if (global_sg_id >= total_subgroups) return;

                int64_t i = global_sg_id / num_q_heads; // query token
                int64_t h = global_sg_id % num_q_heads; // query head
                size_t lane = sg.get_local_linear_id();

                int64_t kv_h = h / gqa_ratio;
                const float* q_vec = Q + (i * num_q_heads + h) * head_dim;
                float* out_vec = out + (i * num_q_heads + h) * head_dim;

                int64_t total_keys = start_pos + i + 1;

                float max_score = -std::numeric_limits<float>::infinity();
                float sum_exp = 0.0f;

                // Query vector cached in registers (up to 16 elements per lane for head_dim <= 256)
                float q_reg[16];
                for (size_t d = 0; d < elems_per_lane; ++d) {
                    size_t idx = lane * elems_per_lane + d;
                    q_reg[d] = q_vec[idx];
                }

                // Accumulator in registers (up to 16 elements per lane for head_dim <= 256)
                float lane_out[16] = {0.0f};

                for (int64_t j = 0; j < total_keys; ++j) {
                    const sycl::half* k_ptr = k_cache + (j * num_kv_heads + kv_h) * head_dim;
                    const sycl::half* v_ptr = v_cache + (j * num_kv_heads + kv_h) * head_dim;

                    // Cooperative dot product Q . K
                    float lane_dot = 0.0f;
                    for (size_t d = 0; d < elems_per_lane; ++d) {
                        size_t idx = lane * elems_per_lane + d;
                        lane_dot += q_reg[d] * static_cast<float>(k_ptr[idx]);
                    }
                    float dot = sycl::reduce_over_group(sg, lane_dot, sycl::plus<float>());
                    float score = dot * scale;

                    // Online softmax update
                    float new_max = sycl::max(max_score, score);
                    float exp_old = (max_score == -std::numeric_limits<float>::infinity()) ? 0.0f : sycl::exp(max_score - new_max);
                    float exp_new = sycl::exp(score - new_max);

                    sum_exp = sum_exp * exp_old + exp_new;

                    // Update accumulated V
                    for (size_t d = 0; d < elems_per_lane; ++d) {
                        size_t idx = lane * elems_per_lane + d;
                        lane_out[d] = lane_out[d] * exp_old + exp_new * static_cast<float>(v_ptr[idx]);
                    }
                    max_score = new_max;
                }

                // Final normalization and store
                float inv_sum = 1.0f / (sum_exp > 0.0f ? sum_exp : 1.0f);
                for (size_t d = 0; d < elems_per_lane; ++d) {
                    size_t idx = lane * elems_per_lane + d;
                    out_vec[idx] = lane_out[d] * inv_sum;
                }
            });
    });
}

// Dynamic device-pointer overload for command-graph capture/replay
sycl::event sdpa_causal_cached_dynamic(sycl::queue& q,
                                 float* out,
                                 const float* Q,
                                 const sycl::half* k_cache,
                                 const sycl::half* v_cache,
                                 const int64_t* d_start_pos,
                                 int64_t num_q_tokens,
                                 int64_t num_q_heads,
                                 int64_t num_kv_heads,
                                 int64_t head_dim,
                                 float scale) {
    if (num_q_tokens <= 0 || num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) return sycl::event{};

    if (scale <= 0.0f) {
        scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
    }

    int64_t gqa_ratio = num_q_heads / num_kv_heads;

    constexpr size_t SG_SIZE = 16;
    constexpr size_t WG_SIZE = 64;

    size_t total_subgroups = static_cast<size_t>(num_q_tokens * num_q_heads);
    size_t global_threads = total_subgroups * SG_SIZE;
    size_t padded_global = ((global_threads + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;

    size_t elems_per_lane = static_cast<size_t>(head_dim) / SG_SIZE;

    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(padded_global, WG_SIZE),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                sycl::sub_group sg = item.get_sub_group();
                size_t global_sg_id = item.get_global_linear_id() / SG_SIZE;
                if (global_sg_id >= total_subgroups) return;

                int64_t i = global_sg_id / num_q_heads;
                int64_t h = global_sg_id % num_q_heads;
                size_t lane = sg.get_local_linear_id();

                int64_t kv_h = h / gqa_ratio;
                const float* q_vec = Q + (i * num_q_heads + h) * head_dim;
                float* out_vec = out + (i * num_q_heads + h) * head_dim;

                int64_t start_pos = *d_start_pos;
                int64_t total_keys = start_pos + i + 1;

                float max_score = -std::numeric_limits<float>::infinity();
                float sum_exp = 0.0f;

                // Query vector cached in registers (up to 16 elements per lane for head_dim <= 256)
                float q_reg[16];
                for (size_t d = 0; d < elems_per_lane; ++d) {
                    size_t idx = lane * elems_per_lane + d;
                    q_reg[d] = q_vec[idx];
                }

                float lane_out[16] = {0.0f};

                for (int64_t j = 0; j < total_keys; ++j) {
                    const sycl::half* k_ptr = k_cache + (j * num_kv_heads + kv_h) * head_dim;
                    const sycl::half* v_ptr = v_cache + (j * num_kv_heads + kv_h) * head_dim;

                    float lane_dot = 0.0f;
                    for (size_t d = 0; d < elems_per_lane; ++d) {
                        size_t idx = lane * elems_per_lane + d;
                        lane_dot += q_reg[d] * static_cast<float>(k_ptr[idx]);
                    }
                    float dot = sycl::reduce_over_group(sg, lane_dot, sycl::plus<float>());
                    float score = dot * scale;

                    float new_max = sycl::max(max_score, score);
                    float exp_old = (max_score == -std::numeric_limits<float>::infinity()) ? 0.0f : sycl::exp(max_score - new_max);
                    float exp_new = sycl::exp(score - new_max);

                    sum_exp = sum_exp * exp_old + exp_new;

                    for (size_t d = 0; d < elems_per_lane; ++d) {
                        size_t idx = lane * elems_per_lane + d;
                        lane_out[d] = lane_out[d] * exp_old + exp_new * static_cast<float>(v_ptr[idx]);
                    }
                    max_score = new_max;
                }

                float inv_sum = 1.0f / (sum_exp > 0.0f ? sum_exp : 1.0f);
                for (size_t d = 0; d < elems_per_lane; ++d) {
                    size_t idx = lane * elems_per_lane + d;
                    out_vec[idx] = lane_out[d] * inv_sum;
                }
            });
    });
}

} // namespace xinfer::ops
