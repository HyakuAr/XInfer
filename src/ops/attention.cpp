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

namespace {

template <typename QT, typename KT, typename VT, typename OutT>
void sdpa_causal_naive_impl(sycl::queue& q,
                            OutT* out,
                            const QT* Q,
                            const KT* K,
                            const VT* V,
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

    q.parallel_for(sycl::range<2>(static_cast<size_t>(seq_len), static_cast<size_t>(num_q_heads)), [=](sycl::id<2> idx) {
        int64_t i = idx[0]; // Target token position
        int64_t h = idx[1]; // Query head index

        int64_t kv_h = h / gqa_ratio;
        const QT* q_vec = Q + (i * num_q_heads + h) * head_dim;
        OutT* out_vec = out + (i * num_q_heads + h) * head_dim;

        float max_score = -std::numeric_limits<float>::infinity();
        float sum_exp = 0.0f;

        // Temporary float accumulation buffer in private memory (head_dim <= 256)
        float acc[256];
        for (int64_t d = 0; d < head_dim; ++d) {
            acc[d] = 0.0f;
        }

        for (int64_t j = 0; j <= i; ++j) {
            const KT* k_vec = K + (j * num_kv_heads + kv_h) * head_dim;
            const VT* v_vec = V + (j * num_kv_heads + kv_h) * head_dim;

            float dot = 0.0f;
            for (int64_t d = 0; d < head_dim; ++d) {
                dot += static_cast<float>(q_vec[d]) * static_cast<float>(k_vec[d]);
            }
            float score = dot * scale;

            float new_max = sycl::max(max_score, score);
            float exp_old = (max_score == -std::numeric_limits<float>::infinity()) ? 0.0f : sycl::exp(max_score - new_max);
            float exp_new = sycl::exp(score - new_max);

            sum_exp = sum_exp * exp_old + exp_new;
            for (int64_t d = 0; d < head_dim; ++d) {
                acc[d] = acc[d] * exp_old + exp_new * static_cast<float>(v_vec[d]);
            }
            max_score = new_max;
        }

        float inv_sum = 1.0f / (sum_exp > 0.0f ? sum_exp : 1.0f);
        for (int64_t d = 0; d < head_dim; ++d) {
            out_vec[d] = static_cast<OutT>(acc[d] * inv_sum);
        }
    });
}

template <typename T>
void attention_write_kv_cache_impl(sycl::queue& q,
                                   sycl::half* k_cache,
                                   sycl::half* v_cache,
                                   const T* K_in,
                                   const T* V_in,
                                   int64_t start_pos,
                                   int64_t num_tokens,
                                   int64_t num_kv_heads,
                                   int64_t head_dim,
                                   int64_t max_seq_len) {
    if (num_tokens <= 0 || num_kv_heads <= 0 || head_dim <= 0) return;
    if (max_seq_len > 0) {
        if (start_pos >= max_seq_len || start_pos < 0) return;
        if (start_pos + num_tokens > max_seq_len) {
            num_tokens = max_seq_len - start_pos;
        }
    }
    int64_t kv_stride = num_kv_heads * head_dim;
    size_t total_elements = static_cast<size_t>(num_tokens * kv_stride);

    q.parallel_for(sycl::range<1>(total_elements), [=](sycl::id<1> idx) {
        size_t i = idx[0];
        size_t t = i / kv_stride;
        size_t rem = i % kv_stride;
        int64_t pos = start_pos + static_cast<int64_t>(t);
        if (max_seq_len > 0 && (pos < 0 || pos >= max_seq_len)) {
            return;
        }
        size_t cache_idx = static_cast<size_t>(pos) * kv_stride + rem;

        k_cache[cache_idx] = static_cast<sycl::half>(K_in[i]);
        v_cache[cache_idx] = static_cast<sycl::half>(V_in[i]);
    });
}

template <typename T>
sycl::event attention_write_kv_cache_dynamic_impl(sycl::queue& q,
                                                  sycl::half* k_cache,
                                                  sycl::half* v_cache,
                                                  const T* K_in,
                                                  const T* V_in,
                                                  const int64_t* d_start_pos,
                                                  int64_t num_tokens,
                                                  int64_t num_kv_heads,
                                                  int64_t head_dim,
                                                  int64_t max_seq_len) {
    if (num_tokens <= 0 || num_kv_heads <= 0 || head_dim <= 0) return sycl::event{};
    int64_t kv_stride = num_kv_heads * head_dim;
    size_t total_elements = static_cast<size_t>(num_tokens * kv_stride);

    return q.parallel_for(sycl::range<1>(total_elements), [=](sycl::id<1> idx) {
        size_t i = idx[0];
        size_t t = i / kv_stride;
        size_t rem = i % kv_stride;
        int64_t start_pos = *d_start_pos;
        int64_t pos = start_pos + static_cast<int64_t>(t);
        if (pos < 0 || (max_seq_len > 0 && pos >= max_seq_len)) {
            return;
        }
        size_t cache_idx = static_cast<size_t>(pos) * kv_stride + rem;

        k_cache[cache_idx] = static_cast<sycl::half>(K_in[i]);
        v_cache[cache_idx] = static_cast<sycl::half>(V_in[i]);
    });
}

template <typename QT, typename OutT>
void sdpa_causal_cached_impl(sycl::queue& q,
                             OutT* out,
                             const QT* Q,
                             const sycl::half* k_cache,
                             const sycl::half* v_cache,
                             int64_t start_pos,
                             int64_t num_q_tokens,
                             int64_t num_q_heads,
                             int64_t num_kv_heads,
                             int64_t head_dim,
                             float scale,
                             int64_t max_seq_len) {
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
                const QT* q_vec = Q + (i * num_q_heads + h) * head_dim;
                OutT* out_vec = out + (i * num_q_heads + h) * head_dim;

                int64_t total_keys = start_pos + i + 1;
                if (max_seq_len > 0 && total_keys > max_seq_len) {
                    total_keys = max_seq_len;
                }
                if (total_keys <= 0) return;

                float max_score = -std::numeric_limits<float>::infinity();
                float sum_exp = 0.0f;

                // Query vector cached in registers (up to 16 elements per lane for head_dim <= 256)
                float q_reg[16];
                for (size_t d = 0; d < elems_per_lane; ++d) {
                    size_t idx = lane * elems_per_lane + d;
                    q_reg[d] = static_cast<float>(q_vec[idx]);
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
                    out_vec[idx] = static_cast<OutT>(lane_out[d] * inv_sum);
                }
            });
    });
}

template <typename QT, typename OutT>
sycl::event sdpa_causal_cached_dynamic_impl(sycl::queue& q,
                                            OutT* out,
                                            const QT* Q,
                                            const sycl::half* k_cache,
                                            const sycl::half* v_cache,
                                            const int64_t* d_start_pos,
                                            int64_t num_q_tokens,
                                            int64_t num_q_heads,
                                            int64_t num_kv_heads,
                                            int64_t head_dim,
                                            float scale,
                                            int64_t max_seq_len) {
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
                const QT* q_vec = Q + (i * num_q_heads + h) * head_dim;
                OutT* out_vec = out + (i * num_q_heads + h) * head_dim;

                int64_t start_pos = *d_start_pos;
                int64_t total_keys = start_pos + i + 1;
                if (max_seq_len > 0 && total_keys > max_seq_len) {
                    total_keys = max_seq_len;
                }
                if (total_keys <= 0) return;

                float max_score = -std::numeric_limits<float>::infinity();
                float sum_exp = 0.0f;

                // Query vector cached in registers (up to 16 elements per lane for head_dim <= 256)
                float q_reg[16];
                for (size_t d = 0; d < elems_per_lane; ++d) {
                    size_t idx = lane * elems_per_lane + d;
                    q_reg[d] = static_cast<float>(q_vec[idx]);
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
                    out_vec[idx] = static_cast<OutT>(lane_out[d] * inv_sum);
                }
            });
    });
}

template <typename T>
void attention_write_kv_cache_int8_impl(sycl::queue& q,
                                        int8_t* k_cache,
                                        int8_t* v_cache,
                                        float* k_scale,
                                        float* v_scale,
                                        float* k_zero_point,
                                        float* v_zero_point,
                                        const T* K_in,
                                        const T* V_in,
                                        int64_t start_pos,
                                        int64_t num_tokens,
                                        int64_t num_kv_heads,
                                        int64_t head_dim,
                                        int64_t max_seq_len) {
    if (num_tokens <= 0 || num_kv_heads <= 0 || head_dim <= 0) return;
    if (max_seq_len > 0) {
        if (start_pos >= max_seq_len || start_pos < 0) return;
        if (start_pos + num_tokens > max_seq_len) {
            num_tokens = max_seq_len - start_pos;
        }
    }

    size_t total_head_vectors = static_cast<size_t>(num_tokens * num_kv_heads);

    q.parallel_for(sycl::range<1>(total_head_vectors), [=](sycl::id<1> idx) {
        size_t linear_head_idx = idx[0];
        size_t t = linear_head_idx / num_kv_heads;
        size_t h = linear_head_idx % num_kv_heads;
        int64_t pos = start_pos + static_cast<int64_t>(t);
        if (max_seq_len > 0 && (pos < 0 || pos >= max_seq_len)) return;

        const T* k_src = K_in + linear_head_idx * head_dim;
        const T* v_src = V_in + linear_head_idx * head_dim;

        float k_min = static_cast<float>(k_src[0]);
        float k_max = k_min;
        float v_min = static_cast<float>(v_src[0]);
        float v_max = v_min;

        for (int64_t d = 1; d < head_dim; ++d) {
            float k_val = static_cast<float>(k_src[d]);
            if (k_val < k_min) k_min = k_val;
            if (k_val > k_max) k_max = k_val;
            float v_val = static_cast<float>(v_src[d]);
            if (v_val < v_min) v_min = v_val;
            if (v_val > v_max) v_max = v_val;
        }

        float k_scale_val = (k_max > k_min) ? ((k_max - k_min) / 255.0f) : 1.0f;
        float k_inv_scale = (k_max > k_min) ? (255.0f / (k_max - k_min)) : 0.0f;
        float v_scale_val = (v_max > v_min) ? ((v_max - v_min) / 255.0f) : 1.0f;
        float v_inv_scale = (v_max > v_min) ? (255.0f / (v_max - v_min)) : 0.0f;

        size_t cache_head_idx = static_cast<size_t>(pos) * num_kv_heads + h;
        k_scale[cache_head_idx] = k_scale_val;
        k_zero_point[cache_head_idx] = k_min;
        v_scale[cache_head_idx] = v_scale_val;
        v_zero_point[cache_head_idx] = v_min;

        size_t dst_base = cache_head_idx * head_dim;
        for (int64_t d = 0; d < head_dim; ++d) {
            float k_val = static_cast<float>(k_src[d]);
            float k_q = sycl::clamp(sycl::round((k_val - k_min) * k_inv_scale), 0.0f, 255.0f) - 128.0f;
            k_cache[dst_base + d] = static_cast<int8_t>(k_q);

            float v_val = static_cast<float>(v_src[d]);
            float v_q = sycl::clamp(sycl::round((v_val - v_min) * v_inv_scale), 0.0f, 255.0f) - 128.0f;
            v_cache[dst_base + d] = static_cast<int8_t>(v_q);
        }
    });
}

template <typename T>
sycl::event attention_write_kv_cache_int8_dynamic_impl(sycl::queue& q,
                                                       int8_t* k_cache,
                                                       int8_t* v_cache,
                                                       float* k_scale,
                                                       float* v_scale,
                                                       float* k_zero_point,
                                                       float* v_zero_point,
                                                       const T* K_in,
                                                       const T* V_in,
                                                       const int64_t* d_start_pos,
                                                       int64_t num_tokens,
                                                       int64_t num_kv_heads,
                                                       int64_t head_dim,
                                                       int64_t max_seq_len) {
    if (num_tokens <= 0 || num_kv_heads <= 0 || head_dim <= 0) return sycl::event{};
    size_t total_head_vectors = static_cast<size_t>(num_tokens * num_kv_heads);

    return q.parallel_for(sycl::range<1>(total_head_vectors), [=](sycl::id<1> idx) {
        size_t linear_head_idx = idx[0];
        size_t t = linear_head_idx / num_kv_heads;
        size_t h = linear_head_idx % num_kv_heads;
        int64_t start_pos = *d_start_pos;
        int64_t pos = start_pos + static_cast<int64_t>(t);
        if (pos < 0 || (max_seq_len > 0 && pos >= max_seq_len)) return;

        const T* k_src = K_in + linear_head_idx * head_dim;
        const T* v_src = V_in + linear_head_idx * head_dim;

        float k_min = static_cast<float>(k_src[0]);
        float k_max = k_min;
        float v_min = static_cast<float>(v_src[0]);
        float v_max = v_min;

        for (int64_t d = 1; d < head_dim; ++d) {
            float k_val = static_cast<float>(k_src[d]);
            if (k_val < k_min) k_min = k_val;
            if (k_val > k_max) k_max = k_val;
            float v_val = static_cast<float>(v_src[d]);
            if (v_val < v_min) v_min = v_val;
            if (v_val > v_max) v_max = v_val;
        }

        float k_scale_val = (k_max > k_min) ? ((k_max - k_min) / 255.0f) : 1.0f;
        float k_inv_scale = (k_max > k_min) ? (255.0f / (k_max - k_min)) : 0.0f;
        float v_scale_val = (v_max > v_min) ? ((v_max - v_min) / 255.0f) : 1.0f;
        float v_inv_scale = (v_max > v_min) ? (255.0f / (v_max - v_min)) : 0.0f;

        size_t cache_head_idx = static_cast<size_t>(pos) * num_kv_heads + h;
        k_scale[cache_head_idx] = k_scale_val;
        k_zero_point[cache_head_idx] = k_min;
        v_scale[cache_head_idx] = v_scale_val;
        v_zero_point[cache_head_idx] = v_min;

        size_t dst_base = cache_head_idx * head_dim;
        for (int64_t d = 0; d < head_dim; ++d) {
            float k_val = static_cast<float>(k_src[d]);
            float k_q = sycl::clamp(sycl::round((k_val - k_min) * k_inv_scale), 0.0f, 255.0f) - 128.0f;
            k_cache[dst_base + d] = static_cast<int8_t>(k_q);

            float v_val = static_cast<float>(v_src[d]);
            float v_q = sycl::clamp(sycl::round((v_val - v_min) * v_inv_scale), 0.0f, 255.0f) - 128.0f;
            v_cache[dst_base + d] = static_cast<int8_t>(v_q);
        }
    });
}

template <typename OutT>
void attention_read_kv_cache_int8_impl(sycl::queue& q,
                                       OutT* K_out,
                                       OutT* V_out,
                                       const int8_t* k_cache,
                                       const int8_t* v_cache,
                                       const float* k_scale,
                                       const float* v_scale,
                                       const float* k_zero_point,
                                       const float* v_zero_point,
                                       int64_t start_pos,
                                       int64_t num_tokens,
                                       int64_t num_kv_heads,
                                       int64_t head_dim,
                                       int64_t max_seq_len) {
    if (num_tokens <= 0 || num_kv_heads <= 0 || head_dim <= 0) return;
    size_t total_head_vectors = static_cast<size_t>(num_tokens * num_kv_heads);

    q.parallel_for(sycl::range<1>(total_head_vectors), [=](sycl::id<1> idx) {
        size_t linear_head_idx = idx[0];
        size_t t = linear_head_idx / num_kv_heads;
        size_t h = linear_head_idx % num_kv_heads;
        int64_t pos = start_pos + static_cast<int64_t>(t);
        if (max_seq_len > 0 && (pos < 0 || pos >= max_seq_len)) return;

        size_t cache_head_idx = static_cast<size_t>(pos) * num_kv_heads + h;
        float k_s = k_scale[cache_head_idx];
        float k_zp = k_zero_point[cache_head_idx];
        float v_s = v_scale[cache_head_idx];
        float v_zp = v_zero_point[cache_head_idx];

        const int8_t* k_src = k_cache + cache_head_idx * head_dim;
        const int8_t* v_src = v_cache + cache_head_idx * head_dim;

        OutT* k_dst = K_out + linear_head_idx * head_dim;
        OutT* v_dst = V_out + linear_head_idx * head_dim;

        for (int64_t d = 0; d < head_dim; ++d) {
            float k_deq = (static_cast<float>(k_src[d]) + 128.0f) * k_s + k_zp;
            k_dst[d] = static_cast<OutT>(k_deq);

            float v_deq = (static_cast<float>(v_src[d]) + 128.0f) * v_s + v_zp;
            v_dst[d] = static_cast<OutT>(v_deq);
        }
    });
}

template <typename QT, typename OutT>
void sdpa_causal_cached_int8_impl(sycl::queue& q,
                                  OutT* out,
                                  const QT* Q,
                                  const int8_t* k_cache,
                                  const int8_t* v_cache,
                                  const float* k_scale,
                                  const float* v_scale,
                                  const float* k_zero_point,
                                  const float* v_zero_point,
                                  int64_t start_pos,
                                  int64_t num_q_tokens,
                                  int64_t num_q_heads,
                                  int64_t num_kv_heads,
                                  int64_t head_dim,
                                  float scale,
                                  int64_t max_seq_len) {
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

    size_t elems_per_lane = static_cast<size_t>(head_dim) / SG_SIZE;

    q.submit([&](sycl::handler& cgh) {
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
                const QT* q_vec = Q + (i * num_q_heads + h) * head_dim;
                OutT* out_vec = out + (i * num_q_heads + h) * head_dim;

                int64_t total_keys = start_pos + i + 1;
                if (max_seq_len > 0 && total_keys > max_seq_len) {
                    total_keys = max_seq_len;
                }
                if (total_keys <= 0) return;

                float max_score = -std::numeric_limits<float>::infinity();
                float sum_exp = 0.0f;

                float q_reg[16];
                for (size_t d = 0; d < elems_per_lane; ++d) {
                    size_t idx = lane * elems_per_lane + d;
                    q_reg[d] = static_cast<float>(q_vec[idx]);
                }

                float lane_out[16] = {0.0f};

                for (int64_t j = 0; j < total_keys; ++j) {
                    size_t cache_head_idx = static_cast<size_t>(j * num_kv_heads + kv_h);
                    float k_s = k_scale[cache_head_idx];
                    float k_zp = k_zero_point[cache_head_idx];
                    float v_s = v_scale[cache_head_idx];
                    float v_zp = v_zero_point[cache_head_idx];

                    const int8_t* k_ptr = k_cache + cache_head_idx * head_dim;
                    const int8_t* v_ptr = v_cache + cache_head_idx * head_dim;

                    float lane_dot = 0.0f;
                    for (size_t d = 0; d < elems_per_lane; ++d) {
                        size_t idx = lane * elems_per_lane + d;
                        float k_deq = (static_cast<float>(k_ptr[idx]) + 128.0f) * k_s + k_zp;
                        lane_dot += q_reg[d] * k_deq;
                    }
                    float dot = sycl::reduce_over_group(sg, lane_dot, sycl::plus<float>());
                    float score = dot * scale;

                    float new_max = sycl::max(max_score, score);
                    float exp_old = (max_score == -std::numeric_limits<float>::infinity()) ? 0.0f : sycl::exp(max_score - new_max);
                    float exp_new = sycl::exp(score - new_max);

                    sum_exp = sum_exp * exp_old + exp_new;

                    for (size_t d = 0; d < elems_per_lane; ++d) {
                        size_t idx = lane * elems_per_lane + d;
                        float v_deq = (static_cast<float>(v_ptr[idx]) + 128.0f) * v_s + v_zp;
                        lane_out[d] = lane_out[d] * exp_old + exp_new * v_deq;
                    }
                    max_score = new_max;
                }

                float inv_sum = 1.0f / (sum_exp > 0.0f ? sum_exp : 1.0f);
                for (size_t d = 0; d < elems_per_lane; ++d) {
                    size_t idx = lane * elems_per_lane + d;
                    out_vec[idx] = static_cast<OutT>(lane_out[d] * inv_sum);
                }
            });
    });
}

template <typename QT, typename OutT>
sycl::event sdpa_causal_cached_int8_dynamic_impl(sycl::queue& q,
                                                 OutT* out,
                                                 const QT* Q,
                                                 const int8_t* k_cache,
                                                 const int8_t* v_cache,
                                                 const float* k_scale,
                                                 const float* v_scale,
                                                 const float* k_zero_point,
                                                 const float* v_zero_point,
                                                 const int64_t* d_start_pos,
                                                 int64_t num_q_tokens,
                                                 int64_t num_q_heads,
                                                 int64_t num_kv_heads,
                                                 int64_t head_dim,
                                                 float scale,
                                                 int64_t max_seq_len) {
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
                const QT* q_vec = Q + (i * num_q_heads + h) * head_dim;
                OutT* out_vec = out + (i * num_q_heads + h) * head_dim;

                int64_t start_pos = *d_start_pos;
                int64_t total_keys = start_pos + i + 1;
                if (max_seq_len > 0 && total_keys > max_seq_len) {
                    total_keys = max_seq_len;
                }
                if (total_keys <= 0) return;

                float max_score = -std::numeric_limits<float>::infinity();
                float sum_exp = 0.0f;

                float q_reg[16];
                for (size_t d = 0; d < elems_per_lane; ++d) {
                    size_t idx = lane * elems_per_lane + d;
                    q_reg[d] = static_cast<float>(q_vec[idx]);
                }

                float lane_out[16] = {0.0f};

                for (int64_t j = 0; j < total_keys; ++j) {
                    size_t cache_head_idx = static_cast<size_t>(j * num_kv_heads + kv_h);
                    float k_s = k_scale[cache_head_idx];
                    float k_zp = k_zero_point[cache_head_idx];
                    float v_s = v_scale[cache_head_idx];
                    float v_zp = v_zero_point[cache_head_idx];

                    const int8_t* k_ptr = k_cache + cache_head_idx * head_dim;
                    const int8_t* v_ptr = v_cache + cache_head_idx * head_dim;

                    float lane_dot = 0.0f;
                    for (size_t d = 0; d < elems_per_lane; ++d) {
                        size_t idx = lane * elems_per_lane + d;
                        float k_deq = (static_cast<float>(k_ptr[idx]) + 128.0f) * k_s + k_zp;
                        lane_dot += q_reg[d] * k_deq;
                    }
                    float dot = sycl::reduce_over_group(sg, lane_dot, sycl::plus<float>());
                    float score = dot * scale;

                    float new_max = sycl::max(max_score, score);
                    float exp_old = (max_score == -std::numeric_limits<float>::infinity()) ? 0.0f : sycl::exp(max_score - new_max);
                    float exp_new = sycl::exp(score - new_max);

                    sum_exp = sum_exp * exp_old + exp_new;

                    for (size_t d = 0; d < elems_per_lane; ++d) {
                        size_t idx = lane * elems_per_lane + d;
                        float v_deq = (static_cast<float>(v_ptr[idx]) + 128.0f) * v_s + v_zp;
                        lane_out[d] = lane_out[d] * exp_old + exp_new * v_deq;
                    }
                    max_score = new_max;
                }

                float inv_sum = 1.0f / (sum_exp > 0.0f ? sum_exp : 1.0f);
                for (size_t d = 0; d < elems_per_lane; ++d) {
                    size_t idx = lane * elems_per_lane + d;
                    out_vec[idx] = static_cast<OutT>(lane_out[d] * inv_sum);
                }
            });
    });
}

} // anonymous namespace

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
    sdpa_causal_naive_impl(q, out, Q, K, V, seq_len, num_q_heads, num_kv_heads, head_dim, scale);
}

void sdpa_causal_naive(sycl::queue& q,
                       sycl::half* out,
                       const sycl::half* Q,
                       const sycl::half* K,
                       const sycl::half* V,
                       int64_t seq_len,
                       int64_t num_q_heads,
                       int64_t num_kv_heads,
                       int64_t head_dim,
                       float scale) {
    sdpa_causal_naive_impl(q, out, Q, K, V, seq_len, num_q_heads, num_kv_heads, head_dim, scale);
}

void attention_write_kv_cache(sycl::queue& q,
                              sycl::half* k_cache,
                              sycl::half* v_cache,
                              const float* K_in,
                              const float* V_in,
                              int64_t start_pos,
                              int64_t num_tokens,
                              int64_t num_kv_heads,
                              int64_t head_dim,
                              int64_t max_seq_len) {
    attention_write_kv_cache_impl(q, k_cache, v_cache, K_in, V_in, start_pos, num_tokens, num_kv_heads, head_dim, max_seq_len);
}

void attention_write_kv_cache(sycl::queue& q,
                              sycl::half* k_cache,
                              sycl::half* v_cache,
                              const sycl::half* K_in,
                              const sycl::half* V_in,
                              int64_t start_pos,
                              int64_t num_tokens,
                              int64_t num_kv_heads,
                              int64_t head_dim,
                              int64_t max_seq_len) {
    attention_write_kv_cache_impl(q, k_cache, v_cache, K_in, V_in, start_pos, num_tokens, num_kv_heads, head_dim, max_seq_len);
}

sycl::event attention_write_kv_cache_dynamic(sycl::queue& q,
                                       sycl::half* k_cache,
                                       sycl::half* v_cache,
                                       const float* K_in,
                                       const float* V_in,
                                       const int64_t* d_start_pos,
                                       int64_t num_tokens,
                                       int64_t num_kv_heads,
                                       int64_t head_dim,
                                       int64_t max_seq_len) {
    return attention_write_kv_cache_dynamic_impl(q, k_cache, v_cache, K_in, V_in, d_start_pos, num_tokens, num_kv_heads, head_dim, max_seq_len);
}

sycl::event attention_write_kv_cache_dynamic(sycl::queue& q,
                                       sycl::half* k_cache,
                                       sycl::half* v_cache,
                                       const sycl::half* K_in,
                                       const sycl::half* V_in,
                                       const int64_t* d_start_pos,
                                       int64_t num_tokens,
                                       int64_t num_kv_heads,
                                       int64_t head_dim,
                                       int64_t max_seq_len) {
    return attention_write_kv_cache_dynamic_impl(q, k_cache, v_cache, K_in, V_in, d_start_pos, num_tokens, num_kv_heads, head_dim, max_seq_len);
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
                        float scale,
                        int64_t max_seq_len) {
    sdpa_causal_cached_impl(q, out, Q, k_cache, v_cache, start_pos, num_q_tokens, num_q_heads, num_kv_heads, head_dim, scale, max_seq_len);
}

void sdpa_causal_cached(sycl::queue& q,
                        sycl::half* out,
                        const sycl::half* Q,
                        const sycl::half* k_cache,
                        const sycl::half* v_cache,
                        int64_t start_pos,
                        int64_t num_q_tokens,
                        int64_t num_q_heads,
                        int64_t num_kv_heads,
                        int64_t head_dim,
                        float scale,
                        int64_t max_seq_len) {
    sdpa_causal_cached_impl(q, out, Q, k_cache, v_cache, start_pos, num_q_tokens, num_q_heads, num_kv_heads, head_dim, scale, max_seq_len);
}

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
                                 float scale,
                                 int64_t max_seq_len) {
    return sdpa_causal_cached_dynamic_impl(q, out, Q, k_cache, v_cache, d_start_pos, num_q_tokens, num_q_heads, num_kv_heads, head_dim, scale, max_seq_len);
}

sycl::event sdpa_causal_cached_dynamic(sycl::queue& q,
                                 sycl::half* out,
                                 const sycl::half* Q,
                                 const sycl::half* k_cache,
                                 const sycl::half* v_cache,
                                 const int64_t* d_start_pos,
                                 int64_t num_q_tokens,
                                 int64_t num_q_heads,
                                 int64_t num_kv_heads,
                                 int64_t head_dim,
                                 float scale,
                                 int64_t max_seq_len) {
    return sdpa_causal_cached_dynamic_impl(q, out, Q, k_cache, v_cache, d_start_pos, num_q_tokens, num_q_heads, num_kv_heads, head_dim, scale, max_seq_len);
}

void attention_write_kv_cache_int8(sycl::queue& q,
                                   int8_t* k_cache,
                                   int8_t* v_cache,
                                   float* k_scale,
                                   float* v_scale,
                                   float* k_zero_point,
                                   float* v_zero_point,
                                   const float* K_in,
                                   const float* V_in,
                                   int64_t start_pos,
                                   int64_t num_tokens,
                                   int64_t num_kv_heads,
                                   int64_t head_dim,
                                   int64_t max_seq_len) {
    attention_write_kv_cache_int8_impl(q, k_cache, v_cache, k_scale, v_scale, k_zero_point, v_zero_point,
                                       K_in, V_in, start_pos, num_tokens, num_kv_heads, head_dim, max_seq_len);
}

void attention_write_kv_cache_int8(sycl::queue& q,
                                   int8_t* k_cache,
                                   int8_t* v_cache,
                                   float* k_scale,
                                   float* v_scale,
                                   float* k_zero_point,
                                   float* v_zero_point,
                                   const sycl::half* K_in,
                                   const sycl::half* V_in,
                                   int64_t start_pos,
                                   int64_t num_tokens,
                                   int64_t num_kv_heads,
                                   int64_t head_dim,
                                   int64_t max_seq_len) {
    attention_write_kv_cache_int8_impl(q, k_cache, v_cache, k_scale, v_scale, k_zero_point, v_zero_point,
                                       K_in, V_in, start_pos, num_tokens, num_kv_heads, head_dim, max_seq_len);
}

sycl::event attention_write_kv_cache_int8_dynamic(sycl::queue& q,
                                                  int8_t* k_cache,
                                                  int8_t* v_cache,
                                                  float* k_scale,
                                                  float* v_scale,
                                                  float* k_zero_point,
                                                  float* v_zero_point,
                                                  const float* K_in,
                                                  const float* V_in,
                                                  const int64_t* d_start_pos,
                                                  int64_t num_tokens,
                                                  int64_t num_kv_heads,
                                                  int64_t head_dim,
                                                  int64_t max_seq_len) {
    return attention_write_kv_cache_int8_dynamic_impl(q, k_cache, v_cache, k_scale, v_scale, k_zero_point, v_zero_point,
                                                      K_in, V_in, d_start_pos, num_tokens, num_kv_heads, head_dim, max_seq_len);
}

sycl::event attention_write_kv_cache_int8_dynamic(sycl::queue& q,
                                                  int8_t* k_cache,
                                                  int8_t* v_cache,
                                                  float* k_scale,
                                                  float* v_scale,
                                                  float* k_zero_point,
                                                  float* v_zero_point,
                                                  const sycl::half* K_in,
                                                  const sycl::half* V_in,
                                                  const int64_t* d_start_pos,
                                                  int64_t num_tokens,
                                                  int64_t num_kv_heads,
                                                  int64_t head_dim,
                                                  int64_t max_seq_len) {
    return attention_write_kv_cache_int8_dynamic_impl(q, k_cache, v_cache, k_scale, v_scale, k_zero_point, v_zero_point,
                                                      K_in, V_in, d_start_pos, num_tokens, num_kv_heads, head_dim, max_seq_len);
}

void attention_read_kv_cache_int8(sycl::queue& q,
                                  sycl::half* K_out,
                                  sycl::half* V_out,
                                  const int8_t* k_cache,
                                  const int8_t* v_cache,
                                  const float* k_scale,
                                  const float* v_scale,
                                  const float* k_zero_point,
                                  const float* v_zero_point,
                                  int64_t start_pos,
                                  int64_t num_tokens,
                                  int64_t num_kv_heads,
                                  int64_t head_dim,
                                  int64_t max_seq_len) {
    attention_read_kv_cache_int8_impl(q, K_out, V_out, k_cache, v_cache, k_scale, v_scale, k_zero_point, v_zero_point,
                                      start_pos, num_tokens, num_kv_heads, head_dim, max_seq_len);
}

void attention_read_kv_cache_int8(sycl::queue& q,
                                  float* K_out,
                                  float* V_out,
                                  const int8_t* k_cache,
                                  const int8_t* v_cache,
                                  const float* k_scale,
                                  const float* v_scale,
                                  const float* k_zero_point,
                                  const float* v_zero_point,
                                  int64_t start_pos,
                                  int64_t num_tokens,
                                  int64_t num_kv_heads,
                                  int64_t head_dim,
                                  int64_t max_seq_len) {
    attention_read_kv_cache_int8_impl(q, K_out, V_out, k_cache, v_cache, k_scale, v_scale, k_zero_point, v_zero_point,
                                      start_pos, num_tokens, num_kv_heads, head_dim, max_seq_len);
}

void sdpa_causal_cached_int8(sycl::queue& q,
                             float* out,
                             const float* Q,
                             const int8_t* k_cache,
                             const int8_t* v_cache,
                             const float* k_scale,
                             const float* v_scale,
                             const float* k_zero_point,
                             const float* v_zero_point,
                             int64_t start_pos,
                             int64_t num_q_tokens,
                             int64_t num_q_heads,
                             int64_t num_kv_heads,
                             int64_t head_dim,
                             float scale,
                             int64_t max_seq_len) {
    sdpa_causal_cached_int8_impl(q, out, Q, k_cache, v_cache, k_scale, v_scale, k_zero_point, v_zero_point,
                                 start_pos, num_q_tokens, num_q_heads, num_kv_heads, head_dim, scale, max_seq_len);
}

void sdpa_causal_cached_int8(sycl::queue& q,
                             sycl::half* out,
                             const sycl::half* Q,
                             const int8_t* k_cache,
                             const int8_t* v_cache,
                             const float* k_scale,
                             const float* v_scale,
                             const float* k_zero_point,
                             const float* v_zero_point,
                             int64_t start_pos,
                             int64_t num_q_tokens,
                             int64_t num_q_heads,
                             int64_t num_kv_heads,
                             int64_t head_dim,
                             float scale,
                             int64_t max_seq_len) {
    sdpa_causal_cached_int8_impl(q, out, Q, k_cache, v_cache, k_scale, v_scale, k_zero_point, v_zero_point,
                                 start_pos, num_q_tokens, num_q_heads, num_kv_heads, head_dim, scale, max_seq_len);
}

sycl::event sdpa_causal_cached_int8_dynamic(sycl::queue& q,
                                            float* out,
                                            const float* Q,
                                            const int8_t* k_cache,
                                            const int8_t* v_cache,
                                            const float* k_scale,
                                            const float* v_scale,
                                            const float* k_zero_point,
                                            const float* v_zero_point,
                                            const int64_t* d_start_pos,
                                            int64_t num_q_tokens,
                                            int64_t num_q_heads,
                                            int64_t num_kv_heads,
                                            int64_t head_dim,
                                            float scale,
                                            int64_t max_seq_len) {
    return sdpa_causal_cached_int8_dynamic_impl(q, out, Q, k_cache, v_cache, k_scale, v_scale, k_zero_point, v_zero_point,
                                                d_start_pos, num_q_tokens, num_q_heads, num_kv_heads, head_dim, scale, max_seq_len);
}

sycl::event sdpa_causal_cached_int8_dynamic(sycl::queue& q,
                                            sycl::half* out,
                                            const sycl::half* Q,
                                            const int8_t* k_cache,
                                            const int8_t* v_cache,
                                            const float* k_scale,
                                            const float* v_scale,
                                            const float* k_zero_point,
                                            const float* v_zero_point,
                                            const int64_t* d_start_pos,
                                            int64_t num_q_tokens,
                                            int64_t num_q_heads,
                                            int64_t num_kv_heads,
                                            int64_t head_dim,
                                            float scale,
                                            int64_t max_seq_len) {
    return sdpa_causal_cached_int8_dynamic_impl(q, out, Q, k_cache, v_cache, k_scale, v_scale, k_zero_point, v_zero_point,
                                                d_start_pos, num_q_tokens, num_q_heads, num_kv_heads, head_dim, scale, max_seq_len);
}

} // namespace xinfer::ops
