// Citing vendor documentation per AGENTS.md §5:
// - docs/vendor/xe-gpu-architecture.md (lines 22-50):
//   Intel Arc Pro B60: 20 Xe-cores, 8 Vector Engines per core, 8 HW threads per VE
//   = 64 HW threads per core (1280 total). Sub-group size: 16, 32.
//   Vector Engine SIMD ALUs support native FP16 and FP32.
// - docs/vendor/thread-mapping-occupancy.md (lines 9-15):
//   Sub-group size 16 maps to one Vector Engine hardware thread; work-group to Xe-core.
//   sycl::reqd_sub_group_size(16).

#include "linear_attn.h"

namespace xinfer::targets::qwen3_8 {

template <typename InT, typename OutT>
sycl::event causal_conv1d_silu_impl(sycl::queue& q,
                                    OutT* out_qkv,
                                    const InT* in_qkv,
                                    const float* conv_w,
                                    int64_t seq_len,
                                    float* conv_state) {
    constexpr int64_t num_channels = 10240;
    if (seq_len <= 0) return sycl::event{};

    if (seq_len == 1) {
        return q.parallel_for(sycl::range<1>(num_channels), [=](sycl::id<1> idx) {
            int64_t c = idx[0];
            float in_val = static_cast<float>(in_qkv[c]);
            float s0 = conv_state ? conv_state[0 * num_channels + c] : 0.0f;
            float s1 = conv_state ? conv_state[1 * num_channels + c] : 0.0f;
            float s2 = conv_state ? conv_state[2 * num_channels + c] : 0.0f;

            // conv_w has shape [num_channels, 4]
            // k = 0 (src_t = -3): s0
            // k = 1 (src_t = -2): s1
            // k = 2 (src_t = -1): s2
            // k = 3 (src_t = 0):  in_val
            float sum = conv_w[c * 4 + 0] * s0 +
                        conv_w[c * 4 + 1] * s1 +
                        conv_w[c * 4 + 2] * s2 +
                        conv_w[c * 4 + 3] * in_val;

            float silu_val = sum / (1.0f + sycl::exp(-sum));
            out_qkv[c] = static_cast<OutT>(silu_val);

            if (conv_state) {
                conv_state[0 * num_channels + c] = s1;
                conv_state[1 * num_channels + c] = s2;
                conv_state[2 * num_channels + c] = in_val;
            }
        });
    }

    auto e1 = q.parallel_for(sycl::range<2>(seq_len, num_channels), [=](sycl::id<2> idx) {
        int64_t t = idx[0];
        int64_t c = idx[1];

        float sum = 0.0f;
        // conv_w has shape [num_channels, 4]
        for (int k = 0; k < 4; ++k) {
            int64_t src_t = t - (3 - k);
            float val = 0.0f;
            if (src_t >= 0) {
                val = static_cast<float>(in_qkv[src_t * num_channels + c]);
            } else if (conv_state) {
                // src_t is -1, -2, or -3. conv_state has shape [3, 10240]
                int64_t state_idx = 3 + src_t;
                if (state_idx >= 0 && state_idx < 3) {
                    val = conv_state[state_idx * num_channels + c];
                }
            }
            sum += conv_w[c * 4 + k] * val;
        }

        // SiLU: x / (1 + exp(-x))
        float silu_val = sum / (1.0f + sycl::exp(-sum));
        out_qkv[t * num_channels + c] = static_cast<OutT>(silu_val);
    });

    // Update conv_state with the last up to 3 timesteps of in_qkv
    if (conv_state && seq_len > 0) {
        return q.parallel_for(sycl::range<1>(num_channels), [=](sycl::id<1> idx) {
            int64_t c = idx[0];
            if (seq_len >= 3) {
                conv_state[0 * num_channels + c] = static_cast<float>(in_qkv[(seq_len - 3) * num_channels + c]);
                conv_state[1 * num_channels + c] = static_cast<float>(in_qkv[(seq_len - 2) * num_channels + c]);
                conv_state[2 * num_channels + c] = static_cast<float>(in_qkv[(seq_len - 1) * num_channels + c]);
            } else if (seq_len == 1) {
                conv_state[0 * num_channels + c] = conv_state[1 * num_channels + c];
                conv_state[1 * num_channels + c] = conv_state[2 * num_channels + c];
                conv_state[2 * num_channels + c] = static_cast<float>(in_qkv[c]);
            } else if (seq_len == 2) {
                conv_state[0 * num_channels + c] = conv_state[2 * num_channels + c];
                conv_state[1 * num_channels + c] = static_cast<float>(in_qkv[0 * num_channels + c]);
                conv_state[2 * num_channels + c] = static_cast<float>(in_qkv[1 * num_channels + c]);
            }
        });
    }
    return e1;
}

sycl::event causal_conv1d_silu(sycl::queue& q,
                               sycl::half* out_qkv,
                               const sycl::half* in_qkv,
                               const float* conv_w,
                               int64_t seq_len,
                               float* conv_state) {
    return causal_conv1d_silu_impl<sycl::half, sycl::half>(q, out_qkv, in_qkv, conv_w, seq_len, conv_state);
}

sycl::event causal_conv1d_silu(sycl::queue& q,
                               float* out_qkv,
                               const float* in_qkv,
                               const float* conv_w,
                               int64_t seq_len,
                               float* conv_state) {
    return causal_conv1d_silu_impl<float, float>(q, out_qkv, in_qkv, conv_w, seq_len, conv_state);
}

template <typename InT, typename OutT>
sycl::event recurrent_gated_delta_net_impl(sycl::queue& q,
                                           OutT* out,
                                           const InT* qkv,
                                           const InT* z,
                                           const InT* b,
                                           const InT* a,
                                           const float* A_log,
                                           const float* dt_bias,
                                           const float* norm_weight,
                                           float* state_buffer,
                                           int64_t seq_len,
                                           bool zero_state) {
    if (seq_len <= 0) return sycl::event{};
    constexpr int64_t num_v_heads = 48;
    constexpr int64_t num_k_heads = 16;
    constexpr int64_t head_k_dim = 128;
    constexpr int64_t head_v_dim = 128;
    constexpr int64_t total_channels = 10240;
    constexpr int64_t value_dim = 6144;

    return q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> shared_q(sycl::range<1>(128), cgh);
        sycl::local_accessor<float, 1> shared_k(sycl::range<1>(128), cgh);
        sycl::local_accessor<float, 1> shared_inv_std(sycl::range<1>(1), cgh);
        sycl::local_accessor<float, 1> slm_sums(sycl::range<1>(8), cgh); // 8 sub-groups of 16

        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(num_v_heads, 128), sycl::range<2>(1, 128)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                int64_t h = item.get_group(0);
                int64_t j = item.get_local_id(1); // 0..127

                // GQA head grouping: h_k = h / 3.
                // Citing official transformers/models/qwen3_5/modeling_qwen3_5.py (Qwen3_5GatedDeltaNet.forward, lines 428-444, 479-482):
                //   self.num_v_heads = 48, self.num_k_heads = 16
                //   in_proj_qkv outputs [key_dim (2048), key_dim (2048), value_dim (6144)] = 10240 channels.
                //   if self.num_v_heads // self.num_k_heads > 1: (48 // 16 == 3)
                //       query = query.repeat_interleave(3, dim=2)
                //       key = key.repeat_interleave(3, dim=2)
                // repeat_interleave replicates each Q/K head 3 consecutive times:
                // K-head 0 serves V-heads 0, 1, 2; K-head 1 serves V-heads 3, 4, 5; K-head h_k serves V-heads 3*h_k..3*h_k+2.
                // Thus for value head h in [0, 47], the corresponding Q and K head index is exactly h / 3 (in [0, 15]).
                int64_t h_k = h / 3;
                sycl::sub_group sg = item.get_sub_group();
                size_t sg_id = sg.get_group_linear_id();

                float* S = state_buffer + h * (head_k_dim * head_v_dim);

                if (zero_state) {
                    for (int i = 0; i < 128; ++i) {
                        S[i * 128 + j] = 0.0f;
                    }
                }

                float a_log_val = A_log[h];
                float exp_a_log = sycl::exp(a_log_val);
                float dt_bias_val = dt_bias[h];

                for (int64_t t = 0; t < seq_len; ++t) {
                    const InT* cur_qkv = qkv + t * total_channels;
                    const InT* cur_q_in = cur_qkv + (h_k * head_k_dim);
                    const InT* cur_k_in = cur_qkv + (num_k_heads * head_k_dim) + (h_k * head_k_dim);
                    const InT* cur_v_in = cur_qkv + (num_k_heads * head_k_dim * 2) + (h * head_v_dim);

                    // 1. Cooperative L2 norm of q and k
                    float raw_q = static_cast<float>(cur_q_in[j]);
                    float raw_k = static_cast<float>(cur_k_in[j]);
                    float q_sq = raw_q * raw_q;
                    float k_sq = raw_k * raw_k;

                    float sg_q_sq = sycl::reduce_over_group(sg, q_sq, sycl::plus<float>());
                    float sg_k_sq = sycl::reduce_over_group(sg, k_sq, sycl::plus<float>());
                    if (sg.get_local_linear_id() == 0) {
                        slm_sums[sg_id] = sg_q_sq;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (sg_id == 0) {
                        float total_q_sq = 0.0f;
                        if (sg.get_local_linear_id() < 8) {
                            total_q_sq = slm_sums[sg.get_local_linear_id()];
                        }
                        float full_q_sq = sycl::reduce_over_group(sg, total_q_sq, sycl::plus<float>());
                        if (sg.get_local_linear_id() == 0) {
                            shared_inv_std[0] = 1.0f / sycl::sqrt(full_q_sq + 1e-6f);
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    float inv_norm_q = shared_inv_std[0];

                    if (sg.get_local_linear_id() == 0) {
                        slm_sums[sg_id] = sg_k_sq;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (sg_id == 0) {
                        float total_k_sq = 0.0f;
                        if (sg.get_local_linear_id() < 8) {
                            total_k_sq = slm_sums[sg.get_local_linear_id()];
                        }
                        float full_k_sq = sycl::reduce_over_group(sg, total_k_sq, sycl::plus<float>());
                        if (sg.get_local_linear_id() == 0) {
                            shared_inv_std[0] = 1.0f / sycl::sqrt(full_k_sq + 1e-6f);
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    float inv_norm_k = shared_inv_std[0];

                    constexpr float q_scale = 0.08838834764f;
                    shared_q[j] = (raw_q * inv_norm_q) * q_scale;
                    shared_k[j] = raw_k * inv_norm_k;
                    item.barrier(sycl::access::fence_space::local_space);

                    // 2. Beta and decay g
                    float b_val = static_cast<float>(b[t * num_v_heads + h]);
                    float beta = 1.0f / (1.0f + sycl::exp(-b_val));

                    float a_val = static_cast<float>(a[t * num_v_heads + h]) + dt_bias_val;
                    float softplus_a = (a_val > 20.0f) ? a_val : sycl::log(1.0f + sycl::exp(a_val));
                    float g = -exp_a_log * softplus_a;
                    float exp_g = sycl::exp(g);

                    // 3 & 4. Decay column j and compute kv_mem[j] = sum_i (S[i, j] * k[i])
                    float kv_mem_j = 0.0f;
                    for (int i = 0; i < 128; ++i) {
                        float s_val = S[i * 128 + j] * exp_g;
                        S[i * 128 + j] = s_val;
                        kv_mem_j += s_val * shared_k[i];
                    }

                    // 5 & 6 & 7. Fused delta update and core_attn_out in a single pass over S
                    float delta_j = (static_cast<float>(cur_v_in[j]) - kv_mem_j) * beta;
                    float attn_j = 0.0f;
                    for (int i = 0; i < 128; ++i) {
                        float new_s = S[i * 128 + j] + shared_k[i] * delta_j;
                        S[i * 128 + j] = new_s;
                        attn_j += new_s * shared_q[i];
                    }

                    // 8. Workgroup RMSNorm of attn_j across 128 threads
                    float attn_sq = attn_j * attn_j;
                    float sg_attn_sq = sycl::reduce_over_group(sg, attn_sq, sycl::plus<float>());
                    if (sg.get_local_linear_id() == 0) {
                        slm_sums[sg_id] = sg_attn_sq;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (sg_id == 0) {
                        float total_attn_sq = 0.0f;
                        if (sg.get_local_linear_id() < 8) {
                            total_attn_sq = slm_sums[sg.get_local_linear_id()];
                        }
                        float full_attn_sq = sycl::reduce_over_group(sg, total_attn_sq, sycl::plus<float>());
                        if (sg.get_local_linear_id() == 0) {
                            float variance = full_attn_sq / 128.0f;
                            shared_inv_std[0] = 1.0f / sycl::sqrt(variance + 1e-6f);
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    float inv_std = shared_inv_std[0];
                    const InT* cur_z = z + t * value_dim + h * 128;
                    OutT* cur_out = out + t * value_dim + h * 128;

                    float normed = attn_j * inv_std * norm_weight[j];
                    float z_val = static_cast<float>(cur_z[j]);
                    float silu_z = z_val / (1.0f + sycl::exp(-z_val));
                    cur_out[j] = static_cast<OutT>(normed * silu_z);
                }
            });
    });
}

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
                                      bool zero_state) {
    return recurrent_gated_delta_net_impl<sycl::half, sycl::half>(
        q, out, qkv, z, b, a, A_log, dt_bias, norm_weight, state_buffer, seq_len, zero_state);
}

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
                                      bool zero_state) {
    return recurrent_gated_delta_net_impl<float, float>(
        q, out, qkv, z, b, a, A_log, dt_bias, norm_weight, state_buffer, seq_len, zero_state);
}

} // namespace xinfer::targets::qwen3_8
