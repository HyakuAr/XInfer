#include "linear_attn.h"

namespace xinfer::targets::qwen3_8 {

void causal_conv1d_silu(sycl::queue& q,
                        float* out_qkv,
                        const float* in_qkv,
                        const float* conv_w,
                        int64_t seq_len,
                        float* conv_state) {
    constexpr int64_t num_channels = 10240;

    q.parallel_for(sycl::range<2>(seq_len, num_channels), [=](sycl::id<2> idx) {
        int64_t t = idx[0];
        int64_t c = idx[1];

        float sum = 0.0f;
        // conv_w has shape [num_channels, 4]
        for (int k = 0; k < 4; ++k) {
            int64_t src_t = t - (3 - k);
            float val = 0.0f;
            if (src_t >= 0) {
                val = in_qkv[src_t * num_channels + c];
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
        out_qkv[t * num_channels + c] = sum / (1.0f + sycl::exp(-sum));
    });

    // Update conv_state with the last up to 3 timesteps of in_qkv
    if (conv_state && seq_len > 0) {
        q.parallel_for(sycl::range<1>(num_channels), [=](sycl::id<1> idx) {
            int64_t c = idx[0];
            if (seq_len >= 3) {
                conv_state[0 * num_channels + c] = in_qkv[(seq_len - 3) * num_channels + c];
                conv_state[1 * num_channels + c] = in_qkv[(seq_len - 2) * num_channels + c];
                conv_state[2 * num_channels + c] = in_qkv[(seq_len - 1) * num_channels + c];
            } else if (seq_len == 1) {
                conv_state[0 * num_channels + c] = conv_state[1 * num_channels + c];
                conv_state[1 * num_channels + c] = conv_state[2 * num_channels + c];
                conv_state[2 * num_channels + c] = in_qkv[c];
            } else if (seq_len == 2) {
                conv_state[0 * num_channels + c] = conv_state[2 * num_channels + c];
                conv_state[1 * num_channels + c] = in_qkv[0 * num_channels + c];
                conv_state[2 * num_channels + c] = in_qkv[1 * num_channels + c];
            }
        });
    }
}

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
                               int64_t seq_len,
                               bool zero_state) {
    constexpr int64_t num_v_heads = 48;
    constexpr int64_t num_k_heads = 16;
    constexpr int64_t head_k_dim = 128;
    constexpr int64_t head_v_dim = 128;
    constexpr int64_t total_channels = 10240;
    constexpr int64_t value_dim = 6144;

    // Launch 48 parallel work-items, one per value head
    q.parallel_for(sycl::range<1>(num_v_heads), [=](sycl::id<1> idx) {
        int64_t h = idx[0];
        int64_t h_k = h / 3; // Key/query head shared across 3 value heads

        // State pointer for this head: [128, 128]
        float* S = state_buffer + h * (head_k_dim * head_v_dim);

        // Zero state only if explicitly requested (e.g. start of prompt)
        if (zero_state) {
            for (int i = 0; i < head_k_dim * head_v_dim; ++i) {
                S[i] = 0.0f;
            }
        }

        float a_log_val = A_log[h];
        float exp_a_log = sycl::exp(a_log_val);
        float dt_bias_val = dt_bias[h];

        // Step sequentially through time t
        for (int64_t t = 0; t < seq_len; ++t) {
            // Pointers for time step t
            const float* cur_qkv = qkv + t * total_channels;
            const float* cur_q_in = cur_qkv + (h_k * head_k_dim);
            const float* cur_k_in = cur_qkv + (num_k_heads * head_k_dim) + (h_k * head_k_dim);
            const float* cur_v_in = cur_qkv + (num_k_heads * head_k_dim * 2) + (h * head_v_dim);

            // 1. L2 normalize q and k
            float sum_sq_q = 0.0f;
            float sum_sq_k = 0.0f;
            for (int d = 0; d < 128; ++d) {
                float q_val = cur_q_in[d];
                float k_val = cur_k_in[d];
                sum_sq_q += q_val * q_val;
                sum_sq_k += k_val * k_val;
            }
            float inv_norm_q = 1.0f / sycl::sqrt(sum_sq_q + 1e-6f);
            float inv_norm_k = 1.0f / sycl::sqrt(sum_sq_k + 1e-6f);
            constexpr float q_scale = 0.08838834764f; // 1.0 / sqrt(128.0)

            float q_vec[128];
            float k_vec[128];
            for (int d = 0; d < 128; ++d) {
                q_vec[d] = (cur_q_in[d] * inv_norm_q) * q_scale;
                k_vec[d] = cur_k_in[d] * inv_norm_k;
            }

            // 2. Compute beta and decay g
            float b_val = b[t * num_v_heads + h];
            float beta = 1.0f / (1.0f + sycl::exp(-b_val));

            float a_val = a[t * num_v_heads + h] + dt_bias_val;
            float softplus_a = (a_val > 20.0f) ? a_val : sycl::log(1.0f + sycl::exp(a_val));
            float g = -exp_a_log * softplus_a;
            float exp_g = sycl::exp(g);

            // 3. Decay state S *= exp(g)
            for (int i = 0; i < 128 * 128; ++i) {
                S[i] *= exp_g;
            }

            // 4. kv_mem[j] = sum_i (S[i, j] * k[i])
            float kv_mem[128];
            for (int j = 0; j < 128; ++j) {
                float sum = 0.0f;
                for (int i = 0; i < 128; ++i) {
                    sum += S[i * 128 + j] * k_vec[i];
                }
                kv_mem[j] = sum;
            }

            // 5. delta[j] = (v[j] - kv_mem[j]) * beta
            float delta[128];
            for (int j = 0; j < 128; ++j) {
                delta[j] = (cur_v_in[j] - kv_mem[j]) * beta;
            }

            // 6. S[i, j] += k[i] * delta[j]
            for (int i = 0; i < 128; ++i) {
                float k_i = k_vec[i];
                for (int j = 0; j < 128; ++j) {
                    S[i * 128 + j] += k_i * delta[j];
                }
            }

            // 7. core_attn_out[j] = sum_i (S[i, j] * q[i])
            float attn_h[128];
            float sum_sq_out = 0.0f;
            for (int j = 0; j < 128; ++j) {
                float sum = 0.0f;
                for (int i = 0; i < 128; ++i) {
                    sum += S[i * 128 + j] * q_vec[i];
                }
                attn_h[j] = sum;
                sum_sq_out += sum * sum;
            }

            // 8. Qwen3_5RMSNormGated: rmsnorm(attn_h) * silu(z)
            float variance = sum_sq_out / 128.0f;
            float inv_std = 1.0f / sycl::sqrt(variance + 1e-6f);

            const float* cur_z = z + t * value_dim + h * 128;
            float* cur_out = out + t * value_dim + h * 128;

            for (int j = 0; j < 128; ++j) {
                float normed = attn_h[j] * inv_std * norm_weight[j];
                float z_val = cur_z[j];
                float silu_z = z_val / (1.0f + sycl::exp(-z_val));
                cur_out[j] = normed * silu_z;
            }
        }
    });
}

} // namespace xinfer::targets::qwen3_8
