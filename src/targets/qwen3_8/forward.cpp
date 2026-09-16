#include "forward.h"
#include "linear_attn.h"
#include "ops/rmsnorm.h"
#include "ops/rope.h"
#include "ops/attention.h"
#include "ops/linear.h"
#include "ops/elementwise.h"
#include "ops/sampling.h"
#include <iostream>
#include <cstring>

namespace xinfer::targets::qwen3_8 {

void embed_tokens_lookup(sycl::queue& q,
                         float* out_act,
                         const void* embed_table_bf16,
                         const int64_t* d_token_ids,
                         int64_t num_tokens,
                         int64_t hidden_size) {
    const uint16_t* tbl = static_cast<const uint16_t*>(embed_table_bf16);
    q.parallel_for(sycl::range<2>(num_tokens, hidden_size), [=](sycl::id<2> idx) {
        int64_t t = idx[0];
        int64_t h = idx[1];
        int64_t tok = d_token_ids[t];
        uint16_t b = tbl[tok * hidden_size + h];
        uint32_t u = static_cast<uint32_t>(b) << 16;
        float f;
        std::memcpy(&f, &u, sizeof(float));
        out_act[t * hidden_size + h] = f;
    });
}

int64_t forward_next_token(std::shared_ptr<core::DeviceContext> ctx,
                           core::DeviceArena& arena,
                           const qwen3_8_27b::LoadedModel& model,
                           const std::vector<int64_t>& token_ids) {
    if (token_ids.empty()) return 0;
    int64_t seq_len = static_cast<int64_t>(token_ids.size());
    sycl::queue& q = ctx->queue();

    arena.reset();

    // Copy token IDs to device
    int64_t* d_token_ids = static_cast<int64_t*>(arena.allocate(seq_len * sizeof(int64_t)));
    ctx->copy_host_to_device(d_token_ids, token_ids.data(), seq_len * sizeof(int64_t), false);

    // Positions for RoPE
    int64_t* d_positions = static_cast<int64_t*>(arena.allocate(seq_len * sizeof(int64_t)));
    std::vector<int64_t> host_pos(seq_len);
    for (int64_t i = 0; i < seq_len; ++i) host_pos[i] = i;
    ctx->copy_host_to_device(d_positions, host_pos.data(), seq_len * sizeof(int64_t), false);

    // Common activation buffers
    constexpr int64_t hidden_size = 5120;
    constexpr int64_t intermediate_size = 17408;

    float* act_x = static_cast<float*>(arena.allocate(seq_len * hidden_size * sizeof(float)));
    float* act_normed = static_cast<float*>(arena.allocate(seq_len * hidden_size * sizeof(float)));
    float* act_proj_out = static_cast<float*>(arena.allocate(seq_len * hidden_size * sizeof(float)));

    // MLP buffers
    float* act_mlp_gate = static_cast<float*>(arena.allocate(seq_len * intermediate_size * sizeof(float)));
    float* act_mlp_up   = static_cast<float*>(arena.allocate(seq_len * intermediate_size * sizeof(float)));

    // Full Attention buffers
    float* act_q_gate   = static_cast<float*>(arena.allocate(seq_len * 12288 * sizeof(float)));
    float* act_q        = static_cast<float*>(arena.allocate(seq_len * 6144 * sizeof(float)));
    float* act_k        = static_cast<float*>(arena.allocate(seq_len * 1024 * sizeof(float)));
    float* act_v        = static_cast<float*>(arena.allocate(seq_len * 1024 * sizeof(float)));
    float* act_attn_out = static_cast<float*>(arena.allocate(seq_len * 6144 * sizeof(float)));

    // Linear Attention buffers
    float* act_qkv_raw   = static_cast<float*>(arena.allocate(seq_len * 10240 * sizeof(float)));
    float* act_qkv_conv  = static_cast<float*>(arena.allocate(seq_len * 10240 * sizeof(float)));
    float* act_z         = static_cast<float*>(arena.allocate(seq_len * 6144 * sizeof(float)));
    float* act_b         = static_cast<float*>(arena.allocate(seq_len * 48 * sizeof(float)));
    float* act_a         = static_cast<float*>(arena.allocate(seq_len * 48 * sizeof(float)));
    float* act_delta_out = static_cast<float*>(arena.allocate(seq_len * 6144 * sizeof(float)));
    float* d_state_buf   = static_cast<float*>(arena.allocate(48 * 128 * 128 * sizeof(float)));

    // Final logits buffer
    constexpr int64_t vocab_size = 248320;
    float* d_logits = static_cast<float*>(arena.allocate(vocab_size * sizeof(float)));

    // 1. Initial embedding lookup
    embed_tokens_lookup(q, act_x, model.d_embed_tokens(), d_token_ids, seq_len, hidden_size);

    // 2. Loop over 64 layers
    const auto& layers = model.layers();
    for (size_t l = 0; l < layers.size(); ++l) {
        const auto& layer = layers[l];

        // Input RMSNorm
        ops::rmsnorm(q, act_normed, act_x, layer.d_input_layernorm, seq_len, hidden_size);

        if (layer.layer_type == "full_attention") {
            // Full attention:
            // Projections
            ops::linear_int4_naive(q, act_q_gate, act_normed,
                                   static_cast<const uint8_t*>(layer.q_proj.d_weights_int4),
                                   static_cast<const sycl::half*>(layer.q_proj.d_scales),
                                   nullptr, seq_len, 12288, hidden_size);
            ops::linear_int4_naive(q, act_k, act_normed,
                                   static_cast<const uint8_t*>(layer.k_proj.d_weights_int4),
                                   static_cast<const sycl::half*>(layer.k_proj.d_scales),
                                   nullptr, seq_len, 1024, hidden_size);
            ops::linear_int4_naive(q, act_v, act_normed,
                                   static_cast<const uint8_t*>(layer.v_proj.d_weights_int4),
                                   static_cast<const sycl::half*>(layer.v_proj.d_scales),
                                   nullptr, seq_len, 1024, hidden_size);

            // Extract Q from interleaved act_q_gate [seq_len, 24, 512]:
            // For each token t and head h (24 heads), first 256 are Q, and next 256 are Gate.
            q.parallel_for(sycl::range<2>(seq_len, 24), [=](sycl::id<2> idx) {
                int64_t t = idx[0];
                int64_t h = idx[1];
                for (int d = 0; d < 256; ++d) {
                    act_q[(t * 24 + h) * 256 + d] = act_q_gate[t * 12288 + h * 512 + d];
                }
            });

            // Head RMSNorm on Q and K
            ops::rmsnorm(q, act_q, act_q, layer.d_q_norm, seq_len * 24, 256);
            ops::rmsnorm(q, act_k, act_k, layer.d_k_norm, seq_len * 4, 256);

            // RoPE on Q and K (with partial rotary dimension 64 for Qwen3.8)
            ops::rope(q, act_q, act_k, seq_len, 24, 4, 256, d_positions, 10000000.0f, 64);

            // Causal Scaled Dot-Product Attention
            ops::sdpa_causal_naive(q, act_attn_out, act_q, act_k, act_v, seq_len, 24, 4, 256);

            // Output gating: attn_out *= sigmoid(gate)
            // gate for head h and dim d is at act_q_gate[t * 12288 + h * 512 + 256 + d]
            q.parallel_for(sycl::range<2>(seq_len, 24), [=](sycl::id<2> idx) {
                int64_t t = idx[0];
                int64_t h = idx[1];
                for (int d = 0; d < 256; ++d) {
                    float gate_val = act_q_gate[t * 12288 + h * 512 + 256 + d];
                    float sig = 1.0f / (1.0f + sycl::exp(-gate_val));
                    act_attn_out[(t * 24 + h) * 256 + d] *= sig;
                }
            });

            // Out projection
            ops::linear_int4_naive(q, act_proj_out, act_attn_out,
                                   static_cast<const uint8_t*>(layer.o_proj.d_weights_int4),
                                   static_cast<const sycl::half*>(layer.o_proj.d_scales),
                                   nullptr, seq_len, hidden_size, 6144);
        } else {
            // Linear attention:
            ops::linear_int4_naive(q, act_qkv_raw, act_normed,
                                   static_cast<const uint8_t*>(layer.in_proj_qkv.d_weights_int4),
                                   static_cast<const sycl::half*>(layer.in_proj_qkv.d_scales),
                                   nullptr, seq_len, 10240, hidden_size);
            ops::linear_int4_naive(q, act_z, act_normed,
                                   static_cast<const uint8_t*>(layer.in_proj_z.d_weights_int4),
                                   static_cast<const sycl::half*>(layer.in_proj_z.d_scales),
                                   nullptr, seq_len, 6144, hidden_size);
            ops::linear_int4_naive(q, act_b, act_normed,
                                   static_cast<const uint8_t*>(layer.in_proj_b.d_weights_int4),
                                   static_cast<const sycl::half*>(layer.in_proj_b.d_scales),
                                   nullptr, seq_len, 48, hidden_size);
            ops::linear_int4_naive(q, act_a, act_normed,
                                   static_cast<const uint8_t*>(layer.in_proj_a.d_weights_int4),
                                   static_cast<const sycl::half*>(layer.in_proj_a.d_scales),
                                   nullptr, seq_len, 48, hidden_size);

            // Causal Conv1d + SiLU
            causal_conv1d_silu(q, act_qkv_conv, act_qkv_raw, layer.d_conv1d_weight, seq_len);

            // Recurrent Gated Delta Rule + RMSNormGated
            recurrent_gated_delta_net(q, act_delta_out, act_qkv_conv, act_z, act_b, act_a,
                                      layer.d_A_log, layer.d_dt_bias, layer.d_norm_weight,
                                      d_state_buf, seq_len);

            // Out projection
            ops::linear_int4_naive(q, act_proj_out, act_delta_out,
                                   static_cast<const uint8_t*>(layer.out_proj.d_weights_int4),
                                   static_cast<const sycl::half*>(layer.out_proj.d_scales),
                                   nullptr, seq_len, hidden_size, 6144);
        }

        // Residual connection: act_x += act_proj_out
        ops::add_inplace(q, act_x, act_proj_out, seq_len * hidden_size);

        // MLP
        ops::rmsnorm(q, act_normed, act_x, layer.d_post_attention_layernorm, seq_len, hidden_size);

        ops::linear_int4_naive(q, act_mlp_gate, act_normed,
                               static_cast<const uint8_t*>(layer.gate_proj.d_weights_int4),
                               static_cast<const sycl::half*>(layer.gate_proj.d_scales),
                               nullptr, seq_len, intermediate_size, hidden_size);
        ops::linear_int4_naive(q, act_mlp_up, act_normed,
                               static_cast<const uint8_t*>(layer.up_proj.d_weights_int4),
                               static_cast<const sycl::half*>(layer.up_proj.d_scales),
                               nullptr, seq_len, intermediate_size, hidden_size);

        // SwiGLU: SiLU(gate) * up
        ops::swiglu(q, act_mlp_gate, act_mlp_gate, act_mlp_up, seq_len * intermediate_size);

        // Down projection
        ops::linear_int4_naive(q, act_proj_out, act_mlp_gate,
                               static_cast<const uint8_t*>(layer.down_proj.d_weights_int4),
                               static_cast<const sycl::half*>(layer.down_proj.d_scales),
                               nullptr, seq_len, hidden_size, intermediate_size);

        // Residual connection: act_x += act_proj_out
        ops::add_inplace(q, act_x, act_proj_out, seq_len * hidden_size);
    }

    // 3. Final norm on the last token only
    float* last_x = act_x + (seq_len - 1) * hidden_size;
    ops::rmsnorm(q, act_normed, last_x, model.d_final_norm(), 1, hidden_size);

    // 4. LM Head projection on the last token
    const auto& lm_head = model.lm_head();
    ops::linear_int4_naive(q, d_logits, act_normed,
                           static_cast<const uint8_t*>(lm_head.d_weights_int4),
                           static_cast<const sycl::half*>(lm_head.d_scales),
                           nullptr, 1, vocab_size, hidden_size);

    // 5. Greedy argmax
    int64_t next_token = ops::argmax(q, d_logits, vocab_size);
    return next_token;
}

} // namespace xinfer::targets::qwen3_8
