#include "forward.h"
#include "linear_attn.h"
#include "ops/linear.h"
#include "ops/rmsnorm.h"
#include "ops/rope.h"
#include "ops/elementwise.h"
#include "ops/attention.h"
#include "ops/sampling.h"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace xinfer::targets::qwen3_8 {

sycl::event embed_tokens_lookup(sycl::queue& q,
                                 float* out_act,
                                 const void* embed_table_bf16,
                                 const int64_t* d_token_ids,
                                 int64_t num_tokens,
                                 int64_t hidden_size) {
    const uint16_t* table = static_cast<const uint16_t*>(embed_table_bf16);

    return q.parallel_for(sycl::range<2>(num_tokens, hidden_size), [=](sycl::id<2> idx) {
        int64_t t = idx[0];
        int64_t d = idx[1];
        int64_t token_id = d_token_ids[t];

        uint16_t bf16_val = table[token_id * hidden_size + d];
        uint32_t fp32_bits = static_cast<uint32_t>(bf16_val) << 16;
        float val;
        __builtin_memcpy(&val, &fp32_bits, sizeof(float));
        out_act[t * hidden_size + d] = val;
    });
}

void forward_chunk(std::shared_ptr<core::DeviceContext> ctx,
                   core::DeviceArena& arena,
                   const qwen3_8_27b::LoadedModel& model,
                   core::KVCache& kv_cache,
                   const int64_t* token_ids,
                   int64_t seq_len,
                   int64_t start_pos,
                   bool zero_linear_state,
                   float* out_last_token_logits) {
    if (seq_len <= 0) return;
    if (start_pos < 0 || start_pos + seq_len > static_cast<int64_t>(kv_cache.max_seq_len())) {
        std::cerr << "[xinfer::qwen3_8] Error: forward_chunk bounds exceeded: start_pos="
                  << start_pos << ", seq_len=" << seq_len << ", max_seq_len="
                  << kv_cache.max_seq_len() << std::endl;
        return;
    }
    sycl::queue& q = ctx->queue();

    arena.reset();

    // Copy token IDs to device
    int64_t* d_token_ids = static_cast<int64_t*>(arena.allocate(seq_len * sizeof(int64_t)));
    ctx->copy_host_to_device(d_token_ids, token_ids, seq_len * sizeof(int64_t), true);

    // Positions for RoPE (start_pos ... start_pos + seq_len - 1)
    int64_t* d_positions = static_cast<int64_t*>(arena.allocate(seq_len * sizeof(int64_t)));
    std::vector<int64_t> host_pos(seq_len);
    for (int64_t i = 0; i < seq_len; ++i) host_pos[i] = start_pos + i;
    ctx->copy_host_to_device(d_positions, host_pos.data(), seq_len * sizeof(int64_t), true);

    // Common activation buffers
    const auto& cfg = model.config();
    const int64_t hidden_size = cfg.hidden_size;
    const int64_t intermediate_size = cfg.intermediate_size;

    float* act_x = static_cast<float*>(arena.allocate(seq_len * hidden_size * sizeof(float)));
    float* act_normed = static_cast<float*>(arena.allocate(seq_len * hidden_size * sizeof(float)));
    float* act_proj_out = static_cast<float*>(arena.allocate(seq_len * hidden_size * sizeof(float)));

    // MLP buffers
    float* act_mlp_gate = static_cast<float*>(arena.allocate(seq_len * intermediate_size * sizeof(float)));
    float* act_mlp_up   = static_cast<float*>(arena.allocate(seq_len * intermediate_size * sizeof(float)));

    // Full Attention buffers
    float* act_q_gate   = static_cast<float*>(arena.allocate(seq_len * cfg.full_q_gate_dim() * sizeof(float)));
    float* act_q        = static_cast<float*>(arena.allocate(seq_len * cfg.full_q_dim() * sizeof(float)));
    float* act_k        = static_cast<float*>(arena.allocate(seq_len * cfg.full_k_dim() * sizeof(float)));
    float* act_v        = static_cast<float*>(arena.allocate(seq_len * cfg.full_v_dim() * sizeof(float)));
    float* act_attn_out = static_cast<float*>(arena.allocate(seq_len * cfg.full_out_dim() * sizeof(float)));

    // Linear Attention buffers
    float* act_qkv_raw   = static_cast<float*>(arena.allocate(seq_len * cfg.linear_conv_channels * sizeof(float)));
    float* act_qkv_conv  = static_cast<float*>(arena.allocate(seq_len * cfg.linear_conv_channels * sizeof(float)));
    float* act_z         = static_cast<float*>(arena.allocate(seq_len * cfg.linear_z_dim * sizeof(float)));
    float* act_b         = static_cast<float*>(arena.allocate(seq_len * cfg.linear_b_dim * sizeof(float)));
    float* act_a         = static_cast<float*>(arena.allocate(seq_len * cfg.linear_a_dim * sizeof(float)));
    float* act_delta_out = static_cast<float*>(arena.allocate(seq_len * cfg.linear_z_dim * sizeof(float)));

    // 1. Initial embedding lookup
    embed_tokens_lookup(q, act_x, model.d_embed_tokens(), d_token_ids, seq_len, hidden_size);

    // 2. Loop over layers
    const auto& layers = model.layers();
    size_t full_idx = 0;
    size_t linear_idx = 0;

    for (size_t l = 0; l < layers.size(); ++l) {
        const auto& layer = layers[l];

        // Input RMSNorm
        ops::rmsnorm(q, act_normed, act_x, layer.d_input_layernorm, seq_len, hidden_size);

        if (layer.layer_type == "full_attention") {
            // Full attention:
            // Projections
            ops::linear_int4(q, act_q_gate, act_normed,
                                   static_cast<const uint8_t*>(layer.q_proj.d_weights_int4),
                                   static_cast<const sycl::half*>(layer.q_proj.d_scales),
                                   nullptr, seq_len, cfg.full_q_gate_dim(), hidden_size);
            ops::linear_int4(q, act_k, act_normed,
                                   static_cast<const uint8_t*>(layer.k_proj.d_weights_int4),
                                   static_cast<const sycl::half*>(layer.k_proj.d_scales),
                                   nullptr, seq_len, cfg.full_k_dim(), hidden_size);
            ops::linear_int4(q, act_v, act_normed,
                                   static_cast<const uint8_t*>(layer.v_proj.d_weights_int4),
                                   static_cast<const sycl::half*>(layer.v_proj.d_scales),
                                   nullptr, seq_len, cfg.full_v_dim(), hidden_size);

            int64_t num_q_heads = cfg.num_attention_heads;
            int64_t head_dim = cfg.head_dim;
            int64_t q_gate_dim = cfg.full_q_gate_dim();

            // Q and Gate extraction and output gating:
            // Citing official transformers/models/qwen3_5/modeling_qwen3_5.py (Qwen3_5Attention.forward, lines 654-688):
            //   query_states, gate = torch.chunk(
            //       self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
            //   )
            //   gate = gate.reshape(*input_shape, -1)
            //   ...
            //   attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            //   attn_output = attn_output * torch.sigmoid(gate)
            // For each of the num_q_heads (24), q_proj produces 2 * head_dim = 512 channels.
            // Chunk 0 (channels 0..255) is Q; Chunk 1 (channels 256..511) is Gate.
            q.parallel_for(sycl::range<2>(seq_len, num_q_heads), [=](sycl::id<2> idx) {
                int64_t t = idx[0];
                int64_t h = idx[1];
                for (int d = 0; d < head_dim; ++d) {
                    act_q[(t * num_q_heads + h) * head_dim + d] = act_q_gate[t * q_gate_dim + h * 2 * head_dim + d];
                }
            });

            // Head RMSNorm on Q and K (Qwen3_5RMSNorm: 1.0 + weight, only on head_dim)
            ops::rmsnorm(q, act_q, act_q, layer.d_q_norm, seq_len * num_q_heads, head_dim);
            ops::rmsnorm(q, act_k, act_k, layer.d_k_norm, seq_len * cfg.num_key_value_heads, head_dim);

            // RoPE on Q and K
            ops::rope(q, act_q, act_k, seq_len, num_q_heads, cfg.num_key_value_heads, head_dim, d_positions, cfg.rope_theta, cfg.rope_dim);

            // Write K and V into KV cache
            ops::attention_write_kv_cache(q, kv_cache.k_cache(full_idx), kv_cache.v_cache(full_idx),
                                          act_k, act_v, start_pos, seq_len, cfg.num_key_value_heads, head_dim,
                                          static_cast<int64_t>(kv_cache.max_seq_len()));

            // Causal Scaled Dot-Product Attention reading from KV Cache
            ops::sdpa_causal_cached(q, act_attn_out, act_q,
                                    kv_cache.k_cache(full_idx), kv_cache.v_cache(full_idx),
                                    start_pos, seq_len, num_q_heads, cfg.num_key_value_heads, head_dim, 0.0f,
                                    static_cast<int64_t>(kv_cache.max_seq_len()));

            // Output gating: attn_out *= sigmoid(gate) matching torch.sigmoid(gate)
            q.parallel_for(sycl::range<2>(seq_len, num_q_heads), [=](sycl::id<2> idx) {
                int64_t t = idx[0];
                int64_t h = idx[1];
                for (int d = 0; d < head_dim; ++d) {
                    float gate_val = act_q_gate[t * q_gate_dim + h * 2 * head_dim + head_dim + d];
                    float sig = 1.0f / (1.0f + sycl::exp(-gate_val));
                    act_attn_out[(t * num_q_heads + h) * head_dim + d] *= sig;
                }
            });

            // Out projection
            ops::linear_int4(q, act_proj_out, act_attn_out,
                                   static_cast<const uint8_t*>(layer.o_proj.d_weights_int4),
                                   static_cast<const sycl::half*>(layer.o_proj.d_scales),
                                   nullptr, seq_len, hidden_size, cfg.full_out_dim());
            full_idx++;
        } else {
            // Linear attention:
            ops::linear_int4(q, act_qkv_raw, act_normed,
                                   static_cast<const uint8_t*>(layer.in_proj_qkv.d_weights_int4),
                                   static_cast<const sycl::half*>(layer.in_proj_qkv.d_scales),
                                   nullptr, seq_len, cfg.linear_conv_channels, hidden_size);
            ops::linear_int4(q, act_z, act_normed,
                                   static_cast<const uint8_t*>(layer.in_proj_z.d_weights_int4),
                                   static_cast<const sycl::half*>(layer.in_proj_z.d_scales),
                                   nullptr, seq_len, cfg.linear_z_dim, hidden_size);
            ops::linear_int4(q, act_b, act_normed,
                                   static_cast<const uint8_t*>(layer.in_proj_b.d_weights_int4),
                                   static_cast<const sycl::half*>(layer.in_proj_b.d_scales),
                                   nullptr, seq_len, cfg.linear_b_dim, hidden_size);
            ops::linear_int4(q, act_a, act_normed,
                                   static_cast<const uint8_t*>(layer.in_proj_a.d_weights_int4),
                                   static_cast<const sycl::half*>(layer.in_proj_a.d_scales),
                                   nullptr, seq_len, cfg.linear_a_dim, hidden_size);

            // Stateful Causal Conv1d + SiLU
            causal_conv1d_silu(q, act_qkv_conv, act_qkv_raw, layer.d_conv1d_weight, seq_len,
                               kv_cache.conv_state(linear_idx));

            // Stateful Recurrent Gated Delta Rule + RMSNormGated
            recurrent_gated_delta_net(q, act_delta_out, act_qkv_conv, act_z, act_b, act_a,
                                       layer.d_A_log, layer.d_dt_bias, layer.d_norm_weight,
                                       kv_cache.linear_state(linear_idx), seq_len, zero_linear_state);

            // Out projection
            ops::linear_int4(q, act_proj_out, act_delta_out,
                                   static_cast<const uint8_t*>(layer.out_proj.d_weights_int4),
                                   static_cast<const sycl::half*>(layer.out_proj.d_scales),
                                   nullptr, seq_len, hidden_size, cfg.linear_z_dim);
            linear_idx++;
        }

        // Residual connection: act_x += act_proj_out
        ops::add_inplace(q, act_x, act_proj_out, seq_len * hidden_size);

        // MLP
        ops::rmsnorm(q, act_normed, act_x, layer.d_post_attention_layernorm, seq_len, hidden_size);

        ops::linear_int4(q, act_mlp_gate, act_normed,
                               static_cast<const uint8_t*>(layer.gate_proj.d_weights_int4),
                               static_cast<const sycl::half*>(layer.gate_proj.d_scales),
                               nullptr, seq_len, intermediate_size, hidden_size);
        ops::linear_int4(q, act_mlp_up, act_normed,
                               static_cast<const uint8_t*>(layer.up_proj.d_weights_int4),
                               static_cast<const sycl::half*>(layer.up_proj.d_scales),
                               nullptr, seq_len, intermediate_size, hidden_size);

        // SwiGLU: SiLU(gate) * up
        ops::swiglu(q, act_mlp_gate, act_mlp_gate, act_mlp_up, seq_len * intermediate_size);

        // Down projection
        ops::linear_int4(q, act_proj_out, act_mlp_gate,
                               static_cast<const uint8_t*>(layer.down_proj.d_weights_int4),
                               static_cast<const sycl::half*>(layer.down_proj.d_scales),
                               nullptr, seq_len, hidden_size, intermediate_size);

        // Residual connection: act_x += act_proj_out
        ops::add_inplace(q, act_x, act_proj_out, seq_len * hidden_size);
    }

    // 3. Optional LM Head projection for the last token in chunk
    if (out_last_token_logits) {
        float* last_x = act_x + (seq_len - 1) * hidden_size;
        ops::rmsnorm(q, act_normed, last_x, model.d_final_norm(), 1, hidden_size);

        const auto& lm_head = model.lm_head();
        ops::linear_int4(q, out_last_token_logits, act_normed,
                               static_cast<const uint8_t*>(lm_head.d_weights_int4),
                               static_cast<const sycl::half*>(lm_head.d_scales),
                               nullptr, 1, cfg.vocab_size, hidden_size);
    }
}

int64_t prefill_prompt(std::shared_ptr<core::DeviceContext> ctx,
                       core::DeviceArena& arena,
                       const qwen3_8_27b::LoadedModel& model,
                       core::KVCache& kv_cache,
                       const std::vector<int64_t>& prompt_tokens,
                       size_t chunk_size) {
    if (prompt_tokens.empty()) return 0;
    if (prompt_tokens.size() > kv_cache.max_seq_len()) {
        std::cerr << "[xinfer::qwen3_8] Error: Prompt tokens (" << prompt_tokens.size()
                  << ") exceeds KV cache max_seq_len (" << kv_cache.max_seq_len() << ")" << std::endl;
        return -1;
    }
    if (chunk_size == 0) chunk_size = 512;

    kv_cache.clear();

    const auto& cfg = model.config();
    float* d_logits = static_cast<float*>(arena.persistent_buffer(cfg.vocab_size * sizeof(float)));

    size_t total_tokens = prompt_tokens.size();
    size_t offset = 0;

    while (offset < total_tokens) {
        size_t cur_chunk = std::min(chunk_size, total_tokens - offset);
        bool is_last = (offset + cur_chunk == total_tokens);
        bool zero_linear = (offset == 0);

        forward_chunk(ctx, arena, model, kv_cache,
                      prompt_tokens.data() + offset,
                      cur_chunk,
                      offset,
                      zero_linear,
                      is_last ? d_logits : nullptr);

        offset += cur_chunk;
    }

    kv_cache.set_seq_len(total_tokens);

    // Greedy argmax
    int64_t first_token = ops::argmax(ctx->queue(), d_logits, cfg.vocab_size);
    return first_token;
}

int64_t decode_step(std::shared_ptr<core::DeviceContext> ctx,
                    core::DeviceArena& arena,
                    const qwen3_8_27b::LoadedModel& model,
                    core::KVCache& kv_cache,
                    int64_t input_token_id,
                    float* d_logits) {
    if (kv_cache.current_seq_len() >= kv_cache.max_seq_len()) {
        std::cerr << "[xinfer::qwen3_8] Error: decode_step called with current_seq_len ("
                  << kv_cache.current_seq_len() << ") >= max_seq_len ("
                  << kv_cache.max_seq_len() << ")" << std::endl;
        return -1;
    }

    const auto& cfg = model.config();
    if (!d_logits) {
        d_logits = static_cast<float*>(arena.persistent_buffer(cfg.vocab_size * sizeof(float)));
    }

    int64_t cur_pos = static_cast<int64_t>(kv_cache.current_seq_len());

    // Single token forward pass (seq_len = 1)
    forward_chunk(ctx, arena, model, kv_cache,
                  &input_token_id,
                  1,
                  cur_pos,
                  false,
                  d_logits);

    if (!kv_cache.advance(1)) {
        return -1;
    }

    // Greedy argmax
    int64_t next_token = ops::argmax(ctx->queue(), d_logits, cfg.vocab_size);
    return next_token;
}

int64_t forward_next_token(std::shared_ptr<core::DeviceContext> ctx,
                           core::DeviceArena& arena,
                           const qwen3_8_27b::LoadedModel& model,
                           const std::vector<int64_t>& token_ids) {
    core::KVCacheConfig cfg = model.config().create_kv_cache_config(std::max<size_t>(8192, token_ids.size() + 1));
    core::KVCache temp_cache(ctx, cfg);
    temp_cache.allocate();
    return prefill_prompt(ctx, arena, model, temp_cache, token_ids, token_ids.size());
}

} // namespace xinfer::targets::qwen3_8
