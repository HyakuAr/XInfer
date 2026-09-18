#include "forward.h"
#include "decode_graph.h"
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
#include <cassert>

namespace xinfer::targets::qwen3_8 {

template <typename OutT>
sycl::event embed_tokens_lookup_impl(sycl::queue& q,
                                      OutT* out_act,
                                      const void* embed_table_bf16,
                                      const int64_t* d_token_ids,
                                      int64_t num_tokens,
                                      int64_t hidden_size,
                                      int64_t vocab_size = qwen3_8_27b::ModelConfig::kDefaultVocabSize,
                                      const OutT* d_vit_embeddings = nullptr,
                                      int64_t visual_token_start = -1,
                                      int64_t visual_token_end = -1,
                                      const int64_t* d_visual_indices = nullptr) {
    if (!out_act || !embed_table_bf16 || !d_token_ids || num_tokens <= 0 || hidden_size <= 0) {
        return sycl::event{};
    }
    const uint16_t* table = static_cast<const uint16_t*>(embed_table_bf16);

    return q.parallel_for(sycl::range<2>(num_tokens, hidden_size), [=](sycl::id<2> idx) {
        int64_t t = idx[0];
        int64_t d = idx[1];
        int64_t token_id = d_token_ids[t];

        // Zero-Copy Visual Token Injection:
        // Check if token_id falls within the reserved visual token range.
        // If it does and ViT embeddings buffer is provided, read directly from the ViT's output USM buffer.
        if (d_vit_embeddings && visual_token_start >= 0 && token_id >= visual_token_start && token_id <= visual_token_end) {
            int64_t v_idx = d_visual_indices ? d_visual_indices[t] : (token_id - visual_token_start);
            if (v_idx >= 0) {
                out_act[t * hidden_size + d] = d_vit_embeddings[v_idx * hidden_size + d];
                return;
            }
        }

#if defined(_DEBUG)
        assert(token_id >= 0 && (vocab_size <= 0 || token_id < vocab_size));
#endif
        if (token_id < 0 || (vocab_size > 0 && token_id >= vocab_size)) {
            out_act[t * hidden_size + d] = static_cast<OutT>(0);
            return;
        }

        uint16_t bf16_val = table[token_id * hidden_size + d];
        uint32_t fp32_bits = static_cast<uint32_t>(bf16_val) << 16;
        float val;
        __builtin_memcpy(&val, &fp32_bits, sizeof(float));
        out_act[t * hidden_size + d] = static_cast<OutT>(val);
    });
}

sycl::event embed_tokens_lookup(sycl::queue& q,
                                 sycl::half* out_act,
                                 const void* embed_table_bf16,
                                 const int64_t* d_token_ids,
                                 int64_t num_tokens,
                                 int64_t hidden_size,
                                 int64_t vocab_size,
                                 const sycl::half* d_vit_embeddings,
                                 int64_t visual_token_start,
                                 int64_t visual_token_end,
                                 const int64_t* d_visual_indices) {
    return embed_tokens_lookup_impl<sycl::half>(q, out_act, embed_table_bf16, d_token_ids, num_tokens, hidden_size, vocab_size,
                                                d_vit_embeddings, visual_token_start, visual_token_end, d_visual_indices);
}

sycl::event embed_tokens_lookup(sycl::queue& q,
                                 float* out_act,
                                 const void* embed_table_bf16,
                                 const int64_t* d_token_ids,
                                 int64_t num_tokens,
                                 int64_t hidden_size,
                                 int64_t vocab_size,
                                 const float* d_vit_embeddings,
                                 int64_t visual_token_start,
                                 int64_t visual_token_end,
                                 const int64_t* d_visual_indices) {
    return embed_tokens_lookup_impl<float>(q, out_act, embed_table_bf16, d_token_ids, num_tokens, hidden_size, vocab_size,
                                           d_vit_embeddings, visual_token_start, visual_token_end, d_visual_indices);
}

void forward_layer(sycl::queue& q,
                   const qwen3_8_27b::ModelConfig& cfg,
                   const qwen3_8_27b::LayerWeights& layer,
                   core::KVCache& kv_cache,
                   size_t& full_idx,
                   size_t& linear_idx,
                   const LayerActivationBuffers& bufs,
                   int64_t seq_len,
                   const int64_t* d_positions,
                   int64_t start_pos,
                   const int64_t* d_dynamic_pos,
                   bool zero_linear_state) {
    const int64_t hidden_size = cfg.hidden_size;
    const int64_t intermediate_size = cfg.intermediate_size;

    // Input RMSNorm
    ops::rmsnorm(q, bufs.act_normed, bufs.act_x, layer.d_input_layernorm, seq_len, hidden_size);

    if (layer.layer_type == "full_attention") {
        // Full attention:
        // Fused Q, K, V wide GEMV projection launch
        ops::FusedProjectionDesc fa_projs[3] = {
            {bufs.act_q_gate, static_cast<const uint8_t*>(layer.q_proj.d_weights_int4),
             static_cast<const sycl::half*>(layer.q_proj.d_scales), nullptr, cfg.full_q_gate_dim()},
            {bufs.act_k, static_cast<const uint8_t*>(layer.k_proj.d_weights_int4),
             static_cast<const sycl::half*>(layer.k_proj.d_scales), nullptr, cfg.full_k_dim()},
            {bufs.act_v, static_cast<const uint8_t*>(layer.v_proj.d_weights_int4),
             static_cast<const sycl::half*>(layer.v_proj.d_scales), nullptr, cfg.full_v_dim()}
        };
        ops::linear_int4_fused(q, bufs.act_normed, fa_projs, 3, seq_len, hidden_size);

        int64_t num_q_heads = cfg.num_attention_heads;
        int64_t head_dim = cfg.head_dim;
        int64_t q_gate_dim = cfg.full_q_gate_dim();

        sycl::half* q_ptr = bufs.act_q;
        const sycl::half* q_gate_ptr = bufs.act_q_gate;

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
                q_ptr[(t * num_q_heads + h) * head_dim + d] = q_gate_ptr[t * q_gate_dim + h * 2 * head_dim + d];
            }
        });

        // Head RMSNorm on Q and K (Qwen3_5RMSNorm: 1.0 + weight, only on head_dim)
        ops::rmsnorm(q, bufs.act_q, bufs.act_q, layer.d_q_norm, seq_len * num_q_heads, head_dim);
        ops::rmsnorm(q, bufs.act_k, bufs.act_k, layer.d_k_norm, seq_len * cfg.num_key_value_heads, head_dim);

        // RoPE on Q and K
        ops::rope(q, bufs.act_q, bufs.act_k, seq_len, num_q_heads, cfg.num_key_value_heads, head_dim, d_positions, cfg.rope_theta, cfg.rope_dim);

        // Write K and V into KV cache and perform Causal SDPA
        if (kv_cache.is_int8()) {
            if (d_dynamic_pos) {
                ops::attention_write_kv_cache_int8_dynamic(
                    q, kv_cache.k_cache_int8(full_idx), kv_cache.v_cache_int8(full_idx),
                    kv_cache.k_scale(full_idx), kv_cache.v_scale(full_idx),
                    kv_cache.k_zero_point(full_idx), kv_cache.v_zero_point(full_idx),
                    bufs.act_k, bufs.act_v, d_dynamic_pos, seq_len, cfg.num_key_value_heads, head_dim,
                    static_cast<int64_t>(kv_cache.max_seq_len()));

                ops::sdpa_causal_cached_int8_dynamic(
                    q, bufs.act_attn_out, bufs.act_q,
                    kv_cache.k_cache_int8(full_idx), kv_cache.v_cache_int8(full_idx),
                    kv_cache.k_scale(full_idx), kv_cache.v_scale(full_idx),
                    kv_cache.k_zero_point(full_idx), kv_cache.v_zero_point(full_idx),
                    d_dynamic_pos, seq_len, num_q_heads, cfg.num_key_value_heads, head_dim, 0.0f,
                    static_cast<int64_t>(kv_cache.max_seq_len()));
            } else {
                ops::attention_write_kv_cache_int8(
                    q, kv_cache.k_cache_int8(full_idx), kv_cache.v_cache_int8(full_idx),
                    kv_cache.k_scale(full_idx), kv_cache.v_scale(full_idx),
                    kv_cache.k_zero_point(full_idx), kv_cache.v_zero_point(full_idx),
                    bufs.act_k, bufs.act_v, start_pos, seq_len, cfg.num_key_value_heads, head_dim,
                    static_cast<int64_t>(kv_cache.max_seq_len()));

                ops::sdpa_causal_cached_int8(
                    q, bufs.act_attn_out, bufs.act_q,
                    kv_cache.k_cache_int8(full_idx), kv_cache.v_cache_int8(full_idx),
                    kv_cache.k_scale(full_idx), kv_cache.v_scale(full_idx),
                    kv_cache.k_zero_point(full_idx), kv_cache.v_zero_point(full_idx),
                    start_pos, seq_len, num_q_heads, cfg.num_key_value_heads, head_dim, 0.0f,
                    static_cast<int64_t>(kv_cache.max_seq_len()));
            }
        } else {
            if (d_dynamic_pos) {
                ops::attention_write_kv_cache_dynamic(q, kv_cache.k_cache(full_idx), kv_cache.v_cache(full_idx),
                                                      bufs.act_k, bufs.act_v, d_dynamic_pos, seq_len, cfg.num_key_value_heads, head_dim,
                                                      static_cast<int64_t>(kv_cache.max_seq_len()));

                ops::sdpa_causal_cached_dynamic(q, bufs.act_attn_out, bufs.act_q,
                                                kv_cache.k_cache(full_idx), kv_cache.v_cache(full_idx),
                                                d_dynamic_pos, seq_len, num_q_heads, cfg.num_key_value_heads, head_dim, 0.0f,
                                                static_cast<int64_t>(kv_cache.max_seq_len()));
            } else {
                ops::attention_write_kv_cache(q, kv_cache.k_cache(full_idx), kv_cache.v_cache(full_idx),
                                              bufs.act_k, bufs.act_v, start_pos, seq_len, cfg.num_key_value_heads, head_dim,
                                              static_cast<int64_t>(kv_cache.max_seq_len()));

                ops::sdpa_causal_cached(q, bufs.act_attn_out, bufs.act_q,
                                        kv_cache.k_cache(full_idx), kv_cache.v_cache(full_idx),
                                        start_pos, seq_len, num_q_heads, cfg.num_key_value_heads, head_dim, 0.0f,
                                        static_cast<int64_t>(kv_cache.max_seq_len()));
            }
        }

        // Output gating: attn_out *= sigmoid(gate) matching torch.sigmoid(gate)
        sycl::half* attn_out_ptr = bufs.act_attn_out;
        q.parallel_for(sycl::range<2>(seq_len, num_q_heads), [=](sycl::id<2> idx) {
            int64_t t = idx[0];
            int64_t h = idx[1];
            for (int d = 0; d < head_dim; ++d) {
                float gate_val = static_cast<float>(q_gate_ptr[t * q_gate_dim + h * 2 * head_dim + head_dim + d]);
                float sig = 1.0f / (1.0f + sycl::exp(-gate_val));
                float cur_val = static_cast<float>(attn_out_ptr[(t * num_q_heads + h) * head_dim + d]);
                attn_out_ptr[(t * num_q_heads + h) * head_dim + d] = static_cast<sycl::half>(cur_val * sig);
            }
        });

        // Out projection
        ops::linear_int4(q, bufs.act_proj_out, bufs.act_attn_out,
                         static_cast<const uint8_t*>(layer.o_proj.d_weights_int4),
                         static_cast<const sycl::half*>(layer.o_proj.d_scales),
                         nullptr, seq_len, hidden_size, cfg.full_out_dim());
        full_idx++;
    } else {
        // Linear attention:
        // Fused in_proj_qkv + in_proj_z + in_proj_b + in_proj_a wide GEMV launch
        ops::FusedProjectionDesc la_projs[4] = {
            {bufs.act_qkv_raw, static_cast<const uint8_t*>(layer.in_proj_qkv.d_weights_int4),
             static_cast<const sycl::half*>(layer.in_proj_qkv.d_scales), nullptr, cfg.linear_conv_channels},
            {bufs.act_z, static_cast<const uint8_t*>(layer.in_proj_z.d_weights_int4),
             static_cast<const sycl::half*>(layer.in_proj_z.d_scales), nullptr, cfg.linear_z_dim},
            {bufs.act_b, static_cast<const uint8_t*>(layer.in_proj_b.d_weights_int4),
             static_cast<const sycl::half*>(layer.in_proj_b.d_scales), nullptr, cfg.linear_b_dim},
            {bufs.act_a, static_cast<const uint8_t*>(layer.in_proj_a.d_weights_int4),
             static_cast<const sycl::half*>(layer.in_proj_a.d_scales), nullptr, cfg.linear_a_dim}
        };
        ops::linear_int4_fused(q, bufs.act_normed, la_projs, 4, seq_len, hidden_size);

        // Stateful Causal Conv1d + SiLU
        causal_conv1d_silu(q, bufs.act_qkv_conv, bufs.act_qkv_raw, layer.d_conv1d_weight, seq_len,
                           kv_cache.conv_state(linear_idx), cfg.linear_conv_channels);

        // Stateful Recurrent Gated Delta Net + RMSNormGated
        recurrent_gated_delta_net(q, bufs.act_delta_out, bufs.act_qkv_conv, bufs.act_z, bufs.act_b, bufs.act_a,
                                  layer.d_A_log, layer.d_dt_bias, layer.d_norm_weight,
                                  kv_cache.linear_state(linear_idx), seq_len, zero_linear_state,
                                  cfg.linear_num_v_heads, cfg.linear_num_k_heads,
                                  cfg.linear_head_k_dim, cfg.linear_head_v_dim,
                                  cfg.linear_conv_channels, cfg.linear_z_dim);

        // Out projection
        ops::linear_int4(q, bufs.act_proj_out, bufs.act_delta_out,
                         static_cast<const uint8_t*>(layer.out_proj.d_weights_int4),
                         static_cast<const sycl::half*>(layer.out_proj.d_scales),
                         nullptr, seq_len, hidden_size, cfg.linear_z_dim);
        linear_idx++;
    }

    // Residual connection: act_x += act_proj_out
    ops::add_inplace(q, bufs.act_x, bufs.act_proj_out, seq_len * hidden_size);

    // MLP
    ops::rmsnorm(q, bufs.act_normed, bufs.act_x, layer.d_post_attention_layernorm, seq_len, hidden_size);

    // Fused MLP Gate + Up + SwiGLU: SiLU(gate) * up computed directly in sub-group registers
    ops::mlp_gate_up_swiglu_int4(q, bufs.act_mlp_gate, bufs.act_normed,
                                 static_cast<const uint8_t*>(layer.gate_proj.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.gate_proj.d_scales),
                                 static_cast<const uint8_t*>(layer.up_proj.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.up_proj.d_scales),
                                 seq_len, intermediate_size, hidden_size);

    // Down projection
    ops::linear_int4(q, bufs.act_proj_out, bufs.act_mlp_gate,
                     static_cast<const uint8_t*>(layer.down_proj.d_weights_int4),
                     static_cast<const sycl::half*>(layer.down_proj.d_scales),
                     nullptr, seq_len, hidden_size, intermediate_size);

    // Residual connection: act_x += act_proj_out
    ops::add_inplace(q, bufs.act_x, bufs.act_proj_out, seq_len * hidden_size);
}

void forward_lm_head(sycl::queue& q,
                     const qwen3_8_27b::LoadedModel& model,
                     sycl::half* act_normed,
                     const sycl::half* act_x_last,
                     float* out_logits) {
    const auto& cfg = model.config();
    const int64_t hidden_size = cfg.hidden_size;

    ops::rmsnorm(q, act_normed, act_x_last, model.d_final_norm(), 1, hidden_size);

    const auto& lm_head = model.lm_head();
    ops::linear_int4(q, out_logits, act_normed,
                     static_cast<const uint8_t*>(lm_head.d_weights_int4),
                     static_cast<const sycl::half*>(lm_head.d_scales),
                     nullptr, 1, cfg.vocab_size, hidden_size);
}

void forward_chunk(std::shared_ptr<core::DeviceContext> ctx,
                   core::DeviceArena& arena,
                   const qwen3_8_27b::LoadedModel& model,
                   core::KVCache& kv_cache,
                   const int64_t* token_ids,
                   int64_t seq_len,
                   int64_t start_pos,
                   bool zero_linear_state,
                   float* out_last_token_logits,
                   bool is_graph_capture,
                   const LayerActivationBuffers* preallocated_bufs,
                   int64_t* preallocated_token_ids,
                   int64_t* preallocated_positions,
                   const VisionInput* vision_input,
                   const int64_t* d_visual_indices) {
    if (seq_len <= 0) return;
    if (start_pos < 0 || start_pos + seq_len > static_cast<int64_t>(kv_cache.max_seq_len())) {
        std::cerr << "[xinfer::qwen3_8] Error: forward_chunk bounds exceeded: start_pos="
                  << start_pos << ", seq_len=" << seq_len << ", max_seq_len="
                  << kv_cache.max_seq_len() << std::endl;
        return;
    }
    sycl::queue& q = ctx->queue();

    // Graph Capture Contract: never reset arena during graph capture or when using preallocated buffers
    if (!is_graph_capture && !preallocated_bufs) {
        arena.reset();
    }

    // Copy token IDs to device
    int64_t* d_token_ids = preallocated_token_ids ? preallocated_token_ids
        : static_cast<int64_t*>(arena.allocate(seq_len * sizeof(int64_t)));
    ctx->copy_host_to_device(d_token_ids, token_ids, seq_len * sizeof(int64_t), true);

    // Positions for RoPE (start_pos ... start_pos + seq_len - 1)
    int64_t* d_positions = preallocated_positions ? preallocated_positions
        : static_cast<int64_t*>(arena.allocate(seq_len * sizeof(int64_t)));
    std::vector<int64_t> host_pos(seq_len);
    for (int64_t i = 0; i < seq_len; ++i) host_pos[i] = start_pos + i;
    q.parallel_for(sycl::range<1>(seq_len), [=](sycl::id<1> idx) {
        d_positions[idx[0]] = start_pos + static_cast<int64_t>(idx[0]);
    });

    // Common activation buffers (FP16 / sycl::half for 2x GDDR6 bandwidth efficiency)
    const auto& cfg = model.config();
    const int64_t hidden_size = cfg.hidden_size;
    const int64_t intermediate_size = cfg.intermediate_size;

    LayerActivationBuffers bufs;
    if (preallocated_bufs) {
        bufs = *preallocated_bufs;
    } else {
        bufs.act_x        = static_cast<sycl::half*>(arena.allocate(seq_len * hidden_size * sizeof(sycl::half)));
        bufs.act_normed   = static_cast<sycl::half*>(arena.allocate(seq_len * hidden_size * sizeof(sycl::half)));
        bufs.act_proj_out = static_cast<sycl::half*>(arena.allocate(seq_len * hidden_size * sizeof(sycl::half)));
        bufs.act_mlp_gate = static_cast<sycl::half*>(arena.allocate(seq_len * intermediate_size * sizeof(sycl::half)));

        bufs.act_q_gate   = static_cast<sycl::half*>(arena.allocate(seq_len * cfg.full_q_gate_dim() * sizeof(sycl::half)));
        bufs.act_q        = static_cast<sycl::half*>(arena.allocate(seq_len * cfg.full_q_dim() * sizeof(sycl::half)));
        bufs.act_k        = static_cast<sycl::half*>(arena.allocate(seq_len * cfg.full_k_dim() * sizeof(sycl::half)));
        bufs.act_v        = static_cast<sycl::half*>(arena.allocate(seq_len * cfg.full_v_dim() * sizeof(sycl::half)));
        bufs.act_attn_out = static_cast<sycl::half*>(arena.allocate(seq_len * cfg.full_out_dim() * sizeof(sycl::half)));

        bufs.act_qkv_raw   = static_cast<sycl::half*>(arena.allocate(seq_len * cfg.linear_conv_channels * sizeof(sycl::half)));
        bufs.act_qkv_conv  = static_cast<sycl::half*>(arena.allocate(seq_len * cfg.linear_conv_channels * sizeof(sycl::half)));
        bufs.act_z         = static_cast<sycl::half*>(arena.allocate(seq_len * cfg.linear_z_dim * sizeof(sycl::half)));
        bufs.act_b         = static_cast<sycl::half*>(arena.allocate(seq_len * cfg.linear_b_dim * sizeof(sycl::half)));
        bufs.act_a         = static_cast<sycl::half*>(arena.allocate(seq_len * cfg.linear_a_dim * sizeof(sycl::half)));
        bufs.act_delta_out = static_cast<sycl::half*>(arena.allocate(seq_len * cfg.linear_z_dim * sizeof(sycl::half)));
    }

    // 1. Initial embedding lookup with Zero-Copy Visual Token Injection
    const sycl::half* d_vit_embeddings = (vision_input && vision_input->d_vit_embeddings) ? vision_input->d_vit_embeddings : nullptr;
    embed_tokens_lookup(q, bufs.act_x, model.d_embed_tokens(), d_token_ids, seq_len, hidden_size, cfg.vocab_size,
                        d_vit_embeddings, cfg.visual_token_start, cfg.visual_token_end, d_visual_indices);

    // 2. Loop over layers using shared parameterized forward_layer
    const auto& layers = model.layers();
    size_t full_idx = 0;
    size_t linear_idx = 0;

    for (size_t l = 0; l < layers.size(); ++l) {
        forward_layer(q, cfg, layers[l], kv_cache, full_idx, linear_idx, bufs,
                      seq_len, d_positions, start_pos, nullptr, zero_linear_state);
    }

    // 3. Optional LM Head projection for the last token in chunk
    if (out_last_token_logits) {
        forward_lm_head(q, model, bufs.act_normed, bufs.act_x + (seq_len - 1) * hidden_size, out_last_token_logits);
    }
}

int64_t prefill_prompt(std::shared_ptr<core::DeviceContext> ctx,
                       core::DeviceArena& arena,
                       const qwen3_8_27b::LoadedModel& model,
                       core::KVCache& kv_cache,
                       const std::vector<int64_t>& prompt_tokens,
                       size_t chunk_size,
                       bool is_graph_capture,
                       const VisionInput* vision_input) {
    if (prompt_tokens.empty()) return 0;
    if (prompt_tokens.size() > kv_cache.max_seq_len()) {
        std::string err = "context_length_exceeded: Prompt tokens (" + std::to_string(prompt_tokens.size()) +
                          ") exceeds KV cache max_seq_len (" + std::to_string(kv_cache.max_seq_len()) + ")";
        std::cerr << "[xinfer::qwen3_8] Error: " << err << std::endl;
        throw context_length_exceeded(err);
    }
    const auto& cfg = model.config();

    // Visual placeholder token counting and strict fail-loud verification
    size_t visual_token_count = 0;
    std::vector<int64_t> visual_indices_host(prompt_tokens.size(), -1);
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        int64_t tok = prompt_tokens[i];
        if (tok >= cfg.visual_token_start && tok <= cfg.visual_token_end) {
            visual_indices_host[i] = static_cast<int64_t>(visual_token_count);
            visual_token_count++;
        }
    }

    size_t patches_per_img = cfg.patches_per_image > 0 ? static_cast<size_t>(cfg.patches_per_image) : 256;
    size_t expected_images_from_tags = visual_token_count / patches_per_img;

    if (visual_token_count > 0 && visual_token_count % patches_per_img != 0) {
        std::string err = "vision_format_error: Visual placeholder tokens (" + std::to_string(visual_token_count) +
                          ") is not a multiple of expected patches per image (" + std::to_string(patches_per_img) + ")";
        std::cerr << "[xinfer::qwen3_8] Error: " << err << std::endl;
        throw vision_format_error(err);
    }

    // Fail-loud contract checks:
    size_t provided_images = vision_input ? vision_input->num_images : 0;
    if (provided_images > 0 && expected_images_from_tags == 0) {
        std::string err = "vision_format_error: User provided " + std::to_string(provided_images) +
                          " image(s) but prompt contains 0 <image> tags";
        std::cerr << "[xinfer::qwen3_8] Error: " << err << std::endl;
        throw vision_format_error(err);
    }
    if (provided_images == 0 && expected_images_from_tags > 0) {
        std::string err = "vision_format_error: Prompt contains " + std::to_string(expected_images_from_tags) +
                          " <image> tag(s) (" + std::to_string(visual_token_count) + " patches) but no image was provided";
        std::cerr << "[xinfer::qwen3_8] Error: " << err << std::endl;
        throw vision_format_error(err);
    }
    if (provided_images > 0 && vision_input) {
        if (provided_images != expected_images_from_tags) {
            std::string err = "vision_format_error: Number of provided images (" + std::to_string(provided_images) +
                              ") does not match <image> tags in prompt (" + std::to_string(expected_images_from_tags) + ")";
            std::cerr << "[xinfer::qwen3_8] Error: " << err << std::endl;
            throw vision_format_error(err);
        }
        if (vision_input->num_patches != visual_token_count) {
            std::string err = "vision_format_error: ViT output tensor dimensions (" + std::to_string(vision_input->num_patches) +
                              " patches) does not match expected prompt visual patches (" + std::to_string(visual_token_count) + ")";
            std::cerr << "[xinfer::qwen3_8] Error: " << err << std::endl;
            throw vision_format_error(err);
        }
        if (!vision_input->d_vit_embeddings) {
            std::string err = "vision_format_error: ViT output USM buffer pointer is null";
            std::cerr << "[xinfer::qwen3_8] Error: " << err << std::endl;
            throw vision_format_error(err);
        }
    }

    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        if (prompt_tokens[i] < 0 || prompt_tokens[i] >= cfg.vocab_size) {
            std::cerr << "[xinfer::qwen3_8] Error: Prompt token at index " << i << " ("
                      << prompt_tokens[i] << ") out of vocabulary range [0, "
                      << cfg.vocab_size << ")" << std::endl;
            return -1;
        }
    }
    if (chunk_size == 0) chunk_size = 512;

    kv_cache.clear();

    float* d_logits = static_cast<float*>(arena.persistent_buffer(cfg.vocab_size * sizeof(float)));

    int64_t* d_all_visual_indices = nullptr;
    if (visual_token_count > 0) {
        d_all_visual_indices = static_cast<int64_t*>(arena.allocate(prompt_tokens.size() * sizeof(int64_t)));
        ctx->copy_host_to_device(d_all_visual_indices, visual_indices_host.data(), prompt_tokens.size() * sizeof(int64_t), true);
    }

    // Graph Capture Contract: pre-allocate persistent activation buffers for chunk_size in DeviceArena
    // before beginning graph recording / chunk loop to guarantee address stability without resetting arena
    LayerActivationBuffers chunk_bufs;
    int64_t* d_chunk_tokens = nullptr;
    int64_t* d_chunk_positions = nullptr;

    if (is_graph_capture) {
        d_chunk_tokens = static_cast<int64_t*>(arena.allocate_persistent(chunk_size * sizeof(int64_t)));
        d_chunk_positions = static_cast<int64_t*>(arena.allocate_persistent(chunk_size * sizeof(int64_t)));

        chunk_bufs.act_x        = static_cast<sycl::half*>(arena.allocate_persistent(chunk_size * cfg.hidden_size * sizeof(sycl::half)));
        chunk_bufs.act_normed   = static_cast<sycl::half*>(arena.allocate_persistent(chunk_size * cfg.hidden_size * sizeof(sycl::half)));
        chunk_bufs.act_proj_out = static_cast<sycl::half*>(arena.allocate_persistent(chunk_size * cfg.hidden_size * sizeof(sycl::half)));
        chunk_bufs.act_mlp_gate = static_cast<sycl::half*>(arena.allocate_persistent(chunk_size * cfg.intermediate_size * sizeof(sycl::half)));

        chunk_bufs.act_q_gate   = static_cast<sycl::half*>(arena.allocate_persistent(chunk_size * cfg.full_q_gate_dim() * sizeof(sycl::half)));
        chunk_bufs.act_q        = static_cast<sycl::half*>(arena.allocate_persistent(chunk_size * cfg.full_q_dim() * sizeof(sycl::half)));
        chunk_bufs.act_k        = static_cast<sycl::half*>(arena.allocate_persistent(chunk_size * cfg.full_k_dim() * sizeof(sycl::half)));
        chunk_bufs.act_v        = static_cast<sycl::half*>(arena.allocate_persistent(chunk_size * cfg.full_v_dim() * sizeof(sycl::half)));
        chunk_bufs.act_attn_out = static_cast<sycl::half*>(arena.allocate_persistent(chunk_size * cfg.full_out_dim() * sizeof(sycl::half)));

        chunk_bufs.act_qkv_raw   = static_cast<sycl::half*>(arena.allocate_persistent(chunk_size * cfg.linear_conv_channels * sizeof(sycl::half)));
        chunk_bufs.act_qkv_conv  = static_cast<sycl::half*>(arena.allocate_persistent(chunk_size * cfg.linear_conv_channels * sizeof(sycl::half)));
        chunk_bufs.act_z         = static_cast<sycl::half*>(arena.allocate_persistent(chunk_size * cfg.linear_z_dim * sizeof(sycl::half)));
        chunk_bufs.act_b         = static_cast<sycl::half*>(arena.allocate_persistent(chunk_size * cfg.linear_b_dim * sizeof(sycl::half)));
        chunk_bufs.act_a         = static_cast<sycl::half*>(arena.allocate_persistent(chunk_size * cfg.linear_a_dim * sizeof(sycl::half)));
        chunk_bufs.act_delta_out = static_cast<sycl::half*>(arena.allocate_persistent(chunk_size * cfg.linear_z_dim * sizeof(sycl::half)));
    }

    size_t total_tokens = prompt_tokens.size();
    size_t offset = 0;
    int64_t start_pos = 0;

    while (offset < total_tokens) {
        size_t cur_chunk = std::min(chunk_size, total_tokens - offset);
        bool is_last = (offset + cur_chunk == total_tokens);
        bool zero_linear = (offset == 0);

        forward_chunk(ctx, arena, model, kv_cache,
                      prompt_tokens.data() + offset,
                      cur_chunk,
                      start_pos,
                      zero_linear,
                      is_last ? d_logits : nullptr,
                      is_graph_capture,
                      is_graph_capture ? &chunk_bufs : nullptr,
                      d_chunk_tokens,
                      d_chunk_positions,
                      vision_input,
                      d_all_visual_indices ? (d_all_visual_indices + offset) : nullptr);

        start_pos += static_cast<int64_t>(cur_chunk);
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
    if (input_token_id < 0 || input_token_id >= cfg.vocab_size) {
        std::cerr << "[xinfer::qwen3_8] Error: decode_step called with input_token_id ("
                  << input_token_id << ") out of vocabulary range [0, "
                  << cfg.vocab_size << ")" << std::endl;
        return -1;
    }

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

DraftDecodeResult draft_decode_loop(std::shared_ptr<core::DeviceContext> ctx,
                                   core::DeviceArena& draft_arena,
                                   const qwen3_8_27b::LoadedModel& model,
                                   core::KVCache& kv_cache,
                                   DecodeGraph* draft_graph,
                                   int64_t current_token_id,
                                   size_t num_draft_tokens) {
    DraftDecodeResult result;
    if (num_draft_tokens == 0) {
        return result;
    }
    if (kv_cache.current_seq_len() + num_draft_tokens > kv_cache.max_seq_len()) {
        result.success = false;
        result.error_msg = "Draft decode exceeds KV cache max_seq_len: cur=" +
                           std::to_string(kv_cache.current_seq_len()) + " + draft=" +
                           std::to_string(num_draft_tokens) + " > max=" +
                           std::to_string(kv_cache.max_seq_len());
        return result;
    }

    int64_t in_tok = current_token_id;
    result.tokens.reserve(num_draft_tokens);

    for (size_t step = 0; step < num_draft_tokens; ++step) {
        if (kv_cache.current_seq_len() >= kv_cache.max_seq_len()) {
            break;
        }

        int64_t next_tok = -1;
        if (draft_graph && draft_graph->is_captured()) {
            next_tok = draft_graph->decode_step(in_tok, kv_cache.current_seq_len());
            if (next_tok < 0 || !kv_cache.advance(1)) {
                result.success = false;
                result.error_msg = "Draft graph decode step failed or KV cache overflow at step " + std::to_string(step);
                return result;
            }
        } else {
            next_tok = decode_step(ctx, draft_arena, model, kv_cache, in_tok);
            if (next_tok < 0) {
                result.success = false;
                result.error_msg = "Draft eager decode step failed at step " + std::to_string(step);
                return result;
            }
        }

        result.tokens.push_back(next_tok);
        in_tok = next_tok;
    }

    return result;
}

void forward_verify(std::shared_ptr<core::DeviceContext> ctx,
                    core::DeviceArena& arena,
                    const qwen3_8_27b::LoadedModel& model,
                    core::KVCache& kv_cache,
                    const int64_t* draft_tokens,
                    int64_t num_draft_tokens,
                    int64_t prefix_len,
                    float* out_logits) {
    if (num_draft_tokens <= 0 || !draft_tokens || !out_logits) return;

    if (prefix_len < 0 || prefix_len + num_draft_tokens > static_cast<int64_t>(kv_cache.max_seq_len())) {
        std::string err = "context_length_exceeded: forward_verify bounds exceeded: prefix_len=" +
                          std::to_string(prefix_len) + " + draft=" + std::to_string(num_draft_tokens) +
                          " > max=" + std::to_string(kv_cache.max_seq_len());
        std::cerr << "[xinfer::qwen3_8] Error: " << err << std::endl;
        throw context_length_exceeded(err);
    }

    const auto& cfg = model.config();
    for (int64_t i = 0; i < num_draft_tokens; ++i) {
        if (draft_tokens[i] < 0 || draft_tokens[i] >= cfg.vocab_size) {
            throw std::out_of_range("forward_verify: draft token at index " + std::to_string(i) +
                                    " (" + std::to_string(draft_tokens[i]) + ") out of vocab range");
        }
    }

    sycl::queue& q = ctx->queue();
    const int64_t hidden_size = cfg.hidden_size;
    const int64_t intermediate_size = cfg.intermediate_size;

    // Allocate device buffers for the M = num_draft_tokens batch
    int64_t* d_token_ids = static_cast<int64_t*>(arena.allocate(num_draft_tokens * sizeof(int64_t)));
    ctx->copy_host_to_device(d_token_ids, draft_tokens, num_draft_tokens * sizeof(int64_t), true);

    int64_t* d_positions = static_cast<int64_t*>(arena.allocate(num_draft_tokens * sizeof(int64_t)));
    std::vector<int64_t> host_pos(num_draft_tokens);
    for (int64_t i = 0; i < num_draft_tokens; ++i) {
        host_pos[i] = prefix_len + i;
    }
    ctx->copy_host_to_device(d_positions, host_pos.data(), num_draft_tokens * sizeof(int64_t), true);

    LayerActivationBuffers bufs;
    bufs.act_x        = static_cast<sycl::half*>(arena.allocate(num_draft_tokens * hidden_size * sizeof(sycl::half)));
    bufs.act_normed   = static_cast<sycl::half*>(arena.allocate(num_draft_tokens * hidden_size * sizeof(sycl::half)));
    bufs.act_proj_out = static_cast<sycl::half*>(arena.allocate(num_draft_tokens * hidden_size * sizeof(sycl::half)));
    bufs.act_mlp_gate = static_cast<sycl::half*>(arena.allocate(num_draft_tokens * intermediate_size * sizeof(sycl::half)));

    bufs.act_q_gate   = static_cast<sycl::half*>(arena.allocate(num_draft_tokens * cfg.full_q_gate_dim() * sizeof(sycl::half)));
    bufs.act_q        = static_cast<sycl::half*>(arena.allocate(num_draft_tokens * cfg.full_q_dim() * sizeof(sycl::half)));
    bufs.act_k        = static_cast<sycl::half*>(arena.allocate(num_draft_tokens * cfg.full_k_dim() * sizeof(sycl::half)));
    bufs.act_v        = static_cast<sycl::half*>(arena.allocate(num_draft_tokens * cfg.full_v_dim() * sizeof(sycl::half)));
    bufs.act_attn_out = static_cast<sycl::half*>(arena.allocate(num_draft_tokens * cfg.full_out_dim() * sizeof(sycl::half)));

    bufs.act_qkv_raw   = static_cast<sycl::half*>(arena.allocate(num_draft_tokens * cfg.linear_conv_channels * sizeof(sycl::half)));
    bufs.act_qkv_conv  = static_cast<sycl::half*>(arena.allocate(num_draft_tokens * cfg.linear_conv_channels * sizeof(sycl::half)));
    bufs.act_z         = static_cast<sycl::half*>(arena.allocate(num_draft_tokens * cfg.linear_z_dim * sizeof(sycl::half)));
    bufs.act_b         = static_cast<sycl::half*>(arena.allocate(num_draft_tokens * cfg.linear_b_dim * sizeof(sycl::half)));
    bufs.act_a         = static_cast<sycl::half*>(arena.allocate(num_draft_tokens * cfg.linear_a_dim * sizeof(sycl::half)));
    bufs.act_delta_out = static_cast<sycl::half*>(arena.allocate(num_draft_tokens * cfg.linear_z_dim * sizeof(sycl::half)));

    // 1. Initial embedding lookup for all N draft tokens
    embed_tokens_lookup(q, bufs.act_x, model.d_embed_tokens(), d_token_ids, num_draft_tokens, hidden_size, cfg.vocab_size);

    // 2. Forward layers with causal masking across draft window (attending to prefix + prior draft tokens)
    const auto& layers = model.layers();
    size_t full_idx = 0;
    size_t linear_idx = 0;

    for (size_t l = 0; l < layers.size(); ++l) {
        forward_layer(q, cfg, layers[l], kv_cache, full_idx, linear_idx, bufs,
                      num_draft_tokens, d_positions, prefix_len, nullptr, false);
    }

    // 3. Batched LM head projection for all N draft tokens: [num_draft_tokens, vocab_size]
    ops::rmsnorm(q, bufs.act_normed, bufs.act_x, model.d_final_norm(), num_draft_tokens, hidden_size);

    const auto& lm_head = model.lm_head();
    ops::linear_int4(q, out_logits, bufs.act_normed,
                     static_cast<const uint8_t*>(lm_head.d_weights_int4),
                     static_cast<const sycl::half*>(lm_head.d_scales),
                     nullptr, num_draft_tokens, cfg.vocab_size, hidden_size);
    q.wait();
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
