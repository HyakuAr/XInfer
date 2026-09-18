#pragma once

#include "core/device.h"
#include "core/arena.h"
#include "core/kv_cache.h"
#include "targets/qwen3_8_27b/weights.h"
#include <vector>
#include <cstdint>

namespace xinfer::targets::qwen3_8 {

// Looks up BF16 token embeddings and writes FP16/FP32 activations
// vocab_size: bounds check limit (0 = unbounded, defaults to ModelConfig::kDefaultVocabSize)
sycl::event embed_tokens_lookup(sycl::queue& q,
                                 sycl::half* out_act,
                                 const void* embed_table_bf16,
                                 const int64_t* d_token_ids,
                                 int64_t num_tokens,
                                 int64_t hidden_size,
                                 int64_t vocab_size = qwen3_8_27b::ModelConfig::kDefaultVocabSize);

sycl::event embed_tokens_lookup(sycl::queue& q,
                                 float* out_act,
                                 const void* embed_table_bf16,
                                 const int64_t* d_token_ids,
                                 int64_t num_tokens,
                                 int64_t hidden_size,
                                 int64_t vocab_size = qwen3_8_27b::ModelConfig::kDefaultVocabSize);

// Common activation scratchpad buffers for a single forward layer (FP16 / sycl::half)
struct LayerActivationBuffers {
    sycl::half* act_x{nullptr};
    sycl::half* act_normed{nullptr};
    sycl::half* act_proj_out{nullptr};
    sycl::half* act_mlp_gate{nullptr};

    // Full Attention buffers
    sycl::half* act_q_gate{nullptr};
    sycl::half* act_q{nullptr};
    sycl::half* act_k{nullptr};
    sycl::half* act_v{nullptr};
    sycl::half* act_attn_out{nullptr};

    // Linear Attention buffers
    sycl::half* act_qkv_raw{nullptr};
    sycl::half* act_qkv_conv{nullptr};
    sycl::half* act_z{nullptr};
    sycl::half* act_b{nullptr};
    sycl::half* act_a{nullptr};
    sycl::half* act_delta_out{nullptr};
};

// Parameterized per-layer forward pass callable from both eager execution and graph capture
// Computes RMSNorm -> Attention (Full or Linear) -> Residual Add -> RMSNorm -> MLP SwiGLU -> Residual Add.
// If d_dynamic_pos is non-null, uses dynamic device-pointer overloads for KV cache write & SDPA (required for graph replay).
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
                   const int64_t* d_dynamic_pos = nullptr,
                   bool zero_linear_state = false);

// Final RMSNorm and LM Head projection computing logits from final hidden states
void forward_lm_head(sycl::queue& q,
                     const qwen3_8_27b::LoadedModel& model,
                     sycl::half* act_normed,
                     const sycl::half* act_x_last,
                     float* out_logits);

// Forward pass on a chunk of tokens using persistent KV cache and recurrent states
// If out_last_token_logits is non-null, computes the LM head logits for token at (seq_len - 1)
void forward_chunk(std::shared_ptr<core::DeviceContext> ctx,
                   core::DeviceArena& arena,
                   const qwen3_8_27b::LoadedModel& model,
                   core::KVCache& kv_cache,
                   const int64_t* token_ids,
                   int64_t seq_len,
                   int64_t start_pos,
                   bool zero_linear_state,
                   float* out_last_token_logits = nullptr);

// Prefills the prompt in chunks (chunk_size tokens each) and returns the first generated token ID
int64_t prefill_prompt(std::shared_ptr<core::DeviceContext> ctx,
                       core::DeviceArena& arena,
                       const qwen3_8_27b::LoadedModel& model,
                       core::KVCache& kv_cache,
                       const std::vector<int64_t>& prompt_tokens,
                       size_t chunk_size = 512);

// Decodes a single token at the current KV-cache sequence length and returns the next token ID
int64_t decode_step(std::shared_ptr<core::DeviceContext> ctx,
                    core::DeviceArena& arena,
                    const qwen3_8_27b::LoadedModel& model,
                    core::KVCache& kv_cache,
                    int64_t input_token_id,
                    float* d_logits = nullptr);

// Legacy forward pass without persistent cache (M5 baseline)
int64_t forward_next_token(std::shared_ptr<core::DeviceContext> ctx,
                           core::DeviceArena& arena,
                           const qwen3_8_27b::LoadedModel& model,
                           const std::vector<int64_t>& token_ids);

} // namespace xinfer::targets::qwen3_8
