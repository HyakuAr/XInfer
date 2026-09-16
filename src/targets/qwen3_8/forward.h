#pragma once

#include "core/device.h"
#include "core/arena.h"
#include "core/kv_cache.h"
#include "targets/qwen3_8_27b/weights.h"
#include <vector>
#include <cstdint>

namespace xinfer::targets::qwen3_8 {

// Looks up BF16 token embeddings and writes FP32 activations
void embed_tokens_lookup(sycl::queue& q,
                         float* out_act,
                         const void* embed_table_bf16,
                         const int64_t* d_token_ids,
                         int64_t num_tokens,
                         int64_t hidden_size);

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
                    int64_t input_token_id);

// Legacy forward pass without persistent cache (M5 baseline)
int64_t forward_next_token(std::shared_ptr<core::DeviceContext> ctx,
                           core::DeviceArena& arena,
                           const qwen3_8_27b::LoadedModel& model,
                           const std::vector<int64_t>& token_ids);

} // namespace xinfer::targets::qwen3_8
