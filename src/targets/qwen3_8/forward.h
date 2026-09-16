#pragma once

#include "core/device.h"
#include "core/arena.h"
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

// Forward pass for single sequence of tokens
// Returns the next predicted token ID (greedy argmax)
int64_t forward_next_token(std::shared_ptr<core::DeviceContext> ctx,
                           core::DeviceArena& arena,
                           const qwen3_8_27b::LoadedModel& model,
                           const std::vector<int64_t>& token_ids);

} // namespace xinfer::targets::qwen3_8
