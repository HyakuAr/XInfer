#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace xinfer::ops {

// Result of a cross-entropy loss computation over a sequence.
struct CrossEntropyResult {
    double total_loss{0.0};       // Sum of per-token NLL losses
    size_t num_tokens{0};         // Number of tokens evaluated (sequence_len - 1 for causal)
    double perplexity{0.0};       // exp(total_loss / num_tokens)

    // Per-token diagnostics (populated when collect_per_token is true)
    std::vector<float> per_token_loss;    // NLL for each causal position
    std::vector<int64_t> per_token_rank;  // Rank of ground-truth token in logit ordering
};

// Numerically stable fused log_softmax + NLL loss kernel.
//
// Computes the causal cross-entropy loss for a sequence of logits:
//   For each position i in [0, seq_len-1):
//     loss_i = -log_softmax(logits[i])[target_token_ids[i+1]]
//
// This kernel runs entirely on the GPU to avoid copying vocab_size-wide
// logit vectors (248,320 floats ≈ 1MB per token) back over PCIe.
//
// Implementation uses a two-pass approach per row:
//   Pass 1: Compute row-max for numerical stability (workgroup reduction)
//   Pass 2: Compute log(sum(exp(x - max))) and extract target log-prob
//
// The partial losses are reduced across the sequence via sub-group
// reductions and a final host-side summation over workgroup partials.
//
// Parameters:
//   q             - SYCL queue for the target device
//   logits        - [seq_len, vocab_size] float logits (device USM)
//   token_ids     - [seq_len] int64 token IDs (device USM)
//                   token_ids[i+1] is the ground-truth label for logits[i]
//   seq_len       - Total number of tokens (including the prompt prefix)
//   vocab_size    - Vocabulary size (e.g. 248320 for Qwen3.8)
//   collect_per_token - If true, populate per_token_loss and per_token_rank
//
// Returns CrossEntropyResult with total_loss, num_tokens, and perplexity.
CrossEntropyResult cross_entropy_loss(
    sycl::queue& q,
    const float* logits,
    const int64_t* token_ids,
    int64_t seq_len,
    int64_t vocab_size,
    bool collect_per_token = false);

// Helper to calculate the required scratch element capacity for workgroup
// partial reductions. Capacity = ceil((seq_len - 1) / WG_SIZE) where WG_SIZE = 256.
inline size_t cross_entropy_scratch_size(int64_t seq_len) {
    if (seq_len <= 1) return 0;
    constexpr size_t WG_SIZE = 256;
    size_t num_positions = static_cast<size_t>(seq_len - 1);
    return (num_positions + WG_SIZE - 1) / WG_SIZE;
}

} // namespace xinfer::ops
