#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstddef>

namespace xinfer::ops {

// Helper to calculate the required scratch element capacity for partial reduction.
// Capacity needed: ceil(vocab_size / WG_SIZE), where WG_SIZE = 256.
inline size_t argmax_scratch_size(int64_t vocab_size) {
    if (vocab_size <= 0) return 0;
    constexpr size_t WG_SIZE = 256;
    return (static_cast<size_t>(vocab_size) + WG_SIZE - 1) / WG_SIZE;
}

// Greedy argmax sampling: returns the token ID with the maximum logit value.
// Callers can supply preallocated USM shared-memory scratch buffers (partial_max and partial_idx)
// of size >= argmax_scratch_size(vocab_size) to avoid per-call allocations.
// If nullptr, temporary shared USM is allocated and freed within the call.
int64_t argmax(sycl::queue& q,
               const float* logits,
               int64_t vocab_size,
               float* partial_max = nullptr,
               int64_t* partial_idx = nullptr);

struct SpeculativeAcceptanceResult {
    std::vector<int64_t> accepted_tokens;
    int64_t bonus_or_resampled_token{-1};
    bool has_rejected_token{false};
    int64_t rejected_token{-1};
    size_t num_accepted{0};
    size_t num_rejected{0};
    size_t num_resampled{0};
    size_t num_remaining{0};
};

// Speculative acceptance/rejection logic (Substep 2)
// Compares draft model's probability distribution P_draft with verified distribution P_target.
// Evaluates ratio r < P_target(x_i) / P_draft(x_i).
// Enforces fail-loud invariant: accepted + 1 (rejected) + remaining == N.
SpeculativeAcceptanceResult speculative_accept_reject(
    sycl::queue& q,
    const int64_t* draft_tokens,
    size_t num_draft_tokens,
    const float* target_logits,
    int64_t vocab_size,
    const float* draft_probs = nullptr,
    const float* rand_uniform = nullptr);

} // namespace xinfer::ops
