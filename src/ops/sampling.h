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

} // namespace xinfer::ops
