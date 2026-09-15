#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

namespace xinfer::ops {

// Greedy argmax sampling: returns the token ID with the maximum logit value
int64_t argmax(sycl::queue& q, const float* logits, int64_t vocab_size);

} // namespace xinfer::ops
