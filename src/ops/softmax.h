#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

namespace xinfer::ops {

// Numerically stable row-wise softmax:
// out[r, c] = exp(in[r, c] - max(in[r])) / sum(exp(in[r] - max(in[r])))
void softmax(sycl::queue& q, float* out, const float* in, int64_t rows, int64_t cols);
void softmax(sycl::queue& q, sycl::half* out, const sycl::half* in, int64_t rows, int64_t cols);

// Row-wise softmax with causal masking (for row r, ignore col c > r)
void softmax_causal(sycl::queue& q, float* out, const float* in, int64_t rows, int64_t cols);
void softmax_causal(sycl::queue& q, sycl::half* out, const sycl::half* in, int64_t rows, int64_t cols);

} // namespace xinfer::ops
