// Citing vendor documentation per AGENTS.md §5:
// - docs/vendor/xe-gpu-architecture.md (lines 48-50):
//   Vector Engine (VE) ALUs support native FP16 and FP32 operations.

#include "softmax.h"
#include <cmath>
#include <limits>

namespace xinfer::ops {

namespace {

template <typename T>
void softmax_impl(sycl::queue& q, T* out, const T* in, int64_t rows, int64_t cols) {
    if (rows <= 0 || cols <= 0) return;

    q.parallel_for(sycl::range<1>(static_cast<size_t>(rows)), [=](sycl::id<1> idx) {
        int64_t r = idx[0];
        const T* row_in = in + r * cols;
        T* row_out = out + r * cols;

        // 1. Max (FP32)
        float max_val = -std::numeric_limits<float>::infinity();
        for (int64_t c = 0; c < cols; ++c) {
            float val = static_cast<float>(row_in[c]);
            if (val > max_val) max_val = val;
        }

        // 2. Exp and Sum (FP32)
        float sum = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            float e = sycl::exp(static_cast<float>(row_in[c]) - max_val);
            row_out[c] = static_cast<T>(e);
            sum += e;
        }

        // 3. Normalize
        float inv_sum = 1.0f / (sum > 0.0f ? sum : 1e-12f);
        for (int64_t c = 0; c < cols; ++c) {
            row_out[c] = static_cast<T>(static_cast<float>(row_out[c]) * inv_sum);
        }
    }).wait();
}

template <typename T>
void softmax_causal_impl(sycl::queue& q, T* out, const T* in, int64_t rows, int64_t cols) {
    if (rows <= 0 || cols <= 0) return;

    q.parallel_for(sycl::range<1>(static_cast<size_t>(rows)), [=](sycl::id<1> idx) {
        int64_t r = idx[0];
        const T* row_in = in + r * cols;
        T* row_out = out + r * cols;

        int64_t max_c = sycl::min(r + 1, cols);

        // 1. Max (FP32)
        float max_val = -std::numeric_limits<float>::infinity();
        for (int64_t c = 0; c < max_c; ++c) {
            float val = static_cast<float>(row_in[c]);
            if (val > max_val) max_val = val;
        }

        // 2. Exp and Sum (FP32)
        float sum = 0.0f;
        for (int64_t c = 0; c < max_c; ++c) {
            float e = sycl::exp(static_cast<float>(row_in[c]) - max_val);
            row_out[c] = static_cast<T>(e);
            sum += e;
        }
        for (int64_t c = max_c; c < cols; ++c) {
            row_out[c] = static_cast<T>(0.0f);
        }

        // 3. Normalize
        float inv_sum = 1.0f / (sum > 0.0f ? sum : 1e-12f);
        for (int64_t c = 0; c < max_c; ++c) {
            row_out[c] = static_cast<T>(static_cast<float>(row_out[c]) * inv_sum);
        }
    }).wait();
}

} // anonymous namespace

void softmax(sycl::queue& q, float* out, const float* in, int64_t rows, int64_t cols) {
    softmax_impl<float>(q, out, in, rows, cols);
}

void softmax(sycl::queue& q, sycl::half* out, const sycl::half* in, int64_t rows, int64_t cols) {
    softmax_impl<sycl::half>(q, out, in, rows, cols);
}

void softmax_causal(sycl::queue& q, float* out, const float* in, int64_t rows, int64_t cols) {
    softmax_causal_impl<float>(q, out, in, rows, cols);
}

void softmax_causal(sycl::queue& q, sycl::half* out, const sycl::half* in, int64_t rows, int64_t cols) {
    softmax_causal_impl<sycl::half>(q, out, in, rows, cols);
}

} // namespace xinfer::ops
