#include "softmax.h"
#include <cmath>
#include <limits>

namespace xinfer::ops {

void softmax(sycl::queue& q, float* out, const float* in, int64_t rows, int64_t cols) {
    if (rows <= 0 || cols <= 0) return;

    q.parallel_for(sycl::range<1>(static_cast<size_t>(rows)), [=](sycl::id<1> idx) {
        int64_t r = idx[0];
        const float* row_in = in + r * cols;
        float* row_out = out + r * cols;

        // 1. Max
        float max_val = -std::numeric_limits<float>::infinity();
        for (int64_t c = 0; c < cols; ++c) {
            float val = row_in[c];
            if (val > max_val) max_val = val;
        }

        // 2. Exp and Sum
        float sum = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            float e = sycl::exp(row_in[c] - max_val);
            row_out[c] = e;
            sum += e;
        }

        // 3. Normalize
        float inv_sum = 1.0f / (sum > 0.0f ? sum : 1e-12f);
        for (int64_t c = 0; c < cols; ++c) {
            row_out[c] *= inv_sum;
        }
    }).wait();
}

void softmax_causal(sycl::queue& q, float* out, const float* in, int64_t rows, int64_t cols) {
    if (rows <= 0 || cols <= 0) return;

    q.parallel_for(sycl::range<1>(static_cast<size_t>(rows)), [=](sycl::id<1> idx) {
        int64_t r = idx[0];
        const float* row_in = in + r * cols;
        float* row_out = out + r * cols;

        int64_t max_c = sycl::min(r + 1, cols);

        // 1. Max
        float max_val = -std::numeric_limits<float>::infinity();
        for (int64_t c = 0; c < max_c; ++c) {
            float val = row_in[c];
            if (val > max_val) max_val = val;
        }

        // 2. Exp and Sum
        float sum = 0.0f;
        for (int64_t c = 0; c < max_c; ++c) {
            float e = sycl::exp(row_in[c] - max_val);
            row_out[c] = e;
            sum += e;
        }
        for (int64_t c = max_c; c < cols; ++c) {
            row_out[c] = 0.0f;
        }

        // 3. Normalize
        float inv_sum = 1.0f / (sum > 0.0f ? sum : 1e-12f);
        for (int64_t c = 0; c < max_c; ++c) {
            row_out[c] *= inv_sum;
        }
    }).wait();
}

} // namespace xinfer::ops
