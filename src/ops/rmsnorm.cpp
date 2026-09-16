#include "rmsnorm.h"
#include <cmath>

namespace xinfer::ops {

void rmsnorm(sycl::queue& q,
             float* out,
             const float* in,
             const float* weight,
             int64_t num_tokens,
             int64_t hidden_size,
             float eps) {
    if (num_tokens <= 0 || hidden_size <= 0) return;

    // Launch 1 work-item per token for naive correctness
    q.parallel_for(sycl::range<1>(static_cast<size_t>(num_tokens)), [=](sycl::id<1> idx) {
        int64_t t = idx[0];
        const float* x = in + t * hidden_size;
        float* y = out + t * hidden_size;

        float sum_sq = 0.0f;
        for (int64_t i = 0; i < hidden_size; ++i) {
            float val = x[i];
            sum_sq += val * val;
        }

        float mean_sq = sum_sq / static_cast<float>(hidden_size);
        float rsqrt_val = 1.0f / sycl::sqrt(mean_sq + eps);

        for (int64_t i = 0; i < hidden_size; ++i) {
            y[i] = x[i] * rsqrt_val * weight[i];
        }
    });
}

void rmsnorm_residual(sycl::queue& q,
                      float* out,
                      float* residual,
                      const float* in,
                      const float* weight,
                      int64_t num_tokens,
                      int64_t hidden_size,
                      float eps) {
    if (num_tokens <= 0 || hidden_size <= 0) return;

    q.parallel_for(sycl::range<1>(static_cast<size_t>(num_tokens)), [=](sycl::id<1> idx) {
        int64_t t = idx[0];
        const float* x = in + t * hidden_size;
        float* res = residual + t * hidden_size;
        float* y = out + t * hidden_size;

        float sum_sq = 0.0f;
        for (int64_t i = 0; i < hidden_size; ++i) {
            res[i] += x[i];
            float val = res[i];
            sum_sq += val * val;
        }

        float mean_sq = sum_sq / static_cast<float>(hidden_size);
        float rsqrt_val = 1.0f / sycl::sqrt(mean_sq + eps);

        for (int64_t i = 0; i < hidden_size; ++i) {
            y[i] = res[i] * rsqrt_val * weight[i];
        }
    });
}

} // namespace xinfer::ops
