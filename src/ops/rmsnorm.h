#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

namespace xinfer::ops {

// Standard RMSNorm: out = (in / sqrt(mean(in^2) + eps)) * weight
sycl::event rmsnorm(sycl::queue& q,
                    float* out,
                    const float* in,
                    const float* weight,
                    int64_t num_tokens,
                    int64_t hidden_size,
                    float eps = 1e-6f);

sycl::event rmsnorm(sycl::queue& q,
                    sycl::half* out,
                    const sycl::half* in,
                    const float* weight,
                    int64_t num_tokens,
                    int64_t hidden_size,
                    float eps = 1e-6f);

// Fused RMSNorm with residual: residual += in; out = rmsnorm(residual, weight)
sycl::event rmsnorm_residual(sycl::queue& q,
                             float* out,
                             float* residual,
                             const float* in,
                             const float* weight,
                             int64_t num_tokens,
                             int64_t hidden_size,
                             float eps = 1e-6f);

sycl::event rmsnorm_residual(sycl::queue& q,
                             sycl::half* out,
                             sycl::half* residual,
                             const sycl::half* in,
                             const float* weight,
                             int64_t num_tokens,
                             int64_t hidden_size,
                             float eps = 1e-6f);

} // namespace xinfer::ops
