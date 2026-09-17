#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

namespace xinfer::ops {

// SwiGLU: out = SiLU(gate) * up = (gate / (1 + exp(-gate))) * up
sycl::event swiglu(sycl::queue& q, float* out, const float* gate, const float* up, int64_t num_elements);
sycl::event swiglu(sycl::queue& q, sycl::half* out, const sycl::half* gate, const sycl::half* up, int64_t num_elements);

// SiLU: out = in / (1 + exp(-in))
sycl::event silu(sycl::queue& q, float* out, const float* in, int64_t num_elements);
sycl::event silu(sycl::queue& q, sycl::half* out, const sycl::half* in, int64_t num_elements);

// Elementwise addition: out = a + b
sycl::event add(sycl::queue& q, float* out, const float* a, const float* b, int64_t num_elements);
sycl::event add(sycl::queue& q, sycl::half* out, const sycl::half* a, const sycl::half* b, int64_t num_elements);

// In-place addition: a += b
sycl::event add_inplace(sycl::queue& q, float* a, const float* b, int64_t num_elements);
sycl::event add_inplace(sycl::queue& q, sycl::half* a, const sycl::half* b, int64_t num_elements);

// Elementwise multiplication: out = a * b
sycl::event mul(sycl::queue& q, float* out, const float* a, const float* b, int64_t num_elements);
sycl::event mul(sycl::queue& q, sycl::half* out, const sycl::half* a, const sycl::half* b, int64_t num_elements);

} // namespace xinfer::ops
