#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

namespace xinfer::ops {

// XMX-accelerated GEMM: C = A * B
// A: [M, K] in FP16/half
// B: [K, N] in FP16/half
// C: [M, N] in FP32
// Accelerated using SYCL Joint Matrix with sub_group DPAS systolic hardware units.
void gemm_xmx(sycl::queue& q,
              float* C,
              const sycl::half* A,
              const sycl::half* B,
              int64_t M,
              int64_t N,
              int64_t K);

// FP32 interface wrapper: converts FP32 inputs to FP16 in device memory/scratchpad
// and executes XMX systolic GEMM
void gemm_xmx(sycl::queue& q,
              float* C,
              const float* A,
              const float* B,
              int64_t M,
              int64_t N,
              int64_t K);

} // namespace xinfer::ops
