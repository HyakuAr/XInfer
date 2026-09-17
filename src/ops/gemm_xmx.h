#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

namespace xinfer::ops {

// =============================================================================
// [RETIRED / REFERENCE-ONLY] Dense FP16 XMX Systolic GEMM
// =============================================================================
// Note: This kernel is retired from the production inference path.
// The Intel Arc Pro B60 XMX systolic units do NOT natively support INT4
// (verified in docs/vendor/b60-matrix-caps.md).
// For M=1 decode, empirical testing proves that unpacking INT4 to SLM to feed XMX
// is 7.5x slower than direct Vector Engine streaming (51 GB/s vs 383.7 GB/s).
// Production decode uses Vector Engine GEMV (linear_int4 in src/ops/linear.h).
// This file is retained as an oracle/reference implementation for dense FP16 GEMM.

// XMX-accelerated dense FP16 GEMM: C = A * B
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
