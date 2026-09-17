#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

namespace xinfer::ops {

// =============================================================================
// Vector Engine INT4 GEMV Kernels (Milestone 7 / M10)
// =============================================================================

// High-performance INT4 Vector Engine GEMV: Y = X * W^T + bias
// Sub-group cooperative SIMD16 reduction with register activation reuse and ALU bit-shift dequantization.
// Note: Intel Arc Pro B60 XMX has no native INT4 support (per docs/vendor/b60-matrix-caps.md).
// For M=1 decode, arithmetic intensity is ~4 FLOP/byte (100% memory bandwidth-bound).
// Vector Engine streaming achieves 383.7 GB/s (84% peak), whereas SLM-unpacking to XMX is 7.5x slower.
// X: [M, K] (FP32 activations)
// W_int4: [N, K/2] (symmetric INT4 weights packed 2 nibbles per byte: low=even, high=odd)
// scales: [N, K/group_size] (FP16 per-group scales)
// bias: [N] (optional FP32 bias)
// Y: [M, N] (FP32 output)
sycl::event linear_int4(sycl::queue& q,
                  float* Y,
                  const float* X,
                  const uint8_t* W_int4,
                  const sycl::half* scales,
                  const float* bias,
                  int64_t M,
                  int64_t N,
                  int64_t K,
                  int group_size = 128);

// =============================================================================
// Reference-Only Naive Kernels (Preserved for numerical oracle validation)
// =============================================================================

// [Reference-Only] Standard naive FP32 GEMM: C = A * B
void gemm_naive(sycl::queue& q, float* C, const float* A, const float* B, int64_t M, int64_t N, int64_t K);

// [Reference-Only] Naive FP32 Linear projection: Y = X * W^T + bias
void linear_naive(sycl::queue& q,
                  float* Y,
                  const float* X,
                  const float* W,
                  const float* bias,
                  int64_t M,
                  int64_t N,
                  int64_t K);

// [Reference-Only] Naive INT4 quantized linear projection
void linear_int4_naive(sycl::queue& q,
                       float* Y,
                       const float* X,
                       const uint8_t* W_int4,
                       const sycl::half* scales,
                       const float* bias,
                       int64_t M,
                       int64_t N,
                       int64_t K,
                       int group_size = 128);

} // namespace xinfer::ops
