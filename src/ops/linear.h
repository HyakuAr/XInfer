#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

namespace xinfer::ops {

// =============================================================================
// XMX and Hardware-Accelerated Kernels (Milestone 7)
// =============================================================================

// XMX-accelerated GEMM using SYCL Joint Matrix: C = A * B
// A: [M, K], B: [K, N], C: [M, N]
void gemm_xmx(sycl::queue& q,
              float* C,
              const sycl::half* A,
              const sycl::half* B,
              int64_t M,
              int64_t N,
              int64_t K);

void gemm_xmx(sycl::queue& q,
              float* C,
              const float* A,
              const float* B,
              int64_t M,
              int64_t N,
              int64_t K);

// High-performance INT4 linear projection: Y = X * W^T + bias
// Fully coalesced sub-group cooperative GEMV and XMX systolic acceleration on Intel Arc Pro B60
// X: [M, K] (FP32 activations)
// W_int4: [N, K/2] (symmetric INT4 weights packed 2 nibbles per byte: low=even, high=odd)
// scales: [N, K/group_size] (FP16 per-group scales)
// bias: [N] (optional FP32 bias)
// Y: [M, N] (FP32 output)
void linear_int4(sycl::queue& q,
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
