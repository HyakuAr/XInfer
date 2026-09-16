// Citing vendor documentation per AGENTS.md §5:
// - docs/vendor/xmx-joint-matrix.md (lines 41-52, 77-85):
//   SYCL Joint Matrix API (sycl::ext::oneapi::experimental::matrix)
//   Tile primitives: joint_matrix<sub_group, ...>, joint_matrix_fill,
//   joint_matrix_load, joint_matrix_mad, joint_matrix_store.
//   Supported combinations on Arc Pro B60: M=16, N=16, K=16 (FP16/FP16 -> FP32).
// - docs/vendor/xe-gpu-architecture.md (lines 22-39):
//   Intel Arc Pro B60: 20 Xe-cores, 8 Vector Engines per core, 8 HW threads per VE
//   = 64 HW threads per core (1280 total). Sub-group size: 16, 32.
// - docs/vendor/thread-mapping-occupancy.md (lines 9-15):
//   Sub-group size 16 maps to one Vector Engine hardware thread; work-group to Xe-core.
//   sycl::reqd_sub_group_size(16).
// - docs/vendor/xetla-gemm.md (lines 23-51):
//   Subgroup-level reduction and coalesced loads along the reduction (K) dimension.

#include "linear.h"
#include <sycl/ext/oneapi/matrix/matrix.hpp>

namespace xinfer::ops {

using namespace sycl::ext::oneapi::experimental::matrix;

// =============================================================================
// Milestone 7 Accelerated INT4 Linear Kernel
// =============================================================================

void linear_int4(sycl::queue& q,
                 float* Y,
                 const float* X,
                 const uint8_t* W_int4,
                 const sycl::half* scales,
                 const float* bias,
                 int64_t M,
                 int64_t N,
                 int64_t K,
                 int group_size) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    int64_t num_groups = K / group_size;

    // Launch with sub-group size 16.
    // Each sub-group of 16 work-items cooperatively processes 1 output element (m, n).
    // Work-group size = 64 (4 sub-groups per work-group).
    constexpr size_t SG_SIZE = 16;
    constexpr size_t WG_SIZE = 64;

    size_t total_subgroups = static_cast<size_t>(M * N);
    size_t global_threads = total_subgroups * SG_SIZE;
    size_t padded_global = ((global_threads + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(padded_global, WG_SIZE),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                sycl::sub_group sg = item.get_sub_group();
                size_t global_sg_id = item.get_global_linear_id() / SG_SIZE;
                if (global_sg_id >= total_subgroups) return;

                int64_t m = global_sg_id / N;
                int64_t n = global_sg_id % N;
                size_t lane = sg.get_local_linear_id(); // 0..15

                const float* row_x = X + m * K;
                const uint8_t* row_w = W_int4 + n * (K / 2);
                const sycl::half* row_scales = scales + n * num_groups;

                float total_acc = 0.0f;

                for (int64_t g = 0; g < num_groups; ++g) {
                    float scale = static_cast<float>(row_scales[g]);
                    int64_t base_k = g * group_size;
                    int64_t base_byte = base_k / 2;

                    // Each group has group_size / 2 = 64 bytes.
                    // The 16 lanes cooperatively load 4 bytes each = 64 bytes in a single coalesced transaction.
                    const uint32_t* row_w_u32 = reinterpret_cast<const uint32_t*>(row_w + base_byte);
                    uint32_t packed = row_w_u32[lane]; // 8 INT4 nibbles

                    int64_t k_offset = base_k + lane * 8;
                    float lane_dot = 0.0f;

                    // Unpack 8 nibbles
                    #pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        uint8_t byte_val = static_cast<uint8_t>((packed >> (b * 8)) & 0xFF);

                        int8_t low = static_cast<int8_t>(byte_val & 0x0F);
                        if (low >= 8) low = static_cast<int8_t>(low - 16);

                        int8_t high = static_cast<int8_t>((byte_val >> 4) & 0x0F);
                        if (high >= 8) high = static_cast<int8_t>(high - 16);

                        int64_t k0 = k_offset + 2 * b;
                        int64_t k1 = k0 + 1;

                        lane_dot += row_x[k0] * static_cast<float>(low);
                        lane_dot += row_x[k1] * static_cast<float>(high);
                    }

                    // Parallel reduction across the 16 sub-group lanes
                    float group_dot = sycl::reduce_over_group(sg, lane_dot, sycl::plus<float>());
                    total_acc += group_dot * scale;
                }

                if (lane == 0) {
                    if (bias) {
                        total_acc += bias[n];
                    }
                    Y[m * N + n] = total_acc;
                }
            });
    });
}

// =============================================================================
// Reference-Only Naive Implementations (Preserved for numerical validation)
// =============================================================================

void gemm_naive(sycl::queue& q, float* C, const float* A, const float* B, int64_t M, int64_t N, int64_t K) {
    if (M <= 0 || N <= 0 || K <= 0) return;

    q.parallel_for(sycl::range<2>(static_cast<size_t>(M), static_cast<size_t>(N)), [=](sycl::id<2> idx) {
        int64_t m = idx[0];
        int64_t n = idx[1];

        float acc = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            acc += A[m * K + k] * B[k * N + n];
        }
        C[m * N + n] = acc;
    });
}

void linear_naive(sycl::queue& q,
                  float* Y,
                  const float* X,
                  const float* W,
                  const float* bias,
                  int64_t M,
                  int64_t N,
                  int64_t K) {
    if (M <= 0 || N <= 0 || K <= 0) return;

    q.parallel_for(sycl::range<2>(static_cast<size_t>(M), static_cast<size_t>(N)), [=](sycl::id<2> idx) {
        int64_t m = idx[0];
        int64_t n = idx[1];

        const float* row_x = X + m * K;
        const float* row_w = W + n * K;

        float acc = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            acc += row_x[k] * row_w[k];
        }
        if (bias) {
            acc += bias[n];
        }
        Y[m * N + n] = acc;
    });
}

void linear_int4_naive(sycl::queue& q,
                       float* Y,
                       const float* X,
                       const uint8_t* W_int4,
                       const sycl::half* scales,
                       const float* bias,
                       int64_t M,
                       int64_t N,
                       int64_t K,
                       int group_size) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    int64_t num_groups = K / group_size;

    q.parallel_for(sycl::range<2>(static_cast<size_t>(M), static_cast<size_t>(N)), [=](sycl::id<2> idx) {
        int64_t m = idx[0];
        int64_t n = idx[1];

        const float* row_x = X + m * K;
        const uint8_t* row_w = W_int4 + n * (K / 2);
        const sycl::half* row_scales = scales + n * num_groups;

        float acc = 0.0f;
        for (int64_t g = 0; g < num_groups; ++g) {
            float scale = static_cast<float>(row_scales[g]);
            int64_t base_k = g * group_size;
            int64_t base_byte = base_k / 2;

            for (int64_t b = 0; b < group_size / 2; ++b) {
                uint8_t byte_val = row_w[base_byte + b];

                int8_t low = static_cast<int8_t>(byte_val & 0x0F);
                if (low >= 8) low = static_cast<int8_t>(low - 16);

                int8_t high = static_cast<int8_t>((byte_val >> 4) & 0x0F);
                if (high >= 8) high = static_cast<int8_t>(high - 16);

                int64_t k0 = base_k + 2 * b;
                int64_t k1 = k0 + 1;

                acc += row_x[k0] * (static_cast<float>(low) * scale);
                acc += row_x[k1] * (static_cast<float>(high) * scale);
            }
        }

        if (bias) {
            acc += bias[n];
        }
        Y[m * N + n] = acc;
    });
}

} // namespace xinfer::ops
