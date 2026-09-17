// Citing vendor documentation per AGENTS.md §5:
// - docs/vendor/b60-matrix-caps.md (Sections 1-4):
//   Verified matrix combinations on Intel Arc Pro B60: FP16 M=16, N=16, K=16 supported;
//   no native INT4 support. For M=1 decode, SLM unpack + XMX is 7.5x slower than Vector Engine GEMV.
// - docs/vendor/xmx-joint-matrix.md (lines 41-52, 77-85):
//   SYCL Joint Matrix API (sycl::ext::oneapi::experimental::matrix)
//   Tile primitives: joint_matrix<sub_group, ...>, joint_matrix_fill,
//   joint_matrix_load, joint_matrix_mad, joint_matrix_store.
// - docs/vendor/xe-gpu-architecture.md (lines 22-39):
//   Intel Arc Pro B60 (Battlemage Xe2-HPG): 20 Xe-cores, sub-group sizes 16, 32.
// - docs/vendor/thread-mapping-occupancy.md (lines 9-15):
//   Sub-group size 16 maps to one Vector Engine hardware thread; work-group to Xe-core.

// NOTE: This kernel is RETIRED from the active engine and retained strictly as a
// reference-only implementation for dense FP16 GEMM oracle tests. The production
// Qwen3.8-27B INT4 decode path uses linear_int4 (Vector Engine SIMD16 GEMV).

#include "gemm_xmx.h"
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <vector>

namespace xinfer::ops {

using namespace sycl::ext::oneapi::experimental::matrix;

constexpr size_t TM = 16;
constexpr size_t TN = 16;
constexpr size_t TK = 16;

void gemm_xmx(sycl::queue& q,
              float* C,
              const sycl::half* A,
              const sycl::half* B,
              int64_t M,
              int64_t N,
              int64_t K) {
    if (M <= 0 || N <= 0 || K <= 0) return;

    // Pad dimensions to multiples of 16 for systolic hardware tiling
    int64_t M_tiles = (M + TM - 1) / TM;
    int64_t N_tiles = (N + TN - 1) / TN;
    int64_t K_tiles = (K + TK - 1) / TK;

    // Check if dimensions are exactly tile-aligned
    bool is_aligned = (M % TM == 0) && (N % TN == 0) && (K % TK == 0);

    if (is_aligned) {
        // Fast path: Each sub-group of 16 threads handles one TM x TN (16x16) output tile.
        sycl::range<2> global_range(static_cast<size_t>(M_tiles), static_cast<size_t>(N_tiles * 16));
        sycl::range<2> local_range(1, 16);

        q.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<2>(global_range, local_range),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                    sycl::sub_group sg = item.get_sub_group();
                    int64_t tile_m = item.get_global_id(0);
                    int64_t tile_n = item.get_global_id(1) / 16;

                    joint_matrix<sycl::sub_group, sycl::half, use::a, TM, TK, layout::row_major> sub_a;
                    joint_matrix<sycl::sub_group, sycl::half, use::b, TK, TN, layout::row_major> sub_b;
                    joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> sub_c;

                    joint_matrix_fill(sg, sub_c, 0.0f);

                    for (int64_t k = 0; k < K_tiles; ++k) {
                        const sycl::half* a_ptr_raw = A + (tile_m * TM) * K + (k * TK);
                        const sycl::half* b_ptr_raw = B + (k * TK) * N + (tile_n * TN);

                        auto a_ptr = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                              sycl::access::decorated::no>(a_ptr_raw);
                        auto b_ptr = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                              sycl::access::decorated::no>(b_ptr_raw);

                        joint_matrix_load(sg, sub_a, a_ptr, K);
                        joint_matrix_load(sg, sub_b, b_ptr, N);
                        joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
                    }

                    float* c_ptr_raw = C + (tile_m * TM) * N + (tile_n * TN);
                    auto c_ptr = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                          sycl::access::decorated::no>(c_ptr_raw);
                    joint_matrix_store(sg, sub_c, c_ptr, N, layout::row_major);
                });
        });
    } else {
        // Unaligned fallback: use temporary padded buffers or compute boundary with sub-groups
        // Allocate padded buffers on device
        size_t padded_M = M_tiles * TM;
        size_t padded_N = N_tiles * TN;
        size_t padded_K = K_tiles * TK;

        sycl::half* pad_A = sycl::malloc_device<sycl::half>(padded_M * padded_K, q);
        sycl::half* pad_B = sycl::malloc_device<sycl::half>(padded_K * padded_N, q);
        float* pad_C = sycl::malloc_device<float>(padded_M * padded_N, q);

        // Zero out padded buffers and copy data
        q.fill(pad_A, sycl::half{0.0f}, padded_M * padded_K);
        q.fill(pad_B, sycl::half{0.0f}, padded_K * padded_N);
        q.wait();

        q.parallel_for(sycl::range<2>(static_cast<size_t>(M), static_cast<size_t>(K)), [=](sycl::id<2> idx) {
            pad_A[idx[0] * padded_K + idx[1]] = A[idx[0] * K + idx[1]];
        });

        q.parallel_for(sycl::range<2>(static_cast<size_t>(K), static_cast<size_t>(N)), [=](sycl::id<2> idx) {
            pad_B[idx[0] * padded_N + idx[1]] = B[idx[0] * N + idx[1]];
        });
        q.wait();

        // Run systolic GEMM on padded dimensions
        gemm_xmx(q, pad_C, pad_A, pad_B, padded_M, padded_N, padded_K);
        q.wait();

        // Copy back unpadded result
        q.parallel_for(sycl::range<2>(static_cast<size_t>(M), static_cast<size_t>(N)), [=](sycl::id<2> idx) {
            C[idx[0] * N + idx[1]] = pad_C[idx[0] * padded_N + idx[1]];
        }).wait();

        sycl::free(pad_A, q);
        sycl::free(pad_B, q);
        sycl::free(pad_C, q);
    }
}

void gemm_xmx(sycl::queue& q,
              float* C,
              const float* A,
              const float* B,
              int64_t M,
              int64_t N,
              int64_t K) {
    if (M <= 0 || N <= 0 || K <= 0) return;

    // Convert FP32 inputs to FP16 for XMX systolic units
    sycl::half* A_half = sycl::malloc_device<sycl::half>(M * K, q);
    sycl::half* B_half = sycl::malloc_device<sycl::half>(K * N, q);

    q.parallel_for(sycl::range<1>(static_cast<size_t>(M * K)), [=](sycl::id<1> idx) {
        A_half[idx] = static_cast<sycl::half>(A[idx]);
    });
    q.parallel_for(sycl::range<1>(static_cast<size_t>(K * N)), [=](sycl::id<1> idx) {
        B_half[idx] = static_cast<sycl::half>(B[idx]);
    });
    q.wait();

    gemm_xmx(q, C, A_half, B_half, M, N, K);
    q.wait();

    sycl::free(A_half, q);
    sycl::free(B_half, q);
}

} // namespace xinfer::ops
