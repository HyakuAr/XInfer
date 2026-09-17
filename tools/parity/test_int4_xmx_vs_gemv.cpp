// Citing vendor documentation per AGENTS.md §5:
// - docs/vendor/b60-matrix-caps.md:
//   Verified B60 combinations: fp16/fp16 -> fp32 with msize=1, nsize=64, ksize=16.
// - docs/vendor/xmx-joint-matrix.md:
//   SYCL Joint Matrix API (sycl::ext::oneapi::experimental::matrix)
//   joint_matrix<sub_group, ...>, joint_matrix_fill, joint_matrix_load, joint_matrix_mad, joint_matrix_store.
// - docs/vendor/xe-gpu-architecture.md:
//   Battlemage Xe2 sub-group size 16.

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>

using namespace sycl::ext::oneapi::experimental::matrix;

// Reference Vector-Engine INT4 GEMV (as implemented in src/ops/linear.cpp)
sycl::event run_vector_engine_gemv(sycl::queue& q,
                                   float* Y,
                                   const float* X,
                                   const uint8_t* W_int4,
                                   const sycl::half* scales,
                                   int64_t M,
                                   int64_t N,
                                   int64_t K,
                                   int group_size = 128) {
    int64_t num_groups = K / group_size;
    constexpr size_t SG_SIZE = 16;
    constexpr size_t WG_SIZE = 64;
    constexpr int64_t ROWS_PER_SG = 2;

    int64_t sgs_per_m = (N + ROWS_PER_SG - 1) / ROWS_PER_SG;
    size_t total_subgroups = static_cast<size_t>(M * sgs_per_m);
    size_t global_threads = total_subgroups * SG_SIZE;
    size_t padded_global = ((global_threads + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;

    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(padded_global, WG_SIZE),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                sycl::sub_group sg = item.get_sub_group();
                size_t global_sg_id = item.get_global_linear_id() / SG_SIZE;
                if (global_sg_id >= total_subgroups) return;

                int64_t m = global_sg_id / sgs_per_m;
                int64_t sg_idx = global_sg_id % sgs_per_m;
                int64_t n0 = sg_idx * ROWS_PER_SG;
                int64_t n1 = n0 + 1;
                size_t lane = sg.get_local_linear_id();

                const float* row_x = X + m * K;
                const uint8_t* row_w0 = W_int4 + n0 * (K / 2);
                const uint8_t* row_w1 = (n1 < N) ? (W_int4 + n1 * (K / 2)) : nullptr;
                const sycl::half* scales0 = scales + n0 * num_groups;
                const sycl::half* scales1 = (n1 < N) ? (scales + n1 * num_groups) : nullptr;

                float lane_acc0 = 0.0f;
                float lane_acc1 = 0.0f;

                for (int64_t g = 0; g < num_groups; ++g) {
                    int64_t base_k = g * group_size;
                    int64_t base_byte = base_k / 2;
                    int64_t k_offset = base_k + lane * 8;

                    float x0 = row_x[k_offset + 0];
                    float x1 = row_x[k_offset + 1];
                    float x2 = row_x[k_offset + 2];
                    float x3 = row_x[k_offset + 3];
                    float x4 = row_x[k_offset + 4];
                    float x5 = row_x[k_offset + 5];
                    float x6 = row_x[k_offset + 6];
                    float x7 = row_x[k_offset + 7];

                    float scale0 = static_cast<float>(scales0[g]);
                    const uint32_t* w0_u32 = reinterpret_cast<const uint32_t*>(row_w0 + base_byte);
                    uint32_t p0 = w0_u32[lane];
                    int32_t sp0 = static_cast<int32_t>(p0);
                    float dot0 =
                        x0 * static_cast<float>((sp0 << 28) >> 28) +
                        x1 * static_cast<float>((sp0 << 24) >> 28) +
                        x2 * static_cast<float>((sp0 << 20) >> 28) +
                        x3 * static_cast<float>((sp0 << 16) >> 28) +
                        x4 * static_cast<float>((sp0 << 12) >> 28) +
                        x5 * static_cast<float>((sp0 << 8) >> 28) +
                        x6 * static_cast<float>((sp0 << 4) >> 28) +
                        x7 * static_cast<float>(sp0 >> 28);
                    lane_acc0 += dot0 * scale0;

                    if (row_w1) {
                        float scale1 = static_cast<float>(scales1[g]);
                        const uint32_t* w1_u32 = reinterpret_cast<const uint32_t*>(row_w1 + base_byte);
                        uint32_t p1 = w1_u32[lane];
                        int32_t sp1 = static_cast<int32_t>(p1);
                        float dot1 =
                            x0 * static_cast<float>((sp1 << 28) >> 28) +
                            x1 * static_cast<float>((sp1 << 24) >> 28) +
                            x2 * static_cast<float>((sp1 << 20) >> 28) +
                            x3 * static_cast<float>((sp1 << 16) >> 28) +
                            x4 * static_cast<float>((sp1 << 12) >> 28) +
                            x5 * static_cast<float>((sp1 << 8) >> 28) +
                            x6 * static_cast<float>((sp1 << 4) >> 28) +
                            x7 * static_cast<float>(sp1 >> 28);
                        lane_acc1 += dot1 * scale1;
                    }
                }

                float total0 = sycl::reduce_over_group(sg, lane_acc0, sycl::plus<float>());
                if (lane == 0) Y[m * N + n0] = total0;

                if (row_w1) {
                    float total1 = sycl::reduce_over_group(sg, lane_acc1, sycl::plus<float>());
                    if (lane == 0) Y[m * N + n1] = total1;
                }
            });
    });
}

// Experimental INT4 Unpack -> SLM -> XMX Joint Matrix GEMV (M=1, N=64, K=16)
sycl::event run_int4_unpack_xmx(sycl::queue& q,
                                float* Y,
                                const sycl::half* X_half,
                                const uint8_t* W_int4,
                                const sycl::half* scales,
                                int64_t M,
                                int64_t N,
                                int64_t K,
                                int group_size = 128) {
    constexpr size_t TM = 1;
    constexpr size_t TN = 64;
    constexpr size_t TK = 16;
    constexpr size_t SG_SIZE = 16;
    constexpr size_t WG_SIZE = 16; // 1 sub-group of 16 threads per work-group for single tile

    int64_t n_tiles = N / TN;
    int64_t k_tiles = K / TK;
    int64_t num_groups = K / group_size;

    return q.submit([&](sycl::handler& cgh) {
        // SLM buffer for unpacked FP16 weights tile [TK, TN] = 16 x 64 = 1024 halves
        sycl::local_accessor<sycl::half, 1> slm_b(sycl::range<1>(TK * TN), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(n_tiles * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                sycl::sub_group sg = item.get_sub_group();
                int64_t tile_n = item.get_group(0);
                size_t tid = item.get_local_id(0); // 0..15

                joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> sub_c;
                joint_matrix_fill(sg, sub_c, 0.0f);

                for (int64_t kt = 0; kt < k_tiles; ++kt) {
                    int64_t base_k = kt * TK;
                    int64_t g = base_k / group_size;

                    // Cooperative unpacking: 16 threads unpack 16 x 64 = 1024 weights
                    // Each thread unpacks 1024 / 16 = 64 weights (32 bytes of INT4)
                    // Thread tid handles rows [tid] across all 64 columns in slm_b
                    int64_t k_local = tid;
                    int64_t k_global = base_k + k_local;

                    for (int64_t col_chunk = 0; col_chunk < TN; col_chunk += 2) {
                        int64_t n0 = tile_n * TN + col_chunk;
                        int64_t n1 = n0 + 1;

                        uint8_t b0 = W_int4[n0 * (K / 2) + (k_global / 2)];
                        int8_t nib0 = (k_global % 2 == 0) ? (b0 & 0x0F) : ((b0 >> 4) & 0x0F);
                        if (nib0 >= 8) nib0 -= 16;
                        float s0 = static_cast<float>(scales[n0 * num_groups + g]);

                        uint8_t b1 = W_int4[n1 * (K / 2) + (k_global / 2)];
                        int8_t nib1 = (k_global % 2 == 0) ? (b1 & 0x0F) : ((b1 >> 4) & 0x0F);
                        if (nib1 >= 8) nib1 -= 16;
                        float s1 = static_cast<float>(scales[n1 * num_groups + g]);

                        // Store in row-major layout in SLM [TK x TN]: row k_local, col col_chunk
                        slm_b[k_local * TN + col_chunk + 0] = static_cast<sycl::half>(nib0 * s0);
                        slm_b[k_local * TN + col_chunk + 1] = static_cast<sycl::half>(nib1 * s1);
                    }

                    // Synchronize SLM across work-group before Joint Matrix load
                    item.barrier(sycl::access::fence_space::local_space);

                    joint_matrix<sycl::sub_group, sycl::half, use::a, TM, TK, layout::row_major> sub_a;
                    joint_matrix<sycl::sub_group, sycl::half, use::b, TK, TN, layout::row_major> sub_b;

                    const sycl::half* a_ptr_raw = X_half + base_k;
                    auto a_ptr = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                          sycl::access::decorated::no>(a_ptr_raw);
                    joint_matrix_load(sg, sub_a, a_ptr, TK);

                    auto b_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                          sycl::access::decorated::no>(slm_b.get_multi_ptr<sycl::access::decorated::no>().get());
                    joint_matrix_load(sg, sub_b, b_ptr, TN);

                    joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);

                    item.barrier(sycl::access::fence_space::local_space);
                }

                // Store sub_c tile (1 x 64) into Y
                float* y_ptr_raw = Y + tile_n * TN;
                auto y_ptr = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                      sycl::access::decorated::no>(y_ptr_raw);
                joint_matrix_store(sg, sub_c, y_ptr, TN, layout::row_major);
            });
    });
}

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    auto dev = q.get_device();
    std::cout << "==================================================================" << std::endl;
    std::cout << "  Microbenchmark: Vector Engine GEMV vs INT4-Unpack+XMX Joint Matrix" << std::endl;
    std::cout << "  Device:       " << dev.get_info<sycl::info::device::name>() << std::endl;
#ifdef XINFER_BUILD_CONFIG
    std::cout << "  Build Config: " << XINFER_BUILD_CONFIG << std::endl;
#endif
    std::cout << "==================================================================" << std::endl;

    const int64_t M = 1;
    const int64_t K = 5120;
    const int64_t N = 17408; // Realistic Qwen3.8 MLP intermediate dimension
    const int group_size = 128;
    const int64_t num_groups = K / group_size;

    size_t x_bytes = K * sizeof(float);
    size_t x_half_bytes = K * sizeof(sycl::half);
    size_t w_bytes = (N * K / 2);
    size_t scale_bytes = (N * num_groups) * sizeof(sycl::half);
    size_t y_bytes = N * sizeof(float);

    std::cout << "Problem Shape: M = " << M << ", K = " << K << ", N = " << N << std::endl;
    std::cout << "Weight Data Size: " << (w_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;

    float* d_X = sycl::malloc_device<float>(K, q);
    sycl::half* d_X_half = sycl::malloc_device<sycl::half>(K, q);
    uint8_t* d_W = sycl::malloc_device<uint8_t>(w_bytes, q);
    sycl::half* d_scales = sycl::malloc_device<sycl::half>(N * num_groups, q);
    float* d_Y_ve = sycl::malloc_device<float>(N, q);
    float* d_Y_xmx = sycl::malloc_device<float>(N, q);

    q.fill(d_X, 1.0f, K);
    q.fill(d_X_half, sycl::half{1.0f}, K);
    q.fill(d_W, static_cast<uint8_t>(0x22), w_bytes);
    q.fill(d_scales, sycl::half{0.1f}, N * num_groups);
    q.wait();

    // 1. Warm up both kernels
    run_vector_engine_gemv(q, d_Y_ve, d_X, d_W, d_scales, M, N, K, group_size).wait();
    run_int4_unpack_xmx(q, d_Y_xmx, d_X_half, d_W, d_scales, M, N, K, group_size).wait();

    // Verify numerical equivalence
    std::vector<float> h_Y_ve(N), h_Y_xmx(N);
    q.memcpy(h_Y_ve.data(), d_Y_ve, y_bytes).wait();
    q.memcpy(h_Y_xmx.data(), d_Y_xmx, y_bytes).wait();

    float max_diff = 0.0f;
    for (int64_t i = 0; i < N; ++i) {
        float diff = std::abs(h_Y_ve[i] - h_Y_xmx[i]);
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "Numerical Parity Check (VE vs XMX): max abs diff = " << max_diff << std::endl;
    std::cout << "VE Y[0] = " << h_Y_ve[0] << ", XMX Y[0] = " << h_Y_xmx[0] << std::endl;

    // 2. Measure Vector-Engine GEMV
    constexpr int ITERS = 20;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        run_vector_engine_gemv(q, d_Y_ve, d_X, d_W, d_scales, M, N, K, group_size);
    }
    q.wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    double time_ve_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;
    double bw_ve = (w_bytes + scale_bytes) / (time_ve_ms * 1e6); // GB/s

    // 3. Measure INT4-Unpack + XMX Joint Matrix
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        run_int4_unpack_xmx(q, d_Y_xmx, d_X_half, d_W, d_scales, M, N, K, group_size);
    }
    q.wait();
    auto t3 = std::chrono::high_resolution_clock::now();
    double time_xmx_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / ITERS;
    double bw_xmx = (w_bytes + scale_bytes) / (time_xmx_ms * 1e6); // GB/s

    std::cout << "\n============================== RESULTS ==============================" << std::endl;
    std::cout << "  1. Vector Engine SIMD16 GEMV (linear_int4):" << std::endl;
    std::cout << "     Latency:   " << std::fixed << std::setprecision(3) << time_ve_ms << " ms" << std::endl;
    std::cout << "     Bandwidth: " << std::fixed << std::setprecision(1) << bw_ve << " GB/s" << std::endl;
    std::cout << "  2. INT4-Unpack-to-SLM + XMX Joint Matrix (1x64x16):" << std::endl;
    std::cout << "     Latency:   " << std::fixed << std::setprecision(3) << time_xmx_ms << " ms" << std::endl;
    std::cout << "     Bandwidth: " << std::fixed << std::setprecision(1) << bw_xmx << " GB/s" << std::endl;
    std::cout << "  Ratio (XMX / VE Latency): " << (time_xmx_ms / time_ve_ms) << "x" << std::endl;
    std::cout << "=====================================================================" << std::endl;

    sycl::free(d_X, q);
    sycl::free(d_X_half, q);
    sycl::free(d_W, q);
    sycl::free(d_scales, q);
    sycl::free(d_Y_ve, q);
    sycl::free(d_Y_xmx, q);

    return 0;
}
