#include "linear.h"

namespace xinfer::ops {

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

                // Unpack low nibble (even k index)
                int8_t low = static_cast<int8_t>(byte_val & 0x0F);
                if (low >= 8) low = static_cast<int8_t>(low - 16);

                // Unpack high nibble (odd k index)
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
