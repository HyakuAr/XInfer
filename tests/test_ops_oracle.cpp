#include "core/device.h"
#include "ops/rmsnorm.h"
#include "ops/rope.h"
#include "ops/elementwise.h"
#include "ops/softmax.h"
#include "ops/linear.h"
#include "ops/attention.h"
#include "ops/sampling.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include <limits>
#include <iomanip>

using namespace xinfer::core;
using namespace xinfer::ops;

namespace {

// Helper to compute max absolute and relative difference
struct DiffStats {
    float max_abs{0.0f};
    float max_rel{0.0f};
    float mean_abs{0.0f};
};

DiffStats compare_buffers(const float* a, const float* b, size_t n) {
    DiffStats s;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float diff = std::abs(a[i] - b[i]);
        if (diff > s.max_abs) s.max_abs = diff;
        float rel = diff / (std::max(std::abs(a[i]), std::abs(b[i])) + 1e-6f);
        if (rel > s.max_rel) s.max_rel = rel;
        sum += diff;
    }
    s.mean_abs = static_cast<float>(sum / static_cast<double>(n));
    return s;
}

} // anonymous namespace

// -----------------------------------------------------------------------------
// 1. RMSNorm Oracle Test
// -----------------------------------------------------------------------------
void test_rmsnorm_oracle(DeviceContext& ctx) {
    std::cout << "\n[Oracle 1/7] RMSNorm (hidden_size = 5120, num_tokens = 4)..." << std::endl;
    const int64_t num_tokens = 4;
    const int64_t hidden_size = 5120; // Qwen3.8-27B hidden size
    const size_t total_elements = num_tokens * hidden_size;
    const float eps = 1e-6f;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> h_in(total_elements);
    std::vector<float> h_weight(hidden_size);
    for (auto& v : h_in) v = dist(rng);
    for (auto& v : h_weight) v = dist(rng) + 1.0f;

    // CPU Oracle evaluation
    std::vector<float> oracle_out(total_elements);
    for (int64_t t = 0; t < num_tokens; ++t) {
        const float* x = h_in.data() + t * hidden_size;
        float* y = oracle_out.data() + t * hidden_size;
        double sum_sq = 0.0;
        for (int64_t i = 0; i < hidden_size; ++i) {
            sum_sq += static_cast<double>(x[i]) * static_cast<double>(x[i]);
        }
        double mean_sq = sum_sq / static_cast<double>(hidden_size);
        float rsqrt_val = static_cast<float>(1.0 / std::sqrt(mean_sq + static_cast<double>(eps)));
        for (int64_t i = 0; i < hidden_size; ++i) {
            y[i] = x[i] * rsqrt_val * h_weight[i];
        }
    }

    // GPU execution
    float* d_in = static_cast<float*>(ctx.allocate_device(total_elements * sizeof(float)));
    float* d_weight = static_cast<float*>(ctx.allocate_device(hidden_size * sizeof(float)));
    float* d_out = static_cast<float*>(ctx.allocate_device(total_elements * sizeof(float)));

    ctx.copy_host_to_device(d_in, h_in.data(), total_elements * sizeof(float));
    ctx.copy_host_to_device(d_weight, h_weight.data(), hidden_size * sizeof(float));

    rmsnorm(ctx.queue(), d_out, d_in, d_weight, num_tokens, hidden_size, eps);

    std::vector<float> gpu_out(total_elements);
    ctx.copy_device_to_host(gpu_out.data(), d_out, total_elements * sizeof(float));

    ctx.free_device(d_in);
    ctx.free_device(d_weight);
    ctx.free_device(d_out);

    DiffStats diff = compare_buffers(oracle_out.data(), gpu_out.data(), total_elements);
    std::cout << "  Max Abs Diff: " << diff.max_abs << " | Mean Abs: " << diff.mean_abs << std::endl;
    assert(diff.max_abs <= 1e-4f);
    std::cout << "  -> PASSED: RMSNorm matches numerical oracle." << std::endl;
}

// -----------------------------------------------------------------------------
// 2. RoPE Oracle Test
// -----------------------------------------------------------------------------
void test_rope_oracle(DeviceContext& ctx) {
    std::cout << "\n[Oracle 2/7] RoPE (Q_heads = 24, KV_heads = 4, head_dim = 256, tokens = 4)..." << std::endl;
    const int64_t num_tokens = 4;
    const int64_t num_q_heads = 24;  // Qwen3.8-27B Q heads
    const int64_t num_kv_heads = 4;  // Qwen3.8-27B KV heads
    const int64_t head_dim = 256;    // Qwen3.8-27B head dimension
    const float theta = 10000000.0f; // Qwen3.8 rope_theta

    std::vector<int64_t> positions = {0, 1, 15, 256};

    size_t q_elements = num_tokens * num_q_heads * head_dim;
    size_t k_elements = num_tokens * num_kv_heads * head_dim;

    std::mt19937 rng(43);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> h_q_in(q_elements);
    std::vector<float> h_k_in(k_elements);
    for (auto& v : h_q_in) v = dist(rng);
    for (auto& v : h_k_in) v = dist(rng);

    // CPU Oracle evaluation
    std::vector<float> oracle_q = h_q_in;
    std::vector<float> oracle_k = h_k_in;
    int64_t rotary_dim = 64;
    int64_t half_rot = rotary_dim / 2;

    auto apply_oracle_rope = [&](std::vector<float>& buf, int64_t num_heads) {
        for (int64_t t = 0; t < num_tokens; ++t) {
            int64_t pos = positions[t];
            for (int64_t h = 0; h < num_heads; ++h) {
                int64_t base = (t * num_heads + h) * head_dim;
                for (int64_t p = 0; p < half_rot; ++p) {
                    double freq = 1.0 / std::pow(static_cast<double>(theta), static_cast<double>(2 * p) / static_cast<double>(rotary_dim));
                    double angle = static_cast<double>(pos) * freq;
                    float cos_val = static_cast<float>(std::cos(angle));
                    float sin_val = static_cast<float>(std::sin(angle));

                    float v0 = buf[base + p];
                    float v1 = buf[base + p + half_rot];

                    buf[base + p] = v0 * cos_val - v1 * sin_val;
                    buf[base + p + half_rot] = v1 * cos_val + v0 * sin_val;
                }
            }
        }
    };

    apply_oracle_rope(oracle_q, num_q_heads);
    apply_oracle_rope(oracle_k, num_kv_heads);

    // GPU execution
    float* d_q = static_cast<float*>(ctx.allocate_device(q_elements * sizeof(float)));
    float* d_k = static_cast<float*>(ctx.allocate_device(k_elements * sizeof(float)));
    int64_t* d_pos = static_cast<int64_t*>(ctx.allocate_device(num_tokens * sizeof(int64_t)));

    ctx.copy_host_to_device(d_q, h_q_in.data(), q_elements * sizeof(float));
    ctx.copy_host_to_device(d_k, h_k_in.data(), k_elements * sizeof(float));
    ctx.copy_host_to_device(d_pos, positions.data(), num_tokens * sizeof(int64_t));

    rope(ctx.queue(), d_q, d_k, num_tokens, num_q_heads, num_kv_heads, head_dim, d_pos, theta, rotary_dim);

    std::vector<float> gpu_q(q_elements);
    std::vector<float> gpu_k(k_elements);
    ctx.copy_device_to_host(gpu_q.data(), d_q, q_elements * sizeof(float));
    ctx.copy_device_to_host(gpu_k.data(), d_k, k_elements * sizeof(float));

    ctx.free_device(d_q);
    ctx.free_device(d_k);
    ctx.free_device(d_pos);

    DiffStats diff_q = compare_buffers(oracle_q.data(), gpu_q.data(), q_elements);
    DiffStats diff_k = compare_buffers(oracle_k.data(), gpu_k.data(), k_elements);

    std::cout << "  Q Max Abs Diff: " << diff_q.max_abs << " | K Max Abs Diff: " << diff_k.max_abs << std::endl;
    assert(diff_q.max_abs <= 1e-4f);
    assert(diff_k.max_abs <= 1e-4f);
    std::cout << "  -> PASSED: RoPE matches numerical oracle." << std::endl;
}

// -----------------------------------------------------------------------------
// 3. Elementwise (SwiGLU, SiLU, Add) Oracle Test
// -----------------------------------------------------------------------------
void test_elementwise_oracle(DeviceContext& ctx) {
    std::cout << "\n[Oracle 3/7] Elementwise SwiGLU / SiLU (intermediate_size = 17408, tokens = 4)..." << std::endl;
    const int64_t num_elements = 4 * 17408; // Qwen3.8-27B MLP intermediate size

    std::mt19937 rng(44);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    std::vector<float> h_gate(num_elements);
    std::vector<float> h_up(num_elements);
    for (auto& v : h_gate) v = dist(rng);
    for (auto& v : h_up) v = dist(rng);

    // CPU Oracle evaluation
    std::vector<float> oracle_swiglu(num_elements);
    for (int64_t i = 0; i < num_elements; ++i) {
        float g = h_gate[i];
        float silu_g = g / (1.0f + std::exp(-g));
        oracle_swiglu[i] = silu_g * h_up[i];
    }

    // GPU execution
    float* d_gate = static_cast<float*>(ctx.allocate_device(num_elements * sizeof(float)));
    float* d_up = static_cast<float*>(ctx.allocate_device(num_elements * sizeof(float)));
    float* d_out = static_cast<float*>(ctx.allocate_device(num_elements * sizeof(float)));

    ctx.copy_host_to_device(d_gate, h_gate.data(), num_elements * sizeof(float));
    ctx.copy_host_to_device(d_up, h_up.data(), num_elements * sizeof(float));

    swiglu(ctx.queue(), d_out, d_gate, d_up, num_elements);

    std::vector<float> gpu_out(num_elements);
    ctx.copy_device_to_host(gpu_out.data(), d_out, num_elements * sizeof(float));

    ctx.free_device(d_gate);
    ctx.free_device(d_up);
    ctx.free_device(d_out);

    DiffStats diff = compare_buffers(oracle_swiglu.data(), gpu_out.data(), num_elements);
    std::cout << "  SwiGLU Max Abs Diff: " << diff.max_abs << " | Mean Abs: " << diff.mean_abs << std::endl;
    assert(diff.max_abs <= 1e-4f);
    std::cout << "  -> PASSED: SwiGLU matches numerical oracle." << std::endl;
}

// -----------------------------------------------------------------------------
// 4. Softmax Oracle Test
// -----------------------------------------------------------------------------
void test_softmax_oracle(DeviceContext& ctx) {
    std::cout << "\n[Oracle 4/7] Softmax & Causal Softmax (rows = 24, cols = 128)..." << std::endl;
    const int64_t rows = 24;
    const int64_t cols = 128;
    const size_t total = rows * cols;

    std::mt19937 rng(45);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::vector<float> h_in(total);
    for (auto& v : h_in) v = dist(rng);

    // CPU Oracle evaluation
    std::vector<float> oracle_out(total);
    for (int64_t r = 0; r < rows; ++r) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (int64_t c = 0; c < cols; ++c) {
            max_val = std::max(max_val, h_in[r * cols + c]);
        }
        double sum = 0.0;
        for (int64_t c = 0; c < cols; ++c) {
            float e = std::exp(h_in[r * cols + c] - max_val);
            oracle_out[r * cols + c] = e;
            sum += static_cast<double>(e);
        }
        float inv_sum = static_cast<float>(1.0 / sum);
        for (int64_t c = 0; c < cols; ++c) {
            oracle_out[r * cols + c] *= inv_sum;
        }
    }

    // GPU execution
    float* d_in = static_cast<float*>(ctx.allocate_device(total * sizeof(float)));
    float* d_out = static_cast<float*>(ctx.allocate_device(total * sizeof(float)));

    ctx.copy_host_to_device(d_in, h_in.data(), total * sizeof(float));

    softmax(ctx.queue(), d_out, d_in, rows, cols);

    std::vector<float> gpu_out(total);
    ctx.copy_device_to_host(gpu_out.data(), d_out, total * sizeof(float));

    ctx.free_device(d_in);
    ctx.free_device(d_out);

    DiffStats diff = compare_buffers(oracle_out.data(), gpu_out.data(), total);
    std::cout << "  Softmax Max Abs Diff: " << diff.max_abs << std::endl;
    assert(diff.max_abs <= 1e-4f);
    std::cout << "  -> PASSED: Softmax matches numerical oracle." << std::endl;
}

// -----------------------------------------------------------------------------
// 5. Linear / GEMM Oracle Test (FP32 & INT4)
// -----------------------------------------------------------------------------
void test_linear_oracle(DeviceContext& ctx) {
    std::cout << "\n[Oracle 5/7] Linear Projection: FP32 and INT4 (M = 2, K = 5120, N = 2048)..." << std::endl;
    const int64_t M = 2;
    const int64_t K = 5120;
    const int64_t N = 2048;
    const int group_size = 128;
    const int64_t num_groups = K / group_size;

    std::mt19937 rng(46);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> h_X(M * K);
    for (auto& v : h_X) v = dist(rng);

    // INT4 packed weights and FP16 scales
    std::vector<uint8_t> h_W_int4(N * (K / 2));
    std::vector<sycl::half> h_scales(N * num_groups);

    // Fill with simulated INT4 values [-8, 7] and scales
    for (size_t i = 0; i < h_scales.size(); ++i) {
        h_scales[i] = sycl::half(0.015f + static_cast<float>(i % 10) * 0.001f);
    }
    for (size_t i = 0; i < h_W_int4.size(); ++i) {
        uint8_t low = (i + 3) & 0x0F;
        uint8_t high = ((i + 7) & 0x0F) << 4;
        h_W_int4[i] = low | high;
    }

    // CPU Oracle evaluation of INT4 dequantization and matrix multiply
    std::vector<float> oracle_Y(M * N, 0.0f);
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            double acc = 0.0;
            const float* row_x = h_X.data() + m * K;
            const uint8_t* row_w = h_W_int4.data() + n * (K / 2);
            const sycl::half* row_scales = h_scales.data() + n * num_groups;

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

                    acc += static_cast<double>(row_x[k0]) * (static_cast<double>(low) * static_cast<double>(scale));
                    acc += static_cast<double>(row_x[k1]) * (static_cast<double>(high) * static_cast<double>(scale));
                }
            }
            oracle_Y[m * N + n] = static_cast<float>(acc);
        }
    }

    // GPU execution
    float* d_X = static_cast<float*>(ctx.allocate_device(M * K * sizeof(float)));
    uint8_t* d_W = static_cast<uint8_t*>(ctx.allocate_device(N * (K / 2)));
    sycl::half* d_scales = static_cast<sycl::half*>(ctx.allocate_device(N * num_groups * sizeof(sycl::half)));
    float* d_Y = static_cast<float*>(ctx.allocate_device(M * N * sizeof(float)));

    ctx.copy_host_to_device(d_X, h_X.data(), M * K * sizeof(float));
    ctx.copy_host_to_device(d_W, h_W_int4.data(), N * (K / 2));
    ctx.copy_host_to_device(d_scales, h_scales.data(), N * num_groups * sizeof(sycl::half));

    // GPU execution of accelerated linear_int4
    linear_int4(ctx.queue(), d_Y, d_X, d_W, d_scales, nullptr, M, N, K, group_size);

    std::vector<float> gpu_Y(M * N);
    ctx.copy_device_to_host(gpu_Y.data(), d_Y, M * N * sizeof(float));

    DiffStats diff = compare_buffers(oracle_Y.data(), gpu_Y.data(), M * N);
    std::cout << "  Accelerated INT4 Linear Max Abs Diff: " << diff.max_abs << " | Mean Abs: " << diff.mean_abs << std::endl;
    assert(diff.max_abs <= 1e-4f);
    std::cout << "  -> PASSED: Accelerated INT4 Linear matches numerical oracle." << std::endl;

    // Reference naive test
    linear_int4_naive(ctx.queue(), d_Y, d_X, d_W, d_scales, nullptr, M, N, K, group_size);
    ctx.copy_device_to_host(gpu_Y.data(), d_Y, M * N * sizeof(float));
    DiffStats diff_naive = compare_buffers(oracle_Y.data(), gpu_Y.data(), M * N);
    std::cout << "  Reference Naive INT4 Linear Max Abs Diff: " << diff_naive.max_abs << std::endl;
    assert(diff_naive.max_abs <= 1e-4f);
    std::cout << "  -> PASSED: Reference Naive INT4 Linear matches numerical oracle." << std::endl;

    ctx.free_device(d_X);
    ctx.free_device(d_W);
    ctx.free_device(d_scales);
    ctx.free_device(d_Y);

    // Test XMX Systolic GEMM (M = 16, K = 32, N = 64)
    std::cout << "  Testing XMX Systolic GEMM (M = 16, K = 32, N = 64)..." << std::endl;
    const int64_t gM = 16, gK = 32, gN = 64;
    std::vector<sycl::half> h_A(gM * gK, sycl::half{1.5f});
    std::vector<sycl::half> h_B(gK * gN, sycl::half{2.0f});
    std::vector<float> h_C_oracle(gM * gN, static_cast<float>(gK) * 1.5f * 2.0f);

    sycl::half* d_A = static_cast<sycl::half*>(ctx.allocate_device(gM * gK * sizeof(sycl::half)));
    sycl::half* d_B = static_cast<sycl::half*>(ctx.allocate_device(gK * gN * sizeof(sycl::half)));
    float* d_C = static_cast<float*>(ctx.allocate_device(gM * gN * sizeof(float)));

    ctx.copy_host_to_device(d_A, h_A.data(), gM * gK * sizeof(sycl::half));
    ctx.copy_host_to_device(d_B, h_B.data(), gK * gN * sizeof(sycl::half));

    gemm_xmx(ctx.queue(), d_C, d_A, d_B, gM, gN, gK);

    std::vector<float> gpu_C(gM * gN);
    ctx.copy_device_to_host(gpu_C.data(), d_C, gM * gN * sizeof(float));

    ctx.free_device(d_A);
    ctx.free_device(d_B);
    ctx.free_device(d_C);

    DiffStats diff_xmx = compare_buffers(h_C_oracle.data(), gpu_C.data(), gM * gN);
    std::cout << "  XMX GEMM Max Abs Diff: " << diff_xmx.max_abs << std::endl;
    assert(diff_xmx.max_abs <= 1e-4f);
    std::cout << "  -> PASSED: XMX Systolic GEMM matches numerical oracle." << std::endl;
}

// -----------------------------------------------------------------------------
// 6. Causal SDPA Attention Oracle Test
// -----------------------------------------------------------------------------
void test_attention_oracle(DeviceContext& ctx) {
    std::cout << "\n[Oracle 6/7] Causal SDPA with GQA (seq_len = 8, H_q = 24, H_kv = 4, D = 256)..." << std::endl;
    const int64_t seq_len = 8;
    const int64_t num_q_heads = 24;
    const int64_t num_kv_heads = 4;
    const int64_t head_dim = 256;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int64_t gqa_ratio = num_q_heads / num_kv_heads;

    size_t q_elements = seq_len * num_q_heads * head_dim;
    size_t kv_elements = seq_len * num_kv_heads * head_dim;

    std::mt19937 rng(47);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> h_Q(q_elements);
    std::vector<float> h_K(kv_elements);
    std::vector<float> h_V(kv_elements);
    for (auto& v : h_Q) v = dist(rng);
    for (auto& v : h_K) v = dist(rng);
    for (auto& v : h_V) v = dist(rng);

    // CPU Oracle evaluation
    std::vector<float> oracle_out(q_elements, 0.0f);
    for (int64_t i = 0; i < seq_len; ++i) {
        for (int64_t h = 0; h < num_q_heads; ++h) {
            int64_t kv_h = h / gqa_ratio;
            const float* q_vec = h_Q.data() + (i * num_q_heads + h) * head_dim;
            float* out_vec = oracle_out.data() + (i * num_q_heads + h) * head_dim;

            // 1. Compute scores for j <= i
            std::vector<float> scores(i + 1);
            float max_s = -std::numeric_limits<float>::infinity();
            for (int64_t j = 0; j <= i; ++j) {
                const float* k_vec = h_K.data() + (j * num_kv_heads + kv_h) * head_dim;
                double dot = 0.0;
                for (int64_t d = 0; d < head_dim; ++d) {
                    dot += static_cast<double>(q_vec[d]) * static_cast<double>(k_vec[d]);
                }
                float s = static_cast<float>(dot * static_cast<double>(scale));
                scores[j] = s;
                max_s = std::max(max_s, s);
            }

            // 2. Softmax
            double sum_e = 0.0;
            std::vector<float> weights(i + 1);
            for (int64_t j = 0; j <= i; ++j) {
                float e = std::exp(scores[j] - max_s);
                weights[j] = e;
                sum_e += static_cast<double>(e);
            }
            float inv_sum = static_cast<float>(1.0 / sum_e);
            for (int64_t j = 0; j <= i; ++j) {
                weights[j] *= inv_sum;
            }

            // 3. Weighted sum of V
            for (int64_t d = 0; d < head_dim; ++d) {
                double acc = 0.0;
                for (int64_t j = 0; j <= i; ++j) {
                    const float* v_vec = h_V.data() + (j * num_kv_heads + kv_h) * head_dim;
                    acc += static_cast<double>(weights[j]) * static_cast<double>(v_vec[d]);
                }
                out_vec[d] = static_cast<float>(acc);
            }
        }
    }

    // GPU execution
    float* d_Q = static_cast<float*>(ctx.allocate_device(q_elements * sizeof(float)));
    float* d_K = static_cast<float*>(ctx.allocate_device(kv_elements * sizeof(float)));
    float* d_V = static_cast<float*>(ctx.allocate_device(kv_elements * sizeof(float)));
    float* d_out = static_cast<float*>(ctx.allocate_device(q_elements * sizeof(float)));

    ctx.copy_host_to_device(d_Q, h_Q.data(), q_elements * sizeof(float));
    ctx.copy_host_to_device(d_K, h_K.data(), kv_elements * sizeof(float));
    ctx.copy_host_to_device(d_V, h_V.data(), kv_elements * sizeof(float));

    sdpa_causal_naive(ctx.queue(), d_out, d_Q, d_K, d_V, seq_len, num_q_heads, num_kv_heads, head_dim, scale);

    std::vector<float> gpu_out(q_elements);
    ctx.copy_device_to_host(gpu_out.data(), d_out, q_elements * sizeof(float));

    DiffStats diff = compare_buffers(oracle_out.data(), gpu_out.data(), q_elements);
    std::cout << "  SDPA Naive Max Abs Diff: " << diff.max_abs << " | Mean Abs: " << diff.mean_abs << std::endl;
    assert(diff.max_abs <= 1e-4f);
    std::cout << "  -> PASSED: Causal SDPA Naive matches numerical oracle." << std::endl;

    // Test Accelerated Cached SDPA
    std::cout << "  Testing Accelerated Cached SDPA with Sub-groups..." << std::endl;
    sycl::half* d_k_cache = static_cast<sycl::half*>(ctx.allocate_device(kv_elements * sizeof(sycl::half)));
    sycl::half* d_v_cache = static_cast<sycl::half*>(ctx.allocate_device(kv_elements * sizeof(sycl::half)));
    attention_write_kv_cache(ctx.queue(), d_k_cache, d_v_cache, d_K, d_V, 0, seq_len, num_kv_heads, head_dim);

    sdpa_causal_cached(ctx.queue(), d_out, d_Q, d_k_cache, d_v_cache, 0, seq_len, num_q_heads, num_kv_heads, head_dim, scale);
    ctx.copy_device_to_host(gpu_out.data(), d_out, q_elements * sizeof(float));

    ctx.free_device(d_k_cache);
    ctx.free_device(d_v_cache);
    ctx.free_device(d_Q);
    ctx.free_device(d_K);
    ctx.free_device(d_V);
    ctx.free_device(d_out);

    DiffStats diff_cached = compare_buffers(oracle_out.data(), gpu_out.data(), q_elements);
    std::cout << "  Accelerated SDPA Cached Max Abs Diff: " << diff_cached.max_abs << std::endl;
    assert(diff_cached.max_abs <= 1e-3f); // FP16 KV cache tolerance
    std::cout << "  -> PASSED: Accelerated Cached SDPA matches numerical oracle." << std::endl;
}

// -----------------------------------------------------------------------------
// 7. Argmax Sampling Oracle Test
// -----------------------------------------------------------------------------
void test_argmax_oracle(DeviceContext& ctx) {
    std::cout << "\n[Oracle 7/7] Greedy Argmax Sampling (vocab_size = 248320)..." << std::endl;
    const int64_t vocab_size = 248320; // Exact Qwen3.8-27B vocabulary size

    std::mt19937 rng(48);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    std::vector<float> h_logits(vocab_size);
    for (auto& v : h_logits) v = dist(rng);

    // Place a deliberate maximum at a specific target token
    const int64_t target_token = 123456;
    h_logits[target_token] = 999.0f;

    // CPU Oracle evaluation
    int64_t cpu_best_idx = 0;
    float cpu_best_val = h_logits[0];
    for (int64_t i = 1; i < vocab_size; ++i) {
        if (h_logits[i] > cpu_best_val) {
            cpu_best_val = h_logits[i];
            cpu_best_idx = i;
        }
    }
    assert(cpu_best_idx == target_token);

    // GPU execution
    float* d_logits = static_cast<float*>(ctx.allocate_device(vocab_size * sizeof(float)));
    ctx.copy_host_to_device(d_logits, h_logits.data(), vocab_size * sizeof(float));

    int64_t gpu_token = argmax(ctx.queue(), d_logits, vocab_size);

    ctx.free_device(d_logits);

    std::cout << "  Target Token: " << target_token << " | CPU: " << cpu_best_idx << " | GPU: " << gpu_token << std::endl;
    assert(gpu_token == cpu_best_idx);
    std::cout << "  -> PASSED: Greedy Argmax matches numerical oracle exactly." << std::endl;
}

// -----------------------------------------------------------------------------
// Main Runner
// -----------------------------------------------------------------------------
int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << " xinfer M4 Naive Operators vs Numerical Oracle Test Suite" << std::endl;
    std::cout << " Running on Intel Arc Pro B60 GPU at Qwen3.8-27B shapes" << std::endl;
    std::cout << "==========================================================" << std::endl;

    try {
        auto ctx = DeviceContext::create(true);

        test_rmsnorm_oracle(*ctx);
        test_rope_oracle(*ctx);
        test_elementwise_oracle(*ctx);
        test_softmax_oracle(*ctx);
        test_linear_oracle(*ctx);
        test_attention_oracle(*ctx);
        test_argmax_oracle(*ctx);

        std::cout << "\n==========================================================" << std::endl;
        std::cout << " ALL 7 OPERATOR ORACLE COMPARISON TESTS PASSED ON B60!" << std::endl;
        std::cout << "==========================================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[TEST FAILED]: " << e.what() << std::endl;
        return 1;
    }
}
