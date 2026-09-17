#include "core/device.h"
#include "core/kv_cache.h"
#include "ops/attention.h"
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    std::cout << "=== Running KVCache Unit Test ===\n";

    auto ctx = xinfer::core::DeviceContext::create(true);
    if (!ctx) {
        std::cerr << "FAILED: Unable to create DeviceContext\n";
        return 1;
    }

    xinfer::core::KVCacheConfig config;
    config.max_seq_len = 512;
    config.num_full_layers = 16;
    config.num_linear_layers = 48;
    config.num_kv_heads = 4;
    config.head_dim = 256;
    config.linear_num_v_heads = 48;
    config.linear_head_k_dim = 128;
    config.linear_head_v_dim = 128;
    config.linear_conv_channels = 10240;
    config.linear_conv_kernel_dim = 4;

    xinfer::core::KVCache cache(ctx, config);
    if (!cache.allocate()) {
        std::cerr << "FAILED: KVCache allocation failed\n";
        return 1;
    }

    std::cout << "[PASS] Allocated KVCache on GPU: "
              << (cache.total_allocated_bytes() / (1024 * 1024)) << " MB\n";

    // Verify all pointers
    for (size_t l = 0; l < 16; ++l) {
        if (!cache.k_cache(l) || !cache.v_cache(l)) {
            std::cerr << "FAILED: Null KV pointer at layer " << l << "\n";
            return 1;
        }
    }
    for (size_t l = 0; l < 48; ++l) {
        if (!cache.linear_state(l) || !cache.conv_state(l)) {
            std::cerr << "FAILED: Null linear/conv pointer at layer " << l << "\n";
            return 1;
        }
    }
    std::cout << "[PASS] Verified all 16 full-attention and 48 linear-attention pointers\n";

    // Verify out-of-range throws std::out_of_range
    try {
        cache.k_cache(16);
        std::cerr << "FAILED: k_cache(16) did not throw out_of_range\n";
        return 1;
    } catch (const std::out_of_range&) {
        std::cout << "[PASS] k_cache(16) correctly threw std::out_of_range\n";
    }

    try {
        cache.linear_state(48);
        std::cerr << "FAILED: linear_state(48) did not throw out_of_range\n";
        return 1;
    } catch (const std::out_of_range&) {
        std::cout << "[PASS] linear_state(48) correctly threw std::out_of_range\n";
    }

    // Test attention write and cached SDPA
    auto& q = ctx->queue();
    int64_t num_tokens = 4;
    int64_t num_q_heads = 24;
    int64_t num_kv_heads = 4;
    int64_t head_dim = 256;

    std::vector<float> h_k(num_tokens * num_kv_heads * head_dim, 0.1f);
    std::vector<float> h_v(num_tokens * num_kv_heads * head_dim, 0.2f);
    std::vector<float> h_q(1 * num_q_heads * head_dim, 0.05f);
    std::vector<float> h_out(1 * num_q_heads * head_dim, 0.0f);

    float* d_k = sycl::malloc_device<float>(h_k.size(), q);
    float* d_v = sycl::malloc_device<float>(h_v.size(), q);
    float* d_q = sycl::malloc_device<float>(h_q.size(), q);
    float* d_out = sycl::malloc_device<float>(h_out.size(), q);

    q.memcpy(d_k, h_k.data(), h_k.size() * sizeof(float));
    q.memcpy(d_v, h_v.data(), h_v.size() * sizeof(float));
    q.memcpy(d_q, h_q.data(), h_q.size() * sizeof(float));
    q.wait();

    // Write 4 tokens into layer 0 KV cache
    xinfer::ops::attention_write_kv_cache(q, cache.k_cache(0), cache.v_cache(0),
                                          d_k, d_v, 0, num_tokens, num_kv_heads, head_dim);

    // Run cached SDPA for 1 query token attending to the 4 cached tokens
    xinfer::ops::sdpa_causal_cached(q, d_out, d_q, cache.k_cache(0), cache.v_cache(0),
                                    3, 1, num_q_heads, num_kv_heads, head_dim);
    q.wait();

    q.memcpy(h_out.data(), d_out, h_out.size() * sizeof(float)).wait();

    // Check output: since all V elements are 0.2f and softmax sums to 1, output should be ~0.2f
    float diff = std::abs(h_out[0] - 0.2f);
    if (diff > 1e-3f) {
        std::cerr << "FAILED: Cached SDPA output mismatch. Expected ~0.2, got " << h_out[0] << "\n";
        return 1;
    }
    std::cout << "[PASS] Cached SDPA correctly attends to past tokens (val=" << h_out[0] << ")\n";

    sycl::free(d_k, q);
    sycl::free(d_v, q);
    sycl::free(d_q, q);
    sycl::free(d_out, q);

    cache.clear();
    if (cache.current_seq_len() != 0) {
        std::cerr << "FAILED: KVCache::clear() did not reset current_seq_len\n";
        return 1;
    }
    std::cout << "[PASS] KVCache::clear() succeeded\n";

    // Test advance and bounds checking
    if (cache.max_seq_len() != 512) {
        std::cerr << "FAILED: KVCache::max_seq_len mismatch\n";
        return 1;
    }
    if (!cache.can_advance(512) || cache.can_advance(513)) {
        std::cerr << "FAILED: KVCache::can_advance check failed\n";
        return 1;
    }
    cache.set_seq_len(510);
    if (!cache.advance(2) || cache.current_seq_len() != 512) {
        std::cerr << "FAILED: KVCache::advance to limit failed\n";
        return 1;
    }
    if (cache.advance(1)) {
        std::cerr << "FAILED: KVCache::advance past max_seq_len did not return false\n";
        return 1;
    }
    if (cache.current_seq_len() != 512) {
        std::cerr << "FAILED: KVCache::advance past max_seq_len did not clamp current_seq_len\n";
        return 1;
    }
    std::cout << "[PASS] KVCache bounds checking and advance() clamping verified\n";

    std::cout << "All KVCache tests PASSED!\n";
    return 0;
}
