#include "core/device.h"
#include "core/kv_cache.h"
#include "core/arena.h"
#include "ops/attention.h"
#include "ops/sampling.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

int main() {
    std::cout << "=== Running Speculative MTP Unit Test ===\n";

    auto ctx = xinfer::core::DeviceContext::create(true);
    if (!ctx) {
        std::cerr << "FAILED: Unable to create DeviceContext\n";
        return 1;
    }
    auto& q = ctx->queue();

    // 1. Test sdpa_causal_verify with FP16 KV cache
    std::cout << "\n--- Testing sdpa_causal_verify (M = N batched attention) ---\n";
    xinfer::core::KVCacheConfig config;
    config.max_seq_len = 256;
    config.num_full_layers = 1;
    config.num_linear_layers = 0;
    config.num_kv_heads = 4;
    config.head_dim = 256;

    xinfer::core::KVCache cache(ctx, config);
    if (!cache.allocate()) {
        std::cerr << "FAILED: KVCache allocation failed\n";
        return 1;
    }

    int64_t prefix_len = 8;
    int64_t num_draft = 4;
    int64_t total_tokens = prefix_len + num_draft;
    int64_t num_q_heads = 24;
    int64_t num_kv_heads = 4;
    int64_t head_dim = 256;

    std::vector<float> h_k(total_tokens * num_kv_heads * head_dim, 0.1f);
    std::vector<float> h_v(total_tokens * num_kv_heads * head_dim, 0.2f);
    std::vector<float> h_q(num_draft * num_q_heads * head_dim, 0.05f);

    float* d_k = sycl::malloc_device<float>(h_k.size(), q);
    float* d_v = sycl::malloc_device<float>(h_v.size(), q);
    float* d_q = sycl::malloc_device<float>(h_q.size(), q);
    float* d_out = sycl::malloc_device<float>(num_draft * num_q_heads * head_dim, q);

    q.memcpy(d_k, h_k.data(), h_k.size() * sizeof(float)).wait();
    q.memcpy(d_v, h_v.data(), h_v.size() * sizeof(float)).wait();
    q.memcpy(d_q, h_q.data(), h_q.size() * sizeof(float)).wait();

    // Write all prefix + draft tokens into KV cache
    xinfer::ops::attention_write_kv_cache(q, cache.k_cache(0), cache.v_cache(0),
                                          d_k, d_v, 0, total_tokens, num_kv_heads, head_dim);
    q.wait();

    // Run sdpa_causal_verify for the M = 4 draft tokens at prefix_len = 8
    xinfer::ops::sdpa_causal_verify(q, d_out, d_q, cache.k_cache(0), cache.v_cache(0),
                                    prefix_len, num_draft, num_q_heads, num_kv_heads, head_dim);
    q.wait();

    std::vector<float> h_out(num_draft * num_q_heads * head_dim);
    q.memcpy(h_out.data(), d_out, h_out.size() * sizeof(float)).wait();

    for (int64_t i = 0; i < num_draft; ++i) {
        float val = h_out[i * num_q_heads * head_dim];
        if (std::isnan(val) || std::isinf(val) || val <= 0.0f) {
            std::cerr << "FAILED: Invalid attention output for draft token " << i << ": " << val << "\n";
            return 1;
        }
    }
    std::cout << "[PASS] sdpa_causal_verify executed successfully across all " << num_draft << " draft tokens\n";

    sycl::free(d_k, q);
    sycl::free(d_v, q);
    sycl::free(d_q, q);
    sycl::free(d_out, q);

    // 2. Test speculative acceptance logic and fail-loud contract
    std::cout << "\n--- Testing speculative_accept_reject & Fail-Loud Contract ---\n";
    constexpr int64_t vocab_size = 1000;
    size_t N = 4;

    // Synthetic verified target logits for 4 positions
    std::vector<float> h_target_logits(N * vocab_size, -10.0f);
    // Position 0: target argmax is token 42
    h_target_logits[0 * vocab_size + 42] = 10.0f;
    // Position 1: target argmax is token 100
    h_target_logits[1 * vocab_size + 100] = 10.0f;
    // Position 2: target argmax is token 200
    h_target_logits[2 * vocab_size + 200] = 10.0f;
    // Position 3: target argmax is token 300
    h_target_logits[3 * vocab_size + 300] = 10.0f;

    float* d_target_logits = sycl::malloc_device<float>(h_target_logits.size(), q);
    q.memcpy(d_target_logits, h_target_logits.data(), h_target_logits.size() * sizeof(float)).wait();

    // Scenario A: All N tokens match (100% acceptance)
    {
        std::vector<int64_t> draft_tokens = {42, 100, 200, 300};
        auto res = xinfer::ops::speculative_accept_reject(q, draft_tokens.data(), N, d_target_logits, vocab_size);

        if (res.has_rejected_token) {
            std::cerr << "FAILED Scenario A: expected all accepted, got rejected token\n";
            return 1;
        }
        if (res.accepted_tokens.size() != N) {
            std::cerr << "FAILED Scenario A: accepted_tokens.size() = " << res.accepted_tokens.size() << " != " << N << "\n";
            return 1;
        }
        if (res.num_accepted != N || res.num_rejected != 0 || res.num_remaining != 0) {
            std::cerr << "FAILED Scenario A: accounting mismatch\n";
            return 1;
        }
        std::cout << "[PASS] Scenario A: All " << N << " tokens accepted (bonus_token="
                  << res.bonus_or_resampled_token << ")\n";
    }

    // Scenario B: Rejection at token index 2
    {
        std::vector<int64_t> draft_tokens = {42, 100, 999 /* wrong */, 300};
        auto res = xinfer::ops::speculative_accept_reject(q, draft_tokens.data(), N, d_target_logits, vocab_size);

        if (!res.has_rejected_token) {
            std::cerr << "FAILED Scenario B: expected rejection at index 2\n";
            return 1;
        }
        if (res.num_accepted != 2) {
            std::cerr << "FAILED Scenario B: num_accepted = " << res.num_accepted << " != 2\n";
            return 1;
        }
        if (res.num_rejected != 1) {
            std::cerr << "FAILED Scenario B: num_rejected = " << res.num_rejected << " != 1\n";
            return 1;
        }
        if (res.num_remaining != 1) {
            std::cerr << "FAILED Scenario B: num_remaining = " << res.num_remaining << " != 1\n";
            return 1;
        }
        // Check fail-loud invariant: accepted (2) + 1 (rejected) + remaining (1) == N (4)
        if (res.num_accepted + 1 + res.num_remaining != N) {
            std::cerr << "FAILED Scenario B: fail-loud invariant sum != N\n";
            return 1;
        }
        // Resampled replacement token must be target's argmax (200)
        if (res.bonus_or_resampled_token != 200) {
            std::cerr << "FAILED Scenario B: resampled token = " << res.bonus_or_resampled_token << " != 200\n";
            return 1;
        }
        std::cout << "[PASS] Scenario B: Correctly accepted 2, rejected 1 (token 999), remaining 1, resampled replacement=200\n";
    }

    // Scenario C: Rejection at token index 0
    {
        std::vector<int64_t> draft_tokens = {999 /* wrong */, 100, 200, 300};
        auto res = xinfer::ops::speculative_accept_reject(q, draft_tokens.data(), N, d_target_logits, vocab_size);

        if (res.num_accepted != 0 || res.num_rejected != 1 || res.num_remaining != 3) {
            std::cerr << "FAILED Scenario C: accounting mismatch: " << res.num_accepted << " + "
                      << res.num_rejected << " + " << res.num_remaining << "\n";
            return 1;
        }
        if (res.num_accepted + 1 + res.num_remaining != N) {
            std::cerr << "FAILED Scenario C: invariant violated\n";
            return 1;
        }
        std::cout << "[PASS] Scenario C: First token rejected (0 accepted, 1 rejected, 3 remaining, resampled="
                  << res.bonus_or_resampled_token << ")\n";
    }

    // Scenario D: Stochastic ratio acceptance (P_target / P_draft)
    {
        std::vector<int64_t> draft_tokens = {42, 100, 200, 300};
        // High draft probability (1.0), low random number (0.1) -> accepted
        std::vector<float> draft_probs = {1.0f, 1.0f, 1.0f, 1.0f};
        std::vector<float> rand_vals = {0.1f, 0.1f, 0.1f, 0.1f};

        auto res = xinfer::ops::speculative_accept_reject(q, draft_tokens.data(), N, d_target_logits, vocab_size,
                                                          draft_probs.data(), rand_vals.data());
        if (res.num_accepted != N) {
            std::cerr << "FAILED Scenario D: expected all accepted with low r\n";
            return 1;
        }
        std::cout << "[PASS] Scenario D: Stochastic acceptance with probability ratio passed\n";
    }

    sycl::free(d_target_logits, q);

    std::cout << "\nAll Speculative MTP Unit Tests PASSED successfully!\n";
    return 0;
}
