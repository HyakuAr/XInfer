// tests/test_eval_oracle.cpp
//
// Numerical oracle test for the fused cross-entropy loss kernel (src/ops/eval.cpp).
// Per AGENTS.md §7: every kernel that touches model math needs one independent
// oracle — a naive FP32 reference implementation that does NOT copy the production
// kernel's staging casts, reduction order, or intermediate dtype.
//
// Test cases:
//   1. Small vocab (10), known logits → exact NLL match vs hand-computed values
//   2. Realistic vocab (248,320), synthetic logits → tolerance-based comparison
//   3. Rank computation correctness
//   4. Edge cases: single token, uniform logits, extreme values
//   5. Sequence-level perplexity computation

#include "ops/eval.h"
#include "core/device.h"
#include <sycl/sycl.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
#include <limits>
#include <algorithm>
#include <random>
#include <cassert>
#include <iomanip>

namespace {

// FP64 reference oracle for cross-entropy loss — deliberately different
// implementation from the production kernel (FP32, SLM tree reductions).
// This uses FP64 sequential computation as the ground truth.
struct OracleResult {
    std::vector<double> per_token_nll;
    std::vector<int64_t> per_token_rank;
    double total_loss{0.0};
    double perplexity{0.0};
};

OracleResult oracle_cross_entropy(
    const std::vector<float>& logits,
    const std::vector<int64_t>& token_ids,
    int64_t seq_len,
    int64_t vocab_size) {

    OracleResult result;
    int64_t num_positions = seq_len - 1;

    result.per_token_nll.resize(num_positions);
    result.per_token_rank.resize(num_positions);

    for (int64_t pos = 0; pos < num_positions; ++pos) {
        const float* row = logits.data() + pos * vocab_size;
        int64_t target_id = token_ids[pos + 1];

        // FP64 log_softmax: max-subtract → exp → sum → log → extract
        double row_max = -std::numeric_limits<double>::infinity();
        for (int64_t v = 0; v < vocab_size; ++v) {
            double val = static_cast<double>(row[v]);
            if (val > row_max) row_max = val;
        }

        double sum_exp = 0.0;
        for (int64_t v = 0; v < vocab_size; ++v) {
            sum_exp += std::exp(static_cast<double>(row[v]) - row_max);
        }

        double log_sum_exp = std::log(sum_exp);
        double target_logit = static_cast<double>(row[target_id]);
        double log_prob = (target_logit - row_max) - log_sum_exp;
        result.per_token_nll[pos] = -log_prob;

        // Rank: number of tokens with strictly higher logit
        int64_t rank = 0;
        for (int64_t v = 0; v < vocab_size; ++v) {
            if (static_cast<double>(row[v]) > target_logit) rank++;
        }
        result.per_token_rank[pos] = rank;
    }

    for (int64_t i = 0; i < num_positions; ++i) {
        result.total_loss += result.per_token_nll[i];
    }
    result.perplexity = std::exp(result.total_loss / static_cast<double>(num_positions));

    return result;
}

int tests_passed = 0;
int tests_failed = 0;

void check(bool condition, const std::string& name) {
    if (condition) {
        std::cout << "  [PASS] " << name << "\n";
        tests_passed++;
    } else {
        std::cerr << "  [FAIL] " << name << "\n";
        tests_failed++;
    }
}

void check_close(double actual, double expected, double atol, const std::string& name) {
    double err = std::abs(actual - expected);
    if (err <= atol) {
        std::cout << "  [PASS] " << name
                  << " (actual=" << std::fixed << std::setprecision(6) << actual
                  << ", expected=" << expected
                  << ", err=" << std::scientific << err << ")\n";
        tests_passed++;
    } else {
        std::cerr << "  [FAIL] " << name
                  << " (actual=" << std::fixed << std::setprecision(6) << actual
                  << ", expected=" << expected
                  << ", err=" << std::scientific << err
                  << ", atol=" << atol << ")\n";
        tests_failed++;
    }
}

} // anonymous namespace

int main() {
    std::cout << "=== Cross-Entropy Loss Kernel Oracle Test ===\n\n";

    // Initialize SYCL device
    auto ctx = xinfer::core::DeviceContext::create(true);
    if (!ctx) {
        std::cerr << "FATAL: Failed to initialize Intel GPU DeviceContext\n";
        return 2;
    }
    auto& q = ctx->queue();
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n\n";

    // -------------------------------------------------------
    // Test 1: Small vocab, known logits
    // -------------------------------------------------------
    {
        std::cout << "Test 1: Small vocab (V=10), seq_len=4\n";

        constexpr int64_t V = 10;
        constexpr int64_t S = 4;

        // logits[0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  → predict token_ids[1]=9 (argmax)
        // logits[1] = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]  → predict token_ids[2]=0 (argmax)
        // logits[2] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  → predict token_ids[3]=5 (uniform)
        std::vector<float> logits(S * V);
        for (int v = 0; v < V; ++v) logits[0 * V + v] = static_cast<float>(v);
        for (int v = 0; v < V; ++v) logits[1 * V + v] = static_cast<float>(9 - v);
        for (int v = 0; v < V; ++v) logits[2 * V + v] = 1.0f;
        for (int v = 0; v < V; ++v) logits[3 * V + v] = 0.0f; // unused (last position)

        std::vector<int64_t> token_ids = {0, 9, 0, 5};

        // Compute oracle
        auto oracle = oracle_cross_entropy(logits, token_ids, S, V);

        // Upload to device
        float* d_logits = sycl::malloc_device<float>(S * V, q);
        int64_t* d_tokens = sycl::malloc_device<int64_t>(S, q);
        q.memcpy(d_logits, logits.data(), S * V * sizeof(float)).wait();
        q.memcpy(d_tokens, token_ids.data(), S * sizeof(int64_t)).wait();

        // Run kernel
        auto result = xinfer::ops::cross_entropy_loss(q, d_logits, d_tokens, S, V, true);

        check(result.num_tokens == 3, "num_tokens == 3");
        check_close(result.total_loss, oracle.total_loss, 1e-4, "total_loss matches oracle");
        check_close(result.perplexity, oracle.perplexity, 1e-3, "perplexity matches oracle");

        // Per-token checks
        for (size_t i = 0; i < oracle.per_token_nll.size(); ++i) {
            check_close(result.per_token_loss[i], oracle.per_token_nll[i], 1e-4,
                        "per_token_loss[" + std::to_string(i) + "]");
        }

        // Rank checks
        // Position 0: logits=[0..9], target=9 (max). Rank should be 0.
        check(result.per_token_rank[0] == 0, "rank[0] == 0 (target is argmax)");
        // Position 1: logits=[9..0], target=0 (also max). Rank should be 0.
        check(result.per_token_rank[1] == 0, "rank[1] == 0 (target is argmax)");
        // Position 2: uniform logits, target=5. Rank should be 0 (all equal).
        check(result.per_token_rank[2] == 0, "rank[2] == 0 (uniform logits)");

        // For uniform distribution with V=10, each NLL should be log(10)
        double expected_uniform_nll = std::log(10.0);
        check_close(result.per_token_loss[2], expected_uniform_nll, 1e-4,
                    "uniform NLL == log(10)");

        sycl::free(d_logits, q);
        sycl::free(d_tokens, q);
    }

    // -------------------------------------------------------
    // Test 2: Realistic vocab size with random logits
    // -------------------------------------------------------
    {
        std::cout << "\nTest 2: Realistic vocab (V=248320), seq_len=16\n";

        constexpr int64_t V = 248320;
        constexpr int64_t S = 16;

        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 3.0f);

        std::vector<float> logits(S * V);
        for (auto& x : logits) x = dist(rng);

        std::uniform_int_distribution<int64_t> tok_dist(0, V - 1);
        std::vector<int64_t> token_ids(S);
        for (auto& t : token_ids) t = tok_dist(rng);

        // Compute oracle (this is slow for V=248K but it's a test)
        std::cout << "  Computing FP64 oracle (may take a moment)...\n";
        auto oracle = oracle_cross_entropy(logits, token_ids, S, V);

        // Upload
        float* d_logits = sycl::malloc_device<float>(S * V, q);
        int64_t* d_tokens = sycl::malloc_device<int64_t>(S, q);
        q.memcpy(d_logits, logits.data(), S * V * sizeof(float)).wait();
        q.memcpy(d_tokens, token_ids.data(), S * sizeof(int64_t)).wait();

        // Run kernel
        auto result = xinfer::ops::cross_entropy_loss(q, d_logits, d_tokens, S, V, true);

        check(result.num_tokens == static_cast<size_t>(S - 1), "num_tokens == 15");

        // FP32 kernel vs FP64 oracle: allow more tolerance for large reductions
        // over 248K elements. Empirically, relative error should be < 0.01%.
        double rel_err = std::abs(result.total_loss - oracle.total_loss) / oracle.total_loss;
        check(rel_err < 1e-3, "total_loss relative error < 0.1% (actual: " +
              std::to_string(rel_err * 100.0) + "%)");

        double ppl_rel_err = std::abs(result.perplexity - oracle.perplexity) / oracle.perplexity;
        check(ppl_rel_err < 1e-2, "perplexity relative error < 1% (actual: " +
              std::to_string(ppl_rel_err * 100.0) + "%)");

        // Check per-token losses individually
        int per_tok_ok = 0;
        double max_per_tok_err = 0.0;
        for (int64_t i = 0; i < S - 1; ++i) {
            double err = std::abs(static_cast<double>(result.per_token_loss[i]) - oracle.per_token_nll[i]);
            if (err > max_per_tok_err) max_per_tok_err = err;
            if (err < 0.01) per_tok_ok++;
        }
        check(per_tok_ok == S - 1,
              "all per-token losses within 0.01 atol (passed " + std::to_string(per_tok_ok) +
              "/" + std::to_string(S - 1) + ", max_err=" + std::to_string(max_per_tok_err) + ")");

        // Check ranks match oracle
        int rank_ok = 0;
        for (int64_t i = 0; i < S - 1; ++i) {
            if (result.per_token_rank[i] == oracle.per_token_rank[i]) rank_ok++;
        }
        check(rank_ok == S - 1,
              "all per-token ranks match oracle (" + std::to_string(rank_ok) +
              "/" + std::to_string(S - 1) + ")");

        sycl::free(d_logits, q);
        sycl::free(d_tokens, q);
    }

    // -------------------------------------------------------
    // Test 3: Edge case — seq_len=2 (single prediction)
    // -------------------------------------------------------
    {
        std::cout << "\nTest 3: Edge case — seq_len=2 (single prediction)\n";

        constexpr int64_t V = 100;
        std::vector<float> logits(2 * V, 0.0f);
        logits[0 * V + 42] = 10.0f;  // Strong prediction for token 42
        std::vector<int64_t> token_ids = {0, 42};

        auto oracle = oracle_cross_entropy(logits, token_ids, 2, V);

        float* d_logits = sycl::malloc_device<float>(2 * V, q);
        int64_t* d_tokens = sycl::malloc_device<int64_t>(2, q);
        q.memcpy(d_logits, logits.data(), 2 * V * sizeof(float)).wait();
        q.memcpy(d_tokens, token_ids.data(), 2 * sizeof(int64_t)).wait();

        auto result = xinfer::ops::cross_entropy_loss(q, d_logits, d_tokens, 2, V, true);

        check(result.num_tokens == 1, "num_tokens == 1");
        check_close(result.per_token_loss[0], oracle.per_token_nll[0], 1e-4,
                    "single-token NLL matches oracle");
        check(result.per_token_rank[0] == 0, "rank == 0 (target has highest logit)");

        // With logit of 10.0 and 99 others at 0.0, the NLL should be very small
        check(result.per_token_loss[0] < 0.01, "NLL << 1 for confident prediction");

        sycl::free(d_logits, q);
        sycl::free(d_tokens, q);
    }

    // -------------------------------------------------------
    // Test 4: Edge case — extremely large/small logit values
    // -------------------------------------------------------
    {
        std::cout << "\nTest 4: Numerical stability — extreme logit values\n";

        constexpr int64_t V = 10;
        constexpr int64_t S = 3;

        std::vector<float> logits(S * V);
        // Row 0: very large values (should not overflow exp)
        for (int v = 0; v < V; ++v) logits[0 * V + v] = 1000.0f + static_cast<float>(v);
        // Row 1: very small values (should not underflow)
        for (int v = 0; v < V; ++v) logits[1 * V + v] = -1000.0f + static_cast<float>(v);

        std::vector<int64_t> token_ids = {0, 9, 9};

        auto oracle = oracle_cross_entropy(logits, token_ids, S, V);

        float* d_logits = sycl::malloc_device<float>(S * V, q);
        int64_t* d_tokens = sycl::malloc_device<int64_t>(S, q);
        q.memcpy(d_logits, logits.data(), S * V * sizeof(float)).wait();
        q.memcpy(d_tokens, token_ids.data(), S * sizeof(int64_t)).wait();

        auto result = xinfer::ops::cross_entropy_loss(q, d_logits, d_tokens, S, V, false);

        check(std::isfinite(result.total_loss), "total_loss is finite (no overflow/underflow)");
        check(std::isfinite(result.perplexity), "perplexity is finite");
        check_close(result.total_loss, oracle.total_loss, 1e-3,
                    "total_loss matches oracle even with extreme logits");

        sycl::free(d_logits, q);
        sycl::free(d_tokens, q);
    }

    // -------------------------------------------------------
    // Test 5: Edge case — seq_len <= 1 returns empty
    // -------------------------------------------------------
    {
        std::cout << "\nTest 5: Edge case — seq_len=0 and seq_len=1\n";

        auto r0 = xinfer::ops::cross_entropy_loss(q, nullptr, nullptr, 0, 10);
        check(r0.num_tokens == 0, "seq_len=0 → num_tokens=0");
        check(r0.total_loss == 0.0, "seq_len=0 → total_loss=0");

        auto r1 = xinfer::ops::cross_entropy_loss(q, nullptr, nullptr, 1, 10);
        check(r1.num_tokens == 0, "seq_len=1 → num_tokens=0");
    }

    // -------------------------------------------------------
    // Summary
    // -------------------------------------------------------
    std::cout << "\n=== Oracle Test Summary ===\n";
    std::cout << "  Passed: " << tests_passed << "\n";
    std::cout << "  Failed: " << tests_failed << "\n";

    if (tests_failed > 0) {
        std::cerr << "\nFAILURE: " << tests_failed << " oracle test(s) failed.\n";
        return 1;
    }
    std::cout << "\nAll oracle tests passed.\n";
    return 0;
}
