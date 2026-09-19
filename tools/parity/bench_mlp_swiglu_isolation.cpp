// Substep 1 of perf/reconcile-microbenchmark-vs-real-decode-path:
// Diagnose WHY mlp_gate_up_swiglu_int4 runs ~22x slower in the real decode
// step (5.41 ms/layer) vs. the tight-loop microbenchmark (~0.24 ms for
// gate+up at N=17408, K=5120).
//
// Two live hypotheses from the roadmap:
// 1. Register pressure: mlp_gate_up_swiglu_int4_impl holds lane_acc_g AND
//    lane_acc_u simultaneously — double the live registers vs. linear_int4_impl.
//    If this causes register spilling, occupancy drops below the formula's
//    prediction (which only counts sub-groups, not register-file pressure).
// 2. Cold-start / cache-miss pattern: 13 distinct kernel shapes fired
//    back-to-back per layer (64 layers) means the GPU never reaches the
//    steady-state clock/cache warmth the tight-loop microbenchmark enjoys.
//
// Citing vendor documentation per AGENTS.md §5:
// - docs/vendor/xe-gpu-architecture.md (lines 31-32):
//   General register file per thread: 128 / 256 (regular / large-register mode).
//   Register width: 512 bits.
// - docs/vendor/thread-mapping-occupancy.md (lines 27-29):
//   Using the large register file mode cuts Xe-Core occupancy in half.
//   This directly trades off against XMX tuning recommendations.
// - docs/vendor/b60-matrix-caps.md (Sections 2-4):
//   No native INT4 XMX support. Vector Engine GEMV is the production path.

#include "core/device.h"
#include "ops/linear.h"
#include "ops/rmsnorm.h"
#include "ops/elementwise.h"

#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace xinfer;

// Qwen3.8-27B model dimensions for realistic shapes
constexpr int64_t HIDDEN     = 5120;
constexpr int64_t INTER      = 17408;  // intermediate_size (MLP gate/up N)
constexpr int     GROUP_SIZE = 128;

// Hardware reference
constexpr double B60_PEAK_BW_GBS = 456.0;
constexpr int    B60_HW_THREADS  = 1280;

struct BenchResult {
    double avg_ms;
    double min_ms;
    double max_ms;
    double stddev_ms;
    double bw_gbs;
    std::vector<double> per_iter_ms;
};

// Compute bandwidth given bytes moved and time
static double calc_bw(size_t bytes, double ms) {
    return (ms > 0.0) ? (static_cast<double>(bytes) / 1e9) / (ms / 1000.0) : 0.0;
}

// Measure a kernel via SYCL event profiling over N iterations
// Returns per-iteration device-side timings
static BenchResult measure_events(const std::vector<sycl::event>& events, size_t bytes_per_iter) {
    BenchResult r;
    r.per_iter_ms.reserve(events.size());

    for (const auto& ev : events) {
        uint64_t s = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        uint64_t e = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        r.per_iter_ms.push_back(static_cast<double>(e - s) * 1e-6);
    }

    double sum = std::accumulate(r.per_iter_ms.begin(), r.per_iter_ms.end(), 0.0);
    r.avg_ms = sum / static_cast<double>(r.per_iter_ms.size());
    r.min_ms = *std::min_element(r.per_iter_ms.begin(), r.per_iter_ms.end());
    r.max_ms = *std::max_element(r.per_iter_ms.begin(), r.per_iter_ms.end());

    double var = 0.0;
    for (double v : r.per_iter_ms) var += (v - r.avg_ms) * (v - r.avg_ms);
    r.stddev_ms = std::sqrt(var / static_cast<double>(r.per_iter_ms.size()));

    r.bw_gbs = calc_bw(bytes_per_iter, r.avg_ms);
    return r;
}

int main() {
    std::cout << "==================================================================" << std::endl;
    std::cout << " Substep 1: MLP SwiGLU Isolation vs Cold-Context Diagnostic" << std::endl;
    std::cout << " Goal: Find the actual limiter behind the 22x microbench-vs-decode gap" << std::endl;
#ifdef XINFER_BUILD_CONFIG
    std::cout << " Build Config: " << XINFER_BUILD_CONFIG << std::endl;
#endif
    std::cout << "==================================================================" << std::endl;

    auto ctx = core::DeviceContext::create(true);
    sycl::queue& q = ctx->queue();
    auto dev = q.get_device();

    std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << std::endl;

    // =========================================================================
    // Allocate realistic-shape buffers
    // =========================================================================
    const int64_t M = 1;
    const int64_t K = HIDDEN;
    const int64_t N = INTER;

    size_t w_bytes = static_cast<size_t>(N) * (K / 2);          // INT4 weights per matrix
    size_t s_bytes = static_cast<size_t>(N) * (K / GROUP_SIZE) * sizeof(sycl::half);
    size_t x_bytes = static_cast<size_t>(K) * sizeof(float);
    size_t y_bytes = static_cast<size_t>(N) * sizeof(float);

    // MLP SwiGLU reads: gate_weights + up_weights + gate_scales + up_scales + X
    // Writes: Y_swiglu
    size_t swiglu_bytes = 2 * w_bytes + 2 * s_bytes + x_bytes + y_bytes;

    // Single linear_int4 reads: weights + scales + X, writes Y
    size_t linear_bytes = w_bytes + s_bytes + x_bytes + y_bytes;

    float*      d_X     = sycl::malloc_device<float>(K, q);
    float*      d_Y     = sycl::malloc_device<float>(N, q);
    uint8_t*    d_Wg    = sycl::malloc_device<uint8_t>(w_bytes, q);
    uint8_t*    d_Wu    = sycl::malloc_device<uint8_t>(w_bytes, q);
    sycl::half* d_Sg    = sycl::malloc_device<sycl::half>(N * (K / GROUP_SIZE), q);
    sycl::half* d_Su    = sycl::malloc_device<sycl::half>(N * (K / GROUP_SIZE), q);

    // Additional buffers for "cold-context" interleaving kernels
    float*      d_norm_w = sycl::malloc_device<float>(K, q);
    float*      d_normed = sycl::malloc_device<float>(K, q);
    float*      d_X2     = sycl::malloc_device<float>(K, q);

    // Tiny-N projection buffers (simulating attention B/A, N=48)
    constexpr int64_t TINY_N = 48;
    size_t tiny_w_bytes = static_cast<size_t>(TINY_N) * (K / 2);
    float*      d_Y_tiny = sycl::malloc_device<float>(TINY_N, q);
    uint8_t*    d_W_tiny = sycl::malloc_device<uint8_t>(tiny_w_bytes, q);
    sycl::half* d_S_tiny = sycl::malloc_device<sycl::half>(TINY_N * (K / GROUP_SIZE), q);

    // Initialize with non-zero data
    q.fill(d_X, 1.0f, K);
    q.fill(d_Y, 0.0f, N);
    q.fill(d_Wg, static_cast<uint8_t>(0x33), w_bytes);
    q.fill(d_Wu, static_cast<uint8_t>(0x55), w_bytes);
    q.fill(d_Sg, sycl::half{0.1f}, N * (K / GROUP_SIZE));
    q.fill(d_Su, sycl::half{0.1f}, N * (K / GROUP_SIZE));
    q.fill(d_norm_w, 1.0f, K);
    q.fill(d_normed, 0.0f, K);
    q.fill(d_X2, 0.0f, K);
    q.fill(d_Y_tiny, 0.0f, TINY_N);
    q.fill(d_W_tiny, static_cast<uint8_t>(0x22), tiny_w_bytes);
    q.fill(d_S_tiny, sycl::half{0.1f}, TINY_N * (K / GROUP_SIZE));
    q.wait();

    // =========================================================================
    // Warmup: compile all kernel variants
    // =========================================================================
    std::cout << "\nWarming up all kernel variants..." << std::endl;
    ops::mlp_gate_up_swiglu_int4(q, d_Y, d_X, d_Wg, d_Sg, d_Wu, d_Su, M, N, K, GROUP_SIZE);
    ops::linear_int4(q, d_Y, d_X, d_Wg, d_Sg, nullptr, M, N, K, GROUP_SIZE);
    ops::linear_int4(q, d_Y_tiny, d_X, d_W_tiny, d_S_tiny, nullptr, M, TINY_N, K, GROUP_SIZE);
    ops::rmsnorm(q, d_normed, d_X, d_norm_w, 1, K);
    ops::add_inplace(q, d_X2, d_normed, K);
    q.wait();

    constexpr int WARM_ITERS = 20;
    constexpr int COLD_LAYERS = 16; // Simulate 16 layers of interleaved kernels

    // =========================================================================
    // TEST 1: Tight-loop (warm) MLP SwiGLU — same as M7 microbenchmark style
    // =========================================================================
    std::cout << "\n--- TEST 1: MLP SwiGLU in tight loop (warm, " << WARM_ITERS << " iters) ---" << std::endl;
    {
        std::vector<sycl::event> events;
        events.reserve(WARM_ITERS);
        q.wait();

        for (int i = 0; i < WARM_ITERS; ++i) {
            events.push_back(
                ops::mlp_gate_up_swiglu_int4(q, d_Y, d_X, d_Wg, d_Sg, d_Wu, d_Su,
                                             M, N, K, GROUP_SIZE)
            );
        }
        q.wait();

        auto r = measure_events(events, swiglu_bytes);
        std::cout << "  Avg: " << std::fixed << std::setprecision(3) << r.avg_ms << " ms"
                  << "  Min: " << r.min_ms << " ms"
                  << "  Max: " << r.max_ms << " ms"
                  << "  StdDev: " << r.stddev_ms << " ms" << std::endl;
        std::cout << "  Bandwidth: " << std::setprecision(1) << r.bw_gbs << " GB/s"
                  << " (" << (r.bw_gbs / B60_PEAK_BW_GBS * 100.0) << "% of B60 peak)" << std::endl;

        // Print per-iteration to show warm-up curve
        std::cout << "  Per-iter (ms): ";
        for (int i = 0; i < std::min(WARM_ITERS, 10); ++i)
            std::cout << std::setprecision(3) << r.per_iter_ms[i] << " ";
        if (WARM_ITERS > 10) std::cout << "...";
        std::cout << std::endl;
    }

    // =========================================================================
    // TEST 2: Tight-loop (warm) linear_int4 at same shape — baseline comparison
    // =========================================================================
    std::cout << "\n--- TEST 2: linear_int4 in tight loop (warm, " << WARM_ITERS << " iters, same N=" << N << " K=" << K << ") ---" << std::endl;
    {
        std::vector<sycl::event> events;
        events.reserve(WARM_ITERS);
        q.wait();

        for (int i = 0; i < WARM_ITERS; ++i) {
            events.push_back(
                ops::linear_int4(q, d_Y, d_X, d_Wg, d_Sg, nullptr, M, N, K, GROUP_SIZE)
            );
        }
        q.wait();

        auto r = measure_events(events, linear_bytes);
        std::cout << "  Avg: " << std::fixed << std::setprecision(3) << r.avg_ms << " ms"
                  << "  Min: " << r.min_ms << " ms"
                  << "  Max: " << r.max_ms << " ms"
                  << "  StdDev: " << r.stddev_ms << " ms" << std::endl;
        std::cout << "  Bandwidth: " << std::setprecision(1) << r.bw_gbs << " GB/s"
                  << " (" << (r.bw_gbs / B60_PEAK_BW_GBS * 100.0) << "% of B60 peak)" << std::endl;
    }

    // =========================================================================
    // TEST 3: Register pressure comparison — swiglu vs 2x sequential linear
    // =========================================================================
    std::cout << "\n--- TEST 3: 2x sequential linear_int4 (gate then up, same total work) ---" << std::endl;
    {
        std::vector<sycl::event> events_g, events_u;
        events_g.reserve(WARM_ITERS);
        events_u.reserve(WARM_ITERS);
        q.wait();

        for (int i = 0; i < WARM_ITERS; ++i) {
            events_g.push_back(
                ops::linear_int4(q, d_Y, d_X, d_Wg, d_Sg, nullptr, M, N, K, GROUP_SIZE)
            );
            events_u.push_back(
                ops::linear_int4(q, d_Y, d_X, d_Wu, d_Su, nullptr, M, N, K, GROUP_SIZE)
            );
        }
        q.wait();

        auto rg = measure_events(events_g, linear_bytes);
        auto ru = measure_events(events_u, linear_bytes);
        double total_avg = rg.avg_ms + ru.avg_ms;
        std::cout << "  Gate avg: " << std::setprecision(3) << rg.avg_ms << " ms"
                  << "  Up avg: " << ru.avg_ms << " ms"
                  << "  Combined: " << total_avg << " ms" << std::endl;
        std::cout << "  If SwiGLU is >> combined, register pressure is the cause." << std::endl;
    }

    // =========================================================================
    // TEST 4: Cold-context simulation — interleave SwiGLU with other kernel
    // shapes to simulate the real decode step's pattern
    // =========================================================================
    std::cout << "\n--- TEST 4: Cold-context interleaved (" << COLD_LAYERS << " simulated layers) ---" << std::endl;
    std::cout << "  Pattern per layer: RMSNorm -> tiny_linear(N=48) -> linear(N=" << N << ") -> SwiGLU -> RMSNorm -> add" << std::endl;
    {
        std::vector<sycl::event> swiglu_events;
        std::vector<sycl::event> all_events; // everything else
        swiglu_events.reserve(COLD_LAYERS);
        all_events.reserve(COLD_LAYERS * 5);
        q.wait();

        for (int l = 0; l < COLD_LAYERS; ++l) {
            // RMSNorm (pre-attention)
            all_events.push_back(
                ops::rmsnorm(q, d_normed, d_X, d_norm_w, 1, K)
            );
            // Tiny-N linear (attention B or A projection, N=48)
            all_events.push_back(
                ops::linear_int4(q, d_Y_tiny, d_normed, d_W_tiny, d_S_tiny,
                                 nullptr, M, TINY_N, K, GROUP_SIZE)
            );
            // Large linear (attention Q or Out projection, N=INTER)
            all_events.push_back(
                ops::linear_int4(q, d_Y, d_normed, d_Wg, d_Sg,
                                 nullptr, M, N, K, GROUP_SIZE)
            );
            // MLP SwiGLU — the kernel we're profiling
            swiglu_events.push_back(
                ops::mlp_gate_up_swiglu_int4(q, d_Y, d_normed, d_Wg, d_Sg,
                                             d_Wu, d_Su, M, N, K, GROUP_SIZE)
            );
            // Post-MLP RMSNorm + residual add
            all_events.push_back(
                ops::rmsnorm(q, d_normed, d_X, d_norm_w, 1, K)
            );
            all_events.push_back(
                ops::add_inplace(q, d_X2, d_normed, K)
            );
        }
        q.wait();

        auto r_swiglu = measure_events(swiglu_events, swiglu_bytes);
        std::cout << "  SwiGLU in cold context:" << std::endl;
        std::cout << "    Avg: " << std::fixed << std::setprecision(3) << r_swiglu.avg_ms << " ms"
                  << "  Min: " << r_swiglu.min_ms << " ms"
                  << "  Max: " << r_swiglu.max_ms << " ms"
                  << "  StdDev: " << r_swiglu.stddev_ms << " ms" << std::endl;
        std::cout << "    Bandwidth: " << std::setprecision(1) << r_swiglu.bw_gbs << " GB/s" << std::endl;

        // Compute total "other" kernel time
        double other_total = 0.0;
        for (const auto& ev : all_events) {
            uint64_t s = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
            uint64_t e = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
            other_total += static_cast<double>(e - s) * 1e-6;
        }
        std::cout << "    Other kernels total: " << std::setprecision(2) << other_total << " ms" << std::endl;
    }

    // =========================================================================
    // TEST 5: Submission overhead measurement — time between end of one
    // kernel and start of the next
    // =========================================================================
    std::cout << "\n--- TEST 5: Inter-kernel dispatch gaps (submission overhead) ---" << std::endl;
    {
        // Fire a mixed sequence and measure the gap between consecutive events
        constexpr int SEQ_LEN = 32;
        std::vector<sycl::event> seq_events;
        seq_events.reserve(SEQ_LEN);
        q.wait();

        for (int i = 0; i < SEQ_LEN; ++i) {
            if (i % 4 == 0) {
                seq_events.push_back(
                    ops::rmsnorm(q, d_normed, d_X, d_norm_w, 1, K));
            } else if (i % 4 == 1) {
                seq_events.push_back(
                    ops::linear_int4(q, d_Y_tiny, d_normed, d_W_tiny, d_S_tiny,
                                     nullptr, M, TINY_N, K, GROUP_SIZE));
            } else if (i % 4 == 2) {
                seq_events.push_back(
                    ops::mlp_gate_up_swiglu_int4(q, d_Y, d_normed, d_Wg, d_Sg,
                                                 d_Wu, d_Su, M, N, K, GROUP_SIZE));
            } else {
                seq_events.push_back(
                    ops::linear_int4(q, d_Y, d_normed, d_Wg, d_Sg,
                                     nullptr, M, N, K, GROUP_SIZE));
            }
        }
        q.wait();

        std::vector<double> gaps_us;
        for (size_t i = 1; i < seq_events.size(); ++i) {
            uint64_t prev_end = seq_events[i-1].get_profiling_info<sycl::info::event_profiling::command_end>();
            uint64_t curr_start = seq_events[i].get_profiling_info<sycl::info::event_profiling::command_start>();
            double gap_us = static_cast<double>(curr_start - prev_end) * 1e-3; // ns -> µs
            gaps_us.push_back(gap_us);
        }

        double avg_gap = std::accumulate(gaps_us.begin(), gaps_us.end(), 0.0) / static_cast<double>(gaps_us.size());
        double max_gap = *std::max_element(gaps_us.begin(), gaps_us.end());
        double min_gap = *std::min_element(gaps_us.begin(), gaps_us.end());

        std::cout << "  Avg inter-kernel gap: " << std::fixed << std::setprecision(2) << avg_gap << " µs" << std::endl;
        std::cout << "  Min: " << min_gap << " µs  Max: " << max_gap << " µs" << std::endl;
        std::cout << "  Total gap for 960 ops: " << std::setprecision(2) << (avg_gap * 960.0 / 1000.0) << " ms" << std::endl;
    }

    // =========================================================================
    // TEST 6: Sub-group occupancy analysis
    // =========================================================================
    std::cout << "\n--- TEST 6: Occupancy analysis (sub-group count vs HW threads) ---" << std::endl;
    {
        // mlp_gate_up_swiglu_int4 uses 1 sub-group per output row (no ROWS_PER_SG pairing)
        int64_t swiglu_sgs = M * N;  // = 17408 sub-groups
        // linear_int4 uses ROWS_PER_SG=2
        int64_t linear_sgs = M * ((N + 1) / 2);  // = 8704 sub-groups
        // Tiny-N linear
        int64_t tiny_sgs = M * ((TINY_N + 1) / 2);  // = 24 sub-groups

        auto print_occ = [](const char* name, int64_t sgs) {
            double occ_pct = static_cast<double>(sgs) / B60_HW_THREADS * 100.0;
            std::cout << "  " << std::left << std::setw(40) << name
                      << std::right << std::setw(8) << sgs << " sub-groups"
                      << std::setw(10) << std::fixed << std::setprecision(1) << occ_pct << "% occ"
                      << (occ_pct >= 100.0 ? "  [FULL]" : (occ_pct < 10.0 ? "  [STARVED]" : ""))
                      << std::endl;
        };

        print_occ("mlp_gate_up_swiglu_int4 (N=17408)", swiglu_sgs);
        print_occ("linear_int4 ROWS_PER_SG=2 (N=17408)", linear_sgs);
        print_occ("linear_int4 ROWS_PER_SG=2 (N=48)", tiny_sgs);
        print_occ("linear_int4 ROWS_PER_SG=2 (N=1024)", M * ((1024 + 1) / 2));
        print_occ("linear_int4 ROWS_PER_SG=2 (N=5120)", M * ((5120 + 1) / 2));

        // Key insight: swiglu launches 2x more sub-groups than linear at the
        // same N because it doesn't use ROWS_PER_SG=2 pairing. BUT each
        // sub-group holds double the registers (gate + up accumulators).
        std::cout << "\n  Note: SwiGLU has 2x sub-groups but 2x register pressure per SG." << std::endl;
        std::cout << "  If register file limits co-resident SGs per Xe-core below the" << std::endl;
        std::cout << "  sub-group formula's prediction, effective occupancy is lower." << std::endl;
    }

    // =========================================================================
    // Summary
    // =========================================================================
    std::cout << "\n==================================================================" << std::endl;
    std::cout << " INTERPRETATION GUIDE:" << std::endl;
    std::cout << " - If TEST 1 (warm SwiGLU) matches TEST 2 (warm linear):" << std::endl;
    std::cout << "     Register pressure is NOT the problem." << std::endl;
    std::cout << "     The gap is in cold-start / kernel dispatch." << std::endl;
    std::cout << " - If TEST 1 >> TEST 3 (2x sequential linear):" << std::endl;
    std::cout << "     Register pressure IS the dominant factor." << std::endl;
    std::cout << " - If TEST 4 (cold SwiGLU) >> TEST 1 (warm SwiGLU):" << std::endl;
    std::cout << "     Cold cache / clock ramp IS a significant contributor." << std::endl;
    std::cout << " - TEST 5 shows absolute dispatch overhead (should be ~1.5 µs/op)." << std::endl;
    std::cout << "==================================================================" << std::endl;

    // Cleanup
    sycl::free(d_X, q);
    sycl::free(d_Y, q);
    sycl::free(d_Wg, q);
    sycl::free(d_Wu, q);
    sycl::free(d_Sg, q);
    sycl::free(d_Su, q);
    sycl::free(d_norm_w, q);
    sycl::free(d_normed, q);
    sycl::free(d_X2, q);
    sycl::free(d_Y_tiny, q);
    sycl::free(d_W_tiny, q);
    sycl::free(d_S_tiny, q);

    return 0;
}
