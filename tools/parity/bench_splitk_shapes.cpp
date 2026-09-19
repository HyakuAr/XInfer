// Benchmark: linear_int4 split-K path vs standard path at small-N shapes.
// Measures the bandwidth improvement from split-K on the N=48 shapes
// (in_proj_b, in_proj_a) that triggered only 1.9% occupancy without split-K.
//
// Citing vendor documentation per AGENTS.md §5:
// - docs/vendor/thread-mapping-occupancy.md (lines 22-24):
//   64 HW threads per Xe-Core, 1280 total. Sub-group = one HW thread.
// - docs/vendor/xe-gpu-architecture.md (lines 22-38):
//   B60: 20 Xe-cores, 8 Vector Engines per core, 8 HW threads per VE.

#include "core/device.h"
#include "ops/linear.h"

#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>

using namespace xinfer;

constexpr double B60_PEAK_BW_GBS = 456.0;
constexpr int    B60_HW_THREADS  = 1280;

struct ShapeSpec {
    const char* name;
    int64_t N;
    int64_t K;
};

int main() {
    try {
        std::cout << "==================================================================" << std::endl;
        std::cout << " Substep 2 Verification: Split-K vs Standard Path Bandwidth" << std::endl;
#ifdef XINFER_BUILD_CONFIG
        std::cout << " Build Config: " << XINFER_BUILD_CONFIG << std::endl;
#endif
        std::cout << "==================================================================" << std::endl;

        auto ctx = core::DeviceContext::create(true);
        sycl::queue& q = ctx->queue();

        std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    // All shapes from the model's linear projections
    const std::vector<ShapeSpec> shapes = {
        {"in_proj_b (N=48)",        48,    5120},
        {"in_proj_a (N=48)",        48,    5120},
        {"Full-Attn K (N=1024)",    1024,  5120},
        {"Full-Attn V (N=1024)",    1024,  5120},
        {"Linear-Attn QKV (N=640)", 640,   5120},
        {"Linear-Attn Z (N=5120)",  5120,  5120},
        {"MLP Gate (N=17408)",      17408, 5120},
        {"MLP Down (N=5120)",       5120,  17408},
        {"LM Head (N=248320)",      248320, 5120},
    };

    constexpr int ITERS = 20;
    constexpr int64_t M = 1;
    constexpr int GROUP_SIZE = 128;

    // Find max dimensions and buffer requirements across actual shapes
    int64_t max_N = 0, max_K = 0;
    size_t max_w_bytes = 0;
    size_t max_s_count = 0;
    for (const auto& s : shapes) {
        if (s.N > max_N) max_N = s.N;
        if (s.K > max_K) max_K = s.K;
        size_t wb = static_cast<size_t>(s.N) * (s.K / 2);
        size_t sc = static_cast<size_t>(s.N) * (s.K / GROUP_SIZE);
        if (wb > max_w_bytes) max_w_bytes = wb;
        if (sc > max_s_count) max_s_count = sc;
    }

    // Allocate buffers
    float*      d_X = sycl::malloc_device<float>(max_K, q);
    float*      d_Y = sycl::malloc_device<float>(max_N, q);
    uint8_t*    d_W = sycl::malloc_device<uint8_t>(max_w_bytes, q);
    sycl::half* d_S = sycl::malloc_device<sycl::half>(max_s_count, q);

    q.fill(d_X, 1.0f, max_K);
    q.fill(d_W, static_cast<uint8_t>(0x33), max_w_bytes);
    q.fill(d_S, sycl::half{0.1f}, max_s_count);
    q.wait();

    // Warmup all shapes
    for (const auto& s : shapes) {
        ops::linear_int4(q, d_Y, d_X, d_W, d_S, nullptr, M, s.N, s.K, GROUP_SIZE);
    }
    q.wait();

    // Benchmark each shape
    std::cout << "\n" << std::left << std::setw(30) << "Shape"
              << std::right << std::setw(8) << "SGs"
              << std::setw(8) << "Occ%"
              << std::setw(10) << "Path"
              << std::setw(12) << "Avg (ms)"
              << std::setw(12) << "Min (ms)"
              << std::setw(13) << "BW (GB/s)"
              << std::setw(10) << "% Peak" << std::endl;
    std::cout << std::string(103, '-') << std::endl;

    for (const auto& s : shapes) {
        int64_t num_groups = s.K / GROUP_SIZE;
        int64_t sgs = (s.N + 1) / 2;
        double occ = static_cast<double>(sgs) / B60_HW_THREADS * 100.0;
        bool is_splitk = (sgs < B60_HW_THREADS / 10) && (num_groups >= 4);
        const char* path = is_splitk ? "SPLIT-K" : "STANDARD";

        size_t w_bytes = static_cast<size_t>(s.N) * (s.K / 2);
        size_t s_bytes = static_cast<size_t>(s.N) * num_groups * sizeof(sycl::half);
        size_t total_bytes = w_bytes + s_bytes + static_cast<size_t>(s.K) * sizeof(float)
                           + static_cast<size_t>(s.N) * sizeof(float);

        std::vector<sycl::event> events;
        events.reserve(ITERS);
        q.wait();

        for (int i = 0; i < ITERS; ++i) {
            events.push_back(
                ops::linear_int4(q, d_Y, d_X, d_W, d_S, nullptr, M, s.N, s.K, GROUP_SIZE)
            );
        }
        q.wait();

        std::vector<double> times_ms;
        for (const auto& ev : events) {
            uint64_t t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
            uint64_t t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
            times_ms.push_back(static_cast<double>(t1 - t0) * 1e-6);
        }

        double avg = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();
        double min_t = *std::min_element(times_ms.begin(), times_ms.end());
        double bw = (avg > 0.0) ? (static_cast<double>(total_bytes) / 1e9) / (avg / 1000.0) : 0.0;
        double pct_peak = bw / B60_PEAK_BW_GBS * 100.0;

        std::cout << std::left << std::setw(30) << s.name
                  << std::right << std::setw(8) << sgs
                  << std::setw(7) << std::fixed << std::setprecision(1) << occ << "%"
                  << std::setw(10) << path
                  << std::setw(12) << std::setprecision(3) << avg
                  << std::setw(12) << min_t
                  << std::setw(10) << std::setprecision(1) << bw << " GB/s"
                  << std::setw(8) << std::setprecision(1) << pct_peak << "%" << std::endl;
    }

    std::cout << std::string(103, '=') << std::endl;

    sycl::free(d_X, q);
    sycl::free(d_Y, q);
    sycl::free(d_W, q);
    sycl::free(d_S, q);

    return 0;
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL Exception: " << e.what() << " (code: " << e.code() << ")" << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
        return 1;
    }
}
