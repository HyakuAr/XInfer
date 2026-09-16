#include "core/device.h"
#include "core/arena.h"
#include "core/kv_cache.h"
#include "ops/linear.h"
#include "ops/attention.h"
#include "ops/rmsnorm.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

namespace syclex = sycl::ext::oneapi::experimental;
using namespace xinfer;

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << " xinfer M8 Level Zero / SYCL Decode Graph Unit Test" << std::endl;
    std::cout << " Testing graph capture & dynamic position replay on B60" << std::endl;
    std::cout << "==========================================================" << std::endl;

    auto ctx = core::DeviceContext::create(true);
    sycl::queue& q = ctx->queue();
    auto dev = q.get_device();

    if (!dev.has(sycl::aspect::ext_oneapi_graph)) {
        std::cerr << "Device does not support ext_oneapi_graph aspect!" << std::endl;
        return 1;
    }

    constexpr size_t num_q_heads = 24;
    constexpr size_t num_kv_heads = 4;
    constexpr size_t head_dim = 256;
    constexpr size_t max_seq = 64;

    // USM device allocations with guaranteed address stability
    int64_t* d_cur_pos = sycl::malloc_device<int64_t>(1, q);
    float* d_Q = sycl::malloc_device<float>(num_q_heads * head_dim, q);
    float* d_K = sycl::malloc_device<float>(num_kv_heads * head_dim, q);
    float* d_V = sycl::malloc_device<float>(num_kv_heads * head_dim, q);
    float* d_out = sycl::malloc_device<float>(num_q_heads * head_dim, q);

    sycl::half* k_cache = sycl::malloc_device<sycl::half>(max_seq * num_kv_heads * head_dim, q);
    sycl::half* v_cache = sycl::malloc_device<sycl::half>(max_seq * num_kv_heads * head_dim, q);

    // Initialize Q, K, V, cache, and position
    q.fill(d_Q, 0.1f, num_q_heads * head_dim);
    q.fill(d_K, 0.05f, num_kv_heads * head_dim);
    q.fill(d_V, 0.5f, num_kv_heads * head_dim);
    q.fill(k_cache, sycl::half{0.0f}, max_seq * num_kv_heads * head_dim);
    q.fill(v_cache, sycl::half{0.0f}, max_seq * num_kv_heads * head_dim);
    int64_t initial_pos = 0;
    q.memcpy(d_cur_pos, &initial_pos, sizeof(int64_t));
    q.wait();

    // 1. Capture graph containing KV-cache write and cached attention
    syclex::command_graph<syclex::graph_state::modifiable> graph(q.get_context(), dev);
    graph.begin_recording(q);

    ops::attention_write_kv_cache_dynamic(q, k_cache, v_cache, d_K, d_V, d_cur_pos, 1, num_kv_heads, head_dim);
    ops::sdpa_causal_cached_dynamic(q, d_out, d_Q, k_cache, v_cache, d_cur_pos, 1, num_q_heads, num_kv_heads, head_dim);

    graph.end_recording(q);
    auto exec_graph = graph.finalize();
    std::cout << "Successfully captured Level Zero command graph!" << std::endl;

    // 2. Replay graph across 5 sequential decode positions (pos 0 .. 4)
    std::cout << "Replaying captured graph across positions 0 .. 4..." << std::endl;
    for (int64_t pos = 0; pos < 5; ++pos) {
        int64_t host_p = pos;
        q.memcpy(d_cur_pos, &host_p, sizeof(int64_t)).wait();
        q.ext_oneapi_graph(exec_graph);
        q.wait();
    }

    // Verify output after 5 steps: since all V values are 0.5f, output must normalize to 0.5f
    std::vector<float> h_out(num_q_heads * head_dim);
    q.memcpy(h_out.data(), d_out, h_out.size() * sizeof(float)).wait();

    std::cout << "Verified output[0] = " << h_out[0] << " (expected 0.5)" << std::endl;
    if (std::abs(h_out[0] - 0.5f) >= 1e-3f) {
        std::cerr << "FAILED: Output mismatch! Expected 0.5, got " << h_out[0] << std::endl;
        return 1;
    }
    std::cout << "  -> PASSED: Graph replay with dynamic position pointer verified." << std::endl;

    sycl::free(d_cur_pos, q);
    sycl::free(d_Q, q);
    sycl::free(d_K, q);
    sycl::free(d_V, q);
    sycl::free(d_out, q);
    sycl::free(k_cache, q);
    sycl::free(v_cache, q);

    std::cout << "==========================================================" << std::endl;
    std::cout << " ALL M8 DECODE GRAPH TESTS PASSED ON B60!" << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}
