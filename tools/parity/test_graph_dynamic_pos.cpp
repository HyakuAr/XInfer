#include "core/device.h"
#include "core/arena.h"
#include "core/kv_cache.h"
#include "targets/qwen3_8_27b/weights.h"
#include "targets/qwen3_8/forward.h"
#include "ops/linear.h"
#include "ops/attention.h"
#include "ops/rmsnorm.h"
#include "ops/rope.h"
#include "ops/elementwise.h"
#include "ops/sampling.h"
#include "targets/qwen3_8/linear_attn.h"
#include "artifact/reader.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <iostream>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;
using namespace xinfer;

int main() {
    std::cout << "Testing Dynamic Position Graph Replay on Intel Arc Pro B60..." << std::endl;
    auto ctx = core::DeviceContext::create(true);
    sycl::queue& q = ctx->queue();
    auto dev = q.get_device();

    // Verify small micro-graph with dynamic pointer
    int64_t* d_pos = sycl::malloc_device<int64_t>(1, q);
    float* d_arr = sycl::malloc_device<float>(10, q);
    q.fill(d_arr, 0.0f, 10).wait();

    int64_t host_pos = 0;
    q.memcpy(d_pos, &host_pos, sizeof(int64_t)).wait();

    // Capture graph where kernel reads *d_pos
    syclex::command_graph<syclex::graph_state::modifiable> graph(q.get_context(), dev);
    graph.begin_recording(q);

    q.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
        int64_t p = *d_pos;
        if (p >= 0 && p < 10) {
            d_arr[p] += 10.0f * static_cast<float>(p + 1);
        }
    });

    graph.end_recording(q);
    auto exec_graph = graph.finalize();

    // Replay across different positions
    for (int p = 0; p < 5; ++p) {
        int64_t new_pos = p;
        q.memcpy(d_pos, &new_pos, sizeof(int64_t)).wait();
        q.ext_oneapi_graph(exec_graph);
        q.wait();
    }

    std::vector<float> h_arr(10);
    q.memcpy(h_arr.data(), d_arr, 10 * sizeof(float)).wait();

    std::cout << "Dynamic graph replay results:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        float expected = 10.0f * static_cast<float>(i + 1);
        std::cout << "  d_arr[" << i << "] = " << h_arr[i] << " (expected " << expected << ")" << std::endl;
        if (std::abs(h_arr[i] - expected) > 1e-3f) {
            std::cerr << "FAIL: dynamic pos not updated!" << std::endl;
            return 1;
        }
    }

    std::cout << "SUCCESS! Dynamic position via device memory pointer verified in graph replay." << std::endl;
    sycl::free(d_pos, q);
    sycl::free(d_arr, q);
    return 0;
}
