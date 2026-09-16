#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <iostream>
#include <vector>
#include <chrono>

namespace syclex = sycl::ext::oneapi::experimental;

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    auto dev = q.get_device();
    std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << std::endl;

    constexpr size_t N = 1024;
    float* d_buf = sycl::malloc_device<float>(N, q);
    q.fill(d_buf, 0.0f, N).wait();

    try {
        // 1. Create modifiable command graph
        syclex::command_graph<syclex::graph_state::modifiable> graph(q.get_context(), dev);

        // 2. Begin recording on the queue
        graph.begin_recording(q);

        // Record a series of operations in order
        for (int step = 0; step < 50; ++step) {
            q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                d_buf[idx[0]] += 1.0f;
            });
        }

        // 3. End recording
        graph.end_recording(q);

        // 4. Finalize to executable graph
        auto exec_graph = graph.finalize();
        std::cout << "Successfully captured command graph with 50 nodes!" << std::endl;

        // Replay 100 times
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            q.ext_oneapi_graph(exec_graph);
        }
        q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        double graph_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::vector<float> h_buf(N);
        q.memcpy(h_buf.data(), d_buf, N * sizeof(float)).wait();
        std::cout << "Result check: d_buf[0] = " << h_buf[0] << " (expected 5000)" << std::endl;

        std::cout << "Time for 100 replays: " << graph_ms << " ms" << std::endl;
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
    }

    sycl::free(d_buf, q);
    return 0;
}
