#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    try {
        sycl::queue q(sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
        std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

        constexpr size_t N = 1024 * 1024;
        float* d_buf = sycl::malloc_device<float>(N, q);

        sycl::event e = q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
            d_buf[idx] = idx[0] * 2.0f;
        });
        e.wait();

        auto start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto end   = e.get_profiling_info<sycl::info::event_profiling::command_end>();
        double duration_ms = static_cast<double>(end - start) * 1e-6;

        std::cout << "Kernel start: " << start << " ns, end: " << end << " ns" << std::endl;
        std::cout << "Kernel duration: " << duration_ms << " ms" << std::endl;

        sycl::free(d_buf, q);
        return 0;
    } catch (const sycl::exception& ex) {
        std::cerr << "SYCL Exception: " << ex.what() << std::endl;
        return 1;
    }
}
