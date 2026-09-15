#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    try {
        // Enumerate all platforms and devices
        std::cout << "=== Available SYCL Platforms and Devices ===" << std::endl;
        auto platforms = sycl::platform::get_platforms();
        for (const auto& plat : platforms) {
            std::cout << "Platform: " << plat.get_info<sycl::info::platform::name>()
                      << " (" << plat.get_info<sycl::info::platform::version>() << ")" << std::endl;
            for (const auto& dev : plat.get_devices()) {
                std::cout << "  Device: " << dev.get_info<sycl::info::device::name>()
                          << " [Type: " << (dev.is_gpu() ? "GPU" : (dev.is_cpu() ? "CPU" : "Other")) << "]"
                          << std::endl;
            }
        }
        std::cout << std::endl;

        // Select GPU device
        sycl::device dev{sycl::gpu_selector_v};
        std::cout << "=== Selected Target Device ===" << std::endl;
        std::cout << "Device Name:    " << dev.get_info<sycl::info::device::name>() << std::endl;
        std::cout << "Vendor:         " << dev.get_info<sycl::info::device::vendor>() << std::endl;
        std::cout << "Driver Version: " << dev.get_info<sycl::info::device::driver_version>() << std::endl;
        std::cout << "Backend:        "
                  << (dev.get_backend() == sycl::backend::ext_oneapi_level_zero ? "Level Zero (ext_oneapi_level_zero)" : "Other")
                  << std::endl;

        // Run a tiny test kernel on the GPU
        sycl::queue q{dev};
        int result = 0;
        {
            sycl::buffer<int, 1> buf(&result, sycl::range<1>(1));
            q.submit([&](sycl::handler& cgh) {
                auto acc = buf.get_access<sycl::access::mode::write>(cgh);
                cgh.single_task([=]() {
                    acc[0] = 42;
                });
            }).wait();
        }

        std::cout << "\n=== Kernel Test ===" << std::endl;
        std::cout << "Kernel Output: " << result << " (Expected: 42)" << std::endl;

        if (result == 42) {
            std::cout << "STATUS: SYCL GPU Test PASSED on " << dev.get_info<sycl::info::device::name>() << "!" << std::endl;
            return 0;
        } else {
            std::cerr << "STATUS: SYCL GPU Test FAILED (wrong result)" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
        return 1;
    }
}
