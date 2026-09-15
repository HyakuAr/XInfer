#include "device.h"
#include <iostream>
#include <stdexcept>

namespace xinfer::core {

std::shared_ptr<DeviceContext> DeviceContext::create(bool prefer_b60) {
    auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.empty()) {
        throw std::runtime_error("No SYCL GPU devices found on the system.");
    }

    sycl::device selected_device = gpus[0];
    bool found = false;

    if (prefer_b60) {
        for (const auto& dev : gpus) {
            std::string name = dev.get_info<sycl::info::device::name>();
            auto backend = dev.get_backend();
            if (name.find("B60") != std::string::npos && backend == sycl::backend::ext_oneapi_level_zero) {
                selected_device = dev;
                found = true;
                break;
            }
        }
    }

    if (!found) {
        for (const auto& dev : gpus) {
            if (dev.get_backend() == sycl::backend::ext_oneapi_level_zero) {
                selected_device = dev;
                found = true;
                break;
            }
        }
    }

    return std::make_shared<DeviceContext>(selected_device);
}

DeviceContext::DeviceContext(const sycl::device& dev)
    : queue_(dev, sycl::property_list{
          sycl::property::queue::in_order{},
          sycl::property::queue::enable_profiling{}
      }) {
    query_arch_info();
}

DeviceContext::~DeviceContext() {
    synchronize();
}

void DeviceContext::query_arch_info() {
    auto dev = device();
    arch_info_.device_name = dev.get_info<sycl::info::device::name>();
    arch_info_.global_mem_bytes = dev.get_info<sycl::info::device::global_mem_size>();
    arch_info_.slm_bytes = static_cast<uint32_t>(dev.get_info<sycl::info::device::local_mem_size>());
    arch_info_.max_work_group_size = dev.get_info<sycl::info::device::max_work_group_size>();
    arch_info_.sub_group_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();

    // Query Xe2 hardware architecture metrics as specified in docs/vendor/xe-gpu-architecture.md
    try {
        uint32_t slices = dev.get_info<sycl::ext::intel::info::device::gpu_slices>();
        uint32_t subslices = dev.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
        uint32_t eus = dev.get_info<sycl::ext::intel::info::device::gpu_eu_count_per_subslice>();
        uint32_t threads = dev.get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>();

        arch_info_.xe_core_count = slices * subslices;
        arch_info_.vector_engine_count = arch_info_.xe_core_count * eus;
        arch_info_.hw_threads_per_ve = threads;
        arch_info_.total_hw_threads = arch_info_.vector_engine_count * threads;
    } catch (...) {
        // Fallback default for Xe2-HPG Battlemage B60 (20 Xe-Cores, 8 VE/core, 8 hw threads/VE)
        arch_info_.xe_core_count = 20;
        arch_info_.vector_engine_count = 160;
        arch_info_.hw_threads_per_ve = 8;
        arch_info_.total_hw_threads = 1280;
    }
}

bool DeviceContext::has_native_level_zero() const noexcept {
    return device().get_backend() == sycl::backend::ext_oneapi_level_zero;
}

void* DeviceContext::native_l0_device() const {
    if (!has_native_level_zero()) return nullptr;
    return reinterpret_cast<void*>(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device()));
}

void* DeviceContext::native_l0_context() const {
    if (!has_native_level_zero()) return nullptr;
    return reinterpret_cast<void*>(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(context()));
}

void* DeviceContext::native_l0_queue() const {
    if (!has_native_level_zero()) return nullptr;
    auto native_q = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue_);
    if (std::holds_alternative<ze_command_queue_handle_t>(native_q)) {
        return reinterpret_cast<void*>(std::get<ze_command_queue_handle_t>(native_q));
    } else if (std::holds_alternative<ze_command_list_handle_t>(native_q)) {
        return reinterpret_cast<void*>(std::get<ze_command_list_handle_t>(native_q));
    }
    return nullptr;
}

void* DeviceContext::allocate_device(size_t bytes, size_t alignment) {
    if (bytes == 0) return nullptr;
    void* ptr = sycl::aligned_alloc_device(alignment, bytes, queue_);
    if (!ptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

void DeviceContext::free_device(void* ptr) {
    if (ptr) {
        sycl::free(ptr, queue_);
    }
}

void* DeviceContext::allocate_host(size_t bytes, size_t alignment) {
    if (bytes == 0) return nullptr;
    void* ptr = sycl::aligned_alloc_host(alignment, bytes, queue_);
    if (!ptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

void DeviceContext::free_host(void* ptr) {
    if (ptr) {
        sycl::free(ptr, queue_);
    }
}

void DeviceContext::copy_host_to_device(void* dst_device, const void* src_host, size_t bytes, bool blocking) {
    if (bytes == 0 || !dst_device || !src_host) return;
    auto event = queue_.memcpy(dst_device, src_host, bytes);
    if (blocking) {
        event.wait();
    }
}

void DeviceContext::copy_device_to_host(void* dst_host, const void* src_device, size_t bytes, bool blocking) {
    if (bytes == 0 || !dst_host || !src_device) return;
    auto event = queue_.memcpy(dst_host, src_device, bytes);
    if (blocking) {
        event.wait();
    }
}

void DeviceContext::copy_device_to_device(void* dst_device, const void* src_device, size_t bytes, bool blocking) {
    if (bytes == 0 || !dst_device || !src_device) return;
    auto event = queue_.memcpy(dst_device, src_device, bytes);
    if (blocking) {
        event.wait();
    }
}

void DeviceContext::fill_device(void* dst_device, uint8_t value, size_t bytes, bool blocking) {
    if (bytes == 0 || !dst_device) return;
    auto event = queue_.memset(dst_device, value, bytes);
    if (blocking) {
        event.wait();
    }
}

void DeviceContext::synchronize() {
    queue_.wait();
}

} // namespace xinfer::core
