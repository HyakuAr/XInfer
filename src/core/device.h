#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace xinfer::core {

// Hardware architectural specifications queried directly at runtime
// Grounding source: docs/vendor/xe-gpu-architecture.md
struct DeviceArchInfo {
    std::string device_name;
    uint32_t    xe_core_count{0};        // gpu_slices * gpu_subslices_per_slice
    uint32_t    vector_engine_count{0};  // xe_core_count * gpu_eu_count_per_subslice
    uint32_t    hw_threads_per_ve{0};    // gpu_hw_threads_per_eu
    uint32_t    total_hw_threads{0};     // vector_engine_count * hw_threads_per_ve
    uint64_t    global_mem_bytes{0};     // Global VRAM size
    uint32_t    slm_bytes{0};            // Shared Local Memory per work-group
    size_t      max_work_group_size{0};  // Max work-group size
    std::vector<size_t> sub_group_sizes; // Supported sub-group sizes (16, 32 on Xe2)
};

class DeviceContext {
public:
    // Create device context selecting Intel Arc Pro B60 or first available Level Zero GPU
    static std::shared_ptr<DeviceContext> create(bool prefer_b60 = true);

    explicit DeviceContext(const sycl::device& dev);
    ~DeviceContext();

    DeviceContext(const DeviceContext&) = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;

    sycl::queue& queue() noexcept { return queue_; }
    const sycl::queue& queue() const noexcept { return queue_; }

    sycl::device device() const noexcept { return queue_.get_device(); }
    sycl::context context() const noexcept { return queue_.get_context(); }

    const DeviceArchInfo& arch_info() const noexcept { return arch_info_; }

    // Native Level Zero handles (available when backend is ext_oneapi_level_zero)
    bool has_native_level_zero() const noexcept;
    void* native_l0_device() const;
    void* native_l0_context() const;
    void* native_l0_queue() const;

    // USM device allocation primitives (64-byte aligned for Xe2 cache line DMA)
    void* allocate_device(size_t bytes, size_t alignment = 64);
    void  free_device(void* ptr);

    // USM host allocation primitives
    void* allocate_host(size_t bytes, size_t alignment = 64);
    void  free_host(void* ptr);

    // Synchronous and asynchronous copy operations
    void copy_host_to_device(void* dst_device, const void* src_host, size_t bytes, bool blocking = true);
    void copy_device_to_host(void* dst_host, const void* src_device, size_t bytes, bool blocking = true);
    void copy_device_to_device(void* dst_device, const void* src_device, size_t bytes, bool blocking = true);
    void fill_device(void* dst_device, uint8_t value, size_t bytes, bool blocking = true);

    void synchronize();

private:
    void query_arch_info();

    sycl::queue queue_;
    DeviceArchInfo arch_info_;
};

} // namespace xinfer::core
