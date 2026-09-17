#pragma once

#include "device.h"
#include "tensor.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <vector>

namespace xinfer::core {

class DeviceArena {
public:
    // Default 64-byte alignment matches Xe2 cache-line size (docs/vendor/xe-gpu-architecture.md)
    static constexpr size_t DEFAULT_ALIGNMENT = 64;

    DeviceArena(std::shared_ptr<DeviceContext> ctx, size_t capacity_bytes, size_t default_alignment = DEFAULT_ALIGNMENT);
    ~DeviceArena();

    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;

    DeviceArena(DeviceArena&& other) noexcept;
    DeviceArena& operator=(DeviceArena&& other) noexcept;

    // Bump allocate raw device memory with required alignment
    void* allocate(size_t bytes, size_t alignment = 0);

    // Allocate a TensorView backed by the arena buffer
    TensorView allocate_tensor(TensorShape shape, DataType dtype, size_t alignment = 0);

    // Allocate raw device memory outside the bump arena that persists across reset()
    // calls and is freed when the DeviceArena is destroyed.
    void* allocate_persistent(size_t bytes, size_t alignment = 0);

    // Get or allocate a dedicated persistent buffer (e.g. for fallback logits) outside the bump arena.
    // Preserves address stability across decode steps and survives reset().
    void* persistent_buffer(size_t bytes = 0, size_t alignment = 0);
    const void* persistent_buffer() const noexcept { return persistent_buffer_; }

    // Reset the bump pointer to 0 for reuse in the next decode step without deallocating USM
    void reset() noexcept;

    size_t capacity() const noexcept { return capacity_; }
    size_t allocated_bytes() const noexcept { return offset_; }
    size_t remaining_bytes() const noexcept { return (offset_ < capacity_) ? (capacity_ - offset_) : 0; }
    size_t peak_allocated_bytes() const noexcept { return peak_offset_; }

    void* base_ptr() noexcept { return base_ptr_; }
    const void* base_ptr() const noexcept { return base_ptr_; }

private:
    std::shared_ptr<DeviceContext> ctx_;
    void*  base_ptr_{nullptr};
    size_t capacity_{0};
    size_t offset_{0};
    size_t peak_offset_{0};
    size_t default_alignment_{DEFAULT_ALIGNMENT};
    std::vector<void*> persistent_allocations_;
    void* persistent_buffer_{nullptr};
};

} // namespace xinfer::core
