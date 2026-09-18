#include "arena.h"
#include <cassert>
#include <new>
#include <utility>

namespace xinfer::core {

DeviceArena::DeviceArena(std::shared_ptr<DeviceContext> ctx, size_t capacity_bytes, size_t default_alignment)
    : ctx_(std::move(ctx)), capacity_(capacity_bytes), default_alignment_(default_alignment) {
    if (!ctx_) {
        throw std::invalid_argument("DeviceArena requires a valid DeviceContext");
    }
    if (capacity_ > 0) {
        base_ptr_ = ctx_->allocate_device(capacity_, default_alignment_);
    }
}

DeviceArena::~DeviceArena() {
    if (ctx_) {
        if (base_ptr_) {
            ctx_->free_device(base_ptr_);
            base_ptr_ = nullptr;
        }
        for (void* ptr : persistent_allocations_) {
            if (ptr) {
                ctx_->free_device(ptr);
            }
        }
        persistent_allocations_.clear();
        persistent_buffer_ = nullptr;
    }
}

DeviceArena::DeviceArena(DeviceArena&& other) noexcept
    : ctx_(std::move(other.ctx_)),
      base_ptr_(other.base_ptr_),
      capacity_(other.capacity_),
      offset_(other.offset_),
      peak_offset_(other.peak_offset_),
      default_alignment_(other.default_alignment_),
      persistent_allocations_(std::move(other.persistent_allocations_)),
      persistent_buffer_(other.persistent_buffer_) {
    other.base_ptr_ = nullptr;
    other.capacity_ = 0;
    other.offset_ = 0;
    other.peak_offset_ = 0;
    other.persistent_buffer_ = nullptr;
}

DeviceArena& DeviceArena::operator=(DeviceArena&& other) noexcept {
    if (this != &other) {
        if (ctx_) {
            if (base_ptr_) {
                ctx_->free_device(base_ptr_);
            }
            for (void* ptr : persistent_allocations_) {
                if (ptr) {
                    ctx_->free_device(ptr);
                }
            }
        }
        ctx_ = std::move(other.ctx_);
        base_ptr_ = other.base_ptr_;
        capacity_ = other.capacity_;
        offset_ = other.offset_;
        peak_offset_ = other.peak_offset_;
        default_alignment_ = other.default_alignment_;
        persistent_allocations_ = std::move(other.persistent_allocations_);
        persistent_buffer_ = other.persistent_buffer_;

        other.base_ptr_ = nullptr;
        other.capacity_ = 0;
        other.offset_ = 0;
        other.peak_offset_ = 0;
        other.persistent_buffer_ = nullptr;
    }
    return *this;
}

void* DeviceArena::allocate(size_t bytes, size_t alignment) {
    if (bytes == 0) return nullptr;
    if (!base_ptr_) throw std::bad_alloc();

    size_t align = (alignment > 0) ? alignment : default_alignment_;
    uintptr_t current_addr = reinterpret_cast<uintptr_t>(base_ptr_) + offset_;
    uintptr_t aligned_addr = (current_addr + align - 1) & ~(align - 1);
    size_t new_offset = (aligned_addr - reinterpret_cast<uintptr_t>(base_ptr_)) + bytes;

    if (new_offset > capacity_) {
        throw std::bad_alloc();
    }

    offset_ = new_offset;
    peak_offset_ = std::max(peak_offset_, offset_);
    return reinterpret_cast<void*>(aligned_addr);
}

TensorView DeviceArena::allocate_tensor(TensorShape shape, DataType dtype, size_t alignment) {
    size_t bytes = compute_tensor_bytes(shape.numel(), dtype);
    void* ptr = allocate(bytes, alignment);
    return TensorView(ptr, std::move(shape), dtype);
}

void* DeviceArena::allocate_persistent(size_t bytes, size_t alignment) {
    if (bytes == 0) return nullptr;
    if (!ctx_) throw std::invalid_argument("DeviceArena requires a valid DeviceContext");

    size_t align = (alignment > 0) ? alignment : default_alignment_;
    void* ptr = ctx_->allocate_device(bytes, align);
    if (!ptr) {
        throw std::bad_alloc();
    }
    persistent_allocations_.push_back(ptr);
    return ptr;
}

void* DeviceArena::persistent_buffer(size_t bytes, size_t alignment) {
    if (!persistent_buffer_) {
        assert(bytes > 0 && "DeviceArena::persistent_buffer: initial allocation must specify non-zero bytes");
        if (bytes == 0) {
            throw std::invalid_argument("DeviceArena::persistent_buffer: initial allocation must specify non-zero bytes");
        }
        persistent_buffer_ = allocate_persistent(bytes, alignment);
    }
    return persistent_buffer_;
}

void DeviceArena::reset() noexcept {
    offset_ = 0;
}

} // namespace xinfer::core
