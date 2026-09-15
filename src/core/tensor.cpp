#include "tensor.h"
#include "device.h"
#include <stdexcept>
#include <cstring>
#include <utility>

namespace xinfer::core {

TensorView TensorView::slice(size_t dim_idx, int64_t start, int64_t length) const {
    if (dim_idx >= shape_.ndim()) {
        throw std::out_of_range("Slice dimension index out of range");
    }
    int64_t dim_len = shape_.dim(dim_idx);
    if (start < 0 || length < 0 || start + length > dim_len) {
        throw std::out_of_range("Slice range [start, start + length) out of bounds");
    }

    auto new_dims = shape_.dims();
    new_dims[dim_idx] = length;
    TensorShape new_shape(std::move(new_dims));

    // Pointer offset
    int64_t element_offset = start * strides_[dim_idx];
    size_t byte_offset = 0;
    if (dtype_ == DataType::Int4) {
        if (element_offset % 2 != 0) {
            throw std::runtime_error("INT4 slice offset must be byte-aligned (even element index)");
        }
        byte_offset = static_cast<size_t>(element_offset / 2);
    } else {
        byte_offset = static_cast<size_t>(element_offset) * dtype_size_bytes(dtype_);
    }

    uint8_t* new_ptr = static_cast<uint8_t*>(data_) + byte_offset;
    return TensorView(new_ptr, std::move(new_shape), strides_, dtype_);
}

TensorView TensorView::reshape(TensorShape new_shape) const {
    if (!is_contiguous()) {
        throw std::runtime_error("Cannot reshape non-contiguous TensorView");
    }
    if (new_shape.numel() != shape_.numel()) {
        throw std::invalid_argument("Reshape element count mismatch: " +
                                   std::to_string(shape_.numel()) + " vs " +
                                   std::to_string(new_shape.numel()));
    }
    return TensorView(data_, std::move(new_shape), dtype_);
}

DeviceTensor DeviceTensor::allocate(DeviceContext& ctx, TensorShape shape, DataType dtype, size_t alignment) {
    size_t bytes = compute_tensor_bytes(shape.numel(), dtype);
    void* ptr = ctx.allocate_device(bytes, alignment);
    return DeviceTensor(&ctx, ptr, std::move(shape), dtype);
}

DeviceTensor::~DeviceTensor() {
    if (ctx_ && data_) {
        ctx_->free_device(data_);
        data_ = nullptr;
    }
}

DeviceTensor::DeviceTensor(DeviceTensor&& other) noexcept
    : ctx_(other.ctx_), data_(other.data_), shape_(std::move(other.shape_)), dtype_(other.dtype_) {
    other.ctx_ = nullptr;
    other.data_ = nullptr;
}

DeviceTensor& DeviceTensor::operator=(DeviceTensor&& other) noexcept {
    if (this != &other) {
        if (ctx_ && data_) {
            ctx_->free_device(data_);
        }
        ctx_ = other.ctx_;
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;

        other.ctx_ = nullptr;
        other.data_ = nullptr;
    }
    return *this;
}

void DeviceTensor::copy_from_host(const void* host_src, size_t bytes) {
    if (!ctx_ || !data_) {
        throw std::runtime_error("DeviceTensor is not allocated");
    }
    size_t copy_bytes = (bytes == 0) ? byte_size() : bytes;
    ctx_->copy_host_to_device(data_, host_src, copy_bytes, true);
}

void DeviceTensor::copy_to_host(void* host_dst, size_t bytes) const {
    if (!ctx_ || !data_) {
        throw std::runtime_error("DeviceTensor is not allocated");
    }
    size_t copy_bytes = (bytes == 0) ? byte_size() : bytes;
    ctx_->copy_device_to_host(host_dst, data_, copy_bytes, true);
}

} // namespace xinfer::core
