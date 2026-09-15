#pragma once

#include "types.h"
#include <vector>
#include <initializer_list>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cassert>

namespace xinfer::core {

class DeviceContext;

class TensorShape {
public:
    TensorShape() = default;
    TensorShape(std::initializer_list<int64_t> dims) : dims_(dims) {}
    explicit TensorShape(std::vector<int64_t> dims) : dims_(std::move(dims)) {}

    size_t ndim() const noexcept { return dims_.size(); }
    int64_t dim(size_t i) const {
        if (i >= dims_.size()) {
            throw std::out_of_range("Tensor dimension index out of range");
        }
        return dims_[i];
    }
    int64_t operator[](size_t i) const noexcept { return dims_[i]; }

    int64_t numel() const noexcept {
        if (dims_.empty()) return 0;
        int64_t count = 1;
        for (int64_t d : dims_) count *= d;
        return count;
    }

    const std::vector<int64_t>& dims() const noexcept { return dims_; }

    std::vector<int64_t> compute_contiguous_strides() const {
        if (dims_.empty()) return {};
        std::vector<int64_t> strides(dims_.size(), 1);
        for (int i = static_cast<int>(dims_.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * dims_[i + 1];
        }
        return strides;
    }

    bool operator==(const TensorShape& other) const noexcept {
        return dims_ == other.dims_;
    }

    bool operator!=(const TensorShape& other) const noexcept {
        return dims_ != other.dims_;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < dims_.size(); ++i) {
            oss << dims_[i];
            if (i + 1 < dims_.size()) oss << ", ";
        }
        oss << "]";
        return oss.str();
    }

private:
    std::vector<int64_t> dims_;
};

class TensorView {
public:
    TensorView() : data_(nullptr), dtype_(DataType::Float32) {}

    TensorView(void* data, TensorShape shape, DataType dtype)
        : data_(data), shape_(shape), strides_(shape.compute_contiguous_strides()), dtype_(dtype) {}

    TensorView(void* data, TensorShape shape, std::vector<int64_t> strides, DataType dtype)
        : data_(data), shape_(std::move(shape)), strides_(std::move(strides)), dtype_(dtype) {}

    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }

    template<typename T>
    T* data_as() noexcept { return static_cast<T*>(data_); }

    template<typename T>
    const T* data_as() const noexcept { return static_cast<const T*>(data_); }

    const TensorShape& shape() const noexcept { return shape_; }
    const std::vector<int64_t>& strides() const noexcept { return strides_; }
    DataType dtype() const noexcept { return dtype_; }

    int64_t numel() const noexcept { return shape_.numel(); }
    size_t byte_size() const noexcept { return compute_tensor_bytes(shape_.numel(), dtype_); }

    bool is_contiguous() const noexcept {
        if (shape_.ndim() <= 1) return true;
        auto expected = shape_.compute_contiguous_strides();
        return strides_ == expected;
    }

    // Slices a dimension [start, start + length)
    TensorView slice(size_t dim_idx, int64_t start, int64_t length) const;

    // Reshapes a contiguous view
    TensorView reshape(TensorShape new_shape) const;

private:
    void* data_{nullptr};
    TensorShape shape_;
    std::vector<int64_t> strides_;
    DataType dtype_;
};

class DeviceTensor {
public:
    DeviceTensor() = default;
    ~DeviceTensor();

    DeviceTensor(DeviceTensor&& other) noexcept;
    DeviceTensor& operator=(DeviceTensor&& other) noexcept;

    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;

    // Allocate USM device tensor
    static DeviceTensor allocate(DeviceContext& ctx, TensorShape shape, DataType dtype, size_t alignment = 64);

    TensorView view() noexcept {
        return TensorView(data_, shape_, dtype_);
    }

    TensorView view() const noexcept {
        return TensorView(data_, shape_, dtype_);
    }

    void* raw_data() noexcept { return data_; }
    const void* raw_data() const noexcept { return data_; }

    template<typename T>
    T* data_as() noexcept { return static_cast<T*>(data_); }

    template<typename T>
    const T* data_as() const noexcept { return static_cast<const T*>(data_); }

    const TensorShape& shape() const noexcept { return shape_; }
    DataType dtype() const noexcept { return dtype_; }
    size_t byte_size() const noexcept { return compute_tensor_bytes(shape_.numel(), dtype_); }

    void copy_from_host(const void* host_src, size_t bytes = 0);
    void copy_to_host(void* host_dst, size_t bytes = 0) const;

private:
    DeviceTensor(DeviceContext* ctx, void* data, TensorShape shape, DataType dtype)
        : ctx_(ctx), data_(data), shape_(std::move(shape)), dtype_(dtype) {}

    DeviceContext* ctx_{nullptr};
    void* data_{nullptr};
    TensorShape shape_;
    DataType dtype_{DataType::Float32};
};

} // namespace xinfer::core
