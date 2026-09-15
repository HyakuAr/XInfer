#pragma once

#include <cstdint>
#include <cstddef>
#include <string_view>
#include <string>

namespace xinfer::core {

enum class DataType : uint32_t {
    Float32   = 0x0001,
    Float16   = 0x0002,
    BFloat16  = 0x0003,
    Int32     = 0x0004,
    Int8      = 0x0005,
    Uint8     = 0x0006,
    Int4      = 0x0007, // 4-bit packed integer: two values per byte
};

constexpr size_t dtype_size_bytes(DataType dtype) noexcept {
    switch (dtype) {
        case DataType::Float32:  return 4;
        case DataType::Float16:  return 2;
        case DataType::BFloat16: return 2;
        case DataType::Int32:    return 4;
        case DataType::Int8:     return 1;
        case DataType::Uint8:    return 1;
        case DataType::Int4:     return 0; // Sub-byte, use compute_tensor_bytes
    }
    return 1;
}

constexpr std::string_view dtype_name(DataType dtype) noexcept {
    switch (dtype) {
        case DataType::Float32:  return "Float32";
        case DataType::Float16:  return "Float16";
        case DataType::BFloat16: return "BFloat16";
        case DataType::Int32:    return "Int32";
        case DataType::Int8:     return "Int8";
        case DataType::Uint8:    return "Uint8";
        case DataType::Int4:     return "Int4";
    }
    return "Unknown";
}

constexpr bool is_floating_point(DataType dtype) noexcept {
    return dtype == DataType::Float32 || dtype == DataType::Float16 || dtype == DataType::BFloat16;
}

constexpr bool is_quantized(DataType dtype) noexcept {
    return dtype == DataType::Int8 || dtype == DataType::Int4 || dtype == DataType::Uint8;
}

inline size_t compute_tensor_bytes(int64_t num_elements, DataType dtype) noexcept {
    if (num_elements <= 0) return 0;
    if (dtype == DataType::Int4) {
        return static_cast<size_t>((num_elements + 1) / 2);
    }
    return static_cast<size_t>(num_elements) * dtype_size_bytes(dtype);
}

} // namespace xinfer::core
