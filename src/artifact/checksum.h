#pragma once

#include <cstdint>
#include <cstddef>

namespace xinfer::artifact {

// CRC64-ECMA-182 implementation
class Crc64 {
public:
    Crc64();

    void update(const void* data, size_t length);
    uint64_t digest() const;
    void reset();

    // Convenience one-shot function
    static uint64_t calculate(const void* data, size_t length);

private:
    uint64_t crc_;
};

} // namespace xinfer::artifact
