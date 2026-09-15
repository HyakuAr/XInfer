#include "checksum.h"

namespace xinfer::artifact {

namespace {

// Standard reflected ECMA-182 polynomial: 0xC96C5795D7870F42ULL
constexpr uint64_t POLY64_ECMA = 0xC96C5795D7870F42ULL;

constexpr auto generate_crc64_table() {
    struct Table {
        uint64_t entries[256];
    } table{};

    for (uint64_t b = 0; b < 256; ++b) {
        uint64_t crc = b;
        for (int i = 0; i < 8; ++i) {
            if (crc & 1) {
                crc = (crc >> 1) ^ POLY64_ECMA;
            } else {
                crc >>= 1;
            }
        }
        table.entries[b] = crc;
    }
    return table;
}

constexpr auto CRC_TABLE = generate_crc64_table();

} // anonymous namespace

const uint64_t Crc64::table_[256] = {};

Crc64::Crc64() : crc_(0ULL) {}

void Crc64::reset() {
    crc_ = 0ULL;
}

void Crc64::update(const void* data, size_t length) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    uint64_t c = crc_ ^ 0xFFFFFFFFFFFFFFFFULL;
    for (size_t i = 0; i < length; ++i) {
        uint8_t index = static_cast<uint8_t>(c ^ bytes[i]);
        c = (c >> 8) ^ CRC_TABLE.entries[index];
    }
    crc_ = c ^ 0xFFFFFFFFFFFFFFFFULL;
}

uint64_t Crc64::digest() const {
    return crc_;
}

uint64_t Crc64::calculate(const void* data, size_t length) {
    Crc64 c;
    c.update(data, length);
    return c.digest();
}

} // namespace xinfer::artifact
