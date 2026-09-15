#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#include <string_view>

namespace xinfer::artifact {

// 8-byte magic: 'X', 'I', 'N', 'F', 'E', 'R', '\0', '\0'
inline constexpr std::array<uint8_t, 8> MAGIC_BYTES = {'X', 'I', 'N', 'F', 'E', 'R', '\0', '\0'};

// 8-byte footer magic: 'X', 'I', 'N', 'F', 'F', 'O', 'O', 'T'
inline constexpr std::array<uint8_t, 8> FOOTER_MAGIC_BYTES = {'X', 'I', 'N', 'F', 'F', 'O', 'O', 'T'};

inline constexpr uint32_t CURRENT_FORMAT_VERSION = 1;
inline constexpr uint64_t SECTION_ALIGNMENT = 64; // 64-byte alignment for cache line & GPU DMA
inline constexpr size_t MAX_SECTION_NAME_LEN = 64;

enum class SectionType : uint32_t {
    RawBlob       = 0x0000,
    MetadataJson  = 0x0001,
    TokenizerData = 0x0002,
    TensorWeights = 0x0003,
    TensorScales  = 0x0004,
    ChatTemplate  = 0x0005,
    Custom        = 0x00FF
};

#pragma pack(push, 1)

// Container Header: 64 bytes total
struct FileHeader {
    uint8_t  magic[8];             // "XINFER\0\0"
    uint32_t format_version;       // 1
    uint32_t header_size;          // sizeof(FileHeader) == 64
    uint32_t flags;                // Reserved flags (0)
    uint32_t section_count;        // Number of sections in section table
    uint64_t metadata_offset;      // Offset to metadata JSON block
    uint64_t metadata_size;        // Size in bytes of metadata JSON block
    uint64_t section_table_offset; // Offset to section table
    uint64_t section_table_size;   // Size in bytes of section table
    uint64_t total_file_size;      // Expected total file size including footer
};
static_assert(sizeof(FileHeader) == 64, "FileHeader must be exactly 64 bytes");

// Section Table Entry: 96 bytes total
struct SectionEntry {
    char     name[MAX_SECTION_NAME_LEN]; // Null-terminated ASCII/UTF-8 section name
    uint32_t type_tag;                   // SectionType
    uint32_t flags;                      // Alignment or compression flags (0 for uncompressed)
    uint64_t offset;                     // File offset where section data starts (64-byte aligned)
    uint64_t size;                       // Byte length of payload
    uint64_t checksum;                   // CRC-64 of section payload
};
static_assert(sizeof(SectionEntry) == 96, "SectionEntry must be exactly 96 bytes");

// Trailing Checksum Footer: 16 bytes total
struct FileFooter {
    uint64_t file_checksum;        // CRC-64 of all preceding bytes (from 0 to footer_offset - 1)
    uint8_t  footer_magic[8];      // "XINFFOOT"
};
static_assert(sizeof(FileFooter) == 16, "FileFooter must be exactly 16 bytes");

#pragma pack(pop)

} // namespace xinfer::artifact
