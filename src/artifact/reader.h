#pragma once

#include "container.h"
#include "metadata.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <fstream>

namespace xinfer::artifact {

class ArtifactReader {
public:
    ArtifactReader();
    ~ArtifactReader();

    // Disallow copy, allow move
    ArtifactReader(const ArtifactReader&) = delete;
    ArtifactReader& operator=(const ArtifactReader&) = delete;
    ArtifactReader(ArtifactReader&&) noexcept;
    ArtifactReader& operator=(ArtifactReader&&) noexcept;

    // Open and parse container
    bool open(const std::string& filepath, std::string* error_msg = nullptr);
    void close();
    bool is_open() const { return is_open_; }

    // Header and footer access
    const FileHeader& header() const { return header_; }
    const FileFooter& footer() const { return footer_; }

    // Metadata access
    const ArtifactMetadata& metadata() const { return metadata_; }
    ArtifactMetadata& mutable_metadata() { return metadata_; }
    const std::string& raw_metadata_json() const { return raw_metadata_json_; }

    // Section inspection
    std::vector<std::string> list_sections() const;
    bool has_section(const std::string& name) const;
    std::optional<SectionEntry> get_section_info(const std::string& name) const;

    // Read full section payload into vector
    bool read_section(const std::string& name, std::vector<uint8_t>& out_data, std::string* error_msg = nullptr);

    // Read section into caller-provided buffer
    bool read_section(const std::string& name, void* dst_buffer, size_t buffer_size, std::string* error_msg = nullptr);

    // Validate entire file checksum (CRC-64)
    bool validate_checksum(std::string* error_msg = nullptr, size_t buffer_size = 4 * 1024 * 1024);

private:
    std::string filepath_;
    mutable std::ifstream file_stream_;
    bool is_open_{false};

    FileHeader header_{};
    FileFooter footer_{};
    std::string raw_metadata_json_;
    ArtifactMetadata metadata_{};

    std::vector<SectionEntry> sections_;
    std::unordered_map<std::string, size_t> section_index_map_;
};

} // namespace xinfer::artifact
