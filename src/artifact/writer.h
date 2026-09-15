#pragma once

#include "container.h"
#include "metadata.h"
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace xinfer::artifact {

struct PendingSection {
    std::string name;
    SectionType type{SectionType::RawBlob};
    uint32_t flags{0};
    std::vector<uint8_t> data;
};

class ArtifactWriter {
public:
    ArtifactWriter();
    ~ArtifactWriter() = default;

    // Set container metadata
    void set_metadata(const ArtifactMetadata& metadata);
    void set_raw_metadata_json(std::string json_str);

    // Add named section blobs
    bool add_section(std::string name, SectionType type, const void* data, size_t size, uint32_t flags = 0);
    bool add_section(std::string name, SectionType type, std::vector<uint8_t> data, uint32_t flags = 0);

    // Write container to file
    bool write_to_file(const std::string& filepath, std::string* error_msg = nullptr);

    // Get number of pending sections
    size_t section_count() const { return sections_.size(); }

private:
    std::string metadata_json_;
    std::vector<PendingSection> sections_;
};

} // namespace xinfer::artifact
