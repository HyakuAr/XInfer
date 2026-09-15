#include "writer.h"
#include "checksum.h"
#include <fstream>
#include <cstring>
#include <algorithm>
#include <vector>

namespace xinfer::artifact {

ArtifactWriter::ArtifactWriter() = default;

void ArtifactWriter::set_metadata(const ArtifactMetadata& metadata) {
    metadata_json_ = metadata.to_json();
}

void ArtifactWriter::set_raw_metadata_json(std::string json_str) {
    metadata_json_ = std::move(json_str);
}

bool ArtifactWriter::add_section(std::string name, SectionType type, const void* data, size_t size, uint32_t flags) {
    if (name.empty() || name.size() >= MAX_SECTION_NAME_LEN) {
        return false;
    }
    // Check for duplicate names
    for (const auto& s : sections_) {
        if (s.name == name) {
            return false;
        }
    }

    PendingSection sec;
    sec.name = std::move(name);
    sec.type = type;
    sec.flags = flags;
    if (size > 0 && data != nullptr) {
        const auto* ptr = static_cast<const uint8_t*>(data);
        sec.data.assign(ptr, ptr + size);
    }
    sections_.push_back(std::move(sec));
    return true;
}

bool ArtifactWriter::add_section(std::string name, SectionType type, std::vector<uint8_t> data, uint32_t flags) {
    if (name.empty() || name.size() >= MAX_SECTION_NAME_LEN) {
        return false;
    }
    for (const auto& s : sections_) {
        if (s.name == name) {
            return false;
        }
    }

    PendingSection sec;
    sec.name = std::move(name);
    sec.type = type;
    sec.flags = flags;
    sec.data = std::move(data);
    sections_.push_back(std::move(sec));
    return true;
}

bool ArtifactWriter::write_to_file(const std::string& filepath, std::string* error_msg) {
    std::ofstream out(filepath, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        if (error_msg) *error_msg = "Failed to open file for writing: " + filepath;
        return false;
    }

    FileHeader header{};
    std::memcpy(header.magic, MAGIC_BYTES.data(), 8);
    header.format_version = CURRENT_FORMAT_VERSION;
    header.header_size = static_cast<uint32_t>(sizeof(FileHeader));
    header.flags = 0;
    header.section_count = static_cast<uint32_t>(sections_.size());

    // Write placeholder header
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    uint64_t current_offset = sizeof(FileHeader);

    // Write metadata
    header.metadata_offset = current_offset;
    header.metadata_size = metadata_json_.size();
    if (header.metadata_size > 0) {
        out.write(metadata_json_.data(), static_cast<std::streamsize>(header.metadata_size));
        current_offset += header.metadata_size;
    }

    // Write sections with 64-byte alignment
    std::vector<SectionEntry> entries;
    entries.reserve(sections_.size());

    for (const auto& sec : sections_) {
        uint64_t pad = (SECTION_ALIGNMENT - (current_offset % SECTION_ALIGNMENT)) % SECTION_ALIGNMENT;
        if (pad > 0) {
            std::vector<char> zeros(pad, 0);
            out.write(zeros.data(), static_cast<std::streamsize>(pad));
            current_offset += pad;
        }

        SectionEntry entry{};
        std::memset(&entry, 0, sizeof(entry));
        std::strncpy(entry.name, sec.name.c_str(), sizeof(entry.name) - 1);
        entry.type_tag = static_cast<uint32_t>(sec.type);
        entry.flags = sec.flags;
        entry.offset = current_offset;
        entry.size = sec.data.size();
        entry.checksum = sec.data.empty() ? 0ULL : Crc64::calculate(sec.data.data(), sec.data.size());

        if (!sec.data.empty()) {
            out.write(reinterpret_cast<const char*>(sec.data.data()), static_cast<std::streamsize>(sec.data.size()));
            current_offset += sec.data.size();
        }

        entries.push_back(entry);
    }

    // Write Section Table aligned to 64 bytes
    uint64_t pad = (SECTION_ALIGNMENT - (current_offset % SECTION_ALIGNMENT)) % SECTION_ALIGNMENT;
    if (pad > 0) {
        std::vector<char> zeros(pad, 0);
        out.write(zeros.data(), static_cast<std::streamsize>(pad));
        current_offset += pad;
    }

    header.section_table_offset = current_offset;
    header.section_table_size = entries.size() * sizeof(SectionEntry);

    if (!entries.empty()) {
        out.write(reinterpret_cast<const char*>(entries.data()), static_cast<std::streamsize>(header.section_table_size));
        current_offset += header.section_table_size;
    }

    header.total_file_size = current_offset + sizeof(FileFooter);

    // Seek back to write final header
    out.seekp(0, std::ios::beg);
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.flush();

    if (!out.good()) {
        if (error_msg) *error_msg = "I/O error during artifact generation.";
        return false;
    }
    out.close();

    // Now calculate full file checksum up to current_offset and append FileFooter
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        if (error_msg) *error_msg = "Failed to reopen file for checksumming.";
        return false;
    }

    Crc64 file_crc;
    std::vector<char> buffer(64 * 1024);
    uint64_t bytes_to_read = current_offset;

    while (bytes_to_read > 0) {
        size_t chunk = static_cast<size_t>(std::min<uint64_t>(bytes_to_read, buffer.size()));
        in.read(buffer.data(), static_cast<std::streamsize>(chunk));
        std::streamsize bytes_read = in.gcount();
        if (bytes_read <= 0) {
            if (error_msg) *error_msg = "Premature EOF while computing file checksum.";
            return false;
        }
        file_crc.update(buffer.data(), static_cast<size_t>(bytes_read));
        bytes_to_read -= static_cast<uint64_t>(bytes_read);
    }
    in.close();

    // Append FileFooter
    std::ofstream append_out(filepath, std::ios::binary | std::ios::app);
    if (!append_out.is_open()) {
        if (error_msg) *error_msg = "Failed to reopen file to append footer.";
        return false;
    }

    FileFooter footer{};
    footer.file_checksum = file_crc.digest();
    std::memcpy(footer.footer_magic, FOOTER_MAGIC_BYTES.data(), 8);

    append_out.write(reinterpret_cast<const char*>(&footer), sizeof(footer));
    append_out.flush();

    if (!append_out.good()) {
        if (error_msg) *error_msg = "Failed to write artifact footer.";
        return false;
    }

    return true;
}

} // namespace xinfer::artifact
