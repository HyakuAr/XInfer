#include "reader.h"
#include "checksum.h"
#include <cstring>
#include <algorithm>
#include <sstream>

namespace xinfer::artifact {

ArtifactReader::ArtifactReader() = default;

ArtifactReader::~ArtifactReader() {
    close();
}

ArtifactReader::ArtifactReader(ArtifactReader&& other) noexcept
    : filepath_(std::move(other.filepath_)),
      file_stream_(std::move(other.file_stream_)),
      is_open_(other.is_open_),
      header_(other.header_),
      footer_(other.footer_),
      raw_metadata_json_(std::move(other.raw_metadata_json_)),
      metadata_(std::move(other.metadata_)),
      sections_(std::move(other.sections_)),
      section_index_map_(std::move(other.section_index_map_)) {
    other.is_open_ = false;
}

ArtifactReader& ArtifactReader::operator=(ArtifactReader&& other) noexcept {
    if (this != &other) {
        close();
        filepath_ = std::move(other.filepath_);
        file_stream_ = std::move(other.file_stream_);
        is_open_ = other.is_open_;
        header_ = other.header_;
        footer_ = other.footer_;
        raw_metadata_json_ = std::move(other.raw_metadata_json_);
        metadata_ = std::move(other.metadata_);
        sections_ = std::move(other.sections_);
        section_index_map_ = std::move(other.section_index_map_);
        other.is_open_ = false;
    }
    return *this;
}

void ArtifactReader::close() {
    if (file_stream_.is_open()) {
        file_stream_.close();
    }
    is_open_ = false;
    filepath_.clear();
    raw_metadata_json_.clear();
    metadata_ = ArtifactMetadata{};
    sections_.clear();
    section_index_map_.clear();
    std::memset(&header_, 0, sizeof(header_));
    std::memset(&footer_, 0, sizeof(footer_));
}

bool ArtifactReader::open(const std::string& filepath, std::string* error_msg) {
    close();

    filepath_ = filepath;
    file_stream_.open(filepath, std::ios::binary);
    if (!file_stream_.is_open()) {
        if (error_msg) *error_msg = "Could not open file: " + filepath;
        return false;
    }

    file_stream_.seekg(0, std::ios::end);
    uint64_t actual_file_size = static_cast<uint64_t>(file_stream_.tellg());

    if (actual_file_size < sizeof(FileHeader) + sizeof(FileFooter)) {
        if (error_msg) *error_msg = "File is smaller than minimum required header and footer size.";
        close();
        return false;
    }

    // 1. Read and validate Header
    file_stream_.seekg(0, std::ios::beg);
    file_stream_.read(reinterpret_cast<char*>(&header_), sizeof(FileHeader));
    if (!file_stream_.good()) {
        if (error_msg) *error_msg = "Failed to read container header.";
        close();
        return false;
    }

    if (std::memcmp(header_.magic, MAGIC_BYTES.data(), 8) != 0) {
        if (error_msg) *error_msg = "Invalid file magic; not a .xinfer artifact.";
        close();
        return false;
    }

    if (header_.format_version != CURRENT_FORMAT_VERSION) {
        if (error_msg) *error_msg = "Unsupported .xinfer format version: " + std::to_string(header_.format_version);
        close();
        return false;
    }

    if (header_.header_size != sizeof(FileHeader)) {
        if (error_msg) *error_msg = "Corrupted header size field: " + std::to_string(header_.header_size);
        close();
        return false;
    }

    if (header_.total_file_size != actual_file_size) {
        if (error_msg) *error_msg = "File size mismatch: header expects " + std::to_string(header_.total_file_size) +
                                    " bytes, actual on disk is " + std::to_string(actual_file_size) + " bytes.";
        close();
        return false;
    }

    // 2. Read and validate Footer
    file_stream_.seekg(static_cast<std::streamoff>(actual_file_size - sizeof(FileFooter)), std::ios::beg);
    file_stream_.read(reinterpret_cast<char*>(&footer_), sizeof(FileFooter));
    if (!file_stream_.good()) {
        if (error_msg) *error_msg = "Failed to read container footer.";
        close();
        return false;
    }

    if (std::memcmp(footer_.footer_magic, FOOTER_MAGIC_BYTES.data(), 8) != 0) {
        if (error_msg) *error_msg = "Invalid footer magic; file may be corrupted or truncated.";
        close();
        return false;
    }

    // 3. Read metadata
    if (header_.metadata_size > 0) {
        if (header_.metadata_offset + header_.metadata_size > actual_file_size) {
            if (error_msg) *error_msg = "Metadata offset/size points out of bounds.";
            close();
            return false;
        }

        raw_metadata_json_.resize(header_.metadata_size);
        file_stream_.seekg(static_cast<std::streamoff>(header_.metadata_offset), std::ios::beg);
        file_stream_.read(raw_metadata_json_.data(), static_cast<std::streamsize>(header_.metadata_size));
        if (!file_stream_.good()) {
            if (error_msg) *error_msg = "Failed reading metadata block.";
            close();
            return false;
        }

        auto parsed = ArtifactMetadata::from_json(raw_metadata_json_);
        if (parsed.has_value()) {
            metadata_ = std::move(parsed.value());
        }
    }

    // 4. Read section table
    if (header_.section_count > 0) {
        uint64_t expected_table_size = header_.section_count * sizeof(SectionEntry);
        if (header_.section_table_size != expected_table_size) {
            if (error_msg) *error_msg = "Section table size does not match section count.";
            close();
            return false;
        }

        if (header_.section_table_offset + header_.section_table_size > actual_file_size) {
            if (error_msg) *error_msg = "Section table points out of bounds.";
            close();
            return false;
        }

        sections_.resize(header_.section_count);
        file_stream_.seekg(static_cast<std::streamoff>(header_.section_table_offset), std::ios::beg);
        file_stream_.read(reinterpret_cast<char*>(sections_.data()), static_cast<std::streamsize>(expected_table_size));
        if (!file_stream_.good()) {
            if (error_msg) *error_msg = "Failed reading section table.";
            close();
            return false;
        }

        for (size_t i = 0; i < sections_.size(); ++i) {
            const auto& entry = sections_[i];
            std::string name(entry.name);
            if (entry.offset + entry.size > actual_file_size) {
                if (error_msg) *error_msg = "Section '" + name + "' offset + size points out of bounds.";
                close();
                return false;
            }
            section_index_map_[name] = i;
        }
    }

    is_open_ = true;
    return true;
}

std::vector<std::string> ArtifactReader::list_sections() const {
    std::vector<std::string> names;
    names.reserve(sections_.size());
    for (const auto& sec : sections_) {
        names.emplace_back(sec.name);
    }
    return names;
}

bool ArtifactReader::has_section(const std::string& name) const {
    return section_index_map_.find(name) != section_index_map_.end();
}

std::optional<SectionEntry> ArtifactReader::get_section_info(const std::string& name) const {
    auto it = section_index_map_.find(name);
    if (it == section_index_map_.end()) {
        return std::nullopt;
    }
    return sections_[it->second];
}

bool ArtifactReader::read_section(const std::string& name, std::vector<uint8_t>& out_data, std::string* error_msg) {
    auto info = get_section_info(name);
    if (!info.has_value()) {
        if (error_msg) *error_msg = "Section not found: " + name;
        return false;
    }

    out_data.resize(info->size);
    if (info->size == 0) {
        return true;
    }

    return read_section(name, out_data.data(), out_data.size(), error_msg);
}

bool ArtifactReader::read_section(const std::string& name, void* dst_buffer, size_t buffer_size, std::string* error_msg) {
    if (!is_open_) {
        if (error_msg) *error_msg = "ArtifactReader is not open.";
        return false;
    }

    auto info = get_section_info(name);
    if (!info.has_value()) {
        if (error_msg) *error_msg = "Section not found: " + name;
        return false;
    }

    if (buffer_size < info->size) {
        if (error_msg) *error_msg = "Destination buffer too small for section: " + name;
        return false;
    }

    if (info->size == 0) {
        return true;
    }

    file_stream_.seekg(static_cast<std::streamoff>(info->offset), std::ios::beg);
    file_stream_.read(reinterpret_cast<char*>(dst_buffer), static_cast<std::streamsize>(info->size));
    if (!file_stream_.good()) {
        if (error_msg) *error_msg = "I/O error while reading section: " + name;
        return false;
    }

    // Validate section checksum
    uint64_t computed_crc = Crc64::calculate(dst_buffer, info->size);
    if (computed_crc != info->checksum) {
        if (error_msg) *error_msg = "Checksum mismatch for section: " + name +
                                    " (expected: " + std::to_string(info->checksum) +
                                    ", got: " + std::to_string(computed_crc) + ")";
        return false;
    }

    return true;
}

bool ArtifactReader::validate_checksum(std::string* error_msg, size_t buffer_size) {
    if (!is_open_) {
        if (error_msg) *error_msg = "ArtifactReader is not open.";
        return false;
    }

    if (buffer_size == 0) {
        buffer_size = 4 * 1024 * 1024;
    }

    uint64_t bytes_to_read = header_.total_file_size - sizeof(FileFooter);
    file_stream_.clear();
    file_stream_.seekg(0, std::ios::beg);

    Crc64 file_crc;
    std::vector<char> buffer(buffer_size);

    while (bytes_to_read > 0) {
        size_t chunk = static_cast<size_t>(std::min<uint64_t>(bytes_to_read, static_cast<uint64_t>(buffer.size())));
        file_stream_.read(buffer.data(), static_cast<std::streamsize>(chunk));
        std::streamsize bytes_read = file_stream_.gcount();
        if (bytes_read <= 0) {
            if (error_msg) *error_msg = "Premature EOF while validating file checksum.";
            file_stream_.clear();
            return false;
        }
        file_crc.update(buffer.data(), static_cast<size_t>(bytes_read));
        bytes_to_read -= static_cast<uint64_t>(bytes_read);
    }

    file_stream_.clear();

    if (file_crc.digest() != footer_.file_checksum) {
        std::ostringstream ss;
        ss << "File trailing checksum mismatch! (expected: 0x"
           << std::hex << footer_.file_checksum
           << ", got: 0x" << file_crc.digest() << std::dec << ")";
        if (error_msg) *error_msg = ss.str();
        return false;
    }

    return true;
}

} // namespace xinfer::artifact
