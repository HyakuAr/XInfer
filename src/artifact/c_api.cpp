#include "c_api.h"
#include "checksum.h"
#include <fstream>
#include <vector>
#include <algorithm>

extern "C" {

XINFER_EXPORT uint64_t xinfer_crc64(const void* data, size_t length) {
    if (!data || length == 0) return 0ULL;
    return xinfer::artifact::Crc64::calculate(data, length);
}

XINFER_EXPORT uint64_t xinfer_crc64_file(const char* filepath, uint64_t max_bytes) {
    if (!filepath) return 0ULL;
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return 0ULL;

    file.seekg(0, std::ios::end);
    uint64_t file_len = static_cast<uint64_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    uint64_t bytes_to_read = (max_bytes > 0 && max_bytes < file_len) ? max_bytes : file_len;

    xinfer::artifact::Crc64 crc;
    std::vector<char> buffer(4 * 1024 * 1024); // 4MB chunks for max I/O throughput

    while (bytes_to_read > 0) {
        size_t chunk = static_cast<size_t>(std::min<uint64_t>(bytes_to_read, buffer.size()));
        file.read(buffer.data(), static_cast<std::streamsize>(chunk));
        std::streamsize bytes_read = file.gcount();
        if (bytes_read <= 0) break;
        crc.update(buffer.data(), static_cast<size_t>(bytes_read));
        bytes_to_read -= static_cast<uint64_t>(bytes_read);
    }

    return crc.digest();
}

}
