#pragma once

#include <cstdint>
#include <cstddef>

#ifdef _WIN32
  #define XINFER_EXPORT __declspec(dllexport)
#else
  #define XINFER_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

// Compute CRC-64 over in-memory buffer
XINFER_EXPORT uint64_t xinfer_crc64(const void* data, size_t length);

// Compute CRC-64 over file up to max_bytes (0 = entire file)
XINFER_EXPORT uint64_t xinfer_crc64_file(const char* filepath, uint64_t max_bytes);

}
