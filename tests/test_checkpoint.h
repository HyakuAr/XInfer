#pragma once

#include <cstdlib>
#include <filesystem>
#include <string>

namespace xinfer::test {

inline std::filesystem::path get_checkpoint_dir() {
    const char* env = std::getenv("XINFER_CHECKPOINT_DIR");
    if (env && *env != '\0') {
        std::string s(env);
        size_t start = s.find_first_not_of(" \t\r\n");
        size_t end = s.find_last_not_of(" \t\r\n");
        if (start != std::string::npos && end != std::string::npos) {
            return std::filesystem::path(s.substr(start, end - start + 1));
        }
    }
#ifdef XINFER_DEFAULT_CHECKPOINT_DIR
    return std::filesystem::path(XINFER_DEFAULT_CHECKPOINT_DIR);
#else
    return std::filesystem::path(R"(H:\Models\Qwen3.8-27B)");
#endif
}

} // namespace xinfer::test
