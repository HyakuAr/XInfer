#pragma once

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <cstdint>

namespace xinfer {

struct GenerationConfig {
    int     max_new_tokens{256};
    float   temperature{0.0f};           // 0.0 = greedy argmax sampling
    int64_t eos_token_id{248044};        // <|endoftext|>
    int64_t im_end_token_id{248046};     // <|im_end|>
    bool    apply_chat_template{true};
};

struct EngineConfig {
    std::string artifact_path;           // Path to .xinfer file
    bool        prefer_b60{true};        // Prefer Intel Arc Pro B60 GPU
    size_t      arena_capacity_bytes{256 * 1024 * 1024}; // 256MB scratchpad arena
};

// Streaming token callback: returns false to halt generation
using TokenCallback = std::function<bool(const std::string& token_piece, int64_t token_id)>;

struct GenerationResult {
    std::string          text;
    std::vector<int64_t> token_ids;
    size_t               prompt_tokens{0};
    size_t               generated_tokens{0};
    double               time_to_first_token_sec{0.0};
    double               decode_tokens_per_sec{0.0};
    double               total_time_sec{0.0};
};

// Public Engine interface (PIMPL pattern per AGENTS.md §4)
class Engine {
public:
    Engine();
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;

    // Load .xinfer container artifact into GPU device memory
    bool load(const EngineConfig& config, std::string* error_msg = nullptr);

    // Query if model is loaded and ready for inference
    bool is_loaded() const noexcept;

    // Execute end-to-end single-request generation (prompt -> tokens -> detokenize)
    GenerationResult generate(const std::string& prompt,
                              const GenerationConfig& gen_config = {},
                              TokenCallback callback = nullptr);

    // Reset runtime state
    void reset();

private:
    class EngineImpl;
    std::unique_ptr<EngineImpl> impl_;
};

} // namespace xinfer
