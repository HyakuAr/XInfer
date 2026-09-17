#pragma once

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <cstdint>

namespace xinfer {

struct ChatMessage {
    std::string role;              // "system", "user", "assistant", "tool"
    std::string content;
    std::string reasoning_content; // optional reasoning content for assistant messages
};

struct GenerationConfig {
    int     max_new_tokens{256};
    float   temperature{0.0f};           // 0.0 = greedy argmax sampling
    int64_t eos_token_id{-1};            // Sentinel: default (-1) resolves dynamically to loaded model/tokenizer EOS (<|endoftext|>)
    int64_t im_end_token_id{-1};         // Sentinel: default (-1) resolves dynamically to loaded model/tokenizer IM_END (<|im_end|>)
    bool    apply_chat_template{true};
};

struct EngineConfig {
    std::string artifact_path;           // Path to .xinfer file
    bool        prefer_b60{true};        // Prefer Intel Arc Pro B60 GPU
    size_t      arena_capacity_bytes{256 * 1024 * 1024}; // 256MB scratchpad arena
    size_t      max_seq_len{8192};       // Maximum sequence length supported by KV cache
    size_t      prefill_chunk_size{512}; // Chunk size for chunked prefill
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
    std::string          finish_reason{"stop"}; // "stop" or "length"
    bool                 success{true};
    std::string          error_msg;
    std::string          error_code;
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

    // Query maximum sequence length supported by the model/KV cache
    size_t max_seq_len() const noexcept;

    // Query special token IDs resolved dynamically from loaded model/tokenizer (source of truth)
    int64_t eos_token_id() const noexcept;
    int64_t im_end_token_id() const noexcept;

    // Format messages with the loaded model's real chat template (chat_template.jinja)
    std::string apply_chat_template(const std::vector<ChatMessage>& messages) const;
    std::string apply_chat_template(const std::string& user_prompt, const std::string& system_prompt = "") const;

    // Tokenize text or messages and return token count without generating
    size_t count_tokens(const std::string& text, bool apply_chat_template = true) const;
    size_t count_tokens(const std::vector<ChatMessage>& messages) const;

    // Validate if prompt tokens and generation config fit within max_seq_len
    bool validate_tokens(size_t prompt_tokens, int max_new_tokens, std::string* error_msg = nullptr) const;

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
