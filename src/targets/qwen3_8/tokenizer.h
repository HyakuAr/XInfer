#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstddef>
#include <utility>

namespace xinfer::targets::qwen3_8 {

class QwenTokenizer {
public:
    QwenTokenizer();
    ~QwenTokenizer();

    // Load tokenizer from tokenizer.json raw buffer (e.g. from .xinfer tokenizer.data section)
    bool load_from_json_buffer(const void* data, size_t size, std::string* error_msg = nullptr);

    // Load tokenizer directly from tokenizer.json or checkpoint directory containing tokenizer.json / config.json
    bool load_from_file(const std::string& path, std::string* error_msg = nullptr);

    // Validate tokenizer special tokens against checkpoint config.json (buffer or path)
    bool validate_against_config_buffer(const void* data, size_t size, std::string* error_msg = nullptr);
    bool validate_against_config(const std::string& config_path, std::string* error_msg = nullptr);

    // Encode string to token IDs via rank-ordered BPE merging
    std::vector<int64_t> encode(const std::string& text) const;

    // Decode single token ID to string piece
    std::string decode_token(int64_t token_id) const;

    // Decode list of token IDs to full string
    std::string decode(const std::vector<int64_t>& token_ids) const;

    // Format user prompt with Qwen chat template (<|im_start|>system...<|im_start|>user...<|im_start|>assistant\n)
    std::string apply_chat_template(const std::string& user_prompt,
                                    const std::string& system_prompt = "You are a helpful assistant.") const;

    bool is_loaded() const noexcept { return !vocab_.empty() && !bpe_ranks_.empty(); }
    size_t vocab_size() const noexcept { return vocab_.size(); }
    size_t merges_size() const noexcept { return bpe_ranks_.size(); }

    // Special token IDs dynamically read from loaded tokenizer.json / config.json (source of truth)
    int64_t eos_token_id() const noexcept { return eos_token_id_; }
    int64_t im_start_token_id() const noexcept { return im_start_token_id_; }
    int64_t im_end_token_id() const noexcept { return im_end_token_id_; }
    int64_t think_start_token_id() const noexcept { return think_start_token_id_; }
    int64_t think_end_token_id() const noexcept { return think_end_token_id_; }

    // Lookup special token ID by exact text (e.g. "<|im_start|>"), returns -1 if not registered
    int64_t special_token_to_id(const std::string& name) const;
    bool has_special_token(const std::string& name) const;

private:
    void init_byte_encoder();
    std::vector<std::string> bpe_merge(const std::string& piece) const;

    std::unordered_map<std::string, int64_t> vocab_;
    std::unordered_map<int64_t, std::string> id_to_token_;
    std::unordered_map<uint8_t, std::string> byte_to_unicode_;
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;
    std::unordered_map<std::string, int64_t> bpe_ranks_;
    mutable std::unordered_map<std::string, std::vector<std::string>> bpe_cache_;

    std::vector<std::pair<std::string, int64_t>> special_tokens_;
    int64_t eos_token_id_{-1};
    int64_t im_start_token_id_{-1};
    int64_t im_end_token_id_{-1};
    int64_t think_start_token_id_{-1};
    int64_t think_end_token_id_{-1};
};

} // namespace xinfer::targets::qwen3_8
