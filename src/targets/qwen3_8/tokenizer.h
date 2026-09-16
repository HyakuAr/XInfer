#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstddef>

namespace xinfer::targets::qwen3_8 {

class QwenTokenizer {
public:
    static constexpr int64_t EOS_TOKEN_ID       = 248044; // <|endoftext|>
    static constexpr int64_t IM_START_TOKEN_ID   = 248045; // <|im_start|>
    static constexpr int64_t IM_END_TOKEN_ID     = 248046; // <|im_end|>
    static constexpr int64_t THINK_START_TOKEN_ID = 248068; // <think>
    static constexpr int64_t THINK_END_TOKEN_ID   = 248069; // </think>

    QwenTokenizer();
    ~QwenTokenizer();

    // Load tokenizer from tokenizer.json raw buffer (e.g. from .xinfer tokenizer.data section)
    bool load_from_json_buffer(const void* data, size_t size);

    // Load tokenizer directly from tokenizer.json or vocab.json / merges.txt file path
    bool load_from_file(const std::string& path);

    // Encode string to token IDs
    std::vector<int64_t> encode(const std::string& text) const;

    // Decode single token ID to string piece
    std::string decode_token(int64_t token_id) const;

    // Decode list of token IDs to full string
    std::string decode(const std::vector<int64_t>& token_ids) const;

    // Format user prompt with Qwen chat template (<|im_start|>system...<|im_start|>user...<|im_start|>assistant\n)
    std::string apply_chat_template(const std::string& user_prompt,
                                    const std::string& system_prompt = "You are a helpful assistant.") const;

    bool is_loaded() const noexcept { return !vocab_.empty(); }
    size_t vocab_size() const noexcept { return vocab_.size(); }

private:
    void init_byte_encoder();

    std::unordered_map<std::string, int64_t> vocab_;
    std::unordered_map<int64_t, std::string> id_to_token_;
    std::unordered_map<uint8_t, std::string> byte_to_unicode_;
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;
    std::unordered_map<std::string, int64_t> bpe_ranks_;
};

} // namespace xinfer::targets::qwen3_8
