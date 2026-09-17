#pragma once

#include "xinfer/engine.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <string_view>
#include <cstddef>

namespace xinfer::targets::qwen3_8 {

using xinfer::ChatMessage;

struct ChatTemplateOptions {
    bool        add_generation_prompt{true};
    bool        enable_thinking{true};
    std::string reasoning_effort{"xhigh"}; // "xhigh", "medium", "low"
    bool        preserve_thinking{true};
};

class QwenChatTemplate {
public:
    QwenChatTemplate();
    ~QwenChatTemplate() = default;

    // Load from raw Jinja template buffer (e.g. from .xinfer container artifact section 'chat_template.jinja')
    bool load_from_buffer(const void* data, size_t size, std::string* error_msg = nullptr);

    // Load from raw Jinja template string
    bool load_from_string(const std::string& template_str, std::string* error_msg = nullptr);

    // Load from file (e.g. checkpoint directory chat_template.jinja)
    bool load_from_file(const std::string& path, std::string* error_msg = nullptr);

    // True if a template has been loaded
    bool is_loaded() const noexcept { return is_loaded_; }

    // Render messages to prompt string faithfully matching chat_template.jinja specification
    std::string render(const std::vector<ChatMessage>& messages,
                       const ChatTemplateOptions& options = {},
                       std::string* error_msg = nullptr) const;

    // Render single turn convenience
    std::string render(const std::string& user_prompt,
                       const std::string& system_prompt = "",
                       const ChatTemplateOptions& options = {},
                       std::string* error_msg = nullptr) const;

    // Template metadata access
    const std::string& raw_template() const noexcept { return raw_template_; }
    const std::string& default_reasoning_effort() const noexcept { return default_reasoning_effort_; }
    std::string get_reasoning_instructions(const std::string& effort) const;

private:
    void init_defaults();
    void parse_template();

    bool is_loaded_{false};
    std::string raw_template_;
    std::string default_reasoning_effort_{"xhigh"};
    bool default_enable_thinking_{true};
    std::unordered_map<std::string, std::string> reasoning_instructions_;
};

} // namespace xinfer::targets::qwen3_8
