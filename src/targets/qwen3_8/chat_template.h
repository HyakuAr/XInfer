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
    std::string reasoning_effort{""}; // Empty defaults to template's default_reasoning_effort ("xhigh", "medium", "low")
    bool        preserve_thinking{true};
};

// ============================================================================
// Architectural Note: Dedicated Compiled C++ ChatML Renderer for Qwen3.8
// ============================================================================
// Per AGENTS.md §1 & §4, xinfer is targeted exclusively to Intel Arc Pro B60 +
// Qwen3.8-27B. A generic dynamic Jinja2 runtime (AST interpreter) is explicitly
// out of scope to avoid runtime dependencies, memory allocations, and parsing
// latency on the hot inference serving path.
//
// Instead, QwenChatTemplate implements an explicitly verified, compiled C++
// ChatML renderer faithfully implementing the official Qwen3.8 chat_template.jinja
// specification (ChatML role frames, thinking tags, tool interactions, generation prompt).
//
// Fail-Loudly Contract:
// To ensure the compiled renderer never silently drifts from the checkpoint's
// actual template:
// 1. parse_template() validates all structural ChatML tokens (<|im_start|>,
//    <|im_end|>, <think>, </think>, <tool_response>, role conditionals).
// 2. It extracts the dynamic parameters directly from the template:
//    - default reasoning effort ('reasoning_effort|default(...)')
//    - per-effort reasoning instructions ('xhigh', 'low', 'medium')
// 3. If any expected pattern or structural invariant cannot be verified or extracted,
//    load_from_*() FAILS LOUDLY (returns false, populates error_msg, logs to stderr),
//    strictly prohibiting silent fallbacks to guessed hardcoded strings.
// ============================================================================
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
    bool parse_template(std::string* error_msg = nullptr);

    bool is_loaded_{false};
    std::string raw_template_;
    std::string default_reasoning_effort_{"xhigh"};
    bool default_enable_thinking_{true};
    std::unordered_map<std::string, std::string> reasoning_instructions_;
};

} // namespace xinfer::targets::qwen3_8
