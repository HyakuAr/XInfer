#include "chat_template.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace xinfer::targets::qwen3_8 {

namespace {

std::string trim_whitespace(std::string_view str) {
    size_t start = 0;
    while (start < str.size() && (str[start] == ' ' || str[start] == '\t' || str[start] == '\r' || str[start] == '\n')) {
        ++start;
    }
    size_t end = str.size();
    while (end > start && (str[end - 1] == ' ' || str[end - 1] == '\t' || str[end - 1] == '\r' || str[end - 1] == '\n')) {
        --end;
    }
    return std::string(str.substr(start, end - start));
}

bool starts_with(std::string_view str, std::string_view prefix) {
    return str.size() >= prefix.size() && str.substr(0, prefix.size()) == prefix;
}

bool ends_with(std::string_view str, std::string_view suffix) {
    return str.size() >= suffix.size() && str.substr(str.size() - suffix.size()) == suffix;
}

} // anonymous namespace

QwenChatTemplate::QwenChatTemplate() {
    init_defaults();
}

void QwenChatTemplate::init_defaults() {
    default_reasoning_effort_ = "xhigh";
    default_enable_thinking_ = true;
    reasoning_instructions_["xhigh"] =
        "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
        "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
    reasoning_instructions_["low"] =
        "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
        "without unnecessary elaboration.";
    reasoning_instructions_["medium"] = "";
}

void QwenChatTemplate::parse_template() {
    if (raw_template_.empty()) return;

    // 1. Parse default reasoning effort: reasoning_effort|default('...')
    std::string needle = "reasoning_effort|default(";
    size_t p = raw_template_.find(needle);
    if (p != std::string::npos) {
        p += needle.size();
        while (p < raw_template_.size() && (raw_template_[p] == ' ' || raw_template_[p] == '\'' || raw_template_[p] == '"')) {
            char quote = raw_template_[p];
            if (quote == '\'' || quote == '"') {
                size_t q_end = raw_template_.find(quote, p + 1);
                if (q_end != std::string::npos) {
                    default_reasoning_effort_ = raw_template_.substr(p + 1, q_end - (p + 1));
                }
                break;
            }
            ++p;
        }
    }

    // 2. Parse reasoning instructions for each effort level
    std::vector<std::string> known_efforts = {"xhigh", "low", "medium"};
    for (const auto& effort : known_efforts) {
        std::string effort_needle = "resolved_reasoning_effort == '" + effort + "'";
        size_t e_pos = raw_template_.find(effort_needle);
        if (e_pos == std::string::npos) {
            effort_needle = "resolved_reasoning_effort == \"" + effort + "\"";
            e_pos = raw_template_.find(effort_needle);
        }

        if (e_pos != std::string::npos) {
            std::string set_needle = "reasoning_instructions = '";
            size_t s_pos = raw_template_.find(set_needle, e_pos);
            char quote_char = '\'';
            if (s_pos == std::string::npos || s_pos > e_pos + 400) {
                set_needle = "reasoning_instructions = \"";
                s_pos = raw_template_.find(set_needle, e_pos);
                quote_char = '"';
            }

            if (s_pos != std::string::npos && s_pos < e_pos + 400) {
                size_t val_start = s_pos + set_needle.size();
                size_t val_end = val_start;
                while (val_end < raw_template_.size()) {
                    if (raw_template_[val_end] == quote_char && raw_template_[val_end - 1] != '\\') {
                        break;
                    }
                    ++val_end;
                }
                if (val_end < raw_template_.size()) {
                    reasoning_instructions_[effort] = raw_template_.substr(val_start, val_end - val_start);
                }
            }
        }
    }
}

bool QwenChatTemplate::load_from_string(const std::string& template_str, std::string* error_msg) {
    if (template_str.empty()) {
        if (error_msg) *error_msg = "Empty chat template string provided";
        return false;
    }
    raw_template_ = template_str;
    parse_template();
    is_loaded_ = true;
    return true;
}

bool QwenChatTemplate::load_from_buffer(const void* data, size_t size, std::string* error_msg) {
    if (!data || size == 0) {
        if (error_msg) *error_msg = "Empty chat template buffer provided";
        return false;
    }
    std::string str(static_cast<const char*>(data), size);
    return load_from_string(str, error_msg);
}

bool QwenChatTemplate::load_from_file(const std::string& path, std::string* error_msg) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        if (error_msg) *error_msg = "Could not open chat template file: " + path;
        return false;
    }
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return load_from_string(content, error_msg);
}

std::string QwenChatTemplate::get_reasoning_instructions(const std::string& effort) const {
    auto it = reasoning_instructions_.find(effort);
    return (it != reasoning_instructions_.end()) ? it->second : "";
}

std::string QwenChatTemplate::render(const std::vector<ChatMessage>& messages,
                                     const ChatTemplateOptions& options,
                                     std::string* error_msg) const {
    if (messages.empty()) {
        if (error_msg) *error_msg = "No messages provided.";
        return "";
    }

    // Validate system message position: system message must be at the beginning
    for (size_t i = 1; i < messages.size(); ++i) {
        if (messages[i].role == "system") {
            if (error_msg) *error_msg = "System message must be at the beginning.";
            return "";
        }
    }

    // Determine last user query index for multi-step tool interactions
    size_t last_query_index = messages.size() - 1;
    bool found_user_query = false;
    for (size_t idx = messages.size(); idx > 0; --idx) {
        size_t i = idx - 1;
        if (messages[i].role == "user") {
            std::string c = trim_whitespace(messages[i].content);
            if (!(starts_with(c, "<tool_response>") && ends_with(c, "</tool_response>"))) {
                last_query_index = i;
                found_user_query = true;
                break;
            }
        }
    }

    if (!found_user_query) {
        if (error_msg) *error_msg = "No user query found in messages.";
        return "";
    }

    // Resolve reasoning effort
    std::string effort = options.reasoning_effort.empty() ? default_reasoning_effort_ : options.reasoning_effort;
    if (effort != "xhigh" && effort != "medium" && effort != "low") {
        if (error_msg) {
            *error_msg = "Unexpected reasoning effort " + effort + ". Supported types are xhigh (default), medium, and low.";
        }
        return "";
    }

    std::string reasoning_instructions;
    if (options.enable_thinking) {
        reasoning_instructions = get_reasoning_instructions(effort);
    }

    std::string out;
    out.reserve(512);

    // Render system prompt
    if (messages[0].role == "system") {
        std::string content = trim_whitespace(messages[0].content);
        if (!content.empty()) {
            out += "<|im_start|>system\n";
            if (!reasoning_instructions.empty()) {
                out += reasoning_instructions + "\n\n";
            }
            out += content + "<|im_end|>\n";
        } else if (!reasoning_instructions.empty()) {
            out += "<|im_start|>system\n" + reasoning_instructions + "<|im_end|>\n";
        }
    } else if (!reasoning_instructions.empty()) {
        out += "<|im_start|>system\n" + reasoning_instructions + "<|im_end|>\n";
    }

    // Render conversation messages
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& msg = messages[i];
        std::string content = trim_whitespace(msg.content);

        if (msg.role == "system") {
            // Already rendered in system prompt block
            continue;
        } else if (msg.role == "user") {
            out += "<|im_start|>user\n" + content + "<|im_end|>\n";
        } else if (msg.role == "assistant") {
            std::string reasoning = trim_whitespace(msg.reasoning_content);
            out += "<|im_start|>assistant\n";
            if (options.preserve_thinking || i > last_query_index) {
                out += "<think>\n" + reasoning + "\n</think>\n\n" + content;
            } else {
                out += content;
            }
            out += "<|im_end|>\n";
        } else if (msg.role == "tool") {
            if (i > 0 && messages[i - 1].role != "tool") {
                out += "<|im_start|>user";
            }
            out += "\n<tool_response>\n" + content + "\n</tool_response>";
            if (i + 1 == messages.size() || messages[i + 1].role != "tool") {
                out += "<|im_end|>\n";
            }
        } else {
            if (error_msg) *error_msg = "Unexpected message role: " + msg.role;
            return "";
        }
    }

    // Generation prompt
    if (options.add_generation_prompt) {
        out += "<|im_start|>assistant\n";
        if (!options.enable_thinking) {
            out += "<think>\n\n</think>\n\n";
        } else {
            out += "<think>\n";
        }
    }

    return out;
}

std::string QwenChatTemplate::render(const std::string& user_prompt,
                                     const std::string& system_prompt,
                                     const ChatTemplateOptions& options,
                                     std::string* error_msg) const {
    std::vector<ChatMessage> msgs;
    if (!system_prompt.empty()) {
        msgs.push_back(ChatMessage{.role = "system", .content = system_prompt, .reasoning_content = ""});
    }
    msgs.push_back(ChatMessage{.role = "user", .content = user_prompt, .reasoning_content = ""});
    return render(msgs, options, error_msg);
}

} // namespace xinfer::targets::qwen3_8
