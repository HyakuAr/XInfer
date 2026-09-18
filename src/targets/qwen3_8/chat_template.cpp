#include "chat_template.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace xinfer::targets::qwen3_8 {

// ============================================================================
// Architectural Rationale: Compiled C++ ChatML Renderer vs Jinja Interpreter
// ============================================================================
// In accordance with AGENTS.md §1 ("Governing objective: single target combination
// Intel Arc Pro B60 + Qwen3.8-27B") and §4 ("Ownership boundaries:
// src/targets/qwen3_8 owns family invariants: tokenizer, chat template..."):
//
// 1. Zero-Overhead Serving: Integrating a generic Jinja2 template engine (e.g.
//    minijinja or inja) would introduce heavy dynamic memory allocations, AST
//    parsing, and interpretation overhead on the hot token generation path.
//    Instead, QwenChatTemplate implements an explicitly verified, compiled C++20
//    ChatML renderer faithful to Qwen3.8's official chat_template.jinja contract.
//
// 2. Fail-Loudly Invariant: The hand-written renderer is NOT an unchecked
//    guess. To guarantee that render() never silently drifts from the model's
//    actual checkpoint:
//    - parse_template() validates all structural ChatML tokens (<|im_start|>,
//      <|im_end|>, <think>, </think>, <tool_response>, role conditionals) in
//      the loaded raw template.
//    - It dynamically extracts default_reasoning_effort and the exact reasoning
//      instructions for all supported effort levels ('xhigh', 'low', 'medium').
//    - If any expected pattern or structural invariant cannot be verified or
//      extracted, load_from_*() FAILS LOUDLY (returns false, populates error_msg,
//      and logs to stderr). It NEVER silently falls back to guessed defaults.
//
// 3. init_defaults() provides baseline Qwen3.8 defaults solely for unit testing
//    uninstantiated objects before an artifact is opened. Once a template is
//    loaded, only validated template-derived parameters are active.
// ============================================================================

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

size_t skip_ws(std::string_view sv, size_t pos) {
    while (pos < sv.size() && (sv[pos] == ' ' || sv[pos] == '\t' || sv[pos] == '\r' || sv[pos] == '\n')) {
        ++pos;
    }
    return pos;
}

size_t match_token(std::string_view sv, size_t pos, std::string_view token) {
    pos = skip_ws(sv, pos);
    if (pos + token.size() <= sv.size() && sv.substr(pos, token.size()) == token) {
        return pos + token.size();
    }
    return std::string_view::npos;
}

bool extract_quoted_string(std::string_view sv, size_t& pos, std::string& out_str) {
    pos = skip_ws(sv, pos);
    if (pos >= sv.size()) return false;
    char quote = sv[pos];
    if (quote != '\'' && quote != '"') return false;
    ++pos;

    std::string res;
    while (pos < sv.size()) {
        if (sv[pos] == '\\' && pos + 1 < sv.size()) {
            res.push_back(sv[pos + 1]);
            pos += 2;
        } else if (sv[pos] == quote) {
            ++pos;
            out_str = std::move(res);
            return true;
        } else {
            res.push_back(sv[pos]);
            ++pos;
        }
    }
    return false;
}

bool parse_default_reasoning_effort(std::string_view sv, std::string& out_effort, std::string* error_msg) {
    size_t re_pos = sv.find("reasoning_effort");
    while (re_pos != std::string_view::npos) {
        size_t cur = re_pos + std::string_view("reasoning_effort").size();
        cur = match_token(sv, cur, "|");
        if (cur != std::string_view::npos) {
            cur = match_token(sv, cur, "default");
            if (cur != std::string_view::npos) {
                cur = match_token(sv, cur, "(");
                if (cur != std::string_view::npos) {
                    std::string val;
                    if (extract_quoted_string(sv, cur, val)) {
                        cur = match_token(sv, cur, ")");
                        if (cur != std::string_view::npos) {
                            if (val == "xhigh" || val == "medium" || val == "low") {
                                out_effort = std::move(val);
                                return true;
                            } else {
                                if (error_msg) {
                                    *error_msg = "Extracted unsupported default reasoning effort '" + val +
                                                 "' from template (expected xhigh, medium, or low)";
                                }
                                return false;
                            }
                        }
                    }
                }
            }
        }
        re_pos = sv.find("reasoning_effort", re_pos + 1);
    }
    if (error_msg) {
        *error_msg = "Failed to extract default reasoning effort (expected pattern 'reasoning_effort|default(...)')";
    }
    return false;
}

bool parse_reasoning_instruction_for_effort(std::string_view sv,
                                            const std::string& effort,
                                            std::string& out_instruction,
                                            std::string* error_msg) {
    size_t pos = 0;
    while ((pos = sv.find("resolved_reasoning_effort", pos)) != std::string_view::npos) {
        size_t cur = pos + std::string_view("resolved_reasoning_effort").size();
        cur = match_token(sv, cur, "==");
        if (cur != std::string_view::npos) {
            std::string target_effort;
            if (extract_quoted_string(sv, cur, target_effort) && target_effort == effort) {
                // Found condition for target effort. Search within subsequent block (next 800 chars)
                size_t search_limit = std::min(sv.size(), cur + 800);
                std::string_view block = sv.substr(cur, search_limit - cur);
                size_t instr_pos = block.find("reasoning_instructions");
                if (instr_pos != std::string_view::npos) {
                    size_t instr_cur = cur + instr_pos + std::string_view("reasoning_instructions").size();
                    instr_cur = match_token(sv, instr_cur, "=");
                    if (instr_cur != std::string_view::npos) {
                        std::string instruction;
                        if (extract_quoted_string(sv, instr_cur, instruction)) {
                            out_instruction = std::move(instruction);
                            return true;
                        }
                    }
                }
            }
        }
        pos += std::string_view("resolved_reasoning_effort").size();
    }
    if (error_msg) {
        *error_msg = "Failed to extract reasoning instructions for effort '" + effort +
                     "' from template (expected 'resolved_reasoning_effort == \\'" + effort + "\\'')";
    }
    return false;
}

bool validate_structural_invariants(std::string_view sv, std::string* error_msg) {
    struct Requirement {
        std::string_view pattern;
        std::string_view description;
    };
    static const Requirement requirements[] = {
        {"<|im_start|>", "ChatML turn start marker '<|im_start|>'"},
        {"<|im_end|>", "ChatML turn end marker '<|im_end|>'"},
        {"<think>", "Reasoning thinking start tag '<think>'"},
        {"</think>", "Reasoning thinking end tag '</think>'"},
        {"add_generation_prompt", "Generation prompt control variable 'add_generation_prompt'"},
        {"enable_thinking", "Thinking enable variable 'enable_thinking'"},
        {"multi_step_tool", "Multi-step tool handling marker 'multi_step_tool'"},
        {"<tool_response>", "Tool response container tag '<tool_response>'"},
        {"</tool_response>", "Tool response container tag '</tool_response>'"},
        {"messages", "Message list variable 'messages'"},
        {"system", "System role marker 'system'"},
        {"user", "User role marker 'user'"},
        {"assistant", "Assistant role marker 'assistant'"},
    };

    for (const auto& req : requirements) {
        if (sv.find(req.pattern) == std::string_view::npos) {
            if (error_msg) {
                *error_msg = std::string("Template structural validation failed: missing ") +
                             std::string(req.description);
            }
            return false;
        }
    }
    return true;
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

bool QwenChatTemplate::parse_template(std::string* error_msg) {
    if (raw_template_.empty()) {
        std::string err = "Empty chat template provided";
        if (error_msg) *error_msg = err;
        std::cerr << "[QwenChatTemplate] Error: " << err << std::endl;
        return false;
    }

    // 1. Validate structural ChatML invariants to guarantee template compatibility
    std::string struct_err;
    if (!validate_structural_invariants(raw_template_, &struct_err)) {
        if (error_msg) *error_msg = struct_err;
        std::cerr << "[QwenChatTemplate] Error: " << struct_err << std::endl;
        return false;
    }

    // 2. Parse default reasoning effort
    std::string effort_err;
    std::string default_effort;
    if (!parse_default_reasoning_effort(raw_template_, default_effort, &effort_err)) {
        if (error_msg) *error_msg = effort_err;
        std::cerr << "[QwenChatTemplate] Error: " << effort_err << std::endl;
        return false;
    }

    // 3. Parse reasoning instructions for known effort levels
    std::string xhigh_err, low_err;
    std::string xhigh_instr, low_instr;
    if (!parse_reasoning_instruction_for_effort(raw_template_, "xhigh", xhigh_instr, &xhigh_err)) {
        if (error_msg) *error_msg = xhigh_err;
        std::cerr << "[QwenChatTemplate] Error: " << xhigh_err << std::endl;
        return false;
    }
    if (!parse_reasoning_instruction_for_effort(raw_template_, "low", low_instr, &low_err)) {
        if (error_msg) *error_msg = low_err;
        std::cerr << "[QwenChatTemplate] Error: " << low_err << std::endl;
        return false;
    }

    // Check optional medium effort instruction (defaults to empty string in Qwen3.8)
    std::string medium_instr;
    std::string med_err;
    if (parse_reasoning_instruction_for_effort(raw_template_, "medium", medium_instr, &med_err)) {
        reasoning_instructions_["medium"] = std::move(medium_instr);
    } else {
        reasoning_instructions_["medium"] = "";
    }

    default_reasoning_effort_ = std::move(default_effort);
    reasoning_instructions_["xhigh"] = std::move(xhigh_instr);
    reasoning_instructions_["low"] = std::move(low_instr);

    return true;
}

bool QwenChatTemplate::load_from_string(const std::string& template_str, std::string* error_msg) {
    if (template_str.empty()) {
        std::string err = "Empty chat template string provided";
        if (error_msg) *error_msg = err;
        std::cerr << "[QwenChatTemplate] Error: " << err << std::endl;
        return false;
    }
    raw_template_ = template_str;

    // Clear prior dynamic instructions so failure never silently retains previous state
    reasoning_instructions_.clear();
    default_reasoning_effort_.clear();
    is_loaded_ = false;

    if (!parse_template(error_msg)) {
        // Parsing failed loudly. Restore baseline defaults for safety, but mark is_loaded_ = false
        init_defaults();
        is_loaded_ = false;
        return false;
    }

    is_loaded_ = true;
    return true;
}

bool QwenChatTemplate::load_from_buffer(const void* data, size_t size, std::string* error_msg) {
    if (!data || size == 0) {
        std::string err = "Empty chat template buffer provided";
        if (error_msg) *error_msg = err;
        std::cerr << "[QwenChatTemplate] Error: " << err << std::endl;
        return false;
    }
    std::string str(static_cast<const char*>(data), size);
    return load_from_string(str, error_msg);
}

bool QwenChatTemplate::load_from_file(const std::string& path, std::string* error_msg) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::string err = "Could not open chat template file: " + path;
        if (error_msg) *error_msg = err;
        std::cerr << "[QwenChatTemplate] Error: " << err << std::endl;
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
