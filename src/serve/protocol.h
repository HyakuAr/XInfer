#pragma once

#include <string>
#include <vector>
#include <optional>
#include <memory>
#include <cstdint>
#include <string_view>

namespace xinfer::serve {

// Lightweight JSON DOM for OpenAI schema handling
struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type{Type::Null};

    bool bool_val{false};
    double num_val{0.0};
    std::string str_val;
    std::vector<JsonValue> arr_val;
    std::vector<std::pair<std::string, JsonValue>> obj_val;

    const JsonValue* find(const std::string& key) const;
    std::string get_string(const std::string& key, const std::string& def = "") const;
    int64_t get_int(const std::string& key, int64_t def = 0) const;
    double get_double(const std::string& key, double def = 0.0) const;
    bool get_bool(const std::string& key, bool def = false) const;
};

bool parse_json(std::string_view input, JsonValue& out, std::string& error_msg);
void escape_json_string(std::string& out, std::string_view str);

struct ChatMessage {
    std::string role;    // "system", "user", "assistant"
    std::string content;
};

struct ChatCompletionRequest {
    std::string model{"qwen3.8-27b"};
    std::vector<ChatMessage> messages;
    int max_tokens{256};
    float temperature{0.0f};
    bool stream{false};

    // Formats conversation messages into standard Qwen chat template prompt
    std::string format_prompt() const;
};

struct ChatChoice {
    int index{0};
    ChatMessage message;
    std::string finish_reason{"stop"};
};

struct UsageInfo {
    size_t prompt_tokens{0};
    size_t completion_tokens{0};
    size_t total_tokens{0};
};

struct ChatCompletionResponse {
    std::string id;
    std::string object{"chat.completion"};
    int64_t created{0};
    std::string model{"qwen3.8-27b"};
    std::vector<ChatChoice> choices;
    UsageInfo usage;

    std::string to_json() const;
};

struct ChunkDelta {
    std::optional<std::string> role;
    std::optional<std::string> content;
};

struct ChunkChoice {
    int index{0};
    ChunkDelta delta;
    std::optional<std::string> finish_reason;
};

struct ChatCompletionChunk {
    std::string id;
    std::string object{"chat.completion.chunk"};
    int64_t created{0};
    std::string model{"qwen3.8-27b"};
    std::vector<ChunkChoice> choices;

    std::string to_sse_event() const;
};

struct ApiError {
    int status_code{400};
    std::string message;
    std::string type{"invalid_request_error"};
    std::string code{"bad_request"};

    std::string to_json() const;
};

// High-level request parser
bool parse_chat_completion_request(std::string_view json_str,
                                   ChatCompletionRequest& out_req,
                                   ApiError& out_err);

// Helper to generate unique chat completion IDs: "chatcmpl-<hex>"
std::string generate_completion_id();

} // namespace xinfer::serve
