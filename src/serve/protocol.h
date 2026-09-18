#pragma once

#include "xinfer/engine.h"
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

// Non-blocking, chunked base64 decoder
bool decode_base64_chunked(std::string_view input,
                           std::vector<uint8_t>& out_bytes,
                           std::string* error_msg = nullptr,
                           size_t chunk_size = 4096);

// Lightweight multipart/form-data parser
struct MultipartPart {
    std::string name;
    std::string filename;
    std::string content_type;
    std::string data;
};

struct MultipartFormData {
    std::vector<MultipartPart> parts;
    const MultipartPart* find_part(const std::string& name) const;
};

bool parse_multipart_form_data(std::string_view body,
                               std::string_view boundary,
                               MultipartFormData& out_form,
                               std::string* error_msg = nullptr);

struct ImagePayload {
    std::vector<uint8_t> raw_bytes;
    int64_t width{0};
    int64_t height{0};
    int64_t channels{3};
    std::string format; // "jpeg", "png", "bmp", "raw"
};

using xinfer::ChatMessage;

struct ChatCompletionRequest {
    std::string model{"qwen3.8-27b"};
    std::vector<ChatMessage> messages;
    std::vector<ImagePayload> images;
    int max_tokens{256};
    float temperature{0.0f};
    float top_p{1.0f};
    bool stream{false};
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
    std::optional<std::string> reasoning_content;
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
    std::string param;

    std::string to_json() const;
};

// Creates standard OpenAI context_length_exceeded error
ApiError make_context_length_exceeded_error(size_t max_seq_len, size_t prompt_tokens, int max_new_tokens);

// High-level request parser
bool parse_chat_completion_request(std::string_view json_str,
                                   ChatCompletionRequest& out_req,
                                   ApiError& out_err);

// High-level multipart/form-data request parser
bool parse_multipart_chat_completion_request(std::string_view body,
                                             std::string_view boundary,
                                             ChatCompletionRequest& out_req,
                                             ApiError& out_err);

// Helper to generate unique chat completion IDs: "chatcmpl-<hex>"
std::string generate_completion_id();

} // namespace xinfer::serve
