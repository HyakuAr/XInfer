#include "protocol.h"
#include <sstream>
#include <iomanip>
#include <random>
#include <chrono>
#include <cctype>

namespace xinfer::serve {

void escape_json_string(std::string& out, std::string_view str) {
    out.push_back('"');
    for (char c : str) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(c));
                    out += buf;
                } else {
                    out.push_back(c);
                }
                break;
        }
    }
    out.push_back('"');
}

const JsonValue* JsonValue::find(const std::string& key) const {
    if (type != Type::Object) return nullptr;
    for (const auto& [k, v] : obj_val) {
        if (k == key) return &v;
    }
    return nullptr;
}

std::string JsonValue::get_string(const std::string& key, const std::string& def) const {
    const auto* v = find(key);
    if (v && v->type == Type::String) return v->str_val;
    return def;
}

int64_t JsonValue::get_int(const std::string& key, int64_t def) const {
    const auto* v = find(key);
    if (v && v->type == Type::Number) return static_cast<int64_t>(v->num_val);
    return def;
}

double JsonValue::get_double(const std::string& key, double def) const {
    const auto* v = find(key);
    if (v && v->type == Type::Number) return v->num_val;
    return def;
}

bool JsonValue::get_bool(const std::string& key, bool def) const {
    const auto* v = find(key);
    if (v && v->type == Type::Bool) return v->bool_val;
    return def;
}

namespace {

class Parser {
public:
    explicit Parser(std::string_view input) : in_(input), pos_(0) {}

    bool parse(JsonValue& out, std::string& err) {
        skip_whitespace();
        if (pos_ >= in_.size()) {
            err = "Empty JSON input";
            return false;
        }
        if (!parse_value(out, err)) return false;
        skip_whitespace();
        if (pos_ < in_.size()) {
            err = "Unexpected trailing characters after JSON";
            return false;
        }
        return true;
    }

private:
    std::string_view in_;
    size_t pos_{0};

    char peek() const { return pos_ < in_.size() ? in_[pos_] : '\0'; }
    char get() { return pos_ < in_.size() ? in_[pos_++] : '\0'; }

    void skip_whitespace() {
        while (pos_ < in_.size()) {
            char c = in_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                pos_++;
            } else {
                break;
            }
        }
    }

    bool consume(char expected) {
        skip_whitespace();
        if (peek() == expected) {
            pos_++;
            return true;
        }
        return false;
    }

    bool parse_value(JsonValue& out, std::string& err) {
        skip_whitespace();
        char c = peek();
        if (c == '{') return parse_object(out, err);
        if (c == '[') return parse_array(out, err);
        if (c == '"') return parse_string(out.str_val, err) ? (out.type = JsonValue::Type::String, true) : false;
        if (c == 't' || c == 'f') return parse_bool(out.bool_val, err) ? (out.type = JsonValue::Type::Bool, true) : false;
        if (c == 'n') return parse_null(err) ? (out.type = JsonValue::Type::Null, true) : false;
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
            return parse_number(out.num_val, err) ? (out.type = JsonValue::Type::Number, true) : false;
        }
        err = std::string("Unexpected character in JSON: '") + c + "'";
        return false;
    }

    bool parse_object(JsonValue& out, std::string& err) {
        if (!consume('{')) { err = "Expected '{'"; return false; }
        out.type = JsonValue::Type::Object;
        out.obj_val.clear();

        skip_whitespace();
        if (consume('}')) return true;

        while (true) {
            skip_whitespace();
            if (peek() != '"') {
                err = "Expected string key in object";
                return false;
            }
            std::string key;
            if (!parse_string(key, err)) return false;

            if (!consume(':')) {
                err = "Expected ':' after object key";
                return false;
            }

            JsonValue val;
            if (!parse_value(val, err)) return false;
            out.obj_val.emplace_back(std::move(key), std::move(val));

            skip_whitespace();
            if (consume('}')) return true;
            if (!consume(',')) {
                err = "Expected ',' or '}' in object";
                return false;
            }
        }
    }

    bool parse_array(JsonValue& out, std::string& err) {
        if (!consume('[')) { err = "Expected '['"; return false; }
        out.type = JsonValue::Type::Array;
        out.arr_val.clear();

        skip_whitespace();
        if (consume(']')) return true;

        while (true) {
            JsonValue val;
            if (!parse_value(val, err)) return false;
            out.arr_val.push_back(std::move(val));

            skip_whitespace();
            if (consume(']')) return true;
            if (!consume(',')) {
                err = "Expected ',' or ']' in array";
                return false;
            }
        }
    }

    bool parse_string(std::string& out, std::string& err) {
        if (get() != '"') { err = "Expected '\"'"; return false; }
        out.clear();

        while (pos_ < in_.size()) {
            char c = get();
            if (c == '"') return true;
            if (c == '\\') {
                if (pos_ >= in_.size()) { err = "Unexpected EOF in string escape"; return false; }
                char esc = get();
                switch (esc) {
                    case '"':  out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/':  out.push_back('/'); break;
                    case 'b':  out.push_back('\b'); break;
                    case 'f':  out.push_back('\f'); break;
                    case 'n':  out.push_back('\n'); break;
                    case 'r':  out.push_back('\r'); break;
                    case 't':  out.push_back('\t'); break;
                    case 'u': {
                        if (pos_ + 4 > in_.size()) { err = "Invalid unicode escape"; return false; }
                        std::string hex_str(in_.substr(pos_, 4));
                        pos_ += 4;
                        try {
                            uint32_t cp = std::stoul(hex_str, nullptr, 16);
                            if (cp < 0x80) {
                                out.push_back(static_cast<char>(cp));
                            } else if (cp < 0x800) {
                                out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                            } else {
                                out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                                out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                            }
                        } catch (...) {
                            err = "Failed to parse unicode escape";
                            return false;
                        }
                        break;
                    }
                    default:
                        out.push_back(esc);
                        break;
                }
            } else {
                out.push_back(c);
            }
        }
        err = "Unterminated string in JSON";
        return false;
    }

    bool parse_number(double& out, std::string& err) {
        size_t start = pos_;
        if (peek() == '-') pos_++;
        if (pos_ >= in_.size() || !std::isdigit(static_cast<unsigned char>(in_[pos_]))) {
            err = "Invalid number format";
            return false;
        }
        while (pos_ < in_.size() && std::isdigit(static_cast<unsigned char>(in_[pos_]))) pos_++;
        if (pos_ < in_.size() && in_[pos_] == '.') {
            pos_++;
            while (pos_ < in_.size() && std::isdigit(static_cast<unsigned char>(in_[pos_]))) pos_++;
        }
        if (pos_ < in_.size() && (in_[pos_] == 'e' || in_[pos_] == 'E')) {
            pos_++;
            if (pos_ < in_.size() && (in_[pos_] == '+' || in_[pos_] == '-')) pos_++;
            while (pos_ < in_.size() && std::isdigit(static_cast<unsigned char>(in_[pos_]))) pos_++;
        }
        std::string num_str(in_.substr(start, pos_ - start));
        try {
            out = std::stod(num_str);
            return true;
        } catch (...) {
            err = "Failed to convert number: " + num_str;
            return false;
        }
    }

    bool parse_bool(bool& out, std::string& err) {
        if (in_.substr(pos_, 4) == "true") {
            pos_ += 4;
            out = true;
            return true;
        }
        if (in_.substr(pos_, 5) == "false") {
            pos_ += 5;
            out = false;
            return true;
        }
        err = "Invalid boolean literal";
        return false;
    }

    bool parse_null(std::string& err) {
        if (in_.substr(pos_, 4) == "null") {
            pos_ += 4;
            return true;
        }
        err = "Invalid null literal";
        return false;
    }
};

} // anonymous namespace

bool parse_json(std::string_view input, JsonValue& out, std::string& error_msg) {
    Parser parser(input);
    return parser.parse(out, error_msg);
}

std::string ChatCompletionRequest::format_prompt() const {
    std::string prompt;
    for (const auto& msg : messages) {
        prompt += "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n";
    }
    prompt += "<|im_start|>assistant\n";
    return prompt;
}

std::string generate_completion_id() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    uint64_t r1 = rng();
    uint64_t r2 = rng();

    std::ostringstream ss;
    ss << "chatcmpl-" << std::hex << std::setfill('0')
       << std::setw(16) << r1 << std::setw(8) << (r2 & 0xFFFFFFFF);
    return ss.str();
}

std::string ChatCompletionResponse::to_json() const {
    std::string out;
    out.reserve(512 + (choices.empty() ? 0 : choices[0].message.content.size() * 2));
    out += "{";

    out += "\"id\":";
    escape_json_string(out, id);
    out += ",\"object\":\"chat.completion\"";
    out += ",\"created\":" + std::to_string(created);
    out += ",\"model\":";
    escape_json_string(out, model);

    out += ",\"choices\":[";
    for (size_t i = 0; i < choices.size(); ++i) {
        if (i > 0) out += ",";
        out += "{";
        out += "\"index\":" + std::to_string(choices[i].index);
        out += ",\"message\":{\"role\":";
        escape_json_string(out, choices[i].message.role);
        out += ",\"content\":";
        escape_json_string(out, choices[i].message.content);
        out += "}";
        out += ",\"finish_reason\":";
        escape_json_string(out, choices[i].finish_reason);
        out += "}";
    }
    out += "]";

    out += ",\"usage\":{";
    out += "\"prompt_tokens\":" + std::to_string(usage.prompt_tokens);
    out += ",\"completion_tokens\":" + std::to_string(usage.completion_tokens);
    out += ",\"total_tokens\":" + std::to_string(usage.total_tokens);
    out += "}}";

    return out;
}

std::string ChatCompletionChunk::to_sse_event() const {
    std::string json;
    json += "{";
    json += "\"id\":";
    escape_json_string(json, id);
    json += ",\"object\":\"chat.completion.chunk\"";
    json += ",\"created\":" + std::to_string(created);
    json += ",\"model\":";
    escape_json_string(json, model);

    json += ",\"choices\":[";
    for (size_t i = 0; i < choices.size(); ++i) {
        if (i > 0) json += ",";
        json += "{";
        json += "\"index\":" + std::to_string(choices[i].index);
        json += ",\"delta\":{";
        bool first = true;
        if (choices[i].delta.role.has_value()) {
            json += "\"role\":";
            escape_json_string(json, *choices[i].delta.role);
            first = false;
        }
        if (choices[i].delta.content.has_value()) {
            if (!first) json += ",";
            json += "\"content\":";
            escape_json_string(json, *choices[i].delta.content);
        }
        json += "}";
        if (choices[i].finish_reason.has_value()) {
            json += ",\"finish_reason\":";
            escape_json_string(json, *choices[i].finish_reason);
        } else {
            json += ",\"finish_reason\":null";
        }
        json += "}";
    }
    json += "]}";

    return "data: " + json + "\n\n";
}

std::string ApiError::to_json() const {
    std::string out;
    out += "{\"error\":{";
    out += "\"message\":";
    escape_json_string(out, message);
    out += ",\"type\":";
    escape_json_string(out, type);
    out += ",\"param\":";
    if (param.empty()) {
        out += "null";
    } else {
        escape_json_string(out, param);
    }
    out += ",\"code\":";
    escape_json_string(out, code);
    out += "}}";
    return out;
}

ApiError make_context_length_exceeded_error(size_t max_seq_len, size_t prompt_tokens, int max_new_tokens) {
    ApiError err;
    err.status_code = 400;
    err.type = "invalid_request_error";
    err.code = "context_length_exceeded";
    err.param = "messages";
    if (prompt_tokens > max_seq_len) {
        err.message = "This model's maximum context length is " + std::to_string(max_seq_len) +
                      " tokens. However, your messages resulted in " + std::to_string(prompt_tokens) +
                      " tokens. Please reduce the length of the messages.";
    } else {
        size_t total_requested = prompt_tokens + static_cast<size_t>(std::max(0, max_new_tokens));
        err.message = "This model's maximum context length is " + std::to_string(max_seq_len) +
                      " tokens. However, you requested " + std::to_string(total_requested) +
                      " tokens (" + std::to_string(prompt_tokens) + " in the messages, " +
                      std::to_string(max_new_tokens) + " in the completion). Please reduce the length of the messages or completion.";
    }
    return err;
}

bool parse_chat_completion_request(std::string_view json_str,
                                   ChatCompletionRequest& out_req,
                                   ApiError& out_err) {
    JsonValue root;
    std::string parse_err;
    if (!parse_json(json_str, root, parse_err)) {
        out_err.status_code = 400;
        out_err.type = "invalid_request_error";
        out_err.code = "parse_error";
        out_err.message = "Failed to parse JSON request body: " + parse_err;
        return false;
    }

    if (root.type != JsonValue::Type::Object) {
        out_err.status_code = 400;
        out_err.type = "invalid_request_error";
        out_err.code = "bad_request";
        out_err.message = "Request body must be a JSON object";
        return false;
    }

    out_req.model = root.get_string("model", "qwen3.8-27b");
    out_req.max_tokens = static_cast<int>(root.get_int("max_tokens", 256));
    out_req.temperature = static_cast<float>(root.get_double("temperature", 0.0));
    out_req.stream = root.get_bool("stream", false);

    const auto* msgs = root.find("messages");
    if (!msgs || msgs->type != JsonValue::Type::Array || msgs->arr_val.empty()) {
        out_err.status_code = 400;
        out_err.type = "invalid_request_error";
        out_err.code = "missing_required_field";
        out_err.message = "Field 'messages' is required and must be a non-empty array";
        return false;
    }

    out_req.messages.clear();
    for (size_t i = 0; i < msgs->arr_val.size(); ++i) {
        const auto& item = msgs->arr_val[i];
        if (item.type != JsonValue::Type::Object) {
            out_err.status_code = 400;
            out_err.type = "invalid_request_error";
            out_err.code = "invalid_message";
            out_err.message = "Each item in 'messages' must be a JSON object";
            return false;
        }

        ChatMessage msg;
        msg.role = item.get_string("role", "");
        msg.content = item.get_string("content", "");

        if (msg.role.empty()) {
            out_err.status_code = 400;
            out_err.type = "invalid_request_error";
            out_err.code = "missing_role";
            out_err.message = "Message at index " + std::to_string(i) + " is missing required field 'role'";
            return false;
        }

        out_req.messages.push_back(std::move(msg));
    }

    return true;
}

} // namespace xinfer::serve
