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
                        for (char h : hex_str) {
                            if (!std::isxdigit(static_cast<unsigned char>(h))) {
                                err = "Invalid hex digit in unicode escape";
                                return false;
                            }
                        }
                        pos_ += 4;
                        try {
                            uint32_t cp = std::stoul(hex_str, nullptr, 16);
                            if (cp >= 0xD800 && cp <= 0xDBFF) {
                                // High surrogate - check for following low surrogate \uXXXX
                                if (pos_ + 6 <= in_.size() && in_[pos_] == '\\' && in_[pos_ + 1] == 'u') {
                                    std::string low_hex(in_.substr(pos_ + 2, 4));
                                    for (char h : low_hex) {
                                        if (!std::isxdigit(static_cast<unsigned char>(h))) {
                                            err = "Invalid hex digit in low surrogate";
                                            return false;
                                        }
                                    }
                                    uint32_t low = std::stoul(low_hex, nullptr, 16);
                                    if (low >= 0xDC00 && low <= 0xDFFF) {
                                        pos_ += 6;
                                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                                    } else {
                                        err = "Invalid UTF-16 low surrogate in unicode escape";
                                        return false;
                                    }
                                } else {
                                    err = "Expected UTF-16 low surrogate after high surrogate";
                                    return false;
                                }
                            } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                                err = "Unpaired UTF-16 low surrogate in unicode escape";
                                return false;
                            }

                            if (cp < 0x80) {
                                out.push_back(static_cast<char>(cp));
                            } else if (cp < 0x800) {
                                out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                            } else if (cp < 0x10000) {
                                out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                                out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                            } else if (cp <= 0x10FFFF) {
                                out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
                                out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
                                out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                            } else {
                                err = "Unicode code point out of range";
                                return false;
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
        if (!choices[i].message.reasoning_content.empty()) {
            out += ",\"reasoning_content\":";
            escape_json_string(out, choices[i].message.reasoning_content);
        }
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
            first = false;
        }
        if (choices[i].delta.reasoning_content.has_value()) {
            if (!first) json += ",";
            json += "\"reasoning_content\":";
            escape_json_string(json, *choices[i].delta.reasoning_content);
            first = false;
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

static const signed char kBase64Table[256] = {
    -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,62, -1,-1,-1,63,
    52,53,54,55, 56,57,58,59, 60,61,-1,-1, -1,-1,-1,-1,
    -1, 0, 1, 2,  3, 4, 5, 6,  7, 8, 9,10, 11,12,13,14,
    15,16,17,18, 19,20,21,22, 23,24,25,-1, -1,-1,-1,-1,
    -1,26,27,28, 29,30,31,32, 33,34,35,36, 37,38,39,40,
    41,42,43,44, 45,46,47,48, 49,50,51,-1, -1,-1,-1,-1,
    -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1
};

bool decode_base64_chunked(std::string_view input,
                           std::vector<uint8_t>& out_bytes,
                           std::string* error_msg,
                           size_t chunk_size) {
    // Strip data URI prefix if present (e.g. "data:image/jpeg;base64,")
    size_t comma_pos = input.find(',');
    if (comma_pos != std::string_view::npos && input.substr(0, comma_pos).find("base64") != std::string_view::npos) {
        input = input.substr(comma_pos + 1);
    }

    out_bytes.clear();
    if (chunk_size == 0) chunk_size = 4096;
    out_bytes.reserve((input.size() / 4) * 3);

    uint32_t buf = 0;
    int bits = 0;

    for (size_t i = 0; i < input.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(input[i]);
        if (std::isspace(c)) continue;
        if (c == '=') break; // padding reached

        signed char val = kBase64Table[c];
        if (val < 0) {
            if (error_msg) *error_msg = std::string("Invalid base64 character '") + static_cast<char>(c) + "' at position " + std::to_string(i);
            return false;
        }

        buf = (buf << 6) | static_cast<uint32_t>(val);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out_bytes.push_back(static_cast<uint8_t>((buf >> bits) & 0xFF));
        }
    }

    return true;
}

const MultipartPart* MultipartFormData::find_part(const std::string& name) const {
    for (const auto& p : parts) {
        if (p.name == name) return &p;
    }
    return nullptr;
}

bool parse_multipart_form_data(std::string_view body,
                               std::string_view boundary,
                               MultipartFormData& out_form,
                               std::string* error_msg) {
    out_form.parts.clear();
    if (boundary.empty()) {
        if (error_msg) *error_msg = "Empty multipart boundary";
        return false;
    }

    if (boundary.size() >= 2 && boundary.front() == '"' && boundary.back() == '"') {
        boundary = boundary.substr(1, boundary.size() - 2);
    }

    std::string delim = "--" + std::string(boundary);
    size_t pos = body.find(delim);
    if (pos == std::string_view::npos) {
        if (error_msg) *error_msg = "Boundary delimiter not found in multipart body";
        return false;
    }

    while (pos != std::string_view::npos) {
        pos += delim.size();
        if (pos + 2 <= body.size() && body.substr(pos, 2) == "--") {
            break; // End delimiter
        }
        if (pos + 2 <= body.size() && body.substr(pos, 2) == "\r\n") {
            pos += 2;
        } else if (pos + 1 <= body.size() && body[pos] == '\n') {
            pos += 1;
        }

        size_t next_delim = body.find(delim, pos);
        if (next_delim == std::string_view::npos) break;

        std::string_view part_raw = body.substr(pos, next_delim - pos);
        size_t header_end = part_raw.find("\r\n\r\n");
        size_t sep_len = 4;
        if (header_end == std::string_view::npos) {
            header_end = part_raw.find("\n\n");
            sep_len = 2;
        }

        if (header_end != std::string_view::npos) {
            std::string_view header_section = part_raw.substr(0, header_end);
            std::string_view part_data = part_raw.substr(header_end + sep_len);
            if (part_data.size() >= 2 && part_data.substr(part_data.size() - 2) == "\r\n") {
                part_data = part_data.substr(0, part_data.size() - 2);
            } else if (!part_data.empty() && part_data.back() == '\n') {
                part_data = part_data.substr(0, part_data.size() - 1);
            }

            MultipartPart part;
            part.data = std::string(part_data);

            std::string h_str(header_section);
            std::istringstream h_stream(h_str);
            std::string line;
            while (std::getline(h_stream, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                size_t colon = line.find(':');
                if (colon == std::string::npos) continue;

                std::string h_name = line.substr(0, colon);
                std::string h_val = line.substr(colon + 1);
                while (!h_val.empty() && (h_val.front() == ' ' || h_val.front() == '\t')) h_val.erase(0, 1);

                std::string h_name_lower = h_name;
                std::transform(h_name_lower.begin(), h_name_lower.end(), h_name_lower.begin(), ::tolower);

                if (h_name_lower == "content-disposition") {
                    size_t name_pos = h_val.find("name=\"");
                    if (name_pos != std::string::npos) {
                        size_t start = name_pos + 6;
                        size_t end = h_val.find('"', start);
                        if (end != std::string::npos) {
                            part.name = h_val.substr(start, end - start);
                        }
                    }
                    size_t fn_pos = h_val.find("filename=\"");
                    if (fn_pos != std::string::npos) {
                        size_t start = fn_pos + 10;
                        size_t end = h_val.find('"', start);
                        if (end != std::string::npos) {
                            part.filename = h_val.substr(start, end - start);
                        }
                    }
                } else if (h_name_lower == "content-type") {
                    part.content_type = h_val;
                }
            }

            out_form.parts.push_back(std::move(part));
        }

        pos = next_delim;
    }

    return true;
}

bool parse_multipart_chat_completion_request(std::string_view body,
                                             std::string_view boundary,
                                             ChatCompletionRequest& out_req,
                                             ApiError& out_err) {
    MultipartFormData form;
    std::string form_err;
    if (!parse_multipart_form_data(body, boundary, form, &form_err)) {
        out_err.status_code = 400;
        out_err.type = "invalid_request_error";
        out_err.code = "multipart_parse_error";
        out_err.message = "Failed to parse multipart form-data: " + form_err;
        return false;
    }

    const auto* msg_part = form.find_part("messages");
    if (!msg_part) msg_part = form.find_part("json");
    if (!msg_part) msg_part = form.find_part("request");

    if (msg_part) {
        if (!parse_chat_completion_request(msg_part->data, out_req, out_err)) {
            return false;
        }
    } else {
        const auto* prompt_part = form.find_part("prompt");
        if (prompt_part) {
            out_req.messages.push_back(ChatMessage{.role = "user", .content = prompt_part->data, .reasoning_content = ""});
        }
    }

    for (const auto& part : form.parts) {
        if (part.name == "image" || part.name == "file" || part.content_type.rfind("image/", 0) == 0) {
            ImagePayload img;
            img.raw_bytes.assign(part.data.begin(), part.data.end());
            img.format = part.content_type;
            out_req.images.push_back(std::move(img));
        }
    }

    if (out_req.messages.empty()) {
        out_err.status_code = 400;
        out_err.type = "invalid_request_error";
        out_err.code = "missing_required_field";
        out_err.message = "Multipart request must contain a 'messages' or 'prompt' field";
        return false;
    }

    if (!out_req.images.empty()) {
        bool has_image_tag = false;
        for (const auto& msg : out_req.messages) {
            if (msg.content.find("<image>") != std::string::npos) {
                has_image_tag = true;
                break;
            }
        }
        if (!has_image_tag) {
            for (auto& msg : out_req.messages) {
                if (msg.role == "user") {
                    msg.content = "<image>\n" + msg.content;
                    break;
                }
            }
        }
    }

    return true;
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
    out_req.temperature = static_cast<float>(root.get_double("temperature", 0.0));
    out_req.top_p = static_cast<float>(root.get_double("top_p", 1.0));
    out_req.stream = root.get_bool("stream", false);

    const auto* mt = root.find("max_tokens");
    if (mt) {
        if (mt->type != JsonValue::Type::Number || static_cast<int64_t>(mt->num_val) <= 0) {
            out_err.status_code = 400;
            out_err.type = "invalid_request_error";
            out_err.param = "max_tokens";
            out_err.code = "invalid_parameter";
            out_err.message = "Invalid 'max_tokens': must be greater than 0";
            return false;
        }
        out_req.max_tokens = static_cast<int>(mt->num_val);
    } else {
        out_req.max_tokens = 256;
    }

    if (out_req.max_tokens <= 0) {
        out_err.status_code = 400;
        out_err.type = "invalid_request_error";
        out_err.param = "max_tokens";
        out_err.code = "invalid_parameter";
        out_err.message = "Invalid 'max_tokens': must be greater than 0";
        return false;
    }

    const auto* msgs = root.find("messages");
    if (!msgs || msgs->type != JsonValue::Type::Array || msgs->arr_val.empty()) {
        out_err.status_code = 400;
        out_err.type = "invalid_request_error";
        out_err.code = "missing_required_field";
        out_err.message = "Field 'messages' is required and must be a non-empty array";
        return false;
    }

    out_req.messages.clear();
    out_req.images.clear();
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

        const auto* content_val = item.find("content");
        if (content_val && content_val->type == JsonValue::Type::Array) {
            // Multimodal content array: [{"type": "text", "text": "..."}, {"type": "image_url", ...}]
            std::string text_accum;
            for (const auto& part : content_val->arr_val) {
                if (part.type != JsonValue::Type::Object) continue;
                std::string part_type = part.get_string("type", "");
                if (part_type == "text") {
                    text_accum += part.get_string("text", "");
                } else if (part_type == "image_url") {
                    const auto* url_obj = part.find("image_url");
                    std::string url = url_obj ? url_obj->get_string("url", "") : part.get_string("url", "");
                    if (!url.empty()) {
                        ImagePayload img;
                        std::string b64_err;
                        if (!decode_base64_chunked(url, img.raw_bytes, &b64_err)) {
                            out_err.status_code = 400;
                            out_err.type = "invalid_request_error";
                            out_err.code = "invalid_image_base64";
                            out_err.message = "Failed to decode base64 image: " + b64_err;
                            return false;
                        }
                        out_req.images.push_back(std::move(img));
                        if (!text_accum.empty() && text_accum.back() != '\n') {
                            text_accum += "\n";
                        }
                        text_accum += "<image>";
                    }
                }
            }
            msg.content = std::move(text_accum);
        } else {
            msg.content = item.get_string("content", "");
        }

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
