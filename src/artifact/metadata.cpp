#include "metadata.h"
#include <sstream>
#include <cctype>

namespace xinfer::artifact {

namespace {

void escape_json_string(std::ostream& os, std::string_view s) {
    os << '"';
    for (char c : s) {
        switch (c) {
            case '"':  os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\b': os << "\\b";  break;
            case '\f': os << "\\f";  break;
            case '\n': os << "\\n";  break;
            case '\r': os << "\\r";  break;
            case '\t': os << "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(c));
                    os << buf;
                } else {
                    os << c;
                }
                break;
        }
    }
    os << '"';
}

class JsonParser {
public:
    explicit JsonParser(std::string_view input) : input_(input), pos_(0) {}

    bool parse(ArtifactMetadata& out) {
        skip_whitespace();
        if (!consume('{')) return false;

        while (true) {
            skip_whitespace();
            if (peek() == '}') {
                consume('}');
                return true;
            }

            std::string key;
            if (!parse_string(key)) return false;

            skip_whitespace();
            if (!consume(':')) return false;
            skip_whitespace();

            if (key == "model_name") {
                if (!parse_string(out.model_name)) return false;
            } else if (key == "quant_scheme") {
                if (!parse_string(out.quant_scheme)) return false;
            } else if (key == "tokenizer_type") {
                if (!parse_string(out.tokenizer_type)) return false;
            } else if (key == "properties") {
                if (!parse_properties(out.properties)) return false;
            } else {
                // Skip unknown value
                if (!skip_value()) return false;
            }

            skip_whitespace();
            if (peek() == ',') {
                consume(',');
            } else if (peek() == '}') {
                consume('}');
                return true;
            } else {
                return false;
            }
        }
    }

private:
    std::string_view input_;
    size_t pos_;

    char peek() const {
        return pos_ < input_.size() ? input_[pos_] : '\0';
    }

    char get() {
        return pos_ < input_.size() ? input_[pos_++] : '\0';
    }

    bool consume(char expected) {
        if (peek() == expected) {
            pos_++;
            return true;
        }
        return false;
    }

    void skip_whitespace() {
        while (pos_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[pos_]))) {
            pos_++;
        }
    }

    bool parse_string(std::string& out) {
        out.clear();
        if (!consume('"')) return false;
        while (pos_ < input_.size()) {
            char c = get();
            if (c == '"') {
                return true;
            }
            if (c == '\\') {
                if (pos_ >= input_.size()) return false;
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
                        if (pos_ + 4 > input_.size()) return false;
                        // Skip 4 hex digits for simplicity or decode ascii
                        std::string hex(input_.substr(pos_, 4));
                        pos_ += 4;
                        try {
                            auto codepoint = std::stoul(hex, nullptr, 16);
                            if (codepoint < 0x80) {
                                out.push_back(static_cast<char>(codepoint));
                            } else {
                                out.push_back('?');
                            }
                        } catch (...) {
                            return false;
                        }
                        break;
                    }
                    default: return false;
                }
            } else {
                out.push_back(c);
            }
        }
        return false;
    }

    bool parse_properties(std::map<std::string, std::string>& props) {
        skip_whitespace();
        if (!consume('{')) return false;

        while (true) {
            skip_whitespace();
            if (peek() == '}') {
                consume('}');
                return true;
            }

            std::string k;
            if (!parse_string(k)) return false;
            skip_whitespace();
            if (!consume(':')) return false;
            skip_whitespace();

            std::string v;
            if (!parse_string(v)) return false;
            props[std::move(k)] = std::move(v);

            skip_whitespace();
            if (peek() == ',') {
                consume(',');
            } else if (peek() == '}') {
                consume('}');
                return true;
            } else {
                return false;
            }
        }
    }

    bool skip_value() {
        skip_whitespace();
        char c = peek();
        if (c == '"') {
            std::string dummy;
            return parse_string(dummy);
        }
        if (c == '{') {
            consume('{');
            int depth = 1;
            while (depth > 0 && pos_ < input_.size()) {
                char ch = get();
                if (ch == '{') depth++;
                else if (ch == '}') depth--;
                else if (ch == '"') {
                    pos_--;
                    std::string dummy;
                    if (!parse_string(dummy)) return false;
                }
            }
            return depth == 0;
        }
        if (c == '[') {
            consume('[');
            int depth = 1;
            while (depth > 0 && pos_ < input_.size()) {
                char ch = get();
                if (ch == '[') depth++;
                else if (ch == ']') depth--;
                else if (ch == '"') {
                    pos_--;
                    std::string dummy;
                    if (!parse_string(dummy)) return false;
                }
            }
            return depth == 0;
        }
        // number / bool / null
        while (pos_ < input_.size() && input_[pos_] != ',' && input_[pos_] != '}' && input_[pos_] != ']' && !std::isspace(static_cast<unsigned char>(input_[pos_]))) {
            pos_++;
        }
        return true;
    }
};

} // anonymous namespace

std::string ArtifactMetadata::to_json() const {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"model_name\": ";
    escape_json_string(ss, model_name);
    ss << ",\n  \"quant_scheme\": ";
    escape_json_string(ss, quant_scheme);
    ss << ",\n  \"tokenizer_type\": ";
    escape_json_string(ss, tokenizer_type);
    ss << ",\n  \"properties\": {\n";

    bool first = true;
    for (const auto& [k, v] : properties) {
        if (!first) {
            ss << ",\n";
        }
        first = false;
        ss << "    ";
        escape_json_string(ss, k);
        ss << ": ";
        escape_json_string(ss, v);
    }
    ss << "\n  }\n}";
    return ss.str();
}

std::optional<ArtifactMetadata> ArtifactMetadata::from_json(std::string_view json_str) {
    ArtifactMetadata meta;
    JsonParser parser(json_str);
    if (!parser.parse(meta)) {
        return std::nullopt;
    }
    return meta;
}

} // namespace xinfer::artifact
