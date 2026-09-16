#include "tokenizer.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <set>

namespace xinfer::targets::qwen3_8 {

namespace {

// Converts UTF-8 string to unicode codepoints
std::vector<uint32_t> utf8_to_codepoints(const std::string& str) {
    std::vector<uint32_t> cps;
    size_t i = 0;
    while (i < str.size()) {
        uint8_t c = static_cast<uint8_t>(str[i]);
        uint32_t cp = 0;
        size_t len = 0;
        if (c < 0x80) {
            cp = c;
            len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F;
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F;
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = c & 0x07;
            len = 4;
        } else {
            i++;
            continue;
        }
        if (i + len > str.size()) break;
        for (size_t j = 1; j < len; ++j) {
            cp = (cp << 6) | (static_cast<uint8_t>(str[i + j]) & 0x3F);
        }
        cps.push_back(cp);
        i += len;
    }
    return cps;
}

// Converts a unicode codepoint to UTF-8 string
std::string codepoint_to_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return s;
}

} // anonymous namespace

QwenTokenizer::QwenTokenizer() {
    init_byte_encoder();
}

QwenTokenizer::~QwenTokenizer() = default;

void QwenTokenizer::init_byte_encoder() {
    // Standard GPT-2 / Qwen byte-to-unicode mapping
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        bool printable = (b >= '!' && b <= '~') || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        uint32_t cp = printable ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + (n++));
        std::string utf8_char = codepoint_to_utf8(cp);
        byte_to_unicode_[static_cast<uint8_t>(b)] = utf8_char;
        unicode_to_byte_[utf8_char] = static_cast<uint8_t>(b);
    }
}

bool QwenTokenizer::load_from_json_buffer(const void* data, size_t size) {
    if (!data || size == 0) return false;
    const char* p = static_cast<const char*>(data);
    const char* end = p + size;

    // Fast linear parser for "token_string": token_id
    while (p < end) {
        while (p < end && *p != '"') ++p;
        if (p >= end) break;
        ++p; // Skip opening quote

        std::string token;
        while (p < end && *p != '"') {
            if (*p == '\\' && p + 1 < end) {
                char next = *(p + 1);
                if (next == '"' || next == '\\' || next == '/') {
                    token.push_back(next);
                    p += 2;
                } else if (next == 'n') {
                    token.push_back('\n');
                    p += 2;
                } else if (next == 'r') {
                    token.push_back('\r');
                    p += 2;
                } else if (next == 't') {
                    token.push_back('\t');
                    p += 2;
                } else if (next == 'u' && p + 5 < end) {
                    // 4 hex digits
                    char hex[5] = {p[2], p[3], p[4], p[5], '\0'};
                    uint32_t cp = static_cast<uint32_t>(std::strtoul(hex, nullptr, 16));
                    token += codepoint_to_utf8(cp);
                    p += 6;
                } else {
                    token.push_back(*p);
                    ++p;
                }
            } else {
                token.push_back(*p);
                ++p;
            }
        }
        if (p >= end) break;
        ++p; // Skip closing quote

        // Look for colon
        while (p < end && *p != ':' && *p != ',' && *p != '}') ++p;
        if (p < end && *p == ':') {
            ++p;
            while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
            if (p < end && (*p >= '0' && *p <= '9')) {
                int64_t id = 0;
                while (p < end && *p >= '0' && *p <= '9') {
                    id = id * 10 + (*p - '0');
                    ++p;
                }
                vocab_[token] = id;
                id_to_token_[id] = token;
            }
        }
    }

    // Register known special tokens
    vocab_["<|endoftext|>"] = EOS_TOKEN_ID;
    id_to_token_[EOS_TOKEN_ID] = "<|endoftext|>";

    vocab_["<|im_start|>"] = IM_START_TOKEN_ID;
    id_to_token_[IM_START_TOKEN_ID] = "<|im_start|>";

    vocab_["<|im_end|>"] = IM_END_TOKEN_ID;
    id_to_token_[IM_END_TOKEN_ID] = "<|im_end|>";

    vocab_["<think>"] = THINK_START_TOKEN_ID;
    id_to_token_[THINK_START_TOKEN_ID] = "<think>";

    vocab_["</think>"] = THINK_END_TOKEN_ID;
    id_to_token_[THINK_END_TOKEN_ID] = "</think>";

    return !vocab_.empty();
}

bool QwenTokenizer::load_from_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return load_from_json_buffer(content.data(), content.size());
}

std::vector<int64_t> QwenTokenizer::encode(const std::string& text) const {
    if (text.empty() || vocab_.empty()) return {};

    std::vector<int64_t> tokens;

    // Pre-split text around special tokens so regular token matching cannot corrupt them
    const std::vector<std::pair<std::string, int64_t>> special_tokens = {
        {"<|im_start|>", IM_START_TOKEN_ID},
        {"<|im_end|>", IM_END_TOKEN_ID},
        {"<|endoftext|>", EOS_TOKEN_ID},
        {"<think>", THINK_START_TOKEN_ID},
        {"</think>", THINK_END_TOKEN_ID}
    };

    auto encode_segment = [&](const std::string& seg) {
        size_t i = 0;
        while (i < seg.size()) {
            size_t max_len = std::min<size_t>(seg.size() - i, 64);
            bool found = false;

            for (size_t len = max_len; len > 0; --len) {
                std::string sub = seg.substr(i, len);
                std::string encoded_sub;
                for (char ch : sub) {
                    uint8_t byte_val = static_cast<uint8_t>(ch);
                    auto it = byte_to_unicode_.find(byte_val);
                    if (it != byte_to_unicode_.end()) {
                        encoded_sub += it->second;
                    } else {
                        encoded_sub.push_back(ch);
                    }
                }

                auto it = vocab_.find(encoded_sub);
                if (it != vocab_.end()) {
                    tokens.push_back(it->second);
                    i += len;
                    found = true;
                    break;
                }
            }

            if (!found) {
                uint8_t byte_val = static_cast<uint8_t>(seg[i]);
                auto it_u = byte_to_unicode_.find(byte_val);
                if (it_u != byte_to_unicode_.end()) {
                    auto it_v = vocab_.find(it_u->second);
                    if (it_v != vocab_.end()) {
                        tokens.push_back(it_v->second);
                    }
                }
                i++;
            }
        }
    };

    size_t pos = 0;
    while (pos < text.size()) {
        size_t nearest_pos = std::string::npos;
        size_t matched_idx = 0;

        for (size_t s = 0; s < special_tokens.size(); ++s) {
            size_t p = text.find(special_tokens[s].first, pos);
            if (p != std::string::npos && (nearest_pos == std::string::npos || p < nearest_pos)) {
                nearest_pos = p;
                matched_idx = s;
            }
        }

        size_t seg_end = (nearest_pos == std::string::npos) ? text.size() : nearest_pos;
        if (seg_end > pos) {
            encode_segment(text.substr(pos, seg_end - pos));
        }

        if (nearest_pos != std::string::npos) {
            tokens.push_back(special_tokens[matched_idx].second);
            pos = nearest_pos + special_tokens[matched_idx].first.size();
        } else {
            pos = text.size();
        }
    }

    return tokens;
}

std::string QwenTokenizer::decode_token(int64_t token_id) const {
    auto it = id_to_token_.find(token_id);
    if (it == id_to_token_.end()) return "";

    const std::string& piece = it->second;
    if (piece == "<|im_start|>" || piece == "<|im_end|>" || piece == "<|endoftext|>") {
        return "";
    }

    // Convert GPT-2 unicode codepoints back to raw bytes
    std::string result;
    auto codepoints = utf8_to_codepoints(piece);
    for (uint32_t cp : codepoints) {
        std::string s = codepoint_to_utf8(cp);
        auto it_byte = unicode_to_byte_.find(s);
        if (it_byte != unicode_to_byte_.end()) {
            result.push_back(static_cast<char>(it_byte->second));
        } else {
            result += s;
        }
    }
    return result;
}

std::string QwenTokenizer::decode(const std::vector<int64_t>& token_ids) const {
    std::string text;
    for (int64_t id : token_ids) {
        text += decode_token(id);
    }
    return text;
}

std::string QwenTokenizer::apply_chat_template(const std::string& user_prompt,
                                              const std::string& system_prompt) const {
    std::string formatted;
    formatted += "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
    formatted += "<|im_start|>user\n" + user_prompt + "<|im_end|>\n";
    formatted += "<|im_start|>assistant\n";
    return formatted;
}

} // namespace xinfer::targets::qwen3_8
