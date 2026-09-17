#include "tokenizer.h"
#include "unicode_table.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <cstring>

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

// Skips whitespace characters in JSON stream
void skip_whitespace(const char*& p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) {
        ++p;
    }
}

// Parses JSON string starting at *p == '"'
bool parse_json_string(const char*& p, const char* end, std::string& out) {
    skip_whitespace(p, end);
    if (p >= end || *p != '"') return false;
    ++p; // Skip opening quote
    out.clear();
    while (p < end && *p != '"') {
        if (*p == '\\' && p + 1 < end) {
            char next = *(p + 1);
            if (next == '"' || next == '\\' || next == '/') {
                out.push_back(next);
                p += 2;
            } else if (next == 'b') {
                out.push_back('\b');
                p += 2;
            } else if (next == 'f') {
                out.push_back('\f');
                p += 2;
            } else if (next == 'n') {
                out.push_back('\n');
                p += 2;
            } else if (next == 'r') {
                out.push_back('\r');
                p += 2;
            } else if (next == 't') {
                out.push_back('\t');
                p += 2;
            } else if (next == 'u' && p + 5 < end) {
                char hex[5] = {p[2], p[3], p[4], p[5], '\0'};
                char* hex_end = nullptr;
                uint32_t cp = static_cast<uint32_t>(std::strtoul(hex, &hex_end, 16));
                p += 6;
                // Handle UTF-16 surrogate pairs in JSON: \uD8xx\uDCxx
                if (cp >= 0xD800 && cp <= 0xDBFF && p + 5 < end && p[0] == '\\' && p[1] == 'u') {
                    char hex2[5] = {p[2], p[3], p[4], p[5], '\0'};
                    uint32_t cp2 = static_cast<uint32_t>(std::strtoul(hex2, nullptr, 16));
                    if (cp2 >= 0xDC00 && cp2 <= 0xDFFF) {
                        cp = 0x10000 + (((cp - 0xD800) << 10) | (cp2 - 0xDC00));
                        p += 6;
                    }
                }
                out += codepoint_to_utf8(cp);
            } else {
                out.push_back(*p);
                ++p;
            }
        } else {
            out.push_back(*p);
            ++p;
        }
    }
    if (p >= end || *p != '"') return false;
    ++p; // Skip closing quote
    return true;
}

// Parses JSON integer
bool parse_json_int64(const char*& p, const char* end, int64_t& out) {
    skip_whitespace(p, end);
    if (p >= end) return false;
    bool neg = false;
    if (*p == '-') {
        neg = true;
        ++p;
    }
    if (p >= end || *p < '0' || *p > '9') return false;
    int64_t val = 0;
    while (p < end && *p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        ++p;
    }
    out = neg ? -val : val;
    return true;
}

// Skips arbitrary JSON value (object, array, string, number, bool, null)
bool skip_json_value(const char*& p, const char* end) {
    skip_whitespace(p, end);
    if (p >= end) return false;
    if (*p == '"') {
        std::string dummy;
        return parse_json_string(p, end, dummy);
    } else if (*p == '{') {
        ++p;
        int depth = 1;
        while (p < end && depth > 0) {
            if (*p == '"') {
                std::string dummy;
                if (!parse_json_string(p, end, dummy)) return false;
            } else {
                if (*p == '{') depth++;
                else if (*p == '}') depth--;
                ++p;
            }
        }
        return depth == 0;
    } else if (*p == '[') {
        ++p;
        int depth = 1;
        while (p < end && depth > 0) {
            if (*p == '"') {
                std::string dummy;
                if (!parse_json_string(p, end, dummy)) return false;
            } else {
                if (*p == '[') depth++;
                else if (*p == ']') depth--;
                ++p;
            }
        }
        return depth == 0;
    } else {
        while (p < end && *p != ',' && *p != '}' && *p != ']' && *p != ' ' && *p != '\t' && *p != '\r' && *p != '\n') {
            ++p;
        }
        return true;
    }
}

// Unicode character flags for Qwen3.5 pre-tokenization
struct CptFlags {
    bool is_letter;
    bool is_number;
    bool is_accent_mark;
    bool is_whitespace;
    bool valid;
};

inline CptFlags get_cpt_flags(uint32_t cp) noexcept {
    if (cp == 0xFFFFFFFF) {
        return {false, false, false, false, false};
    }
    uint8_t cat = get_unicode_category(cp);
    bool ws = (cat == CAT_WHITESPACE) || (cp == ' ' || cp == '\t' || cp == '\r' || cp == '\n' || cp == '\v' || cp == '\f');
    return {
        cat == CAT_LETTER,
        cat == CAT_NUMBER,
        cat == CAT_MARK,
        ws,
        cat != CAT_OTHER || ws
    };
}

// Qwen3.5 pre-tokenization regex splitting:
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
std::vector<std::pair<size_t, size_t>> split_qwen35(const std::vector<uint32_t>& cpts) {
    std::vector<std::pair<size_t, size_t>> offsets;
    const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
    const size_t n = cpts.size();

    auto _get_cpt = [&](size_t pos) -> uint32_t {
        return (pos < n) ? cpts[pos] : OUT_OF_RANGE;
    };
    auto _get_flags = [&](size_t pos) -> CptFlags {
        return get_cpt_flags(_get_cpt(pos));
    };

    size_t prev_end = 0;
    auto _add_token = [&](size_t end) {
        if (end > prev_end) {
            offsets.push_back({prev_end, end});
        }
        prev_end = end;
    };

    size_t pos = 0;
    while (pos < n) {
        uint32_t cpt = _get_cpt(pos);
        CptFlags flags = _get_flags(pos);

        // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cpt == '\'' && pos + 1 < n) {
            uint32_t c1 = _get_cpt(pos + 1);
            if (c1 >= 'A' && c1 <= 'Z') c1 += ('a' - 'A');
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                pos += 2;
                _add_token(pos);
                continue;
            }
            if (pos + 2 < n) {
                uint32_t c2 = _get_cpt(pos + 2);
                if (c2 >= 'A' && c2 <= 'Z') c2 += ('a' - 'A');
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                    pos += 3;
                    _add_token(pos);
                    continue;
                }
            }
        }

        // regex: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if (cpt != '\r' && cpt != '\n' && !flags.is_number) {
            if (flags.is_letter || flags.is_accent_mark || _get_flags(pos + 1).is_accent_mark || _get_flags(pos + 1).is_letter) {
                pos++;
                while (_get_flags(pos).is_letter || _get_flags(pos).is_accent_mark) {
                    pos++;
                }
                _add_token(pos);
                continue;
            }
        }

        // regex: \p{N}
        if (flags.is_number) {
            pos++;
            _add_token(pos);
            continue;
        }

        // regex: <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        CptFlags flags2 = (cpt == ' ') ? _get_flags(pos + 1) : flags;
        bool not_ws_lm_num = !(flags2.is_whitespace || flags2.is_letter || flags2.is_accent_mark || flags2.is_number);
        if (not_ws_lm_num && flags.valid) {
            pos += (cpt == ' ' ? 1 : 0);
            while (true) {
                CptFlags f2 = _get_flags(pos);
                if ((f2.is_whitespace || f2.is_letter || f2.is_accent_mark || f2.is_number) || !f2.valid) {
                    break;
                }
                pos++;
            }
            while (_get_cpt(pos) == '\r' || _get_cpt(pos) == '\n') {
                pos++;
            }
            _add_token(pos);
            continue;
        }

        size_t num_ws = 0;
        size_t last_end_r_or_n = 0;
        while (_get_flags(pos + num_ws).is_whitespace) {
            uint32_t c2 = _get_cpt(pos + num_ws);
            if (c2 == '\r' || c2 == '\n') {
                last_end_r_or_n = pos + num_ws + 1;
            }
            num_ws++;
        }

        // regex: \s*[\r\n]+
        if (last_end_r_or_n > 0) {
            pos = last_end_r_or_n;
            _add_token(pos);
            continue;
        }

        // regex: \s+(?!\S)
        if (num_ws > 1 && _get_cpt(pos + num_ws) != OUT_OF_RANGE) {
            pos += num_ws - 1;
            _add_token(pos);
            continue;
        }

        // regex: \s+
        if (num_ws > 0) {
            pos += num_ws;
            _add_token(pos);
            continue;
        }

        pos++;
        _add_token(pos);
    }
    return offsets;
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

bool QwenTokenizer::load_from_json_buffer(const void* data, size_t size, std::string* error_msg) {
    if (!data || size == 0) {
        if (error_msg) *error_msg = "Empty tokenizer JSON buffer provided";
        return false;
    }

    vocab_.clear();
    id_to_token_.clear();
    bpe_ranks_.clear();
    bpe_cache_.clear();
    special_tokens_.clear();

    eos_token_id_ = -1;
    im_start_token_id_ = -1;
    im_end_token_id_ = -1;
    think_start_token_id_ = -1;
    think_end_token_id_ = -1;

    const char* start = static_cast<const char*>(data);
    const char* end = start + size;

    // 1. Locate and parse "added_tokens"
    const char* added_key = std::strstr(start, "\"added_tokens\"");
    if (added_key && added_key < end) {
        const char* p = added_key + 14;
        skip_whitespace(p, end);
        if (p < end && *p == ':') {
            ++p;
            skip_whitespace(p, end);
            if (p < end && *p == '[') {
                ++p;
                while (p < end) {
                    skip_whitespace(p, end);
                    if (p >= end || *p == ']') {
                        if (p < end) ++p;
                        break;
                    }
                    if (*p == ',') {
                        ++p;
                        continue;
                    }
                    if (*p == '{') {
                        ++p;
                        int64_t tok_id = -1;
                        std::string tok_content;
                        bool has_id = false;
                        bool has_content = false;

                        while (p < end && *p != '}') {
                            skip_whitespace(p, end);
                            if (p >= end || *p == '}') break;
                            if (*p == ',') {
                                ++p;
                                continue;
                            }
                            std::string key;
                            if (!parse_json_string(p, end, key)) break;
                            skip_whitespace(p, end);
                            if (p < end && *p == ':') {
                                ++p;
                                skip_whitespace(p, end);
                                if (key == "id") {
                                    if (parse_json_int64(p, end, tok_id)) {
                                        has_id = true;
                                    }
                                } else if (key == "content") {
                                    if (parse_json_string(p, end, tok_content)) {
                                        has_content = true;
                                    }
                                } else {
                                    skip_json_value(p, end);
                                }
                            }
                        }
                        if (p < end && *p == '}') ++p;

                        if (has_id && has_content) {
                            vocab_[tok_content] = tok_id;
                            id_to_token_[tok_id] = tok_content;
                            special_tokens_.push_back({tok_content, tok_id});

                            if (tok_content == "<|endoftext|>") {
                                eos_token_id_ = tok_id;
                            } else if (tok_content == "<|im_start|>") {
                                im_start_token_id_ = tok_id;
                            } else if (tok_content == "<|im_end|>") {
                                im_end_token_id_ = tok_id;
                            } else if (tok_content == "<think>") {
                                think_start_token_id_ = tok_id;
                            } else if (tok_content == "</think>") {
                                think_end_token_id_ = tok_id;
                            }
                        }
                    } else {
                        ++p;
                    }
                }
            }
        }
    }

    // 2. Locate and parse "model" -> "vocab"
    const char* vocab_key = std::strstr(start, "\"vocab\"");
    if (vocab_key && vocab_key < end) {
        const char* p = vocab_key + 7;
        skip_whitespace(p, end);
        if (p < end && *p == ':') {
            ++p;
            skip_whitespace(p, end);
            if (p < end && *p == '{') {
                ++p;
                while (p < end) {
                    skip_whitespace(p, end);
                    if (p >= end || *p == '}') {
                        if (p < end) ++p;
                        break;
                    }
                    if (*p == ',') {
                        ++p;
                        continue;
                    }
                    std::string token;
                    if (!parse_json_string(p, end, token)) break;
                    skip_whitespace(p, end);
                    if (p < end && *p == ':') {
                        ++p;
                        int64_t id = 0;
                        if (parse_json_int64(p, end, id)) {
                            vocab_[token] = id;
                            id_to_token_[id] = token;
                        }
                    }
                }
            }
        }
    }

    // 3. Locate and parse "model" -> "merges"
    const char* merges_key = std::strstr(start, "\"merges\"");
    if (merges_key && merges_key < end) {
        const char* p = merges_key + 8;
        skip_whitespace(p, end);
        if (p < end && *p == ':') {
            ++p;
            skip_whitespace(p, end);
            if (p < end && *p == '[') {
                ++p;
                int64_t rank = 0;
                while (p < end) {
                    skip_whitespace(p, end);
                    if (p >= end || *p == ']') {
                        if (p < end) ++p;
                        break;
                    }
                    if (*p == ',') {
                        ++p;
                        continue;
                    }
                    std::string merge_str;
                    if (!parse_json_string(p, end, merge_str)) break;
                    bpe_ranks_[merge_str] = rank++;
                }
            }
        }
    }

    // Resolve any special tokens not yet discovered from added_tokens by checking vocab
    if (eos_token_id_ < 0) {
        auto it = vocab_.find("<|endoftext|>");
        if (it != vocab_.end()) eos_token_id_ = it->second;
    }
    if (im_start_token_id_ < 0) {
        auto it = vocab_.find("<|im_start|>");
        if (it != vocab_.end()) im_start_token_id_ = it->second;
    }
    if (im_end_token_id_ < 0) {
        auto it = vocab_.find("<|im_end|>");
        if (it != vocab_.end()) im_end_token_id_ = it->second;
    }
    if (think_start_token_id_ < 0) {
        auto it = vocab_.find("<think>");
        if (it != vocab_.end()) think_start_token_id_ = it->second;
    }
    if (think_end_token_id_ < 0) {
        auto it = vocab_.find("</think>");
        if (it != vocab_.end()) think_end_token_id_ = it->second;
    }

    // Sort special tokens by length descending to match longest special token first
    std::sort(special_tokens_.begin(), special_tokens_.end(),
        [](const auto& a, const auto& b) {
            return a.first.size() > b.first.size();
        });

    // Fail loudly if required components or special tokens are missing
    if (vocab_.empty()) {
        std::string err = "Failed to parse vocabulary from tokenizer.json (vocab table is empty)";
        if (error_msg) *error_msg = err;
        return false;
    }
    if (bpe_ranks_.empty()) {
        std::string err = "Failed to parse BPE merge rules from tokenizer.json (merges table is empty)";
        if (error_msg) *error_msg = err;
        return false;
    }
    if (eos_token_id_ < 0) {
        std::string err = "Missing required special token <|endoftext|> in loaded tokenizer";
        if (error_msg) *error_msg = err;
        return false;
    }
    if (im_start_token_id_ < 0) {
        std::string err = "Missing required special token <|im_start|> in loaded tokenizer";
        if (error_msg) *error_msg = err;
        return false;
    }
    if (im_end_token_id_ < 0) {
        std::string err = "Missing required special token <|im_end|> in loaded tokenizer";
        if (error_msg) *error_msg = err;
        return false;
    }

    return true;
}

bool QwenTokenizer::validate_against_config_buffer(const void* data, size_t size, std::string* error_msg) {
    if (!data || size == 0) return true;

    const char* start = static_cast<const char*>(data);
    const char* end = start + size;

    // Search for "eos_token_id" in config.json
    const char* p = std::strstr(start, "\"eos_token_id\"");
    if (!p || p >= end) return true;

    p += 14;
    skip_whitespace(p, end);
    if (p >= end || *p != ':') return true;
    ++p;
    skip_whitespace(p, end);

    int64_t cfg_eos = -1;
    if (p < end && *p == '[') {
        // Can be an array, e.g. in generation_config.json: [248046, 248044]
        ++p;
        skip_whitespace(p, end);
        while (p < end && *p != ']') {
            int64_t val = 0;
            if (parse_json_int64(p, end, val)) {
                if (val == eos_token_id_) {
                    cfg_eos = val;
                    break;
                }
                if (cfg_eos < 0) cfg_eos = val;
            }
            skip_whitespace(p, end);
            if (p < end && *p == ',') ++p;
            skip_whitespace(p, end);
        }
    } else {
        parse_json_int64(p, end, cfg_eos);
    }

    if (cfg_eos >= 0 && eos_token_id_ >= 0) {
        // Check if matching eos_token_id
        if (cfg_eos != eos_token_id_) {
            // If generation_config had multiple or config has different token, report mismatch
            std::string err = "Config eos_token_id (" + std::to_string(cfg_eos) +
                              ") does not match tokenizer <|endoftext|> ID (" +
                              std::to_string(eos_token_id_) + ")";
            if (error_msg) *error_msg = err;
            return false;
        }
    }

    return true;
}

bool QwenTokenizer::validate_against_config(const std::string& config_path, std::string* error_msg) {
    std::ifstream f(config_path, std::ios::binary);
    if (!f.is_open()) return true; // Optional file
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return validate_against_config_buffer(content.data(), content.size(), error_msg);
}

bool QwenTokenizer::load_from_file(const std::string& path, std::string* error_msg) {
    std::string tok_path = path;
    std::filesystem::path fs_path(path);
    if (std::filesystem::is_directory(fs_path)) {
        tok_path = (fs_path / "tokenizer.json").string();
    }

    std::ifstream f(tok_path, std::ios::binary);
    if (!f.is_open()) {
        if (error_msg) *error_msg = "Could not open tokenizer file: " + tok_path;
        return false;
    }
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (!load_from_json_buffer(content.data(), content.size(), error_msg)) {
        return false;
    }

    // Check sibling config.json if present
    std::filesystem::path dir = std::filesystem::path(tok_path).parent_path();
    std::filesystem::path cfg_file = dir / "config.json";
    if (std::filesystem::exists(cfg_file)) {
        if (!validate_against_config(cfg_file.string(), error_msg)) {
            return false;
        }
    }

    // Check sibling chat_template.jinja if present
    std::filesystem::path jinja_file = dir / "chat_template.jinja";
    if (std::filesystem::exists(jinja_file)) {
        chat_template_.load_from_file(jinja_file.string());
    }

    return true;
}

int64_t QwenTokenizer::special_token_to_id(const std::string& name) const {
    auto it = vocab_.find(name);
    return (it != vocab_.end()) ? it->second : -1;
}

bool QwenTokenizer::has_special_token(const std::string& name) const {
    return vocab_.find(name) != vocab_.end();
}

std::vector<std::string> QwenTokenizer::bpe_merge(const std::string& piece) const {
    if (piece.empty()) return {};

    // Check memoization cache
    auto cache_it = bpe_cache_.find(piece);
    if (cache_it != bpe_cache_.end()) {
        return cache_it->second;
    }

    // Split piece into individual unicode codepoints
    auto cps = utf8_to_codepoints(piece);
    std::vector<std::string> word;
    word.reserve(cps.size());
    for (uint32_t cp : cps) {
        word.push_back(codepoint_to_utf8(cp));
    }

    if (word.size() <= 1) {
        bpe_cache_[piece] = word;
        return word;
    }

    // Iteratively apply rank-ordered BPE merges
    while (word.size() > 1) {
        int64_t min_rank = -1;
        std::string best_first;
        std::string best_second;

        for (size_t i = 0; i + 1 < word.size(); ++i) {
            std::string pair_str = word[i] + " " + word[i + 1];
            auto it = bpe_ranks_.find(pair_str);
            if (it != bpe_ranks_.end()) {
                if (min_rank < 0 || it->second < min_rank) {
                    min_rank = it->second;
                    best_first = word[i];
                    best_second = word[i + 1];
                }
            }
        }

        if (min_rank < 0) {
            // No mergeable pairs found in bpe_ranks_
            break;
        }

        std::vector<std::string> new_word;
        new_word.reserve(word.size());
        for (size_t i = 0; i < word.size(); ) {
            if (i + 1 < word.size() && word[i] == best_first && word[i + 1] == best_second) {
                new_word.push_back(word[i] + word[i + 1]);
                i += 2;
            } else {
                new_word.push_back(std::move(word[i]));
                i += 1;
            }
        }
        word = std::move(new_word);
    }

    bpe_cache_[piece] = word;
    return word;
}

std::vector<int64_t> QwenTokenizer::encode(const std::string& text) const {
    if (text.empty() || vocab_.empty()) return {};

    std::vector<int64_t> tokens;

    auto encode_segment = [&](const std::string& seg) {
        if (seg.empty()) return;
        auto cpts = utf8_to_codepoints(seg);
        auto split_ranges = split_qwen35(cpts);

        for (const auto& rng : split_ranges) {
            std::string chunk;
            for (size_t i = rng.first; i < rng.second; ++i) {
                chunk += codepoint_to_utf8(cpts[i]);
            }

            // Convert raw UTF-8 bytes to byte_to_unicode encoding
            std::string encoded_chunk;
            for (char ch : chunk) {
                uint8_t byte_val = static_cast<uint8_t>(ch);
                auto it = byte_to_unicode_.find(byte_val);
                if (it != byte_to_unicode_.end()) {
                    encoded_chunk += it->second;
                } else {
                    encoded_chunk.push_back(ch);
                }
            }

            // Perform rank-ordered BPE merging
            std::vector<std::string> pieces = bpe_merge(encoded_chunk);
            for (const auto& piece : pieces) {
                auto it = vocab_.find(piece);
                if (it != vocab_.end()) {
                    tokens.push_back(it->second);
                } else {
                    // Fallback to individual byte pieces if subword not directly in vocab
                    for (char c : piece) {
                        uint8_t bv = static_cast<uint8_t>(c);
                        auto it_b = byte_to_unicode_.find(bv);
                        if (it_b != byte_to_unicode_.end()) {
                            auto it_v = vocab_.find(it_b->second);
                            if (it_v != vocab_.end()) {
                                tokens.push_back(it_v->second);
                            }
                        }
                    }
                }
            }
        }
    };

    // Pre-split text around dynamically discovered special tokens
    size_t pos = 0;
    while (pos < text.size()) {
        size_t nearest_pos = std::string::npos;
        size_t matched_idx = 0;

        for (size_t s = 0; s < special_tokens_.size(); ++s) {
            const auto& st = special_tokens_[s];
            size_t p = text.find(st.first, pos);
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
            tokens.push_back(special_tokens_[matched_idx].second);
            pos = nearest_pos + special_tokens_[matched_idx].first.size();
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

bool QwenTokenizer::load_chat_template_buffer(const void* data, size_t size, std::string* error_msg) {
    return chat_template_.load_from_buffer(data, size, error_msg);
}

bool QwenTokenizer::load_chat_template_file(const std::string& path, std::string* error_msg) {
    return chat_template_.load_from_file(path, error_msg);
}

std::string QwenTokenizer::apply_chat_template(const std::vector<ChatMessage>& messages,
                                              const ChatTemplateOptions& options) const {
    return chat_template_.render(messages, options);
}

std::string QwenTokenizer::apply_chat_template(const std::string& user_prompt,
                                              const std::string& system_prompt,
                                              const ChatTemplateOptions& options) const {
    return chat_template_.render(user_prompt, system_prompt, options);
}

} // namespace xinfer::targets::qwen3_8
