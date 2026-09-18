#include "targets/qwen3_8/tokenizer.h"
#include "xinfer/engine.h"
#include "test_checkpoint.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <string>
#include <filesystem>

using namespace xinfer::targets::qwen3_8;

#define ASSERT_TRUE(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FAIL] Line " << __LINE__ << ": " << (msg) << std::endl; \
            return 1; \
        } \
    } while (0)

#define ASSERT_EQ(a, b, msg) \
    do { \
        if ((a) != (b)) { \
            std::cerr << "[FAIL] Line " << __LINE__ << ": " << (msg) \
                      << " (expected " << (b) << ", got " << (a) << ")" << std::endl; \
            return 1; \
        } \
    } while (0)

int main() {
    std::cout << "=== Running Qwen3.8 Real BPE Tokenizer & Special Tokens Test ===" << std::endl;

    // 1. Verify GenerationConfig defaults in include/xinfer/engine.h
    std::cout << "[Test 1] Verifying GenerationConfig sentinel defaults..." << std::endl;
    xinfer::GenerationConfig gen_cfg;
    ASSERT_EQ(gen_cfg.eos_token_id, -1, "GenerationConfig::eos_token_id must default to -1 (resolve from loaded model)");
    ASSERT_EQ(gen_cfg.im_end_token_id, -1, "GenerationConfig::im_end_token_id must default to -1 (resolve from loaded model)");
    std::cout << "  Passed (no hardcoded drifting constants in engine.h)" << std::endl;

    // 2. Verify fail-loudly on corrupt / missing special tokens
    std::cout << "[Test 2] Verifying fail-loudly behavior on missing special tokens..." << std::endl;
    {
        QwenTokenizer bad_tok;
        std::string bad_json = R"({"model": {"type": "BPE", "vocab": {"hello": 0}, "merges": []}})";
        std::string err;
        bool ok = bad_tok.load_from_json_buffer(bad_json.data(), bad_json.size(), &err);
        ASSERT_TRUE(!ok, "Tokenizer must fail to load if required special tokens or merges are missing");
        ASSERT_TRUE(!err.empty(), "Tokenizer must provide error message when load fails");
        std::cout << "  Passed (failed loudly as expected: " << err << ")" << std::endl;
    }

    // 3. Verify fail-loudly on config mismatch
    std::cout << "[Test 3] Verifying fail-loudly behavior on config.json mismatch..." << std::endl;
    {
        QwenTokenizer tok;
        std::string minimal_json = R"({
            "added_tokens": [
                {"id": 100, "content": "<|endoftext|>"},
                {"id": 101, "content": "<|im_start|>"},
                {"id": 102, "content": "<|im_end|>"}
            ],
            "model": {
                "type": "BPE",
                "vocab": {"<|endoftext|>": 100, "<|im_start|>": 101, "<|im_end|>": 102, "a": 0, "b": 1, "ab": 2},
                "merges": ["a b"]
            }
        })";
        std::string err;
        bool ok = tok.load_from_json_buffer(minimal_json.data(), minimal_json.size(), &err);
        ASSERT_TRUE(ok, "Minimal tokenizer must load successfully");
        ASSERT_EQ(tok.eos_token_id(), 100, "EOS token ID must match added_tokens");

        // Validate against mismatched config
        std::string bad_cfg = R"({"text_config": {"eos_token_id": 999}})";
        std::string cfg_err;
        bool cfg_ok = tok.validate_against_config_buffer(bad_cfg.data(), bad_cfg.size(), &cfg_err);
        ASSERT_TRUE(!cfg_ok, "Config validation must fail on mismatched eos_token_id");
        ASSERT_TRUE(!cfg_err.empty(), "Config validation must produce error message on mismatch");
        std::cout << "  Passed (failed loudly on mismatch: " << cfg_err << ")" << std::endl;

        // Validate against matching config
        std::string good_cfg = R"({"text_config": {"eos_token_id": 100}})";
        bool good_ok = tok.validate_against_config_buffer(good_cfg.data(), good_cfg.size(), &cfg_err);
        ASSERT_TRUE(good_ok, "Config validation must succeed on matching eos_token_id");
        std::cout << "  Passed (succeeded on matching config)" << std::endl;
    }

    // 4. Load real official checkpoint tokenizer (if present)
    std::filesystem::path checkpoint_dir = xinfer::test::get_checkpoint_dir();
    std::filesystem::path tokenizer_path = checkpoint_dir / "tokenizer.json";

    if (std::filesystem::exists(tokenizer_path)) {
        std::cout << "[Test 4] Loading real tokenizer from " << tokenizer_path.string() << "..." << std::endl;
        QwenTokenizer real_tok;
        std::string load_err;
        bool loaded = real_tok.load_from_file(tokenizer_path.string(), &load_err);
        ASSERT_TRUE(loaded, ("Failed to load real tokenizer: " + load_err).c_str());
        ASSERT_TRUE(real_tok.is_loaded(), "Tokenizer must report is_loaded() == true");
        ASSERT_TRUE(real_tok.vocab_size() >= 248044, "Vocab size must be >= 248044");
        ASSERT_TRUE(real_tok.merges_size() >= 247587, "Merges size must be >= 247587");

        std::cout << "  Vocab size:  " << real_tok.vocab_size() << " tokens" << std::endl;
        std::cout << "  Merges size: " << real_tok.merges_size() << " rules" << std::endl;

        // 5. Verify real special token IDs
        std::cout << "[Test 5] Verifying real special token IDs..." << std::endl;
        ASSERT_EQ(real_tok.eos_token_id(), 248044, "Real <|endoftext|> must be 248044");
        ASSERT_EQ(real_tok.im_start_token_id(), 248045, "Real <|im_start|> must be 248045");
        ASSERT_EQ(real_tok.im_end_token_id(), 248046, "Real <|im_end|> must be 248046");
        ASSERT_EQ(real_tok.think_start_token_id(), 248068, "Real <think> must be 248068");
        ASSERT_EQ(real_tok.think_end_token_id(), 248069, "Real </think> must be 248069");
        std::cout << "  All special token IDs verified against real checkpoint truth" << std::endl;

        // 6. Verify real BPE encode against HuggingFace AutoTokenizer oracle
        std::cout << "[Test 6] Verifying rank-ordered BPE encoding against HF ground truth..." << std::endl;

        // Case 1: Simple greeting
        {
            std::string input = "Hello world!";
            std::vector<int64_t> expected = {9419, 1814, 0};
            auto actual = real_tok.encode(input);
            ASSERT_EQ(actual.size(), expected.size(), "Token count mismatch on 'Hello world!'");
            for (size_t i = 0; i < expected.size(); ++i) {
                ASSERT_EQ(actual[i], expected[i], "Token ID mismatch in 'Hello world!'");
            }
            std::cout << "  Case 1 (Greeting) passed" << std::endl;
        }

        // Case 2: Numbers and punctuation
        {
            std::string input = "How are you doing? 12345";
            std::vector<int64_t> expected = {4199, 513, 488, 3604, 30, 220, 16, 17, 18, 19, 20};
            auto actual = real_tok.encode(input);
            ASSERT_EQ(actual.size(), expected.size(), "Token count mismatch on 'How are you doing? 12345'");
            for (size_t i = 0; i < expected.size(); ++i) {
                ASSERT_EQ(actual[i], expected[i], "Token ID mismatch in 'How are you doing? 12345'");
            }
            std::cout << "  Case 2 (Numbers & Punctuation) passed" << std::endl;
        }

        // Case 3: Contractions
        {
            std::string input = "don't, haven't, we're, I'm, they'll, it's.";
            std::vector<int64_t> expected = {14572, 914, 11, 8719, 914, 11, 567, 2224, 11, 353, 2688, 11, 781, 3172, 11, 424, 579, 13};
            auto actual = real_tok.encode(input);
            ASSERT_EQ(actual.size(), expected.size(), "Token count mismatch on contractions");
            for (size_t i = 0; i < expected.size(); ++i) {
                ASSERT_EQ(actual[i], expected[i], "Token ID mismatch in contractions");
            }
            std::cout << "  Case 3 (Contractions) passed" << std::endl;
        }

        // Case 4: Python code snippet
        {
            std::string input = "def test_fn(x, y):\n    return x + y * 42\n";
            std::vector<int64_t> expected = {727, 1228, 14802, 2007, 11, 374, 1590, 198, 262, 460, 830, 478, 374, 348, 220, 19, 17, 198};
            auto actual = real_tok.encode(input);
            ASSERT_EQ(actual.size(), expected.size(), "Token count mismatch on code snippet");
            for (size_t i = 0; i < expected.size(); ++i) {
                ASSERT_EQ(actual[i], expected[i], "Token ID mismatch in code snippet");
            }
            std::cout << "  Case 4 (Code snippet) passed" << std::endl;
        }

        // Case 5: Special tokens with Chat Template
        {
            std::string input = "<|im_start|>user\nHello world!<|im_end|>\n<|im_start|>assistant\n";
            std::vector<int64_t> expected = {248045, 846, 198, 9419, 1814, 0, 248046, 198, 248045, 74455, 198};
            auto actual = real_tok.encode(input);
            ASSERT_EQ(actual.size(), expected.size(), "Token count mismatch on chat template");
            for (size_t i = 0; i < expected.size(); ++i) {
                ASSERT_EQ(actual[i], expected[i], "Token ID mismatch in chat template");
            }
            std::cout << "  Case 5 (Chat template special tokens) passed" << std::endl;
        }

        // Case 6: Round-trip decode
        {
            std::string text = "The quick brown fox jumps over the lazy dog.";
            auto tokens = real_tok.encode(text);
            std::string decoded = real_tok.decode(tokens);
            ASSERT_EQ(decoded, text, "Round-trip encode/decode mismatch");
            std::cout << "  Case 6 (Round-trip decode) passed" << std::endl;
        }
    } else {
        std::cout << "[Test 4-6] Note: Checkpoint tokenizer.json not accessible ("
                  << tokenizer_path.string() << "), skipping real checkpoint BPE tests." << std::endl;
    }

    std::cout << "\n>>> ALL TOKENIZER BPE TESTS PASSED SUCCESSFULLY! <<<" << std::endl;
    return 0;
}
