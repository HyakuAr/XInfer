#include "xinfer/engine.h"
#include <sycl/sycl.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <new>
#include <exception>

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --model <path>       Path to .xinfer artifact (default: out/qwen3_8_27b.xinfer)\n"
              << "  --prompt <str>       Input prompt text (default: 'Tell me a fun fact about space.')\n"
              << "  --max-tokens <int>   Maximum generated tokens (default: 128)\n"
              << "  --chunk-size <int>   Chunk size for chunked prefill (default: 512)\n"
              << "  --max-seq-len <int>  Maximum context length for KV cache (default: 8192)\n"
              << "  --no-chat-template   Do not apply chat template formatting\n"
              << "  --help, -h           Show this help message\n"
              << std::endl;
}

#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::string model_path = "out/qwen3_8_27b.xinfer";
    std::string prompt = "Tell me a fun fact about space.";
    int max_new_tokens = 128;
    int chunk_size = 512;
    int max_seq_len = 8192;
    bool apply_chat_template = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_new_tokens = std::stoi(argv[++i]);
        } else if (arg == "--chunk-size" && i + 1 < argc) {
            chunk_size = std::stoi(argv[++i]);
        } else if (arg == "--max-seq-len" && i + 1 < argc) {
            max_seq_len = std::stoi(argv[++i]);
        } else if (arg == "--no-chat-template") {
            apply_chat_template = false;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    std::cout << "========================================================\n"
              << " xinfer: Intel Arc Pro B60 + Qwen3.8-27B Inference Engine\n"
              << "========================================================\n"
              << " Model:        " << model_path << "\n"
              << " Prompt:       " << prompt << "\n"
              << " Max Tokens:   " << max_new_tokens << "\n"
              << " Chunk Size:   " << chunk_size << "\n"
              << " Max Seq Len:  " << max_seq_len << "\n"
              << " ChatTemplate: " << (apply_chat_template ? "enabled" : "disabled") << "\n"
              << "========================================================\n"
              << std::endl;

    xinfer::Engine engine;
    xinfer::EngineConfig config;
    config.artifact_path = model_path;
    config.prefer_b60 = true;
    config.max_seq_len = static_cast<size_t>(max_seq_len);
    config.prefill_chunk_size = static_cast<size_t>(chunk_size);

    try {
        std::string error_msg;
        if (!engine.load(config, &error_msg)) {
            std::cerr << "[Error] Failed to load model: " << error_msg << std::endl;
            return 1;
        }
    } catch (const sycl::exception& e) {
        std::cerr << "[Error] SYCL exception during model load: " << e.what() << std::endl;
        return 1;
    } catch (const std::bad_alloc& e) {
        std::cerr << "[Error] Memory allocation failed (std::bad_alloc / OOM) during model load" << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[Error] Exception during model load: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "[Error] Unknown exception during model load" << std::endl;
        return 1;
    }

    xinfer::GenerationConfig gen_config;
    gen_config.max_new_tokens = max_new_tokens;
    gen_config.apply_chat_template = apply_chat_template;

    std::cout << "\n[Generation Output]\n--------------------------------------------------------" << std::endl;

    auto stream_cb = [](const std::string& piece, int64_t /*token_id*/) -> bool {
        std::cout << piece << std::flush;
        return true;
    };

    xinfer::GenerationResult result;
    try {
        result = engine.generate(prompt, gen_config, stream_cb);
    } catch (const sycl::exception& e) {
        std::cerr << "\n[Error] SYCL exception during generation: " << e.what() << std::endl;
        return 1;
    } catch (const std::bad_alloc& e) {
        std::cerr << "\n[Error] Memory allocation failed (std::bad_alloc / arena overflow) during generation" << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\n[Error] Exception during generation: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n[Error] Unknown exception during generation" << std::endl;
        return 1;
    }

    if (!result.success) {
        std::cerr << "\n[Error] Generation failed: " << result.error_msg << std::endl;
        return 1;
    }

    std::cout << "\n--------------------------------------------------------\n"
              << "\n[Performance Statistics]\n"
              << " Prompt Tokens:    " << result.prompt_tokens << "\n"
              << " Generated Tokens: " << result.generated_tokens << "\n"
              << " Time to 1st Tok:  " << result.time_to_first_token_sec << " s\n"
              << " Decode Speed:     " << result.decode_tokens_per_sec << " tokens/sec\n"
              << " Total Duration:   " << result.total_time_sec << " s\n"
              << " Token IDs:        ";
    for (int64_t id : result.token_ids) {
        std::cout << id << " ";
    }
    std::cout << "\n========================================================"
              << std::endl;

    return 0;
}
