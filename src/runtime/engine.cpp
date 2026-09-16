#include "xinfer/engine.h"
#include "core/device.h"
#include "core/arena.h"
#include "artifact/reader.h"
#include "targets/qwen3_8/tokenizer.h"
#include "targets/qwen3_8/forward.h"
#include "targets/qwen3_8_27b/weights.h"
#include <chrono>
#include <iostream>

namespace xinfer {

class Engine::EngineImpl {
public:
    EngineImpl() = default;
    ~EngineImpl() = default;

    bool load(const EngineConfig& config, std::string* error_msg) {
        config_ = config;

        // 1. Initialize Device Context
        ctx_ = core::DeviceContext::create(config.prefer_b60);
        if (!ctx_) {
            if (error_msg) *error_msg = "Failed to initialize Intel GPU DeviceContext";
            return false;
        }

        const auto& arch = ctx_->arch_info();
        std::cout << "[xinfer::Engine] Hardware Target: " << arch.device_name
                  << " (" << arch.xe_core_count << " Xe-cores, "
                  << (arch.global_mem_bytes / (1024 * 1024 * 1024)) << " GB VRAM)" << std::endl;

        // 2. Open .xinfer container artifact
        artifact::ArtifactReader reader;
        if (!reader.open(config.artifact_path, error_msg)) {
            return false;
        }

        // 3. Load tokenizer from container artifact
        if (reader.has_section("tokenizer.data")) {
            std::vector<uint8_t> tok_data;
            if (reader.read_section("tokenizer.data", tok_data, error_msg)) {
                tokenizer_.load_from_json_buffer(tok_data.data(), tok_data.size());
                std::cout << "[xinfer::Engine] Loaded tokenizer from artifact ("
                          << tokenizer_.vocab_size() << " tokens)" << std::endl;
            }
        }

        // Fallback to local tokenizer.json if not embedded
        if (!tokenizer_.is_loaded()) {
            if (tokenizer_.load_from_file(R"(H:\Models\Qwen3.8-27B\tokenizer.json)")) {
                std::cout << "[xinfer::Engine] Loaded fallback tokenizer from local checkpoint ("
                          << tokenizer_.vocab_size() << " tokens)" << std::endl;
            } else {
                if (error_msg) *error_msg = "Failed to load tokenizer from artifact or fallback path";
                return false;
            }
        }

        // 4. Materialize model weights into GPU USM memory
        model_ = targets::qwen3_8_27b::LoadedModel::load_from_artifact(ctx_, reader, error_msg);
        if (!model_) {
            return false;
        }

        // 5. Initialize device scratchpad arena
        arena_ = std::make_unique<core::DeviceArena>(ctx_, config.arena_capacity_bytes);

        is_loaded_ = true;
        return true;
    }

    bool is_loaded() const noexcept {
        return is_loaded_ && model_ != nullptr;
    }

    GenerationResult generate(const std::string& prompt,
                              const GenerationConfig& gen_config,
                              TokenCallback callback) {
        GenerationResult result;
        if (!is_loaded()) {
            std::cerr << "[xinfer::Engine] Error: Model is not loaded" << std::endl;
            return result;
        }

        // Format prompt
        std::string input_text = prompt;
        if (gen_config.apply_chat_template) {
            input_text = tokenizer_.apply_chat_template(prompt);
        }

        // Encode prompt tokens
        std::vector<int64_t> token_seq = tokenizer_.encode(input_text);
        if (token_seq.empty()) {
            std::cerr << "[xinfer::Engine] Error: Tokenizer produced 0 tokens" << std::endl;
            return result;
        }

        result.prompt_tokens = token_seq.size();

        auto t0 = std::chrono::high_resolution_clock::now();
        std::chrono::high_resolution_clock::time_point t_first = t0;
        bool has_first_token = false;

        // Autoregressive decode loop
        for (int step = 0; step < gen_config.max_new_tokens; ++step) {
            // Forward pass: evaluate current sequence and get next greedy token ID
            int64_t next_tok = targets::qwen3_8::forward_next_token(ctx_, *arena_, *model_, token_seq);

            if (!has_first_token) {
                t_first = std::chrono::high_resolution_clock::now();
                has_first_token = true;
            }

            // Check for EOS / stop tokens
            if (next_tok == gen_config.eos_token_id || next_tok == gen_config.im_end_token_id) {
                break;
            }

            token_seq.push_back(next_tok);
            result.token_ids.push_back(next_tok);

            std::string piece = tokenizer_.decode_token(next_tok);
            result.text += piece;

            if (callback) {
                bool keep_going = callback(piece, next_tok);
                if (!keep_going) {
                    break;
                }
            }
        }

        auto t_end = std::chrono::high_resolution_clock::now();

        result.generated_tokens = result.token_ids.size();
        result.total_time_sec = std::chrono::duration<double>(t_end - t0).count();

        if (has_first_token) {
            result.time_to_first_token_sec = std::chrono::duration<double>(t_first - t0).count();
            double decode_time = std::chrono::duration<double>(t_end - t_first).count();
            if (decode_time > 0.0 && result.generated_tokens > 1) {
                result.decode_tokens_per_sec = static_cast<double>(result.generated_tokens - 1) / decode_time;
            }
        }

        return result;
    }

    void reset() {
        if (arena_) {
            arena_->reset();
        }
    }

private:
    std::shared_ptr<core::DeviceContext> ctx_;
    std::unique_ptr<targets::qwen3_8_27b::LoadedModel> model_;
    std::unique_ptr<core::DeviceArena> arena_;
    targets::qwen3_8::QwenTokenizer tokenizer_;
    EngineConfig config_;
    bool is_loaded_{false};
};

Engine::Engine() : impl_(std::make_unique<EngineImpl>()) {}
Engine::~Engine() = default;

Engine::Engine(Engine&&) noexcept = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

bool Engine::load(const EngineConfig& config, std::string* error_msg) {
    return impl_->load(config, error_msg);
}

bool Engine::is_loaded() const noexcept {
    return impl_->is_loaded();
}

GenerationResult Engine::generate(const std::string& prompt,
                                  const GenerationConfig& gen_config,
                                  TokenCallback callback) {
    return impl_->generate(prompt, gen_config, callback);
}

void Engine::reset() {
    impl_->reset();
}

} // namespace xinfer
