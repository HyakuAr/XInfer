#include "xinfer/engine.h"
#include "core/device.h"
#include "core/arena.h"
#include "artifact/reader.h"
#include "targets/qwen3_8/tokenizer.h"
#include "targets/qwen3_8/forward.h"
#include "targets/qwen3_8/decode_graph.h"
#include "core/kv_cache.h"
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
        uint64_t vram_gb = (arch.global_mem_bytes + (1024ULL * 1024 * 1024 - 1)) / (1024ULL * 1024 * 1024);
        std::cout << "[xinfer::Engine] Hardware Target: " << arch.device_name
                  << " (" << arch.xe_core_count << " Xe-cores, "
                  << vram_gb << " GB VRAM)" << std::endl;

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

        // 6. Initialize persistent KV cache and recurrent states
        core::KVCacheConfig kv_cfg;
        kv_cfg.max_seq_len = config.max_seq_len;
        kv_cache_ = std::make_unique<core::KVCache>(ctx_, kv_cfg);
        if (!kv_cache_->allocate()) {
            if (error_msg) *error_msg = "Failed to allocate KV cache on Intel GPU";
            return false;
        }
        std::cout << "[xinfer::Engine] Initialized KV cache (max_seq_len=" << config.max_seq_len
                  << ", " << (kv_cache_->total_allocated_bytes() / (1024 * 1024)) << " MB VRAM)" << std::endl;

        // 7. Initialize and capture Level Zero / SYCL decode graph for fixed-shape decode step (Milestone 8)
        decode_graph_ = std::make_unique<targets::qwen3_8::DecodeGraph>(ctx_, *model_, *kv_cache_);
        if (decode_graph_->capture()) {
            std::cout << "[xinfer::Engine] Captured Level Zero decode command graph (M8 active)" << std::endl;
        } else {
            std::cout << "[xinfer::Engine] Graph capture fallback to standard kernel submission" << std::endl;
        }

        is_loaded_ = true;
        return true;
    }

    bool is_loaded() const noexcept {
        return is_loaded_ && model_ != nullptr && kv_cache_ != nullptr;
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

        // 1. Prefill prompt (supports chunking via config_.prefill_chunk_size)
        auto t0 = std::chrono::high_resolution_clock::now();
        int64_t current_tok = targets::qwen3_8::prefill_prompt(
            ctx_, *arena_, *model_, *kv_cache_, token_seq, config_.prefill_chunk_size);

        auto t_first = std::chrono::high_resolution_clock::now();
        result.time_to_first_token_sec = std::chrono::duration<double>(t_first - t0).count();

        // Check for stop tokens on first token
        if (current_tok == gen_config.eos_token_id || current_tok == gen_config.im_end_token_id) {
            result.generated_tokens = 0;
            result.total_time_sec = result.time_to_first_token_sec;
            return result;
        }

        result.token_ids.push_back(current_tok);
        std::string first_piece = tokenizer_.decode_token(current_tok);
        result.text += first_piece;

        if (callback) {
            if (!callback(first_piece, current_tok)) {
                result.generated_tokens = 1;
                result.total_time_sec = result.time_to_first_token_sec;
                return result;
            }
        }

        // 2. Autoregressive single-token decode loop using persistent KV cache and captured command graph
        for (int step = 1; step < gen_config.max_new_tokens; ++step) {
            int64_t next_tok = 0;
            if (decode_graph_ && decode_graph_->is_captured()) {
                next_tok = decode_graph_->decode_step(current_tok, kv_cache_->current_seq_len());
                kv_cache_->advance(1);
            } else {
                next_tok = targets::qwen3_8::decode_step(
                    ctx_, *arena_, *model_, *kv_cache_, current_tok);
            }

            if (next_tok == gen_config.eos_token_id || next_tok == gen_config.im_end_token_id) {
                break;
            }

            result.token_ids.push_back(next_tok);
            std::string piece = tokenizer_.decode_token(next_tok);
            result.text += piece;

            if (callback) {
                if (!callback(piece, next_tok)) {
                    break;
                }
            }

            current_tok = next_tok;
        }

        auto t_end = std::chrono::high_resolution_clock::now();

        result.generated_tokens = result.token_ids.size();
        result.total_time_sec = std::chrono::duration<double>(t_end - t0).count();

        double decode_time = std::chrono::duration<double>(t_end - t_first).count();
        if (decode_time > 0.0 && result.generated_tokens > 1) {
            result.decode_tokens_per_sec = static_cast<double>(result.generated_tokens - 1) / decode_time;
        }

        return result;
    }

    void reset() {
        if (kv_cache_) {
            kv_cache_->clear();
        }
        if (arena_) {
            arena_->reset();
        }
    }

private:
    std::shared_ptr<core::DeviceContext> ctx_;
    std::unique_ptr<targets::qwen3_8_27b::LoadedModel> model_;
    std::unique_ptr<core::DeviceArena> arena_;
    std::unique_ptr<core::KVCache> kv_cache_;
    std::unique_ptr<targets::qwen3_8::DecodeGraph> decode_graph_;
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
