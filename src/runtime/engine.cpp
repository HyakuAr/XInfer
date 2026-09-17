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
#include <new>
#include <exception>

namespace xinfer {

class Engine::EngineImpl {
public:
    EngineImpl() = default;
    ~EngineImpl() = default;

    bool load(const EngineConfig& config, std::string* error_msg) {
        try {
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
                std::string tok_err;
                if (tokenizer_.load_from_json_buffer(tok_data.data(), tok_data.size(), &tok_err)) {
                    std::cout << "[xinfer::Engine] Loaded tokenizer from artifact ("
                              << tokenizer_.vocab_size() << " tokens, "
                              << tokenizer_.merges_size() << " merges)" << std::endl;
                } else {
                    std::cerr << "[xinfer::Engine] Failed to parse embedded tokenizer: " << tok_err << std::endl;
                }
            }
        }

        // Load chat template from container artifact (chat_template.jinja)
        if (reader.has_section("chat_template.jinja")) {
            std::vector<uint8_t> tmpl_data;
            if (reader.read_section("chat_template.jinja", tmpl_data, error_msg)) {
                std::string tmpl_err;
                if (tokenizer_.load_chat_template_buffer(tmpl_data.data(), tmpl_data.size(), &tmpl_err)) {
                    std::cout << "[xinfer::Engine] Loaded real chat template from artifact ("
                              << tmpl_data.size() << " bytes)" << std::endl;
                } else {
                    std::cerr << "[xinfer::Engine] Warning: Failed to parse embedded chat template: " << tmpl_err << std::endl;
                }
            }
        }

        // Fallback to local tokenizer.json / chat_template.jinja if not embedded
        if (!tokenizer_.is_loaded()) {
            std::string tok_err;
            if (tokenizer_.load_from_file(R"(H:\Models\Qwen3.8-27B\tokenizer.json)", &tok_err)) {
                std::cout << "[xinfer::Engine] Loaded fallback tokenizer from local checkpoint ("
                          << tokenizer_.vocab_size() << " tokens, "
                          << tokenizer_.merges_size() << " merges)" << std::endl;
            } else {
                if (error_msg) *error_msg = "Failed to load tokenizer from artifact or fallback path: " + tok_err;
                return false;
            }
        }
        if (!tokenizer_.has_chat_template()) {
            std::string tmpl_err;
            if (tokenizer_.load_chat_template_file(R"(H:\Models\Qwen3.8-27B\chat_template.jinja)", &tmpl_err)) {
                std::cout << "[xinfer::Engine] Loaded fallback chat template from local checkpoint" << std::endl;
            }
        }

        // Fail loudly if required special tokens are missing from loaded tokenizer
        if (tokenizer_.eos_token_id() < 0) {
            if (error_msg) *error_msg = "Loaded tokenizer missing valid EOS token ID (<|endoftext|>)";
            return false;
        }
        if (tokenizer_.im_end_token_id() < 0) {
            if (error_msg) *error_msg = "Loaded tokenizer missing valid IM_END token ID (<|im_end|>)";
            return false;
        }

        // 4. Materialize model weights into GPU USM memory
        model_ = targets::qwen3_8_27b::LoadedModel::load_from_artifact(ctx_, reader, error_msg);
        if (!model_) {
            return false;
        }

        // 5. Initialize device scratchpad arena
        arena_ = std::make_unique<core::DeviceArena>(ctx_, config.arena_capacity_bytes);

        // 6. Initialize persistent KV cache and recurrent states
        core::KVCacheConfig kv_cfg = model_->config().create_kv_cache_config(config.max_seq_len);
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
        } catch (const sycl::exception& e) {
            std::string msg = "SYCL exception during model load: " + std::string(e.what());
            std::cerr << "[xinfer::Engine] " << msg << std::endl;
            if (error_msg) *error_msg = msg;
            return false;
        } catch (const std::bad_alloc& e) {
            std::string msg = "Memory allocation failed (std::bad_alloc / OOM) during model load";
            std::cerr << "[xinfer::Engine] " << msg << std::endl;
            if (error_msg) *error_msg = msg;
            return false;
        } catch (const std::exception& e) {
            std::string msg = "Exception during model load: " + std::string(e.what());
            std::cerr << "[xinfer::Engine] " << msg << std::endl;
            if (error_msg) *error_msg = msg;
            return false;
        } catch (...) {
            std::string msg = "Unknown exception during model load";
            std::cerr << "[xinfer::Engine] " << msg << std::endl;
            if (error_msg) *error_msg = msg;
            return false;
        }
    }

    bool is_loaded() const noexcept {
        return is_loaded_ && model_ != nullptr && kv_cache_ != nullptr;
    }

    size_t max_seq_len() const noexcept {
        return kv_cache_ ? kv_cache_->max_seq_len() : config_.max_seq_len;
    }

    std::string apply_chat_template(const std::vector<ChatMessage>& messages) const {
        return tokenizer_.apply_chat_template(messages);
    }

    std::string apply_chat_template(const std::string& user_prompt, const std::string& system_prompt = "") const {
        return tokenizer_.apply_chat_template(user_prompt, system_prompt);
    }

    size_t count_tokens(const std::string& text, bool apply_chat_template) const {
        if (!tokenizer_.is_loaded()) return 0;
        std::string input = apply_chat_template ? tokenizer_.apply_chat_template(text) : text;
        return tokenizer_.encode(input).size();
    }

    size_t count_tokens(const std::vector<ChatMessage>& messages) const {
        if (!tokenizer_.is_loaded()) return 0;
        std::string input = tokenizer_.apply_chat_template(messages);
        return tokenizer_.encode(input).size();
    }

    bool validate_tokens(size_t prompt_tokens, int max_new_tokens, std::string* error_msg = nullptr) const {
        size_t limit = max_seq_len();
        if (prompt_tokens > limit) {
            if (error_msg) {
                *error_msg = "This model's maximum context length is " + std::to_string(limit) +
                             " tokens. However, your messages resulted in " + std::to_string(prompt_tokens) +
                             " tokens. Please reduce the length of the messages.";
            }
            return false;
        }
        if (prompt_tokens + static_cast<size_t>(std::max(0, max_new_tokens)) > limit) {
            if (error_msg) {
                *error_msg = "This model's maximum context length is " + std::to_string(limit) +
                             " tokens. However, you requested " +
                             std::to_string(prompt_tokens + static_cast<size_t>(std::max(0, max_new_tokens))) +
                             " tokens (" + std::to_string(prompt_tokens) + " in the messages, " +
                             std::to_string(max_new_tokens) + " in the completion). Please reduce the length of the messages or completion.";
            }
            return false;
        }
        return true;
    }

    GenerationResult generate(const std::string& prompt,
                              const GenerationConfig& gen_config,
                              TokenCallback callback) {
        GenerationResult result;
        try {
            if (!is_loaded()) {
                std::cerr << "[xinfer::Engine] Error: Model is not loaded" << std::endl;
                result.success = false;
                result.error_code = "model_not_loaded";
                result.error_msg = "Model is not loaded";
                return result;
            }

            if (gen_config.temperature < 0.0f) {
                std::cerr << "[xinfer::Engine] Error: Invalid temperature (" << gen_config.temperature << " < 0.0)" << std::endl;
                result.success = false;
                result.error_code = "invalid_parameter";
                result.error_msg = "Invalid temperature: must be >= 0.0";
                return result;
            }
            if (gen_config.temperature != 0.0f) {
                std::cerr << "[xinfer::Engine] Error: Non-zero temperature (" << gen_config.temperature
                          << ") is not supported (only greedy argmax sampling temperature=0.0 is implemented)" << std::endl;
                result.success = false;
                result.error_code = "unsupported_parameter";
                result.error_msg = "Currently only greedy decoding (temperature=0.0) is supported. Received temperature=" +
                                   std::to_string(gen_config.temperature) + ". Non-zero temperature sampling is not yet supported.";
                return result;
            }
            if (gen_config.top_p != 1.0f) {
                std::cerr << "[xinfer::Engine] Error: Non-default top_p (" << gen_config.top_p
                          << ") is not supported (only greedy argmax sampling top_p=1.0 is implemented)" << std::endl;
                result.success = false;
                result.error_code = "unsupported_parameter";
                result.error_msg = "Currently only greedy decoding (top_p=1.0) is supported. Received top_p=" +
                                   std::to_string(gen_config.top_p) + ". Nucleus (top-p) sampling is not yet supported.";
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
                result.success = false;
                result.error_code = "empty_prompt";
                result.error_msg = "Tokenizer produced 0 tokens";
                return result;
            }

            result.prompt_tokens = token_seq.size();

            // Validate context length before any GPU-writing call
            std::string val_err;
            if (!validate_tokens(token_seq.size(), gen_config.max_new_tokens, &val_err)) {
                std::cerr << "[xinfer::Engine] Error: " << val_err << std::endl;
                result.success = false;
                result.error_code = "context_length_exceeded";
                result.error_msg = val_err;
                return result;
            }

            // 1. Prefill prompt (supports chunking via config_.prefill_chunk_size)
            auto t0 = std::chrono::high_resolution_clock::now();
            int64_t current_tok = targets::qwen3_8::prefill_prompt(
                ctx_, *arena_, *model_, *kv_cache_, token_seq, config_.prefill_chunk_size);
            if (current_tok < 0) {
                result.success = false;
                result.error_code = "prefill_failed";
                result.error_msg = "Prefill failed: context length exceeded";
                return result;
            }

            auto t_first = std::chrono::high_resolution_clock::now();
            result.time_to_first_token_sec = std::chrono::duration<double>(t_first - t0).count();

            // Resolve stop tokens dynamically from loaded tokenizer/config unless explicitly overridden (>= 0)
            int64_t eos_tok = (gen_config.eos_token_id >= 0) ? gen_config.eos_token_id : tokenizer_.eos_token_id();
            int64_t im_end_tok = (gen_config.im_end_token_id >= 0) ? gen_config.im_end_token_id : tokenizer_.im_end_token_id();
            if (eos_tok < 0) {
                result.success = false;
                result.error_code = "missing_eos_token";
                result.error_msg = "No valid EOS token ID configured or found in loaded tokenizer";
                return result;
            }

            // Check for stop tokens on first token
            if (current_tok == eos_tok || (im_end_tok >= 0 && current_tok == im_end_tok)) {
                result.generated_tokens = 0;
                result.finish_reason = "stop";
                result.total_time_sec = result.time_to_first_token_sec;
                return result;
            }

            result.token_ids.push_back(current_tok);
            std::string first_piece = tokenizer_.decode_token(current_tok);
            result.text += first_piece;

            if (callback) {
                if (!callback(first_piece, current_tok)) {
                    result.generated_tokens = 1;
                    result.finish_reason = "stop";
                    result.total_time_sec = result.time_to_first_token_sec;
                    return result;
                }
            }

            // 2. Autoregressive single-token decode loop using persistent KV cache and captured command graph
            result.finish_reason = "length"; // Default if max_new_tokens limit is reached
            for (int step = 1; step < gen_config.max_new_tokens; ++step) {
                // Guard against writing past the KV cache buffer capacity
                if (kv_cache_->current_seq_len() >= kv_cache_->max_seq_len()) {
                    std::cout << "[xinfer::Engine] Context length limit reached ("
                              << kv_cache_->current_seq_len() << "/" << kv_cache_->max_seq_len() << ")" << std::endl;
                    result.finish_reason = "length";
                    break;
                }

                int64_t next_tok = 0;
                if (decode_graph_ && decode_graph_->is_captured()) {
                    next_tok = decode_graph_->decode_step(current_tok, kv_cache_->current_seq_len());
                    if (next_tok < 0) {
                        result.finish_reason = "length";
                        break;
                    }
                    if (!kv_cache_->advance(1)) {
                        result.finish_reason = "length";
                        break;
                    }
                } else {
                    next_tok = targets::qwen3_8::decode_step(
                        ctx_, *arena_, *model_, *kv_cache_, current_tok);
                    if (next_tok < 0) {
                        result.finish_reason = "length";
                        break;
                    }
                }

                if (next_tok == eos_tok || (im_end_tok >= 0 && next_tok == im_end_tok)) {
                    result.finish_reason = "stop";
                    break;
                }

                result.token_ids.push_back(next_tok);
                std::string piece = tokenizer_.decode_token(next_tok);
                result.text += piece;

                if (callback) {
                    if (!callback(piece, next_tok)) {
                        result.finish_reason = "stop";
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
        } catch (const sycl::exception& e) {
            std::cerr << "[xinfer::Engine] SYCL exception during generation: " << e.what() << std::endl;
            reset();
            result.success = false;
            result.error_code = "sycl_exception";
            result.error_msg = std::string("SYCL exception during generation: ") + e.what();
            return result;
        } catch (const std::bad_alloc& e) {
            std::cerr << "[xinfer::Engine] Memory allocation failed (std::bad_alloc / arena overflow) during generation" << std::endl;
            reset();
            result.success = false;
            result.error_code = "bad_alloc";
            result.error_msg = "Memory allocation failed (out of memory or arena overflow) during generation";
            return result;
        } catch (const std::exception& e) {
            std::cerr << "[xinfer::Engine] Exception during generation: " << e.what() << std::endl;
            reset();
            result.success = false;
            result.error_code = "internal_error";
            result.error_msg = std::string("Exception during generation: ") + e.what();
            return result;
        } catch (...) {
            std::cerr << "[xinfer::Engine] Unknown exception during generation" << std::endl;
            reset();
            result.success = false;
            result.error_code = "internal_error";
            result.error_msg = "Unknown exception during generation";
            return result;
        }
    }

    int64_t eos_token_id() const noexcept {
        return tokenizer_.eos_token_id();
    }

    int64_t im_end_token_id() const noexcept {
        return tokenizer_.im_end_token_id();
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

size_t Engine::max_seq_len() const noexcept {
    return impl_->max_seq_len();
}

int64_t Engine::eos_token_id() const noexcept {
    return impl_->eos_token_id();
}

int64_t Engine::im_end_token_id() const noexcept {
    return impl_->im_end_token_id();
}

std::string Engine::apply_chat_template(const std::vector<ChatMessage>& messages) const {
    return impl_->apply_chat_template(messages);
}

std::string Engine::apply_chat_template(const std::string& user_prompt, const std::string& system_prompt) const {
    return impl_->apply_chat_template(user_prompt, system_prompt);
}

size_t Engine::count_tokens(const std::string& text, bool apply_chat_template) const {
    return impl_->count_tokens(text, apply_chat_template);
}

size_t Engine::count_tokens(const std::vector<ChatMessage>& messages) const {
    return impl_->count_tokens(messages);
}

bool Engine::validate_tokens(size_t prompt_tokens, int max_new_tokens, std::string* error_msg) const {
    return impl_->validate_tokens(prompt_tokens, max_new_tokens, error_msg);
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
