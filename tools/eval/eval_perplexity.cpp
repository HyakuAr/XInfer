// tools/eval/eval_perplexity.cpp
//
// XInfer perplexity evaluation tool.
// Loads a .xinfer artifact, tokenizes an input text file, runs prefill to
// obtain logits for every position, computes causal cross-entropy loss
// on the GPU via the fused kernel in src/ops/eval.cpp, and reports
// perplexity along with optional per-token divergence diagnostics.
//
// Usage:
//   eval_perplexity --artifact <path.xinfer> --input <text_file>
//                   [--tokenizer <tokenizer.json>]
//                   [--max-tokens <N>]
//                   [--threshold <pct>]
//                   [--dump-divergence]
//                   [--hf-baseline <baseline.json>]
//
// Exit codes:
//   0 - Perplexity within threshold (or no baseline comparison)
//   1 - Perplexity diverges from HF baseline beyond threshold (fail-loud)
//   2 - Fatal error (load failure, OOM, etc.)

#include "xinfer/engine.h"
#include "core/device.h"
#include "core/arena.h"
#include "core/kv_cache.h"
#include "artifact/reader.h"
#include "targets/qwen3_8/tokenizer.h"
#include "targets/qwen3_8/forward.h"
#include "targets/qwen3_8_27b/weights.h"
#include "ops/eval.h"
#include "ops/sampling.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <algorithm>

namespace {

struct EvalConfig {
    std::string artifact_path;
    std::string input_path;
    std::string tokenizer_path;
    std::string hf_baseline_path;
    int64_t max_tokens{4096};
    double threshold_pct{0.5};  // Max acceptable divergence from HF baseline (%)
    bool dump_divergence{false};
    bool use_int8_kv{false};
    size_t arena_mb{512};
};

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " --artifact <path.xinfer> --input <text_file>\n"
              << "  [--tokenizer <tokenizer.json>]    External tokenizer (if not embedded)\n"
              << "  [--max-tokens <N>]                 Max tokens to evaluate (default: 4096)\n"
              << "  [--threshold <pct>]                Max divergence from HF baseline in % (default: 0.5)\n"
              << "  [--dump-divergence]                Dump per-token divergence details\n"
              << "  [--hf-baseline <baseline.json>]    Path to HF ground-truth perplexity JSON\n"
              << "  [--int8-kv]                        Use INT8 KV cache\n"
              << "  [--arena-mb <MB>]                  Arena capacity in MB (default: 512)\n";
}

EvalConfig parse_args(int argc, char** argv) {
    EvalConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--artifact" && i + 1 < argc) cfg.artifact_path = argv[++i];
        else if (arg == "--input" && i + 1 < argc) cfg.input_path = argv[++i];
        else if (arg == "--tokenizer" && i + 1 < argc) cfg.tokenizer_path = argv[++i];
        else if (arg == "--max-tokens" && i + 1 < argc) cfg.max_tokens = std::atoll(argv[++i]);
        else if (arg == "--threshold" && i + 1 < argc) cfg.threshold_pct = std::atof(argv[++i]);
        else if (arg == "--dump-divergence") cfg.dump_divergence = true;
        else if (arg == "--hf-baseline" && i + 1 < argc) cfg.hf_baseline_path = argv[++i];
        else if (arg == "--int8-kv") cfg.use_int8_kv = true;
        else if (arg == "--arena-mb" && i + 1 < argc) cfg.arena_mb = std::atoll(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
    }
    return cfg;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open input file: " + path);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Minimal JSON parser for HF baseline: extracts "perplexity" double value
// Expected format: { "perplexity": 12.345, ... }
double parse_hf_baseline_perplexity(const std::string& path) {
    std::string content = read_file(path);
    size_t pos = content.find("\"perplexity\"");
    if (pos == std::string::npos) {
        throw std::runtime_error("HF baseline JSON missing 'perplexity' key: " + path);
    }
    pos = content.find(':', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("HF baseline JSON malformed after 'perplexity' key");
    }
    ++pos;
    // Skip whitespace
    while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t')) ++pos;
    double val = std::stod(content.substr(pos));
    return val;
}

// Parse per-token losses from HF baseline JSON if available
// Expected: { ..., "per_token_loss": [0.123, 0.456, ...], ... }
std::vector<float> parse_hf_per_token_losses(const std::string& path) {
    std::vector<float> losses;
    std::string content = read_file(path);
    size_t pos = content.find("\"per_token_loss\"");
    if (pos == std::string::npos) return losses;
    pos = content.find('[', pos);
    if (pos == std::string::npos) return losses;
    ++pos;
    while (pos < content.size()) {
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' || content[pos] == '\n' || content[pos] == '\r' || content[pos] == ',')) ++pos;
        if (pos >= content.size() || content[pos] == ']') break;
        size_t end;
        float val = std::stof(content.substr(pos), &end);
        losses.push_back(val);
        pos += end;
    }
    return losses;
}

} // anonymous namespace

int main(int argc, char** argv) {
    EvalConfig cfg = parse_args(argc, argv);

    if (cfg.artifact_path.empty() || cfg.input_path.empty()) {
        print_usage(argv[0]);
        return 2;
    }

    std::cout << "========================================\n";
    std::cout << " XInfer Perplexity Evaluation Tool\n";
    std::cout << "========================================\n";
    std::cout << "  Artifact:    " << cfg.artifact_path << "\n";
    std::cout << "  Input:       " << cfg.input_path << "\n";
    std::cout << "  Max tokens:  " << cfg.max_tokens << "\n";
    std::cout << "  Threshold:   " << cfg.threshold_pct << "%\n";
    if (!cfg.hf_baseline_path.empty()) {
        std::cout << "  HF baseline: " << cfg.hf_baseline_path << "\n";
    }
    std::cout << "========================================\n\n";

    try {
        // 1. Initialize device
        auto ctx = xinfer::core::DeviceContext::create(true);
        if (!ctx) {
            std::cerr << "[eval] FATAL: Failed to initialize Intel GPU DeviceContext\n";
            return 2;
        }
        auto& q = ctx->queue();

        // 2. Open artifact and load tokenizer
        xinfer::artifact::ArtifactReader reader;
        std::string err;
        if (!reader.open(cfg.artifact_path, &err)) {
            std::cerr << "[eval] FATAL: Failed to open artifact: " << err << "\n";
            return 2;
        }

        xinfer::targets::qwen3_8::QwenTokenizer tokenizer;
        if (reader.has_section("tokenizer.data")) {
            std::vector<uint8_t> tok_data;
            if (reader.read_section("tokenizer.data", tok_data, &err)) {
                std::string tok_err;
                if (!tokenizer.load_from_json_buffer(tok_data.data(), tok_data.size(), &tok_err)) {
                    std::cerr << "[eval] Warning: Failed to parse embedded tokenizer: " << tok_err << "\n";
                }
            }
        }
        if (!tokenizer.is_loaded() && !cfg.tokenizer_path.empty()) {
            std::string tok_err;
            if (!tokenizer.load_from_file(cfg.tokenizer_path, &tok_err)) {
                std::cerr << "[eval] FATAL: Failed to load tokenizer: " << tok_err << "\n";
                return 2;
            }
        }
        if (!tokenizer.is_loaded()) {
            std::cerr << "[eval] FATAL: No tokenizer available\n";
            return 2;
        }
        std::cout << "[eval] Tokenizer loaded (" << tokenizer.vocab_size() << " tokens)\n";

        // 3. Load model weights
        auto model = xinfer::targets::qwen3_8_27b::LoadedModel::load_from_artifact(ctx, reader, &err);
        if (!model) {
            std::cerr << "[eval] FATAL: Failed to load model: " << err << "\n";
            return 2;
        }
        const auto& model_cfg = model->config();
        std::cout << "[eval] Model loaded (" << model_cfg.num_hidden_layers << " layers, "
                  << "vocab=" << model_cfg.vocab_size << ", hidden=" << model_cfg.hidden_size << ")\n";

        // 4. Initialize arena and KV cache
        size_t arena_bytes = cfg.arena_mb * 1024ULL * 1024ULL;
        auto arena = std::make_unique<xinfer::core::DeviceArena>(ctx, arena_bytes);

        size_t max_seq = static_cast<size_t>(cfg.max_tokens);
        xinfer::core::KVCacheConfig kv_cfg = model_cfg.create_kv_cache_config(max_seq, cfg.use_int8_kv);
        auto kv_cache = std::make_unique<xinfer::core::KVCache>(ctx, kv_cfg);
        if (!kv_cache->allocate()) {
            std::cerr << "[eval] FATAL: Failed to allocate KV cache\n";
            return 2;
        }
        std::cout << "[eval] KV cache allocated (" << (kv_cache->total_allocated_bytes() / (1024 * 1024)) << " MB)\n";

        // 5. Read and tokenize input
        std::string input_text = read_file(cfg.input_path);
        std::vector<int64_t> tokens = tokenizer.encode(input_text);
        if (tokens.empty()) {
            std::cerr << "[eval] FATAL: Tokenizer produced 0 tokens from input\n";
            return 2;
        }

        // Truncate to max_tokens if necessary
        if (static_cast<int64_t>(tokens.size()) > cfg.max_tokens) {
            std::cout << "[eval] Truncating input from " << tokens.size()
                      << " to " << cfg.max_tokens << " tokens\n";
            tokens.resize(cfg.max_tokens);
        }

        int64_t seq_len = static_cast<int64_t>(tokens.size());
        std::cout << "[eval] Input tokenized: " << seq_len << " tokens\n";

        if (seq_len <= 1) {
            std::cerr << "[eval] FATAL: Need at least 2 tokens for perplexity evaluation\n";
            return 2;
        }

        // 6. Allocate logits buffer: [seq_len, vocab_size] on device
        size_t logits_bytes = static_cast<size_t>(seq_len) * model_cfg.vocab_size * sizeof(float);
        std::cout << "[eval] Allocating logits buffer: " << (logits_bytes / (1024 * 1024)) << " MB\n";

        float* d_all_logits = sycl::malloc_device<float>(
            static_cast<size_t>(seq_len) * model_cfg.vocab_size, q);
        if (!d_all_logits) {
            std::cerr << "[eval] FATAL: Failed to allocate logits buffer ("
                      << (logits_bytes / (1024 * 1024)) << " MB)\n";
            return 2;
        }

        // Copy token IDs to device for the loss kernel
        int64_t* d_token_ids = sycl::malloc_device<int64_t>(seq_len, q);
        if (!d_token_ids) {
            sycl::free(d_all_logits, q);
            std::cerr << "[eval] FATAL: Failed to allocate device token ID buffer\n";
            return 2;
        }
        q.memcpy(d_token_ids, tokens.data(), seq_len * sizeof(int64_t)).wait();

        // 7. Run prefill in chunks to collect logits for all positions
        std::cout << "[eval] Running forward pass to collect logits...\n";
        auto t_fwd_start = std::chrono::high_resolution_clock::now();

        // Process in chunks, collecting per-chunk logits
        size_t chunk_size = 512;
        int64_t processed = 0;

        while (processed < seq_len) {
            int64_t chunk_len = std::min(static_cast<int64_t>(chunk_size), seq_len - processed);
            bool is_first_chunk = (processed == 0);
            int64_t start_pos = static_cast<int64_t>(kv_cache->current_seq_len());

            // For the last token in each chunk, we get logits via forward_chunk.
            // But for eval, we need logits for EVERY position in the chunk.
            // We process token-by-token to collect all logits.
            for (int64_t t = 0; t < chunk_len; ++t) {
                int64_t cur_pos = processed + t;
                float* logits_ptr = d_all_logits + cur_pos * model_cfg.vocab_size;

                if (cur_pos == 0) {
                    // First token: prefill single token
                    xinfer::targets::qwen3_8::forward_chunk(
                        ctx, *arena, *model, *kv_cache,
                        tokens.data(), 1, 0,
                        /*zero_linear_state=*/true,
                        logits_ptr);
                } else {
                    // Decode step: feed previous token, get logits
                    xinfer::targets::qwen3_8::decode_step(
                        ctx, *arena, *model, *kv_cache,
                        tokens[cur_pos], logits_ptr);
                }
            }

            processed += chunk_len;

            if (processed % 256 == 0 || processed == seq_len) {
                std::cout << "[eval]   Progress: " << processed << "/" << seq_len
                          << " tokens (" << std::fixed << std::setprecision(1)
                          << (100.0 * processed / seq_len) << "%)\n";
            }
        }

        auto t_fwd_end = std::chrono::high_resolution_clock::now();
        double fwd_sec = std::chrono::duration<double>(t_fwd_end - t_fwd_start).count();
        std::cout << "[eval] Forward pass complete in " << std::fixed << std::setprecision(2)
                  << fwd_sec << " s (" << (seq_len / fwd_sec) << " tok/s)\n";

        // 8. Compute cross-entropy loss on GPU
        std::cout << "[eval] Computing cross-entropy loss on GPU...\n";
        auto t_loss_start = std::chrono::high_resolution_clock::now();

        auto ce_result = xinfer::ops::cross_entropy_loss(
            q, d_all_logits, d_token_ids, seq_len, model_cfg.vocab_size,
            cfg.dump_divergence || !cfg.hf_baseline_path.empty());

        auto t_loss_end = std::chrono::high_resolution_clock::now();
        double loss_sec = std::chrono::duration<double>(t_loss_end - t_loss_start).count();

        // 9. Report results
        std::cout << "\n========================================\n";
        std::cout << " Perplexity Evaluation Results\n";
        std::cout << "========================================\n";
        std::cout << "  Tokens evaluated:  " << ce_result.num_tokens << "\n";
        std::cout << "  Total NLL loss:    " << std::fixed << std::setprecision(4) << ce_result.total_loss << "\n";
        std::cout << "  Mean NLL loss:     " << std::fixed << std::setprecision(4)
                  << (ce_result.total_loss / ce_result.num_tokens) << "\n";
        std::cout << "  Perplexity:        " << std::fixed << std::setprecision(4) << ce_result.perplexity << "\n";
        std::cout << "  Loss kernel time:  " << std::fixed << std::setprecision(2) << loss_sec << " s\n";
        std::cout << "  Total eval time:   " << std::fixed << std::setprecision(2)
                  << (fwd_sec + loss_sec) << " s\n";
        std::cout << "========================================\n";

        // 10. HF Baseline comparison (fail-loud contract)
        int exit_code = 0;

        if (!cfg.hf_baseline_path.empty()) {
            std::cout << "\n[eval] Comparing against HF baseline: " << cfg.hf_baseline_path << "\n";

            double hf_ppl = parse_hf_baseline_perplexity(cfg.hf_baseline_path);
            double divergence_pct = std::abs(ce_result.perplexity - hf_ppl) / hf_ppl * 100.0;

            std::cout << "  HF Perplexity:     " << std::fixed << std::setprecision(4) << hf_ppl << "\n";
            std::cout << "  XInfer Perplexity: " << std::fixed << std::setprecision(4) << ce_result.perplexity << "\n";
            std::cout << "  Divergence:        " << std::fixed << std::setprecision(4)
                      << divergence_pct << "%\n";
            std::cout << "  Threshold:         " << std::fixed << std::setprecision(4)
                      << cfg.threshold_pct << "%\n";

            if (divergence_pct > cfg.threshold_pct) {
                // Fail-loud: exit with non-zero status code
                std::cerr << "\n[eval] FAIL: Perplexity divergence (" << divergence_pct
                          << "%) exceeds threshold (" << cfg.threshold_pct << "%)!\n";
                std::cerr << "[eval] This indicates silent numerical drift in XInfer kernels\n"
                          << "       (INT4 dequantization, RMSNorm, attention, etc.).\n";

                // Dump per-token divergence details
                if (!ce_result.per_token_loss.empty()) {
                    auto hf_per_token = parse_hf_per_token_losses(cfg.hf_baseline_path);
                    if (!hf_per_token.empty()) {
                        size_t compare_n = std::min(ce_result.per_token_loss.size(), hf_per_token.size());
                        std::cerr << "\n[eval] Per-token loss divergence (top offenders):\n";
                        std::cerr << "  Pos | XInfer Loss | HF Loss    | Delta     | Token ID\n";
                        std::cerr << "  ----|-------------|------------|-----------|--------\n";

                        // Find top-N divergent positions
                        struct DivEntry {
                            size_t pos;
                            float delta;
                        };
                        std::vector<DivEntry> divs;
                        for (size_t i = 0; i < compare_n; ++i) {
                            float d = std::abs(ce_result.per_token_loss[i] - hf_per_token[i]);
                            divs.push_back({i, d});
                        }
                        std::sort(divs.begin(), divs.end(),
                                  [](const DivEntry& a, const DivEntry& b) { return a.delta > b.delta; });

                        size_t dump_n = std::min(divs.size(), size_t(20));
                        for (size_t k = 0; k < dump_n; ++k) {
                            size_t p = divs[k].pos;
                            std::cerr << "  " << std::setw(4) << p << " | "
                                      << std::fixed << std::setprecision(4) << std::setw(11) << ce_result.per_token_loss[p] << " | "
                                      << std::setw(10) << hf_per_token[p] << " | "
                                      << std::setw(9) << divs[k].delta << " | "
                                      << tokens[p + 1] << "\n";
                        }
                    }
                }

                exit_code = 1;
            } else {
                std::cout << "\n[eval] PASS: Perplexity within threshold.\n";
                exit_code = 0;
            }
        }

        // 11. Optional: dump per-token stats
        if (cfg.dump_divergence && !ce_result.per_token_loss.empty()) {
            std::cout << "\n[eval] Per-token loss dump (first 50):\n";
            std::cout << "  Pos | Loss       | Rank    | Token ID\n";
            std::cout << "  ----|------------|---------|--------\n";
            size_t dump_n = std::min(ce_result.per_token_loss.size(), size_t(50));
            for (size_t i = 0; i < dump_n; ++i) {
                std::cout << "  " << std::setw(4) << i << " | "
                          << std::fixed << std::setprecision(4) << std::setw(10) << ce_result.per_token_loss[i] << " | "
                          << std::setw(7) << ce_result.per_token_rank[i] << " | "
                          << tokens[i + 1] << "\n";
            }
        }

        // Cleanup
        sycl::free(d_all_logits, q);
        sycl::free(d_token_ids, q);

        return exit_code;

    } catch (const std::exception& e) {
        std::cerr << "[eval] FATAL: " << e.what() << "\n";
        return 2;
    }
}
