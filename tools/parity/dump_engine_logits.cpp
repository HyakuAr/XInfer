#include "core/device.h"
#include "core/arena.h"
#include "core/kv_cache.h"
#include "targets/qwen3_8_27b/weights.h"
#include "targets/qwen3_8/forward.h"
#include "targets/qwen3_8/tokenizer.h"
#include "artifact/reader.h"
#include "ops/sampling.h"

#include <sycl/sycl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <iomanip>

using namespace xinfer;

struct TokenPred {
    int rank;
    int64_t token_id;
    std::string token_str;
    float logit;
    float prob;
};

int main(int argc, char** argv) {
    std::string artifact_path = "out/qwen3_8_27b.xinfer";
    std::string output_npy = "tools/parity/engine_logits.npy";
    std::string output_json = "tools/parity/engine_logits.json";

    std::vector<int64_t> prompt_tokens = {760, 12515, 369}; // "The sky is"

    std::cout << "========================================================\n"
              << " xinfer: Dump Engine Logits on Intel Arc Pro B60\n"
              << "========================================================\n"
              << " Artifact:   " << artifact_path << "\n"
              << " Output NPY: " << output_npy << "\n"
              << " Output JSON:" << output_json << "\n"
              << " Tokens:     ";
    for (auto t : prompt_tokens) std::cout << t << " ";
    std::cout << "\n========================================================\n" << std::endl;

    auto ctx = core::DeviceContext::create(true);
    if (!ctx) {
        std::cerr << "Failed to create Intel B60 DeviceContext" << std::endl;
        return 1;
    }

    sycl::queue& q = ctx->queue();

    artifact::ArtifactReader reader;
    if (!reader.open(artifact_path)) {
        std::cerr << "Failed to open artifact: " << artifact_path << std::endl;
        return 1;
    }

    targets::qwen3_8::QwenTokenizer tokenizer;
    if (reader.has_section("tokenizer.data")) {
        std::vector<uint8_t> tok_data;
        if (reader.read_section("tokenizer.data", tok_data)) {
            tokenizer.load_from_json_buffer(tok_data.data(), tok_data.size());
        }
    }
    if (!tokenizer.is_loaded()) {
        tokenizer.load_from_file(R"(H:\Models\Qwen3.8-27B\tokenizer.json)");
    }

    std::string err;
    auto model = targets::qwen3_8_27b::LoadedModel::load_from_artifact(ctx, reader, &err);
    if (!model) {
        std::cerr << "Failed to load model: " << err << std::endl;
        return 1;
    }

    const auto& cfg = model->config();
    core::DeviceArena arena(ctx, 256 * 1024 * 1024);
    core::KVCacheConfig kv_cfg = cfg.create_kv_cache_config(8192);
    core::KVCache kv_cache(ctx, kv_cfg);
    if (!kv_cache.allocate()) {
        std::cerr << "Failed to allocate KV cache" << std::endl;
        return 1;
    }

    const int64_t vocab_size = cfg.vocab_size;
    float* d_logits = sycl::malloc_device<float>(vocab_size, q);

    std::cout << "Executing forward pass for prompt on B60..." << std::endl;
    kv_cache.clear();

    targets::qwen3_8::forward_chunk(
        ctx,
        arena,
        *model,
        kv_cache,
        prompt_tokens.data(),
        prompt_tokens.size(),
        0,
        true,
        d_logits
    );
    q.wait();

    std::vector<float> host_logits(vocab_size);
    q.memcpy(host_logits.data(), d_logits, vocab_size * sizeof(float)).wait();

    // Compute softmax
    float max_l = *std::max_element(host_logits.begin(), host_logits.end());
    double sum_exp = 0.0;
    std::vector<float> probs(vocab_size);
    for (int64_t i = 0; i < vocab_size; ++i) {
        float e = std::exp(host_logits[i] - max_l);
        probs[i] = e;
        sum_exp += e;
    }
    for (int64_t i = 0; i < vocab_size; ++i) {
        probs[i] = static_cast<float>(probs[i] / sum_exp);
    }

    // Top-K
    std::vector<int64_t> indices(vocab_size);
    for (int64_t i = 0; i < vocab_size; ++i) indices[i] = i;
    int top_k = 10;
    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                      [&](int64_t a, int64_t b) {
                          return probs[a] > probs[b];
                      });

    std::cout << "\n========================================================================\n"
              << "  ENGINE PREDICTIONS ON INTEL ARC PRO B60 (INT4)\n"
              << "========================================================================\n";
    std::cout << std::left << std::setw(6) << "Rank"
              << std::setw(12) << "Token ID"
              << std::setw(22) << "Token String"
              << std::setw(14) << "Logit"
              << std::setw(14) << "Probability" << "\n";
    std::cout << "------------------------------------------------------------------------\n";

    std::vector<TokenPred> top_tokens;
    for (int i = 0; i < top_k; ++i) {
        int64_t tid = indices[i];
        std::string s = tokenizer.is_loaded() ? tokenizer.decode_token(tid) : "";
        TokenPred p;
        p.rank = i + 1;
        p.token_id = tid;
        p.token_str = s;
        p.logit = host_logits[tid];
        p.prob = probs[tid];
        top_tokens.push_back(p);

        std::cout << std::left << std::setw(6) << p.rank
                  << std::setw(12) << p.token_id
                  << std::setw(22) << ("'" + p.token_str + "'")
                  << std::fixed << std::setprecision(4)
                  << std::setw(14) << p.logit
                  << std::setprecision(6)
                  << std::setw(14) << p.prob << "\n";
    }
    std::cout << "========================================================================\n" << std::endl;

    // Write raw logits binary file (little-endian float32 array)
    {
        std::ofstream out(output_npy, std::ios::binary);
        out.write(reinterpret_cast<const char*>(host_logits.data()), host_logits.size() * sizeof(float));
    }

    // Write JSON summary
    {
        std::ofstream out(output_json);
        out << "{\n";
        out << "  \"engine\": \"xinfer (Intel Arc Pro B60)\",\n";
        out << "  \"quantization\": \"INT4-G128-SYM\",\n";
        out << "  \"prompt_tokens\": [";
        for (size_t i = 0; i < prompt_tokens.size(); ++i) {
            out << prompt_tokens[i] << (i + 1 < prompt_tokens.size() ? ", " : "");
        }
        out << "],\n";
        out << "  \"vocab_size\": " << vocab_size << ",\n";
        out << "  \"top1_token_id\": " << top_tokens[0].token_id << ",\n";
        out << "  \"top1_token_str\": \"" << top_tokens[0].token_str << "\",\n";
        out << "  \"top1_logit\": " << top_tokens[0].logit << ",\n";
        out << "  \"top1_prob\": " << top_tokens[0].prob << ",\n";
        out << "  \"top_predictions\": [\n";
        for (size_t i = 0; i < top_tokens.size(); ++i) {
            const auto& t = top_tokens[i];
            out << "    {\n";
            out << "      \"rank\": " << t.rank << ",\n";
            out << "      \"token_id\": " << t.token_id << ",\n";
            out << "      \"token_str\": \"" << t.token_str << "\",\n";
            out << "      \"logit\": " << t.logit << ",\n";
            out << "      \"probability\": " << t.prob << "\n";
            out << "    }" << (i + 1 < top_tokens.size() ? "," : "") << "\n";
        }
        out << "  ]\n";
        out << "}\n";
    }

    sycl::free(d_logits, q);

    std::cout << "[SUCCESS] Saved engine logits to:\n"
              << "  Raw binary: " << output_npy << "\n"
              << "  JSON:       " << output_json << "\n";
    return 0;
}
