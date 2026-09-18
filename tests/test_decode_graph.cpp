#include "core/device.h"
#include "core/arena.h"
#include "core/kv_cache.h"
#include "ops/linear.h"
#include "ops/attention.h"
#include "ops/rmsnorm.h"
#include "targets/qwen3_8/linear_attn.h"
#include "targets/qwen3_8/forward.h"
#include "targets/qwen3_8_27b/weights.h"
#include "artifact/writer.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <filesystem>

namespace syclex = sycl::ext::oneapi::experimental;
using namespace xinfer;

void test_linear_attn_shape_validation(sycl::queue& q) {
    std::cout << "\nTesting linear attention shape constants and validation..." << std::endl;

    // 1. Verify causal_conv1d_silu rejects invalid num_channels <= 0
    bool caught_conv = false;
    try {
        targets::qwen3_8::causal_conv1d_silu(q, (float*)nullptr, (const float*)nullptr, (const float*)nullptr, 1, nullptr, 0);
    } catch (const std::invalid_argument& e) {
        caught_conv = true;
        std::cout << "  -> PASSED: causal_conv1d_silu threw as expected on num_channels=0: " << e.what() << std::endl;
    }
    assert(caught_conv);

    // 2. Verify recurrent_gated_delta_net rejects shape mismatch against ModelConfig
    bool caught_recurrent = false;
    try {
        targets::qwen3_8::recurrent_gated_delta_net(
            q, (float*)nullptr, (const float*)nullptr, (const float*)nullptr, (const float*)nullptr, (const float*)nullptr,
            (const float*)nullptr, (const float*)nullptr, (const float*)nullptr, (float*)nullptr, 1, false,
            32, 16, 128, 128, 10240, 6144 // num_v_heads = 32 instead of 48
        );
    } catch (const std::invalid_argument& e) {
        caught_recurrent = true;
        std::cout << "  -> PASSED: recurrent_gated_delta_net threw on num_v_heads mismatch: " << e.what() << std::endl;
    }
    assert(caught_recurrent);
}

void test_metadata_validation(std::shared_ptr<core::DeviceContext> ctx) {
    std::cout << "\nTesting LoadedModel::load_from_artifact metadata validation..." << std::endl;

    const std::string dummy_art = "test_meta_val_dummy.xinfer";
    if (std::filesystem::exists(dummy_art)) {
        std::filesystem::remove(dummy_art);
    }

    // 1. Missing required property (missing "full_attention_interval")
    {
        artifact::ArtifactMetadata meta;
        meta.model_name = "Qwen/Qwen3.8-27B";
        meta.quant_scheme = "INT4-G128-SYM";
        meta.tokenizer_type = "qwen3_8_tiktoken";
        meta.properties["hidden_size"] = "5120";
        meta.properties["intermediate_size"] = "17408";
        meta.properties["num_hidden_layers"] = "64";
        meta.properties["num_attention_heads"] = "24";
        meta.properties["num_key_value_heads"] = "4";
        meta.properties["head_dim"] = "256";
        meta.properties["vocab_size"] = "248320";
        meta.properties["rms_norm_eps"] = "0.000001";
        meta.properties["rope_theta"] = "10000000.0";
        // "full_attention_interval" is intentionally missing

        artifact::ArtifactWriter writer;
        writer.set_metadata(meta);
        std::string err;
        assert(writer.write_to_file(dummy_art, &err));

        artifact::ArtifactReader reader;
        assert(reader.open(dummy_art, &err));

        std::string load_err;
        auto model = targets::qwen3_8_27b::LoadedModel::load_from_artifact(ctx, reader, &load_err);
        assert(model == nullptr);
        assert(load_err.find("missing required property 'full_attention_interval'") != std::string::npos);
        std::cout << "  -> PASSED: Successfully caught missing 'full_attention_interval': " << load_err << std::endl;
        reader.close();
        std::filesystem::remove(dummy_art);
    }

    // 2. Missing "hidden_size"
    {
        artifact::ArtifactMetadata meta;
        meta.model_name = "Qwen/Qwen3.8-27B";
        meta.quant_scheme = "INT4-G128-SYM";
        meta.tokenizer_type = "qwen3_8_tiktoken";
        meta.properties["full_attention_interval"] = "4";
        // "hidden_size" is missing

        artifact::ArtifactWriter writer;
        writer.set_metadata(meta);
        std::string err;
        assert(writer.write_to_file(dummy_art, &err));

        artifact::ArtifactReader reader;
        assert(reader.open(dummy_art, &err));

        std::string load_err;
        auto model = targets::qwen3_8_27b::LoadedModel::load_from_artifact(ctx, reader, &load_err);
        assert(model == nullptr);
        assert(load_err.find("missing required property 'hidden_size'") != std::string::npos);
        std::cout << "  -> PASSED: Successfully caught missing 'hidden_size': " << load_err << std::endl;
        reader.close();
        std::filesystem::remove(dummy_art);
    }
}

void test_embed_tokens_lookup_bounds(sycl::queue& q) {
    std::cout << "\nTesting embed_tokens_lookup bounds checking..." << std::endl;

    constexpr int64_t vocab_size = 8;
    constexpr int64_t hidden_size = 16;
    constexpr size_t total_elements = vocab_size * hidden_size;

    // Construct a test BF16 embedding table: table[v, d] has bit pattern for (float)(v * 16 + d)
    std::vector<uint16_t> h_table(total_elements);
    for (int64_t v = 0; v < vocab_size; ++v) {
        for (int64_t d = 0; d < hidden_size; ++d) {
            float val = static_cast<float>(v * 16 + d);
            uint32_t bits;
            __builtin_memcpy(&bits, &val, sizeof(float));
            h_table[v * hidden_size + d] = static_cast<uint16_t>(bits >> 16);
        }
    }

    uint16_t* d_table = sycl::malloc_device<uint16_t>(total_elements, q);
    q.memcpy(d_table, h_table.data(), total_elements * sizeof(uint16_t)).wait();

    // 4 test tokens:
    // token 0: valid (v = 2)
    // token 1: out-of-bounds negative (v = -1)
    // token 2: valid (v = 5)
    // token 3: out-of-bounds exceeds vocab_size (v = 12)
    constexpr int64_t num_tokens = 4;
    std::vector<int64_t> h_token_ids = {2, -1, 5, 12};
    int64_t* d_token_ids = sycl::malloc_device<int64_t>(num_tokens, q);
    q.memcpy(d_token_ids, h_token_ids.data(), num_tokens * sizeof(int64_t)).wait();

    // 1. Test FP32 lookup
    float* d_out_fp32 = sycl::malloc_device<float>(num_tokens * hidden_size, q);
    std::vector<float> canary_fp32(num_tokens * hidden_size, -999.0f);
    q.memcpy(d_out_fp32, canary_fp32.data(), canary_fp32.size() * sizeof(float)).wait();

    targets::qwen3_8::embed_tokens_lookup(q, d_out_fp32, d_table, d_token_ids, num_tokens, hidden_size, vocab_size).wait();

    std::vector<float> h_out_fp32(num_tokens * hidden_size);
    q.memcpy(h_out_fp32.data(), d_out_fp32, h_out_fp32.size() * sizeof(float)).wait();

    for (int64_t t = 0; t < num_tokens; ++t) {
        int64_t tok = h_token_ids[t];
        for (int64_t d = 0; d < hidden_size; ++d) {
            float actual = h_out_fp32[t * hidden_size + d];
            if (tok >= 0 && tok < vocab_size) {
                float expected = static_cast<float>(tok * 16 + d);
                assert(std::abs(actual - expected) < 1e-3f);
            } else {
                // Out of bounds tokens must be zeroed by defensive skip
                assert(actual == 0.0f);
            }
        }
    }
    std::cout << "  -> PASSED: FP32 embed_tokens_lookup correctly reads in-bounds and zeroes out-of-bounds tokens" << std::endl;

    // 2. Test FP16 (sycl::half) lookup
    sycl::half* d_out_fp16 = sycl::malloc_device<sycl::half>(num_tokens * hidden_size, q);
    std::vector<sycl::half> canary_fp16(num_tokens * hidden_size, static_cast<sycl::half>(-999.0f));
    q.memcpy(d_out_fp16, canary_fp16.data(), canary_fp16.size() * sizeof(sycl::half)).wait();

    targets::qwen3_8::embed_tokens_lookup(q, d_out_fp16, d_table, d_token_ids, num_tokens, hidden_size, vocab_size).wait();

    std::vector<sycl::half> h_out_fp16(num_tokens * hidden_size);
    q.memcpy(h_out_fp16.data(), d_out_fp16, h_out_fp16.size() * sizeof(sycl::half)).wait();

    for (int64_t t = 0; t < num_tokens; ++t) {
        int64_t tok = h_token_ids[t];
        for (int64_t d = 0; d < hidden_size; ++d) {
            float actual = static_cast<float>(h_out_fp16[t * hidden_size + d]);
            if (tok >= 0 && tok < vocab_size) {
                float expected = static_cast<float>(tok * 16 + d);
                assert(std::abs(actual - expected) < 1e-1f);
            } else {
                // Out of bounds tokens must be zeroed by defensive skip
                assert(actual == 0.0f);
            }
        }
    }
    std::cout << "  -> PASSED: FP16 embed_tokens_lookup correctly reads in-bounds and zeroes out-of-bounds tokens" << std::endl;

    sycl::free(d_table, q);
    sycl::free(d_token_ids, q);
    sycl::free(d_out_fp32, q);
    sycl::free(d_out_fp16, q);
}

void test_causal_conv1d_chunk_boundaries(sycl::queue& q) {
    std::cout << "\nTesting causal_conv1d_silu seq_len=1 and seq_len=2 chunk boundaries..." << std::endl;
    constexpr int64_t num_channels = 10240;

    // Allocate USM device buffers
    float* conv_w = sycl::malloc_device<float>(num_channels * 4, q);
    float* conv_state = sycl::malloc_device<float>(3 * num_channels, q);
    float* in_qkv = sycl::malloc_device<float>(2 * num_channels, q);
    float* out_qkv = sycl::malloc_device<float>(2 * num_channels, q);

    // Fill conv_w: weights [0.1, 0.2, 0.3, 0.4] per channel
    std::vector<float> h_conv_w(num_channels * 4);
    for (size_t c = 0; c < num_channels; ++c) {
        h_conv_w[c * 4 + 0] = 0.1f;
        h_conv_w[c * 4 + 1] = 0.2f;
        h_conv_w[c * 4 + 2] = 0.3f;
        h_conv_w[c * 4 + 3] = 0.4f;
    }
    q.memcpy(conv_w, h_conv_w.data(), h_conv_w.size() * sizeof(float));

    // Zero out conv_state
    q.fill(conv_state, 0.0f, 3 * num_channels);

    // 1. Test seq_len = 1: pass in token 0 with value 1.0f
    q.fill(in_qkv, 1.0f, num_channels);
    q.wait();

    targets::qwen3_8::causal_conv1d_silu(q, out_qkv, in_qkv, conv_w, 1, conv_state).wait();

    // Verify output for seq_len = 1: sum = 0.4 * 1.0 = 0.4. silu = 0.4 / (1 + exp(-0.4))
    std::vector<float> h_out(num_channels);
    q.memcpy(h_out.data(), out_qkv, num_channels * sizeof(float)).wait();
    float expected_silu_1 = 0.4f / (1.0f + std::exp(-0.4f));
    assert(std::abs(h_out[0] - expected_silu_1) < 1e-4f);

    // Verify conv_state after seq_len = 1: [s1, s2, in_val] = [0.0, 0.0, 1.0]
    std::vector<float> h_state(3 * num_channels);
    q.memcpy(h_state.data(), conv_state, 3 * num_channels * sizeof(float)).wait();
    assert(std::abs(h_state[0 * num_channels + 0] - 0.0f) < 1e-4f);
    assert(std::abs(h_state[1 * num_channels + 0] - 0.0f) < 1e-4f);
    assert(std::abs(h_state[2 * num_channels + 0] - 1.0f) < 1e-4f);

    // 2. Test seq_len = 2: pass in tokens with values [2.0f, 3.0f]
    std::vector<float> h_in_2(2 * num_channels);
    for (size_t c = 0; c < num_channels; ++c) {
        h_in_2[0 * num_channels + c] = 2.0f;
        h_in_2[1 * num_channels + c] = 3.0f;
    }
    q.memcpy(in_qkv, h_in_2.data(), h_in_2.size() * sizeof(float)).wait();

    targets::qwen3_8::causal_conv1d_silu(q, out_qkv, in_qkv, conv_w, 2, conv_state).wait();

    // Verify output:
    // For t = 0 (val = 2.0f):
    // past timesteps: -3 -> s0 (0.0), -2 -> s1 (0.0), -1 -> s2 (1.0), 0 -> 2.0
    // sum0 = 0.1*0 + 0.2*0 + 0.3*1.0 + 0.4*2.0 = 0.3 + 0.8 = 1.1
    // For t = 1 (val = 3.0f):
    // past timesteps: -2 -> s1 (0.0), -1 -> s2 (1.0), 0 -> 2.0, 1 -> 3.0
    // sum1 = 0.1*0 + 0.2*1.0 + 0.3*2.0 + 0.4*3.0 = 0.2 + 0.6 + 1.2 = 2.0
    std::vector<float> h_out_2(2 * num_channels);
    q.memcpy(h_out_2.data(), out_qkv, 2 * num_channels * sizeof(float)).wait();
    float exp_silu_t0 = 1.1f / (1.0f + std::exp(-1.1f));
    float exp_silu_t1 = 2.0f / (1.0f + std::exp(-2.0f));
    assert(std::abs(h_out_2[0 * num_channels + 0] - exp_silu_t0) < 1e-4f);
    assert(std::abs(h_out_2[1 * num_channels + 0] - exp_silu_t1) < 1e-4f);

    // Verify conv_state after seq_len = 2:
    // state shifts by 2:
    // conv_state[0] = old conv_state[2] = 1.0f
    // conv_state[1] = in_qkv[0] = 2.0f
    // conv_state[2] = in_qkv[1] = 3.0f
    q.memcpy(h_state.data(), conv_state, 3 * num_channels * sizeof(float)).wait();
    assert(std::abs(h_state[0 * num_channels + 0] - 1.0f) < 1e-4f);
    assert(std::abs(h_state[1 * num_channels + 0] - 2.0f) < 1e-4f);
    assert(std::abs(h_state[2 * num_channels + 0] - 3.0f) < 1e-4f);

    std::cout << "  -> PASSED: Causal Conv1D chunk-boundary state transitions verified for seq_len=1 and seq_len=2." << std::endl;

    sycl::free(conv_w, q);
    sycl::free(conv_state, q);
    sycl::free(in_qkv, q);
    sycl::free(out_qkv, q);
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << " xinfer M8 Level Zero / SYCL Decode Graph Unit Test" << std::endl;
    std::cout << " Testing graph capture & dynamic position replay on B60" << std::endl;
    std::cout << "==========================================================" << std::endl;

    auto ctx = core::DeviceContext::create(true);
    sycl::queue& q = ctx->queue();
    auto dev = q.get_device();

    if (!dev.has(sycl::aspect::ext_oneapi_graph)) {
        std::cerr << "Device does not support ext_oneapi_graph aspect!" << std::endl;
        return 1;
    }

    constexpr size_t num_q_heads = 24;
    constexpr size_t num_kv_heads = 4;
    constexpr size_t head_dim = 256;
    constexpr size_t max_seq = 64;

    // USM device allocations with guaranteed address stability
    int64_t* d_cur_pos = sycl::malloc_device<int64_t>(1, q);
    float* d_Q = sycl::malloc_device<float>(num_q_heads * head_dim, q);
    float* d_K = sycl::malloc_device<float>(num_kv_heads * head_dim, q);
    float* d_V = sycl::malloc_device<float>(num_kv_heads * head_dim, q);
    float* d_out = sycl::malloc_device<float>(num_q_heads * head_dim, q);

    sycl::half* k_cache = sycl::malloc_device<sycl::half>(max_seq * num_kv_heads * head_dim, q);
    sycl::half* v_cache = sycl::malloc_device<sycl::half>(max_seq * num_kv_heads * head_dim, q);

    // Initialize Q, K, V, cache, and position
    q.fill(d_Q, 0.1f, num_q_heads * head_dim);
    q.fill(d_K, 0.05f, num_kv_heads * head_dim);
    q.fill(d_V, 0.5f, num_kv_heads * head_dim);
    q.fill(k_cache, sycl::half{0.0f}, max_seq * num_kv_heads * head_dim);
    q.fill(v_cache, sycl::half{0.0f}, max_seq * num_kv_heads * head_dim);
    int64_t initial_pos = 0;
    q.memcpy(d_cur_pos, &initial_pos, sizeof(int64_t));
    q.wait();

    // 1. Capture graph containing KV-cache write and cached attention
    syclex::command_graph<syclex::graph_state::modifiable> graph(q.get_context(), dev);
    graph.begin_recording(q);

    ops::attention_write_kv_cache_dynamic(q, k_cache, v_cache, d_K, d_V, d_cur_pos, 1, num_kv_heads, head_dim);
    ops::sdpa_causal_cached_dynamic(q, d_out, d_Q, k_cache, v_cache, d_cur_pos, 1, num_q_heads, num_kv_heads, head_dim);

    graph.end_recording(q);
    auto exec_graph = graph.finalize();
    std::cout << "Successfully captured Level Zero command graph!" << std::endl;

    // 2. Replay graph across 5 sequential decode positions (pos 0 .. 4)
    std::cout << "Replaying captured graph across positions 0 .. 4..." << std::endl;
    for (int64_t pos = 0; pos < 5; ++pos) {
        int64_t host_p = pos;
        q.memcpy(d_cur_pos, &host_p, sizeof(int64_t)).wait();
        q.ext_oneapi_graph(exec_graph);
        q.wait();
    }

    // Verify output after 5 steps: since all V values are 0.5f, output must normalize to 0.5f
    std::vector<float> h_out(num_q_heads * head_dim);
    q.memcpy(h_out.data(), d_out, h_out.size() * sizeof(float)).wait();

    std::cout << "Verified output[0] = " << h_out[0] << " (expected 0.5)" << std::endl;
    if (std::abs(h_out[0] - 0.5f) >= 1e-3f) {
        std::cerr << "FAILED: Output mismatch! Expected 0.5, got " << h_out[0] << std::endl;
        return 1;
    }
    std::cout << "  -> PASSED: Graph replay with dynamic position pointer verified." << std::endl;

    sycl::free(d_cur_pos, q);
    sycl::free(d_Q, q);
    sycl::free(d_K, q);
    sycl::free(d_V, q);
    sycl::free(d_out, q);
    sycl::free(k_cache, q);
    sycl::free(v_cache, q);

    // 3. Run chunk-boundary test
    test_causal_conv1d_chunk_boundaries(q);

    // 4. Run linear attention shape validation test
    test_linear_attn_shape_validation(q);

    // 5. Run metadata validation test
    test_metadata_validation(ctx);

    // 6. Run embedding lookup bounds checking test
    test_embed_tokens_lookup_bounds(q);

    std::cout << "\n==========================================================" << std::endl;
    std::cout << " ALL M8 DECODE GRAPH & LINEAR ATTN TESTS PASSED ON B60!" << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}
