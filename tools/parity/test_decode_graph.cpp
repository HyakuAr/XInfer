#include "core/device.h"
#include "core/arena.h"
#include "core/kv_cache.h"
#include "targets/qwen3_8_27b/weights.h"
#include "targets/qwen3_8/forward.h"
#include "ops/linear.h"
#include "ops/attention.h"
#include "ops/rmsnorm.h"
#include "ops/rope.h"
#include "ops/elementwise.h"
#include "ops/sampling.h"
#include "targets/qwen3_8/linear_attn.h"
#include "artifact/reader.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <iostream>
#include <chrono>

namespace syclex = sycl::ext::oneapi::experimental;
using namespace xinfer;

int main() {
    std::cout << "Testing Decode Graph Capture on Intel Arc Pro B60..." << std::endl;
    auto ctx = core::DeviceContext::create(true);
    sycl::queue& q = ctx->queue();

    auto dev = q.get_device();
    if (!dev.has(sycl::aspect::ext_oneapi_graph)) {
        std::cerr << "Device does not support ext_oneapi_graph aspect!" << std::endl;
        return 1;
    }

    std::string artifact_path = "out/qwen3_8_27b.xinfer";
    std::cout << "Loading artifact: " << artifact_path << std::endl;
    artifact::ArtifactReader reader;
    reader.open(artifact_path);
    auto model = targets::qwen3_8_27b::LoadedModel::load_from_artifact(ctx, reader);
    std::cout << "Model loaded into B60 VRAM." << std::endl;

    const auto& config = model->config();
    core::DeviceArena arena(ctx, 256 * 1024 * 1024);
    core::KVCacheConfig cfg = config.create_kv_cache_config(8192);
    core::KVCache kv_cache(ctx, cfg);
    kv_cache.allocate();

    const int64_t vocab_size = config.vocab_size;
    const int64_t hidden_size = config.hidden_size;
    const int64_t intermediate_size = config.intermediate_size;

    // Fixed USM allocations for decode step
    int64_t* d_token_ids = sycl::malloc_device<int64_t>(1, q);
    int64_t* d_positions = sycl::malloc_device<int64_t>(1, q);
    float* d_logits = sycl::malloc_device<float>(vocab_size, q);

    // Fixed activation scratchpad buffers
    float* act_x = sycl::malloc_device<float>(hidden_size, q);
    float* act_normed = sycl::malloc_device<float>(hidden_size, q);
    float* act_proj_out = sycl::malloc_device<float>(hidden_size, q);
    float* act_mlp_gate = sycl::malloc_device<float>(intermediate_size, q);
    float* act_mlp_up   = sycl::malloc_device<float>(intermediate_size, q);

    float* act_q_gate   = sycl::malloc_device<float>(config.full_q_gate_dim(), q);
    float* act_q        = sycl::malloc_device<float>(config.full_q_dim(), q);
    float* act_k        = sycl::malloc_device<float>(config.full_k_dim(), q);
    float* act_v        = sycl::malloc_device<float>(config.full_v_dim(), q);
    float* act_attn_out = sycl::malloc_device<float>(config.full_out_dim(), q);

    float* act_qkv_raw   = sycl::malloc_device<float>(config.linear_conv_channels, q);
    float* act_qkv_conv  = sycl::malloc_device<float>(config.linear_conv_channels, q);
    float* act_z         = sycl::malloc_device<float>(config.linear_z_dim, q);
    float* act_b         = sycl::malloc_device<float>(config.linear_b_dim, q);
    float* act_a         = sycl::malloc_device<float>(config.linear_a_dim, q);
    float* act_delta_out = sycl::malloc_device<float>(config.linear_out_dim(), q);

    int64_t cur_pos = 10;
    int64_t token_val = 1234;
    q.memcpy(d_token_ids, &token_val, sizeof(int64_t)).wait();
    q.memcpy(d_positions, &cur_pos, sizeof(int64_t)).wait();

    std::cout << "Recording decode kernels into command graph..." << std::endl;
    try {
        syclex::command_graph<syclex::graph_state::modifiable> graph(q.get_context(), dev);

        graph.begin_recording(q);

        // 1. Embedding lookup
        targets::qwen3_8::embed_tokens_lookup(q, act_x, model->d_embed_tokens(), d_token_ids, 1, hidden_size);

        // 2. Loop over layers
        const auto& layers = model->layers();
        size_t full_idx = 0;
        size_t linear_idx = 0;

        const int64_t num_q_heads = config.num_attention_heads;
        const int64_t num_kv_heads = config.num_key_value_heads;
        const int64_t head_dim = config.head_dim;
        const int64_t full_q_gate_dim = config.full_q_gate_dim();
        const int64_t full_k_dim = config.full_k_dim();
        const int64_t full_v_dim = config.full_v_dim();
        const int64_t full_out_dim = config.full_out_dim();
        const float rope_theta = config.rope_theta;
        const int64_t rope_dim = config.rope_dim;

        const int64_t linear_conv_channels = config.linear_conv_channels;
        const int64_t linear_z_dim = config.linear_z_dim;
        const int64_t linear_b_dim = config.linear_b_dim;
        const int64_t linear_a_dim = config.linear_a_dim;
        const int64_t linear_out_dim = config.linear_out_dim();

        for (size_t l = 0; l < layers.size(); ++l) {
            const auto& layer = layers[l];

            ops::rmsnorm(q, act_normed, act_x, layer.d_input_layernorm, 1, hidden_size);

            if (layer.layer_type == "full_attention") {
                ops::linear_int4(q, act_q_gate, act_normed,
                                 static_cast<const uint8_t*>(layer.q_proj.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.q_proj.d_scales),
                                 nullptr, 1, full_q_gate_dim, hidden_size);
                ops::linear_int4(q, act_k, act_normed,
                                 static_cast<const uint8_t*>(layer.k_proj.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.k_proj.d_scales),
                                 nullptr, 1, full_k_dim, hidden_size);
                ops::linear_int4(q, act_v, act_normed,
                                 static_cast<const uint8_t*>(layer.v_proj.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.v_proj.d_scales),
                                 nullptr, 1, full_v_dim, hidden_size);

                q.parallel_for(sycl::range<2>(1, num_q_heads), [=](sycl::id<2> idx) {
                    int64_t t = idx[0];
                    int64_t h = idx[1];
                    for (int d = 0; d < head_dim; ++d) {
                        act_q[(t * num_q_heads + h) * head_dim + d] = act_q_gate[t * full_q_gate_dim + h * (2 * head_dim) + d];
                    }
                });

                ops::rmsnorm(q, act_q, act_q, layer.d_q_norm, num_q_heads, head_dim);
                ops::rmsnorm(q, act_k, act_k, layer.d_k_norm, num_kv_heads, head_dim);

                ops::rope(q, act_q, act_k, 1, num_q_heads, num_kv_heads, head_dim, d_positions, rope_theta, rope_dim);

                ops::attention_write_kv_cache(q, kv_cache.k_cache(full_idx), kv_cache.v_cache(full_idx),
                                              act_k, act_v, cur_pos, 1, num_kv_heads, head_dim);

                ops::sdpa_causal_cached(q, act_attn_out, act_q,
                                        kv_cache.k_cache(full_idx), kv_cache.v_cache(full_idx),
                                        cur_pos, 1, num_q_heads, num_kv_heads, head_dim);

                q.parallel_for(sycl::range<2>(1, num_q_heads), [=](sycl::id<2> idx) {
                    int64_t t = idx[0];
                    int64_t h = idx[1];
                    for (int d = 0; d < head_dim; ++d) {
                        float gate_val = act_q_gate[t * full_q_gate_dim + h * (2 * head_dim) + head_dim + d];
                        float sig = 1.0f / (1.0f + sycl::exp(-gate_val));
                        act_attn_out[(t * num_q_heads + h) * head_dim + d] *= sig;
                    }
                });

                ops::linear_int4(q, act_proj_out, act_attn_out,
                                 static_cast<const uint8_t*>(layer.o_proj.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.o_proj.d_scales),
                                 nullptr, 1, hidden_size, full_out_dim);
                full_idx++;
            } else {
                ops::linear_int4(q, act_qkv_raw, act_normed,
                                 static_cast<const uint8_t*>(layer.in_proj_qkv.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.in_proj_qkv.d_scales),
                                 nullptr, 1, linear_conv_channels, hidden_size);
                ops::linear_int4(q, act_z, act_normed,
                                 static_cast<const uint8_t*>(layer.in_proj_z.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.in_proj_z.d_scales),
                                 nullptr, 1, linear_z_dim, hidden_size);
                ops::linear_int4(q, act_b, act_normed,
                                 static_cast<const uint8_t*>(layer.in_proj_b.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.in_proj_b.d_scales),
                                 nullptr, 1, linear_b_dim, hidden_size);
                ops::linear_int4(q, act_a, act_normed,
                                 static_cast<const uint8_t*>(layer.in_proj_a.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.in_proj_a.d_scales),
                                 nullptr, 1, linear_a_dim, hidden_size);

                targets::qwen3_8::causal_conv1d_silu(q, act_qkv_conv, act_qkv_raw, layer.d_conv1d_weight, 1,
                                                     kv_cache.conv_state(linear_idx));

                targets::qwen3_8::recurrent_gated_delta_net(q, act_delta_out, act_qkv_conv, act_z, act_b, act_a,
                                                            layer.d_A_log, layer.d_dt_bias, layer.d_norm_weight,
                                                            kv_cache.linear_state(linear_idx), 1, false);

                ops::linear_int4(q, act_proj_out, act_delta_out,
                                 static_cast<const uint8_t*>(layer.out_proj.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.out_proj.d_scales),
                                 nullptr, 1, hidden_size, linear_out_dim);
                linear_idx++;
            }

            ops::add_inplace(q, act_x, act_proj_out, hidden_size);

            ops::rmsnorm(q, act_normed, act_x, layer.d_post_attention_layernorm, 1, hidden_size);

            ops::linear_int4(q, act_mlp_gate, act_normed,
                             static_cast<const uint8_t*>(layer.gate_proj.d_weights_int4),
                             static_cast<const sycl::half*>(layer.gate_proj.d_scales),
                             nullptr, 1, intermediate_size, hidden_size);
            ops::linear_int4(q, act_mlp_up, act_normed,
                             static_cast<const uint8_t*>(layer.up_proj.d_weights_int4),
                             static_cast<const sycl::half*>(layer.up_proj.d_scales),
                             nullptr, 1, intermediate_size, hidden_size);

            ops::swiglu(q, act_mlp_gate, act_mlp_gate, act_mlp_up, intermediate_size);

            ops::linear_int4(q, act_proj_out, act_mlp_gate,
                             static_cast<const uint8_t*>(layer.down_proj.d_weights_int4),
                             static_cast<const sycl::half*>(layer.down_proj.d_scales),
                             nullptr, 1, hidden_size, intermediate_size);

            ops::add_inplace(q, act_x, act_proj_out, hidden_size);
        }

        // LM head projection
        ops::rmsnorm(q, act_normed, act_x, model->d_final_norm(), 1, hidden_size);
        const auto& lm_head = model->lm_head();
        ops::linear_int4(q, d_logits, act_normed,
                         static_cast<const uint8_t*>(lm_head.d_weights_int4),
                         static_cast<const sycl::half*>(lm_head.d_scales),
                         nullptr, 1, vocab_size, hidden_size);

        graph.end_recording(q);

        std::cout << "Finalizing executable graph for all 64 layers..." << std::endl;
        auto exec_graph = graph.finalize();
        std::cout << "SUCCESS! Executable graph finalized." << std::endl;

        // Execute via graph replay
        std::cout << "Replaying graph 10 times..." << std::endl;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) {
            q.ext_oneapi_graph(exec_graph);
        }
        q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 10.0;
        std::cout << "Average decode step latency via Graph Replay: " << ms << " ms (" 
                  << (1000.0 / ms) << " tokens/sec)" << std::endl;

        int64_t next_tok = ops::argmax(q, d_logits, vocab_size);
        std::cout << "Sampled next token ID: " << next_tok << std::endl;

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
        return 1;
    }

    sycl::free(d_token_ids, q);
    sycl::free(d_positions, q);
    sycl::free(d_logits, q);
    sycl::free(act_x, q);
    sycl::free(act_normed, q);
    sycl::free(act_proj_out, q);
    sycl::free(act_mlp_gate, q);
    sycl::free(act_mlp_up, q);
    sycl::free(act_q_gate, q);
    sycl::free(act_q, q);
    sycl::free(act_k, q);
    sycl::free(act_v, q);
    sycl::free(act_attn_out, q);
    sycl::free(act_qkv_raw, q);
    sycl::free(act_qkv_conv, q);
    sycl::free(act_z, q);
    sycl::free(act_b, q);
    sycl::free(act_a, q);
    sycl::free(act_delta_out, q);

    return 0;
}
