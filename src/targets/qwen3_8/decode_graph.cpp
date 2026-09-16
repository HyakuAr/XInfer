// Citing vendor documentation per AGENTS.md §5:
// - docs/vendor/level-zero-command-lists.md (lines 10-31, 32-50):
//   Record-once, replay-many pattern using Level Zero regular command lists.
//   Under oneAPI / Level Zero, sycl::ext::oneapi::experimental::command_graph
//   compiles down to regular command lists (zeCommandListCreate + zeCommandListClose +
//   zeCommandQueueExecuteCommandLists) eliminating host launch overhead for fixed-shape steps.
// - docs/vendor/xe-gpu-architecture.md (lines 22-39):
//   Intel Arc Pro B60 (Battlemage Xe2-HPG, device ID 0xE211): 20 Xe-cores, 24 GB VRAM.
// - Address-stability guarantees (per ROADMAP.md M8 & AGENTS.md §7):
//   All USM device allocations (weights, activations, KV cache containers, token ID and
//   position pointers) remain strictly resident at constant virtual addresses across
//   decode steps. Only the scalar values stored in d_token_ids_ and d_positions_ are updated.

#include "decode_graph.h"
#include "forward.h"
#include "linear_attn.h"
#include "ops/linear.h"
#include "ops/attention.h"
#include "ops/rmsnorm.h"
#include "ops/rope.h"
#include "ops/elementwise.h"
#include "ops/sampling.h"

#include <iostream>

namespace syclex = sycl::ext::oneapi::experimental;

namespace xinfer::targets::qwen3_8 {

DecodeGraph::DecodeGraph(std::shared_ptr<core::DeviceContext> ctx,
                         const qwen3_8_27b::LoadedModel& model,
                         core::KVCache& kv_cache)
    : ctx_(std::move(ctx)), model_(model), kv_cache_(kv_cache) {
    sycl::queue& q = ctx_->queue();

    constexpr int64_t vocab_size = 248320;
    constexpr int64_t hidden_size = 5120;
    constexpr int64_t intermediate_size = 17408;

    // Allocate fixed USM device memory pointers with guaranteed address stability
    d_token_ids_ = sycl::malloc_device<int64_t>(1, q);
    d_positions_ = sycl::malloc_device<int64_t>(1, q);
    d_logits_    = sycl::malloc_device<float>(vocab_size, q);

    act_x_        = sycl::malloc_device<float>(hidden_size, q);
    act_normed_   = sycl::malloc_device<float>(hidden_size, q);
    act_proj_out_ = sycl::malloc_device<float>(hidden_size, q);
    act_mlp_gate_ = sycl::malloc_device<float>(intermediate_size, q);
    act_mlp_up_   = sycl::malloc_device<float>(intermediate_size, q);

    act_q_gate_   = sycl::malloc_device<float>(12288, q);
    act_q_        = sycl::malloc_device<float>(6144, q);
    act_k_        = sycl::malloc_device<float>(1024, q);
    act_v_        = sycl::malloc_device<float>(1024, q);
    act_attn_out_ = sycl::malloc_device<float>(6144, q);

    act_qkv_raw_   = sycl::malloc_device<float>(10240, q);
    act_qkv_conv_  = sycl::malloc_device<float>(10240, q);
    act_z_         = sycl::malloc_device<float>(6144, q);
    act_b_         = sycl::malloc_device<float>(48, q);
    act_a_         = sycl::malloc_device<float>(48, q);
    act_delta_out_ = sycl::malloc_device<float>(6144, q);
}

DecodeGraph::~DecodeGraph() {
    sycl::queue& q = ctx_->queue();
    exec_graph_.reset();

    if (d_token_ids_) sycl::free(d_token_ids_, q);
    if (d_positions_) sycl::free(d_positions_, q);
    if (d_logits_)    sycl::free(d_logits_, q);

    if (act_x_)        sycl::free(act_x_, q);
    if (act_normed_)   sycl::free(act_normed_, q);
    if (act_proj_out_) sycl::free(act_proj_out_, q);
    if (act_mlp_gate_) sycl::free(act_mlp_gate_, q);
    if (act_mlp_up_)   sycl::free(act_mlp_up_, q);

    if (act_q_gate_)   sycl::free(act_q_gate_, q);
    if (act_q_)        sycl::free(act_q_, q);
    if (act_k_)        sycl::free(act_k_, q);
    if (act_v_)        sycl::free(act_v_, q);
    if (act_attn_out_) sycl::free(act_attn_out_, q);

    if (act_qkv_raw_)   sycl::free(act_qkv_raw_, q);
    if (act_qkv_conv_)  sycl::free(act_qkv_conv_, q);
    if (act_z_)         sycl::free(act_z_, q);
    if (act_b_)         sycl::free(act_b_, q);
    if (act_a_)         sycl::free(act_a_, q);
    if (act_delta_out_) sycl::free(act_delta_out_, q);
}

bool DecodeGraph::capture() {
    sycl::queue& q = ctx_->queue();
    auto dev = q.get_device();

    if (!dev.has(sycl::aspect::ext_oneapi_graph)) {
        std::cerr << "[DecodeGraph] Device does not support ext_oneapi_graph aspect." << std::endl;
        return false;
    }

    constexpr int64_t vocab_size = 248320;
    constexpr int64_t hidden_size = 5120;
    constexpr int64_t intermediate_size = 17408;

    int64_t initial_token = 0;
    int64_t initial_pos = 0;
    q.memcpy(d_token_ids_, &initial_token, sizeof(int64_t)).wait();
    q.memcpy(d_positions_, &initial_pos, sizeof(int64_t)).wait();

    try {
        syclex::command_graph<syclex::graph_state::modifiable> graph(q.get_context(), dev);

        graph.begin_recording(q);

        // 1. Embedding lookup
        embed_tokens_lookup(q, act_x_, model_.d_embed_tokens(), d_token_ids_, 1, hidden_size);

        // 2. Loop over 64 layers
        const auto& layers = model_.layers();
        size_t full_idx = 0;
        size_t linear_idx = 0;

        for (size_t l = 0; l < layers.size(); ++l) {
            const auto& layer = layers[l];

            ops::rmsnorm(q, act_normed_, act_x_, layer.d_input_layernorm, 1, hidden_size);

            if (layer.layer_type == "full_attention") {
                ops::linear_int4(q, act_q_gate_, act_normed_,
                                 static_cast<const uint8_t*>(layer.q_proj.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.q_proj.d_scales),
                                 nullptr, 1, 12288, hidden_size);
                ops::linear_int4(q, act_k_, act_normed_,
                                 static_cast<const uint8_t*>(layer.k_proj.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.k_proj.d_scales),
                                 nullptr, 1, 1024, hidden_size);
                ops::linear_int4(q, act_v_, act_normed_,
                                 static_cast<const uint8_t*>(layer.v_proj.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.v_proj.d_scales),
                                 nullptr, 1, 1024, hidden_size);

                float* q_ptr = act_q_;
                float* q_gate_ptr = act_q_gate_;
                float* attn_out_ptr = act_attn_out_;

                q.parallel_for(sycl::range<2>(1, 24), [=](sycl::id<2> idx) {
                    int64_t t = idx[0];
                    int64_t h = idx[1];
                    for (int d = 0; d < 256; ++d) {
                        q_ptr[(t * 24 + h) * 256 + d] = q_gate_ptr[t * 12288 + h * 512 + d];
                    }
                });

                ops::rmsnorm(q, act_q_, act_q_, layer.d_q_norm, 24, 256);
                ops::rmsnorm(q, act_k_, act_k_, layer.d_k_norm, 4, 256);

                ops::rope(q, act_q_, act_k_, 1, 24, 4, 256, d_positions_, 10000000.0f, 64);

                // Use d_positions_ as dynamic device pointer for KV-cache write and SDPA
                ops::attention_write_kv_cache_dynamic(q, kv_cache_.k_cache(full_idx), kv_cache_.v_cache(full_idx),
                                                      act_k_, act_v_, d_positions_, 1, 4, 256,
                                                      static_cast<int64_t>(kv_cache_.max_seq_len()));

                ops::sdpa_causal_cached_dynamic(q, act_attn_out_, act_q_,
                                                kv_cache_.k_cache(full_idx), kv_cache_.v_cache(full_idx),
                                                d_positions_, 1, 24, 4, 256, 0.0f,
                                                static_cast<int64_t>(kv_cache_.max_seq_len()));

                q.parallel_for(sycl::range<2>(1, 24), [=](sycl::id<2> idx) {
                    int64_t t = idx[0];
                    int64_t h = idx[1];
                    for (int d = 0; d < 256; ++d) {
                        float gate_val = q_gate_ptr[t * 12288 + h * 512 + 256 + d];
                        float sig = 1.0f / (1.0f + sycl::exp(-gate_val));
                        attn_out_ptr[(t * 24 + h) * 256 + d] *= sig;
                    }
                });

                ops::linear_int4(q, act_proj_out_, act_attn_out_,
                                 static_cast<const uint8_t*>(layer.o_proj.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.o_proj.d_scales),
                                 nullptr, 1, hidden_size, 6144);
                full_idx++;
            } else {
                ops::linear_int4(q, act_qkv_raw_, act_normed_,
                                 static_cast<const uint8_t*>(layer.in_proj_qkv.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.in_proj_qkv.d_scales),
                                 nullptr, 1, 10240, hidden_size);
                ops::linear_int4(q, act_z_, act_normed_,
                                 static_cast<const uint8_t*>(layer.in_proj_z.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.in_proj_z.d_scales),
                                 nullptr, 1, 6144, hidden_size);
                ops::linear_int4(q, act_b_, act_normed_,
                                 static_cast<const uint8_t*>(layer.in_proj_b.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.in_proj_b.d_scales),
                                 nullptr, 1, 48, hidden_size);
                ops::linear_int4(q, act_a_, act_normed_,
                                 static_cast<const uint8_t*>(layer.in_proj_a.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.in_proj_a.d_scales),
                                 nullptr, 1, 48, hidden_size);

                causal_conv1d_silu(q, act_qkv_conv_, act_qkv_raw_, layer.d_conv1d_weight, 1,
                                   kv_cache_.conv_state(linear_idx));

                recurrent_gated_delta_net(q, act_delta_out_, act_qkv_conv_, act_z_, act_b_, act_a_,
                                          layer.d_A_log, layer.d_dt_bias, layer.d_norm_weight,
                                          kv_cache_.linear_state(linear_idx), 1, false);

                ops::linear_int4(q, act_proj_out_, act_delta_out_,
                                 static_cast<const uint8_t*>(layer.out_proj.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.out_proj.d_scales),
                                 nullptr, 1, hidden_size, 6144);
                linear_idx++;
            }

            ops::add_inplace(q, act_x_, act_proj_out_, hidden_size);

            ops::rmsnorm(q, act_normed_, act_x_, layer.d_post_attention_layernorm, 1, hidden_size);

            ops::linear_int4(q, act_mlp_gate_, act_normed_,
                             static_cast<const uint8_t*>(layer.gate_proj.d_weights_int4),
                             static_cast<const sycl::half*>(layer.gate_proj.d_scales),
                             nullptr, 1, intermediate_size, hidden_size);
            ops::linear_int4(q, act_mlp_up_, act_normed_,
                             static_cast<const uint8_t*>(layer.up_proj.d_weights_int4),
                             static_cast<const sycl::half*>(layer.up_proj.d_scales),
                             nullptr, 1, intermediate_size, hidden_size);

            ops::swiglu(q, act_mlp_gate_, act_mlp_gate_, act_mlp_up_, intermediate_size);

            ops::linear_int4(q, act_proj_out_, act_mlp_gate_,
                             static_cast<const uint8_t*>(layer.down_proj.d_weights_int4),
                             static_cast<const sycl::half*>(layer.down_proj.d_scales),
                             nullptr, 1, hidden_size, intermediate_size);

            ops::add_inplace(q, act_x_, act_proj_out_, hidden_size);
        }

        // LM head projection
        ops::rmsnorm(q, act_normed_, act_x_, model_.d_final_norm(), 1, hidden_size);
        const auto& lm_head = model_.lm_head();
        ops::linear_int4(q, d_logits_, act_normed_,
                         static_cast<const uint8_t*>(lm_head.d_weights_int4),
                         static_cast<const sycl::half*>(lm_head.d_scales),
                         nullptr, 1, vocab_size, hidden_size);

        graph.end_recording(q);

        exec_graph_ = std::make_unique<syclex::command_graph<syclex::graph_state::executable>>(graph.finalize());
        is_captured_ = true;
        std::cout << "[DecodeGraph] Successfully captured and finalized 64-layer decode graph!" << std::endl;
        return true;

    } catch (const sycl::exception& e) {
        std::cerr << "[DecodeGraph] SYCL Exception during graph capture: " << e.what() << std::endl;
        is_captured_ = false;
        return false;
    }
}

int64_t DecodeGraph::decode_step(int64_t input_token_id, size_t cur_pos) {
    if (!is_captured_ || !exec_graph_) {
        throw std::runtime_error("DecodeGraph::decode_step called before successful graph capture");
    }

    if (cur_pos >= kv_cache_.max_seq_len()) {
        std::cerr << "[DecodeGraph] Error: cur_pos (" << cur_pos
                  << ") exceeds KV cache max_seq_len (" << kv_cache_.max_seq_len() << ")" << std::endl;
        return -1;
    }

    sycl::queue& q = ctx_->queue();
    int64_t pos_val = static_cast<int64_t>(cur_pos);

    // Update dynamic inputs in-place without altering device virtual addresses
    q.memcpy(d_token_ids_, &input_token_id, sizeof(int64_t));
    q.memcpy(d_positions_, &pos_val, sizeof(int64_t));

    // Replay the pre-compiled command list with zero host launch overhead
    q.ext_oneapi_graph(*exec_graph_);

    // Greedy sampling from logits
    constexpr int64_t vocab_size = 248320;
    int64_t next_token = ops::argmax(q, d_logits_, vocab_size);
    return next_token;
}

} // namespace xinfer::targets::qwen3_8
