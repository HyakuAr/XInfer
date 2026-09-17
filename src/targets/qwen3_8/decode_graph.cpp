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
#include "ops/sampling.h"

#include <iostream>

namespace syclex = sycl::ext::oneapi::experimental;

namespace xinfer::targets::qwen3_8 {

DecodeGraph::DecodeGraph(std::shared_ptr<core::DeviceContext> ctx,
                         const qwen3_8_27b::LoadedModel& model,
                         core::KVCache& kv_cache)
    : ctx_(std::move(ctx)), model_(model), kv_cache_(kv_cache) {
    sycl::queue& q = ctx_->queue();

    const auto& cfg = model_.config();
    const int64_t vocab_size = cfg.vocab_size;
    const int64_t hidden_size = cfg.hidden_size;
    const int64_t intermediate_size = cfg.intermediate_size;

    // Allocate fixed USM device memory pointers with guaranteed address stability
    d_token_ids_ = sycl::malloc_device<int64_t>(1, q);
    d_positions_ = sycl::malloc_device<int64_t>(1, q);
    d_logits_    = sycl::malloc_device<float>(vocab_size, q);

    size_t sample_scratch_size = ops::argmax_scratch_size(vocab_size);
    d_sample_max_ = sycl::malloc_shared<float>(sample_scratch_size, q);
    d_sample_idx_ = sycl::malloc_shared<int64_t>(sample_scratch_size, q);

    bufs_.act_x        = sycl::malloc_device<sycl::half>(hidden_size, q);
    bufs_.act_normed   = sycl::malloc_device<sycl::half>(hidden_size, q);
    bufs_.act_proj_out = sycl::malloc_device<sycl::half>(hidden_size, q);
    bufs_.act_mlp_gate = sycl::malloc_device<sycl::half>(intermediate_size, q);

    bufs_.act_q_gate   = sycl::malloc_device<sycl::half>(cfg.full_q_gate_dim(), q);
    bufs_.act_q        = sycl::malloc_device<sycl::half>(cfg.full_q_dim(), q);
    bufs_.act_k        = sycl::malloc_device<sycl::half>(cfg.full_k_dim(), q);
    bufs_.act_v        = sycl::malloc_device<sycl::half>(cfg.full_v_dim(), q);
    bufs_.act_attn_out = sycl::malloc_device<sycl::half>(cfg.full_out_dim(), q);

    bufs_.act_qkv_raw   = sycl::malloc_device<sycl::half>(cfg.linear_conv_channels, q);
    bufs_.act_qkv_conv  = sycl::malloc_device<sycl::half>(cfg.linear_conv_channels, q);
    bufs_.act_z         = sycl::malloc_device<sycl::half>(cfg.linear_z_dim, q);
    bufs_.act_b         = sycl::malloc_device<sycl::half>(cfg.linear_b_dim, q);
    bufs_.act_a         = sycl::malloc_device<sycl::half>(cfg.linear_a_dim, q);
    bufs_.act_delta_out = sycl::malloc_device<sycl::half>(cfg.linear_z_dim, q);
}

DecodeGraph::~DecodeGraph() {
    sycl::queue& q = ctx_->queue();
    exec_graph_.reset();

    if (d_token_ids_) sycl::free(d_token_ids_, q);
    if (d_positions_) sycl::free(d_positions_, q);
    if (d_logits_)    sycl::free(d_logits_, q);

    if (d_sample_max_) sycl::free(d_sample_max_, q);
    if (d_sample_idx_) sycl::free(d_sample_idx_, q);

    if (bufs_.act_x)        sycl::free(bufs_.act_x, q);
    if (bufs_.act_normed)   sycl::free(bufs_.act_normed, q);
    if (bufs_.act_proj_out) sycl::free(bufs_.act_proj_out, q);
    if (bufs_.act_mlp_gate) sycl::free(bufs_.act_mlp_gate, q);

    if (bufs_.act_q_gate)   sycl::free(bufs_.act_q_gate, q);
    if (bufs_.act_q)        sycl::free(bufs_.act_q, q);
    if (bufs_.act_k)        sycl::free(bufs_.act_k, q);
    if (bufs_.act_v)        sycl::free(bufs_.act_v, q);
    if (bufs_.act_attn_out) sycl::free(bufs_.act_attn_out, q);

    if (bufs_.act_qkv_raw)   sycl::free(bufs_.act_qkv_raw, q);
    if (bufs_.act_qkv_conv)  sycl::free(bufs_.act_qkv_conv, q);
    if (bufs_.act_z)         sycl::free(bufs_.act_z, q);
    if (bufs_.act_b)         sycl::free(bufs_.act_b, q);
    if (bufs_.act_a)         sycl::free(bufs_.act_a, q);
    if (bufs_.act_delta_out) sycl::free(bufs_.act_delta_out, q);
}

bool DecodeGraph::capture() {
    sycl::queue& q = ctx_->queue();
    auto dev = q.get_device();

    if (!dev.has(sycl::aspect::ext_oneapi_graph)) {
        std::cerr << "[DecodeGraph] Device does not support ext_oneapi_graph aspect." << std::endl;
        return false;
    }

    const auto& cfg = model_.config();
    const int64_t hidden_size = cfg.hidden_size;

    int64_t initial_token = 0;
    int64_t initial_pos = 0;
    q.memcpy(d_token_ids_, &initial_token, sizeof(int64_t)).wait();
    q.memcpy(d_positions_, &initial_pos, sizeof(int64_t)).wait();

    try {
        syclex::command_graph<syclex::graph_state::modifiable> graph(q.get_context(), dev);

        graph.begin_recording(q);

        // 1. Embedding lookup
        embed_tokens_lookup(q, bufs_.act_x, model_.d_embed_tokens(), d_token_ids_, 1, hidden_size);

        // 2. Loop over layers using shared parameterized forward_layer
        const auto& layers = model_.layers();
        size_t full_idx = 0;
        size_t linear_idx = 0;

        for (size_t l = 0; l < layers.size(); ++l) {
            forward_layer(q, cfg, layers[l], kv_cache_, full_idx, linear_idx, bufs_,
                          1, d_positions_, 0, d_positions_, false);
        }

        // 3. LM head projection
        forward_lm_head(q, model_, bufs_.act_normed, bufs_.act_x, d_logits_);

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

    // Greedy sampling from logits using caller-owned preallocated shared scratch buffer
    int64_t next_token = ops::argmax(q, d_logits_, model_.config().vocab_size, d_sample_max_, d_sample_idx_);
    return next_token;
}

} // namespace xinfer::targets::qwen3_8
