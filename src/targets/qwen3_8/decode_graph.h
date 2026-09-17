#pragma once

#include "core/device.h"
#include "core/kv_cache.h"
#include "targets/qwen3_8_27b/weights.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <memory>
#include <cstdint>

namespace xinfer::targets::qwen3_8 {

// Milestone 8: Level Zero / SYCL Command Graph Capture & Replay for Fixed-Shape Single-Token Decode
// Eliminates per-step host kernel submission overhead across all 64 layers.
class DecodeGraph {
public:
    DecodeGraph(std::shared_ptr<core::DeviceContext> ctx,
                const qwen3_8_27b::LoadedModel& model,
                core::KVCache& kv_cache);
    ~DecodeGraph();

    DecodeGraph(const DecodeGraph&) = delete;
    DecodeGraph& operator=(const DecodeGraph&) = delete;

    // Captures the single-token forward pass across all 64 layers into an executable command graph
    bool capture();

    // Replays the captured command graph for input_token_id at sequence position cur_pos
    // and returns the sampled greedy token ID
    int64_t decode_step(int64_t input_token_id, size_t cur_pos);

    bool is_captured() const noexcept { return is_captured_; }

private:
    std::shared_ptr<core::DeviceContext> ctx_;
    const qwen3_8_27b::LoadedModel& model_;
    core::KVCache& kv_cache_;

    bool is_captured_{false};

    // Pointers with guaranteed address stability across the engine lifetime
    int64_t* d_token_ids_{nullptr};
    int64_t* d_positions_{nullptr};
    float*   d_logits_{nullptr};

    // Fixed activation buffers
    float* act_x_{nullptr};
    float* act_normed_{nullptr};
    float* act_proj_out_{nullptr};
    float* act_mlp_gate_{nullptr};

    float* act_q_gate_{nullptr};
    float* act_q_{nullptr};
    float* act_k_{nullptr};
    float* act_v_{nullptr};
    float* act_attn_out_{nullptr};

    float* act_qkv_raw_{nullptr};
    float* act_qkv_conv_{nullptr};
    float* act_z_{nullptr};
    float* act_b_{nullptr};
    float* act_a_{nullptr};
    float* act_delta_out_{nullptr};

    // Compiled executable graph handle
    std::unique_ptr<sycl::ext::oneapi::experimental::command_graph<
        sycl::ext::oneapi::experimental::graph_state::executable>> exec_graph_;
};

} // namespace xinfer::targets::qwen3_8
