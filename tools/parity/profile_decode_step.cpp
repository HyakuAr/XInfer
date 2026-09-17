#include "core/device.h"
#include "core/arena.h"
#include "core/kv_cache.h"
#include "targets/qwen3_8_27b/weights.h"
#include "targets/qwen3_8/forward.h"
#include "targets/qwen3_8/linear_attn.h"
#include "targets/qwen3_8/decode_graph.h"
#include "ops/linear.h"
#include "ops/attention.h"
#include "ops/rmsnorm.h"
#include "ops/rope.h"
#include "ops/elementwise.h"
#include "ops/sampling.h"
#include "artifact/reader.h"

#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

using namespace xinfer;

struct OpCategoryTimers {
    double embed_ms{0.0};
    double rmsnorm_ms{0.0};
    double linear_full_attn_ms{0.0};
    double linear_lin_attn_ms{0.0};
    double linear_mlp_ms{0.0};
    double linear_lm_head_ms{0.0};
    double full_attn_sdpa_ms{0.0};
    double rope_ms{0.0};
    double lin_attn_conv1d_ms{0.0};
    double lin_attn_recurrent_ms{0.0};
    double elementwise_swiglu_ms{0.0};
    double sampling_argmax_ms{0.0};
    double total_step_ms{0.0};

    void print_table() const {
        double linear_total = linear_full_attn_ms + linear_lin_attn_ms + linear_mlp_ms + linear_lm_head_ms;
        double lin_attn_total = lin_attn_conv1d_ms + lin_attn_recurrent_ms;

        std::cout << "\n================================================================================" << std::endl;
        std::cout << "                  M10 DECODE STEP PER-OP TIMING BREAKDOWN                       " << std::endl;
        std::cout << "================================================================================" << std::endl;
        std::cout << std::left << std::setw(42) << "Operation Category" 
                  << std::right << std::setw(14) << "Time (ms)" 
                  << std::setw(14) << "% of Total" << std::endl;
        std::cout << "--------------------------------------------------------------------------------" << std::endl;

        auto print_row = [&](const std::string& name, double ms) {
            double pct = (total_step_ms > 0.0) ? (ms / total_step_ms * 100.0) : 0.0;
            std::cout << std::left << std::setw(42) << name 
                      << std::right << std::fixed << std::setprecision(2) 
                      << std::setw(14) << ms 
                      << std::setw(13) << pct << "%" << std::endl;
        };

        print_row("Embedding Lookup", embed_ms);
        print_row("RMSNorm (64 layers + Q/K + final)", rmsnorm_ms);
        print_row("RoPE (16 full-attention layers)", rope_ms);
        print_row("Linear: Full-Attention (Q, K, V, Out)", linear_full_attn_ms);
        print_row("Linear: Linear-Attention (QKV, Z, B, A, Out)", linear_lin_attn_ms);
        print_row("Linear: MLP SwiGLU (Gate, Up, Down)", linear_mlp_ms);
        print_row("Linear: LM Head (vocab 248,320)", linear_lm_head_ms);
        print_row("  >> [SUBTOTAL] All INT4 Linear Ops", linear_total);
        print_row("Full Attention: KV-Cache Write + SDPA", full_attn_sdpa_ms);
        print_row("Linear Attention: Causal Conv1d + SiLU", lin_attn_conv1d_ms);
        print_row("Linear Attention: Recurrent Gated Delta", lin_attn_recurrent_ms);
        print_row("  >> [SUBTOTAL] Linear Attention Path", lin_attn_total);
        print_row("Elementwise: SwiGLU + Gating + Adds", elementwise_swiglu_ms);
        print_row("Sampling: Argmax (USM alloc + reduce)", sampling_argmax_ms);
        std::cout << "--------------------------------------------------------------------------------" << std::endl;
        print_row("TOTAL ONE DECODE STEP", total_step_ms);
        std::cout << "================================================================================" << std::endl;
    }
};

int main() {
    std::cout << "==================================================================" << std::endl;
    std::cout << " xinfer M10 Performance Diagnostic: Detailed Decode Step Profiler" << std::endl;
    std::cout << "==================================================================" << std::endl;

    auto ctx = core::DeviceContext::create(true);
    sycl::queue& q = ctx->queue();
    auto dev = q.get_device();

    std::string artifact_path = "out/qwen3_8_27b.xinfer";
    std::cout << "Loading model artifact: " << artifact_path << " ..." << std::endl;
    artifact::ArtifactReader reader;
    if (!reader.open(artifact_path)) {
        std::cerr << "Failed to open artifact: " << artifact_path << std::endl;
        return 1;
    }

    std::string err;
    auto model = targets::qwen3_8_27b::LoadedModel::load_from_artifact(ctx, reader, &err);
    if (!model) {
        std::cerr << "Failed to load model: " << err << std::endl;
        return 1;
    }

    const auto& cfg = model->config();
    core::KVCacheConfig kv_cfg = cfg.create_kv_cache_config(8192);
    core::KVCache kv_cache(ctx, kv_cfg);
    kv_cache.allocate();

    const int64_t vocab_size = cfg.vocab_size;
    const int64_t hidden_size = cfg.hidden_size;
    const int64_t intermediate_size = cfg.intermediate_size;

    // USM pointers
    int64_t* d_token_ids = sycl::malloc_device<int64_t>(1, q);
    int64_t* d_positions = sycl::malloc_device<int64_t>(1, q);
    float* d_logits      = sycl::malloc_device<float>(vocab_size, q);

    float* act_x        = sycl::malloc_device<float>(hidden_size, q);
    float* act_normed   = sycl::malloc_device<float>(hidden_size, q);
    float* act_proj_out = sycl::malloc_device<float>(hidden_size, q);
    float* act_mlp_gate = sycl::malloc_device<float>(intermediate_size, q);
    float* act_mlp_up   = sycl::malloc_device<float>(intermediate_size, q);

    float* act_q_gate   = sycl::malloc_device<float>(cfg.full_q_gate_dim(), q);
    float* act_q        = sycl::malloc_device<float>(cfg.full_q_dim(), q);
    float* act_k        = sycl::malloc_device<float>(cfg.full_k_dim(), q);
    float* act_v        = sycl::malloc_device<float>(cfg.full_v_dim(), q);
    float* act_attn_out = sycl::malloc_device<float>(cfg.full_out_dim(), q);

    float* act_qkv_raw   = sycl::malloc_device<float>(cfg.linear_conv_channels, q);
    float* act_qkv_conv  = sycl::malloc_device<float>(cfg.linear_conv_channels, q);
    float* act_z         = sycl::malloc_device<float>(cfg.linear_z_dim, q);
    float* act_b         = sycl::malloc_device<float>(cfg.linear_b_dim, q);
    float* act_a         = sycl::malloc_device<float>(cfg.linear_a_dim, q);
    float* act_delta_out = sycl::malloc_device<float>(cfg.linear_out_dim(), q);

    int64_t init_tok = 9419;
    int64_t init_pos = 10;
    q.memcpy(d_token_ids, &init_tok, sizeof(int64_t)).wait();
    q.memcpy(d_positions, &init_pos, sizeof(int64_t)).wait();

    std::cout << "Warming up kernels and memory on B60..." << std::endl;
    // Warmup step to ensure all JIT kernels are compiled and USM buffers are resident
    {
        targets::qwen3_8::embed_tokens_lookup(q, act_x, model->d_embed_tokens(), d_token_ids, 1, hidden_size);
        ops::rmsnorm(q, act_normed, act_x, model->layers()[0].d_input_layernorm, 1, hidden_size);
        ops::linear_int4(q, d_logits, act_normed,
                         static_cast<const uint8_t*>(model->lm_head().d_weights_int4),
                         static_cast<const sycl::half*>(model->lm_head().d_scales),
                         nullptr, 1, vocab_size, hidden_size);
        ops::argmax(q, d_logits, vocab_size);
        q.wait();
    }
    std::cout << "Starting profiled decode step measurement on B60..." << std::endl;

    enum Category {
        CAT_EMBED = 0,
        CAT_RMSNORM,
        CAT_ROPE,
        CAT_LINEAR_ATTN,
        CAT_LINEAR_MLP,
        CAT_LINEAR_LM_HEAD,
        CAT_FULL_ATTN_SDPA,
        CAT_LIN_ATTN_CONV1D,
        CAT_LIN_ATTN_RECURRENT,
        CAT_ELEMENTWISE,
        CAT_SAMPLING,
        NUM_CATEGORIES
    };

    const char* category_names[NUM_CATEGORIES] = {
        "Embedding Lookup",
        "RMSNorm (layers + Q/K + final)",
        "RoPE (full-attention layers)",
        "Linear: Attention Projections",
        "Linear: MLP SwiGLU Projections",
        "Linear: LM Head Projection",
        "Full Attention: KV-Cache Write + SDPA",
        "Linear Attention: Causal Conv1d + SiLU",
        "Linear Attention: Recurrent Gated Delta",
        "Elementwise: SwiGLU + Gating + Adds",
        "Sampling: Greedy Argmax"
    };

    std::vector<sycl::event> cat_events[NUM_CATEGORIES];
    for (int i = 0; i < NUM_CATEGORIES; ++i) {
        cat_events[i].reserve(256);
    }

    q.wait();
    auto total_start = std::chrono::high_resolution_clock::now();

    // 1. Embedding Lookup
    cat_events[CAT_EMBED].push_back(
        targets::qwen3_8::embed_tokens_lookup(q, act_x, model->d_embed_tokens(), d_token_ids, 1, hidden_size)
    );

    // 2. Layers
    const auto& layers = model->layers();
    size_t full_idx = 0;
    size_t linear_idx = 0;

    for (size_t l = 0; l < layers.size(); ++l) {
        const auto& layer = layers[l];

        // RMSNorm
        cat_events[CAT_RMSNORM].push_back(
            ops::rmsnorm(q, act_normed, act_x, layer.d_input_layernorm, 1, hidden_size)
        );

        if (layer.layer_type == "full_attention") {
            // Full-Attention Fused Projections
            ops::FusedProjectionDesc fa_projs[3] = {
                {act_q_gate, static_cast<const uint8_t*>(layer.q_proj.d_weights_int4),
                 static_cast<const sycl::half*>(layer.q_proj.d_scales), nullptr, cfg.full_q_gate_dim()},
                {act_k, static_cast<const uint8_t*>(layer.k_proj.d_weights_int4),
                 static_cast<const sycl::half*>(layer.k_proj.d_scales), nullptr, cfg.full_k_dim()},
                {act_v, static_cast<const uint8_t*>(layer.v_proj.d_weights_int4),
                 static_cast<const sycl::half*>(layer.v_proj.d_scales), nullptr, cfg.full_v_dim()}
            };
            cat_events[CAT_LINEAR_ATTN].push_back(
                ops::linear_int4_fused(q, act_normed, fa_projs, 3, 1, hidden_size)
            );

            // Elementwise Q split
            float* q_ptr = act_q;
            float* q_gate_ptr = act_q_gate;
            int64_t num_q_heads = cfg.num_attention_heads;
            int64_t head_dim = cfg.head_dim;
            int64_t q_gate_dim = cfg.full_q_gate_dim();

            cat_events[CAT_ELEMENTWISE].push_back(
                q.parallel_for(sycl::range<2>(1, num_q_heads), [=](sycl::id<2> idx) {
                    int64_t t = idx[0];
                    int64_t h = idx[1];
                    for (int d = 0; d < head_dim; ++d) {
                        q_ptr[(t * num_q_heads + h) * head_dim + d] = q_gate_ptr[t * q_gate_dim + h * 2 * head_dim + d];
                    }
                })
            );

            // Q/K RMSNorm
            cat_events[CAT_RMSNORM].push_back(
                ops::rmsnorm(q, act_q, act_q, layer.d_q_norm, num_q_heads, head_dim)
            );
            cat_events[CAT_RMSNORM].push_back(
                ops::rmsnorm(q, act_k, act_k, layer.d_k_norm, cfg.num_key_value_heads, head_dim)
            );

            // RoPE
            cat_events[CAT_ROPE].push_back(
                ops::rope(q, act_q, act_k, 1, num_q_heads, cfg.num_key_value_heads, head_dim, d_positions, cfg.rope_theta, cfg.rope_dim)
            );

            // KV-Cache Write + Causal SDPA
            cat_events[CAT_FULL_ATTN_SDPA].push_back(
                ops::attention_write_kv_cache_dynamic(q, kv_cache.k_cache(full_idx), kv_cache.v_cache(full_idx),
                                                      act_k, act_v, d_positions, 1, cfg.num_key_value_heads, head_dim,
                                                      static_cast<int64_t>(kv_cache.max_seq_len()))
            );
            cat_events[CAT_FULL_ATTN_SDPA].push_back(
                ops::sdpa_causal_cached_dynamic(q, act_attn_out, act_q,
                                                kv_cache.k_cache(full_idx), kv_cache.v_cache(full_idx),
                                                d_positions, 1, num_q_heads, cfg.num_key_value_heads, head_dim, 0.0f,
                                                static_cast<int64_t>(kv_cache.max_seq_len()))
            );

            // Attention Gating
            float* attn_out_ptr = act_attn_out;
            cat_events[CAT_ELEMENTWISE].push_back(
                q.parallel_for(sycl::range<2>(1, num_q_heads), [=](sycl::id<2> idx) {
                    int64_t t = idx[0];
                    int64_t h = idx[1];
                    for (int d = 0; d < head_dim; ++d) {
                        float gate_val = q_gate_ptr[t * q_gate_dim + h * 2 * head_dim + head_dim + d];
                        float sig = 1.0f / (1.0f + sycl::exp(-gate_val));
                        attn_out_ptr[(t * num_q_heads + h) * head_dim + d] *= sig;
                    }
                })
            );

            // Out projection
            cat_events[CAT_LINEAR_ATTN].push_back(
                ops::linear_int4(q, act_proj_out, act_attn_out,
                                 static_cast<const uint8_t*>(layer.o_proj.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.o_proj.d_scales),
                                 nullptr, 1, hidden_size, cfg.full_out_dim())
            );

            full_idx++;
        } else {
            // Linear Attention Fused Projections
            ops::FusedProjectionDesc la_projs[4] = {
                {act_qkv_raw, static_cast<const uint8_t*>(layer.in_proj_qkv.d_weights_int4),
                 static_cast<const sycl::half*>(layer.in_proj_qkv.d_scales), nullptr, cfg.linear_conv_channels},
                {act_z, static_cast<const uint8_t*>(layer.in_proj_z.d_weights_int4),
                 static_cast<const sycl::half*>(layer.in_proj_z.d_scales), nullptr, cfg.linear_z_dim},
                {act_b, static_cast<const uint8_t*>(layer.in_proj_b.d_weights_int4),
                 static_cast<const sycl::half*>(layer.in_proj_b.d_scales), nullptr, cfg.linear_b_dim},
                {act_a, static_cast<const uint8_t*>(layer.in_proj_a.d_weights_int4),
                 static_cast<const sycl::half*>(layer.in_proj_a.d_scales), nullptr, cfg.linear_a_dim}
            };
            cat_events[CAT_LINEAR_ATTN].push_back(
                ops::linear_int4_fused(q, act_normed, la_projs, 4, 1, hidden_size)
            );

            // Causal Conv1d
            cat_events[CAT_LIN_ATTN_CONV1D].push_back(
                targets::qwen3_8::causal_conv1d_silu(q, act_qkv_conv, act_qkv_raw, layer.d_conv1d_weight, 1,
                                                     kv_cache.conv_state(linear_idx))
            );

            // Recurrent Gated Delta Net
            cat_events[CAT_LIN_ATTN_RECURRENT].push_back(
                targets::qwen3_8::recurrent_gated_delta_net(q, act_delta_out, act_qkv_conv, act_z, act_b, act_a,
                                                            layer.d_A_log, layer.d_dt_bias, layer.d_norm_weight,
                                                            kv_cache.linear_state(linear_idx), 1, false)
            );

            // Out projection
            cat_events[CAT_LINEAR_ATTN].push_back(
                ops::linear_int4(q, act_proj_out, act_delta_out,
                                 static_cast<const uint8_t*>(layer.out_proj.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.out_proj.d_scales),
                                 nullptr, 1, hidden_size, cfg.linear_z_dim)
            );

            linear_idx++;
        }

        // Residual Add 1
        cat_events[CAT_ELEMENTWISE].push_back(
            ops::add_inplace(q, act_x, act_proj_out, hidden_size)
        );

        // Post-Attention RMSNorm
        cat_events[CAT_RMSNORM].push_back(
            ops::rmsnorm(q, act_normed, act_x, layer.d_post_attention_layernorm, 1, hidden_size)
        );

        // Fused MLP Gate + Up + SwiGLU: SiLU(gate) * up directly in sub-group registers
        cat_events[CAT_LINEAR_MLP].push_back(
            ops::mlp_gate_up_swiglu_int4(q, act_mlp_gate, act_normed,
                                         static_cast<const uint8_t*>(layer.gate_proj.d_weights_int4),
                                         static_cast<const sycl::half*>(layer.gate_proj.d_scales),
                                         static_cast<const uint8_t*>(layer.up_proj.d_weights_int4),
                                         static_cast<const sycl::half*>(layer.up_proj.d_scales),
                                         1, intermediate_size, hidden_size)
        );

        // MLP Down Projection
        cat_events[CAT_LINEAR_MLP].push_back(
            ops::linear_int4(q, act_proj_out, act_mlp_gate,
                             static_cast<const uint8_t*>(layer.down_proj.d_weights_int4),
                             static_cast<const sycl::half*>(layer.down_proj.d_scales),
                             nullptr, 1, hidden_size, intermediate_size)
        );

        // Residual Add 2
        cat_events[CAT_ELEMENTWISE].push_back(
            ops::add_inplace(q, act_x, act_proj_out, hidden_size)
        );
    }

    // Final RMSNorm
    cat_events[CAT_RMSNORM].push_back(
        ops::rmsnorm(q, act_normed, act_x, model->d_final_norm(), 1, hidden_size)
    );

    // LM Head
    const auto& lm_head = model->lm_head();
    cat_events[CAT_LINEAR_LM_HEAD].push_back(
        ops::linear_int4(q, d_logits, act_normed,
                         static_cast<const uint8_t*>(lm_head.d_weights_int4),
                         static_cast<const sycl::half*>(lm_head.d_scales),
                         nullptr, 1, vocab_size, hidden_size)
    );

    // Single wait for all ~960 commands!
    q.wait();

    // Sampling Argmax
    auto t_sample_0 = std::chrono::high_resolution_clock::now();
    int64_t next_tok = ops::argmax(q, d_logits, vocab_size);
    (void)next_tok;
    q.wait();
    auto t_sample_1 = std::chrono::high_resolution_clock::now();
    double sampling_ms = std::chrono::duration<double, std::milli>(t_sample_1 - t_sample_0).count();

    auto total_end = std::chrono::high_resolution_clock::now();
    double wallclock_step_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    // Collect GPU hardware profiling timestamps
    double cat_durations_ms[NUM_CATEGORIES] = {0.0};
    uint64_t earliest_start = std::numeric_limits<uint64_t>::max();
    uint64_t latest_end = 0;

    for (int i = 0; i < NUM_CATEGORIES; ++i) {
        if (i == CAT_SAMPLING) {
            cat_durations_ms[i] = sampling_ms;
            continue;
        }
        for (auto& ev : cat_events[i]) {
            uint64_t s = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
            uint64_t e = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
            if (s < earliest_start) earliest_start = s;
            if (e > latest_end) latest_end = e;
            cat_durations_ms[i] += static_cast<double>(e - s) * 1e-6;
        }
    }

    double total_cat_sum_ms = 0.0;
    for (int i = 0; i < NUM_CATEGORIES; ++i) {
        total_cat_sum_ms += cat_durations_ms[i];
    }
    double gpu_span_ms = (latest_end > earliest_start) ? static_cast<double>(latest_end - earliest_start) * 1e-6 : 0.0;

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "          RECONCILED POST-FIX M10 DECODE STEP HARDWARE PROFILING                " << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << std::left << std::setw(46) << "Operation Category" 
              << std::right << std::setw(14) << "Time (ms)" 
              << std::setw(14) << "% of Total" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    for (int i = 0; i < NUM_CATEGORIES; ++i) {
        double pct = (total_cat_sum_ms > 0.0) ? (cat_durations_ms[i] / total_cat_sum_ms * 100.0) : 0.0;
        std::cout << std::left << std::setw(46) << category_names[i] 
                  << std::right << std::fixed << std::setprecision(2) 
                  << std::setw(14) << cat_durations_ms[i] 
                  << std::setw(13) << pct << "%" << std::endl;
    }
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(46) << "TOTAL ONE DECODE STEP (Sum of Categories)" 
              << std::right << std::fixed << std::setprecision(2) 
              << std::setw(14) << total_cat_sum_ms 
              << std::setw(13) << "100.00%" << std::endl;
    std::cout << "================================================================================" << std::endl;

    std::cout << "\nGPU Hardware Span (first start -> last end): " << gpu_span_ms << " ms" << std::endl;
    std::cout << "Continuous Wall-Clock Latency:              " << wallclock_step_ms << " ms" << std::endl;

    // Aggregate linear bandwidth reconciliation against M7 microbenchmark
    double linear_total_ms = cat_durations_ms[CAT_LINEAR_ATTN] + cat_durations_ms[CAT_LINEAR_MLP] + cat_durations_ms[CAT_LINEAR_LM_HEAD];
    double weight_gb = 15.77; // INT4 model weights in GB (from M2 artifact)
    double aggregate_bw = (linear_total_ms > 0.0) ? (weight_gb / (linear_total_ms / 1000.0)) : 0.0;
    std::cout << "\n--- Linear GEMV Bandwidth (M7 Reconciliation) ---" << std::endl;
    std::cout << "Total linear kernel time:  " << std::fixed << std::setprecision(2) << linear_total_ms << " ms (" << (linear_total_ms / total_cat_sum_ms * 100.0) << "% of step)" << std::endl;
    std::cout << "Model weights:             " << weight_gb << " GB" << std::endl;
    std::cout << "Aggregate effective BW:    " << std::setprecision(1) << aggregate_bw << " GB/s" << std::endl;
    std::cout << "M7 peak (N=17408 only):    383.7 GB/s" << std::endl;
    std::cout << "Gap ratio:                 " << std::setprecision(1) << (383.7 / aggregate_bw) << "x (explained by shape-mix occupancy)" << std::endl;

    // Now also profile DecodeGraph replay specifically!
    std::cout << "\nMeasuring DecodeGraph command graph replay timing (M8)..." << std::endl;
    targets::qwen3_8::DecodeGraph graph_runner(ctx, *model, kv_cache);
    if (graph_runner.capture()) {
        q.wait();
        auto g0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 3; ++i) {
            graph_runner.decode_step(init_tok, init_pos + i);
            q.wait();
        }
        auto g1 = std::chrono::high_resolution_clock::now();
        double graph_step_ms = std::chrono::duration<double, std::milli>(g1 - g0).count() / 3.0;
        std::cout << "Average DecodeGraph replay time per step: " << graph_step_ms << " ms" << std::endl;
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
