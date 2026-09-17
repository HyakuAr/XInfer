#include "core/device.h"
#include "targets/qwen3_8_27b/weights.h"
#include "artifact/reader.h"
#include "ops/linear.h"

#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <numeric>

using namespace xinfer;

// M7 microbenchmark peak: 383.7 GB/s at shape M=1, N=17408, K=5120
// (measured in tools/parity/test_int4_xmx_vs_gemv.cpp on Arc Pro B60)
constexpr double M7_PEAK_BW_GBS = 383.7;
constexpr double B60_PEAK_BW_GBS = 456.0;
constexpr int B60_HW_THREADS = 1280; // 20 Xe-cores * 64 threads/core

struct ProjectionMetrics {
    std::string name;
    int64_t N{0};
    int64_t K{0};
    int count{0};
    std::vector<sycl::event> events;

    double total_time_ms{0.0};
    size_t weight_bytes{0};
    size_t scale_bytes{0};
    size_t act_bytes{0};
    size_t total_bytes{0};
    double achieved_bw_gbs{0.0};
    int64_t subgroups_per_launch{0};
    double occupancy_pct{0.0};
};

int main(int argc, char** argv) {
    std::cout << "==================================================================" << std::endl;
    std::cout << "  xInfer INT4 Linear GEMV Per-Projection Bandwidth Profiler        " << std::endl;
    std::cout << "==================================================================" << std::endl;

    std::string artifact_path = (argc > 1) ? argv[1] : "out/qwen3_8_27b.xinfer";
    auto ctx = core::DeviceContext::create(true);
    sycl::queue& q = ctx->queue();

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
    const int64_t vocab_size = cfg.vocab_size;
    const int64_t hidden_size = cfg.hidden_size;
    const int64_t intermediate_size = cfg.intermediate_size;

    // Allocate activation buffers
    float* act_x        = sycl::malloc_device<float>(hidden_size, q);
    float* act_normed   = sycl::malloc_device<float>(hidden_size, q);
    float* act_proj_out = sycl::malloc_device<float>(hidden_size, q);
    float* act_mlp_gate = sycl::malloc_device<float>(intermediate_size, q);
    float* act_mlp_up   = sycl::malloc_device<float>(intermediate_size, q);

    float* act_q_gate   = sycl::malloc_device<float>(cfg.full_q_gate_dim(), q);
    float* act_k        = sycl::malloc_device<float>(cfg.full_k_dim(), q);
    float* act_v        = sycl::malloc_device<float>(cfg.full_v_dim(), q);
    float* act_attn_out = sycl::malloc_device<float>(cfg.full_out_dim(), q);

    float* act_qkv_raw   = sycl::malloc_device<float>(cfg.linear_conv_channels, q);
    float* act_z         = sycl::malloc_device<float>(cfg.linear_z_dim, q);
    float* act_b         = sycl::malloc_device<float>(cfg.linear_b_dim, q);
    float* act_a         = sycl::malloc_device<float>(cfg.linear_a_dim, q);
    float* act_delta_out = sycl::malloc_device<float>(cfg.linear_z_dim, q);
    float* d_logits      = sycl::malloc_device<float>(vocab_size, q);

    q.memset(act_x, 0, hidden_size * sizeof(float)).wait();
    q.memset(act_normed, 0, hidden_size * sizeof(float)).wait();

    enum ProjType {
        P_MLP_GATE = 0,
        P_MLP_UP,
        P_MLP_DOWN,
        P_FA_Q,
        P_FA_K,
        P_FA_V,
        P_FA_OUT,
        P_LA_QKV,
        P_LA_Z,
        P_LA_B,
        P_LA_A,
        P_LA_OUT,
        P_LM_HEAD,
        NUM_PROJ_TYPES
    };

    std::vector<ProjectionMetrics> projs(NUM_PROJ_TYPES);
    projs[P_MLP_GATE]   = {"MLP Gate",           intermediate_size, hidden_size, static_cast<int>(cfg.num_hidden_layers), {}};
    projs[P_MLP_UP]     = {"MLP Up",             intermediate_size, hidden_size, static_cast<int>(cfg.num_hidden_layers), {}};
    projs[P_MLP_DOWN]   = {"MLP Down",           hidden_size, intermediate_size, static_cast<int>(cfg.num_hidden_layers), {}};
    projs[P_FA_Q]       = {"Full-Attn Q",        cfg.full_q_gate_dim(), hidden_size, static_cast<int>(cfg.num_full_layers()), {}};
    projs[P_FA_K]       = {"Full-Attn K",        cfg.full_k_dim(),  hidden_size, static_cast<int>(cfg.num_full_layers()), {}};
    projs[P_FA_V]       = {"Full-Attn V",        cfg.full_v_dim(),  hidden_size, static_cast<int>(cfg.num_full_layers()), {}};
    projs[P_FA_OUT]     = {"Full-Attn Out",      hidden_size, cfg.full_out_dim(), static_cast<int>(cfg.num_full_layers()), {}};
    projs[P_LA_QKV]     = {"Linear-Attn QKV",    cfg.linear_conv_channels, hidden_size, static_cast<int>(cfg.num_linear_layers()), {}};
    projs[P_LA_Z]       = {"Linear-Attn Z",      cfg.linear_z_dim,  hidden_size, static_cast<int>(cfg.num_linear_layers()), {}};
    projs[P_LA_B]       = {"Linear-Attn B",      cfg.linear_b_dim,    hidden_size, static_cast<int>(cfg.num_linear_layers()), {}};
    projs[P_LA_A]       = {"Linear-Attn A",      cfg.linear_a_dim,    hidden_size, static_cast<int>(cfg.num_linear_layers()), {}};
    projs[P_LA_OUT]     = {"Linear-Attn Out",    hidden_size, cfg.linear_z_dim, static_cast<int>(cfg.num_linear_layers()), {}};
    projs[P_LM_HEAD]    = {"LM Head",            vocab_size, hidden_size, 1, {}};

    for (auto& p : projs) {
        p.events.reserve(p.count);
    }

    // Warmup
    {
        ops::linear_int4(q, act_mlp_gate, act_normed,
                         static_cast<const uint8_t*>(model->layers()[0].gate_proj.d_weights_int4),
                         static_cast<const sycl::half*>(model->layers()[0].gate_proj.d_scales),
                         nullptr, 1, intermediate_size, hidden_size);
        q.wait();
    }

    std::cout << "Profiling all " << ((cfg.num_hidden_layers * 3) + (cfg.num_full_layers() * 4) + (cfg.num_linear_layers() * 5) + 1)
              << " INT4 linear projections across all " << cfg.num_hidden_layers << " layers..." << std::endl;

    q.wait();
    const auto& layers = model->layers();

    for (size_t l = 0; l < layers.size(); ++l) {
        const auto& layer = layers[l];

        if (layer.layer_type == "full_attention") {
            // Q proj
            projs[P_FA_Q].events.push_back(
                ops::linear_int4(q, act_q_gate, act_normed,
                                 static_cast<const uint8_t*>(layer.q_proj.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.q_proj.d_scales),
                                 nullptr, 1, cfg.full_q_gate_dim(), hidden_size)
            );
            // K proj
            projs[P_FA_K].events.push_back(
                ops::linear_int4(q, act_k, act_normed,
                                 static_cast<const uint8_t*>(layer.k_proj.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.k_proj.d_scales),
                                 nullptr, 1, cfg.full_k_dim(), hidden_size)
            );
            // V proj
            projs[P_FA_V].events.push_back(
                ops::linear_int4(q, act_v, act_normed,
                                 static_cast<const uint8_t*>(layer.v_proj.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.v_proj.d_scales),
                                 nullptr, 1, cfg.full_v_dim(), hidden_size)
            );
            // Out proj
            projs[P_FA_OUT].events.push_back(
                ops::linear_int4(q, act_proj_out, act_attn_out,
                                 static_cast<const uint8_t*>(layer.o_proj.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.o_proj.d_scales),
                                 nullptr, 1, hidden_size, cfg.full_out_dim())
            );
        } else {
            // QKV proj
            projs[P_LA_QKV].events.push_back(
                ops::linear_int4(q, act_qkv_raw, act_normed,
                                 static_cast<const uint8_t*>(layer.in_proj_qkv.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.in_proj_qkv.d_scales),
                                 nullptr, 1, cfg.linear_conv_channels, hidden_size)
            );
            // Z proj
            projs[P_LA_Z].events.push_back(
                ops::linear_int4(q, act_z, act_normed,
                                 static_cast<const uint8_t*>(layer.in_proj_z.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.in_proj_z.d_scales),
                                 nullptr, 1, cfg.linear_z_dim, hidden_size)
            );
            // B proj
            projs[P_LA_B].events.push_back(
                ops::linear_int4(q, act_b, act_normed,
                                 static_cast<const uint8_t*>(layer.in_proj_b.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.in_proj_b.d_scales),
                                 nullptr, 1, cfg.linear_b_dim, hidden_size)
            );
            // A proj
            projs[P_LA_A].events.push_back(
                ops::linear_int4(q, act_a, act_normed,
                                 static_cast<const uint8_t*>(layer.in_proj_a.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.in_proj_a.d_scales),
                                 nullptr, 1, cfg.linear_a_dim, hidden_size)
            );
            // Out proj
            projs[P_LA_OUT].events.push_back(
                ops::linear_int4(q, act_proj_out, act_delta_out,
                                 static_cast<const uint8_t*>(layer.out_proj.d_weights_int4),
                                 static_cast<const sycl::half*>(layer.out_proj.d_scales),
                                 nullptr, 1, hidden_size, cfg.linear_z_dim)
            );
        }

        // MLP Gate
        projs[P_MLP_GATE].events.push_back(
            ops::linear_int4(q, act_mlp_gate, act_normed,
                             static_cast<const uint8_t*>(layer.gate_proj.d_weights_int4),
                             static_cast<const sycl::half*>(layer.gate_proj.d_scales),
                             nullptr, 1, intermediate_size, hidden_size)
        );
        // MLP Up
        projs[P_MLP_UP].events.push_back(
            ops::linear_int4(q, act_mlp_up, act_normed,
                             static_cast<const uint8_t*>(layer.up_proj.d_weights_int4),
                             static_cast<const sycl::half*>(layer.up_proj.d_scales),
                             nullptr, 1, intermediate_size, hidden_size)
        );
        // MLP Down
        projs[P_MLP_DOWN].events.push_back(
            ops::linear_int4(q, act_proj_out, act_mlp_gate,
                             static_cast<const uint8_t*>(layer.down_proj.d_weights_int4),
                             static_cast<const sycl::half*>(layer.down_proj.d_scales),
                             nullptr, 1, hidden_size, intermediate_size)
        );
    }

    // LM Head
    const auto& lm_head = model->lm_head();
    projs[P_LM_HEAD].events.push_back(
        ops::linear_int4(q, d_logits, act_normed,
                         static_cast<const uint8_t*>(lm_head.d_weights_int4),
                         static_cast<const sycl::half*>(lm_head.d_scales),
                         nullptr, 1, vocab_size, hidden_size)
    );

    q.wait();

    double grand_total_time_ms = 0.0;
    size_t grand_total_bytes = 0;
    size_t grand_total_weight_bytes = 0;

    for (auto& p : projs) {
        double dur_ms = 0.0;
        for (auto& ev : p.events) {
            uint64_t s = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
            uint64_t e = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
            dur_ms += static_cast<double>(e - s) * 1e-6;
        }
        p.total_time_ms = dur_ms;
        grand_total_time_ms += dur_ms;

        // Bytes calculations
        size_t w_bytes_per_op = static_cast<size_t>(p.N) * (p.K / 2);
        size_t s_bytes_per_op = static_cast<size_t>(p.N) * (p.K / cfg.group_size) * sizeof(sycl::half);
        size_t in_act_per_op  = static_cast<size_t>(p.K) * sizeof(float);
        size_t out_act_per_op = static_cast<size_t>(p.N) * sizeof(float);

        p.weight_bytes = w_bytes_per_op * p.count;
        p.scale_bytes  = s_bytes_per_op * p.count;
        p.act_bytes    = (in_act_per_op + out_act_per_op) * p.count;
        p.total_bytes  = p.weight_bytes + p.scale_bytes + p.act_bytes;

        grand_total_bytes += p.total_bytes;
        grand_total_weight_bytes += (p.weight_bytes + p.scale_bytes);

        if (p.total_time_ms > 0.0) {
            p.achieved_bw_gbs = (static_cast<double>(p.total_bytes) / 1e9) / (p.total_time_ms / 1000.0);
        }
        // Sub-group occupancy: ROWS_PER_SG=2, each sub-group covers 2 output rows
        p.subgroups_per_launch = (p.N + 1) / 2;
        p.occupancy_pct = static_cast<double>(p.subgroups_per_launch) / B60_HW_THREADS * 100.0;
    }

    int total_ops_count = 0;
    for (const auto& p : projs) {
        total_ops_count += p.count;
    }

    std::cout << "\n=========================================================================================================================" << std::endl;
    std::cout << "                                  DETAILED INT4 LINEAR GEMV SHAPE & BANDWIDTH BREAKDOWN                                  " << std::endl;
    std::cout << "                          (M7 Peak Reference: " << M7_PEAK_BW_GBS << " GB/s at N=17408, K=5120)" << std::endl;
    std::cout << "=========================================================================================================================" << std::endl;
    std::cout << std::left << std::setw(18) << "Projection Type"
              << std::right << std::setw(14) << "Shape [N x K]"
              << std::setw(7) << "Count"
              << std::setw(8) << "SGs"
              << std::setw(8) << "Occ%"
              << std::setw(12) << "Time (ms)"
              << std::setw(12) << "Avg/Op(ms)"
              << std::setw(10) << "% Linear"
              << std::setw(13) << "Bandwidth"
              << std::setw(10) << "vs M7"
              << std::setw(14) << "Status" << std::endl;
    std::cout << "-------------------------------------------------------------------------------------------------------------------------" << std::endl;

    for (const auto& p : projs) {
        double pct = (grand_total_time_ms > 0.0) ? (p.total_time_ms / grand_total_time_ms * 100.0) : 0.0;
        double avg_ms = p.total_time_ms / p.count;
        double vs_m7 = (M7_PEAK_BW_GBS > 0.0) ? (p.achieved_bw_gbs / M7_PEAK_BW_GBS * 100.0) : 0.0;

        std::string shape_str = std::to_string(p.N) + "x" + std::to_string(p.K);
        std::string status = (p.achieved_bw_gbs < 50.0) ? "SEVERELY LOW" : (p.achieved_bw_gbs < 100.0 ? "SUBOPTIMAL" : "SATURATED");
        std::string occ_str = (p.occupancy_pct < 100.0) ? std::to_string(static_cast<int>(p.occupancy_pct)) + "%" : "FULL";

        std::cout << std::left << std::setw(18) << p.name
                  << std::right << std::setw(14) << shape_str
                  << std::setw(7) << p.count
                  << std::setw(8) << p.subgroups_per_launch
                  << std::setw(8) << occ_str
                  << std::fixed << std::setprecision(2) << std::setw(12) << p.total_time_ms
                  << std::setprecision(3) << std::setw(12) << avg_ms
                  << std::setprecision(2) << std::setw(9) << pct << "%"
                  << std::setprecision(1) << std::setw(10) << p.achieved_bw_gbs << " GB/s"
                  << std::setprecision(0) << std::setw(8) << vs_m7 << "%"
                  << std::setw(14) << status << std::endl;
    }
    std::cout << "-------------------------------------------------------------------------------------------------------------------------" << std::endl;
    double aggregate_bw = (static_cast<double>(grand_total_bytes) / 1e9) / (grand_total_time_ms / 1000.0);
    double aggregate_weight_bw = (static_cast<double>(grand_total_weight_bytes) / 1e9) / (grand_total_time_ms / 1000.0);
    std::cout << std::left << std::setw(18) << "TOTAL LINEAR GEMV"
              << std::right << std::setw(14) << "All Projections"
              << std::setw(7) << total_ops_count
              << std::fixed << std::setprecision(1) << std::setw(13) << (grand_total_bytes / (1024.0 * 1024.0))
              << std::setprecision(2) << std::setw(12) << grand_total_time_ms
              << std::setw(12) << "-"
              << std::setw(9) << "100.00%"
              << std::setprecision(1) << std::setw(10) << aggregate_bw << " GB/s"
              << std::setw(14) << "-" << std::endl;
    std::cout << "=========================================================================================================================" << std::endl;
    std::cout << "Aggregate Model Weights Read:        " << (grand_total_weight_bytes / 1e9) << " GB" << std::endl;
    std::cout << "Aggregate Weight Memory Bandwidth:   " << aggregate_weight_bw << " GB/s" << std::endl;
    std::cout << "Aggregate Total Bus Traffic Bandwidth:" << aggregate_bw << " GB/s" << std::endl;
    std::cout << "\n--- M7 Microbenchmark Reconciliation ---" << std::endl;
    std::cout << "M7 Peak (N=17408):           " << M7_PEAK_BW_GBS << " GB/s (" << std::fixed << std::setprecision(1) << (M7_PEAK_BW_GBS / B60_PEAK_BW_GBS * 100.0) << "% of B60 peak)" << std::endl;
    std::cout << "Decode Aggregate:            " << std::setprecision(1) << aggregate_bw << " GB/s (" << (aggregate_bw / B60_PEAK_BW_GBS * 100.0) << "% of B60 peak)" << std::endl;
    std::cout << "Gap Ratio (M7 Peak / Aggr):  " << std::setprecision(1) << (M7_PEAK_BW_GBS / aggregate_bw) << "x" << std::endl;
    std::cout << "Explanation: shape-mix weighted average across " << NUM_PROJ_TYPES << " projection shapes at varying GPU occupancy." << std::endl;
    std::cout << "             Smallest shapes (N=48) use 1.9% of HW threads; M7 benchmarked only the largest (N=17408, 680%)." << std::endl;

    // =========================================================================
    // Fused Projections Benchmark: Directly measuring the fused kernels on B60
    // =========================================================================
    std::cout << "\n=========================================================================================================================" << std::endl;
    std::cout << "                              FUSED PROJECTION ACCELERATION ON INTEL ARC PRO B60 (M=1 DECODE)                            " << std::endl;
    std::cout << "=========================================================================================================================" << std::endl;

    std::vector<sycl::event> events_fa_fused;
    events_fa_fused.reserve(cfg.num_full_layers());
    for (size_t l = 0; l < layers.size(); ++l) {
        const auto& layer = layers[l];
        if (layer.layer_type == "full_attention") {
            ops::FusedProjectionDescFP32 fa_projs[3] = {
                {act_q_gate, static_cast<const uint8_t*>(layer.q_proj.d_weights_int4),
                 static_cast<const sycl::half*>(layer.q_proj.d_scales), nullptr, cfg.full_q_gate_dim()},
                {act_k, static_cast<const uint8_t*>(layer.k_proj.d_weights_int4),
                 static_cast<const sycl::half*>(layer.k_proj.d_scales), nullptr, cfg.full_k_dim()},
                {act_v, static_cast<const uint8_t*>(layer.v_proj.d_weights_int4),
                 static_cast<const sycl::half*>(layer.v_proj.d_scales), nullptr, cfg.full_v_dim()}
            };
            events_fa_fused.push_back(
                ops::linear_int4_fused(q, act_normed, fa_projs, 3, 1, hidden_size)
            );
        }
    }

    std::vector<sycl::event> events_la_fused;
    events_la_fused.reserve(cfg.num_linear_layers());
    for (size_t l = 0; l < layers.size(); ++l) {
        const auto& layer = layers[l];
        if (layer.layer_type != "full_attention") {
            ops::FusedProjectionDescFP32 la_projs[4] = {
                {act_qkv_raw, static_cast<const uint8_t*>(layer.in_proj_qkv.d_weights_int4),
                 static_cast<const sycl::half*>(layer.in_proj_qkv.d_scales), nullptr, cfg.linear_conv_channels},
                {act_z, static_cast<const uint8_t*>(layer.in_proj_z.d_weights_int4),
                 static_cast<const sycl::half*>(layer.in_proj_z.d_scales), nullptr, cfg.linear_z_dim},
                {act_b, static_cast<const uint8_t*>(layer.in_proj_b.d_weights_int4),
                 static_cast<const sycl::half*>(layer.in_proj_b.d_scales), nullptr, cfg.linear_b_dim},
                {act_a, static_cast<const uint8_t*>(layer.in_proj_a.d_weights_int4),
                 static_cast<const sycl::half*>(layer.in_proj_a.d_scales), nullptr, cfg.linear_a_dim}
            };
            events_la_fused.push_back(
                ops::linear_int4_fused(q, act_normed, la_projs, 4, 1, hidden_size)
            );
        }
    }

    std::vector<sycl::event> events_mlp_fused;
    events_mlp_fused.reserve(cfg.num_hidden_layers);
    for (size_t l = 0; l < layers.size(); ++l) {
        const auto& layer = layers[l];
        events_mlp_fused.push_back(
            ops::mlp_gate_up_swiglu_int4(q, act_mlp_gate, act_normed,
                                         static_cast<const uint8_t*>(layer.gate_proj.d_weights_int4),
                                         static_cast<const sycl::half*>(layer.gate_proj.d_scales),
                                         static_cast<const uint8_t*>(layer.up_proj.d_weights_int4),
                                         static_cast<const sycl::half*>(layer.up_proj.d_scales),
                                         1, intermediate_size, hidden_size)
        );
    }
    q.wait();

    auto calc_time = [](const std::vector<sycl::event>& evs) {
        double ms = 0.0;
        for (const auto& ev : evs) {
            uint64_t s = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
            uint64_t e = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
            ms += static_cast<double>(e - s) * 1e-6;
        }
        return ms;
    };

    double fa_fused_ms = calc_time(events_fa_fused);
    double la_fused_ms = calc_time(events_la_fused);
    double mlp_fused_ms = calc_time(events_mlp_fused);

    double fa_unfused_ms = projs[P_FA_Q].total_time_ms + projs[P_FA_K].total_time_ms + projs[P_FA_V].total_time_ms;
    double la_unfused_ms = projs[P_LA_QKV].total_time_ms + projs[P_LA_Z].total_time_ms + projs[P_LA_B].total_time_ms + projs[P_LA_A].total_time_ms;
    double mlp_unfused_ms = projs[P_MLP_GATE].total_time_ms + projs[P_MLP_UP].total_time_ms;

    std::cout << std::left << std::setw(32) << "Kernel Group"
              << std::right << std::setw(16) << "Unfused (ms)"
              << std::setw(16) << "Fused (ms)"
              << std::setw(14) << "Time Saved"
              << std::setw(14) << "Speedup"
              << std::setw(20) << "Workgroups Launched" << std::endl;
    std::cout << "-------------------------------------------------------------------------------------------------------------------------" << std::endl;

    auto print_comp = [](const std::string& name, double unfused, double fused, const std::string& wg_info) {
        double saved = unfused - fused;
        double spd = (fused > 0.0) ? (unfused / fused) : 1.0;
        std::cout << std::left << std::setw(32) << name
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(16) << unfused
                  << std::setw(16) << fused
                  << std::setw(12) << saved << " ms"
                  << std::setprecision(2) << std::setw(12) << spd << "x"
                  << std::setw(20) << wg_info << std::endl;
    };

    print_comp("Full-Attn Q+K+V (16 layers)", fa_unfused_ms, fa_fused_ms, "1792 WGs (was 16 WGs on K/V)");
    print_comp("Linear-Attn QKV+Z+B+A (48 layers)", la_unfused_ms, la_fused_ms, "2060 WGs (was 6 WGs on B/A)");
    print_comp("MLP Gate+Up+SwiGLU (64 layers)", mlp_unfused_ms, mlp_fused_ms, "4352 WGs (1 launch vs 3)");
    std::cout << "-------------------------------------------------------------------------------------------------------------------------" << std::endl;
    double total_unfused = fa_unfused_ms + la_unfused_ms + mlp_unfused_ms;
    double total_fused = fa_fused_ms + la_fused_ms + mlp_fused_ms;
    print_comp("TOTAL FUSED KERNELS", total_unfused, total_fused, "Full GPU Occupancy");
    std::cout << "=========================================================================================================================\n" << std::endl;

    sycl::free(act_x, q);
    sycl::free(act_normed, q);
    sycl::free(act_proj_out, q);
    sycl::free(act_mlp_gate, q);
    sycl::free(act_mlp_up, q);
    sycl::free(act_q_gate, q);
    sycl::free(act_k, q);
    sycl::free(act_v, q);
    sycl::free(act_attn_out, q);
    sycl::free(act_qkv_raw, q);
    sycl::free(act_z, q);
    sycl::free(act_b, q);
    sycl::free(act_a, q);
    sycl::free(act_delta_out, q);
    sycl::free(d_logits, q);

    return 0;
}
