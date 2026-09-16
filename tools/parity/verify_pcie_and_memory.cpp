#include "core/device.h"
#include "artifact/reader.h"
#include "targets/qwen3_8_27b/weights.h"
#include "targets/qwen3_8/decode_graph.h"
#include "core/kv_cache.h"

#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace xinfer;

int main(int argc, char** argv) {
    std::cout << "==================================================================" << std::endl;
    std::cout << "  xInfer Memory Residency & PCIe Activity Diagnostic Tool         " << std::endl;
    std::cout << "==================================================================" << std::endl;

    auto ctx = core::DeviceContext::create(true);
    sycl::queue& q = ctx->queue();
    auto dev = q.get_device();

    std::cout << "Target Device: " << dev.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Backend:       " << (ctx->has_native_level_zero() ? "Level Zero (Native)" : "Other") << std::endl;

    size_t total_global_mem = dev.get_info<sycl::info::device::global_mem_size>();
    std::cout << "Reported Device Global Memory: " << (total_global_mem / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;

    std::string artifact_path = (argc > 1) ? argv[1] : "out/qwen3_8_27b.xinfer";
    std::cout << "\nLoading model artifact: " << artifact_path << " ..." << std::endl;
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

    // Verify USM allocation types for model weights
    const auto& lm_head_weights = model->lm_head().d_weights_int4;
    const auto& embed_weights = model->d_embed_tokens();
    const auto& layer0_gate = model->layers()[0].gate_proj.d_weights_int4;

    auto check_ptr_residency = [&](const void* ptr, const std::string& label) {
        auto alloc_type = sycl::get_pointer_type(ptr, ctx->context());
        std::cout << "  " << std::left << std::setw(28) << label << ": ";
        if (alloc_type == sycl::usm::alloc::device) {
            std::cout << "[CONFIRMED] Local GPU VRAM (sycl::usm::alloc::device - NOT over PCIe)" << std::endl;
        } else if (alloc_type == sycl::usm::alloc::host) {
            std::cout << "[WARNING] Host RAM (sycl::usm::alloc::host - streamed over PCIe!)" << std::endl;
        } else if (alloc_type == sycl::usm::alloc::shared) {
            std::cout << "[WARNING] Shared USM (sycl::usm::alloc::shared - migratable over PCIe)" << std::endl;
        } else {
            std::cout << "[UNKNOWN] Unrecognized allocation type" << std::endl;
        }
    };

    std::cout << "\nVerifying Virtual Address and Physical Residency of Model Weights:" << std::endl;
    check_ptr_residency(embed_weights, "Embed Tokens Table");
    check_ptr_residency(layer0_gate, "Layer 0 Gate Proj Weights");
    check_ptr_residency(lm_head_weights, "LM Head Weights");

    // Setup DecodeGraph
    core::KVCacheConfig kv_cfg;
    kv_cfg.max_seq_len = 8192;
    core::KVCache kv_cache(ctx, kv_cfg);
    kv_cache.allocate();

    std::cout << "\nCapturing DecodeGraph..." << std::endl;
    targets::qwen3_8::DecodeGraph graph_runner(ctx, *model, kv_cache);
    if (!graph_runner.capture()) {
        std::cerr << "Failed to capture DecodeGraph!" << std::endl;
        return 1;
    }

    std::cout << "\nRunning continuous decode loop (10 tokens) on Arc Pro B60..." << std::endl;
    q.wait();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < 10; ++step) {
        int64_t next_tok = graph_runner.decode_step(760 + step, 10 + step);
        q.wait();
        std::cout << "  Step " << (step + 1) << "/10: token " << next_tok << std::endl;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Average step time across 10 tokens: " << (total_ms / 10.0) << " ms" << std::endl;

    return 0;
}
