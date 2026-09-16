#include "weights.h"
#include <iostream>
#include <cstring>
#include <sstream>

namespace xinfer::targets::qwen3_8_27b {

namespace {

inline float bf16_to_fp32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

} // anonymous namespace

LoadedModel::LoadedModel(std::shared_ptr<core::DeviceContext> ctx)
    : ctx_(std::move(ctx)) {}

LoadedModel::~LoadedModel() {
    if (ctx_) {
        for (void* ptr : allocated_device_ptrs_) {
            if (ptr) {
                ctx_->free_device(ptr);
            }
        }
    }
    allocated_device_ptrs_.clear();
}

std::unique_ptr<LoadedModel> LoadedModel::load_from_artifact(
    std::shared_ptr<core::DeviceContext> ctx,
    artifact::ArtifactReader& reader,
    std::string* error_msg
) {
    if (!ctx) {
        if (error_msg) *error_msg = "DeviceContext is null";
        return nullptr;
    }
    if (!reader.is_open()) {
        if (error_msg) *error_msg = "ArtifactReader is not open";
        return nullptr;
    }

    auto model = std::unique_ptr<LoadedModel>(new LoadedModel(ctx));

    // Helper lambda to allocate device memory and track for cleanup
    auto alloc_dev = [&](size_t bytes) -> void* {
        void* ptr = ctx->allocate_device(bytes);
        if (ptr) {
            model->allocated_device_ptrs_.push_back(ptr);
        }
        return ptr;
    };

    // Helper lambda to load INT4 LinearWeightView
    auto load_linear = [&](const std::string& weight_name,
                           int64_t out_f, int64_t in_f,
                           int group_size = 128) -> LinearWeightView {
        LinearWeightView view;
        view.in_features = in_f;
        view.out_features = out_f;
        view.group_size = group_size;

        size_t expected_w_bytes = static_cast<size_t>(out_f) * (in_f / 2);
        size_t expected_s_bytes = static_cast<size_t>(out_f) * (in_f / group_size) * sizeof(uint16_t);

        // Read weights
        std::vector<uint8_t> host_w;
        if (!reader.read_section(weight_name, host_w, error_msg)) {
            std::cerr << "[Error] Failed to read section: " << weight_name << std::endl;
            return view;
        }
        if (host_w.size() != expected_w_bytes) {
            std::cerr << "[Warning] Weight size mismatch for " << weight_name
                      << ": expected " << expected_w_bytes << ", got " << host_w.size() << std::endl;
        }

        view.d_weights_int4 = alloc_dev(host_w.size());
        ctx->copy_host_to_device(view.d_weights_int4, host_w.data(), host_w.size(), true);

        // Read scales
        std::string scale_name = weight_name + ".scales";
        std::vector<uint8_t> host_s;
        if (!reader.read_section(scale_name, host_s, error_msg)) {
            std::cerr << "[Error] Failed to read section: " << scale_name << std::endl;
            return view;
        }
        if (host_s.size() != expected_s_bytes) {
            std::cerr << "[Warning] Scales size mismatch for " << scale_name
                      << ": expected " << expected_s_bytes << ", got " << host_s.size() << std::endl;
        }

        view.d_scales = alloc_dev(host_s.size());
        ctx->copy_host_to_device(view.d_scales, host_s.data(), host_s.size(), true);

        return view;
    };

    // Helper lambda to load BF16 tensor and convert to FP32 on device
    // If add_unit_offset is true, scales as (1.0 + weight) matching Qwen3_5RMSNorm specification
    auto load_bf16_as_fp32 = [&](const std::string& section_name, size_t num_elements, bool add_unit_offset = false) -> float* {
        std::vector<uint8_t> host_raw;
        if (!reader.read_section(section_name, host_raw, error_msg)) {
            std::cerr << "[Error] Failed to read BF16 section: " << section_name << std::endl;
            return nullptr;
        }

        size_t count = host_raw.size() / sizeof(uint16_t);
        if (count != num_elements) {
            std::cerr << "[Warning] Element count mismatch for " << section_name
                      << ": expected " << num_elements << ", got " << count << std::endl;
        }

        const uint16_t* bf16_ptr = reinterpret_cast<const uint16_t*>(host_raw.data());
        std::vector<float> host_fp32(count);
        for (size_t i = 0; i < count; ++i) {
            float val = bf16_to_fp32(bf16_ptr[i]);
            if (add_unit_offset) {
                val += 1.0f;
            }
            host_fp32[i] = val;
        }

        size_t bytes = count * sizeof(float);
        float* d_ptr = static_cast<float*>(alloc_dev(bytes));
        ctx->copy_host_to_device(d_ptr, host_fp32.data(), bytes, true);
        return d_ptr;
    };

    std::cout << "[xinfer] Loading model weights into Intel Arc Pro B60 VRAM..." << std::endl;

    // 1. Embed tokens: [248320, 5120] BF16 (~2.54 GB)
    std::string embed_name = "model.language_model.embed_tokens.weight";
    std::vector<uint8_t> host_embed;
    if (!reader.read_section(embed_name, host_embed, error_msg)) {
        if (error_msg) *error_msg = "Failed to load embed_tokens section";
        return nullptr;
    }
    model->d_embed_tokens_ = alloc_dev(host_embed.size());
    ctx->copy_host_to_device(model->d_embed_tokens_, host_embed.data(), host_embed.size(), true);
    std::cout << "[xinfer] Loaded embed_tokens (" << (host_embed.size() / (1024 * 1024)) << " MB)" << std::endl;

    // 2. Final norm: [5120] FP32 (Qwen3_5RMSNorm: 1.0 + weight)
    std::string final_norm_name = "model.language_model.norm.weight";
    model->d_final_norm_ = load_bf16_as_fp32(final_norm_name, model->config_.hidden_size, true);
    if (!model->d_final_norm_) {
        if (error_msg) *error_msg = "Failed to load final norm";
        return nullptr;
    }

    // 3. LM head: [248320, 5120] INT4
    std::string lm_head_name = "lm_head.weight";
    model->lm_head_ = load_linear(lm_head_name, model->config_.vocab_size, model->config_.hidden_size, 128);
    if (!model->lm_head_.is_valid()) {
        if (error_msg) *error_msg = "Failed to load lm_head";
        return nullptr;
    }
    std::cout << "[xinfer] Loaded lm_head (INT4)" << std::endl;

    // 4. 64 Layers
    model->layers_.resize(model->config_.num_hidden_layers);

    for (int l = 0; l < model->config_.num_hidden_layers; ++l) {
        LayerWeights& layer = model->layers_[l];
        layer.layer_idx = l;
        std::string prefix = "model.language_model.layers." + std::to_string(l);

        // Layer norms (Qwen3_5RMSNorm: 1.0 + weight)
        layer.d_input_layernorm = load_bf16_as_fp32(prefix + ".input_layernorm.weight", model->config_.hidden_size, true);
        layer.d_post_attention_layernorm = load_bf16_as_fp32(prefix + ".post_attention_layernorm.weight", model->config_.hidden_size, true);

        // MLP
        layer.gate_proj = load_linear(prefix + ".mlp.gate_proj.weight", model->config_.intermediate_size, model->config_.hidden_size, 128);
        layer.up_proj   = load_linear(prefix + ".mlp.up_proj.weight", model->config_.intermediate_size, model->config_.hidden_size, 128);
        layer.down_proj = load_linear(prefix + ".mlp.down_proj.weight", model->config_.hidden_size, model->config_.intermediate_size, 128);

        // Token Mixer: alternating 3 linear attention and 1 full attention
        if (l % 4 == 3) {
            layer.layer_type = "full_attention";
            // Full attention:
            // q_proj: [12288, 5120]
            // k_proj: [1024, 5120]
            // v_proj: [1024, 5120]
            // o_proj: [5120, 6144]
            // q_norm: [256]
            // k_norm: [256]
            layer.q_proj = load_linear(prefix + ".self_attn.q_proj.weight", 12288, model->config_.hidden_size, 128);
            layer.k_proj = load_linear(prefix + ".self_attn.k_proj.weight", 1024, model->config_.hidden_size, 128);
            layer.v_proj = load_linear(prefix + ".self_attn.v_proj.weight", 1024, model->config_.hidden_size, 128);
            layer.o_proj = load_linear(prefix + ".self_attn.o_proj.weight", model->config_.hidden_size, 6144, 128);

            layer.d_q_norm = load_bf16_as_fp32(prefix + ".self_attn.q_norm.weight", 256, true);
            layer.d_k_norm = load_bf16_as_fp32(prefix + ".self_attn.k_norm.weight", 256, true);
        } else {
            layer.layer_type = "linear_attention";
            // Linear attention:
            // in_proj_qkv: [10240, 5120]
            // in_proj_z:   [6144, 5120]
            // in_proj_b:   [48, 5120]
            // in_proj_a:   [48, 5120]
            // out_proj:    [5120, 6144]
            // conv1d:      [10240, 1, 4] = 40960
            // A_log:       [48]
            // dt_bias:     [48]
            // norm:        [128]
            layer.in_proj_qkv = load_linear(prefix + ".linear_attn.in_proj_qkv.weight", 10240, model->config_.hidden_size, 128);
            layer.in_proj_z   = load_linear(prefix + ".linear_attn.in_proj_z.weight", 6144, model->config_.hidden_size, 128);
            layer.in_proj_b   = load_linear(prefix + ".linear_attn.in_proj_b.weight", 48, model->config_.hidden_size, 128);
            layer.in_proj_a   = load_linear(prefix + ".linear_attn.in_proj_a.weight", 48, model->config_.hidden_size, 128);
            layer.out_proj    = load_linear(prefix + ".linear_attn.out_proj.weight", model->config_.hidden_size, 6144, 128);

            layer.d_conv1d_weight = load_bf16_as_fp32(prefix + ".linear_attn.conv1d.weight", 10240 * 4);
            layer.d_A_log         = load_bf16_as_fp32(prefix + ".linear_attn.A_log", 48);
            layer.d_dt_bias       = load_bf16_as_fp32(prefix + ".linear_attn.dt_bias", 48);
            layer.d_norm_weight   = load_bf16_as_fp32(prefix + ".linear_attn.norm.weight", 128);
        }

        if ((l + 1) % 16 == 0 || l == model->config_.num_hidden_layers - 1) {
            std::cout << "[xinfer] Loaded layer " << (l + 1) << " / " << model->config_.num_hidden_layers << std::endl;
        }
    }

    std::cout << "[xinfer] Successfully loaded all 64 layers into Intel Arc Pro B60 VRAM!" << std::endl;
    return model;
}

} // namespace xinfer::targets::qwen3_8_27b
