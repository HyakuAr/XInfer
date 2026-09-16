#pragma once

#include "core/device.h"
#include "artifact/reader.h"
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace xinfer::targets::qwen3_8_27b {

struct LinearWeightView {
    void*  d_weights_int4{nullptr}; // Packed INT4 device pointer
    void*  d_scales{nullptr};       // FP16 scales device pointer
    int64_t in_features{0};
    int64_t out_features{0};
    int     group_size{128};

    bool is_valid() const noexcept {
        return d_weights_int4 != nullptr && d_scales != nullptr;
    }
};

struct LayerWeights {
    int         layer_idx{0};
    std::string layer_type; // "linear_attention" or "full_attention"

    // Layer norms
    float* d_input_layernorm{nullptr};
    float* d_post_attention_layernorm{nullptr};

    // Full Attention weights (layer_idx % 4 == 3)
    LinearWeightView q_proj;
    LinearWeightView k_proj;
    LinearWeightView v_proj;
    LinearWeightView o_proj;
    float*           d_q_norm{nullptr};
    float*           d_k_norm{nullptr};

    // Linear Attention weights (layer_idx % 4 != 3)
    LinearWeightView in_proj_qkv;
    LinearWeightView in_proj_z;
    LinearWeightView in_proj_b;
    LinearWeightView in_proj_a;
    LinearWeightView out_proj;
    float*           d_conv1d_weight{nullptr};
    float*           d_A_log{nullptr};
    float*           d_dt_bias{nullptr};
    float*           d_norm_weight{nullptr};

    // MLP weights (all layers)
    LinearWeightView gate_proj;
    LinearWeightView up_proj;
    LinearWeightView down_proj;
};

struct ModelConfig {
    int64_t hidden_size{5120};
    int64_t intermediate_size{17408};
    int64_t num_hidden_layers{64};
    int64_t num_attention_heads{24};
    int64_t num_key_value_heads{4};
    int64_t head_dim{256};
    int64_t vocab_size{248320};
    float   rms_norm_eps{1e-6f};
    float   rope_theta{10000000.0f};
};

class LoadedModel {
public:
    static std::unique_ptr<LoadedModel> load_from_artifact(
        std::shared_ptr<core::DeviceContext> ctx,
        artifact::ArtifactReader& reader,
        std::string* error_msg = nullptr
    );

    ~LoadedModel();

    LoadedModel(const LoadedModel&) = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;

    const ModelConfig& config() const noexcept { return config_; }
    const std::vector<LayerWeights>& layers() const noexcept { return layers_; }

    const void* d_embed_tokens() const noexcept { return d_embed_tokens_; }
    const float* d_final_norm() const noexcept { return d_final_norm_; }
    const LinearWeightView& lm_head() const noexcept { return lm_head_; }

private:
    explicit LoadedModel(std::shared_ptr<core::DeviceContext> ctx);

    std::shared_ptr<core::DeviceContext> ctx_;
    ModelConfig config_;

    void*  d_embed_tokens_{nullptr}; // [vocab_size, hidden_size] BF16
    float* d_final_norm_{nullptr};   // [hidden_size] FP32
    LinearWeightView lm_head_;       // [vocab_size, hidden_size] INT4

    std::vector<LayerWeights> layers_;
    std::vector<void*> allocated_device_ptrs_;
};

} // namespace xinfer::targets::qwen3_8_27b
