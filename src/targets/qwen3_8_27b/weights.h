#pragma once

#include "core/device.h"
#include "core/kv_cache.h"
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
    static constexpr int64_t kDefaultHiddenSize = 5120;
    static constexpr int64_t kDefaultIntermediateSize = 17408;
    static constexpr int64_t kDefaultNumHiddenLayers = 64;
    static constexpr int64_t kDefaultNumAttentionHeads = 24;
    static constexpr int64_t kDefaultNumKeyValueHeads = 4;
    static constexpr int64_t kDefaultHeadDim = 256;
    static constexpr int64_t kDefaultVocabSize = 248320;
    static constexpr float   kDefaultRmsNormEps = 1e-6f;
    static constexpr float   kDefaultRopeTheta = 10000000.0f;
    static constexpr int64_t kDefaultRopeDim = 64;
    static constexpr int     kDefaultGroupSize = 128;

    // Linear attention channel dimensions
    static constexpr int64_t kDefaultLinearConvChannels = 10240;
    static constexpr int64_t kDefaultLinearConvKernelDim = 4;
    static constexpr int64_t kDefaultLinearNumVHeads = 48;
    static constexpr int64_t kDefaultLinearNumKHeads = 16;
    static constexpr int64_t kDefaultLinearHeadKDim = 128;
    static constexpr int64_t kDefaultLinearHeadVDim = 128;
    static constexpr int64_t kDefaultLinearZDim = 6144;
    static constexpr int64_t kDefaultLinearBDim = 48;
    static constexpr int64_t kDefaultLinearADim = 48;
    static constexpr int64_t kDefaultLinearNormDim = 128;
    static constexpr int64_t kDefaultFullAttentionInterval = 4;

    int64_t hidden_size{kDefaultHiddenSize};
    int64_t intermediate_size{kDefaultIntermediateSize};
    int64_t num_hidden_layers{kDefaultNumHiddenLayers};
    int64_t num_attention_heads{kDefaultNumAttentionHeads};
    int64_t num_key_value_heads{kDefaultNumKeyValueHeads};
    int64_t head_dim{kDefaultHeadDim};
    int64_t vocab_size{kDefaultVocabSize};
    float   rms_norm_eps{kDefaultRmsNormEps};
    float   rope_theta{kDefaultRopeTheta};
    int64_t rope_dim{kDefaultRopeDim};
    int     group_size{kDefaultGroupSize};

    // Linear attention channel dimensions
    int64_t linear_conv_channels{kDefaultLinearConvChannels};
    int64_t linear_conv_kernel_dim{kDefaultLinearConvKernelDim};
    int64_t linear_num_v_heads{kDefaultLinearNumVHeads};
    int64_t linear_num_k_heads{kDefaultLinearNumKHeads};
    int64_t linear_head_k_dim{kDefaultLinearHeadKDim};
    int64_t linear_head_v_dim{kDefaultLinearHeadVDim};
    int64_t linear_z_dim{kDefaultLinearZDim};
    int64_t linear_b_dim{kDefaultLinearBDim};
    int64_t linear_a_dim{kDefaultLinearADim};
    int64_t linear_norm_dim{kDefaultLinearNormDim};
    int64_t full_attention_interval{kDefaultFullAttentionInterval};

    int64_t linear_out_dim() const noexcept { return linear_num_v_heads * linear_head_v_dim; } // 6144

    // Full attention derived dimensions
    int64_t full_q_gate_dim() const noexcept { return num_attention_heads * head_dim * 2; } // 12288
    int64_t full_q_dim() const noexcept { return num_attention_heads * head_dim; }         // 6144
    int64_t full_k_dim() const noexcept { return num_key_value_heads * head_dim; }         // 1024
    int64_t full_v_dim() const noexcept { return num_key_value_heads * head_dim; }         // 1024
    int64_t full_out_dim() const noexcept { return num_attention_heads * head_dim; }       // 6144

    // Dynamic layer types: if loaded from artifact metadata or populated from sections,
    // matches official transformers/models/qwen3_5/modeling_qwen3_5.py (Qwen3_5DecoderLayer, lines 733-739)
    // and config.json (text_config.layer_types).
    std::vector<std::string> layer_types;

    int64_t num_full_layers() const noexcept {
        if (!layer_types.empty()) {
            int64_t count = 0;
            for (const auto& t : layer_types) {
                if (t == "full_attention") count++;
            }
            return count;
        }
        int64_t count = 0;
        int64_t interval = full_attention_interval > 0 ? full_attention_interval : 4;
        for (int64_t l = 0; l < num_hidden_layers; ++l) {
            if (l % interval == interval - 1) count++;
        }
        return count;
    }

    int64_t num_linear_layers() const noexcept {
        return num_hidden_layers - num_full_layers();
    }

    bool is_full_attention_layer(int64_t l) const noexcept {
        if (l >= 0 && l < static_cast<int64_t>(layer_types.size())) {
            return layer_types[l] == "full_attention";
        }
        int64_t interval = full_attention_interval > 0 ? full_attention_interval : 4;
        return (l % interval == interval - 1);
    }

    // Vision encoder configuration (Qwen3.8 Vision ViT)
    static constexpr int64_t kDefaultVisionWidth = 1024;
    static constexpr int64_t kDefaultVisionLayers = 24;
    static constexpr int64_t kDefaultVisionHeads = 16;
    static constexpr int64_t kDefaultVisionMlpRatio = 4;
    static constexpr int64_t kDefaultVisionPatchSize = 14;
    static constexpr int64_t kDefaultPatchesPerImage = 256;
    static constexpr int64_t kDefaultMaxImageResolution = 1024;
    static constexpr float   kDefaultMaxAspectRatio = 4.0f;
    static constexpr int64_t kDefaultVisualTokenStart = 248000;
    static constexpr int64_t kDefaultVisualTokenEnd = 248319;
    static constexpr int64_t kDefaultImagePadTokenId = 248064;

    bool    has_vision{false};
    int64_t vision_width{kDefaultVisionWidth};
    int64_t vision_layers{kDefaultVisionLayers};
    int64_t vision_heads{kDefaultVisionHeads};
    int64_t vision_mlp_ratio{kDefaultVisionMlpRatio};
    int64_t vision_patch_size{kDefaultVisionPatchSize};
    int64_t patches_per_image{kDefaultPatchesPerImage};
    int64_t max_image_resolution{kDefaultMaxImageResolution};
    float   max_aspect_ratio{kDefaultMaxAspectRatio};
    int64_t visual_token_start{kDefaultVisualTokenStart};
    int64_t visual_token_end{kDefaultVisualTokenEnd};
    int64_t image_pad_token_id{kDefaultImagePadTokenId};

    std::string quant_scheme;
    bool is_int8_kv{false};

    core::KVCacheConfig create_kv_cache_config(size_t max_seq_len = 8192, bool use_int8_kv = false) const noexcept {
        core::KVCacheConfig cfg;
        cfg.max_seq_len = max_seq_len;
        cfg.num_full_layers = static_cast<size_t>(num_full_layers());
        cfg.num_linear_layers = static_cast<size_t>(num_linear_layers());
        cfg.num_kv_heads = static_cast<size_t>(num_key_value_heads);
        cfg.head_dim = static_cast<size_t>(head_dim);
        cfg.linear_num_v_heads = static_cast<size_t>(linear_num_v_heads);
        cfg.linear_head_k_dim = static_cast<size_t>(linear_head_k_dim);
        cfg.linear_head_v_dim = static_cast<size_t>(linear_head_v_dim);
        cfg.linear_conv_channels = static_cast<size_t>(linear_conv_channels);
        cfg.linear_conv_kernel_dim = static_cast<size_t>(linear_conv_kernel_dim);
        cfg.dtype = (use_int8_kv || is_int8_kv) ? core::KVCacheDType::INT8 : core::KVCacheDType::FP16;
        cfg.artifact_quant_scheme = quant_scheme;
        return cfg;
    }
};

struct VisionWeights {
    bool is_valid{false};
    int64_t width{1024};
    int64_t layers{24};
    int64_t heads{16};
    int64_t patch_size{14};
    int64_t in_channels{3};
    int64_t patches_per_image{256};

    // Patch embedding weights: [width, in_channels * patch_size * patch_size]
    void* d_patch_embed_weight{nullptr}; // FP16
    void* d_patch_embed_bias{nullptr};   // FP16/FP32
    void* d_pos_embed{nullptr};          // [patches_per_image, width]

    // ViT layers (qkv, out_proj, mlp)
    struct ViTLayerWeights {
        void* d_qkv_weight{nullptr};     // [3 * width, width]
        void* d_qkv_bias{nullptr};       // [3 * width]
        void* d_proj_weight{nullptr};    // [width, width]
        void* d_proj_bias{nullptr};      // [width]
        void* d_mlp_fc1_weight{nullptr}; // [mlp_dim, width]
        void* d_mlp_fc1_bias{nullptr};   // [mlp_dim]
        void* d_mlp_fc2_weight{nullptr}; // [width, mlp_dim]
        void* d_mlp_fc2_bias{nullptr};   // [width]
    };
    std::vector<ViTLayerWeights> layers_weights;

    // Visual Projector: ViT width (1024) -> LLM hidden_size (5120)
    void* d_projector_fc1_weight{nullptr}; // [llm_hidden_size, width]
    void* d_projector_fc1_bias{nullptr};   // [llm_hidden_size]
};

class LoadedModel {
public:
    static std::unique_ptr<LoadedModel> load_from_artifact(
        std::shared_ptr<core::DeviceContext> ctx,
        artifact::ArtifactReader& reader,
        std::string* error_msg = nullptr
    );

    // Mock factory for unit testing and bounds validation without disk artifact
    static std::unique_ptr<LoadedModel> create_mock(
        std::shared_ptr<core::DeviceContext> ctx,
        const ModelConfig& config = {}
    ) {
        auto model = std::unique_ptr<LoadedModel>(new LoadedModel(ctx));
        model->config_ = config;
        return model;
    }

    ~LoadedModel();

    LoadedModel(const LoadedModel&) = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;

    const ModelConfig& config() const noexcept { return config_; }
    const std::vector<LayerWeights>& layers() const noexcept { return layers_; }

    const void* d_embed_tokens() const noexcept { return d_embed_tokens_; }
    const float* d_final_norm() const noexcept { return d_final_norm_; }
    const LinearWeightView& lm_head() const noexcept { return lm_head_; }

    const VisionWeights& vision_weights() const noexcept { return vision_weights_; }
    bool has_vision() const noexcept { return vision_weights_.is_valid; }
    void set_vision_weights(VisionWeights weights) { vision_weights_ = std::move(weights); }

private:
    explicit LoadedModel(std::shared_ptr<core::DeviceContext> ctx);

    std::shared_ptr<core::DeviceContext> ctx_;
    ModelConfig config_;

    void*  d_embed_tokens_{nullptr}; // [vocab_size, hidden_size] BF16
    float* d_final_norm_{nullptr};   // [hidden_size] FP32
    LinearWeightView lm_head_;       // [vocab_size, hidden_size] INT4

    std::vector<LayerWeights> layers_;
    VisionWeights vision_weights_;
    std::vector<void*> allocated_device_ptrs_;
};

} // namespace xinfer::targets::qwen3_8_27b
