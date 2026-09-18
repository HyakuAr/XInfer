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

std::vector<std::string> parse_layer_types_json(std::string_view json_str) {
    std::vector<std::string> types;
    size_t i = 0;
    while (i < json_str.size() && json_str[i] != '[') i++;
    if (i >= json_str.size()) return types;
    i++; // skip '['

    while (i < json_str.size()) {
        while (i < json_str.size() && (json_str[i] == ' ' || json_str[i] == '\t' || json_str[i] == '\r' || json_str[i] == '\n' || json_str[i] == ',')) {
            i++;
        }
        if (i >= json_str.size() || json_str[i] == ']') break;
        if (json_str[i] == '"') {
            i++;
            size_t start = i;
            while (i < json_str.size() && json_str[i] != '"') {
                if (json_str[i] == '\\' && i + 1 < json_str.size()) i += 2;
                else i++;
            }
            types.emplace_back(json_str.substr(start, i - start));
            if (i < json_str.size()) i++; // skip closing '"'
        } else {
            i++;
        }
    }
    return types;
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

    // Validate and cross-check ModelConfig against artifact metadata
    const auto& meta = reader.metadata();
    model->config_.quant_scheme = meta.quant_scheme;
    if (meta.quant_scheme.find("INT8-KV") != std::string::npos ||
        (meta.properties.count("kv_quant_scheme") && meta.properties.at("kv_quant_scheme") == "INT8-KV")) {
        model->config_.is_int8_kv = true;
    }

    if (!meta.quant_scheme.empty()) {
        std::string expected_prefix = "INT4-G";
        if (meta.quant_scheme.rfind(expected_prefix, 0) == 0) {
            size_t end_pos = meta.quant_scheme.find("-SYM", expected_prefix.size());
            if (end_pos != std::string::npos) {
                int parsed_g = std::stoi(meta.quant_scheme.substr(expected_prefix.size(), end_pos - expected_prefix.size()));
                if (parsed_g != model->config_.group_size) {
                    std::string msg = "Artifact metadata quant_scheme mismatch: artifact specifies group_size " +
                                      std::to_string(parsed_g) + " (" + meta.quant_scheme +
                                      "), but model config expected " + std::to_string(model->config_.group_size);
                    std::cerr << "[Error] " << msg << std::endl;
                    if (error_msg) *error_msg = msg;
                    return nullptr;
                }
            }
        }
    }

    auto validate_int_prop = [&](const std::string& key, int64_t expected_val) -> bool {
        auto it = meta.properties.find(key);
        if (it == meta.properties.end() || it->second.empty()) {
            std::string msg = "Artifact metadata is missing required property '" + key + "'";
            std::cerr << "[Error] " << msg << std::endl;
            if (error_msg) *error_msg = msg;
            return false;
        }
        try {
            int64_t actual_val = std::stoll(it->second);
            if (actual_val != expected_val) {
                std::string msg = "Artifact metadata property mismatch for '" + key + "': expected " +
                                  std::to_string(expected_val) + ", got " + std::to_string(actual_val);
                std::cerr << "[Error] " << msg << std::endl;
                if (error_msg) *error_msg = msg;
                return false;
            }
        } catch (const std::exception& e) {
            std::string msg = "Artifact metadata property '" + key + "' is malformed: " + e.what();
            std::cerr << "[Error] " << msg << std::endl;
            if (error_msg) *error_msg = msg;
            return false;
        }
        return true;
    };

    auto validate_float_prop = [&](const std::string& key, float expected_val, float tol) -> bool {
        auto it = meta.properties.find(key);
        if (it == meta.properties.end() || it->second.empty()) {
            std::string msg = "Artifact metadata is missing required property '" + key + "'";
            std::cerr << "[Error] " << msg << std::endl;
            if (error_msg) *error_msg = msg;
            return false;
        }
        try {
            float actual_val = std::stof(it->second);
            if (std::abs(actual_val - expected_val) > tol) {
                std::string msg = "Artifact metadata property mismatch for '" + key + "': expected " +
                                  std::to_string(expected_val) + ", got " + std::to_string(actual_val);
                std::cerr << "[Error] " << msg << std::endl;
                if (error_msg) *error_msg = msg;
                return false;
            }
        } catch (const std::exception& e) {
            std::string msg = "Artifact metadata property '" + key + "' is malformed: " + e.what();
            std::cerr << "[Error] " << msg << std::endl;
            if (error_msg) *error_msg = msg;
            return false;
        }
        return true;
    };

    // Read full_attention_interval back from metadata and validate
    auto it_interval = meta.properties.find("full_attention_interval");
    if (it_interval == meta.properties.end() || it_interval->second.empty()) {
        std::string msg = "Artifact metadata is missing required property 'full_attention_interval'";
        std::cerr << "[Error] " << msg << std::endl;
        if (error_msg) *error_msg = msg;
        return nullptr;
    }
    int64_t parsed_interval = 0;
    try {
        parsed_interval = std::stoll(it_interval->second);
    } catch (const std::exception& e) {
        std::string msg = "Artifact metadata property 'full_attention_interval' is malformed: " + std::string(e.what());
        std::cerr << "[Error] " << msg << std::endl;
        if (error_msg) *error_msg = msg;
        return nullptr;
    }
    if (parsed_interval <= 0) {
        std::string msg = "Artifact metadata property 'full_attention_interval' must be positive, got " +
                          std::to_string(parsed_interval);
        std::cerr << "[Error] " << msg << std::endl;
        if (error_msg) *error_msg = msg;
        return nullptr;
    }
    model->config_.full_attention_interval = parsed_interval;

    if (!validate_int_prop("hidden_size", model->config_.hidden_size) ||
        !validate_int_prop("intermediate_size", model->config_.intermediate_size) ||
        !validate_int_prop("num_hidden_layers", model->config_.num_hidden_layers) ||
        !validate_int_prop("num_attention_heads", model->config_.num_attention_heads) ||
        !validate_int_prop("num_key_value_heads", model->config_.num_key_value_heads) ||
        !validate_int_prop("head_dim", model->config_.head_dim) ||
        !validate_int_prop("vocab_size", model->config_.vocab_size) ||
        !validate_float_prop("rms_norm_eps", model->config_.rms_norm_eps, 1e-7f) ||
        !validate_float_prop("rope_theta", model->config_.rope_theta, 1.0f) ||
        !validate_int_prop("full_attention_interval", model->config_.full_attention_interval)) {
        return nullptr;
    }

    auto validate_optional_int_prop = [&](const std::string& key, int64_t expected_val) -> bool {
        auto it = meta.properties.find(key);
        if (it != meta.properties.end() && !it->second.empty()) {
            try {
                int64_t actual_val = std::stoll(it->second);
                if (actual_val != expected_val) {
                    std::string msg = "Artifact metadata property mismatch for '" + key + "': expected " +
                                      std::to_string(expected_val) + ", got " + std::to_string(actual_val);
                    std::cerr << "[Error] " << msg << std::endl;
                    if (error_msg) *error_msg = msg;
                    return false;
                }
            } catch (const std::exception& e) {
                std::string msg = "Artifact metadata property '" + key + "' is malformed: " + e.what();
                std::cerr << "[Error] " << msg << std::endl;
                if (error_msg) *error_msg = msg;
                return false;
            }
        }
        return true;
    };

    if (!validate_optional_int_prop("linear_conv_channels", model->config_.linear_conv_channels) ||
        !validate_optional_int_prop("linear_conv_kernel_dim", model->config_.linear_conv_kernel_dim) ||
        !validate_optional_int_prop("linear_num_v_heads", model->config_.linear_num_v_heads) ||
        !validate_optional_int_prop("linear_num_k_heads", model->config_.linear_num_k_heads) ||
        !validate_optional_int_prop("linear_head_k_dim", model->config_.linear_head_k_dim) ||
        !validate_optional_int_prop("linear_head_v_dim", model->config_.linear_head_v_dim) ||
        !validate_optional_int_prop("linear_z_dim", model->config_.linear_z_dim)) {
        return nullptr;
    }

    // Parse optional Vision configuration
    auto it_has_vision = meta.properties.find("has_vision");
    if (it_has_vision != meta.properties.end()) {
        model->config_.has_vision = (it_has_vision->second == "true" || it_has_vision->second == "1");
    }
    validate_optional_int_prop("vision_width", model->config_.vision_width);
    validate_optional_int_prop("vision_layers", model->config_.vision_layers);
    validate_optional_int_prop("vision_heads", model->config_.vision_heads);
    validate_optional_int_prop("vision_mlp_ratio", model->config_.vision_mlp_ratio);
    validate_optional_int_prop("vision_patch_size", model->config_.vision_patch_size);
    validate_optional_int_prop("patches_per_image", model->config_.patches_per_image);
    validate_optional_int_prop("max_image_resolution", model->config_.max_image_resolution);
    validate_optional_int_prop("visual_token_start", model->config_.visual_token_start);
    validate_optional_int_prop("visual_token_end", model->config_.visual_token_end);
    validate_optional_int_prop("image_pad_token_id", model->config_.image_pad_token_id);

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
                           int group_size = 0) -> LinearWeightView {
        if (group_size <= 0) group_size = model->config_.group_size;
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
            std::string msg = "Weight size mismatch for " + weight_name +
                              ": expected " + std::to_string(expected_w_bytes) +
                              ", got " + std::to_string(host_w.size());
            std::cerr << "[Error] " << msg << std::endl;
            if (error_msg) *error_msg = msg;
            return view;
        }

        view.d_weights_int4 = alloc_dev(host_w.size());
        ctx->copy_host_to_device(view.d_weights_int4, host_w.data(), host_w.size(), true);

        // Read scales
        std::string scale_name = weight_name + ".scales";
        std::vector<uint8_t> host_s;
        if (!reader.read_section(scale_name, host_s, error_msg)) {
            std::cerr << "[Error] Failed to read section: " << scale_name << std::endl;
            view.d_weights_int4 = nullptr;
            return view;
        }
        if (host_s.size() != expected_s_bytes) {
            std::string msg = "Scales size mismatch for " + scale_name +
                              ": expected " + std::to_string(expected_s_bytes) +
                              ", got " + std::to_string(host_s.size());
            std::cerr << "[Error] " << msg << std::endl;
            if (error_msg) *error_msg = msg;
            view.d_weights_int4 = nullptr;
            return view;
        }

        view.d_scales = alloc_dev(host_s.size());
        ctx->copy_host_to_device(view.d_scales, host_s.data(), host_s.size(), true);

        return view;
    };

    // Helper lambda to load BF16 tensor and convert to FP32 on device.
    // If add_unit_offset is true, scales as (1.0 + weight) matching Qwen3_5RMSNorm specification:
    // Citing official transformers/models/qwen3_5/modeling_qwen3_5.py:
    // - class Qwen3_5RMSNorm (lines 711-727, 807-808):
    //     self.weight = nn.Parameter(torch.zeros(dim))
    //     output = output * (1.0 + self.weight.float())
    //   "We initialize with 0s to be 1 centered as the RMSNorm here does (1 + weight)"
    //   Used for: input_layernorm (line 741), post_attention_layernorm (line 742),
    //   q_norm (line 644), k_norm (line 645), and final norm (line 1218).
    // - In contrast, class Qwen3_5RMSNormGated (lines 175-188, 392) used in linear attention:
    //     self.weight = nn.Parameter(torch.ones(hidden_size))
    //     hidden_states = self.weight * hidden_states
    //   is 1.0-initialized and does NOT use (1.0 + weight) offset.
    auto load_bf16_as_fp32 = [&](const std::string& section_name, size_t num_elements, bool add_unit_offset = false) -> float* {
        std::vector<uint8_t> host_raw;
        if (!reader.read_section(section_name, host_raw, error_msg)) {
            std::cerr << "[Error] Failed to read BF16 section: " << section_name << std::endl;
            return nullptr;
        }

        size_t expected_bytes = num_elements * sizeof(uint16_t);
        if (host_raw.size() != expected_bytes) {
            std::string msg = "Element count/size mismatch for " + section_name +
                              ": expected " + std::to_string(expected_bytes) + " bytes (" +
                              std::to_string(num_elements) + " elements), got " +
                              std::to_string(host_raw.size()) + " bytes (" +
                              std::to_string(host_raw.size() / sizeof(uint16_t)) + " elements)";
            std::cerr << "[Error] " << msg << std::endl;
            if (error_msg) *error_msg = msg;
            return nullptr;
        }

        size_t count = num_elements;
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
        if (error_msg && error_msg->empty()) *error_msg = "Failed to load embed_tokens section";
        return nullptr;
    }
    size_t expected_embed_bytes = static_cast<size_t>(model->config_.vocab_size) * model->config_.hidden_size * sizeof(uint16_t);
    if (host_embed.size() != expected_embed_bytes) {
        std::string msg = "Embed tokens size mismatch for " + embed_name +
                          ": expected " + std::to_string(expected_embed_bytes) +
                          ", got " + std::to_string(host_embed.size());
        std::cerr << "[Error] " << msg << std::endl;
        if (error_msg) *error_msg = msg;
        return nullptr;
    }
    model->d_embed_tokens_ = alloc_dev(host_embed.size());
    ctx->copy_host_to_device(model->d_embed_tokens_, host_embed.data(), host_embed.size(), true);
    std::cout << "[xinfer] Loaded embed_tokens (" << (host_embed.size() / (1024 * 1024)) << " MB)" << std::endl;

    // 2. Final norm: [5120] FP32 (Qwen3_5RMSNorm: 1.0 + weight)
    std::string final_norm_name = "model.language_model.norm.weight";
    model->d_final_norm_ = load_bf16_as_fp32(final_norm_name, model->config_.hidden_size, true);
    if (!model->d_final_norm_) {
        if (error_msg && error_msg->empty()) *error_msg = "Failed to load final norm";
        return nullptr;
    }

    // 3. LM head: INT4
    std::string lm_head_name = "lm_head.weight";
    model->lm_head_ = load_linear(lm_head_name, model->config_.vocab_size, model->config_.hidden_size);
    if (!model->lm_head_.is_valid()) {
        if (error_msg && error_msg->empty()) *error_msg = "Failed to load lm_head";
        return nullptr;
    }
    std::cout << "[xinfer] Loaded lm_head (INT4)" << std::endl;

    // 4. Layers
    model->layers_.resize(model->config_.num_hidden_layers);
    model->config_.layer_types.clear();
    model->config_.layer_types.reserve(model->config_.num_hidden_layers);

    auto it_layer_types = meta.properties.find("layer_types");
    std::vector<std::string> meta_layer_types;
    if (it_layer_types != meta.properties.end() && !it_layer_types->second.empty()) {
        meta_layer_types = parse_layer_types_json(it_layer_types->second);
        if (!meta_layer_types.empty() && meta_layer_types.size() != static_cast<size_t>(model->config_.num_hidden_layers)) {
            std::string msg = "Artifact metadata 'layer_types' count (" + std::to_string(meta_layer_types.size()) +
                              ") does not match num_hidden_layers (" + std::to_string(model->config_.num_hidden_layers) + ")";
            std::cerr << "[Error] " << msg << std::endl;
            if (error_msg) *error_msg = msg;
            return nullptr;
        }
    }

    for (int l = 0; l < model->config_.num_hidden_layers; ++l) {
        LayerWeights& layer = model->layers_[l];
        layer.layer_idx = l;
        std::string prefix = "model.language_model.layers." + std::to_string(l);

        // Layer norms (Qwen3_5RMSNorm: 1.0 + weight)
        layer.d_input_layernorm = load_bf16_as_fp32(prefix + ".input_layernorm.weight", model->config_.hidden_size, true);
        if (!layer.d_input_layernorm) {
            std::string msg = "Failed to load input_layernorm for layer " + std::to_string(l);
            std::cerr << "[Error] " << msg << std::endl;
            if (error_msg && error_msg->empty()) *error_msg = msg;
            return nullptr;
        }

        layer.d_post_attention_layernorm = load_bf16_as_fp32(prefix + ".post_attention_layernorm.weight", model->config_.hidden_size, true);
        if (!layer.d_post_attention_layernorm) {
            std::string msg = "Failed to load post_attention_layernorm for layer " + std::to_string(l);
            std::cerr << "[Error] " << msg << std::endl;
            if (error_msg && error_msg->empty()) *error_msg = msg;
            return nullptr;
        }

        // MLP
        layer.gate_proj = load_linear(prefix + ".mlp.gate_proj.weight", model->config_.intermediate_size, model->config_.hidden_size);
        if (!layer.gate_proj.is_valid()) {
            std::string msg = "Failed to load mlp.gate_proj for layer " + std::to_string(l);
            std::cerr << "[Error] " << msg << std::endl;
            if (error_msg && error_msg->empty()) *error_msg = msg;
            return nullptr;
        }

        layer.up_proj = load_linear(prefix + ".mlp.up_proj.weight", model->config_.intermediate_size, model->config_.hidden_size);
        if (!layer.up_proj.is_valid()) {
            std::string msg = "Failed to load mlp.up_proj for layer " + std::to_string(l);
            std::cerr << "[Error] " << msg << std::endl;
            if (error_msg && error_msg->empty()) *error_msg = msg;
            return nullptr;
        }

        layer.down_proj = load_linear(prefix + ".mlp.down_proj.weight", model->config_.hidden_size, model->config_.intermediate_size);
        if (!layer.down_proj.is_valid()) {
            std::string msg = "Failed to load mlp.down_proj for layer " + std::to_string(l);
            std::cerr << "[Error] " << msg << std::endl;
            if (error_msg && error_msg->empty()) *error_msg = msg;
            return nullptr;
        }

        // Token Mixer: determine layer type directly from artifact sections and metadata.
        // Citing transformers/models/qwen3_5/modeling_qwen3_5.py (Qwen3_5DecoderLayer.__init__, lines 733-739)
        // and config.json (text_config.layer_types and text_config.full_attention_interval):
        //   self.layer_type = config.layer_types[layer_idx]
        //   if self.layer_type == "linear_attention": self.linear_attn = Qwen3_5GatedDeltaNet(...)
        //   elif self.layer_type == "full_attention": self.self_attn = Qwen3_5Attention(...)
        // Inspect actual artifact container sections and cross-check against metadata and full_attention_interval.
        bool has_full_attn = reader.has_section(prefix + ".self_attn.q_proj.weight");
        bool has_linear_attn = reader.has_section(prefix + ".linear_attn.in_proj_qkv.weight");

        int64_t interval = model->config_.full_attention_interval;
        if (interval <= 0) {
            std::string msg = "Invalid full_attention_interval in model config: " + std::to_string(interval);
            std::cerr << "[Error] " << msg << std::endl;
            if (error_msg) *error_msg = msg;
            return nullptr;
        }

        std::string expected_by_interval = ((l % interval) == (interval - 1)) ? "full_attention" : "linear_attention";

        std::string detected_type;
        if (has_full_attn && !has_linear_attn) {
            detected_type = "full_attention";
        } else if (has_linear_attn && !has_full_attn) {
            detected_type = "linear_attention";
        } else {
            // Fallback to metadata layer_types if present, or interval formula using full_attention_interval
            if (l < static_cast<int64_t>(meta_layer_types.size())) {
                detected_type = meta_layer_types[l];
            } else {
                detected_type = expected_by_interval;
            }
        }

        // Fail-loud cross-check: verify detected layer type against metadata and interval
        if (l < static_cast<int64_t>(meta_layer_types.size())) {
            if (detected_type != meta_layer_types[l]) {
                std::string msg = "Layer " + std::to_string(l) + " type mismatch: section detection gave '" +
                                  detected_type + "', but metadata specifies '" + meta_layer_types[l] + "'";
                std::cerr << "[Error] " << msg << std::endl;
                if (error_msg) *error_msg = msg;
                return nullptr;
            }
        } else {
            if (detected_type != expected_by_interval) {
                std::string msg = "Layer " + std::to_string(l) + " type mismatch: section detection gave '" +
                                  detected_type + "', but full_attention_interval=" + std::to_string(interval) +
                                  " requires '" + expected_by_interval + "'";
                std::cerr << "[Error] " << msg << std::endl;
                if (error_msg) *error_msg = msg;
                return nullptr;
            }
        }

        layer.layer_type = detected_type;
        model->config_.layer_types.push_back(layer.layer_type);

        if (layer.layer_type == "full_attention") {
            layer.q_proj = load_linear(prefix + ".self_attn.q_proj.weight", model->config_.full_q_gate_dim(), model->config_.hidden_size);
            if (!layer.q_proj.is_valid()) {
                std::string msg = "Failed to load self_attn.q_proj for layer " + std::to_string(l);
                std::cerr << "[Error] " << msg << std::endl;
                if (error_msg && error_msg->empty()) *error_msg = msg;
                return nullptr;
            }

            layer.k_proj = load_linear(prefix + ".self_attn.k_proj.weight", model->config_.full_k_dim(), model->config_.hidden_size);
            if (!layer.k_proj.is_valid()) {
                std::string msg = "Failed to load self_attn.k_proj for layer " + std::to_string(l);
                std::cerr << "[Error] " << msg << std::endl;
                if (error_msg && error_msg->empty()) *error_msg = msg;
                return nullptr;
            }

            layer.v_proj = load_linear(prefix + ".self_attn.v_proj.weight", model->config_.full_v_dim(), model->config_.hidden_size);
            if (!layer.v_proj.is_valid()) {
                std::string msg = "Failed to load self_attn.v_proj for layer " + std::to_string(l);
                std::cerr << "[Error] " << msg << std::endl;
                if (error_msg && error_msg->empty()) *error_msg = msg;
                return nullptr;
            }

            layer.o_proj = load_linear(prefix + ".self_attn.o_proj.weight", model->config_.hidden_size, model->config_.full_out_dim());
            if (!layer.o_proj.is_valid()) {
                std::string msg = "Failed to load self_attn.o_proj for layer " + std::to_string(l);
                std::cerr << "[Error] " << msg << std::endl;
                if (error_msg && error_msg->empty()) *error_msg = msg;
                return nullptr;
            }

            layer.d_q_norm = load_bf16_as_fp32(prefix + ".self_attn.q_norm.weight", model->config_.head_dim, true);
            if (!layer.d_q_norm) {
                std::string msg = "Failed to load self_attn.q_norm for layer " + std::to_string(l);
                std::cerr << "[Error] " << msg << std::endl;
                if (error_msg && error_msg->empty()) *error_msg = msg;
                return nullptr;
            }

            layer.d_k_norm = load_bf16_as_fp32(prefix + ".self_attn.k_norm.weight", model->config_.head_dim, true);
            if (!layer.d_k_norm) {
                std::string msg = "Failed to load self_attn.k_norm for layer " + std::to_string(l);
                std::cerr << "[Error] " << msg << std::endl;
                if (error_msg && error_msg->empty()) *error_msg = msg;
                return nullptr;
            }
        } else {
            layer.layer_type = "linear_attention";
            layer.in_proj_qkv = load_linear(prefix + ".linear_attn.in_proj_qkv.weight", model->config_.linear_conv_channels, model->config_.hidden_size);
            if (!layer.in_proj_qkv.is_valid()) {
                std::string msg = "Failed to load linear_attn.in_proj_qkv for layer " + std::to_string(l);
                std::cerr << "[Error] " << msg << std::endl;
                if (error_msg && error_msg->empty()) *error_msg = msg;
                return nullptr;
            }

            layer.in_proj_z = load_linear(prefix + ".linear_attn.in_proj_z.weight", model->config_.linear_z_dim, model->config_.hidden_size);
            if (!layer.in_proj_z.is_valid()) {
                std::string msg = "Failed to load linear_attn.in_proj_z for layer " + std::to_string(l);
                std::cerr << "[Error] " << msg << std::endl;
                if (error_msg && error_msg->empty()) *error_msg = msg;
                return nullptr;
            }

            layer.in_proj_b = load_linear(prefix + ".linear_attn.in_proj_b.weight", model->config_.linear_b_dim, model->config_.hidden_size);
            if (!layer.in_proj_b.is_valid()) {
                std::string msg = "Failed to load linear_attn.in_proj_b for layer " + std::to_string(l);
                std::cerr << "[Error] " << msg << std::endl;
                if (error_msg && error_msg->empty()) *error_msg = msg;
                return nullptr;
            }

            layer.in_proj_a = load_linear(prefix + ".linear_attn.in_proj_a.weight", model->config_.linear_a_dim, model->config_.hidden_size);
            if (!layer.in_proj_a.is_valid()) {
                std::string msg = "Failed to load linear_attn.in_proj_a for layer " + std::to_string(l);
                std::cerr << "[Error] " << msg << std::endl;
                if (error_msg && error_msg->empty()) *error_msg = msg;
                return nullptr;
            }

            layer.out_proj = load_linear(prefix + ".linear_attn.out_proj.weight", model->config_.hidden_size, model->config_.linear_z_dim);
            if (!layer.out_proj.is_valid()) {
                std::string msg = "Failed to load linear_attn.out_proj for layer " + std::to_string(l);
                std::cerr << "[Error] " << msg << std::endl;
                if (error_msg && error_msg->empty()) *error_msg = msg;
                return nullptr;
            }

            layer.d_conv1d_weight = load_bf16_as_fp32(prefix + ".linear_attn.conv1d.weight", model->config_.linear_conv_channels * model->config_.linear_conv_kernel_dim);
            if (!layer.d_conv1d_weight) {
                std::string msg = "Failed to load linear_attn.conv1d.weight for layer " + std::to_string(l);
                std::cerr << "[Error] " << msg << std::endl;
                if (error_msg && error_msg->empty()) *error_msg = msg;
                return nullptr;
            }

            layer.d_A_log = load_bf16_as_fp32(prefix + ".linear_attn.A_log", model->config_.linear_num_v_heads);
            if (!layer.d_A_log) {
                std::string msg = "Failed to load linear_attn.A_log for layer " + std::to_string(l);
                std::cerr << "[Error] " << msg << std::endl;
                if (error_msg && error_msg->empty()) *error_msg = msg;
                return nullptr;
            }

            layer.d_dt_bias = load_bf16_as_fp32(prefix + ".linear_attn.dt_bias", model->config_.linear_num_v_heads);
            if (!layer.d_dt_bias) {
                std::string msg = "Failed to load linear_attn.dt_bias for layer " + std::to_string(l);
                std::cerr << "[Error] " << msg << std::endl;
                if (error_msg && error_msg->empty()) *error_msg = msg;
                return nullptr;
            }

            layer.d_norm_weight = load_bf16_as_fp32(prefix + ".linear_attn.norm.weight", model->config_.linear_norm_dim);
            if (!layer.d_norm_weight) {
                std::string msg = "Failed to load linear_attn.norm.weight for layer " + std::to_string(l);
                std::cerr << "[Error] " << msg << std::endl;
                if (error_msg && error_msg->empty()) *error_msg = msg;
                return nullptr;
            }
        }

        if ((l + 1) % 16 == 0 || l == model->config_.num_hidden_layers - 1) {
            std::cout << "[xinfer] Loaded layer " << (l + 1) << " / " << model->config_.num_hidden_layers << std::endl;
        }
    }

    auto load_fp16 = [&](const std::string& section_name, size_t num_elements) -> void* {
        std::vector<uint8_t> host_raw;
        if (!reader.read_section(section_name, host_raw, error_msg)) {
            return nullptr;
        }
        size_t expected_bytes = num_elements * sizeof(uint16_t);
        if (host_raw.size() != expected_bytes) {
            std::string msg = "Size mismatch for " + section_name + ": expected " +
                              std::to_string(expected_bytes) + ", got " + std::to_string(host_raw.size());
            std::cerr << "[Error] " << msg << std::endl;
            if (error_msg) *error_msg = msg;
            return nullptr;
        }
        void* d_ptr = alloc_dev(host_raw.size());
        if (d_ptr) {
            ctx->copy_host_to_device(d_ptr, host_raw.data(), host_raw.size(), true);
        }
        return d_ptr;
    };

    // Load Vision weights if present in artifact
    if (model->config_.has_vision || reader.has_section("visual.patch_embed.weight")) {
        std::cout << "[xinfer] Loading Vision ViT weights into Intel Arc Pro B60 VRAM..." << std::endl;
        VisionWeights v_weights;
        v_weights.width = model->config_.vision_width;
        v_weights.layers = model->config_.vision_layers;
        v_weights.heads = model->config_.vision_heads;
        v_weights.patch_size = model->config_.vision_patch_size;
        v_weights.patches_per_image = model->config_.patches_per_image;

        int64_t patch_dim = v_weights.in_channels * v_weights.patch_size * v_weights.patch_size;
        if (reader.has_section("visual.patch_embed.weight")) {
            v_weights.d_patch_embed_weight = load_fp16("visual.patch_embed.weight", v_weights.width * patch_dim);
        }
        if (reader.has_section("visual.patch_embed.bias")) {
            v_weights.d_patch_embed_bias = load_fp16("visual.patch_embed.bias", v_weights.width);
        }
        if (reader.has_section("visual.pos_embed")) {
            v_weights.d_pos_embed = load_fp16("visual.pos_embed", v_weights.patches_per_image * v_weights.width);
        }

        // Projector
        if (reader.has_section("visual.projector.weight")) {
            v_weights.d_projector_fc1_weight = load_fp16("visual.projector.weight", model->config_.hidden_size * v_weights.width);
        }
        if (reader.has_section("visual.projector.bias")) {
            v_weights.d_projector_fc1_bias = load_fp16("visual.projector.bias", model->config_.hidden_size);
        }

        v_weights.is_valid = (v_weights.d_patch_embed_weight != nullptr);
        model->vision_weights_ = v_weights;
        model->config_.has_vision = v_weights.is_valid;
        if (v_weights.is_valid) {
            std::cout << "[xinfer] Successfully loaded Vision ViT weights!" << std::endl;
        }
    }

    std::cout << "[xinfer] Successfully loaded all 64 layers into Intel Arc Pro B60 VRAM!" << std::endl;
    return model;
}

} // namespace xinfer::targets::qwen3_8_27b
