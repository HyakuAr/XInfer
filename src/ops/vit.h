#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <vector>

namespace xinfer::ops {

struct ViTConfig {
    int64_t width{1024};             // ViT hidden dimension
    int64_t num_layers{24};          // ViT depth
    int64_t num_heads{16};           // ViT attention heads
    int64_t patch_size{14};          // ViT patch size (14x14)
    int64_t in_channels{3};          // RGB channels
    int64_t num_patches{256};        // Number of patches (e.g. 16x16 grid = 256)
    int64_t mlp_ratio{4};            // ViT MLP ratio (4 * 1024 = 4096)
    int64_t llm_hidden_size{5120};   // Target LLM hidden size for projection

    int64_t patch_dim() const noexcept { return in_channels * patch_size * patch_size; }
    int64_t mlp_dim() const noexcept { return width * mlp_ratio; }
    int64_t head_dim() const noexcept { return num_heads > 0 ? width / num_heads : 64; }
};

// ViT patch embedding: projects flattened image patches into ViT embedding dimension
// Supports optional patch bias and positional embedding addition.
sycl::event vit_patch_embed(
    sycl::queue& q,
    sycl::half* d_out_patches,           // [num_patches, width]
    const sycl::half* d_image_pixels,    // [num_patches, patch_dim]
    const sycl::half* d_patch_weight,    // [width, patch_dim]
    const sycl::half* d_patch_bias,      // [width] (optional, nullptr if none)
    const sycl::half* d_pos_embed,       // [num_patches, width] (optional, nullptr if none)
    int64_t num_patches,
    int64_t patch_dim,
    int64_t embed_dim
);

// ViT multi-head attention: self-attention for M patch tokens
sycl::event vit_attn(
    sycl::queue& q,
    sycl::half* d_out,                   // [num_patches, embed_dim]
    const sycl::half* d_in,              // [num_patches, embed_dim]
    const sycl::half* d_qkv_weight,      // [3 * embed_dim, embed_dim]
    const sycl::half* d_qkv_bias,        // [3 * embed_dim] (optional)
    const sycl::half* d_proj_weight,     // [embed_dim, embed_dim]
    const sycl::half* d_proj_bias,       // [embed_dim] (optional)
    int64_t num_patches,
    int64_t embed_dim,
    int64_t num_heads
);

// ViT MLP (feed-forward network): 2-layer MLP with GELU activation and residual
sycl::event vit_mlp(
    sycl::queue& q,
    sycl::half* d_out,                   // [num_patches, embed_dim]
    const sycl::half* d_in,              // [num_patches, embed_dim]
    const sycl::half* d_fc1_weight,      // [intermediate_dim, embed_dim]
    const sycl::half* d_fc1_bias,        // [intermediate_dim] (optional)
    const sycl::half* d_fc2_weight,      // [embed_dim, intermediate_dim]
    const sycl::half* d_fc2_bias,        // [embed_dim] (optional)
    int64_t num_patches,
    int64_t embed_dim,
    int64_t intermediate_dim
);

// ViT Projector: projects ViT output [num_patches, vit_dim] to LLM hidden dimension [num_patches, llm_dim]
sycl::event vit_project(
    sycl::queue& q,
    sycl::half* d_out_llm,               // [num_patches, llm_dim] (e.g. [256, 5120])
    const sycl::half* d_in_vit,          // [num_patches, vit_dim] (e.g. [256, 1024])
    const sycl::half* d_proj_weight,     // [llm_dim, vit_dim]
    const sycl::half* d_proj_bias,       // [llm_dim] (optional)
    int64_t num_patches,
    int64_t vit_dim,
    int64_t llm_dim
);

} // namespace xinfer::ops
