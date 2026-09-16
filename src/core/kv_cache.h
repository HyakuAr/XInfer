#pragma once

#include "device.h"
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>

namespace xinfer::core {

struct KVCacheConfig {
    size_t max_seq_len{8192};
    size_t num_full_layers{16};
    size_t num_linear_layers{48};
    size_t num_kv_heads{4};
    size_t head_dim{256};
    size_t linear_num_v_heads{48};
    size_t linear_head_k_dim{128};
    size_t linear_head_v_dim{128};
    size_t linear_conv_channels{10240};
    size_t linear_conv_kernel_dim{4}; // kernel_size=4, needs (kernel_size - 1) = 3 past timesteps
};

// Physical container for KV cache (full attention) and recurrent/conv states (linear attention)
// per AGENTS.md §4 (owned by src/core).
class KVCache {
public:
    explicit KVCache(std::shared_ptr<DeviceContext> ctx, const KVCacheConfig& config = {});
    ~KVCache();

    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;

    KVCache(KVCache&&) noexcept;
    KVCache& operator=(KVCache&&) noexcept;

    // Allocate USM device buffers for KV cache and recurrent states
    bool allocate();

    // Reset sequence length and zero recurrent states on the GPU
    void clear();

    // Full-attention layer accessors (FP16 / sycl::half)
    // Shape per layer: [max_seq_len, num_kv_heads, head_dim]
    sycl::half* k_cache(size_t full_layer_idx) noexcept;
    const sycl::half* k_cache(size_t full_layer_idx) const noexcept;
    sycl::half* v_cache(size_t full_layer_idx) noexcept;
    const sycl::half* v_cache(size_t full_layer_idx) const noexcept;

    // Linear-attention recurrent state accessors (FP32)
    // S state shape per layer: [linear_num_v_heads, linear_head_k_dim, linear_head_v_dim]
    float* linear_state(size_t linear_layer_idx) noexcept;
    const float* linear_state(size_t linear_layer_idx) const noexcept;

    // Conv state shape per layer: [linear_conv_kernel_dim - 1, linear_conv_channels]
    float* conv_state(size_t linear_layer_idx) noexcept;
    const float* conv_state(size_t linear_layer_idx) const noexcept;

    size_t current_seq_len() const noexcept { return current_seq_len_; }
    void set_seq_len(size_t len) noexcept { current_seq_len_ = len; }
    void advance(size_t delta) noexcept { current_seq_len_ += delta; }

    const KVCacheConfig& config() const noexcept { return config_; }
    size_t total_allocated_bytes() const noexcept;

private:
    std::shared_ptr<DeviceContext> ctx_;
    KVCacheConfig config_;
    size_t current_seq_len_{0};

    std::vector<sycl::half*> k_caches_;
    std::vector<sycl::half*> v_caches_;
    std::vector<float*> linear_states_;
    std::vector<float*> conv_states_;

    void* d_raw_kv_storage_{nullptr};
    void* d_raw_recurrent_storage_{nullptr};
    size_t kv_storage_bytes_{0};
    size_t recurrent_storage_bytes_{0};
};

} // namespace xinfer::core
