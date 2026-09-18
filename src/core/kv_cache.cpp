#include "kv_cache.h"
#include <iostream>
#include <cassert>
#include <stdexcept>

namespace xinfer::core {

KVCache::KVCache(std::shared_ptr<DeviceContext> ctx, const KVCacheConfig& config)
    : ctx_(std::move(ctx)), config_(config) {
}

KVCache::~KVCache() {
    if (ctx_) {
        auto& q = ctx_->queue();
        if (d_raw_kv_storage_) {
            sycl::free(d_raw_kv_storage_, q);
            d_raw_kv_storage_ = nullptr;
        }
        if (d_raw_recurrent_storage_) {
            sycl::free(d_raw_recurrent_storage_, q);
            d_raw_recurrent_storage_ = nullptr;
        }
    }
}

KVCache::KVCache(KVCache&& other) noexcept
    : ctx_(std::move(other.ctx_)),
      config_(other.config_),
      current_seq_len_(other.current_seq_len_),
      k_caches_(std::move(other.k_caches_)),
      v_caches_(std::move(other.v_caches_)),
      k_caches_int8_(std::move(other.k_caches_int8_)),
      v_caches_int8_(std::move(other.v_caches_int8_)),
      k_scales_(std::move(other.k_scales_)),
      v_scales_(std::move(other.v_scales_)),
      k_zero_points_(std::move(other.k_zero_points_)),
      v_zero_points_(std::move(other.v_zero_points_)),
      linear_states_(std::move(other.linear_states_)),
      conv_states_(std::move(other.conv_states_)),
      d_raw_kv_storage_(other.d_raw_kv_storage_),
      d_raw_recurrent_storage_(other.d_raw_recurrent_storage_),
      kv_storage_bytes_(other.kv_storage_bytes_),
      recurrent_storage_bytes_(other.recurrent_storage_bytes_) {
    other.d_raw_kv_storage_ = nullptr;
    other.d_raw_recurrent_storage_ = nullptr;
    other.current_seq_len_ = 0;
}

KVCache& KVCache::operator=(KVCache&& other) noexcept {
    if (this != &other) {
        if (ctx_) {
            auto& q = ctx_->queue();
            if (d_raw_kv_storage_) sycl::free(d_raw_kv_storage_, q);
            if (d_raw_recurrent_storage_) sycl::free(d_raw_recurrent_storage_, q);
        }
        ctx_ = std::move(other.ctx_);
        config_ = other.config_;
        current_seq_len_ = other.current_seq_len_;
        k_caches_ = std::move(other.k_caches_);
        v_caches_ = std::move(other.v_caches_);
        k_caches_int8_ = std::move(other.k_caches_int8_);
        v_caches_int8_ = std::move(other.v_caches_int8_);
        k_scales_ = std::move(other.k_scales_);
        v_scales_ = std::move(other.v_scales_);
        k_zero_points_ = std::move(other.k_zero_points_);
        v_zero_points_ = std::move(other.v_zero_points_);
        linear_states_ = std::move(other.linear_states_);
        conv_states_ = std::move(other.conv_states_);
        d_raw_kv_storage_ = other.d_raw_kv_storage_;
        d_raw_recurrent_storage_ = other.d_raw_recurrent_storage_;
        kv_storage_bytes_ = other.kv_storage_bytes_;
        recurrent_storage_bytes_ = other.recurrent_storage_bytes_;

        other.d_raw_kv_storage_ = nullptr;
        other.d_raw_recurrent_storage_ = nullptr;
        other.current_seq_len_ = 0;
    }
    return *this;
}

bool KVCache::init() {
    static_assert(sizeof(int8_t) == 1, "int8_t must be exactly 1 byte");
    if (config_.dtype == KVCacheDType::INT8) {
        if (config_.artifact_quant_scheme.find("INT8-KV") == std::string::npos) {
            throw std::runtime_error(
                "KVCache::init: Artifact metadata does not explicitly declare quant_scheme = \"INT8-KV\". "
                "Refusing to load INT8 cache path and falling back is forbidden to prevent silent memory exhaustion.");
        }
        int64_t kv_stride = static_cast<int64_t>(config_.num_kv_heads * config_.head_dim);
        if (kv_stride <= 0) {
            throw std::runtime_error("KVCache::init: Invalid kv_stride calculated for INT8 KV cache");
        }
        size_t int8_token_bytes = static_cast<size_t>(kv_stride) * sizeof(int8_t);
        size_t scale_overhead_bytes = config_.num_kv_heads * sizeof(float) * 2; // scale + zero_point per token
        if (int8_token_bytes == 0 || scale_overhead_bytes == 0) {
            throw std::runtime_error("KVCache::init: Incorrect stride or scale overhead for INT8 KV cache");
        }
    }
    return true;
}

bool KVCache::allocate() {
    if (!ctx_) return false;
    init();
    auto& q = ctx_->queue();

    if (config_.dtype == KVCacheDType::INT8) {
        // Allocate full-attention INT8 KV cache + per-layer per-head scale & zero_point tensors
        size_t elements_per_cache = config_.max_seq_len * config_.num_kv_heads * config_.head_dim;
        size_t bytes_per_cache = elements_per_cache * sizeof(int8_t);
        size_t scale_elements = config_.max_seq_len * config_.num_kv_heads;
        size_t scale_bytes = scale_elements * sizeof(float);

        size_t per_layer_kv_bytes = 2 * bytes_per_cache + 4 * scale_bytes;
        kv_storage_bytes_ = config_.num_full_layers * per_layer_kv_bytes;

        d_raw_kv_storage_ = sycl::malloc_device(kv_storage_bytes_, q);
        if (!d_raw_kv_storage_) {
            std::cerr << "[xinfer::KVCache] Failed to allocate " << (kv_storage_bytes_ / (1024 * 1024))
                      << " MB for INT8 full-attention KV cache on GPU.\n";
            return false;
        }

        k_caches_int8_.resize(config_.num_full_layers);
        v_caches_int8_.resize(config_.num_full_layers);
        k_scales_.resize(config_.num_full_layers);
        v_scales_.resize(config_.num_full_layers);
        k_zero_points_.resize(config_.num_full_layers);
        v_zero_points_.resize(config_.num_full_layers);

        uint8_t* kv_ptr = static_cast<uint8_t*>(d_raw_kv_storage_);
        for (size_t l = 0; l < config_.num_full_layers; ++l) {
            k_caches_int8_[l] = reinterpret_cast<int8_t*>(kv_ptr);
            kv_ptr += bytes_per_cache;
            v_caches_int8_[l] = reinterpret_cast<int8_t*>(kv_ptr);
            kv_ptr += bytes_per_cache;
            k_scales_[l] = reinterpret_cast<float*>(kv_ptr);
            kv_ptr += scale_bytes;
            v_scales_[l] = reinterpret_cast<float*>(kv_ptr);
            kv_ptr += scale_bytes;
            k_zero_points_[l] = reinterpret_cast<float*>(kv_ptr);
            kv_ptr += scale_bytes;
            v_zero_points_[l] = reinterpret_cast<float*>(kv_ptr);
            kv_ptr += scale_bytes;
        }
    } else {
        // Allocate full-attention KV cache (FP16 / sycl::half)
        size_t elements_per_cache = config_.max_seq_len * config_.num_kv_heads * config_.head_dim;
        size_t bytes_per_cache = elements_per_cache * sizeof(sycl::half);
        kv_storage_bytes_ = config_.num_full_layers * 2 * bytes_per_cache;

        d_raw_kv_storage_ = sycl::malloc_device(kv_storage_bytes_, q);
        if (!d_raw_kv_storage_) {
            std::cerr << "[xinfer::KVCache] Failed to allocate " << (kv_storage_bytes_ / (1024 * 1024))
                      << " MB for full-attention KV cache on GPU.\n";
            return false;
        }

        k_caches_.resize(config_.num_full_layers);
        v_caches_.resize(config_.num_full_layers);

        uint8_t* kv_ptr = static_cast<uint8_t*>(d_raw_kv_storage_);
        for (size_t l = 0; l < config_.num_full_layers; ++l) {
            k_caches_[l] = reinterpret_cast<sycl::half*>(kv_ptr);
            kv_ptr += bytes_per_cache;
            v_caches_[l] = reinterpret_cast<sycl::half*>(kv_ptr);
            kv_ptr += bytes_per_cache;
        }
    }

    // 2. Allocate linear-attention recurrent states (FP32)
    size_t s_elements_per_layer = config_.linear_num_v_heads * config_.linear_head_k_dim * config_.linear_head_v_dim;
    size_t s_bytes_per_layer = s_elements_per_layer * sizeof(float);

    size_t conv_past_len = (config_.linear_conv_kernel_dim > 1) ? (config_.linear_conv_kernel_dim - 1) : 0;
    size_t conv_elements_per_layer = conv_past_len * config_.linear_conv_channels;
    size_t conv_bytes_per_layer = conv_elements_per_layer * sizeof(float);

    recurrent_storage_bytes_ = config_.num_linear_layers * (s_bytes_per_layer + conv_bytes_per_layer);
    d_raw_recurrent_storage_ = sycl::malloc_device(recurrent_storage_bytes_, q);
    if (!d_raw_recurrent_storage_) {
        std::cerr << "[xinfer::KVCache] Failed to allocate " << (recurrent_storage_bytes_ / (1024 * 1024))
                  << " MB for linear-attention recurrent state on GPU.\n";
        return false;
    }

    linear_states_.resize(config_.num_linear_layers);
    conv_states_.resize(config_.num_linear_layers);

    uint8_t* rec_ptr = static_cast<uint8_t*>(d_raw_recurrent_storage_);
    for (size_t l = 0; l < config_.num_linear_layers; ++l) {
        linear_states_[l] = reinterpret_cast<float*>(rec_ptr);
        rec_ptr += s_bytes_per_layer;
        conv_states_[l] = reinterpret_cast<float*>(rec_ptr);
        rec_ptr += conv_bytes_per_layer;
    }

    clear();
    return true;
}

void KVCache::clear() {
    if (!ctx_) return;
    auto& q = ctx_->queue();
    if (d_raw_recurrent_storage_ && recurrent_storage_bytes_ > 0) {
        q.memset(d_raw_recurrent_storage_, 0, recurrent_storage_bytes_);
    }
    current_seq_len_ = 0;
    q.wait();
}

sycl::half* KVCache::k_cache(size_t full_layer_idx) {
    if (config_.dtype != KVCacheDType::FP16) {
        throw std::logic_error("KVCache::k_cache: Cannot access FP16 k_cache when cache is INT8");
    }
    if (full_layer_idx >= k_caches_.size()) {
        throw std::out_of_range("KVCache::k_cache: full_layer_idx (" +
                                std::to_string(full_layer_idx) + ") >= num_full_layers (" +
                                std::to_string(k_caches_.size()) + ")");
    }
    assert(k_caches_[full_layer_idx] != nullptr);
    return k_caches_[full_layer_idx];
}

const sycl::half* KVCache::k_cache(size_t full_layer_idx) const {
    if (config_.dtype != KVCacheDType::FP16) {
        throw std::logic_error("KVCache::k_cache: Cannot access FP16 k_cache when cache is INT8");
    }
    if (full_layer_idx >= k_caches_.size()) {
        throw std::out_of_range("KVCache::k_cache: full_layer_idx (" +
                                std::to_string(full_layer_idx) + ") >= num_full_layers (" +
                                std::to_string(k_caches_.size()) + ")");
    }
    assert(k_caches_[full_layer_idx] != nullptr);
    return k_caches_[full_layer_idx];
}

sycl::half* KVCache::v_cache(size_t full_layer_idx) {
    if (config_.dtype != KVCacheDType::FP16) {
        throw std::logic_error("KVCache::v_cache: Cannot access FP16 v_cache when cache is INT8");
    }
    if (full_layer_idx >= v_caches_.size()) {
        throw std::out_of_range("KVCache::v_cache: full_layer_idx (" +
                                std::to_string(full_layer_idx) + ") >= num_full_layers (" +
                                std::to_string(v_caches_.size()) + ")");
    }
    assert(v_caches_[full_layer_idx] != nullptr);
    return v_caches_[full_layer_idx];
}

const sycl::half* KVCache::v_cache(size_t full_layer_idx) const {
    if (config_.dtype != KVCacheDType::FP16) {
        throw std::logic_error("KVCache::v_cache: Cannot access FP16 v_cache when cache is INT8");
    }
    if (full_layer_idx >= v_caches_.size()) {
        throw std::out_of_range("KVCache::v_cache: full_layer_idx (" +
                                std::to_string(full_layer_idx) + ") >= num_full_layers (" +
                                std::to_string(v_caches_.size()) + ")");
    }
    assert(v_caches_[full_layer_idx] != nullptr);
    return v_caches_[full_layer_idx];
}

int8_t* KVCache::k_cache_int8(size_t full_layer_idx) {
    if (config_.dtype != KVCacheDType::INT8) {
        throw std::logic_error("KVCache::k_cache_int8: Cannot access INT8 k_cache when cache is FP16");
    }
    if (full_layer_idx >= k_caches_int8_.size()) {
        throw std::out_of_range("KVCache::k_cache_int8: full_layer_idx (" +
                                std::to_string(full_layer_idx) + ") >= num_full_layers (" +
                                std::to_string(k_caches_int8_.size()) + ")");
    }
    assert(k_caches_int8_[full_layer_idx] != nullptr);
    return k_caches_int8_[full_layer_idx];
}

const int8_t* KVCache::k_cache_int8(size_t full_layer_idx) const {
    if (config_.dtype != KVCacheDType::INT8) {
        throw std::logic_error("KVCache::k_cache_int8: Cannot access INT8 k_cache when cache is FP16");
    }
    if (full_layer_idx >= k_caches_int8_.size()) {
        throw std::out_of_range("KVCache::k_cache_int8: full_layer_idx (" +
                                std::to_string(full_layer_idx) + ") >= num_full_layers (" +
                                std::to_string(k_caches_int8_.size()) + ")");
    }
    assert(k_caches_int8_[full_layer_idx] != nullptr);
    return k_caches_int8_[full_layer_idx];
}

int8_t* KVCache::v_cache_int8(size_t full_layer_idx) {
    if (config_.dtype != KVCacheDType::INT8) {
        throw std::logic_error("KVCache::v_cache_int8: Cannot access INT8 v_cache when cache is FP16");
    }
    if (full_layer_idx >= v_caches_int8_.size()) {
        throw std::out_of_range("KVCache::v_cache_int8: full_layer_idx (" +
                                std::to_string(full_layer_idx) + ") >= num_full_layers (" +
                                std::to_string(v_caches_int8_.size()) + ")");
    }
    assert(v_caches_int8_[full_layer_idx] != nullptr);
    return v_caches_int8_[full_layer_idx];
}

const int8_t* KVCache::v_cache_int8(size_t full_layer_idx) const {
    if (config_.dtype != KVCacheDType::INT8) {
        throw std::logic_error("KVCache::v_cache_int8: Cannot access INT8 v_cache when cache is FP16");
    }
    if (full_layer_idx >= v_caches_int8_.size()) {
        throw std::out_of_range("KVCache::v_cache_int8: full_layer_idx (" +
                                std::to_string(full_layer_idx) + ") >= num_full_layers (" +
                                std::to_string(v_caches_int8_.size()) + ")");
    }
    assert(v_caches_int8_[full_layer_idx] != nullptr);
    return v_caches_int8_[full_layer_idx];
}

float* KVCache::k_scale(size_t full_layer_idx) {
    if (config_.dtype != KVCacheDType::INT8) {
        throw std::logic_error("KVCache::k_scale: Cannot access k_scale when cache is FP16");
    }
    if (full_layer_idx >= k_scales_.size()) {
        throw std::out_of_range("KVCache::k_scale: full_layer_idx >= num_full_layers");
    }
    assert(k_scales_[full_layer_idx] != nullptr);
    return k_scales_[full_layer_idx];
}

const float* KVCache::k_scale(size_t full_layer_idx) const {
    if (config_.dtype != KVCacheDType::INT8) {
        throw std::logic_error("KVCache::k_scale: Cannot access k_scale when cache is FP16");
    }
    if (full_layer_idx >= k_scales_.size()) {
        throw std::out_of_range("KVCache::k_scale: full_layer_idx >= num_full_layers");
    }
    assert(k_scales_[full_layer_idx] != nullptr);
    return k_scales_[full_layer_idx];
}

float* KVCache::v_scale(size_t full_layer_idx) {
    if (config_.dtype != KVCacheDType::INT8) {
        throw std::logic_error("KVCache::v_scale: Cannot access v_scale when cache is FP16");
    }
    if (full_layer_idx >= v_scales_.size()) {
        throw std::out_of_range("KVCache::v_scale: full_layer_idx >= num_full_layers");
    }
    assert(v_scales_[full_layer_idx] != nullptr);
    return v_scales_[full_layer_idx];
}

const float* KVCache::v_scale(size_t full_layer_idx) const {
    if (config_.dtype != KVCacheDType::INT8) {
        throw std::logic_error("KVCache::v_scale: Cannot access v_scale when cache is FP16");
    }
    if (full_layer_idx >= v_scales_.size()) {
        throw std::out_of_range("KVCache::v_scale: full_layer_idx >= num_full_layers");
    }
    assert(v_scales_[full_layer_idx] != nullptr);
    return v_scales_[full_layer_idx];
}

float* KVCache::k_zero_point(size_t full_layer_idx) {
    if (config_.dtype != KVCacheDType::INT8) {
        throw std::logic_error("KVCache::k_zero_point: Cannot access k_zero_point when cache is FP16");
    }
    if (full_layer_idx >= k_zero_points_.size()) {
        throw std::out_of_range("KVCache::k_zero_point: full_layer_idx >= num_full_layers");
    }
    assert(k_zero_points_[full_layer_idx] != nullptr);
    return k_zero_points_[full_layer_idx];
}

const float* KVCache::k_zero_point(size_t full_layer_idx) const {
    if (config_.dtype != KVCacheDType::INT8) {
        throw std::logic_error("KVCache::k_zero_point: Cannot access k_zero_point when cache is FP16");
    }
    if (full_layer_idx >= k_zero_points_.size()) {
        throw std::out_of_range("KVCache::k_zero_point: full_layer_idx >= num_full_layers");
    }
    assert(k_zero_points_[full_layer_idx] != nullptr);
    return k_zero_points_[full_layer_idx];
}

float* KVCache::v_zero_point(size_t full_layer_idx) {
    if (config_.dtype != KVCacheDType::INT8) {
        throw std::logic_error("KVCache::v_zero_point: Cannot access v_zero_point when cache is FP16");
    }
    if (full_layer_idx >= v_zero_points_.size()) {
        throw std::out_of_range("KVCache::v_zero_point: full_layer_idx >= num_full_layers");
    }
    assert(v_zero_points_[full_layer_idx] != nullptr);
    return v_zero_points_[full_layer_idx];
}

const float* KVCache::v_zero_point(size_t full_layer_idx) const {
    if (config_.dtype != KVCacheDType::INT8) {
        throw std::logic_error("KVCache::v_zero_point: Cannot access v_zero_point when cache is FP16");
    }
    if (full_layer_idx >= v_zero_points_.size()) {
        throw std::out_of_range("KVCache::v_zero_point: full_layer_idx >= num_full_layers");
    }
    assert(v_zero_points_[full_layer_idx] != nullptr);
    return v_zero_points_[full_layer_idx];
}

float* KVCache::linear_state(size_t linear_layer_idx) {
    if (linear_layer_idx >= linear_states_.size()) {
        throw std::out_of_range("KVCache::linear_state: linear_layer_idx (" +
                                std::to_string(linear_layer_idx) + ") >= num_linear_layers (" +
                                std::to_string(linear_states_.size()) + ")");
    }
    assert(linear_states_[linear_layer_idx] != nullptr);
    return linear_states_[linear_layer_idx];
}

const float* KVCache::linear_state(size_t linear_layer_idx) const {
    if (linear_layer_idx >= linear_states_.size()) {
        throw std::out_of_range("KVCache::linear_state: linear_layer_idx (" +
                                std::to_string(linear_layer_idx) + ") >= num_linear_layers (" +
                                std::to_string(linear_states_.size()) + ")");
    }
    assert(linear_states_[linear_layer_idx] != nullptr);
    return linear_states_[linear_layer_idx];
}

float* KVCache::conv_state(size_t linear_layer_idx) {
    if (linear_layer_idx >= conv_states_.size()) {
        throw std::out_of_range("KVCache::conv_state: linear_layer_idx (" +
                                std::to_string(linear_layer_idx) + ") >= num_linear_layers (" +
                                std::to_string(conv_states_.size()) + ")");
    }
    assert(conv_states_[linear_layer_idx] != nullptr);
    return conv_states_[linear_layer_idx];
}

const float* KVCache::conv_state(size_t linear_layer_idx) const {
    if (linear_layer_idx >= conv_states_.size()) {
        throw std::out_of_range("KVCache::conv_state: linear_layer_idx (" +
                                std::to_string(linear_layer_idx) + ") >= num_linear_layers (" +
                                std::to_string(conv_states_.size()) + ")");
    }
    assert(conv_states_[linear_layer_idx] != nullptr);
    return conv_states_[linear_layer_idx];
}

size_t KVCache::total_allocated_bytes() const noexcept {
    return kv_storage_bytes_ + recurrent_storage_bytes_;
}

} // namespace xinfer::core
