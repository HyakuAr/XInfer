// Citing vendor documentation per AGENTS.md §5:
// - docs/vendor/xe-gpu-architecture.md (lines 48-50):
//   Vector Engine (VE) ALUs support native FP16 and FP32 operations.
// - docs/vendor/thread-mapping-occupancy.md (lines 9-15):
//   Sub-group size 16 maps to one Vector Engine hardware thread. sycl::reqd_sub_group_size(16).

#include "rmsnorm.h"
#include <cmath>

namespace xinfer::ops {

namespace {

template <typename T>
sycl::event rmsnorm_impl(sycl::queue& q,
                         T* out,
                         const T* in,
                         const float* weight,
                         int64_t num_tokens,
                         int64_t hidden_size,
                         float eps) {
    if (num_tokens <= 0 || hidden_size <= 0) return sycl::event{};

    constexpr size_t SG_SIZE = 16;
    constexpr size_t WG_SIZE = 256;
    constexpr size_t NUM_SG = WG_SIZE / SG_SIZE;

    size_t total_wgs = static_cast<size_t>(num_tokens);

    return q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> slm_sums(sycl::range<1>(NUM_SG), cgh);
        sycl::local_accessor<float, 1> slm_rsqrt(sycl::range<1>(1), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(total_wgs * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                size_t t = item.get_group(0);
                size_t tid = item.get_local_linear_id();
                sycl::sub_group sg = item.get_sub_group();
                size_t sg_id = sg.get_group_linear_id();

                const T* x = in + t * hidden_size;
                T* y = out + t * hidden_size;

                // 1. Cooperative strided sum of squares (FP32 accumulation)
                float local_sum_sq = 0.0f;
                for (int64_t i = static_cast<int64_t>(tid); i < hidden_size; i += static_cast<int64_t>(WG_SIZE)) {
                    float val = static_cast<float>(x[i]);
                    local_sum_sq += val * val;
                }

                // 2. Intra-subgroup reduction
                float sg_sum = sycl::reduce_over_group(sg, local_sum_sq, sycl::plus<float>());
                if (sg.get_local_linear_id() == 0) {
                    slm_sums[sg_id] = sg_sum;
                }
                item.barrier(sycl::access::fence_space::local_space);

                // 3. Inter-subgroup reduction by sub-group 0
                if (sg_id == 0) {
                    float total_sum = 0.0f;
                    if (sg.get_local_linear_id() < NUM_SG) {
                        total_sum = slm_sums[sg.get_local_linear_id()];
                    }
                    float full_sum = sycl::reduce_over_group(sg, total_sum, sycl::plus<float>());
                    if (sg.get_local_linear_id() == 0) {
                        float mean_sq = full_sum / static_cast<float>(hidden_size);
                        slm_rsqrt[0] = 1.0f / sycl::sqrt(mean_sq + eps);
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);

                float rsqrt_val = slm_rsqrt[0];

                // 4. Normalized store
                for (int64_t i = static_cast<int64_t>(tid); i < hidden_size; i += static_cast<int64_t>(WG_SIZE)) {
                    float val = static_cast<float>(x[i]);
                    y[i] = static_cast<T>(val * rsqrt_val * weight[i]);
                }
            });
    });
}

template <typename T>
sycl::event rmsnorm_residual_impl(sycl::queue& q,
                                  T* out,
                                  T* residual,
                                  const T* in,
                                  const float* weight,
                                  int64_t num_tokens,
                                  int64_t hidden_size,
                                  float eps) {
    if (num_tokens <= 0 || hidden_size <= 0) return sycl::event{};

    constexpr size_t SG_SIZE = 16;
    constexpr size_t WG_SIZE = 256;
    constexpr size_t NUM_SG = WG_SIZE / SG_SIZE;

    size_t total_wgs = static_cast<size_t>(num_tokens);

    return q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> slm_sums(sycl::range<1>(NUM_SG), cgh);
        sycl::local_accessor<float, 1> slm_rsqrt(sycl::range<1>(1), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(total_wgs * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                size_t t = item.get_group(0);
                size_t tid = item.get_local_linear_id();
                sycl::sub_group sg = item.get_sub_group();
                size_t sg_id = sg.get_group_linear_id();

                const T* x = in + t * hidden_size;
                T* res = residual + t * hidden_size;
                T* y = out + t * hidden_size;

                float local_sum_sq = 0.0f;
                for (int64_t i = static_cast<int64_t>(tid); i < hidden_size; i += static_cast<int64_t>(WG_SIZE)) {
                    float r = static_cast<float>(res[i]) + static_cast<float>(x[i]);
                    res[i] = static_cast<T>(r);
                    local_sum_sq += r * r;
                }

                float sg_sum = sycl::reduce_over_group(sg, local_sum_sq, sycl::plus<float>());
                if (sg.get_local_linear_id() == 0) {
                    slm_sums[sg_id] = sg_sum;
                }
                item.barrier(sycl::access::fence_space::local_space);

                if (sg_id == 0) {
                    float total_sum = 0.0f;
                    if (sg.get_local_linear_id() < NUM_SG) {
                        total_sum = slm_sums[sg.get_local_linear_id()];
                    }
                    float full_sum = sycl::reduce_over_group(sg, total_sum, sycl::plus<float>());
                    if (sg.get_local_linear_id() == 0) {
                        float mean_sq = full_sum / static_cast<float>(hidden_size);
                        slm_rsqrt[0] = 1.0f / sycl::sqrt(mean_sq + eps);
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);

                float rsqrt_val = slm_rsqrt[0];

                for (int64_t i = static_cast<int64_t>(tid); i < hidden_size; i += static_cast<int64_t>(WG_SIZE)) {
                    float r = static_cast<float>(res[i]);
                    y[i] = static_cast<T>(r * rsqrt_val * weight[i]);
                }
            });
    });
}

} // anonymous namespace

sycl::event rmsnorm(sycl::queue& q,
                    float* out,
                    const float* in,
                    const float* weight,
                    int64_t num_tokens,
                    int64_t hidden_size,
                    float eps) {
    return rmsnorm_impl<float>(q, out, in, weight, num_tokens, hidden_size, eps);
}

sycl::event rmsnorm(sycl::queue& q,
                    sycl::half* out,
                    const sycl::half* in,
                    const float* weight,
                    int64_t num_tokens,
                    int64_t hidden_size,
                    float eps) {
    return rmsnorm_impl<sycl::half>(q, out, in, weight, num_tokens, hidden_size, eps);
}

sycl::event rmsnorm_residual(sycl::queue& q,
                             float* out,
                             float* residual,
                             const float* in,
                             const float* weight,
                             int64_t num_tokens,
                             int64_t hidden_size,
                             float eps) {
    return rmsnorm_residual_impl<float>(q, out, residual, in, weight, num_tokens, hidden_size, eps);
}

sycl::event rmsnorm_residual(sycl::queue& q,
                             sycl::half* out,
                             sycl::half* residual,
                             const sycl::half* in,
                             const float* weight,
                             int64_t num_tokens,
                             int64_t hidden_size,
                             float eps) {
    return rmsnorm_residual_impl<sycl::half>(q, out, residual, in, weight, num_tokens, hidden_size, eps);
}

} // namespace xinfer::ops
