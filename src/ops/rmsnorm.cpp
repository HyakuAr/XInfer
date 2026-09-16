#include "rmsnorm.h"
#include <cmath>

namespace xinfer::ops {

sycl::event rmsnorm(sycl::queue& q,
                    float* out,
                    const float* in,
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

                const float* x = in + t * hidden_size;
                float* y = out + t * hidden_size;

                // 1. Cooperative strided sum of squares
                float local_sum_sq = 0.0f;
                for (int64_t i = static_cast<int64_t>(tid); i < hidden_size; i += static_cast<int64_t>(WG_SIZE)) {
                    float val = x[i];
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
                    y[i] = x[i] * rsqrt_val * weight[i];
                }
            });
    });
}

sycl::event rmsnorm_residual(sycl::queue& q,
                             float* out,
                             float* residual,
                             const float* in,
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

                const float* x = in + t * hidden_size;
                float* res = residual + t * hidden_size;
                float* y = out + t * hidden_size;

                float local_sum_sq = 0.0f;
                for (int64_t i = static_cast<int64_t>(tid); i < hidden_size; i += static_cast<int64_t>(WG_SIZE)) {
                    res[i] += x[i];
                    float val = res[i];
                    local_sum_sq += val * val;
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
                    y[i] = res[i] * rsqrt_val * weight[i];
                }
            });
    });
}

} // namespace xinfer::ops
