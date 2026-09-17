#include "sampling.h"
#include <mutex>
#include <limits>
#include <algorithm>

namespace xinfer::ops {

int64_t argmax(sycl::queue& q,
               const float* logits,
               int64_t vocab_size,
               float* partial_max,
               int64_t* partial_idx) {
    if (vocab_size <= 0 || !logits) return 0;
    if (vocab_size == 1) return 0;

    constexpr size_t WG_SIZE = 256;
    size_t num_wgs = (static_cast<size_t>(vocab_size) + WG_SIZE - 1) / WG_SIZE;

    // Caller-owned scratch buffer; fallback to ephemeral USM shared allocation if not provided
    bool allocated_scratch = false;
    float* p_max = partial_max;
    int64_t* p_idx = partial_idx;

    if (!p_max || !p_idx) {
        p_max = sycl::malloc_shared<float>(num_wgs, q);
        p_idx = sycl::malloc_shared<int64_t>(num_wgs, q);
        allocated_scratch = true;
    }

    // Stage 1: Workgroup local reduction
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> local_max(sycl::range<1>(WG_SIZE), cgh);
        sycl::local_accessor<int64_t, 1> local_idx(sycl::range<1>(WG_SIZE), cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(num_wgs * WG_SIZE), sycl::range<1>(WG_SIZE)),
                         [=](sycl::nd_item<1> item) {
            size_t global_i = item.get_global_id(0);
            size_t local_i = item.get_local_id(0);
            size_t wg_id = item.get_group(0);

            float val = -std::numeric_limits<float>::infinity();
            int64_t idx = -1;

            if (global_i < static_cast<size_t>(vocab_size)) {
                val = logits[global_i];
                idx = static_cast<int64_t>(global_i);
            }

            local_max[local_i] = val;
            local_idx[local_i] = idx;
            item.barrier(sycl::access::fence_space::local_space);

            // Workgroup tree reduction in SLM
            for (size_t stride = WG_SIZE / 2; stride > 0; stride /= 2) {
                if (local_i < stride) {
                    if (local_max[local_i + stride] > local_max[local_i]) {
                        local_max[local_i] = local_max[local_i + stride];
                        local_idx[local_i] = local_idx[local_i + stride];
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
            }

            if (local_i == 0) {
                p_max[wg_id] = local_max[0];
                p_idx[wg_id] = local_idx[0];
            }
        });
    }).wait();

    // Stage 2: Reduce partial workgroup results on host
    float best_val = p_max[0];
    int64_t best_idx = p_idx[0];

    for (size_t i = 1; i < num_wgs; ++i) {
        if (p_max[i] > best_val) {
            best_val = p_max[i];
            best_idx = p_idx[i];
        }
    }

    if (allocated_scratch) {
        sycl::free(p_max, q);
        sycl::free(p_idx, q);
    }

    return best_idx;
}

} // namespace xinfer::ops
