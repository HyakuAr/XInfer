#include "sampling.h"
#include <limits>
#include <algorithm>

namespace xinfer::ops {

int64_t argmax(sycl::queue& q, const float* logits, int64_t vocab_size) {
    if (vocab_size <= 0 || !logits) return 0;
    if (vocab_size == 1) return 0;

    constexpr size_t WG_SIZE = 256;
    size_t num_wgs = (static_cast<size_t>(vocab_size) + WG_SIZE - 1) / WG_SIZE;

    // Allocate USM shared memory for inter-workgroup partial reduction
    float* partial_max = sycl::malloc_shared<float>(num_wgs, q);
    int64_t* partial_idx = sycl::malloc_shared<int64_t>(num_wgs, q);

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
                partial_max[wg_id] = local_max[0];
                partial_idx[wg_id] = local_idx[0];
            }
        });
    }).wait();

    // Stage 2: Reduce partial workgroup results on host
    float best_val = partial_max[0];
    int64_t best_idx = partial_idx[0];

    for (size_t i = 1; i < num_wgs; ++i) {
        if (partial_max[i] > best_val) {
            best_val = partial_max[i];
            best_idx = partial_idx[i];
        }
    }

    sycl::free(partial_max, q);
    sycl::free(partial_idx, q);

    return best_idx;
}

} // namespace xinfer::ops
