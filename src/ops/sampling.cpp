#include "sampling.h"
#include <mutex>
#include <limits>
#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <string>

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

SpeculativeAcceptanceResult speculative_accept_reject(
    sycl::queue& q,
    const int64_t* draft_tokens,
    size_t num_draft_tokens,
    const float* target_logits,
    int64_t vocab_size,
    const float* draft_probs,
    const float* rand_uniform) {
    if (num_draft_tokens == 0 || !draft_tokens || !target_logits || vocab_size <= 0) {
        return SpeculativeAcceptanceResult{};
    }

    SpeculativeAcceptanceResult result;
    result.accepted_tokens.reserve(num_draft_tokens);

    bool rejection_occurred = false;

    // Buffer to host-copy logits for evaluating softmax probability P_target(x_i) when stochastic
    std::vector<float> host_logits;
    if (draft_probs != nullptr) {
        host_logits.resize(vocab_size);
    }

    for (size_t i = 0; i < num_draft_tokens; ++i) {
        int64_t draft_tok = draft_tokens[i];
        const float* cur_logits = target_logits + i * vocab_size;

        int64_t target_argmax_tok = argmax(q, cur_logits, vocab_size);

        bool accepted = false;

        if (!draft_probs) {
            // Greedy argmax acceptance
            accepted = (draft_tok == target_argmax_tok);
        } else {
            float p_draft = draft_probs[i];
            float r = rand_uniform ? rand_uniform[i] : 0.0f;

            if (p_draft <= 0.0f) {
                accepted = (draft_tok == target_argmax_tok);
            } else if (draft_tok == target_argmax_tok && r <= 0.0f) {
                accepted = true;
            } else {
                q.memcpy(host_logits.data(), cur_logits, vocab_size * sizeof(float)).wait();

                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t v = 0; v < vocab_size; ++v) {
                    if (host_logits[v] > max_val) max_val = host_logits[v];
                }

                float sum_exp = 0.0f;
                for (int64_t v = 0; v < vocab_size; ++v) {
                    sum_exp += std::exp(host_logits[v] - max_val);
                }

                float p_target = 0.0f;
                if (draft_tok >= 0 && draft_tok < vocab_size && sum_exp > 0.0f) {
                    p_target = std::exp(host_logits[draft_tok] - max_val) / sum_exp;
                }

                float ratio = p_target / p_draft;
                accepted = (r < ratio);
            }
        }

        if (accepted) {
            result.accepted_tokens.push_back(draft_tok);
            result.num_accepted++;
        } else {
            rejection_occurred = true;
            result.has_rejected_token = true;
            result.rejected_token = draft_tok;
            result.num_rejected = 1;
            result.bonus_or_resampled_token = target_argmax_tok;
            result.num_resampled = 1;
            result.num_remaining = num_draft_tokens - (result.num_accepted + 1);
            break;
        }
    }

    // Fail-loud contract assertion:
    // Assert that the sum of accepted tokens + 1 rejected token + remaining resampled tokens equals exactly N.
    // If the math doesn't balance, throw an exception rather than returning a truncated sequence.
    if (rejection_occurred) {
        size_t accounted = result.num_accepted + result.num_rejected + result.num_remaining;
        if (accounted != num_draft_tokens) {
            throw std::logic_error(
                "Speculative decoding fail-loud contract violated: accepted (" +
                std::to_string(result.num_accepted) + ") + 1 (rejected) + remaining (" +
                std::to_string(result.num_remaining) + ") = " +
                std::to_string(accounted) + " != N (" +
                std::to_string(num_draft_tokens) + ")");
        }
    } else {
        if (result.num_accepted != num_draft_tokens) {
            throw std::logic_error(
                "Speculative decoding fail-loud contract violated: all accepted but num_accepted (" +
                std::to_string(result.num_accepted) + ") != N (" +
                std::to_string(num_draft_tokens) + ")");
        }
        // Bonus token from last position
        result.bonus_or_resampled_token = argmax(q, target_logits + (num_draft_tokens - 1) * vocab_size, vocab_size);
    }

    return result;
}

} // namespace xinfer::ops
