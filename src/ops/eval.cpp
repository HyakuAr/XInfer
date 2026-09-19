// Citing vendor documentation per AGENTS.md §5:
// - docs/vendor/xe-gpu-architecture.md (lines 48-50):
//   Vector Engine (VE) ALUs support native FP16 and FP32 operations.
// - docs/vendor/xe-gpu-architecture.md (lines 55-59):
//   Sub-group sizes of 16 and 32 are supported on Xe2-HPG (Battlemage).
//   Sub-group shuffle operations enable efficient intra-sub-group reductions.

#include "eval.h"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <string>

namespace xinfer::ops {

namespace {

// Workgroup size for the per-position log_softmax + NLL kernel.
// Each workgroup handles one sequence position's vocab_size-wide row.
// The vocab dimension (248,320) is too large for SLM, so we tile it
// in a loop within each work-item, using sub-group reductions for
// the max and log-sum-exp passes.
constexpr size_t VOCAB_WG_SIZE = 256;

} // anonymous namespace

CrossEntropyResult cross_entropy_loss(
    sycl::queue& q,
    const float* logits,
    const int64_t* token_ids,
    int64_t seq_len,
    int64_t vocab_size,
    bool collect_per_token) {

    CrossEntropyResult result;

    if (seq_len <= 1 || vocab_size <= 0 || !logits || !token_ids) {
        return result;
    }

    // Number of causal positions: for each position i, logits[i] predicts token_ids[i+1]
    int64_t num_positions = seq_len - 1;
    result.num_tokens = static_cast<size_t>(num_positions);

    // ---- Allocate output buffers in USM shared memory ----
    // per_position_loss[i] = -log_softmax(logits[i])[token_ids[i+1]]
    float* d_per_pos_loss = sycl::malloc_shared<float>(num_positions, q);
    if (!d_per_pos_loss) {
        throw std::runtime_error(
            "cross_entropy_loss: Failed to allocate per-position loss buffer ("
            + std::to_string(num_positions * sizeof(float)) + " bytes)");
    }

    // Optional: per-position rank of the ground-truth token
    int64_t* d_per_pos_rank = nullptr;
    if (collect_per_token) {
        d_per_pos_rank = sycl::malloc_shared<int64_t>(num_positions, q);
        if (!d_per_pos_rank) {
            sycl::free(d_per_pos_loss, q);
            throw std::runtime_error(
                "cross_entropy_loss: Failed to allocate per-position rank buffer");
        }
    }

    // ---- Kernel: fused log_softmax + NLLLoss ----
    // Launch one workgroup per causal position.
    // Each workgroup of VOCAB_WG_SIZE work-items cooperatively processes
    // the vocab_size-wide logit row in a tiled loop.
    //
    // Algorithm per workgroup (position i):
    //   1. Tiled max reduction over logits[i, 0..vocab_size)
    //   2. Tiled log-sum-exp: sum(exp(logits[i,v] - max)) for all v
    //   3. Extract target_logit = logits[i, token_ids[i+1]]
    //   4. loss_i = -(target_logit - max) + log(sum_exp)
    //   5. Optional: count rank = #{v : logits[i,v] > target_logit}

    q.submit([&](sycl::handler& cgh) {
        // SLM for workgroup-level reductions
        sycl::local_accessor<float, 1> slm_max(sycl::range<1>(VOCAB_WG_SIZE), cgh);
        sycl::local_accessor<float, 1> slm_sum(sycl::range<1>(VOCAB_WG_SIZE), cgh);
        sycl::local_accessor<int64_t, 1> slm_rank(sycl::range<1>(VOCAB_WG_SIZE), cgh);

        int64_t v_size = vocab_size;
        int64_t n_pos = num_positions;
        bool do_rank = collect_per_token;

        cgh.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(static_cast<size_t>(num_positions) * VOCAB_WG_SIZE),
                sycl::range<1>(VOCAB_WG_SIZE)),
            [=](sycl::nd_item<1> item) {
                size_t local_id = item.get_local_id(0);
                size_t wg_id = item.get_group(0);
                int64_t pos = static_cast<int64_t>(wg_id);

                if (pos >= n_pos) return;

                const float* row = logits + pos * v_size;
                int64_t target_id = token_ids[pos + 1];

                // Clamp target_id to valid range (fail-safe; the harness
                // should have validated this already)
                if (target_id < 0 || target_id >= v_size) {
                    if (local_id == 0) {
                        d_per_pos_loss[pos] = 0.0f;
                        if (do_rank && d_per_pos_rank) {
                            d_per_pos_rank[pos] = -1;
                        }
                    }
                    return;
                }

                // ---- Pass 1: Row-max for numerical stability ----
                float thread_max = -std::numeric_limits<float>::infinity();
                for (int64_t v = static_cast<int64_t>(local_id); v < v_size;
                     v += static_cast<int64_t>(VOCAB_WG_SIZE)) {
                    float val = row[v];
                    if (val > thread_max) thread_max = val;
                }

                // Store to SLM and tree-reduce
                slm_max[local_id] = thread_max;
                item.barrier(sycl::access::fence_space::local_space);

                for (size_t stride = VOCAB_WG_SIZE / 2; stride > 0; stride /= 2) {
                    if (local_id < stride) {
                        float other = slm_max[local_id + stride];
                        if (other > slm_max[local_id]) {
                            slm_max[local_id] = other;
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                float row_max = slm_max[0];
                item.barrier(sycl::access::fence_space::local_space);

                // ---- Pass 2: Sum of exp(x - max) ----
                float thread_sum = 0.0f;
                for (int64_t v = static_cast<int64_t>(local_id); v < v_size;
                     v += static_cast<int64_t>(VOCAB_WG_SIZE)) {
                    thread_sum += sycl::exp(row[v] - row_max);
                }

                slm_sum[local_id] = thread_sum;
                item.barrier(sycl::access::fence_space::local_space);

                for (size_t stride = VOCAB_WG_SIZE / 2; stride > 0; stride /= 2) {
                    if (local_id < stride) {
                        slm_sum[local_id] += slm_sum[local_id + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                float log_sum_exp = sycl::log(slm_sum[0] > 0.0f ? slm_sum[0] : 1e-12f);

                // ---- Compute NLL loss for this position ----
                // log_softmax(logits[pos, target]) = (logits[pos, target] - max) - log_sum_exp
                // NLL = -log_softmax(logits[pos, target])
                if (local_id == 0) {
                    float target_logit = row[target_id];
                    float log_prob = (target_logit - row_max) - log_sum_exp;
                    d_per_pos_loss[pos] = -log_prob;
                }

                // ---- Optional: compute rank of ground-truth token ----
                if (do_rank && d_per_pos_rank) {
                    float target_logit = row[target_id];
                    int64_t thread_rank = 0;
                    for (int64_t v = static_cast<int64_t>(local_id); v < v_size;
                         v += static_cast<int64_t>(VOCAB_WG_SIZE)) {
                        if (row[v] > target_logit) {
                            thread_rank++;
                        }
                    }

                    slm_rank[local_id] = thread_rank;
                    item.barrier(sycl::access::fence_space::local_space);

                    // Reduce ranks
                    for (size_t stride = VOCAB_WG_SIZE / 2; stride > 0; stride /= 2) {
                        if (local_id < stride) {
                            slm_rank[local_id] += slm_rank[local_id + stride];
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    if (local_id == 0) {
                        d_per_pos_rank[pos] = slm_rank[0];
                    }
                }
            });
    }).wait();

    // ---- Host-side reduction: sum per-position losses ----
    double total_loss = 0.0;
    if (collect_per_token) {
        result.per_token_loss.resize(num_positions);
        result.per_token_rank.resize(num_positions);
    }

    for (int64_t i = 0; i < num_positions; ++i) {
        float loss_i = d_per_pos_loss[i];
        total_loss += static_cast<double>(loss_i);

        if (collect_per_token) {
            result.per_token_loss[i] = loss_i;
            result.per_token_rank[i] = d_per_pos_rank ? d_per_pos_rank[i] : -1;
        }
    }

    result.total_loss = total_loss;
    result.perplexity = std::exp(total_loss / static_cast<double>(num_positions));

    // ---- Cleanup ----
    sycl::free(d_per_pos_loss, q);
    if (d_per_pos_rank) {
        sycl::free(d_per_pos_rank, q);
    }

    return result;
}

} // namespace xinfer::ops
