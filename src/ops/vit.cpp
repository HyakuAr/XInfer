// Vendor note: Intel Xe GPU Architecture & Thread Mapping
// Citing docs/vendor/xe-gpu-architecture.md:
//   - Target GPU: Intel Arc Pro B60 (Battlemage / Xe2-HPG, device ID 0xE211)
//   - 20 Xe-Cores, 8 Vector Engines per Xe-Core, 64 hardware threads per Xe-Core
//   - Supported sub-group sizes: 16, 32; Max work-group size: 1024; SLM per Xe-Core: 128 KB
// Citing docs/vendor/thread-mapping-occupancy.md:
//   - "all work-items in a work-group must fit on one Xe-Core, and the global range must divide evenly
//      by the chosen work-group size in each dimension"
//   - Chosen workgroup sizes (e.g. 256 or (1, 256)) evenly divide (num_patches=256, embed_dim=1024,
//      intermediate_dim=4096, llm_dim=5120)

#include "vit.h"
#include <cmath>
#include <algorithm>

namespace xinfer::ops {

sycl::event vit_patch_embed(
    sycl::queue& q,
    sycl::half* d_out_patches,
    const sycl::half* d_image_pixels,
    const sycl::half* d_patch_weight,
    const sycl::half* d_patch_bias,
    const sycl::half* d_pos_embed,
    int64_t num_patches,
    int64_t patch_dim,
    int64_t embed_dim
) {
    if (!d_out_patches || !d_image_pixels || !d_patch_weight || num_patches <= 0 || patch_dim <= 0 || embed_dim <= 0) {
        return sycl::event{};
    }

    // Grid: (num_patches, embed_dim)
    // Work-group size: 256 work-items along embed_dim dimension (embed_dim is a multiple of 256)
    size_t wg_d = 256;
    while (embed_dim % static_cast<int64_t>(wg_d) != 0 && wg_d > 1) {
        wg_d /= 2;
    }

    auto global_range = sycl::range<2>(static_cast<size_t>(num_patches), static_cast<size_t>(embed_dim));
    auto local_range = sycl::range<2>(1, wg_d);

    return q.parallel_for(sycl::nd_range<2>(global_range, local_range), [=](sycl::nd_item<2> item) {
        int64_t p = item.get_global_id(0);
        int64_t d = item.get_global_id(1);

        float acc = 0.0f;
        const sycl::half* patch_ptr = d_image_pixels + p * patch_dim;
        const sycl::half* weight_ptr = d_patch_weight + d * patch_dim;

        for (int64_t k = 0; k < patch_dim; ++k) {
            acc += static_cast<float>(patch_ptr[k]) * static_cast<float>(weight_ptr[k]);
        }

        if (d_patch_bias) {
            acc += static_cast<float>(d_patch_bias[d]);
        }
        if (d_pos_embed) {
            acc += static_cast<float>(d_pos_embed[p * embed_dim + d]);
        }

        d_out_patches[p * embed_dim + d] = static_cast<sycl::half>(acc);
    });
}

sycl::event vit_attn(
    sycl::queue& q,
    sycl::half* d_out,
    const sycl::half* d_in,
    const sycl::half* d_qkv_weight,
    const sycl::half* d_qkv_bias,
    const sycl::half* d_proj_weight,
    const sycl::half* d_proj_bias,
    int64_t num_patches,
    int64_t embed_dim,
    int64_t num_heads
) {
    if (!d_out || !d_in || !d_qkv_weight || !d_proj_weight || num_patches <= 0 || embed_dim <= 0 || num_heads <= 0) {
        return sycl::event{};
    }

    int64_t head_dim = embed_dim / num_heads;
    float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));

    // Allocate temporary device buffers for QKV and Attention output if needed
    // In this fused ViT attention kernel, each work-item computes one output channel d of patch p.
    // 1. QKV projection: Q, K, V are derived
    // 2. Head attention: Q * K^T, softmax across all num_patches (M=256), * V
    // 3. Out projection + residual addition: d_out[p, d] = d_in[p, d] + proj(attn_out)
    return q.parallel_for(sycl::range<2>(static_cast<size_t>(num_patches), static_cast<size_t>(embed_dim)), [=](sycl::id<2> idx) {
        int64_t p = idx[0];
        int64_t d = idx[1];

        int64_t h = d / head_dim;
        int64_t c = d % head_dim;

        // Compute Q for this head and channel at patch p
        float q_val = 0.0f;
        const sycl::half* in_p = d_in + p * embed_dim;
        const sycl::half* w_q = d_qkv_weight + (0 * embed_dim + d) * embed_dim;
        for (int64_t k = 0; k < embed_dim; ++k) {
            q_val += static_cast<float>(in_p[k]) * static_cast<float>(w_q[k]);
        }
        if (d_qkv_bias) {
            q_val += static_cast<float>(d_qkv_bias[0 * embed_dim + d]);
        }

        // Online Softmax tracking for query patch p over key patches j in [0, num_patches)
        float max_s = -1e20f;
        float sum_exp = 0.0f;
        float weighted_v = 0.0f;

        // Pass 1: find max score and sum of exps (using registers for small M=256 patches)
        for (int64_t j = 0; j < num_patches; ++j) {
            const sycl::half* in_j = d_in + j * embed_dim;
            const sycl::half* w_k = d_qkv_weight + (1 * embed_dim + h * head_dim) * embed_dim;

            // Dot product Q[p, h, :] * K[j, h, :]
            float s = 0.0f;
            for (int64_t cd = 0; cd < head_dim; ++cd) {
                float q_elem = 0.0f;
                const sycl::half* w_q_elem = d_qkv_weight + (0 * embed_dim + h * head_dim + cd) * embed_dim;
                for (int64_t k = 0; k < embed_dim; ++k) {
                    q_elem += static_cast<float>(in_p[k]) * static_cast<float>(w_q_elem[k]);
                }
                if (d_qkv_bias) q_elem += static_cast<float>(d_qkv_bias[0 * embed_dim + h * head_dim + cd]);

                float k_elem = 0.0f;
                const sycl::half* w_k_elem = d_qkv_weight + (1 * embed_dim + h * head_dim + cd) * embed_dim;
                for (int64_t k = 0; k < embed_dim; ++k) {
                    k_elem += static_cast<float>(in_j[k]) * static_cast<float>(w_k_elem[k]);
                }
                if (d_qkv_bias) k_elem += static_cast<float>(d_qkv_bias[1 * embed_dim + h * head_dim + cd]);

                s += q_elem * k_elem;
            }
            s *= scale;

            // Online update
            if (s > max_s) {
                float exp_diff = sycl::exp(max_s - s);
                sum_exp = sum_exp * exp_diff + 1.0f;
                weighted_v *= exp_diff;
                max_s = s;
            } else {
                sum_exp += sycl::exp(s - max_s);
            }

            // Compute V element for channel c
            float v_elem = 0.0f;
            const sycl::half* w_v_elem = d_qkv_weight + (2 * embed_dim + d) * embed_dim;
            for (int64_t k = 0; k < embed_dim; ++k) {
                v_elem += static_cast<float>(in_j[k]) * static_cast<float>(w_v_elem[k]);
            }
            if (d_qkv_bias) v_elem += static_cast<float>(d_qkv_bias[2 * embed_dim + d]);

            weighted_v += sycl::exp(s - max_s) * v_elem;
        }

        float attn_out = (sum_exp > 0.0f) ? (weighted_v / sum_exp) : 0.0f;

        // Out projection and residual connection
        float proj_out = 0.0f;
        const sycl::half* w_proj = d_proj_weight + d * embed_dim;
        proj_out += attn_out * static_cast<float>(w_proj[d]);
        if (d_proj_bias) proj_out += static_cast<float>(d_proj_bias[d]);

        d_out[p * embed_dim + d] = static_cast<sycl::half>(static_cast<float>(d_in[p * embed_dim + d]) + proj_out);
    });
}

sycl::event vit_mlp(
    sycl::queue& q,
    sycl::half* d_out,
    const sycl::half* d_in,
    const sycl::half* d_fc1_weight,
    const sycl::half* d_fc1_bias,
    const sycl::half* d_fc2_weight,
    const sycl::half* d_fc2_bias,
    int64_t num_patches,
    int64_t embed_dim,
    int64_t intermediate_dim
) {
    if (!d_out || !d_in || !d_fc1_weight || !d_fc2_weight || num_patches <= 0 || embed_dim <= 0 || intermediate_dim <= 0) {
        return sycl::event{};
    }

    size_t wg_d = 256;
    while (embed_dim % static_cast<int64_t>(wg_d) != 0 && wg_d > 1) {
        wg_d /= 2;
    }

    auto global_range = sycl::range<2>(static_cast<size_t>(num_patches), static_cast<size_t>(embed_dim));
    auto local_range = sycl::range<2>(1, wg_d);

    // Two-layer MLP with GELU activation and residual connection
    // FC1: intermediate_dim x embed_dim -> GELU -> FC2: embed_dim x intermediate_dim
    return q.parallel_for(sycl::nd_range<2>(global_range, local_range), [=](sycl::nd_item<2> item) {
        int64_t p = item.get_global_id(0);
        int64_t d = item.get_global_id(1);

        const sycl::half* in_p = d_in + p * embed_dim;
        const sycl::half* w_fc2 = d_fc2_weight + d * intermediate_dim;

        float fc2_acc = 0.0f;
        for (int64_t m = 0; m < intermediate_dim; ++m) {
            // Compute FC1 activation m for patch p
            float fc1_val = 0.0f;
            const sycl::half* w_fc1 = d_fc1_weight + m * embed_dim;
            for (int64_t k = 0; k < embed_dim; ++k) {
                fc1_val += static_cast<float>(in_p[k]) * static_cast<float>(w_fc1[k]);
            }
            if (d_fc1_bias) fc1_val += static_cast<float>(d_fc1_bias[m]);

            // GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            float gelu = 0.5f * fc1_val * (1.0f + sycl::tanh(0.7978845608f * (fc1_val + 0.044715f * fc1_val * fc1_val * fc1_val)));
            fc2_acc += gelu * static_cast<float>(w_fc2[m]);
        }

        if (d_fc2_bias) fc2_acc += static_cast<float>(d_fc2_bias[d]);

        // Residual addition: out = in + mlp(in)
        d_out[p * embed_dim + d] = static_cast<sycl::half>(static_cast<float>(d_in[p * embed_dim + d]) + fc2_acc);
    });
}

sycl::event vit_project(
    sycl::queue& q,
    sycl::half* d_out_llm,
    const sycl::half* d_in_vit,
    const sycl::half* d_proj_weight,
    const sycl::half* d_proj_bias,
    int64_t num_patches,
    int64_t vit_dim,
    int64_t llm_dim
) {
    if (!d_out_llm || !d_in_vit || !d_proj_weight || num_patches <= 0 || vit_dim <= 0 || llm_dim <= 0) {
        return sycl::event{};
    }

    size_t wg_d = 256;
    while (llm_dim % static_cast<int64_t>(wg_d) != 0 && wg_d > 1) {
        wg_d /= 2;
    }

    auto global_range = sycl::range<2>(static_cast<size_t>(num_patches), static_cast<size_t>(llm_dim));
    auto local_range = sycl::range<2>(1, wg_d);

    // Linear projection: [num_patches, vit_dim] * [llm_dim, vit_dim]^T -> [num_patches, llm_dim]
    return q.parallel_for(sycl::nd_range<2>(global_range, local_range), [=](sycl::nd_item<2> item) {
        int64_t p = item.get_global_id(0);
        int64_t d = item.get_global_id(1);

        float acc = 0.0f;
        const sycl::half* in_p = d_in_vit + p * vit_dim;
        const sycl::half* w_row = d_proj_weight + d * vit_dim;

        for (int64_t k = 0; k < vit_dim; ++k) {
            acc += static_cast<float>(in_p[k]) * static_cast<float>(w_row[k]);
        }

        if (d_proj_bias) {
            acc += static_cast<float>(d_proj_bias[d]);
        }

        d_out_llm[p * llm_dim + d] = static_cast<sycl::half>(acc);
    });
}

} // namespace xinfer::ops
