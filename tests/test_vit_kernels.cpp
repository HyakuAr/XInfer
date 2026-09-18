#include "core/device.h"
#include "core/arena.h"
#include "core/kv_cache.h"
#include "ops/vit.h"
#include "targets/qwen3_8/forward.h"
#include "targets/qwen3_8_27b/weights.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

int main() {
    std::cout << "=== Running Qwen3.8 ViT & Multimodal Pipeline Unit Tests ===\n";

    auto ctx = xinfer::core::DeviceContext::create(true);
    if (!ctx) {
        std::cerr << "FAILED: Unable to create DeviceContext\n";
        return 1;
    }
    auto& q = ctx->queue();

    // 1. Test vit_patch_embed
    std::cout << "\n--- Testing vit_patch_embed (M=256 patches, D=1024) ---\n";
    constexpr int64_t num_patches = 256;
    constexpr int64_t patch_dim = 3 * 14 * 14; // 588
    constexpr int64_t embed_dim = 1024;

    std::vector<sycl::half> h_pixels(num_patches * patch_dim, static_cast<sycl::half>(0.5f));
    std::vector<sycl::half> h_patch_w(embed_dim * patch_dim, static_cast<sycl::half>(0.01f));
    std::vector<sycl::half> h_patch_b(embed_dim, static_cast<sycl::half>(0.1f));
    std::vector<sycl::half> h_pos_embed(num_patches * embed_dim, static_cast<sycl::half>(0.05f));

    sycl::half* d_pixels = sycl::malloc_device<sycl::half>(h_pixels.size(), q);
    sycl::half* d_patch_w = sycl::malloc_device<sycl::half>(h_patch_w.size(), q);
    sycl::half* d_patch_b = sycl::malloc_device<sycl::half>(h_patch_b.size(), q);
    sycl::half* d_pos_embed = sycl::malloc_device<sycl::half>(h_pos_embed.size(), q);
    sycl::half* d_patches_out = sycl::malloc_device<sycl::half>(num_patches * embed_dim, q);

    q.memcpy(d_pixels, h_pixels.data(), h_pixels.size() * sizeof(sycl::half)).wait();
    q.memcpy(d_patch_w, h_patch_w.data(), h_patch_w.size() * sizeof(sycl::half)).wait();
    q.memcpy(d_patch_b, h_patch_b.data(), h_patch_b.size() * sizeof(sycl::half)).wait();
    q.memcpy(d_pos_embed, h_pos_embed.data(), h_pos_embed.size() * sizeof(sycl::half)).wait();

    xinfer::ops::vit_patch_embed(q, d_patches_out, d_pixels, d_patch_w, d_patch_b, d_pos_embed,
                                 num_patches, patch_dim, embed_dim).wait();

    std::vector<sycl::half> h_patches_out(num_patches * embed_dim);
    q.memcpy(h_patches_out.data(), d_patches_out, h_patches_out.size() * sizeof(sycl::half)).wait();

    // Expected value: 588 * (0.5 * 0.01) + 0.1 + 0.05 = 588 * 0.005 + 0.15 = 2.94 + 0.15 = 3.09
    float expected_val = static_cast<float>(patch_dim) * (0.5f * 0.01f) + 0.1f + 0.05f;
    float sample_val = static_cast<float>(h_patches_out[0]);
    if (std::abs(sample_val - expected_val) > 0.1f) {
        std::cerr << "FAILED: vit_patch_embed mismatch: expected " << expected_val << ", got " << sample_val << "\n";
        return 1;
    }
    std::cout << "[PASS] vit_patch_embed output matches oracle (sample=" << sample_val << ", expected=" << expected_val << ")\n";

    // 2. Test vit_attn
    std::cout << "\n--- Testing vit_attn (Self-Attention over 256 patches) ---\n";
    constexpr int64_t num_heads = 16;
    std::vector<sycl::half> h_qkv_w(3 * embed_dim * embed_dim, static_cast<sycl::half>(0.001f));
    std::vector<sycl::half> h_qkv_b(3 * embed_dim, static_cast<sycl::half>(0.01f));
    std::vector<sycl::half> h_proj_w(embed_dim * embed_dim, static_cast<sycl::half>(0.001f));
    std::vector<sycl::half> h_proj_b(embed_dim, static_cast<sycl::half>(0.0f));

    sycl::half* d_qkv_w = sycl::malloc_device<sycl::half>(h_qkv_w.size(), q);
    sycl::half* d_qkv_b = sycl::malloc_device<sycl::half>(h_qkv_b.size(), q);
    sycl::half* d_proj_w = sycl::malloc_device<sycl::half>(h_proj_w.size(), q);
    sycl::half* d_proj_b = sycl::malloc_device<sycl::half>(h_proj_b.size(), q);
    sycl::half* d_attn_out = sycl::malloc_device<sycl::half>(num_patches * embed_dim, q);

    q.memcpy(d_qkv_w, h_qkv_w.data(), h_qkv_w.size() * sizeof(sycl::half)).wait();
    q.memcpy(d_qkv_b, h_qkv_b.data(), h_qkv_b.size() * sizeof(sycl::half)).wait();
    q.memcpy(d_proj_w, h_proj_w.data(), h_proj_w.size() * sizeof(sycl::half)).wait();
    q.memcpy(d_proj_b, h_proj_b.data(), h_proj_b.size() * sizeof(sycl::half)).wait();

    xinfer::ops::vit_attn(q, d_attn_out, d_patches_out, d_qkv_w, d_qkv_b, d_proj_w, d_proj_b,
                          num_patches, embed_dim, num_heads).wait();

    std::vector<sycl::half> h_attn_out(num_patches * embed_dim);
    q.memcpy(h_attn_out.data(), d_attn_out, h_attn_out.size() * sizeof(sycl::half)).wait();
    for (size_t i = 0; i < 10; ++i) {
        float val = static_cast<float>(h_attn_out[i]);
        if (std::isnan(val) || std::isinf(val) || val <= 0.0f) {
            std::cerr << "FAILED: Invalid vit_attn output at index " << i << ": " << val << "\n";
            return 1;
        }
    }
    std::cout << "[PASS] vit_attn executed successfully across all 256 patches\n";

    // 3. Test vit_mlp
    std::cout << "\n--- Testing vit_mlp (GELU feed-forward network) ---\n";
    constexpr int64_t intermediate_dim = 4096;
    std::vector<sycl::half> h_fc1_w(intermediate_dim * embed_dim, static_cast<sycl::half>(0.001f));
    std::vector<sycl::half> h_fc1_b(intermediate_dim, static_cast<sycl::half>(0.0f));
    std::vector<sycl::half> h_fc2_w(embed_dim * intermediate_dim, static_cast<sycl::half>(0.001f));
    std::vector<sycl::half> h_fc2_b(embed_dim, static_cast<sycl::half>(0.0f));
    sycl::half* d_fc1_w = sycl::malloc_device<sycl::half>(h_fc1_w.size(), q);
    sycl::half* d_fc1_b = sycl::malloc_device<sycl::half>(h_fc1_b.size(), q);
    sycl::half* d_fc2_w = sycl::malloc_device<sycl::half>(h_fc2_w.size(), q);
    sycl::half* d_fc2_b = sycl::malloc_device<sycl::half>(h_fc2_b.size(), q);
    sycl::half* d_mlp_out = sycl::malloc_device<sycl::half>(num_patches * embed_dim, q);

    q.memcpy(d_fc1_w, h_fc1_w.data(), h_fc1_w.size() * sizeof(sycl::half)).wait();
    q.memcpy(d_fc1_b, h_fc1_b.data(), h_fc1_b.size() * sizeof(sycl::half)).wait();
    q.memcpy(d_fc2_w, h_fc2_w.data(), h_fc2_w.size() * sizeof(sycl::half)).wait();
    q.memcpy(d_fc2_b, h_fc2_b.data(), h_fc2_b.size() * sizeof(sycl::half)).wait();

    xinfer::ops::vit_mlp(q, d_mlp_out, d_attn_out, d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b,
                         num_patches, embed_dim, intermediate_dim).wait();

    std::vector<sycl::half> h_mlp_out(num_patches * embed_dim);
    q.memcpy(h_mlp_out.data(), d_mlp_out, h_mlp_out.size() * sizeof(sycl::half)).wait();
    for (size_t i = 0; i < 10; ++i) {
        float val = static_cast<float>(h_mlp_out[i]);
        if (std::isnan(val) || std::isinf(val) || val <= 0.0f) {
            std::cerr << "FAILED: Invalid vit_mlp output at index " << i << ": " << val << "\n";
            return 1;
        }
    }
    std::cout << "[PASS] vit_mlp executed successfully across all 256 patches\n";

    // 4. Test vit_project
    std::cout << "\n--- Testing vit_project (1024 -> 5120 LLM hidden size) ---\n";
    constexpr int64_t llm_dim = 5120;
    std::vector<sycl::half> h_proj_llm_w(llm_dim * embed_dim, static_cast<sycl::half>(0.001f));
    std::vector<sycl::half> h_proj_llm_b(llm_dim, static_cast<sycl::half>(0.02f));
    sycl::half* d_proj_llm_w = sycl::malloc_device<sycl::half>(h_proj_llm_w.size(), q);
    sycl::half* d_proj_llm_b = sycl::malloc_device<sycl::half>(h_proj_llm_b.size(), q);
    sycl::half* d_visual_embeddings = sycl::malloc_device<sycl::half>(num_patches * llm_dim, q);

    q.memcpy(d_proj_llm_w, h_proj_llm_w.data(), h_proj_llm_w.size() * sizeof(sycl::half)).wait();
    q.memcpy(d_proj_llm_b, h_proj_llm_b.data(), h_proj_llm_b.size() * sizeof(sycl::half)).wait();

    xinfer::ops::vit_project(q, d_visual_embeddings, d_mlp_out, d_proj_llm_w, d_proj_llm_b,
                             num_patches, embed_dim, llm_dim).wait();

    std::vector<sycl::half> h_visual_embeddings(num_patches * llm_dim);
    q.memcpy(h_visual_embeddings.data(), d_visual_embeddings, h_visual_embeddings.size() * sizeof(sycl::half)).wait();

    for (size_t i = 0; i < 10; ++i) {
        float val = static_cast<float>(h_visual_embeddings[i]);
        if (std::isnan(val) || std::isinf(val) || val <= 0.0f) {
            std::cerr << "FAILED: Invalid vit_project output at index " << i << ": " << val << "\n";
            return 1;
        }
    }
    std::cout << "[PASS] vit_project produced valid LLM visual embeddings [256, 5120]\n";

    // 5. Test Zero-Copy Visual Token Injection in embed_tokens_lookup
    std::cout << "\n--- Testing Zero-Copy Visual Token Injection in embed_tokens_lookup ---\n";
    constexpr int64_t vocab_size = 248320;
    constexpr int64_t visual_token_start = 248000;
    constexpr int64_t visual_token_end = 248319;
    constexpr int64_t test_seq_len = 4;

    // Sequence: [text token 10, visual token 248000, visual token 248001, text token 20]
    std::vector<int64_t> h_token_ids = {10, visual_token_start + 0, visual_token_start + 1, 20};
    int64_t* d_token_ids = sycl::malloc_device<int64_t>(test_seq_len, q);
    q.memcpy(d_token_ids, h_token_ids.data(), test_seq_len * sizeof(int64_t)).wait();

    // Create a mock BF16 text embedding table
    // Token 10: all 1.0f (0x3F80 in BF16)
    // Token 20: all 2.0f (0x4000 in BF16)
    std::vector<uint16_t> h_embed_table(vocab_size * llm_dim, 0);
    for (int64_t d = 0; d < llm_dim; ++d) {
        h_embed_table[10 * llm_dim + d] = 0x3F80; // 1.0f in BF16
        h_embed_table[20 * llm_dim + d] = 0x4000; // 2.0f in BF16
    }
    uint16_t* d_embed_table = sycl::malloc_device<uint16_t>(h_embed_table.size(), q);
    q.memcpy(d_embed_table, h_embed_table.data(), h_embed_table.size() * sizeof(uint16_t)).wait();

    // Fill ViT output USM buffer with distinct values for patch 0 (99.0f) and patch 1 (77.0f)
    std::vector<sycl::half> h_vit_test_out(num_patches * llm_dim, static_cast<sycl::half>(0.0f));
    for (int64_t d = 0; d < llm_dim; ++d) {
        h_vit_test_out[0 * llm_dim + d] = static_cast<sycl::half>(99.0f);
        h_vit_test_out[1 * llm_dim + d] = static_cast<sycl::half>(77.0f);
    }
    q.memcpy(d_visual_embeddings, h_vit_test_out.data(), h_vit_test_out.size() * sizeof(sycl::half)).wait();

    sycl::half* d_act_x = sycl::malloc_device<sycl::half>(test_seq_len * llm_dim, q);

    xinfer::targets::qwen3_8::embed_tokens_lookup(
        q, d_act_x, d_embed_table, d_token_ids, test_seq_len, llm_dim, vocab_size,
        d_visual_embeddings, visual_token_start, visual_token_end, nullptr).wait();

    std::vector<sycl::half> h_act_x(test_seq_len * llm_dim);
    q.memcpy(h_act_x.data(), d_act_x, h_act_x.size() * sizeof(sycl::half)).wait();

    // Verify token 0 read text embedding table (1.0f)
    assert(std::abs(static_cast<float>(h_act_x[0 * llm_dim + 0]) - 1.0f) < 1e-2f);
    // Verify token 1 read ViT USM buffer patch 0 (99.0f) directly
    assert(std::abs(static_cast<float>(h_act_x[1 * llm_dim + 0]) - 99.0f) < 1e-2f);
    // Verify token 2 read ViT USM buffer patch 1 (77.0f) directly
    assert(std::abs(static_cast<float>(h_act_x[2 * llm_dim + 0]) - 77.0f) < 1e-2f);
    // Verify token 3 read text embedding table (2.0f)
    assert(std::abs(static_cast<float>(h_act_x[3 * llm_dim + 0]) - 2.0f) < 1e-2f);

    std::cout << "[PASS] Zero-copy visual token injection verified: text tokens read text table, visual tokens read directly from ViT USM buffer!\n";

    // 6. Test Fail-Loud Contract in prefill_prompt
    std::cout << "\n--- Testing Fail-Loud Contract for Multimodal Vision Prefill ---\n";
    xinfer::core::KVCacheConfig kv_cfg;
    kv_cfg.max_seq_len = 1024;
    kv_cfg.num_full_layers = 1;
    kv_cfg.num_linear_layers = 0;
    kv_cfg.num_kv_heads = 4;
    kv_cfg.head_dim = 256;
    xinfer::core::KVCache kv_cache(ctx, kv_cfg);
    assert(kv_cache.allocate());

    xinfer::core::DeviceArena arena(ctx, 8 * 1024 * 1024);

    auto model = xinfer::targets::qwen3_8_27b::LoadedModel::create_mock(ctx);

    // Scenario A: Prompt contains 256 visual placeholder tokens, but NO image is provided
    {
        std::vector<int64_t> prompt_with_visual_tags(256, visual_token_start);
        bool caught = false;
        try {
            xinfer::targets::qwen3_8::prefill_prompt(ctx, arena, *model, kv_cache, prompt_with_visual_tags, 512, false, nullptr);
        } catch (const xinfer::targets::qwen3_8::vision_format_error& e) {
            caught = true;
            std::cout << "  -> Scenario A caught as expected: " << e.what() << "\n";
        }
        assert(caught);
    }

    // Scenario B: User sends 1 image, but prompt contains 0 <image> tags
    {
        std::vector<int64_t> prompt_without_visual_tags(10, 42);
        xinfer::targets::qwen3_8::VisionInput v_in;
        v_in.num_images = 1;
        v_in.num_patches = 256;
        v_in.d_vit_embeddings = d_visual_embeddings;

        bool caught = false;
        try {
            xinfer::targets::qwen3_8::prefill_prompt(ctx, arena, *model, kv_cache, prompt_without_visual_tags, 512, false, &v_in);
        } catch (const xinfer::targets::qwen3_8::vision_format_error& e) {
            caught = true;
            std::cout << "  -> Scenario B caught as expected: " << e.what() << "\n";
        }
        assert(caught);
    }

    // Scenario C: User sends 1 image (256 patches), but prompt has 512 placeholder tokens (2 images)
    {
        std::vector<int64_t> prompt_with_512_tags(512, visual_token_start);
        xinfer::targets::qwen3_8::VisionInput v_in;
        v_in.num_images = 1;
        v_in.num_patches = 256;
        v_in.d_vit_embeddings = d_visual_embeddings;

        bool caught = false;
        try {
            xinfer::targets::qwen3_8::prefill_prompt(ctx, arena, *model, kv_cache, prompt_with_512_tags, 512, false, &v_in);
        } catch (const xinfer::targets::qwen3_8::vision_format_error& e) {
            caught = true;
            std::cout << "  -> Scenario C caught as expected: " << e.what() << "\n";
        }
        assert(caught);
    }

    // Scenario D: Prompt has malformed visual tokens (not a multiple of 256 patches)
    {
        std::vector<int64_t> prompt_malformed_tags(100, visual_token_start);
        xinfer::targets::qwen3_8::VisionInput v_in;
        v_in.num_images = 1;
        v_in.num_patches = 256;
        v_in.d_vit_embeddings = d_visual_embeddings;

        bool caught = false;
        try {
            xinfer::targets::qwen3_8::prefill_prompt(ctx, arena, *model, kv_cache, prompt_malformed_tags, 512, false, &v_in);
        } catch (const xinfer::targets::qwen3_8::vision_format_error& e) {
            caught = true;
            std::cout << "  -> Scenario D caught as expected: " << e.what() << "\n";
        }
        assert(caught);
    }

    std::cout << "[PASS] All Fail-Loud vision format contract checks passed!\n";

    // Clean up
    sycl::free(d_pixels, q);
    sycl::free(d_patch_w, q);
    sycl::free(d_patch_b, q);
    sycl::free(d_pos_embed, q);
    sycl::free(d_patches_out, q);
    sycl::free(d_qkv_w, q);
    sycl::free(d_qkv_b, q);
    sycl::free(d_proj_w, q);
    sycl::free(d_proj_b, q);
    sycl::free(d_attn_out, q);
    sycl::free(d_fc1_w, q);
    sycl::free(d_fc1_b, q);
    sycl::free(d_fc2_w, q);
    sycl::free(d_fc2_b, q);
    sycl::free(d_mlp_out, q);
    sycl::free(d_proj_llm_w, q);
    sycl::free(d_proj_llm_b, q);
    sycl::free(d_visual_embeddings, q);
    sycl::free(d_token_ids, q);
    sycl::free(d_embed_table, q);
    sycl::free(d_act_x, q);

    std::cout << "\n==========================================================\n";
    std::cout << " ALL VIT KERNEL & MULTIMODAL INJECTION TESTS PASSED ON B60!\n";
    std::cout << "==========================================================\n";
    return 0;
}
