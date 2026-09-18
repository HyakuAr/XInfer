#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

namespace xinfer::ops {

// Scaled Dot-Product Attention (SDPA) with Grouped-Query Attention (GQA) and Causal Masking
// Q: [seq_len, num_q_heads, head_dim]
// K: [seq_len, num_kv_heads, head_dim]
// V: [seq_len, num_kv_heads, head_dim]
// out: [seq_len, num_q_heads, head_dim]
// scale: default 1.0 / sqrt(head_dim)
void sdpa_causal_naive(sycl::queue& q,
                       float* out,
                       const float* Q,
                       const float* K,
                       const float* V,
                       int64_t seq_len,
                       int64_t num_q_heads,
                       int64_t num_kv_heads,
                       int64_t head_dim,
                       float scale = 0.0f);

void sdpa_causal_naive(sycl::queue& q,
                       sycl::half* out,
                       const sycl::half* Q,
                       const sycl::half* K,
                       const sycl::half* V,
                       int64_t seq_len,
                       int64_t num_q_heads,
                       int64_t num_kv_heads,
                       int64_t head_dim,
                       float scale = 0.0f);

// Write computed K and V into FP16 KV cache at [start_pos, start_pos + num_tokens)
// max_seq_len: bounds check limit (0 = unbounded)
void attention_write_kv_cache(sycl::queue& q,
                              sycl::half* k_cache,
                              sycl::half* v_cache,
                              const float* K_in,
                              const float* V_in,
                              int64_t start_pos,
                              int64_t num_tokens,
                              int64_t num_kv_heads,
                              int64_t head_dim,
                              int64_t max_seq_len = 0);

void attention_write_kv_cache(sycl::queue& q,
                              sycl::half* k_cache,
                              sycl::half* v_cache,
                              const sycl::half* K_in,
                              const sycl::half* V_in,
                              int64_t start_pos,
                              int64_t num_tokens,
                              int64_t num_kv_heads,
                              int64_t head_dim,
                              int64_t max_seq_len = 0);

// Dynamic device-pointer overload for command-graph capture/replay
// max_seq_len: bounds check limit (0 = unbounded)
sycl::event attention_write_kv_cache_dynamic(sycl::queue& q,
                                       sycl::half* k_cache,
                                       sycl::half* v_cache,
                                       const float* K_in,
                                       const float* V_in,
                                       const int64_t* d_start_pos,
                                       int64_t num_tokens,
                                       int64_t num_kv_heads,
                                       int64_t head_dim,
                                       int64_t max_seq_len = 0);

sycl::event attention_write_kv_cache_dynamic(sycl::queue& q,
                                       sycl::half* k_cache,
                                       sycl::half* v_cache,
                                       const sycl::half* K_in,
                                       const sycl::half* V_in,
                                       const int64_t* d_start_pos,
                                       int64_t num_tokens,
                                       int64_t num_kv_heads,
                                       int64_t head_dim,
                                       int64_t max_seq_len = 0);

// Causal SDPA reading from FP16 KV cache (supports single-token decode and chunked prefill)
// Q: [num_q_tokens, num_q_heads, head_dim]
// k_cache, v_cache: [max_seq_len, num_kv_heads, head_dim] (sycl::half)
// out: [num_q_tokens, num_q_heads, head_dim]
// start_pos: absolute sequence position of first query token in Q
void sdpa_causal_cached(sycl::queue& q,
                        float* out,
                        const float* Q,
                        const sycl::half* k_cache,
                        const sycl::half* v_cache,
                        int64_t start_pos,
                        int64_t num_q_tokens,
                        int64_t num_q_heads,
                        int64_t num_kv_heads,
                        int64_t head_dim,
                        float scale = 0.0f,
                        int64_t max_seq_len = 0);

void sdpa_causal_cached(sycl::queue& q,
                        sycl::half* out,
                        const sycl::half* Q,
                        const sycl::half* k_cache,
                        const sycl::half* v_cache,
                        int64_t start_pos,
                        int64_t num_q_tokens,
                        int64_t num_q_heads,
                        int64_t num_kv_heads,
                        int64_t head_dim,
                        float scale = 0.0f,
                        int64_t max_seq_len = 0);

// Dynamic device-pointer overload for command-graph capture/replay
sycl::event sdpa_causal_cached_dynamic(sycl::queue& q,
                                 float* out,
                                 const float* Q,
                                 const sycl::half* k_cache,
                                 const sycl::half* v_cache,
                                 const int64_t* d_start_pos,
                                 int64_t num_q_tokens,
                                 int64_t num_q_heads,
                                 int64_t num_kv_heads,
                                 int64_t head_dim,
                                 float scale = 0.0f,
                                 int64_t max_seq_len = 0);

sycl::event sdpa_causal_cached_dynamic(sycl::queue& q,
                                 sycl::half* out,
                                 const sycl::half* Q,
                                 const sycl::half* k_cache,
                                 const sycl::half* v_cache,
                                 const int64_t* d_start_pos,
                                 int64_t num_q_tokens,
                                 int64_t num_q_heads,
                                 int64_t num_kv_heads,
                                 int64_t head_dim,
                                 float scale = 0.0f,
                                 int64_t max_seq_len = 0);

// Write computed K and V into INT8 KV cache with dynamic min/max scaling per head
void attention_write_kv_cache_int8(sycl::queue& q,
                                   int8_t* k_cache,
                                   int8_t* v_cache,
                                   float* k_scale,
                                   float* v_scale,
                                   float* k_zero_point,
                                   float* v_zero_point,
                                   const float* K_in,
                                   const float* V_in,
                                   int64_t start_pos,
                                   int64_t num_tokens,
                                   int64_t num_kv_heads,
                                   int64_t head_dim,
                                   int64_t max_seq_len = 0);

void attention_write_kv_cache_int8(sycl::queue& q,
                                   int8_t* k_cache,
                                   int8_t* v_cache,
                                   float* k_scale,
                                   float* v_scale,
                                   float* k_zero_point,
                                   float* v_zero_point,
                                   const sycl::half* K_in,
                                   const sycl::half* V_in,
                                   int64_t start_pos,
                                   int64_t num_tokens,
                                   int64_t num_kv_heads,
                                   int64_t head_dim,
                                   int64_t max_seq_len = 0);

sycl::event attention_write_kv_cache_int8_dynamic(sycl::queue& q,
                                                  int8_t* k_cache,
                                                  int8_t* v_cache,
                                                  float* k_scale,
                                                  float* v_scale,
                                                  float* k_zero_point,
                                                  float* v_zero_point,
                                                  const float* K_in,
                                                  const float* V_in,
                                                  const int64_t* d_start_pos,
                                                  int64_t num_tokens,
                                                  int64_t num_kv_heads,
                                                  int64_t head_dim,
                                                  int64_t max_seq_len = 0);

sycl::event attention_write_kv_cache_int8_dynamic(sycl::queue& q,
                                                  int8_t* k_cache,
                                                  int8_t* v_cache,
                                                  float* k_scale,
                                                  float* v_scale,
                                                  float* k_zero_point,
                                                  float* v_zero_point,
                                                  const sycl::half* K_in,
                                                  const sycl::half* V_in,
                                                  const int64_t* d_start_pos,
                                                  int64_t num_tokens,
                                                  int64_t num_kv_heads,
                                                  int64_t head_dim,
                                                  int64_t max_seq_len = 0);

// Dequantize INT8 KV cache back to FP16/FP32
void attention_read_kv_cache_int8(sycl::queue& q,
                                  sycl::half* K_out,
                                  sycl::half* V_out,
                                  const int8_t* k_cache,
                                  const int8_t* v_cache,
                                  const float* k_scale,
                                  const float* v_scale,
                                  const float* k_zero_point,
                                  const float* v_zero_point,
                                  int64_t start_pos,
                                  int64_t num_tokens,
                                  int64_t num_kv_heads,
                                  int64_t head_dim,
                                  int64_t max_seq_len = 0);

void attention_read_kv_cache_int8(sycl::queue& q,
                                  float* K_out,
                                  float* V_out,
                                  const int8_t* k_cache,
                                  const int8_t* v_cache,
                                  const float* k_scale,
                                  const float* v_scale,
                                  const float* k_zero_point,
                                  const float* v_zero_point,
                                  int64_t start_pos,
                                  int64_t num_tokens,
                                  int64_t num_kv_heads,
                                  int64_t head_dim,
                                  int64_t max_seq_len = 0);

// Causal SDPA reading from INT8 KV cache with on-the-fly dequantization
void sdpa_causal_cached_int8(sycl::queue& q,
                             float* out,
                             const float* Q,
                             const int8_t* k_cache,
                             const int8_t* v_cache,
                             const float* k_scale,
                             const float* v_scale,
                             const float* k_zero_point,
                             const float* v_zero_point,
                             int64_t start_pos,
                             int64_t num_q_tokens,
                             int64_t num_q_heads,
                             int64_t num_kv_heads,
                             int64_t head_dim,
                             float scale = 0.0f,
                             int64_t max_seq_len = 0);

void sdpa_causal_cached_int8(sycl::queue& q,
                             sycl::half* out,
                             const sycl::half* Q,
                             const int8_t* k_cache,
                             const int8_t* v_cache,
                             const float* k_scale,
                             const float* v_scale,
                             const float* k_zero_point,
                             const float* v_zero_point,
                             int64_t start_pos,
                             int64_t num_q_tokens,
                             int64_t num_q_heads,
                             int64_t num_kv_heads,
                             int64_t head_dim,
                             float scale = 0.0f,
                             int64_t max_seq_len = 0);

sycl::event sdpa_causal_cached_int8_dynamic(sycl::queue& q,
                                            float* out,
                                            const float* Q,
                                            const int8_t* k_cache,
                                            const int8_t* v_cache,
                                            const float* k_scale,
                                            const float* v_scale,
                                            const float* k_zero_point,
                                            const float* v_zero_point,
                                            const int64_t* d_start_pos,
                                            int64_t num_q_tokens,
                                            int64_t num_q_heads,
                                            int64_t num_kv_heads,
                                            int64_t head_dim,
                                            float scale = 0.0f,
                                            int64_t max_seq_len = 0);

sycl::event sdpa_causal_cached_int8_dynamic(sycl::queue& q,
                                            sycl::half* out,
                                            const sycl::half* Q,
                                            const int8_t* k_cache,
                                            const int8_t* v_cache,
                                            const float* k_scale,
                                            const float* v_scale,
                                            const float* k_zero_point,
                                            const float* v_zero_point,
                                            const int64_t* d_start_pos,
                                            int64_t num_q_tokens,
                                            int64_t num_q_heads,
                                            int64_t num_kv_heads,
                                            int64_t head_dim,
                                            float scale = 0.0f,
                                            int64_t max_seq_len = 0);

// Batched speculative verification attention kernel (M = N):
// Allows draft token i in [0, num_draft_tokens) to attend to prefix KV cache (positions 0 .. prefix_len - 1)
// and preceding draft tokens in the window (positions prefix_len .. prefix_len + i).
void sdpa_causal_verify(sycl::queue& q,
                        float* out,
                        const float* Q,
                        const sycl::half* k_cache,
                        const sycl::half* v_cache,
                        int64_t prefix_len,
                        int64_t num_draft_tokens,
                        int64_t num_q_heads,
                        int64_t num_kv_heads,
                        int64_t head_dim,
                        float scale = 0.0f,
                        int64_t max_seq_len = 0);

void sdpa_causal_verify(sycl::queue& q,
                        sycl::half* out,
                        const sycl::half* Q,
                        const sycl::half* k_cache,
                        const sycl::half* v_cache,
                        int64_t prefix_len,
                        int64_t num_draft_tokens,
                        int64_t num_q_heads,
                        int64_t num_kv_heads,
                        int64_t head_dim,
                        float scale = 0.0f,
                        int64_t max_seq_len = 0);

void sdpa_causal_verify_int8(sycl::queue& q,
                             float* out,
                             const float* Q,
                             const int8_t* k_cache,
                             const int8_t* v_cache,
                             const float* k_scale,
                             const float* v_scale,
                             const float* k_zero_point,
                             const float* v_zero_point,
                             int64_t prefix_len,
                             int64_t num_draft_tokens,
                             int64_t num_q_heads,
                             int64_t num_kv_heads,
                             int64_t head_dim,
                             float scale = 0.0f,
                             int64_t max_seq_len = 0);

void sdpa_causal_verify_int8(sycl::queue& q,
                             sycl::half* out,
                             const sycl::half* Q,
                             const int8_t* k_cache,
                             const int8_t* v_cache,
                             const float* k_scale,
                             const float* v_scale,
                             const float* k_zero_point,
                             const float* v_zero_point,
                             int64_t prefix_len,
                             int64_t num_draft_tokens,
                             int64_t num_q_heads,
                             int64_t num_kv_heads,
                             int64_t head_dim,
                             float scale = 0.0f,
                             int64_t max_seq_len = 0);

} // namespace xinfer::ops
