# Intel Arc Pro B60 Matrix Capabilities (Hardware Query)

- **Device:** Intel(R) Arc(TM) Pro B60 Graphics (Battlemage / Xe2-HPG, Device ID `0xE211`)
- **Query Tool:** `tools/parity/query_matrix_caps.cpp`
- **Compiler / Runtime:** Intel oneAPI DPC++ Compiler 2026.1, Level Zero backend
- **Aspect `sycl::aspect::ext_intel_matrix`:** `true`
- **Total Supported Matrix Combinations:** 53

---

## 1. Raw Query Output

```
Device: Intel(R) Arc(TM) Pro B60 Graphics
Has ext_intel_matrix: true
Total supported matrix combinations: 53
atype   btype   ctype   dtype   max_m   max_n   max_k   msize   nsize   ksize   
--------------------------------------------------------------------------------
uint8   uint8   sint32  sint32  8       0       0       0       16      32      
uint8   sint8   sint32  sint32  8       0       0       0       16      32      
sint8   uint8   sint32  sint32  8       0       0       0       16      32      
sint8   sint8   sint32  sint32  8       0       0       0       16      32      
fp16    fp16    fp32    fp32    8       0       0       0       16      16      
fp16    fp16    fp16    fp32    8       0       0       0       16      16      
fp16    fp16    fp32    fp16    8       0       0       0       16      16      
fp16    fp16    fp16    fp16    8       0       0       0       16      16      
fp16    fp16    fp32    fp32    0       0       0       16      16      16      
fp16    fp16    fp32    fp16    0       0       0       16      16      16      
fp16    fp16    fp16    fp32    0       0       0       16      16      16      
fp16    fp16    fp16    fp16    0       0       0       16      16      16      
fp16    fp16    fp32    fp32    0       0       0       1       64      16      
fp16    fp16    fp16    fp32    0       0       0       1       64      16      
fp16    fp16    fp32    fp16    0       0       0       1       64      16      
fp16    fp16    fp16    fp16    0       0       0       1       64      16      
fp16    fp16    fp32    fp32    0       0       0       32      64      16      
fp16    fp16    fp16    fp32    0       0       0       32      64      16      
fp16    fp16    fp32    fp16    0       0       0       32      64      16      
fp16    fp16    fp16    fp16    0       0       0       32      64      16      
fp16    fp16    fp32    fp32    0       0       0       1       64      32      
fp16    fp16    fp16    fp32    0       0       0       1       64      32      
fp16    fp16    fp32    fp16    0       0       0       1       64      32      
fp16    fp16    fp16    fp16    0       0       0       1       64      32      
fp16    fp16    fp32    fp32    0       0       0       32      64      32      
fp16    fp16    fp16    fp32    0       0       0       32      64      32      
fp16    fp16    fp32    fp16    0       0       0       32      64      32      
fp16    fp16    fp16    fp16    0       0       0       32      64      32      
bf16    bf16    bf16    bf16    8       0       0       0       16      16      
bf16    bf16    fp32    bf16    8       0       0       0       16      16      
bf16    bf16    bf16    fp32    8       0       0       0       16      16      
bf16    bf16    fp32    fp32    8       0       0       0       16      16      
bf16    bf16    fp32    fp32    0       0       0       16      16      16      
bf16    bf16    bf16    fp32    0       0       0       16      16      16      
bf16    bf16    fp32    bf16    0       0       0       16      16      16      
bf16    bf16    bf16    bf16    0       0       0       16      16      16      
bf16    bf16    fp32    fp32    0       0       0       1       64      16      
bf16    bf16    bf16    fp32    0       0       0       1       64      16      
bf16    bf16    fp32    bf16    0       0       0       1       64      16      
bf16    bf16    bf16    bf16    0       0       0       1       64      16      
bf16    bf16    fp32    fp32    0       0       0       32      64      16      
bf16    bf16    bf16    fp32    0       0       0       32      64      16      
bf16    bf16    fp32    bf16    0       0       0       32      64      16      
bf16    bf16    bf16    bf16    0       0       0       32      64      16      
bf16    bf16    fp32    fp32    0       0       0       1       64      32      
bf16    bf16    bf16    fp32    0       0       0       1       64      32      
bf16    bf16    fp32    bf16    0       0       0       1       64      32      
bf16    bf16    bf16    bf16    0       0       0       1       64      32      
bf16    bf16    fp32    fp32    0       0       0       32      64      32      
bf16    bf16    bf16    fp32    0       0       0       32      64      32      
bf16    bf16    fp32    bf16    0       0       0       32      64      32      
bf16    bf16    bf16    bf16    0       0       0       32      64      32      
tf32    tf32    fp32    fp32    8       0       0       0       16      8       
```

---

## 2. Capability Analysis

### Input Data Types Supported by B60 XMX
| Precision | `atype` / `btype` | Accumulator / Output (`ctype` / `dtype`) | Valid Tile Dimensions ($M \times N \times K$) |
|---|---|---|---|
| **INT8** | `uint8`, `sint8` | `sint32` | Dynamic $M \le 8, N=16, K=32$ |
| **FP16** | `fp16` | `fp32`, `fp16` | Dynamic $M \le 8, N=16, K=16$; Static $16 \times 16 \times 16$, $1 \times 64 \times 16$, $32 \times 64 \times 16$, $1 \times 64 \times 32$, $32 \times 64 \times 32$ |
| **BF16** | `bf16` | `fp32`, `bf16` | Dynamic $M \le 8, N=16, K=16$; Static $16 \times 16 \times 16$, $1 \times 64 \times 16$, $32 \times 64 \times 16$, $1 \times 64 \times 32$, $32 \times 64 \times 32$ |
| **TF32** | `tf32` | `fp32` | Dynamic $M \le 8, N=16, K=8$ |

### ABSENT Precisions
- **NO INT4 / UINT4**: The Intel Arc Pro B60 XMX systolic arrays have **no native INT4 input mode**.
- **NO FP8 (E4M3 / E5M2)**: Neither FP8 variant is exposed through SYCL Joint Matrix on Xe2-HPG.

---

## 3. Structural Implication for Qwen3.8-27B INT4 Inference

1. **Single-Token Decode ($M=1$) is Bound by GDDR6 Bandwidth, Not Compute**:
   - Model weights occupy **15.77 GB** in INT4 format.
   - For an $M=1$ projection $Y = X \cdot W^T$, arithmetic intensity is:
     $$\text{Arithmetic Intensity} \approx \frac{2 \cdot 1 \cdot N \cdot K}{(N \cdot K / 2) \text{ bytes}} = 4.0 \text{ FLOP/byte}$$
   - At the B60 peak bandwidth of 456 GB/s, streaming 15.77 GB requires:
     $$T_{\text{min}} = \frac{15.77 \text{ GB}}{456 \text{ GB/s}} = 34.58 \text{ ms}$$
   - The required compute rate is only:
     $$\text{Compute Rate} = 456 \text{ GB/s} \times 4.0 \text{ FLOP/byte} = 1.82 \text{ TFLOPS}$$
   - The Arc Pro B60's 20 Xe-cores provide **~15–20 TFLOPS of FP32 Vector Engine throughput**. Thus, the Vector Engines are operating at **under 10% of their compute peak** during decode.
   - Compute is completely idle waiting on memory; XMX matrix compute throughput (~100+ TFLOPS) is completely irrelevant for $M=1$ decode.

2. **Dequantization to VRAM is Structurally Impossible**:
   - Dequantizing INT4 weights (15.77 GB) to INT8 in VRAM would require **31.54 GB**.
   - Dequantizing to FP16/BF16 would require **63.08 GB**.
   - The B60 has **24 GB GDDR6 total**. Expanding weights in memory causes immediate Out-Of-Memory (OOM).

3. **On-the-Fly Unpacking into SLM / Registers for XMX Overhead**:
   - Streaming INT4 weights from DRAM into SLM, unpacking 4-bit nibbles to FP16 in SLM, executing `joint_matrix_load` + `joint_matrix_mad`, and synchronizing with barriers adds:
     1. Extra SLM round-trip latency.
     2. Workgroup barrier stalls (`item.barrier()`).
     3. SLM bank conflict serialization.
   - It does not save a single byte of DRAM traffic (which is the actual 99% bottleneck).
   - In contrast, the Vector Engine SIMD16 cooperative GEMV (`linear_int4` in `src/ops/linear.cpp`) dequantizes INT4 directly in registers via single-cycle bit-shift arithmetic (`sp0 << 28 >> 28`) and computes FMA concurrently with memory coalesced loads, without any SLM barriers or memory roundtrips.

4. **Where XMX Actually Belongs**:
   - High batch sizes ($M \ge 16$) or prompt prefill ($M = \text{seq\_len} \ge 64$), where operational intensity is $2 \cdot M \text{ FLOP/byte} \gg 100 \text{ FLOP/byte}$, transitioning the kernel from memory-bandwidth bound to compute-bound.

---

## 4. Empirical Microbenchmark: Vector Engine GEMV vs. INT4-Unpack+XMX

- **Benchmark Source:** `tools/parity/test_int4_xmx_vs_gemv.cpp`
- **Test Workload:** Qwen3.8-27B MLP intermediate projection ($M = 1, K = 5120, N = 17408$), group size 128, 42.5 MB weight footprint.
- **Hardware:** Intel Arc Pro B60 Graphics (24 GB GDDR6, 456 GB/s peak).

### Measured Hardware Performance:
| Implementation | Execution Latency | Achieved Bandwidth | Relative Speed | Status |
|---|---|---|---|---|
| **Vector Engine SIMD16 GEMV** (`linear_int4`) | **0.120 ms** | **383.7 GB/s** (84.1% of peak) | **1.0x (Baseline)** | **Production Kernel** |
| **INT4-Unpack-to-SLM + XMX Joint Matrix** ($1 \times 64 \times 16$) | **0.902 ms** | **51.0 GB/s** (11.2% of peak) | **7.5x SLOWER** | **Architecturally Counterproductive** |

### Architectural Conclusion:
Attempting to feed systolic XMX engines during $M=1$ INT4 decode destroys performance. The overhead of cooperative unpacking in SLM and barrier synchronization reduces memory bandwidth utilization from 84% down to 11%, increasing latency by **7.5x**. The sub-group cooperative SIMD16 Vector Engine GEMV with register bit-shift dequantization is the mathematically and physically optimal architecture for single-token decode on Intel Arc Pro B60.
