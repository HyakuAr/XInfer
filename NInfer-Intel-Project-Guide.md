# NInfer-Inspired Specialized Inference Engine for Intel Arc Pro B60

> **Project Codename (suggested):** `xinfer` / `arcinfer` / `b60infer`  
> **Goal:** Maximum single-GPU inference performance for selected model checkpoints on Intel Arc Pro B60, following the extreme specialization philosophy of Neroued/ninfer.

---

## 1. Project Vision & Philosophy

**Inspiration:** [Neroued/ninfer](https://github.com/Neroued/ninfer)

NInfer achieves extreme performance by *refusing to be general*:
- One specific GPU architecture only (RTX 5090 / `sm_120a`)
- A tiny set of explicitly registered model checkpoints
- Custom quantized artifact format (`.ninfer`)
- Hand-written, heavily tuned CUDA kernels + CUDA Graphs
- Fixed residency, startup-fixed concurrency (1–8 requests)
- No multi-GPU, no dynamic continuous batching of arbitrary size, no weight offloading

**This project applies the exact same philosophy to Intel hardware.**

### Core Constraints (Non-Negotiable)
- **Target GPU only:** Intel Arc Pro B60 (Battlemage / Xe2, 24 GB GDDR6)
- **Target model family only:** Qwen3.8-27B (and later carefully chosen variants)
- **Primary quant:** INT4 (group size 64 or 128), custom layout optimized for XMX
- **One resident model** per process
- **Startup-fixed capacity** (e.g. 1–4 or 1–8 active requests)
- From-scratch high-performance engine (C++ + SYCL / Level Zero)
- Custom binary artifact format (equivalent of `.ninfer`)

The product boundary stays intentionally small. Generality is explicitly rejected in favor of peak performance on this exact combination.

---

## 2. Target Hardware

### Intel Arc Pro B60 (Battlemage / Xe2-HPG)

| Spec                        | Value                  |
|----------------------------|------------------------|
| Architecture               | Xe2 (Battlemage)      |
| Xe-cores                   | 20                    |
| XMX Engines                | 160                   |
| Xe Vector Engines          | 160                   |
| Peak INT8 TOPS             | 197                   |
| Peak FP32                  | 12.28 TFLOPS          |
| Memory                     | 24 GB GDDR6           |
| Memory Bus                 | 192-bit               |
| Memory Bandwidth           | 456 GB/s              |
| TBP                        | 120–200 W             |
| PCIe                       | 5.0 x8                |
| Device ID                  | 0xE211                |

**Why this card?**
- 24 GB is enough for a clean single-GPU resident Qwen3.8-27B INT4 (~17–19 GB weights) + useful KV cache + runtime buffers.
- Dedicated XMX matrix engines designed for AI workloads.
- Workstation-class drivers and multi-GPU Linux readiness.
- Good cost/performance entry point for specialized local inference.

---

## 3. Target Model

- **Base model:** Official `Qwen/Qwen3.8-27B` (BF16 / full precision from Hugging Face)
- **Quantization target:** High-quality INT4 (symmetric or asymmetric, group size 64/128)
- **Artifact:** Single self-contained binary (weights + tokenizer + chat template + optional vision frontend + speculative decoding heads)
- **Context:** Aim for practical long context (tens of thousands of tokens initially; push toward native 262k where memory allows with aggressive KV quantization)

Optional later extensions (only after the core is excellent):
- Qwen3.6-27B / 35B-A3B variants
- Multimodal (vision) path
- Speculative decoding (MTP-style)

---

## 4. Technology Stack

### Programming Model
- **Primary language:** Modern C++20/23
- **GPU programming:** **SYCL** (with Level Zero backend)
- **Low-level control:** Intel Level Zero API (closest equivalent to CUDA Driver API)
- **Build system:** CMake + Ninja
- **Compiler:** Intel oneAPI DPC++/C++ Compiler

### Why SYCL + Level Zero?
SYCL is pure modern C++ (single-source). When targeting the Level Zero backend on Intel GPUs it is as close to the metal as CUDA is on NVIDIA. You can drop down to raw Level Zero for command lists, fine-grained memory control, and maximum performance.

### Key Intel Components
| Component                      | Role                                      |
|--------------------------------|-------------------------------------------|
| oneAPI Base Toolkit            | Compilers, libraries, tools               |
| Intel oneAPI DPC++ Compiler    | SYCL compilation                          |
| Level Zero                     | Low-level GPU runtime                     |
| Intel Graphics Compute Runtime | Actual GPU driver/runtime                 |
| OpenVINO / NNCF                | Quantization & reference validation       |
| oneDNN                         | Highly optimized primitives (study source)|
| XeTLA                          | Template library for high-perf GEMM etc.  |
| Intel VTune / Advisor          | Profiling & roofline analysis             |

---

## 5. Required Documentation (Deep Study List)

### Hardware & Product
- [Arc Pro B60 Specifications](https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html)
- [B60 Datasheet (PDF)](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2026-03/datasheet-b60-gpu.pdf)
- [B60 GPU Management User Guide](https://cdrdv2-public.intel.com/860316/860316_IntelArcProB60GPUManagementUserGuide_IntelXeonPlatform_1.4.pdf)

### Architecture & XMX (Critical)
- [oneAPI GPU Optimization Guide (main living document)](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/overview.html)
  - Especially: Intel Xe GPU Architecture, Thread Mapping & Occupancy, Memory hierarchy
- [Programming Intel XMX using SYCL Joint Matrix Extension](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/programming-intel-xmx-using-sycl-joint-matrix.html)
- [What is Xe Matrix Extensions (XMX)?](https://www.intel.com/content/www/us/en/support/articles/000091112/graphics.html)
- [XeTLA (GitHub) – GEMM construction guide](https://github.com/intel/xetla/blob/main/media/docs/construct_a_gemm.md)

### Programming Model
- [Level Zero Specification (latest)](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/)
- [Level Zero Core API](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api.html)
- [Level Zero GitHub (headers + loader)](https://github.com/oneapi-src/level-zero)
- SYCL 2020 Specification (Khronos)

### Drivers & Installation
- [Intel discrete GPU software installation guides](https://dgpu-docs.intel.com/installation-guides/index.html)  
  (Use the “Intel Open Middleware Xe” path for Arc Pro B-series)

### Supporting Tools
- Intel Advisor (roofline, occupancy, offload modeling)
- Intel VTune Profiler
- oneDNN documentation & source (for reference kernels)

---

## 6. High-Level Architecture (Inspired by NInfer)

```
Gateway / HTTP Frontend (OpenAI-compatible)
        ↓
Frontend (prompt processing, tokenizer, chat template, multimodal)
        ↓
Engine (request lifecycle, scheduling, capacity management)
        ↓
Program / Runtime (physical execution, KV cache, checkpoints)
        ↓
Hand-written SYCL / Level Zero Kernels (attention, linear, norms, RoPE, MoE if any, speculative decode)
```

### Key Design Decisions to Mirror
- Custom binary artifact that embeds everything needed for the registered model
- Startup-time materialization of weights into optimal GPU layout
- Device + Host checkpointing / prefix reuse for long context
- Fixed concurrency lanes decided at startup
- Speculative decoding (MTP-style) as a first-class citizen once core decode is fast
- Aggressive use of XMX via Joint Matrix or XeTLA-style kernels
- CUDA-Graph equivalent → Level Zero command lists / graphs for the decode path

---

## 7. Development Phases (Suggested)

### Phase 0 – Foundations
- Set up oneAPI + Level Zero + Arc Pro B60 development environment
- Study the Optimization Guide + XMX Joint Matrix deeply
- Write and profile small GEMM / attention micro-kernels
- Build a minimal weight loader + custom INT4 layout experiment

### Phase 1 – Core Engine Skeleton
- Artifact format design + converter from official Qwen3.8-27B
- Basic Engine with single-request greedy decode
- KV cache (start with BF16/FP16, then add lower precision)
- Simple OpenAI-compatible HTTP endpoint

### Phase 2 – Performance Specialization
- Hand-tuned attention + linear kernels using XMX
- Prefill chunking + CUDA-Graph-style decode path
- Memory residency planner
- Speculative decoding (MTP)

### Phase 3 – Production Polish
- Prefix caching / checkpoint reuse
- Robust serving (streaming, tools, etc.)
- Evaluation harness + quality validation after quantization
- Documentation + reproducible benchmarks

---

## 8. Resource Checklist

### Must Have
- [ ] Official `Qwen/Qwen3.8-27B` weights
- [ ] At least one Arc Pro B60 (24 GB)
- [ ] Intel oneAPI Base Toolkit + DPC++ compiler
- [ ] Up-to-date Arc Pro drivers (Middleware Xe path)
- [ ] Strong host CPU + 64–128 GB+ system RAM
- [ ] NNCF / quantization tooling

### Strongly Recommended
- [ ] Second B60 or dual-B60 card for later multi-GPU experiments
- [ ] Intel VTune + Advisor
- [ ] oneDNN + XeTLA source for reference
- [ ] Fast local storage for model conversion iteration

---

## 9. Success Metrics (Inspired by NInfer)

The project is successful when, on a single Arc Pro B60 with Qwen3.8-27B INT4:

- Single-stream decode significantly beats current general frameworks (vLLM-XPU, SGLang-XPU, llama.cpp SYCL, OpenVINO) on the same hardware
- Prefill rates are competitive given the 456 GB/s bandwidth
- Practical long-context performance is usable
- The engine is stable for continuous local serving
- Quality after custom quantization remains high (measured with standard eval suites)

Absolute numbers will be lower than NInfer on RTX 5090 (different hardware class), but the *relative* achievement — “maximum possible performance for this model on this specific Intel GPU” — is the goal.

---

## 10. Important Notes for AI-Assisted Development

When working with AI coding assistants on this project:

1. Always prefer **Level Zero backend** for maximum performance.
2. Treat XMX utilization and memory bandwidth as first-class constraints.
3. Study existing high-quality kernels in oneDNN and XeTLA before reinventing.
4. Keep the product boundary narrow — resist feature creep.
5. Every kernel should be profiled with real occupancy / roofline data.
6. The custom artifact format and converter are as important as the runtime kernels.
7. Document numeric formats, memory layouts, and kernel contracts rigorously (NInfer style).

---

## 11. License & Origin

This project is an independent effort inspired by the design philosophy of [Neroued/ninfer](https://github.com/Neroued/ninfer) (Apache 2.0).  
It is **not** a port of NInfer’s CUDA codebase. All low-level kernels and the runtime must be written for the Intel stack.

Base model weights remain under their original Qwen license.

---

**Document version:** 1.0  
**Created from full project discussion:** September 2026  
**Primary target:** Arc Pro B60 + Qwen3.8-27B INT4 + SYCL/Level Zero specialization
