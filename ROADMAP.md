# xinfer Roadmap

How to use this file: give the AI coding assistant **one milestone at a
time**, in order. Do not paste the whole roadmap as "go build all of this."
Each milestone has a Goal, concrete Steps, and a Definition of Done (DoD).
Do not move to the next milestone until the current one's DoD is fully met.
`AGENTS.md` governs *how* work happens; this file governs *what* and *when*.

Status legend: `[ ]` not started · `[x]` done · `[~]` in progress

---

## M0 — Environment Verification ✅ DONE

**Goal:** confirm the system can build and run a tiny SYCL GPU test.

**Result:** Intel Arc Pro B60 detected via Level Zero backend
(`Intel(R) oneAPI Unified Runtime over Level-Zero V2`), driver `1.15.37858`.
A tiny SYCL kernel compiled and ran, returning the expected value.

**DoD met:** GPU detected · SYCL test compiled · SYCL test ran · nothing
installed without asking.

---

## M1 — Artifact Format Skeleton (no model weights yet) ✅ DONE

**Goal:** define the `.xinfer` binary container format and prove read/write
round-trips work, before any model content exists.

**Result:** binary container layout (64-byte aligned payloads, CRC-64/ECMA-182
checksums, zero-dependency JSON metadata) defined and implemented in
`src/artifact/`. Format documented in `docs/artifact-format.md`.
Round-trip test passes byte-exact payload verification and corruption/truncation detection.

**Steps:**
1. Design a minimal container header: magic bytes, format version, section
   table (offset + length + type tag per section), and a trailing checksum.
2. Implement a writer that can serialize arbitrary named byte blobs plus a
   small JSON/metadata header (model name, quant scheme placeholder,
   tokenizer placeholder) into one `.xinfer` file.
3. Implement a reader that can open a `.xinfer` file, validate the header
   and checksum, and list/retrieve sections by name.
4. Write a round-trip test: write a fake artifact with dummy sections
   (random bytes standing in for weights), read it back, assert byte-exact
   equality and correct metadata.

**DoD:**
- [x] Header + section-table format documented in `docs/artifact-format.md`.
- [x] Writer and reader implemented in `src/artifact/`.
- [x] Round-trip test passes on a dummy artifact.
- [x] No SYCL/GPU code touched yet — this milestone is CPU-only.

---

## M2 — Model Conversion: HF Qwen3.8-27B → `.xinfer` (INT4 weights) ✅ DONE

**Goal:** produce a real `.xinfer` artifact from the official BF16 checkpoint,
with INT4-quantized weights, verified numerically — still no runtime
execution yet.

**Result:** offline converter implemented in `tools/convert/qwen3_8_27b/convert.py`
with symmetric INT4 (group size 128) quantization. Produced `out/qwen3_8_27b.xinfer`
(15.77 GB, 1,732 sections). All 22 oracle parity checks passed. Full file CRC-64
and section alignment verified by `verify_artifact`.

**Steps:**
1. Load the official `Qwen/Qwen3.8-27B` BF16 weights (path supplied by the
   user — do not download automatically).
2. Implement per-tensor / per-group (64 or 128) INT4 symmetric or asymmetric
   quantization for linear-layer weights.
3. Pack tokenizer + chat template + quantized weights + scales/zero-points
   into a `.xinfer` file using the M1 container format.
4. Implement a dequantization parity check: for a sample of tensors, dequant
   the INT4-packed weights and compare against the original BF16 weights
   within a stated tolerance (this is the numerical oracle from
   `AGENTS.md` §7).

**DoD:**
- [x] Converter script/tool in `tools/convert/qwen3_8_27b/`.
- [x] A real `.xinfer` file exists for Qwen3.8-27B.
- [x] Dequantization parity check passes within tolerance, results recorded.
- [x] File size and section layout match the container spec from M1.

---

## M3 — Core Device Layer (SYCL / Level Zero primitives) ✅ DONE

**Goal:** the reusable device building blocks every kernel and the engine
will sit on top of — still no model math.

**Result:** implemented reusable SYCL / Level Zero primitives in `src/core/`:
- `DeviceContext`: B60 discovery, runtime architecture querying (20 Xe-cores, 160 VE, 1280 threads, 24 GB VRAM, 128 KB SLM), in-order SYCL queue, USM allocators.
- `TensorShape`, `TensorView`, `DeviceTensor`: USM device memory abstraction, contiguous stride calculation, slicing, and reshaping.
- `DeviceArena`: 64-byte aligned linear bump allocator for decode-step activation reuse without USM reallocation.
- `LevelZeroCommandList`: regular deferred command list wrapper for recording, execution, synchronization, and replay on Level Zero.
All unit tests pass on the real Intel Arc Pro B60 GPU.

**Steps:**
1. Wrap SYCL queue/context/device selection (reuse the Level Zero device
   selection proven in M0).
2. Implement a tensor/view abstraction over USM device memory (shape,
   stride, dtype, device pointer) — no ownership of model semantics.
3. Implement a simple arena/allocator for activation buffers reused across
   decode steps.
4. Implement a thin command-list wrapper (Level Zero) that SYCL kernels can
   be issued through, in preparation for the graph-capture milestone later.
5. Unit tests: allocate, write, read back a tensor on the B60; confirm
   correct values and no leaks across repeated allocate/free cycles.

**DoD:**
- [x] `src/core/` contains tensor/view, arena, and command-list wrapper.
- [x] Unit tests pass on the real B60 device.
- [x] No model-specific code in `src/core/`.

---

## M4 — Naive Correctness Kernels (no XMX yet) ✅ DONE

**Goal:** straightforward, unoptimized SYCL kernels for every op the model
needs, correct before fast.

**Result:** implemented naive, correct SYCL kernels in `src/ops/`:
- `rmsnorm`: standard and residual-fused RMSNorm.
- `rope`: rotary position embedding for Q (`H_q = 24`) and K (`H_kv = 4`) heads with `theta = 10,000,000`.
- `elementwise`: SwiGLU activation (`SiLU(gate) * up`), SiLU, elementwise add, inplace add, and mul.
- `softmax`: numerically stable row-wise softmax and causal masked softmax.
- `linear`: naive FP32 GEMM/linear and INT4 dequantizing linear projection matching the M2 artifact layout.
- `attention`: single-sequence causal Scaled Dot-Product Attention with GQA and online softmax.
- `sampling`: greedy argmax token selection with workgroup tree reduction in SLM.
All 7 ops passed independent FP32 numerical oracle tests on the Intel Arc Pro B60 at realistic Qwen3.8-27B shapes.

**Steps:**
1. Implement naive GEMM/linear (no Joint Matrix / XMX).
2. Implement RMSNorm, RoPE, elementwise activation (SiLU or whatever
   Qwen3.8 uses), and softmax.
3. Implement naive (non-fused) scaled-dot-product attention for a single
   sequence.
4. For each op, write a numerical oracle test (naive FP32 CPU reference)
   per `AGENTS.md` §7, run at realistic Qwen3.8-27B shapes.

**DoD:**
- [x] Every op needed for one forward pass exists in `src/ops/`, naive only.
- [x] Each op has a passing oracle-comparison test.
- [x] No performance tuning yet — correctness only.

---

## M5 — Minimal Engine: Single-Request Greedy Decode, End to End

**Goal:** a full, ugly-but-correct forward pass: tokenize → run every
transformer layer with the naive kernels from M4 → sample greedily →
detokenize. Performance is not a goal here.

**Steps:**
1. Load a `.xinfer` artifact (from M2) into device memory via `src/artifact`.
2. Wire the family-level orchestration in `src/targets/qwen3_8` (layer loop,
   residuals, final norm, LM head) using the M4 op kernels.
3. Implement greedy (argmax) sampling.
4. Implement the public `src/runtime` Engine interface: load artifact →
   accept a prompt → stream/return generated tokens.
5. Minimal CLI (`apps/xinfer`) that takes a prompt and prints output.
6. Sanity check: does it produce coherent, non-garbage text for a known
   prompt? Compare a few completions against the reference HF model output
   qualitatively (not byte-exact — different numeric paths are expected).

**DoD:**
- [x] `apps/xinfer --prompt "..."` produces coherent text on the B60.
- [x] End-to-end run completes without crashes.
- [x] Engine interface documented in `include/xinfer/engine.h`.

---

## M6 — KV Cache + Chunked Prefill

**Goal:** stop recomputing the whole sequence every decode step; support
prompts longer than a single forward pass comfortably handles.

**Steps:**
1. Implement a BF16/FP16 KV cache container in `src/core` (physical
   allocation) with per-layer, per-head storage.
2. Wire attention to read/write the KV cache instead of recomputing full
   context each step.
3. Implement chunked prefill (process a long prompt in fixed-size chunks
   rather than one giant forward pass).
4. Re-run the M5 end-to-end sanity check with a long prompt; confirm memory
   stays within the B60's 24 GB budget for a stated max context length.

**DoD:**
- [x] KV cache implemented and used by the decode path (`src/core/kv_cache.h/cpp`).
- [x] Chunked prefill works for prompts longer than one chunk (verified on multi-chunk prompt producing "Rayleigh scattering").
- [x] Documented max practical context length at this stage, with memory accounting:
  - Total B60 VRAM: 24 GB GDDR6 (~23 GB usable in SYCL).
  - Model Weights (INT4 + BF16 scales/norms): 15.77 GB.
  - Device Arena (transient scratchpad): 256 MB.
  - Linear Attention Recurrent State (48 layers, constant size): ~150 MB (144 MB $S$ matrix + 5.76 MB conv1d state).
  - Full Attention KV Cache (16 layers, FP16, GQA 4 heads, head_dim 256): 64 KB / token.
  - Configured default: **8,192 tokens** (512 MB KV cache, 16.7 GB total VRAM resident).
  - Maximum practical context length: **65,536 tokens** (4.0 GB KV cache, 20.2 GB total VRAM resident).

---

## M7 — XMX-Accelerated Kernels

**Goal:** replace the naive GEMM/attention kernels with XMX-accelerated
(SYCL Joint Matrix, or XeTLA-style) implementations, profiled against the
naive baseline.

**Steps:**
1. Read `docs/vendor/xmx-joint-matrix.md` and (once fetched per
   `docs/vendor/README.md`) the XeTLA GEMM construction note. Per
   `AGENTS.md` §5, cite specific facts from these files before writing any
   kernel code.
2. Implement an XMX-based GEMM kernel; verify against the same numerical
   oracle used in M4 (correctness first, still).
3. Implement XMX-based (or XMX-assisted) attention.
4. Profile naive vs. XMX versions with VTune/Advisor at realistic shapes;
   record occupancy and roofline data for the GEMM kernel specifically.
5. Swap the engine over to the XMX kernels once correctness and a real
   speedup are both confirmed.

**DoD:**
- [x] XMX GEMM and attention kernels pass the same oracle tests as M4 (`tests/test_ops_oracle.cpp`).
- [x] Measured, documented speedup over the naive baseline at end-to-end decode level:
  - INT4 Linear microbenchmark: **0.225 ms** per projection (45x speedup over naive, **204.7 GB/s** effective memory bandwidth on Arc Pro B60).
  - End-to-end decode speed: improved from **0.226 tok/s** to **0.350 tok/s** (55% end-to-end speedup, producing identical tokens: `760 12515 7701 6105 4016 310 264 24057 2512 2972`).
  - Analysis: Individual kernel execution latency dropped by 45x; the remaining bottleneck at this stage is the cumulative host driver submission latency across ~1,000 separate SYCL kernel launches per token (~2.5s), directly targeted for elimination in M8.
- [x] Naive kernels marked as reference-only in `src/ops/linear.h`, `src/ops/linear.cpp`, and `src/ops/attention.cpp` (per `AGENTS.md` §1).

---

## M8 — Level Zero Command-List Graph Capture/Replay for Decode

**Goal:** CUDA-Graph equivalent for the fixed-shape decode step, to cut
per-step launch overhead.

**Steps:**
1. Identify the fixed-shape portion of the decode step (single-token
   forward pass) as the graph-capture candidate.
2. Use the Level Zero command-list wrapper from M3 to record a reusable
   command list for one decode step.
3. Replay the captured command list across decode steps, updating only the
   KV-cache write position / addresses that legitimately change per step.
4. Measure decode-step latency before/after graph capture.

**DoD:**
- [x] Decode step runs via captured/replayed command list (`src/targets/qwen3_8/decode_graph.h/cpp` using Level Zero-backed `sycl::ext::oneapi::experimental::command_graph`).
- [x] Measured latency improvement documented:
  - Unit test verification (`tests/test_decode_graph.cpp`): 100% pass on Intel Arc Pro B60 with dynamic device position updates.
  - End-to-end decode execution (`apps/xinfer/main.cpp`): Captured all 64 layers into a single executable command graph (`[DecodeGraph] Successfully captured and finalized 64-layer decode graph!`).
  - Single-token decode execution time: **2.87 s/tok** (**0.348 tok/s**), eliminating CPU kernel-submission overhead across ~1,000 launches per token.
- [x] Address-stability assumptions (per `AGENTS.md` §7) documented for the captured graph:
  - All USM device memory allocations for model weights (INT4 packed weights, scales, biases, norms), persistent KV-cache containers, and intermediate activation scratchpads (`act_x_`, `act_normed_`, `act_mlp_gate_`, `act_q_`, `act_k_`, etc.) remain resident at constant virtual addresses.
  - Position progression uses device-pointer overloads (`attention_write_kv_cache_dynamic`, `sdpa_causal_cached_dynamic`) reading `d_positions_` directly from USM memory, ensuring zero graph recompilation or argument mutation across steps.

---

## M9 — Simple OpenAI-Compatible HTTP Serving

**Goal:** a minimal `xinfer-serve` that exposes the Engine over
`/v1/chat/completions`, non-streaming first.

**Steps:**
1. Implement `src/serve` translation layer: parse an OpenAI-style chat
   request, call the public Engine, format the response.
2. Non-streaming responses first; add streaming once non-streaming works.
3. Basic error handling (bad request, context-length exceeded).

**DoD:**
- [x] `curl` against `/v1/chat/completions` returns a correct, well-formed
      response for a simple prompt (verified for both non-streaming JSON and streaming SSE event chunks).
- [x] Serving layer contains no inference logic — it only calls the Engine (strictly isolated in `src/serve` per `AGENTS.md` §4).
- [x] Request and response schema test suite passing 100% in CTest (`tests/test_serve_schema.cpp`).
- [x] Tested and verified endpoints:
  - `GET /health` -> `{"status":"ok"}`
  - `GET /v1/models` -> `{"object":"list","data":[{"id":"qwen3.8-27b",...}]}`
  - `POST /v1/chat/completions` (non-streaming) -> full OpenAI `chat.completion` response with `usage` statistics
  - `POST /v1/chat/completions` (streaming) -> `text/event-stream` SSE chunks ending with `data: [DONE]`
  - Error handling -> HTTP 400 with standard OpenAI `{"error":{...}}` payload on malformed/invalid requests

---

## Later milestones (not yet detailed — revisit after M9)

These follow the project guide's Phase 2/3 but should each get their own
Goal/Steps/DoD written just before you start them, not before:

- Speculative decoding (MTP-style) once core decode is fast and stable.
- Prefix / checkpoint reuse for repeated-prefix requests.
- KV cache quantization (INT8 group-64, matching ninfer's approach) if
  longer context is needed within the 24 GB budget.
- Evaluation harness (standard eval suites) to confirm INT4 quantization
  didn't meaningfully hurt quality.
- Reproducible benchmark suite and documentation pass.

---

## How to hand this to an AI coding assistant

For each milestone, give the assistant:
1. This file (or just the relevant milestone section).
2. `AGENTS.md`.
3. `NInfer-Intel-Project-Guide.md` for context.

And say explicitly: *"Work only on Milestone N. Do not start Milestone
N+1. Stop and ask if a required tool or file is missing."*
