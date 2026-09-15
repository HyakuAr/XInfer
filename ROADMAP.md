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

## M2 — Model Conversion: HF Qwen3.8-27B → `.xinfer` (INT4 weights)

**Goal:** produce a real `.xinfer` artifact from the official BF16 checkpoint,
with INT4-quantized weights, verified numerically — still no runtime
execution yet.

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
- [ ] Converter script/tool in `tools/convert/qwen3_8_27b/`.
- [ ] A real `.xinfer` file exists for Qwen3.8-27B.
- [ ] Dequantization parity check passes within tolerance, results recorded.
- [ ] File size and section layout match the container spec from M1.

---

## M3 — Core Device Layer (SYCL / Level Zero primitives)

**Goal:** the reusable device building blocks every kernel and the engine
will sit on top of — still no model math.

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
- [ ] `src/core/` contains tensor/view, arena, and command-list wrapper.
- [ ] Unit tests pass on the real B60 device.
- [ ] No model-specific code in `src/core/`.

---

## M4 — Naive Correctness Kernels (no XMX yet)

**Goal:** straightforward, unoptimized SYCL kernels for every op the model
needs, correct before fast.

**Steps:**
1. Implement naive GEMM/linear (no Joint Matrix / XMX).
2. Implement RMSNorm, RoPE, elementwise activation (SiLU or whatever
   Qwen3.8 uses), and softmax.
3. Implement naive (non-fused) scaled-dot-product attention for a single
   sequence.
4. For each op, write a numerical oracle test (naive FP32 CPU reference)
   per `AGENTS.md` §7, run at realistic Qwen3.8-27B shapes.

**DoD:**
- [ ] Every op needed for one forward pass exists in `src/ops/`, naive only.
- [ ] Each op has a passing oracle-comparison test.
- [ ] No performance tuning yet — correctness only.

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
- [ ] `apps/xinfer --prompt "..."` produces coherent text on the B60.
- [ ] End-to-end run completes without crashes for at least a few hundred
      generated tokens.
- [ ] Engine interface documented in `include/xinfer/engine.h`.

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
- [ ] KV cache implemented and used by the decode path.
- [ ] Chunked prefill works for prompts longer than one chunk.
- [ ] Documented max practical context length at this stage, with memory
      accounting.

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
- [ ] XMX GEMM and attention kernels pass the same oracle tests as M4.
- [ ] Measured, documented speedup over the naive baseline at end-to-end
      decode level, not just microbenchmark level.
- [ ] Naive kernels removed or clearly marked as reference-only (per
      `AGENTS.md` §1 on not preserving superseded internal paths).

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
- [ ] Decode step runs via captured/replayed command list.
- [ ] Measured latency improvement documented.
- [ ] Address-stability assumptions (per `AGENTS.md` §7) documented for the
      captured graph.

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
- [ ] `curl` against `/v1/chat/completions` returns a correct, well-formed
      response for a simple prompt.
- [ ] Serving layer contains no inference logic — it only calls the Engine.

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
