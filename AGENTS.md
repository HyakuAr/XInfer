# AGENTS.md

These rules apply to the whole `xinfer` repository and govern how an AI coding
assistant works on this project. They are adapted from the ownership/scope
discipline used by `Neroued/ninfer`, retargeted to Intel Arc Pro B60 + SYCL /
Level Zero, and merged with the project owner's explicit operating rules.

---

## 0. Operating rules (non-negotiable, from the project owner)

- Do not install anything automatically.
- Before using a tool, compiler, driver, or library, verify it is already
  installed / available. Never assume.
- If a required tool is missing, **stop and ask the user** — do not attempt
  to install it, download it, or work around its absence.
- Do not add features, kernels, endpoints, or abstractions beyond the scope
  of the **current milestone** in `ROADMAP.md`. If something looks useful
  but isn't in the current milestone, note it and move on.
- Never regress a milestone that has already passed its Definition of Done
  without the user explicitly asking to revisit it.

## 1. Governing objective

Build the strongest coherent single-GPU inference engine for exactly one
target combination: **Intel Arc Pro B60 + Qwen3.8-27B (INT4)**. Optimize for
architectural integrity, clear ownership, functional and numerical
correctness, and maximum performance on that exact combination — not for
generality, portability, broad framework compatibility, or minimal diff size.

Do not optimize for: supporting other GPUs, other model families, dynamic
batching, multi-tenant serving, or backward compatibility with earlier
in-project code. This project has no external users yet; project-owned code,
formats, and CLIs do not need to preserve compatibility across milestones
unless the user says otherwise.

## 2. Scope control

Before doing substantial work, identify: what does the current milestone in
`ROADMAP.md` ask for, and what is its Definition of Done? Only do work that:

- directly produces that milestone's deliverable,
- is necessary to keep the artifact format / engine internally consistent,
- resolves a real uncertainty that would otherwise block the milestone, or
- fixes a concrete regression the current change introduced.

Do not use "future-proofing" as a reason to build abstractions for models,
GPUs, or features that are not the current target. Do not start a later
milestone's work early, even if it seems efficient. Do not silently expand a
"verify the environment" task into "install/upgrade the environment."

## 3. Current product contract

- **Target GPU:** Intel Arc Pro B60 (Battlemage / Xe2-HPG, 24 GB GDDR6,
  device ID `0xE211`) — one GPU, one process, one resident model.
- **Target model:** `Qwen/Qwen3.8-27B`, quantized to INT4 (group size 64 or
  128), custom XMX-friendly layout.
- **Concurrency:** startup-fixed, small (1–8 active requests served via bounded
  worker pool; single resident model forward passes serialized). No dynamic
  continuous batching, no multi-GPU, no weight offloading.
- **Artifact format:** a single self-contained binary format (working name:
  `.xinfer`) analogous to ninfer's `.ninfer` — weights + tokenizer + chat
  template, produced by an offline converter from the official HF checkpoint.
- **Programming model:** C++20/23, SYCL with the Level Zero backend, Intel
  oneAPI DPC++ compiler, CMake + Ninja.

Any change to this contract (different GPU, different model, a second
resident model, general framework support) is an explicit architecture
decision the user must state — it is not something to infer from a task.

## 4. Ownership boundaries

Mirrors ninfer's layering, retargeted to the Intel stack. Each directory has
exactly one job; no layer reaches into another's private state.

| Path | Owns | Does not own |
|---|---|---|
| `src/core` | SYCL/Level Zero device primitives: USM allocation, tensors/views, arenas, command-list wrappers, KV-cache physical containers, queue/context lifetime | Model math, checkpoint semantics |
| `src/artifact` | Generic `.xinfer` container framing, descriptors, binding, materialization into device memory | Any Qwen-specific execution semantics |
| `src/ops` | Every semantically closed kernel: attention, linear/GEMM (naive and XMX/Joint-Matrix), RMSNorm, RoPE, elementwise, sampling. Ownership follows the math contract, not the first caller. | Model orchestration, scheduling |
| `src/targets/qwen3_8` | Family invariants: tokenizer, chat template, prompt construction, weight-view schema, the planning/Program/decode algorithms shared by any Qwen3.8 variant | Target identity/registry entry, artifact binder for one exact checkpoint |
| `src/targets/qwen3_8_27b` | The exact checkpoint package: storage profile, binder, `LoadedModel`, populated model-view values | Program/schedule/state-transaction algorithms (those live in the family) |
| `src/runtime` | Public Engine interface (PIMPL), request lifecycle, generated-token transaction policy | Model mathematics, target state |
| `src/serve` | Protocol translation (OpenAI-compatible HTTP) and transport only | Inference logic — calls the public Engine only |
| `tools/convert` | Offline HF checkpoint → `.xinfer` converter, INT4 quantization | Runtime execution |
| `tools/parity` | Numerical parity checks against a reference (e.g. OpenVINO/NNCF or naive FP32) | Product code paths |

Do not add a generic model-graph abstraction, plugin discovery, string-driven
dispatch, or a second artifact format "for flexibility." If a future
milestone genuinely needs a second target, extend `src/targets/` the way
ninfer did — a peer variant reusing the family layer — not a rewrite.

## 5. Grounding requirement for vendor-specific code

Before writing any code that depends on a specific Intel API, flag, aspect
string, or numeric limit (SYCL Joint Matrix, Level Zero, XeTLA, driver
behavior), you must first open the corresponding local file in
`docs/vendor/` and cite the specific fact you're using, in the same response,
before the code. See `docs/vendor/README.md` for the full rule and the list
of what's already extracted vs. what still needs fetching.

If the relevant `docs/vendor/*.md` file doesn't exist yet, stop and say so —
do not write the code from memory and present it as if it were checked
against documentation. This applies even when the general shape of the code
"looks right" from training knowledge; APIs like Joint Matrix are new and
narrow enough that plausible-looking code is frequently wrong in specifics
(exact function names, required device aspects, tile-shape assumptions that
don't hold on Battlemage/Xe2).

## 6. Sources of truth

Read only what's relevant to the task at hand — this is a routing map, not a
reading list to consume in full every time.

- `NInfer-Intel-Project-Guide.md` — project vision, hardware, model,
  phases, success metrics.
- `ROADMAP.md` — the current milestone and its Definition of Done.
- `AGENTS.md` (this file) — ownership boundaries and operating rules.
- [oneAPI GPU Optimization Guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/overview.html) — Xe architecture, memory hierarchy, occupancy.
- [Programming Intel XMX via SYCL Joint Matrix](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/programming-intel-xmx-using-sycl-joint-matrix.html) — once a milestone reaches XMX kernels.
- [Level Zero Specification](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/) — once a milestone needs command-list/graph control below SYCL.
- [XeTLA GEMM construction guide](https://github.com/intel/xetla/blob/main/media/docs/construct_a_gemm.md) — reference for high-performance GEMM structure.

## 7. Numerical correctness

Every kernel that touches model math needs one independent oracle:

- A naive FP32 reference implementation of the exact mathematical operation
  (attention, RMSNorm, RoPE, GEMM, quant/dequant), evaluated from the
  represented inputs.
- The oracle does not copy a production kernel's staging casts, reduction
  order, or intermediate dtype. Production kernels are free to choose their
  own intermediate precision as long as the final output meets the criterion
  for that op's precision (exact for exact transforms, tolerance-based for
  floating point).
- For INT4 weight quantization specifically: verify dequantized weights
  against the original BF16 weights (per-group scale/zero-point), not
  against another kernel's output.

## 8. Performance work

- State a performance claim at the level it actually matters: single op,
  decode step, or end-to-end tokens/sec. Measure at that level.
- Use VTune / Advisor roofline analysis only once a specific kernel is
  suspected to be the bottleneck and a profiling answer could change a
  design decision — not as a blanket first step.
- Record enough context to interpret a result (device, driver version,
  build configuration / optimization flags e.g. Release /O3, model/quant
  config, workload) without needing full raw profiler dumps by default.
  Always record the exact build configuration alongside every benchmark number.

## 9. Tests and verification

Add a test only when it protects real behavior: numerical kernel
correctness, `.xinfer` framing/binding round-trip, a real end-to-end decode,
or a reproduced bug. Do not add tests for coverage numbers, trivial
getters, or hypothetical failure modes.

| Change | Relevant evidence |
|---|---|
| SYCL/Level Zero core primitive | targeted unit test + device execution |
| Op kernel | numerical oracle comparison at realistic shapes |
| `.xinfer` reader/writer | round-trip test on a real (small) artifact |
| Converter | parity check against reference dequantization |
| Engine / decode loop | end-to-end token generation sanity check |
| Serving layer | request/response schema test against the OpenAI spec |

## 10. Local environment

Do not assume any of this is installed — verify before use, per Section 0.

| Purpose | Expected location / identity |
|---|---|
| GPU | Intel Arc Pro B60, device ID `0xE211` |
| SYCL/Level Zero runtime | verified present in Milestone 0 |
| Compiler | Intel oneAPI DPC++/C++ Compiler |
| Build | CMake + Ninja, `build/` |
| Model source | official `Qwen/Qwen3.8-27B` BF16 checkpoint (path TBD by user) |
| Artifact output | `out/qwen3_8_27b.xinfer` (once Milestone ≥2) |

## 11. Commits

Create a commit only when the user explicitly asks for one. Use
Conventional Commit style subjects, e.g.:

```
feat(artifact): add .xinfer container header and round-trip test
```

Types: `feat`, `fix`, `perf`, `bench`, `test`, `build`, `refactor`, `docs`,
`chore`.
