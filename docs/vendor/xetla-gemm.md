# Vendor note: XeTLA — Constructing a High-Performance GEMM

- **Source:** https://github.com/intel/xetla/blob/main/media/docs/construct_a_gemm.md
- **Extracted:** 2026-09-15

## Important caveat found during extraction — flag to the user, don't just proceed

**The `intel/xetla` repository is archived (read-only) as of 2024-12-18.**
This wasn't visible from the guide's link alone. This means:
- No further updates, bug fixes, or Battlemage/Xe2-specific additions should
  be expected from this repo going forward.
- Before relying on XeTLA as a reference for M7 kernel structure, confirm
  whether it actually has good Xe2/Battlemage-tuned configurations, or
  whether it's PVC/Xe-HPC-centric (older architecture) — the doc's examples
  don't specify B60/Battlemage-specific tuning.
- Treat XeTLA here as a **conceptual reference for GEMM decomposition
  patterns** (workgroup/subgroup tiling, splitK), not as a library you
  should plan to link against for a maintained/production dependency,
  unless the user explicitly decides otherwise.

## Conceptual GEMM decomposition (the actual reusable content)

XeTLA's approach, independent of whether you use the library itself:

1. The output matrix C is divided into per-workgroup tiles (`wg_tile_m` ×
   `wg_tile_n`), e.g. 256×256.
2. Each workgroup's tile is further divided into per-subgroup tiles
   (`sg_tile_m` × `sg_tile_n`), e.g. 32×64 — one subgroup per hardware
   thread.
3. Subgroup tile operations map to hardware instructions like 2D loads and
   `mma` (matrix multiply-accumulate — the DPAS operation, same underlying
   hardware op the SYCL Joint Matrix note describes).

## SplitK — for GEMM shapes with small M/N but large K

Relevant because attention/decode GEMMs are often exactly this shape
(skinny output, long reduction dimension). If the natural workgroup tiling
produces too few workgroups to use the GPU well (e.g. output shape 256×256
with a 256×256 workgroup tile → only 1 workgroup), split the K dimension
across workgroups instead:

- **Workgroup-level splitK:** each workgroup computes a partial sum over
  a K-slice; partial results across workgroups are combined with
  `atomic_add` — **this constrains the output accumulator to float32**,
  since atomic add doesn't support float16/bfloat16. Relevant constraint if
  a decode kernel's accumulator is meant to stay in a lower precision.
- **Subgroup-level splitK:** accumulates via shared local memory within one
  workgroup instead of a global atomic — supports half-precision output
  since there's no cross-workgroup atomic involved.
- Both can be combined (set both a global and local split factor).

## Structural building blocks (API shape, if using XeTLA directly)

- `gemm_selector_t<...>` — selects the actual GEMM implementation given:
  data types for A/B, memory layout (row/col-major), memory space
  (global/local), buffer alignment, accumulator dtype, tile shape,
  `mma_engine::xmx` (compute engine choice), `gpu_arch::Xe`/`Xe2` (**must
  specify the correct arch enum for Battlemage — verify whether `gpu_arch`
  has an explicit Xe2/Battlemage value or whether Xe2 falls under a shared
  enum value; this doc's examples only show `gpu_arch::Xe`**).
- `epilogue_t<...>` — fuses post-GEMM elementwise ops (relu, bias-add, or a
  custom op) directly at the register level, avoiding an extra memory
  round-trip — directly relevant for fusing RMSNorm/activation into a
  linear-layer kernel.
- `gemm_universal_t<dispatch_policy, gemm_t, epilogue_t>` — the assembled
  kernel; invoked as `gemm_op(item, arg)` inside a SYCL kernel body.

## What this note does NOT cover

- Whether XeTLA's `gpu_arch` enum has Xe2/Battlemage-specific tuning
  tables, or whether B60 falls back to generic Xe-HPG tuning — **check the
  actual XeTLA source (`third_party` if vendored, or the archived repo)
  before assuming performance parity with PVC examples shown here.**
- Streaming-K (`streamK`) algorithm details — the doc mentions it exists as
  an alternative to splitK but this note doesn't cover its mechanics; fetch
  the relevant section specifically if splitK proves insufficient.
- Whether to vendor XeTLA as a dependency at all, given the archived status
  — this is a decision for the user, not something to default into.
