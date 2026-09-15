# Vendor note: Programming Intel XMX Using SYCL Joint Matrix Extension

- **Source:** https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/programming-intel-xmx-using-sycl-joint-matrix.html
- **Doc version:** oneAPI GPU Optimization Guide 2025.2, page last modified 2025-07-10
- **Extracted:** 2026-09-15 by Claude, for the xinfer project. This is a
  paraphrased technical summary, not a copy of the page — treat the source
  URL as authoritative if anything here is ambiguous, and re-fetch it if the
  doc has since been revised.
- **Full extension spec (more detail than the guide page):**
  https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc
- **Worked examples repo:** https://github.com/intel/llvm/tree/sycl/sycl/test-e2e/Matrix

## What Joint Matrix is

Joint Matrix is a SYCL extension for programming matrix-multiply hardware
generically — it targets Intel AMX (CPU), Intel XMX (GPU, i.e. our B60), and
NVIDIA Tensor Cores through one API. It sits below framework-level libraries
like oneDNN: lower abstraction, but you keep control over fusion and custom
op structure, which matters for a hand-tuned decode kernel.

On Intel XMX, the underlying hardware op is called **DPAS** (Dot Product
and Accumulate Systolic) — this is the name that will show up in profiler
output (VTune/Advisor) when a Joint Matrix kernel is actually hitting XMX.

## Required precondition — check before assuming XMX is usable

There is **no emulation or CPU fallback** for Joint Matrix. Before writing
any Joint Matrix kernel, confirm the device actually exposes the capability:

```
sycl-ls --verbose
```

and check for the aspect `ext_intel_matrix` in the output. If it's absent,
Joint Matrix code will not run on that device at all — don't assume the B60
has it; verify it, per `AGENTS.md` §0.

## API surface (function/type names — cite these exactly in code, don't
invent similar-sounding ones)

- `joint_matrix` — the matrix type itself, templated on scope
  (`sycl::sub_group`), element type, `use` (`use::a`, `use::b`,
  `use::accumulator`), tile dimensions, and layout (e.g.
  `layout::row_major`).
- `joint_matrix_load` / `joint_matrix_store` — explicit memory ops moving
  data between global memory and the matrix registers.
- `joint_matrix_load_checked` / `joint_matrix_store_checked` — bounds-checked
  variants, for tiles that may run off the edge of a non-tile-aligned
  matrix (relevant for irregular sequence-length attention tiles).
- `joint_matrix_fill` — initializes a matrix to a constant (e.g. zero the
  accumulator before a k-loop).
- `joint_matrix_mad` — the actual multiply-accumulate: `C += A * B`.
- `joint_matrix_prefetch` — prefetches data from global memory to cache
  ahead of use.
- `joint_matrix_apply` — applies an elementwise lambda directly to the
  matrix while its data is still in registers (no extra memory round-trip).
  A coordinate-aware overload also exists in the
  `sycl::ext::intel::matrix` namespace, passing `(value, row, col)` to the
  callback — relevant if a kernel needs per-position behavior (e.g. RoPE
  applied inside a fused attention kernel).

## Kernel shape (paraphrased structure, not a copy of the guide's example)

A joint-matrix GEMM kernel is written per sub-group, not per work-item — all
work-items in a sub-group cooperate on one set of `joint_matrix_load` /
`joint_matrix_mad` / `joint_matrix_store` calls with no per-work-item
branching. The typical loop shape: declare `sub_a`, `sub_b`, `sub_c` tiles;
zero `sub_c` with `joint_matrix_fill`; loop over the K dimension in tile-sized
(`TK`) steps, loading `A`/`B` tiles and accumulating with `joint_matrix_mad`;
optionally post-process `sub_c` with `joint_matrix_apply` (e.g. a fused
activation); then `joint_matrix_store` the result.

To discover which tile shapes (`TM`/`TN`/`TK`) a given device actually
supports, query at runtime rather than hardcoding a shape from a different
GPU generation's example:

```
sycl::ext::oneapi::experimental::matrix::combination
```
via `device::get_info<...::matrix_combinations>()`. The guide's own example
branches on `combinations[i].nsize` to distinguish Intel AMX vs. PVC
(`nsize == 16`) vs. DG2-class GPUs (`nsize == 8`) — B60 is Xe2/Battlemage,
not DG2 or PVC, so **do not assume the DG2 (`nsize == 8`) shapes apply** —
query the actual supported combinations on B60 rather than reusing the
guide's PVC or DG2 numbers.

## Performance tuning notes (for Milestone 7, not earlier)

- **Sub-group partitioning:** one sub-group can issue multiple `joint_matrix_mad`
  calls reusing loaded tiles — the guide's suggested combination for good
  register utilization is a 32×64×16 effective tile achieved either by
  issuing 16 MAD ops reusing 4 A-tiles/4 B-tiles/16 C-tiles, or by using that
  shape directly if the device supports it.
- **Work-group partitioning / cache blocking:** block the i/j/k GEMM loop
  dimensions for cache locality; i/j blocking shows up in the kernel's
  global range, k-blocking happens inside the kernel body.
- **Register file mode:** a large-register-file compiler mode measurably
  helps GEMM kernels. On the PVC-targeted compiler flags in the guide this
  was `-Xsycl-target-backend "-device pvc -options -ze-opt-large-register-file"`
  — confirm the equivalent flag/device string for Battlemage/B60 rather than
  reusing the PVC device string verbatim.
- **Prefetch:** using `joint_matrix_prefetch` gave roughly 30% additional
  throughput on PVC in Intel's own measurement. Treat this as a hypothesis
  to verify on B60, not a guaranteed number — hardware differs.

## What this note does NOT cover

- Exact tile shapes supported on Battlemage/Xe2 (must be queried live per
  above, not assumed from this note).
- oneDNN/XeTLA reference kernel internals — see the separate XeTLA vendor
  note (fetch before Milestone 7 if not already present in `docs/vendor/`).
- Level Zero command-list/graph mechanics — see the Level Zero vendor note.
