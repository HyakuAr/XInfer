# Vendor note: Thread Mapping and GPU Occupancy

- **Source:** https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/thread-mapping-and-gpu-occupancy.html
- **Doc version:** oneAPI GPU Optimization Guide 2025.2, page last modified 2025-07-10
- **Extracted:** 2026-09-15

## Mapping (memorize this, it's used everywhere)

- **work-item** → one SIMD lane
- **sub-group** (16 or 32 consecutive work-items) → one Vector Engine
  hardware thread
- **work-group** → executes on a single Xe-Core (never split across cores)
- **nd_range** → the grid of work-groups; barriers only synchronize within
  one work-group

SYCL ↔ CUDA rough equivalence, if that mapping is ever useful for
cross-checking a ported kernel's intent: work-item ≈ CUDA thread, work-group
≈ thread block, sub-group (8/16/32) ≈ warp (CUDA warp is fixed at 32).

## The two resources that actually limit occupancy

1. **Vector Engine hardware threads per Xe-Core** — for the B60 (Xe2-HPG):
   8 Vector Engines × 8 hardware threads = **64 hardware threads per
   Xe-Core** (confirmed value from the architecture note, not assumed).
2. **Shared Local Memory (SLM) per Xe-Core** — **128 KB** for the B60.

Using the **large register file mode** cuts Xe-Core occupancy in half — this
directly trades off against the XMX tuning note's recommendation to use
large-register mode for GEMM kernels (`docs/vendor/xmx-joint-matrix.md`).
That's a genuine tension to measure, not assume: large-register mode may
still win on a compute-bound GEMM despite lower occupancy, but confirm with
profiling rather than assuming it always helps.

## Formulas (exact, from the doc — use these, don't approximate)

```
threads_per_work_group = work_group_size / sub_group_size
xe_core_utilization    = threads_per_work_group / max_hw_threads_per_xe_core
```

For the B60, `max_hw_threads_per_xe_core = 64`.

Worked example from the doc (using the Xe-HPC part, but the formula is
architecture-independent — substitute 64 for B60's thread count):
work-group size 1024, sub-group size 32 → 32 threads/work-group →
utilization = 32/64 = 50% for one work-group, 100% if 2 work-groups
co-reside on the same Xe-Core.

**Constraint to respect:** all work-items in a work-group must fit on one
Xe-Core, and the global range must divide evenly by the chosen work-group
size in each dimension — a work-group size that doesn't evenly divide the
global range fails to launch (not just runs slower).

## SLM allocation rounding — don't assume requested == allocated

The GPU may round an SLM request up to a fixed set of allocation sizes
(documented for the Xe-HPC example: 2, 4, 8, 16, 24, 32, 48, 64, 96, 128 KB
— **verify the equivalent rounding table for B60/Battlemage** rather than
reusing the Xe-HPC numbers, since this note doesn't confirm they're
identical across generations). This matters for KV-cache / attention
kernels that request SLM per work-group: a request just over a rounding
boundary can silently halve occupancy by blocking a second work-group from
co-residing.

## Practical sizing procedure (for M3/M4 kernels)

1. Query `max_work_group_size` and `sub_group_sizes` at runtime (see
   `docs/vendor/xe-gpu-architecture.md`) — don't hardcode B60 numbers as
   literals in case of a driver-imposed cap different from the spec sheet.
2. Prefer the largest work-group size that evenly divides your problem's
   global range — fewer work-group dispatches is generally better.
3. Try each supported sub-group size (16 and 32 confirmed for Xe2-HPG) and
   measure — larger sub-groups reduce thread count needed per work-group
   but increase register pressure per thread.
4. If using SLM, compute utilization with the formula above accounting for
   the rounding table, and check whether 100% occupancy with less SLM beats
   50% occupancy with more SLM for your specific kernel — the doc explicitly
   states higher occupancy does not always mean higher performance,
   especially for memory-bound kernels that benefit from more SLM at the
   cost of fewer co-resident work-groups.
5. Verify against the [Intel GPU Occupancy Calculator](https://oneapi-src.github.io/oneAPI-samples/Tools/GPU-Occupancy-Calculator/)
   before trusting hand calculations, and confirm actual (not just
   theoretical) occupancy with VTune or `unitrace` once a kernel is real.

## What this note does NOT cover

- Register-spill diagnosis specifics (separate "Registers and Performance"
  page in the same guide — fetch only if register pressure becomes a live
  issue during M4/M7).
- Multi-Xe-Core / multi-stack scheduling — not relevant, B60 is single-stack.
