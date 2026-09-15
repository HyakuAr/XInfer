# Vendor note: Intel Xe GPU Architecture

- **Source:** https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html
- **Doc version:** oneAPI GPU Optimization Guide 2025.2, page last modified 2025-07-10
- **Extracted:** 2026-09-15

## GPU hierarchy (terms used everywhere else in the guide)

Architecture (Xe) → Generation (Xe, Xe2, Xe3...) → Micro-architecture
(Xe-LP/LPG, Xe-HPG, Xe-HPC, Xe-HP) → Model/SKU (e.g. Arc B580, our B60).

The B60 is **Xe2-HPG** (Battlemage). The guide's own reference table uses
the **Arc B580** as its Xe2-HPG example — B580 and B60 share the same
20-Xe-core, 8-vector-engine-per-core configuration, so numbers quoted for
"Xe2-HPG" in Intel's docs map directly onto the B60 unless a spec explicitly
differs (B60 has 24GB GDDR6 / 192-bit bus vs. B580's smaller consumer specs —
verify VRAM-dependent numbers against the B60-specific datasheet, not this
generic architecture page).

## Xe2-HPG (B60-class) hardware numbers confirmed on this page

| Property | Value |
|---|---|
| Xe-Core count | 20 |
| Vector Engines per Xe-Core | 8 |
| Vector Engine count (total) | 160 |
| Hardware Threads per Vector Engine | 8 |
| Hardware Thread count (total) | 1280 |
| Matrix Engine (XMX/DPAS) support | Yes |
| Native FP64 support | Yes |
| General register file per thread | 128 / 256 (regular / large-register mode) |
| Register width | 512 bits |
| L3 cache size | 18 MB |
| L1 cache size per Xe-Core | 256 KB |
| SLM (Shared Local Memory) size per Xe-Core | 128 KB |
| Max SLM per Work-Group | 128 KB |
| Max Work-Group size | 1024 |
| Supported sub-group sizes | 16, 32 |

Note: "Global Memory Size" listed for the B580 example (12 GB) does **not**
apply to the B60 — the B60 has 24 GB GDDR6 per your project guide's own
hardware table. Everything else in this table is architecture-level (shared
across all Xe2-HPG SKUs), not memory-capacity-level.

## Building blocks, smallest to largest

- **Vector Engine (VE):** smallest thread-level unit. Multithreaded with 8
  hardware threads; each thread runs SIMD instructions (width 16 or 32).
  Contains SIMD ALUs supporting FP64/FP32/FP16/INT64/BF16/INT32/INT16/INT8
  (exact mix varies by generation).
- **Xe-Core:** the fundamental building block. Contains both Vector Engines
  and Matrix Engines (XMX). Has a shared L1 cache and shared SLM accessible
  by all Vector Engines within that Xe-Core.
- **Xe-Stack:** a collection of Xe-Cores + ray tracing units + hardware
  contexts + memory controllers + media engines = one functional GPU. Has
  an L3 cache shared across all its Xe-Cores.
- **Xe GPU:** one or two Xe-Stacks. Only Xe-HPC (data-center) parts are
  multi-stack; Xe-HPG (our B60), Xe-LP, and Xe2-HPG are single-stack.

## Querying hardware characteristics at runtime (don't hardcode)

Some characteristics (Xe-core count, vector engine count, SLM size) are
queryable via SYCL device info at startup rather than hardcoded from a spec
sheet — do this in `src/core` device initialization so the code is correct
even if run on a different Battlemage SKU:

```
q.get_device().get_info<sycl::info::device::name>()
q.get_device().get_info<sycl::ext::intel::info::device::gpu_slices>()
q.get_device().get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>()
q.get_device().get_info<sycl::ext::intel::info::device::gpu_eu_count_per_subslice>()
q.get_device().get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>()
q.get_device().get_info<sycl::info::device::global_mem_size>()
q.get_device().get_info<sycl::info::device::local_mem_size>()
q.get_device().get_info<sycl::info::device::max_work_group_size>()
q.get_device().get_info<sycl::info::device::sub_group_sizes>()
```

(`gpu_slices * gpu_subslices_per_slice` = Xe-core count; multiply further by
`gpu_eu_count_per_subslice` for vector-engine count, per the guide's own
worked example.)

## What this note does NOT cover

- L2 cache specifics, ray-tracing units, media engine details — not needed
  for the inference engine.
- Exact B60 VRAM bandwidth/bus numbers — already in your project guide's
  hardware table (§2), sourced from Intel's product page separately.
- See `docs/vendor/thread-mapping-occupancy.md` for how these numbers turn
  into actual kernel work-group/sub-group sizing decisions.
