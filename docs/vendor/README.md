# Intel Architecture & Vendor Documentation

This directory contains local, extracted vendor specifications and optimization guides for the Intel Arc Pro B60 (Battlemage / Xe2-HPG, device ID `0xE211`), SYCL / Level Zero, and XMX matrix engines.

Per `AGENTS.md` §5, before writing any code that relies on Intel-specific APIs, flags, aspect strings, or numeric limits, the corresponding file in this directory must be consulted and cited.

---

## Index of Local Vendor Documents

| Document | Topic & Scope | Relevant Milestones | Status |
| :--- | :--- | :---: | :---: |
| [`xe-gpu-architecture.md`](xe-gpu-architecture.md) | Xe2 / Battlemage architecture, memory hierarchy (GDDR6, L2, SLM), cache line sizes, XMX hardware units. | M0, M3, M6, M7 | `[x] Extracted & Verified` |
| [`thread-mapping-occupancy.md`](thread-mapping-occupancy.md) | Workgroup sizing, sub-groups (SIMD16 / SIMD32), wavefront scheduling, occupancy rules for Xe2. | M3, M4, M7 | `[x] Extracted & Verified` |
| [`xmx-joint-matrix.md`](xmx-joint-matrix.md) | SYCL Joint Matrix extensions, programming model, DPAS systolic tile operations. | M7 | `[x] Extracted & Verified` |
| [`b60-matrix-caps.md`](b60-matrix-caps.md) | Verified hardware matrix combinations on Intel Arc Pro B60 (53 combinations: INT8, FP16, BF16, TF32; no native INT4). | M7, M10 | `[x] Extracted & Verified` |
| [`level-zero-command-lists.md`](level-zero-command-lists.md) | Level Zero command-list creation, immediate command lists, graph capture/replay, barrier synchronization. | M3, M8 | `[x] Extracted & Verified` |
| [`xetla-gemm.md`](xetla-gemm.md) | XeTLA high-performance GEMM structure, epilogue fusion, systolic tile staging. | M7 | `[x] Extracted & Verified` |

*Reconciliation Note:* All required vendor documentation for milestones M0 through M12 is fully extracted, locally archived in this directory, and verified against the Intel Arc Pro B60 GPU and Intel oneAPI 2026.1 / Level Zero driver stack. There are no remaining "not yet extracted" documents blocking roadmap milestones.
