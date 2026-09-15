# Intel Architecture & Vendor Documentation

This directory contains local, extracted vendor specifications and optimization guides for the Intel Arc Pro B60 (Battlemage / Xe2-HPG, device ID `0xE211`), SYCL / Level Zero, and XMX matrix engines.

Per `AGENTS.md` §5, before writing any code that relies on Intel-specific APIs, flags, aspect strings, or numeric limits, the corresponding file in this directory must be consulted and cited.

---

## Index of Local Vendor Documents

| Document | Topic & Scope | Relevant Milestones |
| :--- | :--- | :---: |
| [`xe-gpu-architecture.md`](xe-gpu-architecture.md) | Xe2 / Battlemage architecture, memory hierarchy (GDDR6, L2, SLM), cache line sizes, XMX hardware units. | M0, M3, M6, M7 |
| [`thread-mapping-occupancy.md`](thread-mapping-occupancy.md) | Workgroup sizing, sub-groups (SIMD16 / SIMD32), wavefront scheduling, occupancy rules for Xe2. | M3, M4, M7 |
| [`xmx-joint-matrix.md`](xmx-joint-matrix.md) | SYCL Joint Matrix extensions, supported matrix shapes ($M \times N \times K$) for INT4/BF16 on Xe2, tile loads and stores. | M7 |
| [`level-zero-command-lists.md`](level-zero-command-lists.md) | Level Zero command-list creation, immediate command lists, graph capture/replay, barrier synchronization. | M3, M8 |
| [`xetla-gemm.md`](xetla-gemm.md) | XeTLA high-performance GEMM structure, epilogue fusion, systolic tile staging. | M7 |
