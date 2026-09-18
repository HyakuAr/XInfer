# xInfer

> Selected checkpoints. Maximum single-GPU inference performance on Intel Arc Pro B60.

xInfer is a from-scratch C++/SYCL/Level Zero inference engine for Qwen3.8 hybrid architectures on a single Intel Arc Pro B60 (Battlemage / Xe2-HPG, 24 GB GDDR6). It runs text prompts through a local CLI or OpenAI-compatible HTTP APIs. The runtime is deliberately specialized: one GPU, one resident model, and a startup-fixed capacity of one to eight active requests.

The engine executes from a single self-contained `.xinfer` binary artifact containing INT4 quantized weights, scales, BPE tokenizer, and native chat template.

| Model | Weights | Artifact | Target GPU |
|---|---|---|---|
| Qwen3.8-27B | `INT4-G128-SYM` | `qwen3_8_27b.xinfer` | Intel Arc Pro B60 (24 GB, `0xE211`) |

Each `.xinfer` artifact carries model configuration, INT4-packed weights, per-group FP16 scales, tokenizer tables, and chat template formatting. Runtime execution uses those facts with compiled hardware kernels. You can [convert official weights](#model-conversion) directly from Hugging Face safetensors checkpoints.

---

## Quick start

xInfer requires 64-bit Windows or Linux, an Intel Arc Pro B60 GPU (device ID `0xE211`), Intel oneAPI DPC++/C++ compiler (`icx`/`icpx` 2025.0 or newer), Level Zero runtime (`ze_api`), CMake 3.20 or newer, and Ninja.

Build the product binaries:

```bash
# Initialize the oneAPI build environment
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"   # Windows
# source /opt/intel/oneapi/setvars.sh                   # Linux

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Tests and diagnostic benchmarks are enabled via CMake testing (`ctest`). The build compiles `xinfer` (standalone CLI), `xinfer-serve` (HTTP serving daemon), and the `xinfer_artifact_c` acceleration library.

There is no separate install target; run xInfer directly from its build tree.

### Model conversion

Convert an official `Qwen/Qwen3.8-27B` Hugging Face checkpoint to `.xinfer`:

```bash
python tools/convert/qwen3_8_27b/convert.py \
  --checkpoint-dir path/to/Qwen3.8-27B \
  --output-path out/qwen3_8_27b.xinfer \
  --group-size 128
```

The converter performs symmetric INT4 per-group quantization, packages tokenizer vocabulary and chat templates, verifies dequantization against an FP32 numerical oracle, and appends a whole-file CRC-64/ECMA-182 checksum.

You can also export `XINFER_CHECKPOINT_DIR=path/to/Qwen3.8-27B` to omit `--checkpoint-dir`.

### Running the server

Start a long-running OpenAI-compatible HTTP server:

```bash
./build/xinfer-serve \
  --model out/qwen3_8_27b.xinfer \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 8 \
  --max-seq-len 8192
```

The server binds the HTTP listener, initializes the resident model in GPU memory, pre-allocates KV cache pools, and captures the 64-layer decode graph. Up to 8 connections can be held concurrently; inference forward passes are serialized through the resident model.

Send an OpenAI-style chat completion request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Explain Level Zero command lists in one short sentence."}],
    "max_tokens": 64
  }'
```

Streaming responses (`"stream": true`) return server-sent events (`text/event-stream`) ending with `data: [DONE]`.

### Running one-shot CLI inference

Run interactive generation with a 8,192-token context ceiling:

```bash
./build/xinfer \
  --model out/qwen3_8_27b.xinfer \
  --prompt "Explain the difference between compute-bound prefill and memory-bound decode." \
  --max-tokens 128 \
  --max-seq-len 8192 \
  --chunk-size 512
```

Answer content is streamed directly to stdout. Model loading, memory allocation diagnostics, and generation timing are written to stderr.

---

## Architecture and runtime specialization

xInfer achieves single-GPU throughput by intentionally refusing generality. The engine implements zero multi-GPU code, no dynamic continuous batching, and no weight paging.

### Level Zero command graph capture (`DecodeGraph`)
In a 64-layer model, a naive decode step issues ~1,000 individual SYCL kernel submissions per token, creating massive host OS and driver launch overhead. `DecodeGraph` captures the entire decode loop into a single executable Level Zero command graph (`sycl::ext::oneapi::experimental::command_graph`). Token positions are updated in-place via device USM memory, allowing continuous graph replay with zero submission bubbles.

### Vector Engine SIMD16 INT4 GEMV
Single-token decode ($M=1$) has an arithmetic intensity of $\approx 4.0\text{ FLOP/byte}$, making it 100% memory bandwidth-bound. Microbenchmarks on the Arc Pro B60 showed that unpacking INT4 weights through SLM into systolic XMX units is **7.5x slower** (51.0 GB/s) due to barrier serialization. xInfer executes decode projections via sub-group cooperative SIMD16 Vector Engine kernels with single-cycle ALU bit-shift dequantization, streaming weights at up to **383.7 GB/s** (84.1% of physical peak bandwidth).

### Native hybrid attention
Qwen3.8 alternates between two distinct token-mixing mechanisms across its 64 layers:
- **48 Linear Attention Layers:** Fused single-pass Causal Conv1d + SiLU and parallelized Recurrent Gated Delta Net maintaining a constant-size recurrent state ($\approx 150\text{ MB}$ total across all layers).
- **16 Full Attention Layers:** Grouped Query Attention (GQA, 24 query heads, 4 KV heads, head dimension 256) backed by a pre-allocated FP16 KV cache.

### Zero-allocation decode loop
All intermediate activation buffers and persistent logit scratchpads are allocated from a 64-byte aligned `DeviceArena`. The bump allocator resets between decode steps without calling driver USM allocation or free routines.

---

## Memory layout (24 GB VRAM)

The resident memory footprint on the Intel Arc Pro B60:

| Allocation | Size | Description |
|---|---|---|
| Model weights | 15.77 GB | INT4 packed weights + BF16 scales + norms |
| Linear attention state | 0.15 GB | 48 recurrent state matrices ($S$) + conv1d state |
| Device scratch arena | 0.25 GB | 64-byte aligned bump arena for intermediate activations |
| Full attention KV cache | 0.51 GB | 16 full-attention layers, FP16, 8,192 token ceiling |
| **Total Resident (8K context)** | **16.68 GB** | **~6.3 GB headroom remaining** |
| *Max context (65,536 tokens)* | *20.19 GB* | *Fits entirely within 24 GB physical budget* |

---

## Repository layout

| Path | Ownership |
|---|---|
| `src/core/` | SYCL/Level Zero primitives: device context, USM arenas, command lists, physical KV cache. |
| `src/artifact/` | `.xinfer` container reader, writer, section tables, and CRC-64 verification. |
| `src/ops/` | Closed hardware kernels: SIMD16 INT4 GEMV, attention, RMSNorm, RoPE, Conv1d, Delta Net. |
| `src/targets/qwen3_8/` | Qwen3.8 family logic: forward pass, `DecodeGraph` replay, BPE tokenizer, chat template. |
| `src/targets/qwen3_8_27b/` | Checkpoint packaging, weight view schema, and artifact loader. |
| `src/runtime/` | Public `Engine` interface (PIMPL) and token transaction lifecycle. |
| `src/serve/` | Multi-threaded HTTP server exposing OpenAI-compatible endpoints. |
| `apps/` | Binaries: `xinfer` (CLI) and `xinfer-serve` (HTTP server). |
| `tools/convert/` | Offline Hugging Face safetensors to `.xinfer` INT4 converter. |
| `tools/parity/` | Hardware profiling, numerical parity checks, and PCIe monitors. |
| `docs/vendor/` | Extracted Intel architecture guides, occupancy rules, and B60 matrix capabilities. |
| `tests/` | CTest test suites for device primitives, numerical oracles, and HTTP schemas. |

---

## Verification and testing

Run the full verification test suite:

```bash
ctest --test-dir build --output-on-failure
```

Tests validate:
- **Numerical Oracles (`tests/test_ops_oracle.cpp`):** Every op kernel (attention, RMSNorm, RoPE, elementwise, dequantizing GEMV) is verified against an independent FP32 CPU oracle evaluated from represented inputs.
- **Container Round-Trip (`tests/test_artifact_roundtrip.cpp`):** Container serialization, section table integrity, and CRC-64 corruption detection.
- **Command Graph (`tests/test_decode_graph.cpp`):** Hardware graph execution and dynamic device-pointer position updates.
- **Serving Schema (`tests/test_serve_schema.cpp`):** OpenAI request/response JSON validation, token usage accounting, and streaming SSE framing.

---

## References and lineage

- Design philosophy, artifact container concepts, and repository structure inspired by [Neroued/ninfer](https://github.com/Neroued/ninfer) (Apache 2.0).
- Hardware execution targeting the [Intel Arc Pro B60](https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html) via [oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) and [Level Zero](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/).
- Target model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen Team.
- See [`AGENTS.md`](AGENTS.md) for architectural boundaries and operating rules.
- See [`ROADMAP.md`](ROADMAP.md) for milestone deliverables and hardware profiling breakdown.
