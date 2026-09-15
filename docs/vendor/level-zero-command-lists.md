# Vendor note: Level Zero Command Lists & Command Queues

- **Sources:**
  - https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/immediate-command-lists.html (oneAPI GPU Optimization Guide 2025.2, last modified 2025-07-10)
  - https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api/apis/command_list.html (Level Zero Spec, v1.18.31 at extraction time)
  - https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api/apis/command_queue.html (same spec)
  - https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/EXT_Exp_CommandListClone.html (Command List Clone extension)
- **Extracted:** 2026-09-15

## THE critical distinction for Milestone 8 (graph capture/replay)

Level Zero has **two different kinds of command list**, and they are opposites
for the "record once, replay many times" pattern your roadmap wants:

| | Regular command list | Immediate command list |
|---|---|---|
| Created via | `zeCommandListCreate` | `zeCommandListCreateImmediate` |
| Execution timing | Deferred — commands are recorded, then submitted together | Immediate — each appended command executes right away |
| Can be replayed via `zeCommandQueueExecuteCommandLists`? | **Yes** — this is exactly the record-once/replay-many pattern | **No** — the spec explicitly states immediate command lists must not be passed to `zeCommandQueueExecuteCommandLists` |
| Analogous to CUDA... | **CUDA Graphs** (capture once, launch repeatedly) | A raw CUDA stream (low-latency, per-call submission) |

**For Milestone 8, use regular command lists (`zeCommandListCreate` +
`zeCommandListClose` + repeated `zeCommandQueueExecuteCommandLists`), not
immediate command lists.** Immediate command lists exist for a different
use case — low-latency one-off kernel launches — not for the decode-step
graph-capture pattern the roadmap describes. This is exactly the kind of
detail a coding assistant is likely to get backwards from training-data
pattern-matching (both are called "command lists," and immediate command
lists are the newer/more-discussed feature in recent Intel material), so
confirm this distinction explicitly before implementing M8.

## Regular command list / queue API surface (for M8)

- `zeCommandListCreate(hContext, hDevice, desc, &hCommandList)` — creates a
  command list in the "open" state. Thread-safe, callable from multiple
  threads simultaneously.
- Commands get appended to it (e.g. `zeCommandListAppendLaunchKernel`,
  `zeCommandListAppendMemoryCopy`) while it's open.
- `zeCommandListClose(hCommandList)` — closes it, making it submittable.
- `zeCommandQueueCreate(hContext, hDevice, desc, &hCommandQueue)` — creates
  the queue the closed list is submitted to.
- `zeCommandQueueExecuteCommandLists(hCommandQueue, numCommandLists,
  phCommandLists, hFence)` — submits one or more closed command lists to
  the queue, **in the order received**. This same closed command list can
  be submitted repeatedly — this is the actual "replay" mechanism.
- `zeFenceHostSynchronize` / events — used to know when execution completed
  (regular command lists still use the fence/queue synchronization model,
  unlike immediate lists which recommend event-based sync since they have
  no queue handle).

## Command List Clone extension — an even more direct match for capture/replay

If a decode-step command list needs to be captured once but run with
per-iteration-varying parameters (e.g. a different KV-cache write offset
each decode step), the **Command List Clone extension**
(`zeCommandListCreateCloneExp`) may be the more direct primitive: build and
close a command list with the `ZE_COMMAND_LIST_FLAG_EXP_CLONEABLE` flag,
execute it once, then clone it cheaply (no synchronization required) for
each subsequent iteration rather than re-recording from scratch. This is
worth evaluating against the plain "reuse the same closed list every step
and just update a device-memory-backed parameter" approach — check which
is actually faster on B60 rather than assuming.

## Immediate command lists (recap — NOT what M8 wants, but useful elsewhere)

- Created via `zeCommandListCreateImmediate`, which takes a **command
  queue descriptor** directly (it implicitly creates its own queue) rather
  than a plain command list descriptor.
- No open/close lifecycle needed.
- Two modes via `cmdQueueDesc.mode`: `ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS`
  (sync via events, since there's no queue handle to call
  `zeCommandQueueSynchronize` on) or `ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS`
  (each appended command blocks until complete — no explicit sync needed).
- Good fit for: short-running kernels launched many times where minimizing
  submission overhead per-call matters more than batching — e.g. possibly
  the M1–M5 naive single-kernel-at-a-time execution path, before M8
  introduces batched graph replay for the decode step specifically.

## What this note does NOT cover

- Full event/fence lifecycle semantics — fetch the Level Zero "Event" and
  "Fence" spec pages specifically if a synchronization bug shows up in M8.
- The Mutable Command List extension (`zeCommandListGetNextCommandIdExp`,
  letting you mutate a *specific* previously-recorded command like a group
  count without rebuilding the whole list) — this is a plausible alternative
  or complement to Command List Clone for M8's varying-per-step parameters;
  worth a dedicated look when M8 starts if the clone approach proves
  awkward, not fetched in full here.
- Multi-queue-group / copy-engine-specific command list details — not
  needed for a single-GPU single-resident-model engine.
