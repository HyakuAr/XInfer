# Vendor Documentation Notes — Index & Grounding Rule

## Why this directory exists

Coding assistants routinely claim to have "read the docs" and then write
code from training-data pattern-matching instead — especially for something
as niche as SYCL Joint Matrix or Level Zero, where confident-but-wrong output
is common. The fix is not a stronger instruction to "please actually read
it" — it's removing the excuse. Every doc a milestone depends on gets pulled
into a local file *before* that milestone starts. After that, "read the
docs" means "open this local file," which is a normal, checkable file-read —
no network access, no claim to verify, no faking.

These notes are paraphrased technical summaries (API names, function
signatures, and structural facts are kept precise; prose is rewritten), not
copies of the original pages. The source URL is always listed and is
authoritative if this note is ever wrong or stale — re-fetch it if in doubt.

## Hard rule for the AI assistant

Before writing any code that implements something covered by a linked doc
below, **cite the specific local file and the specific fact you're using**
(e.g. "per `docs/vendor/xmx-joint-matrix.md`, the accumulator type is
selected via `use::accumulator`") *before* the code, in the same response.

If you cannot point to a specific fact from a local file for something
platform-specific (an exact API name, flag, aspect string, or numeric
limit), that is a signal you don't actually have it — say so and ask for
that doc to be fetched into `docs/vendor/` first. Do not fill the gap from
memory and present it as fact.

## Status

| Topic | Needed for milestone | Local file | Status |
|---|---|---|---|
| SYCL Joint Matrix / XMX programming | M7 (also useful context for M4) | `docs/vendor/xmx-joint-matrix.md` | ✅ extracted |
| Intel Xe GPU Architecture (memory hierarchy, occupancy) | M3, M4 | `docs/vendor/xe-gpu-architecture.md` | ⬜ not yet — fetch before M3 |
| Level Zero (command lists, immediate command lists) | M3, M8 | `docs/vendor/level-zero.md` | ⬜ not yet — fetch before M3/M8 |
| Level Zero Core API spec (full) | M8 (graph capture/replay) | `docs/vendor/level-zero-core-api.md` | ⬜ not yet — fetch before M8 |
| XeTLA GEMM construction guide | M7 (reference for XMX kernel structure) | `docs/vendor/xetla-gemm.md` | ⬜ not yet — fetch before M7 |
| Arc Pro B60 datasheet / management guide | M0 (hardware limits), ongoing reference | `docs/vendor/b60-datasheet.md` | ⬜ not yet — optional, fetch if a spec number is in question |
| oneAPI Base Toolkit install / driver install guide | M0 (already verified manually) | — | not needed as a note; environment already confirmed working |

## How to add a new note

When a milestone is about to start and its doc isn't extracted yet:

1. Ask the assistant (with web access) to fetch the exact URL from
   `NInfer-Intel-Project-Guide.md` §5.
2. It writes a paraphrased note here following the format of
   `xmx-joint-matrix.md`: source URL, version/date, API surface (exact
   names), structural summary, and an explicit "what this note does NOT
   cover" section so gaps are visible instead of silently filled in later.
3. Update the status table above.
4. Only then does the milestone that depends on it start.

Do not pre-fetch every doc in the guide at once — per `AGENTS.md` §2 (scope
control), pull in a doc just before the milestone that needs it, not
speculatively.
