# How bad is a `for` loop in CuTe DSL?

## Question

Say I'm writing an elementwise op in CuTe DSL — think elementwise add. The
simplest thing is to loop over elements with a plain Python `for` loop. Is
that a problem?

Some angles:

- Is there a vectorized add / scale primitive I should reach for instead?
- Or is the thing I really care about just vectorized loads and stores, and
  the `for` over the register tile doesn't actually matter?
- Does it change anything if the loop bound is a compile-time constant vs. a
  runtime value?

## Results (GB200, M=16384, N=8192)

| variant | fp16 | fp32 |
|---|---:|---:|
| `torch.add` (baseline) | 115 us / 6987 GB/s | 228 us / 7077 GB/s |
| `vectorized` (fragment `.load()`) | 117 us / 6863 GB/s | 230 us / 7014 GB/s |
| `for_loop` (per-element Python `for`) | 359 us / 2243 GB/s | 358 us / 4499 GB/s |

Vectorized matches `torch.add` within ~2%. Plain `for i: thrC[i] = thrA[i] +
thrB[i]` drops to ~32% (fp16) / ~64% (fp32) of peak.

## Why the `for` loop is slow — coalescing at the same instant

The hardware caps each thread at a 128-bit (16 B) ld/st per instruction, so
moving 128 fp16 values per thread takes multiple instructions no matter what.
What matters is that **the 32 threads of a warp, executing the same
instruction at the same instant, hit consecutive addresses** — that's what
lets the hardware fold them into a few 128 B sectors.

Our TV layout gives each thread a `(16 rows, 8 cols)` fragment:

- Per instruction: thread `i` loads cols `[8i, 8i+8)` of one row → 32 threads
  together cover 512 consecutive bytes → 4 sectors, fully coalesced.
- 16 such instructions (one per row) per thread.

The `for_loop` variant has the same TV layout; the issue is that indexing
`thrA[i]` / `thrC[i]` one element at a time makes the compiler emit per-element
ld/st instead of packing a row into a single 128-bit instruction, so the
"same instant" still holds but the wide vector op is lost.

A naively different layout — "give each thread 128 consecutive elements" —
would be even worse: within any single 128-bit instruction, thread `i` would
be 256 B away from thread `i+1`, so the warp would fan out to 32 separate
sectors per instruction (~8× the memory traffic) instead of 4.

The outer "16 rows" isn't about packing more work into one instruction (you
can't; 16 B is the hardware cap). It's about giving each thread 16 cleanly
coalesced instructions to issue.

## Confirming it in PTX/SASS

Dumped with `python dump_asm.py`, which sets `CUTE_DSL_KEEP_PTX=1` /
`CUTE_DSL_KEEP_CUBIN=1` and disassembles the cubin with `nvdisasm` against
`sm_100a`.

| variant | dtype | PTX loads | SASS loads | SASS stores | SASS insn count |
|---|---|---|---|---|---:|
| `vectorized` | fp16 | 32 × `ld.global.v4.b32` | 32 × `LDG.E.128` | 16 × `STG.E.128` | 236 |
| `vectorized` | fp32 | 32 × `ld.global.v4.b32` | 32 × `LDG.E.128` | 16 × `STG.E.128` | 236 |
| `for_loop` | fp16 | 256 × `ld.global.b16` | 256 × `LDG.E.U16` | 128 × `STG.E.U16` | 756 |
| `for_loop` | fp32 | 128 × `ld.global.b32` | 128 × `LDG.E` (32b) | 64 × `STG.E` | 380 |

Bytes moved per thread are identical; what changes is the width. Per-element
indexing via `thrA[i]` lowers to `ld.global.b16` / `ld.global.b32` at the PTX
level, and ptxas has no freedom to re-vectorize those back into `LDG.E.128`.
The warp still hits consecutive addresses at the same instant (TV layout is
unchanged), but each instruction now only moves 2 B or 4 B per thread instead
of 16 B.

The SASS instruction-count ratio matches the wall-clock slowdown within a
few percent, consistent with the bottleneck being memory-path issue
throughput, not DRAM bandwidth:

| variant | dtype | SASS insn ratio | wall-clock ratio |
|---|---|---:|---:|
| `for_loop` | fp16 | 756 / 236 = **3.2×** | 359 / 117 = **3.07×** |
| `for_loop` | fp32 | 380 / 236 = **1.61×** | 358 / 230 = **1.56×** |

## Follow-ups

- **Persistent kernel + CTA swizzle** — one block per SM looping over tiles;
  kills tail-wave + launch overhead, and swizzling improves L2 reuse.
