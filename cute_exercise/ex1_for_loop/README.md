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
| `torch.add` (baseline) | 115 us / 6996 GB/s | 228 us / 7075 GB/s |
| `vectorized` (fragment `.load()`) | 117 us / 6858 GB/s | 230 us / 7012 GB/s |
| `vec_ld_scalar_add` (wide ld/st, scalar `for` for the add) | 117 us / 6874 GB/s | 230 us / 7013 GB/s |
| `scalar_ld_vec_add` (scalar `for` filling register frags, then vector add+store) | 118 us / 6846 GB/s | 230 us / 7009 GB/s |
| `for_loop` (per-element `thrA[i] + thrB[i]` over global) | 360 us / 2238 GB/s | 357 us / 4515 GB/s |

Vectorized matches `torch.add` within ~2%. Plain `for i: thrC[i] = thrA[i] +
thrB[i]` drops to ~32% (fp16) / ~64% (fp32) of peak.

The three fast variants are all the same — and `scalar_ld_vec_add` is the
interesting one: it tries to force *scalar* loads by filling register
fragments element-by-element, but the CuTe DSL / MLIR / NVVM frontend
coalesces those scalar assignments back into wide `ld.global.v4` before PTX.
So the plain `for_loop` variant is the only one that actually produces scalar
global ld/st in PTX — because its loop body mixes the load with the store.

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

| variant | dtype | PTX loads | PTX adds | SASS loads | SASS stores | SASS insn |
|---|---|---|---|---|---|---:|
| `vectorized` | fp16 | 32 × `ld.global.v4.b32` | 64 × `add.f16x2` | 32 × `LDG.E.128` | 16 × `STG.E.128` | 236 |
| `vec_ld_scalar_add` | fp16 | 32 × `ld.global.v4.b32` | 128 × `add.f16` | 32 × `LDG.E.128` | 16 × `STG.E.128` | 236 |
| `scalar_ld_vec_add` | fp16 | 32 × `ld.global.v4.b32` | 64 × `add.f16x2` | 32 × `LDG.E.128` | 16 × `STG.E.128` | 236 |
| `for_loop` | fp16 | 256 × `ld.global.b16` | 128 × `add.f16` | 256 × `LDG.E.U16` | 128 × `STG.E.U16` | 756 |
| `vectorized` | fp32 | 32 × `ld.global.v4.b32` | 64 × `add.f32` | 32 × `LDG.E.128` | 16 × `STG.E.128` | 236 |
| `for_loop` | fp32 | 128 × `ld.global.b32` | 64 × `add.f32` | 128 × `LDG.E` (32b) | 64 × `STG.E` | 380 |

Bytes moved per thread are identical; what changes is the width. Per-element
indexing via `thrA[i]` lowers to `ld.global.b16` / `ld.global.b32` at the PTX
level, and ptxas has no freedom to re-vectorize those back into `LDG.E.128`.
The warp still hits consecutive addresses at the same instant (TV layout is
unchanged), but each instruction now only moves 2 B or 4 B per thread instead
of 16 B.

The `vec_ld_scalar_add` row is the clean control: PTX emits 128 scalar
`add.f16` (same as `for_loop`), but the SASS is **byte-identical** to
`vectorized` — ptxas re-packs the scalar adds into `HADD2`/`HFMA2` on its own
because the operands are clearly adjacent in register space. So the scalar
`for` loop over the add is free; it's the scalar indexing into *global* memory
(what the plain `for_loop` variant does) that the compiler can't rescue.

`scalar_ld_vec_add` tries the opposite: scalar `rA[i] = thrA[i]` assignments
filling a register fragment, then a fragment-wide `rA.load() + rB.load()`.
Surprise — the CuTe DSL / MLIR / NVVM frontend re-vectorizes the scalar loads
into wide `ld.global.v4` before PTX, and the SASS again ends up
byte-identical to `vectorized`. So the two compiler layers rescue different
things:

| layer | scalar adds → packed? | scalar loads → wide? |
|---|---|---|
| CuTe DSL → MLIR → NVVM (PTX gen) | no | **yes**, when dataflow into a register fragment is clear |
| ptxas (PTX → SASS) | **yes**, into `HADD2`/`HFMA2` | no |

The only way to actually get scalar global ld/st in PTX is what plain
`for_loop` does: mix the load, add, and store in the same loop body — the
interleaving breaks the frontend's dataflow analysis and it can't lift out
the wide load.

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
