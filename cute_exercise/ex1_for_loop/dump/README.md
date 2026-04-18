# PTX / SASS comparison — `vectorized` vs `for_loop`

Dumped with `python dump_asm.py` (sets `CUTE_DSL_KEEP_PTX=1` /
`CUTE_DSL_KEEP_CUBIN=1`). SASS produced via `nvdisasm` against `sm_100a`.

## Load / store widths (per thread)

| variant | dtype | PTX loads | SASS loads | SASS stores | SASS insn count |
|---|---|---|---|---|---:|
| `vectorized` | fp16 | 32 × `ld.global.v4.b32` (128-bit) | 32 × `LDG.E.128` | 16 × `STG.E.128` | 236 |
| `vectorized` | fp32 | 32 × `ld.global.v4.b32` (128-bit) | 32 × `LDG.E.128` | 16 × `STG.E.128` | 236 |
| `for_loop` | fp16 | 256 × `ld.global.b16` (16-bit) | 256 × `LDG.E.U16` | 128 × `STG.E.U16` | 756 |
| `for_loop` | fp32 | 128 × `ld.global.b32` (32-bit) | 128 × `LDG.E` (32-bit) | 64 × `STG.E` | 380 |

Per thread the work is identical (128 fp16 or 64 fp32 values from each of A
and B, written once to C). The bytes read are the same; what changes is how
many instructions it takes:

- `vectorized`: 2 tensors × 16 rows = **32 wide loads**, each 16 B.
- `for_loop` fp16: **256 scalar 16-bit loads**, one per element.
- `for_loop` fp32: **128 scalar 32-bit loads**, one per element.

## Instruction-count vs wall-clock

| variant | dtype | SASS insn ratio | wall-clock ratio (vs vectorized) |
|---|---|---:|---:|
| `for_loop` | fp16 | 756 / 236 = **3.2×** | 359 / 117 = **3.07×** |
| `for_loop` | fp32 | 380 / 236 = **1.61×** | 358 / 230 = **1.56×** |

Within a few percent, the slowdown tracks the SASS instruction count —
consistent with the bottleneck being issue/throughput on the memory path,
not DRAM bandwidth.

## Takeaway

Per-element indexing via `thrA[i]` defeats the wide-load codegen: PTX emits
`ld.global.b16` / `ld.global.b32` with no vector form, and ptxas has no
freedom to re-vectorize those back into `LDG.E.128`. The warp still hits
consecutive addresses at the same instant (TV layout is unchanged), but
each instruction now only moves 2 B or 4 B per thread instead of 16 B.
