# Blackwell MMA from scratch — following gau-nernst's tcgen05 walkthrough

References:
- <https://gau-nernst.github.io/tcgen05/>
- <https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/>

## Question

How do I build a Blackwell (SM100) matmul from the ground up, step by
step, using `tcgen05.mma` and TMEM? Each step adds one new mechanic
on top of the previous, and the kernel must stay bit-correct against
`torch.matmul` after every step.

The point is to internalize the moving parts — TMEM allocation, SMEM
operand layouts, `tcgen05.mma` issue / wait, TMEM read-back — by
adding them one at a time, not by copying a full reference kernel.

## Setup

Single-tile BF16 GEMM:

- `A`: `(M, K)` BF16 in GMEM, row-major (K contiguous, K-major).
- `B`: `(K, N)` BF16 in GMEM, column-major (K contiguous, K-major).
- `D`: `(M, N)` BF16 in GMEM, row-major. FP32 accumulator lives in
  TMEM and is cast down to BF16 before the GMEM store.

Both operands are K-major because the BF16 `tcgen05.mma` atoms expect
the contraction dim contiguous in SMEM. (If your tutorial / atom
choice expects MN-major, flip the strides and document it here.)

Start with a single tile that the hardware can deliver in one
`tcgen05.mma cta_group::1` instruction, e.g. `(M, N, K) = (128, 128, 64)`.
No K loop, no pipelining, no warp-spec, no cluster.

## Plan

Follow the tutorial in order. Each step keeps the test passing.

1. **Naive cp.async load + manual MMA.** Bring `A` and `B` into SMEM
   with `cp.async`, allocate TMEM for the accumulator, issue one
   `tcgen05.mma`, wait, copy TMEM → RMEM → GMEM.
2. **TMA loads** for `A` and `B` (replace `cp.async`).
3. **Right SMEM swizzle / layout** for the MMA atom (so the operand
   side of the MMA is happy).
4. **K-loop into the same TMEM accumulator** (multiple MMAs into one
   TMEM region, no SMEM pipelining yet).
5. **2-stage SMEM pipeline** with mbarriers — overlap TMA load of
   stage `s+1` with MMA of stage `s`.
6. **Warp specialization** — split into producer (TMA) and consumer
   (MMA + epilogue) warpgroups.

Stretch goals (separate exercises):

- `cta_group::2` pair MMA (see `ex19_2cta_mma`).
- FP8 / NVFP4 with scale-factor metadata (see `ex12`–`ex14`).
- Multi-tile grid + persistent kernel.

## Constraints

- Test (`test_mma.py`) must pass after every step. If a step breaks
  correctness, fix it before moving on.
- Don't paste from a reference kernel — write each piece, then check
  it against the tutorial. The point is the muscle memory.
