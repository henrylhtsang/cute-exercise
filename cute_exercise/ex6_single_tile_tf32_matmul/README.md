# Single-tile TF32 matmul — TMEM + `tcgen05` practice

## Question

How do I write the smallest interesting Blackwell matmul: a single
tile, TF32 inputs, accumulate in FP32, using `tcgen05.mma` and TMEM?

The point isn't performance — it's to get hands-on with the
Blackwell-era pieces:

- TMEM allocation / deallocation (`tcgen05.alloc`, `tcgen05.dealloc`).
- Loading A/B from SMEM into the right TMEM/SMEM operand layout for
  `tcgen05.mma`.
- Issuing `tcgen05.mma` (CTA group ONE for a first pass) and waiting
  on its completion.
- Reading the FP32 accumulator out of TMEM back to registers / SMEM /
  GMEM.
- Picking a tile shape (M, N, K) that's actually legal for the TF32
  `tcgen05.mma` variants.

## Plan

- One CTA, one tile. No pipelining, no warp spec, no cluster.
- Inputs: A (M×K) and B (K×N), TF32. Output: C (M×N), FP32.
- Use TMA loads to bring A and B into SMEM (already practiced in ex3).
- Allocate TMEM for the accumulator, issue a single `tcgen05.mma`,
  wait, then store C back to GMEM.
- Correctness: compare against `torch.matmul` in TF32 mode (or FP32
  reference with TF32-rounded inputs).
- Stretch goals (separate exercises later): K-loop with multiple MMAs
  into the same TMEM accumulator, then SMEM pipelining, then warp
  specialization, then CTA group TWO.
