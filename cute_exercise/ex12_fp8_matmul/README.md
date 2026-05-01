# How do I write an FP8 matmul, with different scaling granularities?

## Question

FP8 matmul on Hopper / Blackwell uses `e4m3` (`float_e4m3_t`):
1 sign + 4 exponent + 3 mantissa, range `±448`. Operands are FP8, the
accumulator is `f32`. (`e5m2` is the other FP8 type, more common for
gradients; this exercise sticks to `e4m3`.)

The MMA instruction itself (`tcgen05.mma` on sm100, `wgmma` on sm90)
takes the FP8 operands directly and accumulates into `f32`. The
*scaling* — how each operand's tensor of FP8 values maps back to a real
number — is purely software; the math problem is "where exactly does
the scale multiply in, and at what granularity?"

This exercise builds the same `D = A @ B` four ways, one for each
common scaling recipe, and verifies they all agree on the math.

### Variant 0 — unscaled FP8 (sanity)

`A`, `B` are FP8 (`e4m3`), `D` is `bf16` or `f32`:

```
D[i, j] = sum_k A_q[i, k] * B_q[k, j]
```

with the sum done in `f32`. Not a real recipe (you can't represent a
typical matrix in `e4m3` without scaling), but it isolates the MMA path
from the scaling.

### Variant 1 — per-tensor scaling (alpha)

Two scalar `f32` scales `s_A`, `s_B` (one per tensor):

```
D[i, j] = s_A * s_B * sum_k A_q[i, k] * B_q[k, j]
```

The scale collapses into the GEMM `alpha`. No change to the inner
loop. This is what `cublasLtMatmul` does in its simplest FP8 mode and
what example 54 in CUTLASS demonstrates.

### Variant 2 — per-row / per-column (also called per-channel)

`s_A` is shape `(M,)`, `s_B` is shape `(N,)`, both `f32`:

```
D[i, j] = s_A[i] * s_B[j] * sum_k A_q[i, k] * B_q[k, j]
```

The scales apply outside the sum, so this is just an epilogue
broadcast: do the FP8 GEMM into an `f32` accumulator, multiply by
`s_A[i] * s_B[j]` per output element, cast to output dtype.

### Variant 3 — blockwise, DeepSeek-style 1×128 / 128×128

`s_A` has shape `(M, K/128)` (one scale per *row*, per *128-wide K
block*). `s_B` has shape `(N/128, K/128)` (one scale per *128×128
block*). With `K = 128 * T`:

```
D[i, j] = sum_t s_A[i, t] * s_B[j // 128, t]
                * sum_{k in block t} A_q[i, k] * B_q[k, j]
```

Now the scale lives *inside* the K reduction — you cannot pull it out.
Each K-tile of `128` elements gets its own pair of scales, so the
running accumulator is "scaled and added in" once per K-tile rather
than once at the very end. CUTLASS calls this "blockwise / groupwise
scaling"; it's what example 67/68/81 implement and what DeepSeek-V3
uses for FP8 training.

## Why this matters

Per-tensor scaling is the cheapest but loses the most range — one
outlier in the tensor sets the scale for everyone. Per-channel
recovers the per-row dynamic range. Blockwise recovers the per-K-tile
dynamic range, which is what you need when activations have very
different magnitude across the K (feature) dimension. Each step costs
more memory for the scale tensor and more work in the kernel.

The math problem to internalize is: in variant 3, you can no longer
treat the scale as `alpha`. The MMA instruction still consumes raw FP8
and produces an `f32` partial sum, but you need to flush that partial
sum, multiply by the right pair of scales, and add into a separate
`f32` accumulator at every K-tile boundary. That's a different
inner-loop shape than variants 0–2.

## Plan

One CuTe DSL kernel per variant, all on the same `(M, N, K) =
(2048, 2048, 2048)` problem with `e4m3` operands and `bf16` output.

1. **V0 — unscaled FP8 GEMM**
   - Generate `A`, `B` as random `e4m3` (no scale), reference =
     `A.to(f32) @ B.to(f32)`.
   - Build a single-tile or multi-tile `tcgen05.mma` / `wgmma`
     pipeline. Accumulator `f32`, output cast to `bf16`.
   - Verify against the reference up to the FP8 rounding error (not
     bit-exact; use a tolerance).

2. **V1 — per-tensor scaling**
   - Quantize an `f32` reference `A_ref` to `(s_A, A_q)` with
     `s_A = A_ref.abs().amax() / 448.0`, `A_q = (A_ref / s_A).to(e4m3)`.
     Same for `B`.
   - Reference `D_ref = A_ref @ B_ref` in `f32` (then cast to `bf16`).
   - Kernel: same as V0 but pass `alpha = s_A * s_B`.

3. **V2 — per-row / per-column**
   - Quantize per-row of `A`: `s_A[i] = A_ref[i].abs().amax() / 448`,
     `A_q[i, :] = A_ref[i, :] / s_A[i]` cast to `e4m3`. Same per-column
     of `B` (i.e. per-row of `B^T`).
   - Reference `D_ref = (s_A[:, None] * A_ref) @ (B_ref * s_B[None, :])`.
   - Kernel: V0 inner loop, then in the epilogue multiply each `D[i,j]`
     by `s_A[i] * s_B[j]` before casting. Bonus: also try doing the
     scale before the cast vs after the cast and measure the precision
     hit.

4. **V3 — blockwise, 1×128 / 128×128**
   - Quantize `A` per `(1, 128)` tile in K: `s_A[i, t] =
     A_ref[i, 128*t:128*(t+1)].abs().amax() / 448`.
   - Quantize `B` per `(128, 128)` tile: `s_B[j_blk, t] =
     B_ref[128*t:128*(t+1), 128*j_blk:128*(j_blk+1)].abs().amax() / 448`.
   - Inner loop now reads one 128-wide K-slab at a time; after the
     `mma` for that slab, multiply the partial `f32` accumulator by
     `s_A[i, t] * s_B[j // 128, t]` and add into the running `f32`
     output accumulator. This is the *only* variant where the K
     reduction is non-uniform, and the place where the kernel structure
     changes vs a normal GEMM.
   - Reference: dequantize `A`, `B` back to `f32` block by block,
     matmul, compare.

## Measurements

For each of `M = N = K = 2048` and `M = N = K = 8192`:

- TFLOP/s achieved (FP8 peak on GB200 is ~5 PFLOP/s). Compare the four
  variants — V3 should be slightly slower than V1/V2 because of the
  per-K-tile rescale.
- Numerical error vs the `f32` reference, reported as
  `(D - D_ref).abs().max()` and `RMSE / D_ref.abs().mean()`. Expect
  V3 < V2 < V1 in error (smaller scale block = less dynamic-range
  loss).
- For V3, sweep the scale block in K (`64, 128, 256`) and confirm
  smaller blocks → lower error.
## Open questions to settle while implementing

- For V3, where exactly should the rescale live? Two options:
  a) flush the `f32` MMA accumulator into a second `f32` register
     accumulator per K-tile (clean math, extra registers);
  b) reset the `f32` MMA accumulator to zero each K-tile and
     pre-scale the operand tile before the MMA (saves registers, but
     pre-scaling FP8 by an `f32` is itself lossy unless you upcast
     first — defeats the point).
  Verify the error matches the reference only for option (a).
- For V2, does the FP8 MMA hardware support an "accumulate-and-scale"
  fused op, or is the per-row multiply always an epilogue pass over
  the `f32` accumulator?
- `e4m3` has no infinity (it has NaN at the all-1s pattern). Does
  that matter for any of the variants when an outlier saturates? Try
  injecting one element near `±448` in `A_ref` for variant 1 and see
  whether the rounded `e4m3` value or the resulting GEMM differs from
  the `f32` reference more than expected.
- For V3, the canonical layout is `s_B` shape `(N/128, K/128)`,
  *transposed* relative to how PyTorch usually stores it. Why does the
  CUTLASS convention require K as the second mode? (Hint: it's the
  same reason A and B are stored K-major / K-major in CuTe — the K
  reduction always walks contiguous addresses.)
- Per-tensor V1: is there a difference between baking the scale into
  `alpha` (one fused multiply at epilogue) vs scaling the FP8 operands
  on the host before the kernel runs? They look mathematically
  identical but the second one re-quantizes and loses precision.
