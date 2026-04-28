# TF32×3 matmul — does the precision actually improve?

(Follow-up to [ex6](../ex6_single_tile_tf32_matmul/).)

## Question

TF32 keeps only 10 mantissa bits, so a single `tcgen05.mma` in TF32
mode loses precision vs true FP32. The "3xTF32" trick splits each FP32
input into a hi + lo TF32 pair (`x = x_hi + x_lo`, where `x_hi` is `x`
rounded to TF32 and `x_lo = x - x_hi`), and then reconstructs

```
A·B ≈ A_hi·B_hi + A_hi·B_lo + A_lo·B_hi
```

with three TF32 MMAs accumulated into the same FP32 accumulator. The
`A_lo·B_lo` term is dropped (below TF32's noise floor).

How much precision do we actually recover vs plain TF32, and what does
it cost?

## Plan

Build on ex6 (single tile, TMEM, `tcgen05.mma`):

- Pre-split A and B in a small kernel (or on the host for the
  exercise): produce `A_hi`, `A_lo`, `B_hi`, `B_lo` as TF32 tensors.
- Issue three `tcgen05.mma` ops accumulating into one TMEM
  accumulator, in this order: `A_hi·B_lo`, `A_lo·B_hi`, `A_hi·B_hi`
  (small terms first to preserve them when the big term lands).
- Read C back as FP32.

Measurements:

- Max abs error and ULP error vs FP64 reference, for:
  - plain TF32 (ex6),
  - TF32×3,
  - native FP32 (`torch.matmul` with TF32 disabled).
- Cost: 3× the MMA issue rate, plus the split. Measure end-to-end
  vs ex6 to see the real slowdown — it should be close to 3× on the
  MMA-bound regime.

Stretch: do the hi/lo split inside the kernel from FP32 GMEM inputs
(saves a memory round-trip vs splitting on the host).
