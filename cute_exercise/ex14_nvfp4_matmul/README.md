# How do I write an NVFP4 matmul?

## Question

NVFP4 is Nvidia's narrow-precision format introduced with Blackwell.
One NVFP4 *block* is:

- **16 consecutive elements** along K, stored as `e2m1` —
  `1 + 2 + 1` bits, range `±6`. There are exactly 16 distinct
  representable magnitudes (including zero), so this is genuinely
  4-bit float.
- **One per-block scale**, stored as `ue4m3` — an unsigned 8-bit
  *FP8* scale (4 exponent + 3 mantissa, no sign), so the scale itself
  has ~3 bits of mantissa precision and a wide exponent range. This
  is what distinguishes NVFP4 from MXFP4 (which uses `ue8m0` like
  MXFP8 — see ex13).
- **One per-tensor outer scale**, stored as `f32`. NVFP4 has *two*
  levels of scale: a fine-grained per-16-element FP8 scale, and a
  per-tensor FP32 scale on top. MXFP8 / MXFP4 have only one level.

Dequantization:

```
A_real[i, k] = global_scale_A
             * (sf_A[i, k // 16].to(f32))
             * A_q[i, k].to(f32)
```

The block size is **16** (not 32 like MX formats), the per-block scale
is **FP8 with mantissa** (not power-of-two), and there is an **extra
per-tensor scale** on top.

## The math

```
D[i, j] = global_scale_A * global_scale_B
       * sum_{t=0}^{K/16 - 1}
            sf_A[i, t] * sf_B[j, t]
          * sum_{k in block t} A_q[i, k] * B_q[k, j]
```

Three reductions stacked: the inner FP8(`e2m1`) dot product over 16
elements, the per-block FP8(`ue4m3`) scale multiply, and the per-tensor
`f32` scale outside the K reduction.

The Blackwell instruction `tcgen05.mma.blockscaled` consumes the
operand `A_q`/`B_q` and the per-block scales `sf_A`/`sf_B` directly
(same instruction as MXFP8, different operand-type / scale-type
template arguments). The `global_scale_A * global_scale_B` factor is
just an `alpha`-style epilogue multiply on the `f32` accumulator.

The CUTLASS type spelling is
`cutlass::nv_float4_t<cutlass::float_e2m1_t>`, with
`ScaleFactorType = float_ue4m3_t`.

## Why this matters

- 4-bit operands → 2× the throughput of FP8 mma instructions and 4× of
  Hopper WGMMA fp8 — this is the format that underpins NVFP4
  inference on B200/GB200.
- The two-level scaling is exactly because `e2m1` has only 16 levels
  and ~6 of dynamic range. Without the FP8 per-block scale, you'd need
  one quantization scale per element to recover any precision; with
  it, 16-element blocks of FP4 land in a usable range. The
  per-tensor `f32` scale sits on top to absorb the global magnitude.
- Compared to MXFP4 (also 4-bit, also 32-element MX-style block, but
  with a **power-of-two** scale and **no** outer scale), NVFP4 spends
  more bits on the scale to win back precision on the operand. The
  question of whether that's worth it is empirical.

## Plan

Three variants on `(M, N, K) = (2048, 2048, 2048)` (K must be a
multiple of 16). All use `nv_float4_t<e2m1>` operands, `f32`
accumulator, `bf16` output.

1. **NVFP4 → bf16** (CUTLASS example 72a)
   - Quantize `A_ref` (`f32`) into `(global_scale_A, sf_A, A_q)`:
     - `global_scale_A = A_ref.abs().amax() / (6.0 * 448.0)`
       (so that the worst-case per-block amax stays inside what
       `ue4m3 * e2m1` can represent; the constant comes from
       `e2m1_max * ue4m3_max = 6 * 448`).
     - For each `(i, t)`:
       `block = A_ref[i, 16*t:16*(t+1)] / global_scale_A`,
       `sf = block.abs().amax() / 6.0` quantized to `ue4m3`,
       `A_q[i, 16*t:16*(t+1)] = (block / sf).to(e2m1)`.
   - Same for `B`.
   - Build the `tcgen05.mma.blockscaled` call with `nv_float4_t`
     operands. Bake `global_scale_A * global_scale_B` into `alpha`,
     bf16 output in the epilogue.
   - Reference: dequantize back to `f32`, matmul, compare.

2. **NVFP4 → NVFP4** (CUTLASS example 72b)
   - Same kernel but the output is also NVFP4: the epilogue produces
     `(global_scale_D, SFD, D_q)`, computing the per-block amax of
     the `f32` accumulator on the fly. SFD is `ue8m0` per the
     example; verify whether the spec actually wants `ue4m3` here
     too. The math problem is "epilogue-side blockwise quantization
     given a streaming f32 accumulator".
   - Reference: do step 1, then quantize the bf16 output to NVFP4
     and compare.

3. **NVFP4 vs MXFP4 head-to-head**
   - Implement an MXFP4 path (`mx_float4_t<e2m1>`, `ue8m0` scale,
     32-element block, no outer scale) on the same input.
   - Compare error vs the `f32` reference. The hypothesis to test is:
     NVFP4 wins on smooth distributions (the FP8 scale captures the
     per-block magnitude precisely) and MXFP4 is more robust on
     pathological / outlier-heavy distributions because of the wider
     E8M0 exponent range. Verify or refute.

## Layout of the scale tensors

Same trap as MXFP8. `sf_A` is logically `(M, K/16)` `ue4m3`, but the
physical layout for `tcgen05.mma.blockscaled` is interleaved per the
SM100 blockscaled config (`Sm1xxBlockScaledConfig<16>`). The exercise
should build SF in PyTorch in the natural layout, pack it via the CuTe
DSL helper, and stage it into TMEM via TMA along with the operands.

The per-tensor `f32` scales are just two scalars passed as kernel
arguments; they don't need any layout.

## Measurements

On `(M, N, K) ∈ {(2048,)*3, (8192,)*3}`:

- TFLOP/s. NVFP4 peak on GB200 is ~10 PFLOP/s (≈2× MXFP8). Verify.
- Numerical error vs `f32` reference: max abs and RMSE.
- Plot RMSE for **NVFP4 vs MXFP4 vs MXFP8 vs DeepSeek-FP8 (ex12 V3)
  vs per-row FP8 (ex12 V2)** on the same input distribution. This is
  the precision/throughput Pareto frontier.
- Sweep input distributions:
  - `randn(M, K)` — well-behaved, NVFP4 should be very close to MXFP8.
  - Normal with one outlier per row at 100× — NVFP4's outer-tensor
    scale should still cope, MXFP4 (no outer scale) starts losing.
  - Heavy-tailed (Cauchy) — both formats struggle; quantify by how
    much.
- Confirm `tcgen05.mma.blockscaled` is the actual instruction emitted
  (PTX / SASS dump, ex5-style).

## Open questions to settle while implementing

- The CUTLASS type `nv_float4_t<e2m1>` says
  `ScaleFactorType = float_ue4m3_t`. Why **unsigned** FP8 for the
  scale instead of `e4m3` proper? (Answer hint: scales are
  positive-only by construction, so the sign bit is wasted; spending
  it on extra mantissa precision wins.)
- The constant `6.0 * 448.0 ≈ 2688` for the per-tensor scale assumes
  the worst-case per-block amax saturates both the `e2m1` operand and
  the `ue4m3` scale. Is that the right normalization, or do you want
  `6.0 * <ue4m3 max>` where the max representable `ue4m3` value is
  smaller than 448? Read the `ue4m3` spec carefully — the exponent
  range and bias differ from signed `e4m3`.
- For variant 2 (NVFP4 output), the epilogue needs the per-block amax
  of the `f32` accumulator *before* it can quantize. But the
  accumulator is striped across threads / TMEM. What's the right
  reduction pattern — warp shuffle within a 16-element block, or one
  more `tcgen05` op? CUTLASS example 72b shows this; pick it apart
  and explain.
- Compared to MXFP4, NVFP4 has a *smaller* block (16 vs 32) but a
  *more expensive* scale (FP8 vs E8M0). Which dominates the memory
  cost? Compute SF byte size = `M * K/16 * 1B` vs `M * K/32 * 1B` and
  confirm NVFP4's SF tensor is exactly 2× MXFP4's, plus the per-tensor
  scalar.
- For the back-to-back use case (NVFP4 GEMM → NVFP4 GEMM, e.g.
  attention `Q @ K^T` followed by `S @ V`), the intermediate output
  must be re-quantized to NVFP4 in the epilogue. How much error does
  the round-trip add on top of one GEMM's worth? Worth measuring.
- Does `tcgen05.mma.blockscaled` have any restriction on `K` (e.g.,
  multiple of 64 or 128 because of the operand alignment) beyond the
  obvious "multiple of 16"? Find out and document.
