# How do I write an MXFP8 matmul?

## Question

MXFP8 is the OCP "Microscaling" FP8 format (the MX in MXFP8 stands for
Microscaling, not Nvidia-specific). One MXFP8 *block* is:

- **32 consecutive elements** along the K dimension, stored as `e4m3`
  (or `e5m2`) — `1 + 4 + 3` (or `1 + 5 + 2`) bits each.
- **One shared scale** for that block, stored as `ue8m0`: an unsigned
  8-bit power-of-two exponent, no sign, no mantissa. The real value is
  `2 ** (ue8m0 - 127)`, so legal scales are exactly the powers of two
  from `2^-127` to `2^127`.

Dequantization is

```
A_real[i, k] = A_q[i, k] * 2 ** (sf_A[i, k // 32] - 127)
```

i.e. each block of 32 consecutive K elements gets its own
power-of-two multiplier. There is no per-tensor outer scale (contrast
NVFP4, ex14, which has both).

The Blackwell SM100 instruction `tcgen05.mma.blockscaled` consumes the
two scale tensors `SFA`, `SFB` directly — the scale multiply is *fused
into the MMA*, not done in software at the K-tile boundary like
DeepSeek-style FP8 (ex12 V3). The math problem here is therefore: what
exactly does that instruction compute, what shape and layout do `SFA`
and `SFB` need to be in, and how do you build them from an `f32`
reference tensor?

## The math

Per output element:

```
D[i, j] = sum_{t=0}^{K/32 - 1}
            2 ** (sf_A[i, t] - 127)
          * 2 ** (sf_B[j, t] - 127)
          * sum_{k in block t} A_q[i, k] * B_q[k, j]
```

i.e., per 32-wide K-block, the inner FP8 dot product is multiplied by
`2 ** (sf_A + sf_B - 254)` and accumulated into an `f32` running sum.
This is structurally the same as ex12 V3 (blockwise FP8) but:

- block size in K is fixed at **32**, not 128;
- scales are E8M0 (power-of-two), so the multiply is a **pure exponent
  add**, no mantissa;
- the multiply is done by the MMA instruction itself, so there's no
  per-K-tile flush in the software loop;
- both `SFA` and `SFB` have the same per-row × per-K-block granularity
  (no asymmetry like DeepSeek's 1×128 vs 128×128).

CUTLASS exposes the operand type as `cutlass::mx_float8_t<e4m3>` (or
`e5m2`), and the scale type as `cutlass::float_ue8m0_t`.

## Why this matters

- Power-of-two scales mean no precision loss in the scale itself; the
  whole error budget goes to the FP8 mantissa. Compared to FP8 with
  `f32` blockwise scales (ex12 V3), MXFP8 trades a tiny bit of scale
  precision for a much cheaper hardware path — the mma.blockscaled
  instruction is essentially free relative to plain `tcgen05.mma`.
- 32-element blocks are small enough that per-K-tile dynamic range is
  preserved across activations with very different magnitudes, which
  is why MXFP8 is the OCP-blessed format for FP8 inference and is
  showing up in vLLM / TRT-LLM weights.
- The "scale layout" is non-trivial: `SFA` and `SFB` are stored in a
  swizzled / interleaved form that matches what
  `tcgen05.mma.blockscaled` expects. Getting that layout wrong is the
  most common bug in a first MXFP8 kernel — the GEMM compiles, runs,
  and produces nonsense.

## Plan

Two variants of the same kernel, both on `(M, N, K) =
(2048, 2048, 2048)` (K must be a multiple of 32):

1. **MXFP8 with `e4m3` data**
   - Quantize an `f32` reference `A_ref` to `(SFA, A_q)`:
     - For each `(i, t)`: take `A_ref[i, 32*t:32*(t+1)]`,
       `amax = block.abs().amax()`,
       `sf = ceil(log2(amax / 448.0))` (saturate to E4M3's max),
       `SFA[i, t] = sf + 127` (E8M0 bias),
       `A_q[i, 32*t:32*(t+1)] = (block / 2**sf).to(e4m3)`.
   - Same for `B` with respect to its K dimension.
   - Build the `tcgen05.mma.blockscaled` call. CuTe DSL should expose
     this through a blockscaled MMA atom; SFA/SFB are loaded into
     dedicated descriptors that the instruction reads alongside A/B.
   - Reference: dequantize `(SFA, A_q)` back to `f32`, dequantize
     `(SFB, B_q)`, compute `f32` matmul, compare.

2. **MXFP8 with `e5m2` data**
   - Same as variant 1 but `e5m2`. Range is wider (`±57344`), mantissa
     is narrower (2 bits). Show that high-dynamic-range inputs benefit
     and low-magnitude inputs lose precision relative to `e4m3`.

3. **(Optional) Mixed: `e4m3` × `e5m2`**
   - One operand `e4m3`, the other `e5m2`. The hardware allows this;
     the question is whether CuTe DSL's blockscaled atom does too, and
     whether SFA/SFB are still both `ue8m0`.

## Layout of the scale tensors

This is where to be careful.

- Logical shape of `SFA` is `(M, K/32)`, of `SFB` is `(N, K/32)`.
- Physical layout for `tcgen05.mma.blockscaled` is *not* row-major.
  CUTLASS calls it `Sm1xxBlockScaledConfig` (see
  `include/cutlass/detail/sm100_blockscaled_layout.hpp`) and it
  interleaves SF values to match the way the MMA instruction
  distributes scales across lanes within a warp.
- The exercise should build the SF tensors in PyTorch in the natural
  `(M, K/32)` layout, then call a CuTe DSL packer (or write one) that
  permutes them into the SM100 layout, and load that into TMEM via TMA.

## Measurements

For each variant, on `(M, N, K) ∈ {(2048,)*3, (8192,)*3}`:

- TFLOP/s. Blackwell MXFP8 peak should be ~2× the FP8 peak for the
  same operand type; verify.
- Numerical error vs the dequantized `f32` reference: max abs and
  RMSE.
- Compare error to ex12 V3 (DeepSeek-style 1×128/128×128). MXFP8
  should win on inputs with K-direction outliers (smaller block) and
  may lose slightly on smooth inputs (E8M0 scale rounds to a power of
  two, which is up to a 2× scale-quantization error per block).
- Sweep the input distribution: uniform, normal, and "K-direction
  outlier" (one element per row inflated 100×). Confirm MXFP8 handles
  the outlier input gracefully where per-tensor FP8 (ex12 V1) does
  not.

## Open questions to settle while implementing

- Where does the `2 ** (sf_A + sf_B - 254)` multiply actually happen
  inside `tcgen05.mma.blockscaled`? Is it a pre-multiply on the
  operands (effectively a per-block exponent shift before the
  multiply), or a post-multiply on the dot-product result before
  accumulating? The visible math is the same; the implementation
  choice shows up in PTX / SASS.
- What's the right way to build SFA / SFB in CuTe DSL? Is there an
  atom like `cute.make_blockscale_atom` that handles the
  interleaving, or do we have to call `Sm1xxBlockScaledConfig`
  manually?
- For inputs where a 32-wide K-block is all zero, what should `sf`
  be? Convention is `0` (which dequantizes to `2^-127 * 0 = 0`,
  fine), but check that the hardware doesn't trap on a zero-amax
  block.
- E8M0 has no representation for zero or negative numbers — only
  positive powers of two. What happens if a block's amax is zero
  (legal) or if rounding pushes the scale outside `[0, 254]`?
- Is there any benefit to padding K to a multiple of 32 with zeros
  so MXFP8 always works, vs. requiring the user to handle the tail
  themselves? The OOB question from ex11 suddenly matters here:
  TMA-loaded zeros at the K tail must produce a *valid* MXFP8 block
  (zero data, zero scale).
- Compare against an emulated MXFP8 path on Hopper (no
  `tcgen05.mma.blockscaled`): software dequant to bf16, then bf16
  GEMM. How much does the hardware blockscaled MMA actually buy you
  over emulation?
