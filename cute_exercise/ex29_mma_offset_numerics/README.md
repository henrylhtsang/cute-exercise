# Does an MMA tile give bitwise-identical numerics when the result is offset?

## Question

Take two dense matrices:

```text
A: 128 x 128
B: 128 x 128
C = A @ B
```

Now embed the same multiplication inside a larger padded problem. For arbitrary
non-negative offsets `m0` and `n0`, define:

```text
A_pad = [ Z_M ; A ]      shape: (m0 + 128) x 128
B_pad = [ Z_N , B ]      shape: 128 x (n0 + 128)
```

where `Z_M` is an `m0 x 128` zero matrix and `Z_N` is a `128 x n0` zero
matrix. The padded product is:

```text
C_pad = A_pad @ B_pad    shape: (m0 + 128) x (n0 + 128)
```

The bottom-right `128 x 128` slice is mathematically the same product:

```text
C_slice = C_pad[m0:m0 + 128, n0:n0 + 128]
        = A @ B
```

The exercise asks a stricter implementation question:

```text
Is C_slice bitwise identical to C when both are computed with the same MMA
dtype, accumulator dtype, and kernel strategy?
```

Sweep `m0` and `n0`, especially values that are not multiples of the MMA tile
shape, shared-memory swizzle period, global-memory alignment, or output vector
width.

## Why this matters

For a fixed output element `(i, j)`, the mathematical reduction is unchanged:

```text
C[i, j]       = sum_k A[i, k] * B[k, j]
C_slice[i, j] = sum_k A_pad[m0 + i, k] * B_pad[k, n0 + j]
              = sum_k A[i, k] * B[k, j]
```

If the kernel issues the same sequence of MMA instructions, feeds each MMA the
same operand values in the same K order, and stores the same accumulator dtype,
the result should be bitwise identical. Floating-point addition is not
associative, but this setup does not change the reduction order by itself.

The interesting cases are the ones where the offset changes the implementation
path even though the math is identical:

- the useful `128 x 128` region lands at a different tile-local coordinate;
- the first or last output tile is partially masked;
- global or shared-memory addresses have different alignment;
- the kernel uses a different vectorized load/store path;
- TMA, `cp.async`, or SMEM swizzling sees a different box coordinate;
- the library/kernel selector chooses a different tile shape for the larger
  padded problem;
- split-K, atomics, or another parallel reduction changes the order in which
  partial sums are combined.

This exercise is meant to distinguish "MMA is deterministic for the same
fragments" from "a larger padded GEMM necessarily computes the same fragments in
the same order."

## Setup

Use square `A` and `B` inputs with shape `(128, 128)`. Start with `float16`
operands and `float32` accumulation, then repeat for the formats we care about
most:

- `tf32` operands with `float32` accumulation;
- `float16` or `bfloat16` operands with `float32` accumulation;
- optionally FP8 operands with `float32` accumulation.

For each dtype:

1. Generate `A` and `B` once.
2. Compute the baseline `C = A @ B` with the exercise kernel.
3. For each `(m0, n0)` pair, build `A_pad` and `B_pad`, compute `C_pad`, and
   compare `C_pad[m0:m0 + 128, n0:n0 + 128]` with `C`.
4. Report both:
   - `torch.equal(C_slice, C)` for bitwise equality;
   - max absolute error and max ULP distance when bitwise equality fails.

Suggested offset sweep:

```text
m0, n0 in {
  0, 1, 2, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128
}
```

Include at least these pairings:

- `(m0, n0) = (0, 0)` as the control;
- M-only offsets `(m0, 0)`;
- N-only offsets `(0, n0)`;
- mixed offsets `(m0, n0)`;
- offsets that are exact multiples of the output tile shape;
- offsets that are one element before and after those multiples.

## Expected answer shape

The answer should not be a single yes/no. It should say which assumptions make
bitwise equality expected, and which implementation choices break those
assumptions.

A useful final table would look like:

```text
dtype   accumulator  m0  n0  same_bits  max_abs_err  max_ulp  suspected_cause
-----   -----------  --  --  ---------  -----------  -------  ---------------
f16     f32          0   0   yes        0            0        control
f16     f32          1   0   ...
tf32    f32          0   1   ...
```

## Things to prove while implementing

- Does a hand-written single-tile MMA kernel stay bitwise identical for every
  offset when the K loop is unchanged?
- Does a tiled multi-CTA kernel remain bitwise identical, or do boundary tiles
  and masks perturb the MMA issue pattern?
- Does changing only the output offset `(n0)` affect the epilogue store path
  without affecting the accumulator bits?
- Does changing the input row offset `(m0)` affect global-load alignment enough
  to change the values fed into MMA?
- If bitwise equality fails, is the first differing value already present in
  the accumulator before the epilogue, or introduced by the output cast/store?

## Answer

_TODO: fill in after implementing the offset sweep._
