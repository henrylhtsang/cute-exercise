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

Results below are from the local GB200. There are two separate experiments:

1. `torch.mm`, including the literal full padded GEMM.
2. A custom CuTe DSL `tcgen05` GEMM path that keeps the CTA block shape at
   `128 x 128` and computes the full padded GEMM before slicing.

Keep these separate. The `torch.mm` padded experiment answers "does the library
produce the same bits when it sees a larger padded GEMM?" The custom `tcgen05`
experiment asks the same padded-GEMM question while holding the CuTe block shape
fixed at `128 x 128`.

### `torch.mm`

The harness uses `torch.mm` explicitly for every GEMM and enables
`torch.use_deterministic_algorithms(True)` while each case runs.

```bash
python -m cute_exercise.ex29_mma_offset_numerics.analysis \
  --dtype tf32 \
  --mode padded \
  --deterministic \
  --output cute_exercise/ex29_mma_offset_numerics/artifacts/tf32_padded_offset_sweep.json
```

Sparse sweep:

```text
m0, n0 in {
  0, 1, 2, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128
}
```

for both of these modes:

- `view`: compute `torch.mm(A_pad[m0:m0+128, :], B_pad[:, n0:n0+128])`. This
  keeps the logical GEMM shape at `128 x 128 x 128` and changes only storage
  offset / stride.
- `padded`: compute the full `(m0+128) x (n0+128)` padded GEMM and compare the
  bottom-right slice. This is the literal mathematical setup, but it also lets
  the library see a different problem shape and boundary structure.

Artifacts:

```text
cute_exercise/ex29_mma_offset_numerics/artifacts/tf32_view_offset_sweep.json
cute_exercise/ex29_mma_offset_numerics/artifacts/tf32_padded_offset_sweep.json
cute_exercise/ex29_mma_offset_numerics/artifacts/bf16_view_offset_sweep.json
cute_exercise/ex29_mma_offset_numerics/artifacts/bf16_padded_offset_sweep.json
cute_exercise/ex29_mma_offset_numerics/artifacts/f16_view_offset_sweep.json
cute_exercise/ex29_mma_offset_numerics/artifacts/f16_padded_offset_sweep.json
```

Sparse-sweep summary:

```text
dtype  mode    cases  mismatches  max_abs_err            max_ulp
-----  ------  -----  ----------  ---------------------  -------
tf32   view    169    0           0                      0
tf32   padded  169    0           0                      0
f16    view    169    0           0                      0
f16    padded  169    0           0                      0
bf16   view    169    0           0                      0
bf16   padded  169    2           4.76837158203125e-07   2
```

The control case `(m0, n0) = (0, 0)` is bitwise identical for all tested dtypes
and modes.

The remaining BF16 mismatches are small, but they are real bitwise mismatches.
With deterministic `torch.mm`, the full set is:

```text
(m0, n0) = (127, 127)
(m0, n0) = (128, 127)
```

Both are in `padded` mode. The `view` mode is bitwise identical for every
offset, which is the stronger check that changing only the base/storage offset
does not perturb this deterministic `torch.mm` path.

Dense BF16 sweep:

I then repeated the BF16 deterministic `torch.mm` sweep with every offset in
`0..256` for several output sizes:

```text
shape MxNxK    mode    cases   mismatches  max_abs_err            max_ulp
------------   ------  ------  ----------  ---------------------  -------
1x1x128        view    66049   0           0                      0
1x1x128        padded  66049   0           0                      0
1x128x128      view    66049   0           0                      0
1x128x128      padded  66049   0           0                      0
128x1x128      view    66049   0           0                      0
128x1x128      padded  66049   0           0                      0
128x128x128    view    66049   0           0                      0
128x128x128    padded  66049   21232       4.76837158203125e-07   2
```

The one-row / one-column cases did not reproduce the mismatch at all. The
mismatch requires the full `128 x 128` output tile and the full padded GEMM
shape. In the dense full-size padded sweep, the mismatches are still tiny
1-2 ULP BF16 differences, but there are many of them once the offset range is
expanded. All dense artifacts are stored as summary JSON:

```text
cute_exercise/ex29_mma_offset_numerics/artifacts/bf16_view_m1_n1_k128_offsets_0_256.json
cute_exercise/ex29_mma_offset_numerics/artifacts/bf16_padded_m1_n1_k128_offsets_0_256.json
cute_exercise/ex29_mma_offset_numerics/artifacts/bf16_view_m1_n128_k128_offsets_0_256.json
cute_exercise/ex29_mma_offset_numerics/artifacts/bf16_padded_m1_n128_k128_offsets_0_256.json
cute_exercise/ex29_mma_offset_numerics/artifacts/bf16_view_m128_n1_k128_offsets_0_256.json
cute_exercise/ex29_mma_offset_numerics/artifacts/bf16_padded_m128_n1_k128_offsets_0_256.json
cute_exercise/ex29_mma_offset_numerics/artifacts/bf16_view_m128_n128_k128_offsets_0_256.json
cute_exercise/ex29_mma_offset_numerics/artifacts/bf16_padded_m128_n128_k128_offsets_0_256.json
```

`torch.mm` conclusion:

- `torch.mm` view mode was bitwise identical for all tested dtypes and offsets.
  This suggests that changing only the storage/base offset of the same logical
  `128 x 128 x 128` GEMM did not perturb the deterministic library path tested
  here.
- `torch.mm` full padded mode is different: the library sees a different GEMM
  shape and boundary structure. TF32 and FP16 stayed bitwise identical in the
  sparse sweep, but BF16 full padded `128 x 128 x 128` showed real 1-2 ULP
  differences.
- Reducing the output to `1x1`, `1x128`, or `128x1` did not reproduce the BF16
  mismatch. The mismatch appears tied to the full `128 x 128` output tile in
  the larger padded GEMM.

Follow-up: a scalar `False` assignment to CUDA's BF16 reduced-precision
reduction flag did not remove the mismatch. On the same GB200 / Torch
`2.11.0+cu130` environment, retrying the known sparse mismatches with:

```python
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
```

left both `(m0, n0) = (127, 127)` and `(128, 127)` mismatched at
`max_abs_err = 4.76837158203125e-07` and `max_ulp = 2`. Comparing the flag-on
and flag-off outputs directly also produced identical baseline bits and
identical padded-GEMM bits for those cases, so this flag was not the source of
the observed BF16 offset sensitivity. Repeating the same check in a fresh
`cute-exercise` conda environment with Torch `2.13.0+cu130` reproduced the same
result: the sparse sweep still had exactly those two mismatches with the same
`max_abs_err` and `max_ulp`, and flag-on vs flag-off outputs were still
bitwise-identical.

The tuple form is different in Torch `2.13.0+cu130`:

```python
torch.backends.cuda.preferred_blas_library("cublaslt")
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
```

Without forcing `cublaslt`, the tuple assignment failed on the first `torch.mm`
with:

```text
RuntimeError: torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction(..., allow_splitk=False) requires the cuBLASLt backend
```

With `cublaslt` forced, the original two known mismatches `(127, 127)` and
`(128, 127)` became bitwise-identical. The sparse sweep as a whole did not
become bitwise-identical, though: it reported 58 mismatches, still with
`max_abs_err = 4.76837158203125e-07` and `max_ulp = 2`, at a different set of
offsets.

#### Optional side quest: default cuBLAS kernel selection

I used Nsight Compute on Torch `2.13.0+cu130` to compare the default cuBLAS
path for the baseline GEMM and the two sparse mismatches. The profiling harness
is:

```text
cute_exercise/ex29_mma_offset_numerics/profile_cublas_mm.py
```

Example capture:

```bash
ncu --profile-from-start off --target-processes all --set full \
  --force-overwrite -o /tmp/ex29_cublas_padded_127_127 \
  conda run -n cute-exercise python -m \
  cute_exercise.ex29_mma_offset_numerics.profile_cublas_mm \
  --case padded --m0 127 --n0 127 --backend cublas
```

NCU does not expose PTX for these closed cuBLAS kernels, but the kernel names
and SASS instruction markers are enough to show an algorithm switch:

```text
case                  full GEMM shape  same bits?  kernel
-------------------   ---------------  ----------  -----------------------------------------------
baseline              128x128x128      yes         nvjet_sm100_tst_64x8_64x16_2x4_h_bz_NNT
padded (127, 127)     255x255x128      no          cutlass_75_tensorop_bf16_s1688gemm_bf16_64x64
padded (128, 127)     256x255x128      no          cutlass_75_tensorop_bf16_s1688gemm_bf16_64x64
padded (127, 128)     255x256x128      yes         nvjet_sm100_tst_64x8_64x16_1x4_h_bz_NNT
padded (128, 128)     256x256x128      yes         nvjet_sm100_tst_64x8_64x16_1x4_h_bz_NNT
```

Launch metadata:

```text
case                  grid  block  cluster  registers/thread  dynamic smem/block
-------------------   ----  -----  -------  ----------------  ------------------
baseline              32    256    8        255               156.17 KiB
padded (127, 127)     16    128    0        96                16.38 KiB
padded (128, 127)     16    128    0        96                16.38 KiB
padded (127, 128)     128   256    4        255               156.17 KiB
padded (128, 128)     128   256    4        255               156.17 KiB
```

SASS markers:

```text
nvjet path:
  8 UTCHMMA
  2 F2FP.BF16.F32.PACK_AB
  STG.3D

cutlass_75 fallback:
  32 HMMA
  64 F2FP.BF16.F32.PACK_AB
  64 STG.E.U16
```

The mismatching `n0 = 127` cases therefore are not using the same Blackwell
tensor-memory MMA path as the baseline. They fall back to a CUTLASS-style
`64x64` BF16 tensorop kernel using warp-level `HMMA` and register accumulators.
The bitwise difference is consistent with a different tile decomposition,
accumulation schedule, and BF16 conversion/store epilogue.

### Custom `tcgen05` MMA

The custom path uses a fixed-shape CuTe DSL launcher:

```text
cute_exercise/ex29_mma_offset_numerics/fixed_tcgen05_mma.py
```

It fixes the tile shape at:

```text
MNK = 128 x 128 x 128
```

and uses the Blackwell CuTe dense GEMM path from `~/cutlass`. The baseline
correctness probe is:

```bash
python -m cute_exercise.ex29_mma_offset_numerics.run_fixed_tcgen05_probe \
  --output cute_exercise/ex29_mma_offset_numerics/artifacts/fixed_tcgen05_mma_baseline.json
```

Current result:

```text
same_bits:   true
max_abs_err: 0
max_ulp:     0
correct:     true
```

The baseline reference for this BF16 kernel is `torch.mm(A, B.T)`, not
`torch.mm(A.float(), B.float().T).to(torch.bfloat16)`. The latter can differ by
1 BF16 ULP because it is not the same BF16 GEMM implementation path.

The offset sweep computes the full padded GEMM, then compares the correct
bottom-right slice with the unpadded baseline. B is stored in the kernel's
canonical `(N, K)` layout, so the script computes `A_pad @ B_pad.T`:

```python
baseline = fixed_mma_interface(a, b)

logical_m = m0 + 128
logical_n = n0 + 128
physical_n = round_up(logical_n, 8)

a_pad = zeros((logical_m, 128))
b_pad = zeros((physical_n, 128))
a_pad[m0:m0+128, :] = a
b_pad[n0:n0+128, :] = b

d_pad = fixed_gemm_interface(a_pad, b_pad)
d_slice = d_pad[m0:m0+128, n0:n0+128]
```

The extra physical B/output columns are zero and are outside the requested
logical padded GEMM. They are present only so the upstream dense GEMM TMA store
path has a supported output leading dimension. The compared slice is the
mathematical `A_pad @ B_pad.T` slice for the logical padded problem.

Artifacts:

```text
cute_exercise/ex29_mma_offset_numerics/artifacts/fixed_tcgen05_mma_baseline.json
cute_exercise/ex29_mma_offset_numerics/artifacts/fixed_tcgen05_mma_offset_sweep.json
cute_exercise/ex29_mma_offset_numerics/artifacts/fixed_tcgen05_mma_offset_sweep_aligned.json
cute_exercise/ex29_mma_offset_numerics/artifacts/fixed_tcgen05_mma_k_padding_sweep.json
```

Summary:

```text
kernel                   offsets               cases  mismatches  max_abs_err  max_ulp
----------------------   --------------------  -----  ----------  -----------  -------
custom tcgen05 fixed MMA sparse mixed offsets  169    0           0            0
custom tcgen05 fixed MMA aligned subset        36     0           0            0
```

Custom `tcgen05` conclusion:

- The fixed `128 x 128 x 128` BF16 `tcgen05` path matches BF16 `torch.mm(A,
  B.T)` bit-for-bit for the baseline.
- The custom padded GEMM path computes `A_pad @ B_pad.T`, slices
  `d_pad[m0:m0+128, n0:n0+128]`, and compares that slice with the unpadded
  fixed-kernel baseline.
- With the CuTe CTA block shape held at `128 x 128`, all tested custom padded
  GEMM cases are bitwise identical: 169 sparse mixed offsets, 0 mismatches,
  max ULP 0.

### K-padding variant

I also checked a different padding pattern where the zero padding is inserted
along the contraction dimension:

```text
A_pad = [zero, A]
B_pad.T = [B.T; zero]
```

Because the custom kernel stores B in canonical `(N, K)` form, the script builds
that as:

```python
a_pad[:, k0:k0+128] = a
b_pad[:, 0:128] = b
d_pad = fixed_gemm_interface(a_pad, b_pad)
```

The physical K dimension is rounded up to a multiple of 128 for the fixed CuTe
K tile; extra physical columns are zero. This is a different question from the
M/N padded-GEMM slice above: here the nonzero K ranges of A and B are shifted
relative to each other.

K-padding artifact:

```text
cute_exercise/ex29_mma_offset_numerics/artifacts/fixed_tcgen05_mma_k_padding_sweep.json
```

Summary:

```text
kernel                   k offsets  cases  mismatches  max_abs_err  max_ulp
----------------------   ---------  -----  ----------  -----------  -------
custom tcgen05 fixed MMA sparse     13     12          73           66596
```

K-padding conclusion:

- `k0 = 0` is the control and matches the baseline.
- Every tested nonzero `k0` mismatches the baseline.
- At `k0 = 128`, the A and B nonzero K ranges do not overlap at all, so the
  result is exactly zero.
- For partial shifts such as `k0 = 1` or `k0 = 8`, the result is neither the
  baseline nor zero; it only accumulates over the overlapping subset of K.

### Overall conclusion

- If the same MMA instructions consume the same operand fragments in the same K
  order and the same accumulator/epilogue path is used, bitwise equality is the
  expected result. Offsetting the mathematical answer does not itself change
  floating-point associativity.
- A larger padded GEMM is not guaranteed to preserve that implementation path.
  The library can see different dimensions, boundary tiles, base alignment, and
  strides, so bitwise equality becomes an implementation property rather than a
  mathematical property.
- In this GB200 run, `torch.mm` full padded BF16 can differ by 1-2 ULP, while
  the custom `tcgen05` padded GEMM path with fixed `128 x 128` blocks is
  bitwise identical over the tested sparse offsets.
- K-padding is different: moving A and B in opposite directions along the
  contraction dimension changes which products are accumulated, so the output is
  not bitwise independent of that offset.
