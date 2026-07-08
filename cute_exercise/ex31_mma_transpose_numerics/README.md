# Does transposing the MMA operand roles affect numerics?

## Question

Take two BF16 matrices in the CuTe MMA canonical shape:

```text
A: 128 x 64
B: 128 x 64
```

The custom kernel consumes both operands in the requested tensor-core operand
major mode and computes:

```text
C_direct = A @ B.T
```

The transpose-equivalent product swaps the MMA operand roles and transposes the
answer back:

```text
C_swapped = (B @ A.T).T
```

Mathematically these are the same values:

```text
C_direct[i, j] = sum_k A[i, k] * B[j, k]
C_swapped[i, j] = sum_k B[j, k] * A[i, k]
```

The exercise asks the stricter tensor-core question:

```text
Are C_direct and C_swapped bitwise identical when both are computed by the
same custom CuTe DSL tcgen05 MMA kernel?
```

It also checks whether source layout matters at the MMA instruction level. For
both `A` and `B`, build the source tensor as either:

- `row`: ordinary row-major / K-contiguous storage, mapped to
  `tcgen05.OperandMajorMode.K`;
- `column`: same logical values, but column-major physical storage via
  `x.T.contiguous().T`, mapped to `tcgen05.OperandMajorMode.MN`.

The tested layout pairs are:

```text
(row, row)
(row, column)
(column, row)
(column, column)
```

## Implementation

The runner uses the custom CuTe DSL tensor-core kernel body from:

```text
cute_exercise/ex21_tcgen05_mma/kernel.py
```

and a layout-parameterized host wrapper:

```text
cute_exercise/ex31_mma_transpose_numerics/layout_tcgen05_mma.py
```

That wrapper builds a Blackwell `tcgen05` BF16 MMA with explicit A/B
`OperandMajorMode` values, FP32 accumulation, and BF16 output. Its tile shape
is:

```text
M x N x K = 128 x 128 x 64
```

Run the sweep with:

```bash
python -m cute_exercise.ex31_mma_transpose_numerics.run_custom_tcgen05_transpose_sweep \
  --output cute_exercise/ex31_mma_transpose_numerics/artifacts/custom_tcgen05_mma_transpose_sweep.json
```

The helper module records:

- whether `A @ B.T` equals `(B @ A.T).T` bit-for-bit;
- max absolute error and max ULP distance;
- the A/B MMA operand-major modes;
- source strides and kernel operand strides;
- whether each direct result also matches the row/row direct baseline.

## Answer

Artifact:

```text
cute_exercise/ex31_mma_transpose_numerics/artifacts/custom_tcgen05_mma_transpose_sweep.json
```

Summary from the local GB200 run:

```text
kernel                  shape          dtype  accumulator  cases  mismatches  max_abs_err  max_ulp
----------------------  -------------  -----  -----------  -----  ----------  -----------  -------
custom_tcgen05_mma_ex21 128x128x64     bf16   f32          4      0           0            0
```

Per-layout result:

```text
A layout  B layout  A MMA mode  B MMA mode  A stride  B stride  same_bits  max_ulp
--------  --------  ----------  ----------  --------  --------  ---------  -------
row       row       K           K           (64, 1)   (64, 1)   yes        0
row       column    K           MN          (64, 1)   (1,128)   yes        0
column    row       MN          K           (1,128)   (64, 1)   yes        0
column    column    MN          MN          (1,128)   (1,128)   yes        0
```

Every direct result also matches the row/row direct baseline bit-for-bit. In
this custom kernel, swapping the MMA operand roles for the transpose-equivalent
product did not perturb the BF16 output bits.

I also dumped PTX with:

```bash
CUTE_DSL_KEEP_PTX=1 CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=/tmp/cute_ex31_ptx_modes \
  python -m cute_exercise.ex31_mma_transpose_numerics.run_custom_tcgen05_transpose_sweep \
  --output /tmp/cute_ex31_ptx_modes/result.json
```

The dump produced four PTX files whose names include the operand modes:

```text
OperandMajorModeK_OperandMajorModeK
OperandMajorModeK_OperandMajorModeMN
OperandMajorModeMN_OperandMajorModeK
OperandMajorModeMN_OperandMajorModeMN
```

The `tcgen05.mma.cta_group::1.kind::f16` instructions are present in all four
files. The generated descriptor constants differ by operand mode: K-major
descriptors use `4611756662049538048`, while MN-major descriptors use
`4611756662083026944`, and mixed-mode files place those constants on the
expected A/B descriptor operands. So this exercise is checking actual
tensor-core MMA operand layout modes, not only source copies into one layout.

Representative PTX snippets:

```ptx
// A=row/K, B=row/K
or.b64    %rd1, %rd12, 4611756662049538048;  // A descriptor: K-major
or.b64    %rd2, %rd13, 4611756662049538048;  // B descriptor: K-major
selp.b32  %r48, 136324240, 136316048, %p2;
tcgen05.mma.cta_group::1.kind::f16 [%r7], %rd1, %rd2, %r8,
    {%r50, %r50, %r50, %r50}, %p14;
```

```ptx
// A=row/K, B=column/MN
or.b64    %rd2, %rd14, 4611756662049538048;  // A descriptor: K-major
or.b64    %rd3, %rd15, 4611756662083026944;  // B descriptor: MN-major
selp.b32  %r53, 136389776, 136381584, %p2;
tcgen05.mma.cta_group::1.kind::f16 [%r7], %rd2, %rd3, %r8,
    {%r55, %r55, %r55, %r55}, %p16;
```

```ptx
// A=column/MN, B=row/K
or.b64    %rd2, %rd14, 4611756662083026944;  // A descriptor: MN-major
or.b64    %rd3, %rd15, 4611756662049538048;  // B descriptor: K-major
selp.b32  %r53, 136357008, 136348816, %p2;
tcgen05.mma.cta_group::1.kind::f16 [%r7], %rd2, %rd3, %r8,
    {%r55, %r55, %r55, %r55}, %p16;
```

```ptx
// A=column/MN, B=column/MN
or.b64    %rd3, %rd16, 4611756662083026944;  // A descriptor: MN-major
or.b64    %rd4, %rd17, 4611756662083026944;  // B descriptor: MN-major
selp.b32  %r58, 136422544, 136414352, %p2;
tcgen05.mma.cta_group::1.kind::f16 [%r7], %rd3, %rd4, %r8,
    {%r60, %r60, %r60, %r60}, %p18;
```
