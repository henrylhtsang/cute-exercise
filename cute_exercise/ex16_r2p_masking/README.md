# How do I write code that lowers to the R2P SASS instruction?

## Question

`R2P` ("Register-To-Predicate") is a SASS instruction on SM80+ that
copies bits from a single 32-bit GPR into the per-thread predicate
register file. One R2P unpacks up to 32 mask bits in one shot; the
follow-up code can then issue 32 cheap predicated selects without
having to AND-test the bit, compare to zero, and branch on each
iteration. The classic use is "apply a 32-wide column mask to a row
of softmax accumulators" — exactly what flash-attention's masking
code does.

The catch is that the compiler will only lower to R2P if you write
the masking loop in a very specific shape. Get the shape wrong and
you get 32 independent `LOP3 + ISETP + SEL` sequences instead of one
`R2P + 32 × predicated SEL`. The exercise is to nail down that shape,
verify R2P actually lands in SASS, and measure the win.

The reference pattern lives in
`flash-attention/flash_attn/cute/mask.py`:

```python
@cute.jit
def mask_r2p_lambda(X, mask_gen_fn, rank1=False):
    ncol = ...
    CHUNK_SIZE = 32
    for s in cutlass.range_constexpr(cute.ceil_div(ncol, CHUNK_SIZE)):
        mask = mask_gen_fn(s)              # Uint32 bitmask, 1=keep
        for i in cutlass.range_constexpr(min(CHUNK_SIZE, ncol - s*CHUNK_SIZE)):
            in_bound = cutlass.Boolean(mask & (Uint32(1) << i))
            c = s * CHUNK_SIZE + i
            X[c] = X[c] if in_bound else -Float32.inf
```

Plus the `shl_u32` / `shr_u32` helpers from `flash_attn/cute/utils.py`
which use **inline PTX** (`shl.b32` / `shr.u32`) instead of the
DSL-level `<<` / `>>` operator — Python's shift goes through LLVM,
where shift-by-32 is UB and the optimizer can poison the result. The
PTX shift clamps. This matters because the natural way to build the
"keep bits below `limit`" mask is `0xFFFFFFFF >> (32 - limit)`, and
`limit == 32` would otherwise produce a poison value that wrecks
later code paths.

## Why the code has to look weird

For R2P lowering the compiler needs to see, in one basic block:

1. A single `Uint32` value holding the mask (call it `m`).
2. Up to 32 selects, each gated on **bit `i` of `m`** for a distinct
   constexpr `i`. The bit test must be of the literal form
   `m & (1u << i)` with `i` a constexpr — anything else (a runtime
   `i`, a precomputed boolean array, a popcount-based path) breaks
   the pattern.
3. The selects must live in the same scope as the bit tests — no
   storing the booleans into a `bool` array first, no pulling the
   inner loop out into a helper that takes the bit as an argument.

Concretely:

- The outer loop over chunks (`s`) **must** be `range_constexpr`,
  not `range`, so the per-chunk mask register is a distinct
  compile-time entity.
- The inner loop over `i` **must** be `range_constexpr`. A runtime
  loop forces the bit index into a register, which kills R2P (you
  end up with `SHF + LOP3 + ISETP` per element).
- The branch **must** be the ternary form
  `X[c] = X[c] if in_bound else -inf`, not an `if in_bound: X[c] =
  ...` statement. The ternary becomes a single `SEL`/`FSEL`; the
  `if`-statement may become a real branch and miss R2P.
- The mask must come from `shl_u32` / `shr_u32` (inline PTX) so the
  edge case `shift == 32` is well-defined and the compiler doesn't
  delete the dependent code as poison.

Get any of these wrong and the SASS dump shows 32 separate
`LOP3.LUT + ISETP.NE + FSEL` sequences instead of one `R2P` followed
by 32 predicated selects.

## Plan

A small standalone kernel that doesn't pretend to be flash-attention
— just enough to drive R2P lowering and prove it shows up in SASS.

### Kernel

Inputs:
- `X`: `(M, N)` `float32` tile in GMEM, `N` a multiple of 32 (start
  with `N = 128`, `M = 64`).
- `limit`: an `int32` per-row threshold. For row `r`, columns
  `>= limit[r]` should be set to `-inf`; columns `< limit[r]` are
  passed through.

Body (inside a single CTA, one row per thread bundle):
1. Load the row into registers.
2. For each 32-column chunk `s`, build the keep-mask
   `m_s = shr_u32(0xFFFFFFFF, max(32 - (limit - s*32), 0))`
   (i.e. keep the leading `min(32, limit - s*32)` bits).
3. Apply the mask in the constexpr-unrolled inner loop above.
4. Store the row back to GMEM.

The "real" computation is irrelevant — what we want is the masked
write to land as one R2P + 32 predicated selects.

### Variants

- **A — R2P-friendly**: the pattern above. Outer `range_constexpr`
  over chunks, inner `range_constexpr` over bits, inline-PTX shift,
  ternary select.
- **B — runtime inner loop**: replace the inner `range_constexpr`
  with a plain `range`. Everything else identical.
- **C — boolean array**: precompute `bool[32]` from the mask in a
  first pass, then a second pass that consumes the booleans. Same
  semantics, different shape.
- **D — Python-level shift**: keep the constexpr shape but replace
  `shl_u32` / `shr_u32` with `Uint32(0xFFFFFFFF) >> Uint32(...)`.
  Verify whether the LLVM-level UB on `shift == 32` actually leaks
  through and produces wrong results, or whether the compiler
  happens to do the right thing here.
- **E — `if`-statement instead of ternary**: keep the constexpr
  shape and inline-PTX shift but write
  `if not in_bound: X[c] = -Float32.inf`. Does the compiler still
  fold to R2P + SEL, or does the `if` introduce a real branch?

For each variant we want to know:
- Does R2P actually appear in SASS? (count R2P opcodes per kernel)
- How many SASS instructions does the masking region take?
- How does runtime compare on a problem big enough to be
  ALU-bound (e.g. a long stripe over `M = 64K` rows)?

## Measurements

- SASS dump (`cuobjdump --dump-sass` or the `dump-ptx-sass` skill).
  Grep for `R2P`, count occurrences, and visually confirm the
  expected instruction sequence. Variant A should be ~1 R2P per
  chunk per row; variants B/C/D/E should have zero (or fewer).
- Instruction count in the masking region: NCU
  `smsp__inst_executed.sum` and `smsp__sass_thread_inst_executed_op_*`
  bucketed by op class.
- Wall-clock time for the kernel on `(M, N) = (65536, 128)`,
  `(65536, 256)`, `(65536, 512)`. The win should grow with `N`
  because the masked region scales with `N`.
- Correctness (variant D especially): bit-exact against a
  reference implementation. The interesting case is `limit == 32`
  / `limit == N` where the LLVM shift hits the UB boundary.

## Open questions to settle while implementing

- Is R2P really one instruction setting all 32 predicates, or does
  it write into a single predicate register (`P0..P6`) one bit at a
  time across the warp? Read the SM90 / SM100 SASS reference and
  confirm. The flash-attention code's effectiveness assumes the
  former.
- Does R2P lowering survive across thread granularities? The
  flash-attention pattern is per-thread (each thread holds its own
  fragment of accumulators); does the same shape work when the
  fragment crosses warp boundaries? Probably not — but worth
  checking with a `(M=1, N=1024)` variant where one warp owns
  the whole row.
- The compiler heuristic that decides "this is an R2P pattern" —
  is it stable across CUDA toolkit versions? Check on at least two
  toolkit versions and document the SASS diff if any.
- How does R2P interact with `select_with_else_pred` / `cute.where`?
  If the DSL ever adds a "select on a bitmask" primitive, it
  presumably wants to lower to R2P internally. Today we have to do
  it by hand.
- For non-Float32 element types (BF16, FP16, FP8), does the same
  pattern lower to R2P + the corresponding typed `SEL`, or does
  type promotion break the pattern? Try BF16 specifically since
  that's what flash-attention's accumulator sometimes is.
- Does R2P fight with `setmaxnreg`? If the kernel's register
  budget is tight, the compiler may spill the mask register and
  defeat the pattern. Try the same kernel with deliberately
  reduced `setmaxnreg` and see at what budget R2P disappears.
- Can we drive R2P lowering for *non-32-wide* chunks? The
  flash-attention code uses 32 because that's R2P's hardware
  width. A 16-wide pattern presumably doesn't lower to R2P; what
  about a 64-wide pattern split into two consecutive R2P regions?
- Is there a way to express the same intent in a "natural" shape
  (e.g. a runtime loop, a `where`-style op) that the compiler
  could in principle pattern-match to R2P, or is the constexpr
  unroll fundamental? If the latter, this is a pretty large papercut
  worth filing.
