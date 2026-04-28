# How do I write a transpose in CuTe DSL?

## Question

Out-of-place 2D transpose: `B[i, j] = A[j, i]` for an `M × N` tensor.
Looks trivial, but it's the canonical "layout matters" exercise —
naive implementations tank on either the load or the store side
because one of them ends up strided in GMEM.

What's the right shape of a CuTe DSL transpose? Specifically:

- How do I express the transpose purely as a layout swap (no data
  movement in the math), and let the copy atoms do the work?
- Where does the transpose actually happen — in the GMEM→SMEM load,
  in SMEM, or in the SMEM→GMEM store? (i.e., do I read row-major and
  write column-major, or read column-major and write row-major, or
  shuffle in SMEM via swizzled layouts?)
- How do I avoid SMEM bank conflicts on the in-SMEM step? (Padding vs
  swizzle.)
- TMA or `cp.async`? TMA descriptors are 2D-aware — can a single TMA
  load + transposed SMEM layout + TMA store give the right answer
  without any thread-level shuffling?

## Plan

Build up in layers, each a separate variant in the same file:

1. **Naive thread-per-element**: each thread reads `A[j,i]`, writes
   `B[i,j]`. Either the load or store is strided. Measure as the
   floor.
2. **SMEM staging, no swizzle, no padding**: tile-load A into SMEM
   row-major, threads read SMEM column-major, store B row-major.
   Expect heavy bank conflicts — quantify them with NCU
   (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld`).
3. **SMEM with `+1` padding**: classic fix, see bank conflicts go to
   zero.
4. **SMEM with swizzled layout**: prefer this over padding (no wasted
   SMEM). Use a CuTe swizzle atom that maps the column-major read to
   conflict-free banks.
5. **TMA load + TMA store with transposed descriptors**: build the B
   descriptor with swapped strides so the SMEM tile stays as-is and
   the store handles the transpose. Or vice versa on the load side.

Measurements vs a reference (`A.T.contiguous()` in PyTorch):

- Bandwidth achieved (GB/s) for each variant at a few shapes
  (square, tall, wide).
- SMEM bank conflicts (NCU).
- GMEM coalescing on both load and store sides.
- Steady-state vs first-call (TMA setup cost, if any).

Open questions to settle while implementing:

- Is there a CuTe DSL idiom for "swap modes 0 and 1 of a tensor's
  layout" that I should be using everywhere instead of building two
  separate descriptors?
- For TMA, is the transposed-descriptor trick actually free, or does
  it just push the shuffling into the TMA engine and end up
  bandwidth-bound the same way?
