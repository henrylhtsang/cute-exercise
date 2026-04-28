# `cp.reduce.async.bulk` for vec add

## Question

Can I implement `C = A + B` using `cp.reduce.async.bulk` (the SMEM→GMEM
TMA-style copy-with-reduction) instead of a normal load/add/store?

The instruction does `gmem op= smem` atomically — there's no two-source
variant — so vec add has to be expressed as a sequence of reduce-adds
into `C`, not a single fused op.

## Plan

Two shapes to try:

1. Zero-init `C`, then issue two `cp.reduce.async.bulk.tensor` (ADD) —
   one with a tile of `A`, one with a tile of `B`.
2. Plain TMA store of `A` into `C`, then one `cp.reduce.async.bulk.tensor`
   (ADD) with a tile of `B`.

In CuTe DSL the only thing that changes vs ex3 is the store atom:

```python
cpasync.CopyReduceBulkTensorTileS2GOp(cute.ReductionOp.ADD)
```

Check correctness against ex3, then benchmark against ex3's TMA store
baseline. Expectation: slower than ex3 for pure vec add (extra GMEM
trip), since this instruction is really meant for split-K accumulation
and scatter-add, not elementwise binary ops. Goal is to quantify by how
much.
