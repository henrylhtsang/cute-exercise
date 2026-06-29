# How do I write a fast LayerNorm for different sizes?

## Question

LayerNorm normalizes each row of an `(M, N)` activation:

```
y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
```

The fast strategy depends heavily on the normalized dimension `N`:

- For small `N` (e.g. `256`–`1024`), a single warp or block can hold the
  whole row in registers/SMEM and do a one-pass reduction.
- For large `N` (e.g. `4096`–`16384`), the row no longer fits comfortably,
  so we need a two-pass (or online/Welford) approach and care about how
  many threads cooperate per row.

So: how do we write a LayerNorm in CuTe DSL that stays fast across this
range of `N`? Specifically:

1. When is one-pass (sum and sum-of-squares together) numerically good
   enough versus a two-pass mean-then-variance scheme?
2. How should the block size and rows-per-block change as `N` grows?
3. How close can we get to the memory-bandwidth roof, given LayerNorm is
   memory bound?

## Setup

Implement a CuTe DSL LayerNorm with:

- vectorized global loads/stores;
- a warp/block-level reduction for mean and variance;
- a sweep over `N` (small → large) comparing against `torch.nn.functional`
  for correctness and against the bandwidth roofline for speed.

## Answer

_TODO: fill in after measurements on GB200._
