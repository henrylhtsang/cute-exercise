# How do I write a top-k kernel?

## Question

Top-k selection — pick the `k` largest values (and their indices) along a
dimension — is the routing primitive behind MoE gating and a common building
block on its own. For MoE, `k` is tiny (often `1`–`8`) and the reduced
dimension (number of experts, or vocab) can be small or large.

So: how do we write a fast top-k in CuTe DSL? Specifically:

1. For small `k`, when is a simple per-thread/per-warp bitonic or
   insertion-based selection better than a full sort?
2. How do we do the within-warp and within-block reduction to merge each
   lane's local top-k into the row's global top-k?
3. How does the best strategy change as the reduced dimension grows from
   a handful of experts to a large vocabulary?

## Setup

Implement a CuTe DSL top-k along the last dimension with:

- a warp/block-cooperative selection (e.g. iterative max-extraction for small
  `k`, or bitonic top-k);
- output of both values and indices;
- correctness checks against `torch.topk` and a benchmark across `k` and the
  reduced dimension size.

## Answer

_TODO: fill in after measurements on GB200._
