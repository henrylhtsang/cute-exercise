# TMA load → elementwise add → TMA store (no warp specialization)

## Question

How do I wire up a TMA load from global into smem, do an elementwise add,
and then TMA-store the result back to global — without warp specialization
yet (producer/consumer split comes later)?

And: is this actually faster than a plain synchronous copy (`cp.async` /
vectorized ld/st) for an elementwise op?

## Plan

Write the TMA version, check correctness, benchmark against the sync-copy
baseline from the previous exercise.
