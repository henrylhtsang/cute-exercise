# How does SMEM / L2 swizzle work?

## Question

Two distinct things go by the name "swizzle" on NVIDIA GPUs and they
solve different problems:

- **SMEM swizzle**: a bit-permutation on the SMEM address used by a
  thread, so that a warp's 32-thread access pattern lands on 32
  distinct banks instead of colliding. The classic alternative is `+1`
  padding; swizzle wastes no SMEM and is the canonical CuTe answer.
- **L2 / CTA-rasterization swizzle**: the order in which CTAs are
  scheduled across a 2D grid. Row-major is the default; "swizzled"
  rasterizations (group-by-N-rows, Hilbert, Morton) trade off
  per-row-stride locality against per-column-stride locality so that
  more of the inputs each CTA needs are already in L2.

This exercise uses **blockwise matrix addition** as the vehicle to see
both. The op is small enough that the kernel is fully memory-bound —
which is exactly the regime where these effects show up in the
numbers.

## Setup

Given:

- `A` of shape `(Q, d)`
- `B` of shape `(d, K)`
- output `C` of shape `(Q, K)`

where `Q` and `K` are divisible by `d` (typical `d = 128`). For each
`(d × d)` output block at block coordinates `(qi, kj)`:

```
C[qi*d:(qi+1)*d, kj*d:(kj+1)*d]
    = A[qi*d:(qi+1)*d, :]            # a (d, d) tile of A
    + B[:, kj*d:(kj+1)*d]            # a (d, d) tile of B
```

i.e. `A` is a column-stack of `Q/d` square tiles, `B` is a row-stack
of `K/d` square tiles, and `C[qi, kj] = A[qi] + B[kj]`. This is a
broadcast-add: each `A` tile is reused `K/d` times across the grid,
each `B` tile is reused `Q/d` times.

Reference (PyTorch):

```python
C = A.view(Q // d, 1, d, d) + B.view(1, K // d, d, d).transpose(-1, -2).contiguous().view(1, K // d, d, d)
# or equivalently, just two broadcasts:
C = A.unsqueeze(1) + B.permute(1, 0).reshape(K // d, d, d).unsqueeze(0)
C = C.permute(0, 2, 1, 3).reshape(Q, K)
```

## Why this op exposes swizzle

- One CTA per `(d × d)` output block. The CTA loads `A[qi]` and
  `B[kj]` into SMEM, adds, and stores `C[qi, kj]`.
- **SMEM side**: with `d = 128` and fp16/bf16, a row of `A[qi]` is
  256 B = 8 banks worth. The natural layout is fine for row-major
  reads, but the moment you want to read column-major (e.g. to mix in
  a transpose, or to feed an MMA atom that wants K-major operands),
  bank conflicts appear. This is where SMEM swizzle earns its keep.
- **L2 side**: each `A[qi]` is reused `K/d` times, each `B[kj]` is
  reused `Q/d` times. Row-major rasterization keeps `A[qi]` hot in L2
  but evicts `B[kj]` between passes. A grouped / Hilbert
  rasterization keeps both warm, at the cost of harder index math.

## Plan

Build up in variants in one file:

1. **Naive, no SMEM**: each thread reads one `A` element and one `B`
   element directly from GMEM, adds, stores. Floor for the comparison.
2. **TMA load A and B to SMEM, unswizzled, row-major add**: tile-load
   the two `(d × d)` operands, add row-major, TMA store. Should be
   conflict-free because the read is row-major matching the layout.
3. **Same, but read column-major from SMEM** (i.e. add `A[qi] + B[kj]^T`
   for the sake of the experiment): expect heavy bank conflicts; quantify
   with NCU
   (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld`).
4. **Column-major read, with `+1` pad**: bank conflicts → 0, SMEM
   wasted = `d * 2 B`.
5. **Column-major read, with CuTe swizzle atom** (e.g. `Swizzle<3,3,3>`
   for 16-bit, 8-bank-group): bank conflicts → 0, no wasted SMEM. Show
   the bit-permutation explicitly and which banks each thread hits.
6. **Row-major rasterization vs grouped rasterization**: same kernel
   as variant 2, but launch with two different
   `(qi, kj) ↔ blockIdx` mappings:
     - row-major: `qi = blockIdx.y, kj = blockIdx.x`
     - grouped: tile the grid into `G × G` super-blocks (`G = 8` is a
       good starting point) and walk super-blocks row-major, then
       walk inside each super-block row-major. This keeps a slab of
       `A[qi]` and a slab of `B[kj]` simultaneously hot in L2.
   Measure L2 hit rate (`lts__t_sector_hit_rate.pct`) and DRAM bytes
   (`dram__bytes.sum`) for each.

## Measurements

For each shape `(Q, K)` ∈ {square, tall, wide} and `d = 128`:

- Achieved bandwidth (GB/s) vs theoretical peak.
- SMEM bank conflicts per variant.
- L2 hit rate and DRAM bytes per rasterization.
- Time vs the PyTorch reference.

## Open questions to settle while implementing

- For a `(d, d)` fp16/bf16 tile with `d = 128`, what is the smallest
  CuTe swizzle (`Swizzle<B, M, S>`) that gives conflict-free
  column-major reads? Why those specific `B`, `M`, `S`?
- Does the SMEM swizzle answer change if `d` drops to 64? To 32?
- For the rasterization sweep, at what `(Q, K)` does the grouped
  schedule actually beat row-major? It can't help when the working
  set already fits in L2.
- Is there any benefit to combining a non-trivial rasterization with
  a hint like `cp.async.bulk.tensor`'s L2 cache policy descriptor, or
  is the rasterization alone sufficient?
- "L2 swizzle" is sometimes used to mean the hardware-level address
  hashing that L2 does internally (so consecutive addresses don't all
  hit the same L2 slice). Is there anything CuTe DSL exposes for that,
  or is it purely transparent?
