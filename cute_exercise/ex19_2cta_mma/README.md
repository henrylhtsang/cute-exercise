# How does 2-CTA MMA (`tcgen05.mma cta_group::2`) work, and when does it help?

## Question

Blackwell (SM100) introduced a new flavor of `tcgen05.mma`: a
**pair-MMA** mode where two peer CTAs in a thread block cluster
**cooperatively drive a single MMA**. PTX: `tcgen05.mma.cta_group::2`,
paired with the new TMA mode `cp.async.bulk.tensor.<N>d.cta_group::2`
that splits an operand tile across the two peer CTAs' SMEM. CUTLASS
exposes the load side as `SM100_TMA_2SM_LOAD_*` (and a multicast
combo, `SM100_TMA_2SM_LOAD_MULTICAST_*`).

The mental model:

- Two CTAs at ranks `2N` and `2N+1` in the same cluster form a
  **pair**.
- They share **one** logical TMEM-resident MMA accumulator. Each
  CTA contributes half the operand SMEM and half the TMEM rows.
- Each CTA issues the `cta_group::2` TMA load (note: **both** CTAs
  issue, unlike multicast where only one issues). The hardware
  consolidates `complete_tx` bytes onto CTA0's mbarrier (CUTLASS
  models this with `Sm100MmaPeerBitMask = 0xFEFFFFFF`, which clears
  the peer bit so the second CTA's contribution is credited to
  CTA0).
- One `tcgen05.mma.cta_group::2` instruction then drives an MMA
  whose effective tile is **2x** the single-CTA tile in one of the
  spatial dims. The compute is split across both SMs; the result
  lives in the pair's shared TMEM.

The pitch is that you can issue larger MMAs without burning more
register pressure or SMEM per CTA, and you can keep the GEMM
mainloop simpler (one MMA per inner step, but it's twice as wide).
The cost is the cluster machinery, the pair-MMA-only TMEM layout,
and the fact that *both* CTAs must execute every TMA load and every
MMA — there's no "free pair" in occupancy terms.

This exercise builds the simplest possible 2-CTA MMA kernel from
scratch (single-tile, no pipelining, no warp-spec) and pulls apart
each piece to figure out what the hardware is really doing and when
it's worth the complexity.

## Setup

A single-tile FP8 (or BF16, both should work; start with BF16 for
debuggability) GEMM:

- `A`: `(M, K)` BF16 in GMEM, `M = 256`, `K = 64`.
- `B`: `(K, N)` BF16 in GMEM, `N = 256`.
- `D`: `(M, N)` FP32 accumulator out.

Pair tile shape: `(M_pair, N_pair, K_pair) = (256, 256, 64)`. Each
CTA in the pair owns half of `M_pair` (or half of `N_pair`,
depending on the pair-tile layout — see open questions). Cluster
shape `(2, 1)`.

This is just ex6 (single-tile TF32 matmul) but with `cta_group::2`
instead of `cta_group::1`. The matmul is correctness-trivial; what
we want to learn is the mechanics.

## Variants

### V0 — single-CTA `tcgen05.mma cta_group::1` baseline

Plain Blackwell single-CTA MMA on a `(128, 256, 64)` tile (so the
tile per CTA matches what one CTA owns in V1). One TMA load for
`A`, one for `B`. No cluster. Get this working first; this is
ex6-equivalent and the reference for correctness.

### V1 — `tcgen05.mma cta_group::2` minimal kernel

Cluster `(2, 1)`. Both CTAs:
1. Allocate their half of the pair-MMA TMEM. CUTLASS's pair-MMA
   layout has CTA0 and CTA1 owning interleaved or split TMEM rows
   — figure out the exact rule from the SM100 PTX docs and the
   `SM100_TMEM_*` helpers.
2. Issue `cp.async.bulk.tensor.<N>d.cta_group::2` for `A` and `B`.
   Each CTA gets its half of each operand in SMEM.
3. Set up the *pair* mbarrier (the one whose transaction bytes get
   consolidated via the peer-bit-mask trick).
4. Wait for the pair mbarrier — both CTAs proceed once both halves
   have landed.
5. Issue `tcgen05.mma.cta_group::2`. Both CTAs execute the
   instruction; the hardware drives one logical MMA across the
   pair.
6. Wait on the MMA completion mbarrier.
7. Each CTA reads its half of the TMEM accumulator into RMEM and
   stores its half of `D` to GMEM.

### V2 — V1 + multicast on `B` (or `A`)

For a non-square cluster you'd combine pair-MMA with multicast on
the operand that's shared across pairs. With cluster `(2, 1)` the
multicast-on-top doesn't kick in (mask of size 1 in the broadcast
dim). For this variant, expand the cluster to `(2, 2)`: two pairs,
each pair owns a different `N` slab, and `A` is multicast across
both pairs (since both pairs share the same `M` slab). Confirm the
PTX dumps `cta_group::2 ... multicast::cluster` and verify the HBM
read of `A` goes from `2x` (one per pair) to `1x` (one cluster-wide).

### V3 — V1 with deliberately mis-sized pair tile

Run V1 but with a pair tile that the hardware can't actually
deliver as a single MMA (e.g. `M_pair = 128`, where the pair-MMA
shape rules want `>= 256` in the M dim). Goal: document the failure
mode. Either the kernel fails to compile, the PTX falls back to two
separate single-CTA MMAs, or it runs but wastes the pair lanes.

### V4 — V1 split into two `cta_group::1` MMAs for direct comparison

Equivalent compute, two single-CTA MMAs in two different CTAs (no
cluster, no pair). Same TMEM accumulator size in aggregate; same
operand SMEM. Same total HBM bytes. The question this answers:
**given the same compute, does the pair-MMA path actually run
faster, or is it just a cosmetic API?** If V1 is faster, by how
much, and where does the time go (SMEM bytes? TMEM bandwidth? MMA
issue overhead)? If V4 is faster, why even bother with `cta_group::2`?

## Measurements

Hardware: GB200 (SM100). Compare V0 / V1 / V4 head-to-head.

For each variant:

- TFLOP/s on the pair tile, with K replicated to make the kernel
  long enough to be a steady-state measurement (e.g. iterate K=64
  → K=4096 by replication; that does redundant compute but it's
  fine for measuring throughput).
- HBM read bytes (`dram__bytes_read.sum`). V1 and V4 should be
  identical here; if they're not, the hardware is doing something
  different about operand fetch.
- TMEM accesses: NCU `tmem__*` counters (verify name in
  `--query-metrics`). V1 should show *fewer* TMEM access events
  per output element (one per pair vs one per CTA).
- MMA instruction count: `sm__inst_executed_pipe_tensor.sum` and
  the `tcgen05` per-pipeline counters. V1 should issue half as
  many MMAs as V4 (one per pair vs one per CTA) but each MMA does
  twice the work.
- Cluster network traffic for V2: confirm the multicast actually
  fired and `lts__t_sectors_aperture_device_op_multicast.sum` is
  nonzero.
- Per-CTA register count, SMEM usage, and TMEM usage. V1 should be
  cheaper *per pair* but each CTA in the pair still pays full
  occupancy cost — document the actual numbers.

## Plan

### Step 0 — get V0 working

Single-CTA MMA, tile `(128, 256, 64)`. This is mostly ex6; pull
that kernel forward, swap to BF16/FP8, confirm bit-exact against
PyTorch.

### Step 1 — pair-MMA mechanics

Read the SM100 PTX section on `tcgen05.mma.cta_group::2` carefully.
Specifically nail down:

- The TMEM layout: which rows belong to which peer CTA?
- Operand SMEM split: does CTA0 hold `A[:M/2]` and CTA1 hold
  `A[M/2:]`, or is it interleaved? Same for `B`.
- The pair mbarrier semantics: is there one shared mbarrier, or two
  with the peer bit trick? CUTLASS uses the peer-bit trick to
  consolidate; the PTX-level invariant is "all transaction-bytes
  arrive on CTA0's mbarrier".
- Which thread in the pair issues the MMA? In CUTLASS, both CTAs
  execute the MMA instruction; the hardware coalesces. Verify by
  predicating the MMA on "rank 0 only" and watching the kernel
  produce wrong output.

### Step 2 — V1, the minimal pair-MMA kernel

Implement the steps in V1 above. Get bit-exact against the V0
reference on a single tile. Then dump PTX/SASS and verify:

- `cp.async.bulk.tensor.<N>d.cta_group::2` is emitted (twice — once
  per CTA in the pair).
- The mbarrier setup uses the peer-bit-mask trick.
- `tcgen05.mma.cta_group::2` shows up.
- The TMEM allocation uses the pair-aware allocator.

### Step 3 — V4, the side-by-side reference

Convert V1 into two independent single-CTA MMAs in two CTAs of the
same grid (no cluster). Same total work, no pair-MMA. This is the
direct comparator to V1 — same compute, same HBM, different
mechanics.

### Step 4 — V2, multicast on top

Expand cluster to `(2, 2)` and add `cta_group::2 ... multicast::cluster`
to the operand that's shared across pairs. Confirm PTX, confirm
HBM bytes drop, measure speedup over V1.

### Step 5 — write up the numbers

Build the perf table. Document:
- V1 vs V0: is the pair speedup just "two CTAs do work faster than
  one"? (Yes, but quantify how close to 2x.)
- V1 vs V4: same compute, different mechanics — does the pair-MMA
  hardware path actually win, and why?
- V2 vs V1: what does multicast add on top.

## Open questions to settle while implementing

- Pair-MMA TMEM layout. The SM100 docs give the formal rule; the
  practical question is "given a `(M_pair, N_pair)` MMA shape, how
  many TMEM rows do I need in CTA0 vs CTA1, and at what column
  offsets?". Read off the rule, write it down explicitly here, and
  verify with a kernel that allocates the wrong amount and watches
  for a TMEM out-of-bounds fault.
- What happens if only one CTA in the pair executes the
  `tcgen05.mma.cta_group::2`? Does the hardware hang, produce wrong
  output, or does it actually still drive a half-MMA on just the
  one peer? PTX says both CTAs must issue — verify the failure mode
  empirically.
- The peer-bit-mask trick (`Sm100MmaPeerBitMask = 0xFEFFFFFF`) makes
  CTA1's contribution to `complete_tx` count toward CTA0's
  mbarrier. What does CTA1's mbarrier look like — does it just
  never get arrived-on? Or do you arrive on it too with a non-tx
  arrive? Read the cute pipeline code (`SM100`-flavored `pipeline_*`)
  and document.
- Pair-MMA + warp-spec: in a real Blackwell kernel the producer and
  consumer warpgroups are separate, and the producer issues the
  TMA / MMA. With `cta_group::2` both CTAs have their own producer
  warpgroup that has to be in lockstep enough to issue the MMA
  together. How tight does the lockstep have to be? Probably "both
  arrive at the same MMA in the same iteration" — confirm.
- Does `cta_group::2` *require* a cluster of exactly 2, or can the
  cluster be larger (e.g. `(4, 1)` with two pairs)? Looking at
  CUTLASS's `SM100_TMA_2SM_LOAD_MULTICAST_*` it clearly composes
  with multicast across multiple pairs. Verify the hardware
  constraint: cluster size must be even; pairs must align on
  `cluster_rank % 2 == 0` boundaries.
- What's the actual perf of a `cta_group::2` MMA vs two
  `cta_group::1` MMAs of the same total shape? Hypothesis: pair-MMA
  is faster because it amortizes operand fetch and MMA issue over
  twice the work. Measure. If it's *not* faster, the only reason to
  use it is the larger logical tile per cluster — and that benefit
  evaporates if you can't pipeline pairs efficiently.
- Pair-MMA + FP4 / NVFP4: ex14 already uses single-CTA blockscaled
  MMAs. Does `cta_group::2` change the scale-factor TMEM layout?
  (Almost certainly yes — the scale factor metadata has to be
  partitioned across the pair the same way the accumulator is.)
- Does TMEM-to-RMEM load (the read-back at the end) work the same
  way per-CTA, or does each CTA only "see" its half of the TMEM
  accumulator? Verify that CTA0 reading TMEM rows that "logically
  belong" to CTA1 either errors or returns garbage.
- Cluster shape constraints: pair-MMA wants the pair axis along a
  specific cluster mode. For a `(2, 2)` cluster, are the pairs
  along the row or the column? CUTLASS picks one — verify and
  document.
- Pair-MMA in MoE: each expert's MMA is independent, but if two
  experts in the same cluster could share `A` (the activations),
  pair-MMA + multicast on `A` would be a clean fit. Out of scope
  for this exercise; flag for a follow-up.
- What exactly does `cta_group::2` cost in instruction-issue
  latency on each peer? Is the MMA issue blocking on the peer's
  arrival at the same instruction, or is it fire-and-forget once
  both CTAs have issued? Microbenchmark a kernel that deliberately
  delays one peer (e.g. one extra `__nanosleep`) and see whether
  the other peer waits.
