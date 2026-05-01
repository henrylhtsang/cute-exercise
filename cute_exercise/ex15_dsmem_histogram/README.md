# How does distributed shared memory work? (cluster histogram)

## Question

Hopper introduced thread block clusters: a small group of CTAs (up to
8 portably, 16 with the opt-in attribute) whose shared memory is
addressable as one distributed pool. Any thread in any CTA of the
cluster can issue `ld.shared::cluster`, `st.shared::cluster`, or
`red.shared::cluster.<op>` against the SMEM of any *other* CTA in the
same cluster. The hardware routes the request over a cluster-internal
network; the visible programming model is "one big SMEM partitioned
across the cluster".

This exercise uses **a histogram with one shared bin table per
cluster** to see DSMEM end-to-end. We deliberately scope to a single
cluster — make it as large as the hardware allows — so the focus is
the DSMEM mechanics, not the grid-level reduction.

### Setup

Given:

- `x`: an `int32` (or `uint32`) input tensor of `N` elements with
  values in `[0, B)`.
- `B`: the number of bins. Pick `B` such that the bin table does
  *not* fit in one CTA's SMEM but *does* fit in the cluster's
  aggregate SMEM. On a GB200 / B200 the per-CTA SMEM ceiling is ~228
  KB, so an `int32` table that doesn't fit a single CTA needs
  `B > 228 KiB / 4 = 58368`. With a 16-CTA cluster we have
  `16 × 228 KiB ≈ 3.6 MiB` of DSMEM, i.e. up to `~900K` `int32` bins.
  A natural target is `B = 65536` (256 KB table, just above the
  per-CTA ceiling, comfortably inside the cluster).
- Output `hist`: an `int32` tensor of length `B` in GMEM.

Per spec:

```
hist[b] = sum_i (x[i] == b)   for b in [0, B)
```

### Why this is the right vehicle

- A histogram is *all atomics, no math*, so the kernel time is
  dominated by where those atomics land. That's exactly what we want
  to measure when comparing global atomics, per-CTA SMEM atomics, and
  cluster-DSMEM atomics.
- The bin table is the natural thing to "distribute across the
  cluster": owning CTA `c` holds bins
  `[c * B/cluster_size, (c+1) * B/cluster_size)` in its own SMEM, and
  every thread in the cluster issues a cluster-shared atomic against
  whichever CTA owns the bin its input mapped to. No CTA-local copy
  to reduce at the end of the cluster's work — the table *is* the
  cluster's SMEM.
- A bin table that doesn't fit in one CTA's SMEM forces DSMEM. A
  bin table that does fit becomes the obvious baseline (variant 2
  below).

## Variants

One CuTe DSL kernel, several launch configurations + small variations.
All single-cluster (one cluster covers the whole input):

### Variant 0 — global atomics baseline

Each thread reads its slice of `x`, computes `b = x[i]`, issues
`atomicAdd(&hist[b], 1)` directly in GMEM. No SMEM, no DSMEM. This is
the floor: gives us the number to beat.

### Variant 1 — per-thread / per-warp privatization in registers, then GMEM atomics

Standard "reduce contention" trick: each thread (or warp) keeps a
small private histogram in registers / SMEM, then dumps it to GMEM at
the end. Useful only when `B` is small; with `B = 65536` you can't
afford a per-thread copy. Skip if it doesn't fit; include if it does
for the comparison.

### Variant 2 — single-CTA SMEM histogram (no cluster)

Pick a smaller `B` (e.g. `B = 16384`, table = 64 KB) so the table
fits in one CTA's SMEM. Each CTA in the grid:
1. Zeros its SMEM table.
2. Streams its slice of `x`, atomic-adds into SMEM
   (`atom.shared.add.u32`).
3. After a CTA-wide barrier, reduces the SMEM table into the GMEM
   output via per-bin `atomicAdd`.

This is the canonical pattern when the table fits a CTA. We include
it to show the *cliff* — at `B = 65536` it stops working, which
motivates DSMEM.

### Variant 3 — one cluster, one DSMEM-distributed histogram (the point of the exercise)

Single cluster, size `C` (try `C = 8` and `C = 16`; the latter
requires `cudaFuncSetAttribute(...,
cudaFuncAttributeNonPortableClusterSizeAllowed, 1)`). The cluster
covers the whole input; the grid is `1 × 1 × 1` clusters.

Layout:
- Cluster-wide bin table of length `B`, partitioned: CTA `c` owns
  `bins[c * B/C : (c+1) * B/C]` in its own SMEM. Each CTA allocates
  `B/C * 4 B` of SMEM for its slice.
- Cluster startup: each CTA zeros its slice, then a *cluster-wide*
  arrival barrier (cluster `mbarrier`, not just CTA-local) gates the
  start of the atomic phase.
- Atomic phase: every thread reads its inputs, computes
  `b = x[i]`, derives `(owner, offset) = divmod(b, B/C)` (or a
  bit-shift if `B/C` is a power of two), and issues
  `red.shared::cluster.add.u32` with a cluster-distributed pointer
  into CTA `owner`'s slice. The DSMEM hardware routes the request.
- Cluster-wide barrier again to make sure all atomics have landed.
- Each CTA writes its slice of the table to GMEM (no atomic — the
  cluster owns the answer; we just dump it).

Math/correctness check is trivial: it's the same histogram, the
distribution and reduction are bit-exact integer.

### Variant 4 — cluster, but each CTA holds a *full* private copy in SMEM, then cluster-reduce

For comparison: each CTA in the cluster does a CTA-local SMEM
histogram on its slice of `x`, then the cluster cooperates to add
the per-CTA tables together (e.g. ring-style additions across DSMEM)
and dumps the sum. This needs `B * 4 B` per CTA, so it only works
for small-ish `B`. The point is to compare:

- Variant 3: distributed table, atomics travel across DSMEM at insert
  time.
- Variant 4: replicated table, atomics stay CTA-local at insert time;
  DSMEM is only used for the cross-CTA reduce at the end.

For `B = 65536` variant 4 is infeasible (256 KB × C doesn't fit per
CTA). For `B = 16384` it does fit — run both and see which wins, and
why.

## Measurements

Hardware: GB200 / B200 (cluster size 16 supported with opt-in).
Single cluster across the entire input.

For each variant on inputs of `N ∈ {1M, 16M, 256M}` `int32` values
sampled uniformly in `[0, B)`:

- Time / throughput (input GB/s; DRAM read bandwidth is the ceiling).
- DSMEM traffic, where applicable: NCU
  `lts__t_sectors_srcunit_tex_aperture_distributed.sum` and the
  `smsp__inst_executed_pipe_*` counters for cluster-shared atomics.
- L2 vs DRAM split (`lts__t_sector_hit_rate.pct`,
  `dram__bytes.sum`).
- Sweep input distribution:
  - **uniform**: every bin equally hot, atomics spread evenly across
    cluster CTAs in variant 3.
  - **skewed (Zipf or one-hot)**: most atomics land on a single
    owning CTA — DSMEM contention concentrated. Quantify the slowdown
    vs uniform.
  - **monotonic / sorted**: atomics march from the first owner to
    the last — exposes the cluster network's behavior under sequential
    access.
- Sweep cluster size: `C ∈ {1, 2, 4, 8, 16}`. `C = 1` is variant 2
  with the same code path. The interesting question is whether
  variant 3 scales linearly with `C` or whether the cluster network
  caps out earlier.

## Plan

1. Stand up variant 0 (GMEM atomics) and variant 2 (per-CTA SMEM
   atomics) first as baselines. Both are well-known and let you
   sanity-check correctness against `torch.bincount(x, minlength=B)`.
2. Move to variant 3:
   - Allocate a cluster of `C = 8`, then `C = 16` (with the opt-in
     attribute).
   - Use a cluster-aware `mbarrier` for the zero-init → atomic-phase
     → drain transitions. CuTe DSL should expose this through a
     cluster barrier atom; if not, drop to PTX.
   - The cluster-distributed pointer is the key DSMEM primitive:
     given an SMEM symbol and a target CTA rank within the cluster,
     produce a generic pointer that `red.shared::cluster` can
     consume. Look up what CuTe DSL exposes (`cute.cluster_smem_ptr`
     or similar) and confirm with a PTX dump.
3. Add variant 4 for the small-`B` regime. Measure side-by-side
   against variant 3 on `B = 16384` to learn whether atomics-during
   (variant 3) or atomics-then-reduce (variant 4) wins for histograms.

## Open questions to settle while implementing

- What's the maximum cluster size on this GPU? `cudaDeviceGetAttribute
  (cudaDevAttrClusterLaunchMaxBlocks)` says 8 portably; the
  non-portable attribute lets you go to 16 on Hopper / Blackwell but
  with caveats (occupancy implications). Verify experimentally and
  document.
- How much SMEM can one CTA actually carve out for the bin slice?
  The 228 KiB ceiling is shared with other SMEM uses (input staging,
  barriers); we want as much of it as possible for the bin table.
  What's the practical max per CTA, and therefore the max table
  length per cluster?
- Does `red.shared::cluster.add.u32` cost the same as a CTA-local
  `atom.shared.add.u32`, or is there a measurable per-hop latency
  premium? Try a microbenchmark: 1M atomics, all to the local CTA,
  vs 1M atomics, all routed to one specific remote CTA in the
  cluster.
- For the skewed-input case (one bin gets `>50%` of the inputs), can
  we re-balance the partitioning? Two ideas:
  a) hash the bin index before the divmod so hot bins spread across
     CTAs (breaks the natural ordering of the output table — needs
     un-hashing in the dump phase);
  b) detect the hot bin in a pre-pass and split it across CTAs.
  Worth measuring whether a hash partition is enough.
- Cluster-wide barriers: a cluster `mbarrier` lives in one CTA's
  SMEM but is arrived-at by threads from all CTAs in the cluster.
  Where do you put it, and is there a contention story for the
  arrive transactions? CUDA's cluster `barrier.cluster.arrive`
  primitive vs. an explicit `mbarrier` with cluster-wide arrive
  count — which one does CuTe DSL expose, and is it the right
  abstraction here?
- Variant 3 dumps the partitioned table to GMEM at the end with no
  atomics needed. But what if two clusters were running on disjoint
  input slices and writing to the *same* GMEM `hist`? (That's the
  natural extension to multi-cluster.) The dump becomes an atomic
  add per bin — is that still cheaper than variant 0 because the
  cluster has already coalesced the per-cluster contribution? (Out
  of scope for this exercise; flag it for ex16.)
- DSMEM atomic ordering: are two `red.shared::cluster.add` from the
  same thread to the same address strictly ordered? PTX docs say
  yes for same-warp / same-CTA, but the cluster case is worth
  reading carefully. We don't depend on ordering for histogram
  (add is commutative + associative), but it matters if you ever
  port this pattern to a non-commutative reduction.
- Does the partition stride matter for the cluster network? Bins
  `[c*B/C, (c+1)*B/C)` is a contiguous-per-CTA partition. An
  interleaved partition (`bin b lives on CTA b % C`) would change
  which atomics travel where. Try both and see whether the network
  prefers one.
