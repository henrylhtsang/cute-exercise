# Does TMA multicast actually save bandwidth on a memory-bound op?

## Question

`cp.async.bulk.tensor.<N>d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster`
(a.k.a. `TMA_LOAD_MULTICAST` in CUTLASS) is the SM90+ TMA variant
that delivers **the same** GMEM tile to the SMEM of multiple CTAs in
the same thread block cluster, in **one** instruction issued by **one**
CTA. The other receiver CTAs are passive — they only set up their
mbarriers with `expect_tx` for their share of the bytes. The hardware
fans the data out to every receiver's SMEM and credits each receiver's
mbarrier.

The pitch is straightforward: if `K` CTAs in a cluster all need the
same tile, multicast turns `K` HBM reads into `1`. In a GEMM that's
the textbook case (A multicast along the cluster N axis, B multicast
along the M axis). For *memory-bound* ops the win can be more
dramatic, because there's no compute to hide behind — saved bytes
turn directly into faster wall-clock.

This exercise builds a deliberately memory-bound kernel where the
same input tile is consumed by `K` consumers, and asks whether
multicast actually delivers the bandwidth amplification it promises
on a real GB200, and what the cliffs are.

## The setup

A "many-to-many broadcast" elementwise op:

- Input `X`: `(M, N)` `bf16` tensor in GMEM.
- Coefficient bank `W`: `(K, N)` `bf16` matrix in GMEM, `K` small
  (e.g. `K = 8` or `K = 16`).
- Output `Y`: `(K, M, N)` `bf16` tensor in GMEM, where
  `Y[k, m, n] = X[m, n] * W[k, n] + bias`.

This is memory-bound by construction:
- One FMA per element, so arithmetic intensity is ~1 op/byte
  (well under any SM's roofline).
- Each output `Y[k]` reads the *same* `X` tile and the *same* `W[k]`
  row.
- `K` is the GQA-ish reuse factor: `K = 8` ≈ Llama-style group size.

The cluster shape is `(K, 1)`. Each CTA in the cluster owns one
`k` slice of the output. Within a cluster they all consume the same
`X` tile, but each consumes a different `W[k]` row.

## Variants

### V0 — no cluster, no multicast (baseline)

Plain TMA load of `X` per CTA, plain TMA load of `W[k]` per CTA, no
cluster. Each of the `K` CTAs reads `X` from HBM independently. L2
*may* absorb most of the redundant `X` traffic if the working set is
small; we want to measure how well it actually does on a tile size
where the working set spills out of L2.

### V1 — cluster, but no multicast

Launch the same workload as `(K, 1)` clusters of `K` CTAs each, but
each CTA still issues its own `cp.async.bulk.tensor` for `X`. No
multicast mask. This isolates "cluster overhead" (cluster-mode
launch, mbarrier setup) from "multicast benefit".

### V2 — cluster + multicast on `X`

The CTA at cluster rank 0 issues a `TMA_LOAD_MULTICAST` for `X`
with `multicast_mask = (1 << K) - 1` (all `K` CTAs in the cluster).
Each receiver CTA sets up its mbarrier with `expect_tx` for the
full `X` tile size. `W[k]` is still per-CTA (no sharing). This is
the headline configuration.

### V3 — multicast on `X`, leader rotation

Same as V2 but rotate which CTA in the cluster is the issuer across
stages (CTA 0 issues stage 0, CTA 1 issues stage 1, …). In theory
this spreads the issue cost across SMs; in practice the issue cost
of multicast TMA is tiny compared to the load itself, so we expect
this to be a wash. Worth measuring once.

### V4 — multicast on `X`, but `K` only partial

Multicast `X` to a *subset* of the cluster (e.g. `K = 8` cluster but
multicast mask covers only 4 CTAs at a time, then issue a second
multicast for the other 4). This costs an extra `cp.async.bulk.tensor`
issue but each issue delivers fewer bytes. We want to know whether
the per-issue overhead ever matters, and whether splitting helps in
any regime (e.g. when `K` is bigger than what the cluster network
can broadcast efficiently in one shot).

### V5 — cooperative non-multicast: "round-robin loads + DSMEM share"

For comparison, do *not* use multicast. Instead, each CTA in the
cluster loads a **slice** of `X` (CTA 0 loads the first `1/K` of
`X`, CTA 1 loads the next `1/K`, …), and each CTA then `ld.shared::cluster`-reads
the slices it doesn't own from peers' SMEM. Same total HBM traffic
as V2, more cluster-internal traffic. The point is to see whether
multicast is a hardware optimization over what we could do "by hand"
with DSMEM, or whether it's the only sane way to get the bandwidth
back.

## Measurements

Hardware: GB200 (cluster size 16 supported with the opt-in
attribute). All variants use the same `(M, N) = (1024, 8192)` tile
size unless noted; pick a tile that's bigger than L2 / `K` so V0's
"L2 absorbs the reuse" path can't quietly win.

For each variant:

- HBM read bytes: NCU `dram__bytes_read.sum`. The expected ranking:
  V2 ≈ V5 ≈ V0_with_warm_L2 < V1 < V0_cold. Confirm.
- L2 read bytes: `lts__t_bytes.sum.read`. V0 is the L2-thrash case;
  V2 should drop both DRAM and L2 read traffic on `X`.
- Multicast traffic counters: NCU
  `lts__t_sectors_aperture_device_op_multicast.sum` (or whatever the
  current counter name is — verify in `ncu --query-metrics`). This
  tells you the multicast actually fired and not a fallback.
- Time / throughput (GB/s of `Y`-write equivalent). Memory-bound, so
  the right ceiling to compare against is HBM peak times the
  reuse factor `K + 1` (read `X` once, read `W` `K` times, write `Y`
  `K` times).
- Cluster scaling: sweep `K ∈ {2, 4, 8, 16}`. Hypothesis: V2
  improves close to linearly in `K` until it hits HBM peak for `W +
  Y` traffic, then plateaus.
- Tile size sweep: `(M, N) ∈ {(512, 4096), (1024, 8192), (2048,
  16384)}`. Tiny tiles probably make multicast indistinguishable
  from V0+L2. Find the crossover.
- Cold vs warm L2: launch with `cudaCtxResetPersistingL2Cache` (or
  equivalent flush) before each measurement to kill L2 carryover.
  Then re-measure with L2 warm. The gap V0_cold − V0_warm is "what
  L2 was already doing for you"; the gap V0_warm − V2 is "what
  multicast adds on top".

## Plan

### Step 0 — V0 baseline

Vanilla single-CTA-per-output TMA-load kernel. One TMA atom for `X`,
one TMA atom for `W[k]`. Confirm correctness against PyTorch
`torch.einsum('mn,kn->kmn', X, W) + bias`, get the V0 number cold
and warm.

### Step 1 — V1, cluster without multicast

Add `__cluster_dims__ = (K, 1)`. Don't change the loads. Verify the
cluster launches (PTX should now show `cluster_dim`), measure
whether cluster-mode alone moves anything. It shouldn't — but we
want the number documented.

### Step 2 — V2, the headline kernel

Replace the `X` load with a multicast atom. CuTe DSL exposes this
via `cute.make_tma_atom_load_multicast` (or similar — verify what's
in the current cute_dsl version). The mask is `(1 << K) - 1` for
"all peers". Each CTA's mbarrier needs `expect_tx(X_tile_bytes)`.
Only the elected leader CTA executes the multicast issue; the rest
just wait on the mbarrier.

Confirm in SASS / PTX that:
- The kernel emits exactly one `cp.async.bulk.tensor.<N>d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster` per `X` load.
- The instruction is gated behind a "this is the leader CTA" predicate.
- The mbarrier expect-tx is per-CTA.

### Step 3 — V5, manual DSMEM split

Same data flow, no multicast. Each CTA loads `1/K` of `X` and the
cluster wires up DSMEM reads for the rest. Use this to check
whether the multicast hardware path beats the "obvious" cluster-aware
implementation in raw GB/s, latency, and instruction count.

### Step 4 — sweep & write up

Run the variants across the sweeps in the Measurements section.
Build a single results table. For each cell record: GB/s, HBM read,
L2 read, kernel time, and a comment on what changed.

## Open questions to settle while implementing

- Which CTA "should" be the multicast issuer? CUTLASS picks the CTA
  whose rank lines up with the source data's layout. On a `(K, 1)`
  cluster broadcasting `X` (which is identical for all CTAs), any
  rank works — does it matter for cluster-network routing? Try
  rank-0 vs rank-`K-1` and look for asymmetry.
- Does the mbarrier `expect_tx` count have to match exactly? PTX
  says yes (the transaction-bytes are credited to the receiver's
  mbarrier; under-reporting hangs, over-reporting can delay arrive).
  Confirm by deliberately mis-sizing it and watching the kernel
  hang.
- What happens if a CTA is in the multicast mask but never
  `mbarrier::wait`s on the corresponding mbarrier? Likely "data
  lands but nobody reads it" — confirm there's no penalty beyond
  the wasted SMEM, and document.
- Multicast and persistent kernels: if the same cluster issues many
  multicast loads back-to-back, does the cluster network pipeline
  them, or do they serialize on the issuer CTA? Microbenchmark with
  N back-to-back multicasts and look at the per-issue latency.
- Multicast + swizzle: the descriptor's swizzle mode applies to the
  receiver SMEM layout. Are there interactions with the cluster
  network we should worry about (e.g. is the TMA multicast unit OK
  with each receiver having a different SMEM swizzle?). Probably not
  — but it's worth confirming with two receivers using *different*
  SMEM layouts in their `make_tma_atom_load_multicast`.
- `cluster_size > 8` requires the non-portable opt-in attribute.
  Does the multicast network's effective bandwidth scale linearly
  past 8, or does it cap at the GPC's TPC count? Sweep `K ∈ {2, 4,
  8, 16}` and look for a knee.
- For GQA-decode-style workloads (Q per-CTA, KV multicast across
  the cluster), the HBM win on KV is roughly `K`. But the *useful*
  arithmetic per byte goes up by `K` too — at what `K` does the
  kernel become compute-bound and stop benefiting? Estimate from
  the roofline and confirm with a real flash-attention decode
  kernel as a follow-up.
- Multicast on the **store** side: PTX has
  `cp.reduce.async.bulk.tensor.*` but no `cp.async.bulk.tensor.store.multicast`
  AFAIK — i.e. you can multicast a load but you can't multicast a
  store. Confirm in the current PTX ISA reference. If true, document
  this asymmetry — it constrains the kinds of fused ops where
  multicast is useful.
- L2 caching policy interactions: the multicast load also takes a
  `cache_hint`. Does setting `EVICT_FIRST` on the multicast load
  hurt subsequent kernels that wanted `X` warm in L2, or does the
  multicast path bypass L2 entirely?
