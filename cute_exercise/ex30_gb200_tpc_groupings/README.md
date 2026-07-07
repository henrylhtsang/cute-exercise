# What are the TPC groupings on GB200?

Reference:
- <https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor>

## Question

Given a GB200 / SM100 GPU with 152 visible SMs, can we recover the physical
TPC grouping of those SMs from software-visible behavior?

The target output is a map like:

```text
tpc_id  smids
------  -----
0       ...
1       ...
...
```

and, if the evidence supports it, a higher-level grouping by GPC:

```text
gpc_id  tpc_ids        smids
------  -------        -----
0       ...
1       ...
...
```

Do not assume the logical SM id order is the physical TPC order. The point of
the exercise is to prove the grouping, then compare the result with the
Blackwell floorplan / floorsweep discussion in the SemiAnalysis article.

## Why this matters

Several Blackwell features are cluster-local or topology-sensitive:

- `tcgen05.mma cta_group::2` pairs two CTAs / SMs into one logical MMA;
- TMA multicast routes one GMEM tile to multiple CTAs in a cluster;
- DSMEM traffic moves through the cluster fabric;
- cluster sizes above 8 may expose GPC / TPC boundaries;
- floorswept parts can leave non-obvious holes in the visible SM id space.

For kernel work, "cluster shape `(2, 1)`" is not enough. We also need to know
which physical SMs the scheduler places together, whether adjacent `smid`
values are actually peers, and whether performance cliffs line up with TPC or
GPC boundaries.

## Setup

Run on a GB200 system. Start with one GPU visible via `CUDA_VISIBLE_DEVICES`,
and record:

- `torch.cuda.get_device_name()`;
- compute capability;
- `multiProcessorCount`;
- driver and CUDA runtime versions;
- whether NCU hardware counters are available;
- whether cluster launch attributes are enabled for cluster sizes up to 16.

The investigation should be repeatable. Every probe should dump raw rows, not
just the inferred grouping.

## Probes

### V0 -- SM id inventory

Write a tiny kernel that records `%smid` for each resident CTA. Launch enough
CTAs to observe every visible SM, repeat many times, and build the set of SM ids
that the driver exposes.

Questions to answer:

- Are the visible ids dense from `0` to `multiProcessorCount - 1`?
- If not, which ids are missing?
- Is the observed set stable across launches and processes?

### V1 -- cluster placement

Launch clustered kernels with cluster sizes:

```text
2, 4, 8, 16
```

For each CTA, record:

- `blockIdx`;
- cluster rank;
- `%smid`;
- a launch iteration id.

Build tables showing which SM ids appear in the same cluster for each cluster
size. Repeat enough times to separate deterministic placement from scheduler
noise.

### V2 -- peer latency / bandwidth

For candidate SM pairs, run a DSMEM ping-pong or producer/consumer microbench
inside a cluster and measure:

- one-way latency;
- sustained shared-cluster load bandwidth;
- atomic / reduction throughput if useful.

The goal is to see whether inferred TPC peers have measurably different
behavior from non-peers in the same larger cluster.

### V3 -- 2-SM MMA adjacency

Use a minimal `tcgen05.mma cta_group::2` kernel, or adapt
`ex19_2cta_mma`, and record the two `%smid` values that participate in each
pair-MMA. Check whether the hardware / scheduler always forms pairs that match
the inferred TPC grouping.

If the kernel is not implemented yet, a PTX-level or CUTLASS-based probe is
fine. The exercise is about topology evidence, not CuTe DSL purity.

### V4 -- multicast and cluster-size cliffs

Adapt the multicast microbench from `ex18_tma_multicast` and sweep cluster size.
Look for knees in:

- multicast load bandwidth;
- DSMEM bandwidth;
- kernel time;
- NCU cluster / L2 multicast counters.

Compare the cliffs with the inferred TPC and GPC boundaries.

## Expected answer shape

The answer should include both raw evidence and the final grouping. A useful
minimum output is:

```text
smid  cluster2_peer  cluster4_group  cluster8_group  cluster16_group  notes
----  -------------  --------------  --------------  ---------------  -----
0     ...
1     ...
```

and:

```text
tpc_id  smids  evidence
------  -----  --------
0       ...    cluster2 placement, pair-MMA placement, DSMEM latency
1       ...
```

If a GB200 part is floorswept, explicitly distinguish:

- physical TPCs described by the article;
- visible SM ids reported by software;
- disabled or missing SMs/TPCs inferred from holes in the mapping;
- any remapping the scheduler appears to do.

## Things to prove while implementing

- Does `%smid` order correspond to physical TPC adjacency?
- Does cluster size 2 always place CTAs on the two SMs in one TPC?
- For cluster sizes 4, 8, and 16, do the groups align with TPC and GPC
  boundaries?
- Does `cta_group::2` pair the same SMs that cluster placement suggests?
- Do DSMEM / multicast measurements show a topology-dependent cliff?
- Is the grouping stable across repeated launches, process restarts, and
  different GPU instances?
- How does the recovered map compare with the SemiAnalysis Blackwell floorplan
  and floorsweep discussion?

## Answer

Measured on the local GB200 with CUDA 13.3 `nvcc`:

```bash
python cute_exercise/ex30_gb200_tpc_groupings/run_probe.py \
  --cluster-sizes 1,2,4,8,16 \
  --iters 20 \
  --output cute_exercise/ex30_gb200_tpc_groupings/artifacts/gb200_tpc_probe_latest.json
```

Device / toolchain:

```text
GPU: NVIDIA GB200
compute capability: 10.0
visible SMs: 152
CUDA driver version reported by runtime: 13000
CUDA runtime version: 13030
nvcc: release 13.3, V13.3.33
NCU binary: /usr/local/bin/ncu
cluster launch: works for cluster sizes 1, 2, 4, 8, and non-portable 16
```

The probe records one raw row per CTA:

```text
cluster_size, iteration, block_idx, cluster_id, cluster_rank, smid
```

It uses 232448 bytes of dynamic shared memory per CTA so a launch wave cannot
place multiple CTAs on one SM. The Python runner calibrates the largest
one-wave block count for each cluster size and rejects launches with duplicate
SM ids inside an iteration.

Raw evidence is in:

```text
cute_exercise/ex30_gb200_tpc_groupings/artifacts/gb200_tpc_probe_latest.json
```

### V0 -- SM id inventory

All 152 visible SM ids are dense:

```text
observed: 0..151
missing in dense range: none
```

### V1 -- cluster placement

For cluster size 2, all 76 groups appeared in every one of 20 iterations, with
no ambiguity:

```text
tpc_id  smids
------  -----
0       0,1
1       2,3
2       4,5
...
73      146,147
74      148,149
75      150,151
```

Equivalently:

```text
tpc_id n maps to SM ids (2*n, 2*n + 1), for n in [0, 75].
```

This was stable across three separate process invocations:

```text
run 1: 76 pairs, first=(0,1), last=(150,151)
run 2: 76 pairs, first=(0,1), last=(150,151)
run 3: 76 pairs, first=(0,1), last=(150,151)
all_equal: true
```

The cluster-placement compact grouping array for the measured placement is:

```text
[8, 8, 8, 8, 8, 8, 8, 4, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1]
```

This sums to 76 visible TPCs. The exact TPC id groups behind that array are:

```text
[1, 9, 17, 25, 33, 41, 48, 55]
[2, 10, 18, 26, 34, 42, 49, 56]
[3, 11, 19, 27, 35, 43, 50, 57]
[4, 12, 20, 28, 36, 44, 51, 58]
[5, 13, 21, 29, 37, 45, 52, 59]
[6, 14, 22, 30, 38, 46, 53, 60]
[7, 15, 23, 31, 39, 47, 54, 61]
[0, 8, 16, 24]
[32, 40]
[64, 69]
[65, 70]
[66, 71]
[67, 72]
[68, 73]
[62]
[63]
[74]
[75]
```

This is not the same thing as the SemiAnalysis physical-GPC count array. It is
the connected-component hierarchy induced by CUDA cluster co-placement. It is
useful evidence that the scheduler sees strided neighborhoods, but it is not a
floorplan-level physical GPC map.

Cluster sizes above 2 show that logical TPC id order is not the larger
physical-neighborhood order. Expressed in TPC ids:

```text
cluster4:
  (0,8) .. (7,15)
  (16,24) .. (23,31)
  (32,40) .. (39,47)
  (48,55) .. (54,61)
  (64,69) .. (68,73)

cluster8:
  (0,8,16,24) .. (7,15,23,31)
  (33,41,48,55) .. (39,47,54,61)

cluster16:
  for k in 1..7:
    (k, k+8, k+16, k+24, k+32, k+40, k+47, k+54)
```

The calibrated one-wave launches used these block counts. The
`max_cluster_groups` column is the maximum number of cluster groups of that size
that launched in one clean wave without duplicate SM ids:

```text
cluster_size  blocks  max_cluster_groups  unique_smids
------------  ------  ------------------  ------------
1             152     152                 152
2             152     76                  152
4             144     36                  144
8             120     15                  120
16            112     7                   112
```

The larger-cluster data is therefore good evidence for physical adjacency among
the groups shown above, but it does not expose a complete unambiguous GPC map
for every visible SM. In particular, the one-wave larger-cluster placements do
not cover all 152 SMs:

```text
cluster4 missing:  124,125,126,127,148,149,150,151
cluster8 missing:  64,65,80,81,124..151
cluster16 missing: 0,1,16,17,32,33,48,49,64,65,80,81,124..151
```

The software-visible SM id space is dense, so any floorsweep is hidden behind
logical remapping rather than holes in `%smid`. The stable 2-SM TPC map is
complete; the higher-level GPC grouping is only partially recoverable from
cluster placement alone on this system.

### L2 pointer-chase physical-layout probe

To try to reproduce the SemiAnalysis physical-GPC method, I added a separate
L2 pointer-chase probe:

```bash
python cute_exercise/ex30_gb200_tpc_groupings/run_l2_probe.py \
  --concurrent \
  --buckets 4096 \
  --samples-per-bucket 16 \
  --stride-words 1024 \
  --output cute_exercise/ex30_gb200_tpc_groupings/artifacts/gb200_l2_physical_probe_concurrent_256mb.json
```

This uses a 256 MiB pointer-chase working set and emits one latency signature
per `%smid`. The SASS contains real dependent global loads in the timed loop:

```text
LDG.E.STRONG.GPU
```

However, this run did not produce a stable physical-GPC count array. The
hierarchical cuts depend materially on pointer permutation seed. For example:

```text
seed=1, fixed K=8:  [14, 12, 12, 10, 8, 8, 8, 4]
seed=1, fixed K=11: [14, 10, 10, 8, 8, 8, 8, 4, 3, 2, 1]

seed=2, fixed K=8:  [18, 16, 10, 10, 8, 8, 4, 2]
seed=2, fixed K=11: [18, 16, 10, 8, 6, 6, 4, 4, 2, 1, 1]
```

Raw artifacts:

```text
cute_exercise/ex30_gb200_tpc_groupings/artifacts/gb200_l2_physical_probe_concurrent_256mb.json
cute_exercise/ex30_gb200_tpc_groupings/artifacts/gb200_l2_physical_probe_concurrent_256mb_seed2.json
```

The SM-SM latency-difference matrix can be rendered from the latency
signatures:

```bash
python cute_exercise/ex30_gb200_tpc_groupings/plot_l2_matrix.py \
  --order smid \
  --output cute_exercise/ex30_gb200_tpc_groupings/artifacts/gb200_l2_latency_difference_smid.svg

python cute_exercise/ex30_gb200_tpc_groupings/plot_l2_matrix.py \
  --order clustered \
  --cluster-groups 8 \
  --output cute_exercise/ex30_gb200_tpc_groupings/artifacts/gb200_l2_latency_difference_clustered.svg
```

Generated graph artifacts:

```text
cute_exercise/ex30_gb200_tpc_groupings/artifacts/gb200_l2_latency_difference_smid.svg
cute_exercise/ex30_gb200_tpc_groupings/artifacts/gb200_l2_latency_difference_clustered.svg
cute_exercise/ex30_gb200_tpc_groupings/artifacts/gb200_l2_latency_difference_probe_order.svg
cute_exercise/ex30_gb200_tpc_groupings/artifacts/gb200_l2_latency_difference_tpc_stride8.svg
cute_exercise/ex30_gb200_tpc_groupings/artifacts/gb200_l2_latency_difference_nearest.svg
cute_exercise/ex30_gb200_tpc_groupings/artifacts/gb200_l2_latency_difference_nearest_tpc.svg
```

The nearest-neighbor reorders make local neighborhoods much clearer than raw
SM id order:

```text
SMID order average adjacent matrix value:        3.24 cycles
nearest-SM order average adjacent matrix value:  0.89 cycles
nearest-TPC order average adjacent matrix value: 1.25 cycles
```

The nearest-SM order is visually useful but can split the two SMs in a known
TPC. The nearest-TPC order preserves each `(2*n, 2*n+1)` pair and is the better
view for guessing TPC groupings.

I also tried forcing a conservative "no inferred physical group may exceed 10
visible TPCs" cap on the nearest-matrix clustering. That produced this
candidate:

```text
heuristic counts: [10, 10, 10, 10, 8, 8, 8, 8, 4]

[0, 1, 33, 34, 39, 40, 41, 55, 63, 73]
[3, 6, 11, 12, 14, 59, 69, 70, 71, 72]
[4, 5, 19, 21, 22, 35, 36, 38, 64, 67]
[8, 9, 24, 25, 26, 31, 48, 54, 74, 75]
[2, 7, 10, 15, 42, 47, 49, 61]
[13, 43, 44, 45, 46, 57, 58, 60]
[16, 17, 18, 23, 32, 56, 62, 68]
[27, 28, 29, 30, 50, 51, 52, 53]
[20, 37, 65, 66]
```

I reject this candidate. It does not line up with the cluster-size-8 placement
evidence. Cluster size 8 exposes exactly 15 groups of 4 TPCs:

```text
(0,8,16,24) .. (7,15,23,31)
(33,41,48,55) .. (39,47,54,61)
```

Those 15 groups cover 60 of 76 visible TPCs. The missing TPCs are:

```text
32,40,62,63,64,65,66,67,68,69,70,71,72,73,74,75
```

The forced nearest-matrix candidate mixes those missing TPCs into several
groups and is not supported by the cluster8 evidence. The useful conclusion is
negative: the current latency graph and cluster-placement graph do not yet give
a defensible physical-GPC grouping.

So the physical-GPC count is unresolved from the current L2 probe. Do not quote
the cluster-placement array as the physical GPC array. A floorplan-compatible
hypothesis for 76 visible TPCs on an 80-TPC physical GPU would need eight
physical GPC counts summing to 76, but this probe has not provided stable
evidence for which physical GPCs have the disabled TPCs.

### Conclusions

- `%smid` order does correspond to physical TPC adjacency at the 2-SM level:
  TPC `n` is exactly `(2*n, 2*n+1)`.
- Cluster size 2 always placed CTAs on those two-SM TPC pairs in the measured
  runs.
- Cluster sizes 4, 8, and 16 do not group consecutive TPC ids. They group
  strided TPC ids, mostly with stride 8 in the lower logical range.
- The GB200 exposes 152 dense logical SM ids, i.e. 76 visible TPCs. The raw
  larger-cluster placement has missing tails consistent with a floorswept or
  remapped part, but cluster placement alone is not enough to label every
  physical GPC.
- I added an L2 pointer-chase probe for physical-GPC inference, but the current
  distance-matrix clustering is not stable enough to publish a physical count
  array.
- I did not use DSMEM latency, TMA multicast counters, or a separate
  `tcgen05.mma cta_group::2` probe to define the TPC map. Those would still be
  useful independent confirmations.
