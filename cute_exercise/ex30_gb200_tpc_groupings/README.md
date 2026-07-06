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

_TODO: fill in after implementing the topology probes._
