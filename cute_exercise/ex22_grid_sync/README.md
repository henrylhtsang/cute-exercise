# Can I fuse two kernels into one with a grid-wide barrier in CuTe DSL?

## Question

Sometimes two consecutive kernels are really one algorithm split by a
data dependency the GPU can't express *within* a single kernel without
a **grid-wide barrier**: phase 1 has every CTA write a partial result
to GMEM, then *every* CTA must see *all* of phase 1's output before
phase 2 starts. The canonical CUDA tool for this is a cooperative
launch + `cooperative_groups::this_grid().sync()`: launch exactly as
many CTAs as can be co-resident, run phase 1, `grid.sync()`, run
phase 2 — all in one kernel, no second launch, no GMEM round-trip
through the driver, no relaunch latency.

The questions this exercise answers:

1. **Does CuTe DSL expose a grid barrier at all?** Short answer: it
   exposes the *launch* side (`cooperative=True`) but **not** a
   device-side `grid.sync()` primitive. You have to build the barrier
   yourself.
2. **How do you build it?** Two routes — a hand-rolled GMEM-flag
   barrier (exactly flash-attention's split-K combine semaphore), or a
   proper reusable arrival-counter grid barrier — both via inline PTX
   through `llvm.inline_asm`.
3. **Is the fusion actually worth it** vs. (a) two plain kernel
   launches and (b) two launches overlapped with PDL
   (`griddepcontrol`)?

### What CuTe DSL gives you today

- **Cooperative launch flag.** `LaunchConfig` in
  `cutlass/base_dsl/dsl.py` carries `cooperative: bool = False`, plumbed
  through `cutlass/cutlass_dsl/cutlass.py` to
  `cuda_dialect.launch_cfg_cooperative(...)`. Setting it makes the
  launch use `cudaLaunchCooperativeKernel`, which is the *prerequisite*
  for a grid barrier: it guarantees all requested CTAs are co-resident
  (the launch fails if the grid exceeds the device's co-resident
  capacity). It does **not** give you a barrier — it just makes one
  safe to write.
- **No `grid.sync()` device intrinsic.** Grep
  `cute/arch/nvvm_wrappers.py`: there's `bar.sync` (CTA-scope),
  `fence_acq_rel_gpu`, and the PDL pair `griddepcontrol_wait` /
  `griddepcontrol_launch_dependents` — but nothing grid-scope. A
  grid barrier is not a single hardware instruction anyway; CUDA's
  `this_grid().sync()` is itself a global-atomic arrival counter + spin
  with a generation flip. You reproduce that in DSL with inline PTX.
- **The building blocks already exist in flash-attention.**
  `flash_attn/cute/barrier.py` is a GMEM-flag inter-CTA barrier built
  entirely from `llvm.inline_asm`:
  - `ld_acquire` → `ld.global.acquire.gpu.b32`
  - `red_release` / `red_relaxed` → `red.release.gpu.global.add.s32`
  - `wait_eq(lock_ptr, tid, off, val)` → thread 0 spins on
    `ld_acquire` until the flag hits `val`
  - `arrive_inc(lock_ptr, tid, off, val)` → thread 0 does a
    `red.release` add

  FA uses this as a *producer→consumer* semaphore between the split-KV
  attention kernel and the combine kernel (`flash_fwd_combine.py`,
  `semaphore_to_reset`), not as a symmetric all-CTA barrier — but it's
  the same primitives a grid barrier needs.

### Why this is the right vehicle

A two-pass operation that *genuinely* needs the whole grid finished
before pass 2: **global L2-normalize over a large 1-D tensor**, or
**full-tensor layernorm**. Pass 1: each CTA reduces its slab to a
partial `sum` / `sum-of-squares`, atomic-accumulates into a small GMEM
scratch. Barrier. Pass 2: each CTA re-reads its slab, divides by the
now-global statistic, writes out. It's arithmetically trivial
(correctness is a one-liner against PyTorch) so all the signal is in
the barrier mechanics and the launch-strategy comparison.

This is also exactly the shape of a fused **GEMM → row/col-softmax** or
**GEMM → global-reduction epilogue**, so the lesson transfers.

## Setup

- `x`: `(N,)` `float32` in GMEM, `N` large (start `N = 64 Mi` so the
  kernel is solidly DRAM-bound and launch overhead is *not* the whole
  story — then also try `N = 256 Ki` where launch overhead dominates
  and fusion should win biggest).
- `scratch`: a tiny GMEM buffer — one `float32` accumulator (the global
  sum-of-squares) plus the barrier flag(s).
- `out`: `(N,)` `float32`, `out[i] = x[i] / sqrt(mean(x^2) + eps)`.

Reference: `out = x * torch.rsqrt((x*x).mean() + eps)`. Bit-exact
isn't required (atomic-add order is nondeterministic); use `atol/rtol`
or do the global reduction in a deterministic tree if you want
exactness.

**Persistent grid.** For grid.sync to not deadlock, every CTA must be
co-resident. Launch `grid = num_SMs * cta_per_sm` where `cta_per_sm`
comes from the occupancy of *this* kernel (query it, or pin it with
`min_blocks_per_mp`). Each CTA then strides over the input
(grid-stride loop) so a fixed CTA count covers arbitrary `N`. Confirm
the cooperative launch *rejects* an over-large grid rather than
silently hanging.

## Variants

### V0 — two separate kernel launches (baseline)

Kernel A: partial reduction → atomic into `scratch`. Kernel B: read
`scratch`, normalize. Two `compile`/`launch` pairs, ordered by the
stream. No cooperative launch, no persistence. This is the
"obviously correct, obviously has a launch gap + a full DRAM
round-trip of `x`" reference. Measure the launch gap explicitly.

### V1 — single cooperative kernel, hand-rolled GMEM-flag barrier

One persistent kernel, `cooperative=True`. Phase 1 reduces + atomics
into `scratch`; then a **grid barrier** built from the FA `barrier.py`
primitives:

- Each CTA's thread 0 `red.release`-adds 1 into a GMEM arrival counter.
- Each CTA's thread 0 spins (`ld.global.acquire`) until the counter
  reaches `gridDim` (i.e. all CTAs arrived).
- CTA-internal `bar.sync` before/after so the whole CTA respects the
  grid barrier, not just thread 0.

Then phase 2 normalizes. The subtlety is **reusability / the
generation flip**: a naive "spin until count == gridDim" barrier is
single-shot and races on reset. Implement the standard two-phase
(arrive / depart) or generation-parity barrier so it's safe even if
you later want multiple grid barriers in one kernel. Document the
fence/scope requirements: the phase-1 stores must be visible
grid-wide *before* the arrival is observed → `red.release` +
`ld.acquire` at `.gpu` scope, and a `fence_acq_rel_gpu` if needed.

### V2 — single cooperative kernel, arrival-counter "real" grid barrier

Same as V1 but implement the barrier the way
`cooperative_groups::this_grid().sync()` does internally (one global
counter, all-CTA arrive + spin with generation flip), and compare it
head-to-head with V1's FA-style flag. Are they the same thing? (Mostly
yes.) Does either lower to fewer instructions / less spin traffic?

### V3 — no grid barrier: PDL-overlapped two launches

Keep V0's two kernels but overlap them with **Programmatic Dependent
Launch**: kernel B issues `griddepcontrol_wait()` at entry (waits for
A's grid to finish + memflush), kernel A issues
`griddepcontrol_launch_dependents()` once its writes are done. This is
*not* a barrier — it's a launch-dependency hint that lets B's CTAs
start spinning up while A drains. The comparison question: for this
workload does PDL recover most of what fusion buys (hidden launch
latency) **without** the cooperative-launch occupancy tax? Or does the
full DRAM round-trip of `x` (which fusion avoids but PDL does not)
dominate, making V1/V2 win regardless?

### V4 — fusion that actually reuses `x` in SMEM/regs across the barrier

The real payoff of grid.sync isn't just skipping a launch — it's that
phase 2 can reuse phase-1 state. Keep each CTA's slab of `x` resident
(SMEM or registers) across the barrier so phase 2 doesn't reload it
from DRAM. Now fusion saves a *full read of `x`*, not just a launch.
This is the variant where fusion should decisively beat V0/V3. Watch
the occupancy cost: holding the slab resident cuts `cta_per_sm`, which
shrinks the persistent grid, which can slow phase 1. Find the
break-even.

## Measurements

Hardware: GB200.

For each variant, at `N ∈ {256 Ki, 4 Mi, 64 Mi, 256 Mi}`:

- Wall-clock and effective GB/s (`(read x + write out) / time`; V4
  reads `x` once, V0/V3 read it twice).
- **Launch gap**: NCU/Nsight-systems timeline gap between the two
  grids in V0 vs. the in-kernel barrier cost in V1/V2 vs. the PDL
  overlap in V3. This is the headline number.
- DRAM bytes (`dram__bytes_read.sum`, `dram__bytes_write.sum`):
  confirm V0/V3 read `x` twice and V4 reads it once.
- Barrier cost in isolation: a microbench kernel that does *only* N
  grid barriers back-to-back (no work) → ns per grid barrier, and how
  it scales with grid size. Compare V1 flag-barrier vs V2
  arrival-counter.
- Occupancy / `cta_per_sm` for the cooperative kernels and how the
  cooperative launch caps the grid. Record the max grid the
  cooperative launch accepts before it errors.
- PTX/SASS of the barrier region (`dump-ptx-sass` skill): confirm
  `red.release.gpu`, `ld.global.acquire.gpu`, the spin loop, and for
  V3 the `griddepcontrol.wait` / `.launch_dependents`.

## Plan

1. **V0 first** — get the two-launch version bit-close to PyTorch.
   This pins correctness and gives the launch-gap baseline.
2. **Lift the FA barrier primitives** — copy/adapt `ld_acquire`,
   `red_release`, `wait_eq`, `arrive_inc` from
   `flash_attn/cute/barrier.py`. Write a *standalone* test: a
   cooperative kernel where CTA `c` writes `c` to `scratch[c]` in
   phase 1, grid-barriers, then phase 2 has every CTA sum `scratch[:]`
   and check it equals `sum(range(gridDim))`. If any CTA sees a stale
   slot, the barrier (or its fences) is wrong. Get this green before
   touching the real workload.
3. **V1** — wire the barrier into L2-normalize. Watch for the
   single-shot-barrier reset race; add the generation flip.
4. **V2** — the arrival-counter barrier; A/B against V1.
5. **V3** — PDL-overlapped two-launch. Pure launch-side change, no
   barrier.
6. **V4** — keep the slab resident across the barrier; find the
   occupancy break-even.
7. **Write up**: when does fusing with grid.sync beat just launching
   twice (or PDL-overlapping)? Hypothesis: fusion wins when (a) launch
   overhead is a large fraction of runtime (small `N`) **or** (b) phase
   2 can reuse phase-1 state to avoid a DRAM re-read (V4). When the op
   is big and phase 2 must re-read from DRAM anyway, PDL (V3) likely
   recovers most of the benefit without the cooperative occupancy tax.

## Open questions to settle while implementing

- **Does the cooperative launch actually go through
  `cudaLaunchCooperativeKernel`?** Confirm by over-sizing the grid and
  checking the launch errors (cooperative grids are capped at
  co-resident capacity) rather than hanging. Where does the DSL
  surface that error?
- **What scope/fence is the minimum correct barrier?** Is
  `red.release.gpu` + `ld.acquire.gpu` enough to make phase-1 GMEM
  stores visible grid-wide, or do you need an explicit
  `fence_acq_rel_gpu` (or even `.sys` scope) before arrival? Try the
  weakest and look for stale reads under `compute-sanitizer
  --tool racecheck`.
- **CTA-internal sync around the grid barrier.** Only thread 0
  arrives/spins; the rest of the CTA must `bar.sync` before trusting
  the barrier. Get the ordering of `bar.sync` vs. the GMEM
  arrive/wait right — a missing one is a classic "works at small grid,
  races at large grid" bug.
- **Single-shot vs. reusable.** The naive count-to-gridDim barrier
  can't be reused (reset races the next arrive). Implement the
  two-phase / generation-parity version. Does CuTe DSL's
  `cooperative_groups` story have anything reusable, or is hand-rolling
  the only option?
- **Is there a hidden built-in?** Double-check `cute.arch` and
  `cutlass.pipeline` for any grid-scope barrier atom before committing
  to inline PTX. The pipeline classes are cluster/CTA-scope — verify
  nothing is grid-scope.
- **Cluster + cooperative.** Can a cooperative launch also use thread
  block clusters? If so, does the grid barrier compose with cluster
  barriers, or do they fight over occupancy?
- **PDL vs. grid.sync semantics.** `griddepcontrol.wait` waits for the
  *entire previous grid* to finish and memflush — it's a launch
  dependency, not a mid-kernel barrier. Enumerate exactly which
  problems need a true grid barrier (mid-kernel, state reuse) vs. which
  are satisfied by PDL (just hide launch latency). The L2-normalize
  here is the borderline case — measure which camp it falls in.
- **Persistent grid sizing.** What's the right `cta_per_sm`? Too few
  CTAs underfills phase 1; too many and the cooperative launch caps
  you anyway. Sweep and find where effective bandwidth peaks.
- **Does fusion ever *lose*?** If phase 2 must re-read `x` from DRAM
  (V1/V2 without the V4 resident trick), fusion saves only the launch
  gap but pays the cooperative occupancy tax. Is there an `N` where
  V0/V3 actually beat V1? Document it.
