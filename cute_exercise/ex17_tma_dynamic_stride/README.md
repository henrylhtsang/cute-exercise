# How does TMA work when one stride needs to be dynamic?

## Question

A TMA descriptor (a.k.a. "tensormap") is a 128 B blob that the TMA
engine consumes to issue a `cp.async.bulk.tensor` against a tile of
GMEM. The descriptor encodes, among other things:

- The base GMEM pointer.
- The tensor shape (per dim).
- The tensor **stride** (per dim, in bytes).
- The box (tile) shape.
- Swizzle / fill mode / element type.

Most CuTe DSL examples build the descriptor from a `cute.Tensor`
whose strides are entirely *static* — e.g. `(N, 1)` for a row-major
`(M, N)` matrix where `N` is a compile-time constant. That's the
happy path: `make_tma_atom_*` bakes the stride into the descriptor on
the host once and the kernel never has to think about it.

The unhappy path is when one stride is **only known at launch
time**. Examples:

- **Variable-K matmul**: the K stride of an `(M, N, K)` activation
  tile is `K`, but `K` may differ across calls.
- **Batched / strided batched matmul**: the batch stride for `A`
  is `M * K`, where `M` and/or `K` is a runtime parameter.
- **Padded language-model batches**: the row stride of a packed
  `(B, max_seqlen, H)` tensor changes when the upstream pads to a
  rounded length you don't control.
- **Sliced tensor arguments**: any time PyTorch hands you a view
  with a runtime stride (e.g. a chunk of a larger tensor), you don't
  get to pick.

The exercise is to figure out what CuTe DSL actually does in that
case, what your options are if the default doesn't work, and what
each option costs.

## Context: PTX vs CuTe DSL

At the PTX level, `cp.async.bulk.tensor.<N>d.shared::cluster.global`
takes a tensormap pointer; the strides inside the tensormap are
plain 64-bit integers that *can* be runtime values. There's no PTX
requirement for static strides. The descriptor is just a struct in
memory.

At the CuTe DSL level, things are more constrained:

- `make_tma_atom*` typically wants a `cute.Tensor` (or layout) whose
  strides are statically known. The tensor's strides feed into the
  generated descriptor on the host *and* into the kernel-side
  layout the atom uses to translate logical coordinates into byte
  offsets. If a stride is dynamic, the host-side descriptor build
  is fine — but does the DSL still want a constexpr stride for the
  kernel-side coord math?
- Even if the DSL accepts a runtime stride, the descriptor needs to
  be built somewhere. The two physical options are:
  1. **Host-side build**: PyTorch hands you a stride, host builds
     the descriptor, kernel gets the descriptor pointer. One CPU
     descriptor per launch.
  2. **Device-side build / patch**: kernel writes the dynamic
     stride into a pre-built descriptor template at kernel start
     using `tensormap.replace.tile`/`tensormap.cp_fenceproxy.*`. One
     kernel-resident descriptor.

  Option 1 is simpler; option 2 is what you need for "the stride
  changes per CTA / per persistent kernel iteration".

We want to find which of these CuTe DSL exposes today, what the
defaults are, and what it costs.

## Plan

### Step 0 — establish the static baseline

Pick a simple TMA-load kernel (the one from ex3 or ex11 will do).
All strides static, single tile, copy GMEM → SMEM, store SMEM →
GMEM. Get bit-exact correctness and a baseline GB/s number.

### Step 1 — make one stride dynamic, naive approach

Take the same kernel but pass the major stride of the source as a
runtime argument (`stride: Int32`, not a constexpr). Rebuild the
`cute.Tensor` with that runtime stride and pass it through
`make_tma_atom_*` as before. Two outcomes are possible:

- The DSL accepts it — it builds the descriptor on the host using
  the runtime stride at launch time. Confirm this by:
  - Dumping the descriptor bytes from the host before launch and
    spotting the stride in the expected slot.
  - Sweeping the runtime stride and verifying correctness.
  - Comparing GB/s vs the static-stride baseline. Should be
    identical at steady state since the descriptor is the same
    size and the kernel-side code is unchanged.
- The DSL rejects it — `make_tma_atom_*` errors out at compile
  time, or the layout type system refuses the runtime stride. In
  that case, document the exact error and move to step 2.

### Step 2 — what changes if the stride changes *between* launches?

A descriptor built at launch time means each launch with a different
stride pays one host-side descriptor build. Quantify it:

- Time the host-side `cute.make_tma_atom_*` call (or whatever
  function CuTe DSL exposes for descriptor build) in a tight loop.
  How many microseconds per build? Is it big enough to matter for
  small kernels?
- Compare with: build once, cache, hand to the kernel. Implement a
  tiny LRU keyed on the stride and measure the savings on a
  workload that calls the kernel 10K times with 5 distinct strides.

### Step 3 — descriptor patch on device

For workloads where the stride changes *within* a single launch
(e.g. a persistent kernel walking a list of variable-stride tiles),
host rebuild isn't an option. PTX has
`tensormap.replace.tile.global_stride` and
`tensormap.cp_fenceproxy.*` for exactly this case: copy the
template descriptor into SMEM (or a fresh GMEM slot), patch the
stride with `tensormap.replace`, fence, then point the next
`cp.async.bulk.tensor` at the patched copy.

Find what CuTe DSL exposes:
- Is there a `cute.tensormap_replace_*` / `cute.update_tma_*`
  primitive? If yes, use it.
- If not, drop to PTX via the inline-PTX path used in
  `flash-attention/flash_attn/cute/utils.py` for `shl_u32` (cf.
  ex16). The instruction list is `tensormap.replace.tile.{...}`
  for the field of interest, then `tensormap.cp_fenceproxy.global.shared`
  to make the patched descriptor visible to the TMA engine.

Build a kernel that, per CTA, picks one of K different runtime
strides and patches the descriptor before issuing the TMA. Compare
against:
- A naive kernel that does K static-stride launches (one per
  stride bucket).
- A kernel that does generic loads (no TMA) for the variable-stride
  path.

### Step 4 — descriptor cache and patch sequencing

Document the rules:
- Where does the patched descriptor live — SMEM, GMEM, or a fresh
  L2-resident slot? PTX requires it to be in shared or global; the
  TMA proxy needs `cp_fenceproxy.global.shared` (SMEM source) or a
  fence chain on GMEM.
- How many CTAs can patch the same descriptor concurrently? (Spoiler:
  one per descriptor instance — you need a per-CTA descriptor copy
  if the strides differ across CTAs.)
- Is there an alignment requirement (yes — 128 B, naturally aligned)
  and how does CuTe DSL allocate it? Check whether the DSL gives you
  an "aligned tensormap" SMEM allocator.
- What's the latency of `tensormap.replace.tile` + the proxy fence?
  Microbenchmark a kernel that does N descriptor patches and N TMA
  loads vs. N TMA loads with no patch.

## Measurements

For each step where it's meaningful:

- Throughput: GB/s of the TMA load (or store), compared against the
  static-stride baseline.
- Per-launch overhead (step 2): host-side time spent in the
  descriptor build path, measured with `chrono::steady_clock` over
  10K iterations.
- Patch overhead (step 3): kernel time delta with vs. without the
  patch, isolated by running the kernel with the patch as a no-op
  (replace stride with the value it already has).
- SASS / PTX dump confirming whether the kernel issues
  `tensormap.replace.tile` and `cp_fenceproxy` (step 3) and which
  descriptor pointer the `cp.async.bulk.tensor` consumes.

## Open questions to settle while implementing

- Does CuTe DSL today distinguish "stride is dynamic but constant
  across the launch" from "stride is constexpr"? The first should
  be free at steady state; the second is a constexpr win only if
  it lets the compiler fold address math elsewhere in the kernel
  (e.g. into the SMEM layout). How big is the actual perf delta?
  (Hypothesis: zero for the TMA itself, possibly nonzero for
  kernel-side index math that *also* uses the same stride.)
- Which dimension's stride are you allowed to make dynamic in a
  single descriptor? PTX allows any of them; does CuTe DSL?
  Specifically: can the **innermost** stride be dynamic? (Usually 1
  for contiguous tensors, but not always — e.g. an interleaved
  layout.)
- For batched matmul, the typical pattern is a single TMA descriptor
  with the batch stride either (a) baked into the descriptor and
  re-computed per-batch CTA, or (b) handled by issuing a fresh TMA
  with a base-pointer offset. Which is faster, and does CuTe DSL
  prefer one over the other?
- Is there a way to query, at compile time, whether a given stride
  is constexpr? Useful for writing a kernel that has a fast path
  for the all-static case and a `tensormap.replace` path for the
  dynamic case.
- `tensormap.replace.tile` updates are sequenced by
  `cp_fenceproxy.global.shared`. Is there a cluster-scope variant,
  or do you need per-CTA descriptor copies even within a cluster?
- For descriptors that live in GMEM (host-built and uploaded), can
  you legally patch them with a kernel-side `tensormap.replace`?
  PTX docs imply yes if the descriptor is in `.global` and you
  fence properly. Verify.
- How does cudaGraph capture interact with this? A captured kernel
  with a runtime-stride descriptor: does the captured graph hold a
  pointer to the descriptor (fine, descriptor must stay alive) or
  copy it in (fine, but stale if you intended to mutate)? Likely
  needs a per-graph descriptor allocation — check the docs.
- Practical recipe: when should I just *not* use TMA and fall back
  to vectorized `cp.async.ca.shared.global`? E.g. when the dynamic
  stride means the descriptor patch overhead exceeds the TMA win on
  small tiles. Find the crossover point empirically on GB200.
