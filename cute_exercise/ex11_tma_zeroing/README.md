# How does TMA load and store zeroing work?

## Question

TMA (`cp.async.bulk.tensor`) is the canonical way to move a 2D/3D tile
between GMEM and SMEM. The tricky bit at the edges of a tensor is what
happens when the tile box overhangs the tensor extent: are out-of-bounds
elements defined? Who writes (or doesn't write) them?

Concretely:

- **TMA load, OOB region**: when the SMEM tile is larger than the
  in-bounds slice of the GMEM tensor (e.g. `M = 100`, tile `M = 128`),
  what ends up in SMEM for the OOB rows? The PTX rule is that
  `cp.async.bulk.tensor` with the default OOB-fill mode writes **zero**
  into the OOB elements of the destination SMEM. There is also an
  "OOB NaN" mode for floating-point types. What does CuTe DSL expose
  here — is the OOB-fill mode part of the TMA atom, the tensormap
  descriptor, or the copy call site? Do we ever need to memset SMEM
  ourselves before a TMA load, or is the descriptor enough?
- **TMA store, OOB region**: by contrast, TMA *store* does **not**
  write zeros to the OOB region of GMEM — it simply skips OOB elements
  so the existing GMEM contents are preserved. Verify this empirically
  (initialize the destination GMEM with a sentinel pattern, run a TMA
  store whose tile overhangs, check the sentinel is intact in the OOB
  area). What's the right mental model: "TMA load fills SMEM
  rectangularly using zero for OOB, TMA store fills GMEM with a
  *masked* rectangle that drops OOB"?
- **Zeroing GMEM with TMA**: is there any way to use TMA to zero a
  region of GMEM (e.g. for output initialization), or is the only
  option a regular vectorized store / `cp.async.bulk` with a
  zeroed-SMEM source? `cp.async.bulk` has reduction variants
  (`.add`, `.min`, `.max`) but not a "store-immediate-zero" — confirm
  this and document the workaround.

## Why this matters

This shows up everywhere with non-multiple shapes:

- Flash attention and other tile kernels with `seq_len % BLOCK != 0`.
  The query/key/value tail tile has OOB rows; if TMA load zero-fills
  them, those rows contribute zeros to the QK matmul, which is
  almost-but-not-always what you want (softmax mask still needed for
  the K side).
- Persistent / split-K kernels that initialize a partial accumulator
  and want to TMA-store back without clobbering the OOB pad.
- Workspace buffers that must be zeroed before a kernel writes into
  them — knowing whether TMA store leaves them alone tells you whether
  to zero-init from the host, from a separate kernel, or to lean on
  TMA-load's zero-fill on the read side instead.

## Plan

Two micro-kernels, each with a deliberately overhanging tile:

### Variant A — TMA load OOB fill

1. Allocate a GMEM tensor of shape `(M, N)` with `M = 100`, `N = 128`.
2. Initialize it with a non-zero pattern (e.g. `arange + 1`).
3. TMA-load a `128 × 128` tile starting at row 0 into SMEM.
4. Copy SMEM back to a host-visible buffer; check rows `[100, 128)`.
   Expected: zeros (default OOB fill).
5. Repeat with the OOB-fill mode flipped to NaN if CuTe DSL exposes
   it; check we get NaN there.
6. Bonus: do the same with an integer dtype; what does "OOB NaN" do
   then? (PTX docs say the NaN mode only applies to floating-point;
   integer types fall back to zero — verify.)

### Variant B — TMA store OOB skip

1. Allocate a GMEM destination of shape `(M, N) = (100, 128)`,
   initialize with a sentinel (e.g. `-1`).
2. Build a SMEM tile of `128 × 128` filled with a known value
   (e.g. `7`).
3. TMA-store the SMEM tile to GMEM starting at row 0.
4. Read the GMEM destination back; verify rows `[0, 100)` are `7` and
   rows `[100, 128)` are still `-1`. Confirm TMA store does not
   touch the OOB region.

### Variant C — TMA-as-memset

1. Try to use TMA store to zero a region of GMEM. Allocate a SMEM
   buffer, zero it (regular store + `fence.proxy.async`), TMA-store.
   Measure the achieved BW and compare against a plain
   `cudaMemsetAsync` and a hand-rolled vectorized-`st.global` zeroing
   kernel.
2. Is TMA actually faster for this trivial workload, or does the
   per-CTA descriptor / barrier overhead drown out the bulk-store
   benefit at small sizes?

## Measurements

- Bit-exact OOB region check (`torch.equal(oob_slice, expected)`)
  for both load-zero and store-skip.
- For variant C: GB/s vs `cudaMemsetAsync` and vs a vectorized
  `st.global.v4` kernel, swept over sizes from 1 MiB to 1 GiB.
- SASS / PTX dump confirming the actual instruction is
  `cp.async.bulk.tensor.{2d,Nd}.shared::cluster.global` with the
  expected OOB-fill mode encoded in the descriptor.

## Open questions to settle while implementing

- Where exactly does the OOB-fill mode live — in the tensormap
  descriptor (i.e. you bake it in once on the host), in the
  per-instruction modifier, or both? CuTe DSL `make_tma_atom` /
  `make_tiled_tma_atom`-style APIs — do they let me pick zero vs NaN
  fill, or is it always zero?
- For 3D / 5D TMA descriptors, does the OOB rule apply per-mode
  (any one OOB axis ⇒ zero) or only when the *origin* of the tile is
  OOB? PTX docs say per-element, but it's worth a one-liner empirical
  check.
- Does the OOB-fill happen at the L2 / TMA-engine boundary, or does
  it cost SMEM-side bandwidth like a normal store? In other words,
  is a 50%-OOB tile half as expensive as a fully-in-bounds tile, or
  the same cost?
- For TMA store, can `cp.reduce.async.bulk.tensor` (the reduction
  variant from ex4) be abused to "store zero" by reducing with
  `.add` against a zeroed SMEM source? Probably nonsensical but
  worth ruling out.
- If I want both behaviors in one kernel — TMA load with zero fill
  on inputs, TMA store with skip on outputs — do I need two distinct
  TMA atoms / descriptors, or is it the same atom used in opposite
  directions?
