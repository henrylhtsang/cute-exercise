# Skip the `fence.proxy.async` between generic SMEM writes and TMA — what breaks?

## Question

TMA reads/writes SMEM through the **async proxy**, while normal thread
loads/stores hit SMEM through the **generic proxy**. The two proxies
don't observe each other's writes by default — you need a
`fence.proxy.async` (CuTe DSL: `cute.arch.fence_proxy_async` /
`cpasync.fence_async_proxy`-style helpers) to publish generic-proxy
writes to the async proxy before issuing a TMA store, and the mbarrier
wait that ends a TMA load acts as the fence in the other direction.

If we *omit* that fence around a TMA store, does the kernel actually
break? When? On every run, only at certain shapes, only under load?

The point is to build intuition for when the fence is load-bearing vs.
when you're "lucky" — and to have a reproducer we can point at later.

## Plan

Use ex3's pipeline as the victim: TMA load A,B → generic-proxy add in
SMEM → TMA store C.

The bug-injection point is between the generic-proxy add (threads
write the sum into SMEM) and the TMA store (async proxy reads SMEM
into GMEM). That's where `fence.proxy.async` belongs.

Variants to run:

1. **Correct**: fence in place. Should match reference.
2. **No fence, with `__syncthreads()`**: a thread-level barrier does
   *not* cross proxies. Expect this to still be broken, just
   deterministically broken — useful to confirm the fence is the
   thing that matters, not just the sync.
3. **No fence, no syncthreads**: maximally broken.
4. **Fence on store side only / load side only**: isolate which
   direction matters for this kernel.
5. **Threads only *read* SMEM before TMA reads it, no fence**: e.g.,
   threads consume the loaded tile (compute something, write to a
   different SMEM buffer or to registers), and only then does the
   TMA store kick off on that same source SMEM. Two reads shouldn't
   race in theory — verify empirically that this case is in fact
   safe to skip the fence on, so we know the rule is "generic
   *write* → async read" and not just "any thread access → TMA".

What to measure / look for:

- Bit-exact mismatch vs reference, counted per element.
- Is the mismatch deterministic across runs? Across launches in the
  same process? Across processes?
- Does it scale with tile size, number of stages, number of CTAs?
  (Bigger pipelines tend to surface the bug more often because the
  async proxy has more in-flight traffic.)
- Run under `compute-sanitizer --tool racecheck` — it should flag the
  proxy ordering violation directly.
- Skim the SASS diff between "with fence" and "without fence" to
  confirm the only delta is the fence instruction (no other codegen
  shifted under us).

Stretch:

- Try the same omission with `cp.async` (generic-proxy bulk load) +
  TMA store. Different proxy pairing — does the same fence apply or
  is a different one needed?
- Document the failure mode in a short note next to the kernel so
  future-us doesn't have to re-derive it.
