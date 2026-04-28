# Ship a CuTe DSL kernel as pre-compiled PTX

## Question

I have a CuTe DSL kernel I'm happy with. I want to keep the DSL source
as the source of truth (still readable, still editable in Python), but
ship the kernel as **pre-compiled PTX** so end users don't pay the
CuTe DSL JIT cost on first call.

What's the cleanest workflow?

1. **Extract**: dump the PTX that CuTe DSL produces for this kernel
   (already done in ex1 via `CUTE_DSL_DUMP_DIR` — reuse that).
2. **Embed**: store the PTX as a string / `.ptx` file alongside the
   Python module that owns the kernel.
3. **Load + launch from Python**: at import / first-call time, load
   the embedded PTX with the CUDA driver API
   (`cuModuleLoadData` → `cuModuleGetFunction` → `cuLaunchKernel`),
   cache the `CUmodule`, and expose a Python entry point with the
   same signature as the DSL kernel.
4. **Keep both paths**: the DSL source stays in the repo and stays
   runnable (for editing, debugging, re-dumping PTX). A flag /
   env var picks DSL-JIT vs pre-compiled PTX at runtime.

## Plan

- Pick a small kernel from an earlier exercise (ex3's TMA vec add is a
  good candidate) as the guinea pig.
- Add a `dump_ptx.py` helper that compiles the DSL kernel once and
  writes `<kernel>.ptx` next to it. Re-run whenever the DSL source
  changes.
- Add a `ptx_runner.py` that:
  - reads the embedded `.ptx`,
  - `cuModuleLoadData` once (cache module + function on a module-level
    dict keyed by device + kernel name),
  - packs args and calls `cuLaunchKernel` with the same grid/block as
    the DSL launch.
- Wire both paths into `benchmark.py` behind a `--mode {dsl,ptx}`
  flag.

What to verify:

- Numeric output of the PTX path matches the DSL path bit-for-bit
  (same kernel, same PTX → should be identical).
- First-call latency: DSL JIT vs `cuModuleLoadData`. Expect the PTX
  path to be much faster on cold start.
- Steady-state throughput: identical (same SASS).
- Re-dumping PTX is a one-line workflow when the DSL source changes,
  so the DSL stays the editable source of truth.

Open questions to settle while implementing:

- Does CuTe DSL emit PTX that's portable across SM versions, or do we
  need one `.ptx` per arch (SM90, SM100, …)? If per-arch, fold the
  arch into the filename and pick at load time.
- Arg-packing: does the DSL kernel's PTX expect the same ABI as a
  hand-written kernel, or does CuTe wrap args in a struct? Inspect
  the dumped PTX's `.entry` signature to confirm.
- Is there a CuTe DSL helper that already does "compile to PTX once,
  reuse later" without us reaching for the driver API directly? If so
  prefer that — fall back to driver API if not.
