# Upstreamability to NVIDIA cutlass

**Verdict: upstreamable as a new example, not as a core library feature.**

The pattern would live at `examples/python/CuTeDSL/cute/export/ship_ptx.py`,
alongside the existing cubin-shipping example
[`load_in_python.py`](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/export/load_in_python.py).
Upstream today loads kernels exclusively via cubin: `jit_executor.py`'s
`load_kernels_from_ir_module` calls `cuda_helpers.load_library_data(cubin_data)`
on a symbol named `<prefix>_cubin`, and the public `cute.runtime.load_module()`
expects an `ExternalBinaryModule` wrapped around an MLIR execution engine.
PTX shipping isn't *blocked* — `cuLibraryLoadData` and `cuModuleLoadData`
accept PTX bytes and driver-JIT to SASS — it's just absent. To upstream
ex5, three things would need to change: (1) the JSON manifest would either
need to align with the existing `kernel_info` schema produced by
`encode_metadata_into_ir_module` in `export/export.py:127`, or be justified
as a deliberately thinner sidecar for the static-shape PTX path; (2) the
`.version`-rewriting trick in `dump_ptx.py:60` (needed because
DSL-bundled `ptxas` can emit newer ISA than the deployment driver supports)
has no upstream analogue and would need to ship as a small utility or be
documented as a pre-ship step; (3) since `ExternalBinaryModule` assumes an
MLIR host-launch wrapper that doesn't exist for raw PTX, upstream would
need either a new thin `load_ptx_module()` helper in `cute/runtime.py`
that does `cuModuleLoadData` + manifest lookup without the MLIR engine,
or the example would call the driver directly (as ex5 does) and document
that explicitly. No hard blockers, but the PTX-only loader abstraction
and version-rewriting helper are net-new additions, not just glue around
existing APIs.
