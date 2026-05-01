# Ship a CuTe DSL kernel as pre-compiled PTX

## Question

I have a CuTe DSL kernel I'm happy with. I want to keep the DSL source as
the source of truth (still readable, still editable in Python), but ship
the kernel as **pre-compiled PTX** so end users don't pay the CuTe DSL JIT
cost on first call.

What's the cleanest workflow?

## Files

- `dump_ptx.py`: compiles the DSL kernel (re-using `ElementwiseAdd` from
  ex1) and writes `artifacts/elementwise_add_<dtype>_<MxN>_<sm>.ptx` plus
  a `.manifest.json` sidecar with the kernel symbol, grid/block, and ABI
  description. Re-run whenever the DSL source or tile geometry changes.
- `ptx_runner.py`: load `.ptx` via `cuModuleLoadData`, cache the
  `CUmodule` + `CUfunction`, launch via `cuLaunchKernel` with 3 device
  pointers as kernel params. No CuTe DSL import on the hot path.
- `test_elementwise_add_ptx.py`: PTX result == DSL result == `torch.add`.
- `bench.py`: subprocess-isolated cold-start timing + in-process
  steady-state.

## How to use

### 1. Generate the PTX (only if you change the DSL source or want a new shape/arch)

The `artifacts/` directory already ships the PTX + manifest for fp16/fp32
at `16384x8192` on `sm_100a`. You only need to re-dump if:

- you edited `ElementwiseAdd` in `cute_exercise/ex1_for_loop/kernel.py`,
- you want a different `(dtype, M, N)`,
- you're targeting a different SM arch (the build runs on whatever GPU is
  visible to PyTorch — set `CUDA_VISIBLE_DEVICES` to pick).

```bash
# Default: fp16 + fp32 at 16384x8192, host-driver PTX ISA ceiling
python -m cute_exercise.ex5_ship_pure_ptx.dump_ptx

# Custom shape / dtype
python -m cute_exercise.ex5_ship_pure_ptx.dump_ptx --dtype fp16 --shape 8192 4096

# Cross-host shipping: pin the PTX ISA version to the deployment driver's
# ceiling instead of the build host's (CUDA 13.0 driver -> 9.0; 12.4 -> 8.4)
python -m cute_exercise.ex5_ship_pure_ptx.dump_ptx --ptx-version 9.0
```

Output (in `artifacts/`):

```
elementwise_add_<dtype>_<MxN>_<arch>.ptx
elementwise_add_<dtype>_<MxN>_<arch>.manifest.json
```

Both files are required at runtime — the manifest carries the kernel
symbol, grid/block, and ABI description.

### 2. Run the kernel (no `cute.compile` on the hot path)

```python
import torch
from cute_exercise.ex5_ship_pure_ptx.ptx_runner import elementwise_add_ptx

a = torch.randn(16384, 8192, device="cuda", dtype=torch.float16)
b = torch.randn_like(a)

# allocates output (must match a shape/dtype/contiguous/cuda)
c = elementwise_add_ptx(a, b)

# or in-place into a pre-allocated buffer
out = torch.empty_like(a)
elementwise_add_ptx(a, b, out=out)

# explicit stream (must be on the same device as the tensors)
stream = torch.cuda.current_stream(a.device)
elementwise_add_ptx(a, b, out=out, stream=stream)
```

The first call for a given `(dtype, shape, device)` reads the matching
artifact and runs `cuModuleLoadData` (~tens of ms). Subsequent calls hit
the cache and only pay `cuLaunchKernel`.

Importing `ptx_runner` does **not** import `cutlass` / CuTe DSL.

### 3. Verify and benchmark

```bash
# Correctness: PTX vs DSL JIT vs torch.add (bit-exact)
pytest -q cute_exercise/ex5_ship_pure_ptx/test_elementwise_add_ptx.py

# Cold-start (subprocess-isolated) + steady-state benchmark
python -m cute_exercise.ex5_ship_pure_ptx.bench
```

## Results (GB200, M=16384, N=8192)

### Cold-start: process-boot → first result

Each cell is a fresh `python -c '<allocate inputs; import + first call>'`
subprocess, sync'd. Median of 3 trials.

| mode                | fp16    | fp32    |
|---------------------|--------:|--------:|
| `torch.add`         | 5.4 ms  | 5.8 ms  |
| DSL (JIT)           | 427 ms  | 447 ms  |
| **pre-compiled PTX**| **14 ms** | **15 ms** |

The DSL pays a full `cute.compile` (MLIR → LLVM → ptxas → cubin) on first
call. The PTX path pays only a file read + `cuModuleLoadData` (driver
JITs PTX → SASS, much smaller). **~30× faster cold start.**

### Steady-state (same kernel, same SASS)

| variant             | fp16               | fp32               |
|---------------------|-------------------:|-------------------:|
| `torch.add`         | 115 us / 6996 GB/s | 228 us / 7074 GB/s |
| DSL (JIT, cached)   | 117 us / 6866 GB/s | 230 us / 7012 GB/s |
| pre-compiled PTX    | 118 us / 6820 GB/s | 230 us / 7012 GB/s |

All within 1% — as expected (same kernel, same SASS, same launch
config).

## Open questions, settled

**Does CuTe DSL emit portable PTX?** PTX is portable across SM versions
*if the kernel doesn't use SM-specific features and you target a base
arch* (e.g. `sm_80`). For an elementwise kernel that's true. But the DSL
defaults to the host's `sm_<major><minor>a` (architecture-suffixed — on
GB200 that's `sm_100a`, which includes Blackwell-specific features). So
the PTX as dumped is **arch-specific**: ship one `.ptx` per (arch,
dtype). The artifact filename encodes both.

**ABI: what does the device kernel's `.entry` expect?** With static
shape (no `mark_layout_dynamic`), each `cute.Tensor` flattens to a single
8-byte device pointer (shape and strides are baked into the cubin as
constants). So the kernel signature is just `(void* a, void* b, void* c)`
— easy to pack from Python with the cuda-python `(values, ctypes)` form
of `cuLaunchKernel`. Verified by inspecting the dumped `.entry`:

```
.visible .entry kernel_..._param_0[8],   // a ptr
                kernel_..._param_1[8],   // b ptr
                kernel_..._param_2[8]    // c ptr
```

If we instead `mark_layout_dynamic()`, each tensor becomes a 64-byte
struct (ptr + dynamic shape + dynamic strides + padding). One PTX would
then serve all `(M, N)` divisible by the tile, but Python has to
hand-pack that struct — fragile. We took the simpler path and let the
artifact filename carry the shape.

**Is there an official CuTe helper for "compile once, ship later"?** Yes,
but it ships **cubin** (not PTX) wrapped in a `.o` file together with a
generated host launch entry:
`JitCompiledFunction.export_to_c(...)` + `cute.runtime.load_module(...)`.
See `cutlass/examples/python/CuTeDSL/cute/export/`. That path is more
ergonomic (handles `mark_layout_dynamic` correctly and exposes the same
Python signature as the DSL) but ties the artifact to one SM. The
pure-PTX path here is the lower-level alternative — useful when you want
the driver to re-translate PTX → SASS at load time, or when you want a
human-readable artifact you can inspect/diff.

**Driver vs PTX version mismatch.** The DSL ships its own `ptxas` (CUDA
13.1 in this env), which emits `.version 9.1` PTX. A host driver at CUDA
13.0 only accepts up to `.version 9.0` and rejects the dumped PTX with
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. `dump_ptx.py` detects the host
driver version and rewrites the `.version` line down to the highest
supported value before saving. Safe iff the kernel doesn't use post-9.0
instructions (true for elementwise; revisit for kernels that use
`tcgen05` / cluster TMA / ...).

## Why this matters

For a one-shot benchmark, full JIT is fine. For a library or a deployment
that imports the kernel as part of a larger startup path — PyTorch op
libraries, MoE expert dispatch, anything where users measure TTFT —
paying 400+ ms per kernel per cold start is unacceptable. The PTX path
turns that into ~15 ms while keeping the DSL source as the source of
truth.
