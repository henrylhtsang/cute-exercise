# VectorAdd

## Question

Try to get a really optimized float16 vector-add kernel for square tensors:

```text
input:  (A, B), A/B shape = (N, N), dtype = torch.float16
output: C,      C shape   = (N, N), dtype = torch.float16
N in {1024, 2048, 4096, 8192, 16384}
```

The tensors are normal-distributed and used once.  The target machine is B200 /
GB200 class hardware.  The benchmark host used here is an NVIDIA GB200 with 152
SMs, read at runtime with `torch.cuda.get_device_properties`.  This is a pure
memory-bandwidth exercise: coalesced global traffic, CTA scheduling, register
pressure, and half2 add throughput.

## Kernel

`kernel.py` implements a CuTe DSL kernel with compile-time configs for:

- threads per CTA: `128`, `256`, `512`
- CTAs per SM: `1`, `2`, `4`, `8`
- elements per thread: `8` or `16`
- vector load width: scalar `u32`, 128-bit `v4.u32`, or 256-bit `v8.u32`
- load cache policy: default, `.cs`, `L1::no_allocate`, `.ca`, or `.cg`
- store cache policy: default, `.cs`, `L1::no_allocate`, `.wt`, `.wb`, or `.cg`
- math instruction: `add.rn.f16x2`, `fma.rn.f16x2`, or an interleaved mix
- manual unroll factor: `1`, `2`, or `4`
- scalar operation ordering: interleaved load/math, all loads before math, or
  one bundled inline-PTX block
- DLPack alignment assumption: `16`, `32`, `64`, or `128` bytes
- scheduling: one tile per CTA, fixed tiles per CTA, or persistent CTAs
- compile-time launch constants: `num_ctas`, `tile_count`, and the first-tile
  assumption are constructor-specialized into `VectorAdd`

The PTX variants intentionally use inline PTX via `llvm.inline_asm` for the
instructions that matter:

```ptx
ld.global[.cs|.L1::no_allocate].v4.u32
ld.global[.cs|.L1::no_allocate].v8.u32
add.rn.f16x2
fma.rn.f16x2
st.global[.cs|.L1::no_allocate].v4.u32
st.global[.cs|.L1::no_allocate].v8.u32
```

The scalar `u32` path has three ordering experiments.  The original
`interleaved` order does `ld A`, `ld B`, half2 math, then repeats four times
before storing.  `loads_first` issues all four A/B load pairs first, then all
four half2 ops, then all four stores; this gives ptxas more independent memory
ops before the math dependency chain.  `bundled_scalar` puts the four loads,
four half2 ops, and four stores into a single inline PTX block.  That is a
tighter scheduling request: the CuTe/MLIR layer sees one side-effecting asm
operation, while ptxas gets the exact local instruction stream.

The default dispatch table is generated from CUDA graph replay measurements on
GB200.  Because the real target is B200, `kernel.py` also has an opt-in
just-in-time autotuner: for each shape it times a small candidate set on the
current device, caches the winner by `(device name, SM count, CUDA version,
shape)`, and dispatches to that config.  Use `config_name="jit_autotune"` or
`autotune=True` from Python.

| N | selected config | measured time | bandwidth |
| ---: | --- | ---: | ---: |
| 1024 | `ptx_h2_v4_fma_mix` | 9.120 us | 689.9 GB/s |
| 2048 | `dsl_vectorized_t512` | 9.408 us | 2674.9 GB/s |
| 4096 | `ptx_h2_v8_noalloc_one_tile` | 14.224 us | 7077.0 GB/s |
| 8192 | `ptx_h2_v4_cs_t512_one_tile` | 59.328 us | 6786.9 GB/s |
| 16384 | `ptx_h2_scalar_one_tile_fma_unroll4` | 223.136 us | 7218.1 GB/s |

Full results are in `benchmark_results_cuda_graph_2026_07_02.txt`.

The large-shape winners intentionally launch many CTAs.  N=8192 uses a
512-thread one-tile CTA with `v4.u32` cache-streaming loads/stores.  N=16384
uses `128` threads, `8` fp16 elements per thread, scalar `u32` half2
loads/stores, `fma.rn.f16x2`, and manual unroll4 with `schedule="one_tile"`.
For N=16384 this launches the same 262144 CTA count as `torch.add`.

The latest short 16K graph replay with the new scalar variants measured:

| config | measured time | bandwidth |
| --- | ---: | ---: |
| `ptx_h2_scalar_one_tile_fma_unroll4` | 221.120 us | 7283.9 GB/s |
| `ptx_h2_scalar_one_tile_fma_unroll4_loads_first` | 221.120 us | 7283.9 GB/s |
| `ptx_h2_scalar_one_tile_fma_unroll4_bundle` | 221.352 us | 7276.3 GB/s |
| `torch.add` | 223.136 us | 7218.1 GB/s |

## Run

Correctness:

```bash
pytest -q cute_exercise/ex29_best_vector_add/test_best_vector_add.py
```

Benchmark every knob for every requested shape.  By default this captures each
candidate in a CUDA graph and measures graph replay, so Python/CUDA launch
overhead is excluded:

```bash
python -m cute_exercise.ex29_best_vector_add.bench
```

Use `--eager` only when you explicitly want launch-inclusive timings.

Emit a replacement dispatch table from the measured winners:

```bash
python -m cute_exercise.ex29_best_vector_add.bench --emit-dispatch
```

Include the just-in-time autotuned path in the benchmark.  The first call tunes
only a few candidates for each size using CUDA graph replay, so this is the path
to use on B200 when the GB200 table is only a starting point:

```bash
python -m cute_exercise.ex29_best_vector_add.bench --jit-autotune --emit-dispatch
```

From Python:

```python
out = vector_add_interface(a, b, config_name="jit_autotune")
# equivalent:
out = vector_add_interface(a, b, autotune=True)
```

Run the focused grid search.  The report includes a top-10% option analysis so
config dimensions that never appear in the top decile for every measured shape
can be trimmed.  The expanded generator includes scalar `v1`, unroll4, separate
load/store cache policies, `16`/`32`/`64`/`128` byte alignment assumptions, and
the scalar ordering variants.  It is intentionally broad; use `--size` and
`--limit` for quick slices before running the whole grid:

```bash
python -m cute_exercise.ex29_best_vector_add.autotune --size 8192 --size 16384 --output cute_exercise/ex29_best_vector_add/autotune_results_cuda_graph_2026_07_02.txt
python -m cute_exercise.ex29_best_vector_add.autotune --size 16384 --limit 200 --top-k 20
```

Dump PTX/SASS for selected configs:

```bash
python -m cute_exercise.ex29_best_vector_add.dump_asm --size 2048 --config ptx_h2_v4_fma_mix
python -m cute_exercise.ex29_best_vector_add.dump_asm --size 8192 --config ptx_h2_v4_cs_t512_one_tile
python -m cute_exercise.ex29_best_vector_add.dump_asm --size 16384 --config ptx_h2_scalar_one_tile_fma_unroll4
python -m cute_exercise.ex29_best_vector_add.dump_asm --size 16384 --config ptx_h2_scalar_one_tile_fma_unroll4_loads_first
python -m cute_exercise.ex29_best_vector_add.dump_asm --size 16384 --config ptx_h2_scalar_one_tile_fma_unroll4_bundle
```

This writes raw `.ptx`, `.cubin`, `.sass`, and a combined `.txt` document with
the PTX and SASS code in one file.  The new scalar ordering dumps are in
`dump/ptx_h2_scalar_one_tile_fma_unroll4_loads_first_16384.txt` and
`dump/ptx_h2_scalar_one_tile_fma_unroll4_bundle_16384.txt`.

Profile against `torch.add` with Nsight Compute:

```bash
ncu --profile-from-start off --target-processes all --section LaunchStats --section Occupancy --section SpeedOfLight --section MemoryWorkloadAnalysis --section InstructionStats --section SourceCounters --export cute_exercise/ex29_best_vector_add/ncu/torch_add_N16384 python -m cute_exercise.ex29_best_vector_add.profile_ncu_replay --mode torch --size 16384 --replays 1
ncu --profile-from-start off --target-processes all --section LaunchStats --section Occupancy --section SpeedOfLight --section MemoryWorkloadAnalysis --section InstructionStats --section SourceCounters --export cute_exercise/ex29_best_vector_add/ncu/vector_scalar_one_tile_N16384 python -m cute_exercise.ex29_best_vector_add.profile_ncu_replay --mode vector --size 16384 --config ptx_h2_scalar_one_tile_fma_unroll4 --replays 1
```

The NCU summary and exported SASS source views are in `ncu/`.

## Notes

The 256-bit path exists because CUTLASS has SM100 no-allocation copy atoms and
`ex27_256bit_load` showed that GB200 can keep `LDG.E.256` coalesced.  It still
loses on the largest shape because the persistent/block-limited variants do not
produce enough active work.  The current large-shape winner is less obviously
"vectorized", but it gets the CTA count and occupancy profile closer to
`torch.add` while using fewer registers.
