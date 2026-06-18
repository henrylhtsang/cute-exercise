# Where does a contended global atomic add happen?

## Question

When every thread does many `atomicAdd` operations to the same global
memory accumulator, where is the actual add performed?

Two possible mental models:

1. Each thread loads the accumulator from global memory into a register,
   adds its value in the SM, then stores it back with some acquire/release
   protocol.
2. The SM issues an atomic memory operation, and the read-modify-write is
   performed by the memory hierarchy, with the contended cache line
   serialized near L2.

Also: if every thread in every CTA is hammering the same accumulator, what
does Nsight Compute report as the main stall reason?

## Setup

Use a CuTe DSL microbenchmark with one kernel:

- `hot`: every thread repeatedly atomically adds `1` into `counter[0]`.
- `striped`: each thread atomically adds into one of many counters. This
  keeps global atomics but reduces single-address contention, so it is the
  control case.

The benchmark should:

- verify the final sum equals `num_ctas * threads * iters`;
- dump PTX/SASS so we can see whether the instruction is a global
  reduction/atomic instead of an explicit load/add/store sequence;
- run `ncu` on the hot and striped cases, collecting warp stall reasons and
  L2/atomic-related counters.

## Commands

```bash
python -m cute_exercise.ex23_global_atomic_add.bench
python -m cute_exercise.ex23_global_atomic_add.dump_asm
python -m cute_exercise.ex23_global_atomic_add.profile_ncu
```

The answer belongs in this file after the measurements are taken.

## Answer

Measured on GB200 with:

```bash
python -m cute_exercise.ex23_global_atomic_add.bench
python -m cute_exercise.ex23_global_atomic_add.dump_asm
python -m cute_exercise.ex23_global_atomic_add.profile_ncu
```

Configuration:

- `num_ctas = 608` (`4 * #SMs`)
- `threads = 256`
- `iters = 2048`
- total atomics per measured kernel = `318,767,104`
- striped control = `8192` counters

Benchmark:

| variant | time | throughput |
| --- | ---: | ---: |
| `hot` | `6658.912 us` | `47,870.7 atomic-adds/us` |
| `striped` | `606.096 us` | `525,935.0 atomic-adds/us` |

So the single-counter contention costs roughly `11x` here.

### Where the add happens

It is not compiled as an explicit load into a thread register, register
add, then store. The PTX from CuTe DSL is:

```ptx
atom.global.add.u32 %r1, [%rd1], 1;
```

but because the returned old value is unused, ptxas lowers it to a
global reduction in SASS:

```sass
REDG.E.ADD.STRONG.GPU desc[UR4][R2.64], R5 ;
```

That is a memory-system operation. The SM issues a global reduction
request; the read-modify-write is serialized by the global-memory/L2
atomic path for the target line. For the hot case, almost all requests
target one 32-byte sector, so the line stays L2-resident and DRAM is
basically irrelevant: NCU reports only `4.10 KiB` DRAM reads and `0`
DRAM writes during the profiled hot kernel, with `97.72%` L2 hit rate.

The important consequence: when the return value is unused, the warp is
not waiting for an old accumulator value to come back to a register.
It is mostly blocked because it cannot keep issuing more global-memory
reduction operations.

### Stall reason from NCU

NCU's top stall reason is **LG Throttle**.

Hot single-counter case:

- duration: `7.39 ms`
- scheduler `No Eligible`: `99.76%`
- `Warp Cycles Per Issued Instruction`: `2316.53`
- top stall: `1960.6 cycles` waiting for the L1 instruction queue for
  local/global memory operations to not be full
- stall share: `84.6%`

Striped control:

- duration: `707.42 us`
- scheduler `No Eligible`: `96.90%`
- `Warp Cycles Per Issued Instruction`: `166.91`
- top stall: `153.9 cycles` waiting for the same LG queue
- stall share: `92.2%`

The stall category stays LG Throttle in both cases because both kernels
are intentionally just streams of global reduction atomics. The hot
case is much slower because all warps feed the same L2 atomic
serialization point; the striped case has the same instruction shape
but spreads requests across many sectors, so the memory system can make
far more concurrent progress.
