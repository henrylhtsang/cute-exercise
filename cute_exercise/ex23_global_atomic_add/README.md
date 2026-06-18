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
