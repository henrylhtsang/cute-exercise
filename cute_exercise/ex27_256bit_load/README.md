# Do 256-bit PTX global loads actually coalesce?

## Question

In `ex1_for_loop`, the working model is that each thread can issue a
128-bit global load/store instruction (`ld.global.v4.b32` in PTX,
`LDG.E.128` in SASS), and the warp-level coalescer combines the 32
per-thread addresses into contiguous memory sectors when the layout is good.

Newer PTX exposes 256-bit global load forms. So: if each thread issues a
256-bit global load from consecutive addresses, what actually happens?
Specifically:

1. Does ptxas lower the 256-bit PTX load to a real single 256-bit global
   memory instruction, or split it into multiple 128-bit instructions?
2. If it splits, are the two halves independently coalesced across the warp?
3. Does the wider PTX load reduce instruction count or improve bandwidth, or
   does it mostly increase register pressure / scheduling constraints?
4. Is 128-bit still the practical per-thread sweet spot for coalesced global
   memory traffic on current NVIDIA GPUs?

## Setup

Write a small CuTe DSL / inline-PTX microbenchmark that compares:

- the existing 128-bit vectorized load/store pattern from `ex1_for_loop`;
- an equivalent 256-bit PTX load/store path, if CuTe DSL / inline PTX can
  express it cleanly;
- scalar or 64-bit variants as controls.

For each variant:

- dump PTX and SASS, and record whether the final SASS contains one wider
  instruction or multiple 128-bit/64-bit instructions;
- benchmark bandwidth on a large contiguous elementwise copy/add workload;
- inspect memory-sector counters in NCU to confirm how the warp accesses
  coalesce.

## Answer

_TODO: fill in after measurements on GB200._
