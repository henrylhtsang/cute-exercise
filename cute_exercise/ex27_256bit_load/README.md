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

Write a small plain-CUDA / inline-PTX microbenchmark that compares:

- the existing 128-bit vectorized load/store pattern from `ex1_for_loop`;
- an equivalent 256-bit PTX load/store path;
- scalar or 64-bit variants as controls.

For each variant:

- dump PTX and SASS, and record whether the final SASS contains one wider
  instruction or multiple 128-bit/64-bit instructions;
- benchmark bandwidth on a large contiguous elementwise copy/add workload;
- inspect memory-sector counters in NCU to confirm how the warp accesses
  coalesce.

## Answer

Measured on GB200 (`sm_100`) with CUDA 13.3:

```bash
sudo dnf install -y cuda-13-3 --allowerasing
sudo alternatives --set cuda /usr/local/cuda-13.3
python cute_exercise/ex27_256bit_load/build.py --run --n 67108864 --iters 100
python cute_exercise/ex27_256bit_load/dump_asm.py
python cute_exercise/ex27_256bit_load/profile_ncu.py --n 16777216 --iters 5
```

`nvcc --version`:

```text
Cuda compilation tools, release 13.3, V13.3.33
```

The plain CUDA probe uses inline PTX for the vector paths:

- `v4_u32`: `ld.global.v4.u32` / `st.global.v4.u32`
- `v8_u32`: `ld.global.v8.u32` / `st.global.v8.u32`

On GB200, ptxas does not split the 256-bit load. The SASS contains:

```text
LDG.E.128        ...  # v4_u32
STG.E.128        ...
LDG.E.ENL2.256   ...  # v8_u32
STG.E.ENL2.256   ...
```

ptxas register counts:

| variant | registers |
| --- | ---: |
| `u32` | 14 |
| `v4_u32` | 16 |
| `v8_u32` | 20 |

NCU load-coalescing counters for `n=16,777,216` `uint32` words and
`iters=5`:

| variant | load requests | load sectors | sectors/request |
| --- | ---: | ---: | ---: |
| `u32` | 2,621,440 | 10,485,760 | 4 |
| `v4_u32` | 655,360 | 10,485,760 | 16 |
| `v8_u32` | 327,680 | 10,485,760 | 32 |

The sector count is exactly the ideal useful byte count divided by the 32-byte
sector size:

```text
16,777,216 words * 4 bytes/word * 5 iters / 32 bytes/sector = 10,485,760 sectors
```

So the 256-bit per-thread load is still coalesced across the warp. It creates
half as many warp-level load requests as the 128-bit path while touching the
same ideal sector count.

Timing outside NCU for `n=67,108,864`, `iters=100`:

| variant | kernel time | load GiB/s |
| --- | ---: | ---: |
| `u32` | 16.67 ms | 1,500 |
| `v4_u32` | 8.66 ms | 2,887 |
| `v8_u32` | 9.02 ms | 2,772 |

The 256-bit form reduces dynamic load requests, but it did not improve this
copy microbenchmark over 128-bit. It also costs more registers. For this
simple contiguous traffic pattern on GB200, 128-bit remains the better
practical per-thread width unless a kernel has a specific reason to consume a
32-byte contiguous fragment per thread.
