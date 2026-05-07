# CuTe DSL transpose notes

This exercise implements an out-of-place 2D transpose:

```python
B[i, j] = A[j, i]
```

The final checked-in kernel is a hybrid:

1. TMA-load a contiguous `32 x 32` tile from `A` into shared memory using a
   SW128-swizzled SMEM layout to reduce bank conflicts on the transposed
   read.
2. Wait on the TMA mbarrier.
3. Read the tile transposed from shared memory into registers.
4. Store the output tile with a thread-level `cute.copy` to `B`.

This is intentionally not a pure TMA-load plus TMA-store solution. For this
exercise, the hybrid is the clearest correct version and exposes the important
CuTe/TMA synchronization and layout lessons.

## Current design

The SMEM layout is built as a `Swizzle ∘ Layout` composition, then tiled to
the full `32 x 32` shape:

```python
swizzle = cute.make_swizzle(3, 4, 2)
smem_atom_layout = cute.make_layout((8, 32), stride=(32, 1))
smem_layout = cute.make_composed_layout(
    inner=swizzle, offset=0, outer=smem_atom_layout,
)
smem_layout = cute.tile_to_shape(smem_layout, (32, 32), order=(1, 0))
```

The swizzle XOR-permutes the linear SMEM offset so that the column-strided
read `sA[tidx, i]` no longer collides on a single bank. Logical indexing
(`sA[m, n]`) is unchanged; the swizzle is transparent to the user, and TMA's
hardware swizzle field in the descriptor matches the layout, so what TMA
writes is what `cute` reads.

The host then builds a TMA load atom for `A` using this swizzled layout:

```python
tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
    cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
    A,
    smem_layout,
    (32, 32),
)
```

The output tensor `B` is tiled with `cute.zipped_divide`, because the store side
is a normal thread-level copy:

```python
gB = cute.zipped_divide(B, (32, 32))
```

In the kernel, CTA `(bidx, bidy)` reads `A` tile `(bidx, bidy)` and writes the
corresponding transposed tile of `B` at tile coordinate `(bidy, bidx)`.

```python
gA_tile = cute.local_tile(tma_tensor_a, (32, 32), (bidx, bidy))
gB_tile = gB[((None, None), (bidy, bidx))]
```

`tma_tensor_a` is not a pre-tiled `((32, 32), tile_grid)` view. It still carries
the logical tensor coordinate space that the TMA descriptor understands, so the
per-CTA tile is selected with `cute.local_tile`.

## Why TMA for the load

The load is a good fit for TMA:

- Each CTA reads one contiguous `32 x 32` fp32 tile from row-major `A`.
- The TMA descriptor handles the 2D global-to-shared transfer.
- The transfer is independent of the participating threads once issued.
- It gives a direct way to learn the TMA descriptor and mbarrier protocol.

The required mbarrier sequence is:

```python
with cute.arch.elect_one():
    cute.arch.mbarrier_arrive_and_expect_tx(mbar, 32 * 32 * 4)

cute.copy(tma_atom_a, tAgA, tAsA, tma_bar_ptr=mbar)
cute.arch.mbarrier_wait(mbar, phase=0)
```

`cute.copy` for the TMA load must be outside the `elect_one` region. The elected
thread sets up the mbarrier transaction count; the copy itself is issued through
the CuTe TMA copy path.

The `sync_threads()` after `mbarrier_init_fence()` is still needed because only
one elected thread initializes the mbarrier. The later `sync_threads()` after
`mbarrier_wait()` is redundant for this kernel because all participating threads
wait on the mbarrier before reading shared memory.

## Why `cute.copy` for the store

The transpose happens when each thread reads shared memory in the opposite
order:

```python
for i in cutlass.range_constexpr(32):
    tBrB[i] = sA[tidx, i]
```

Then the thread stores a contiguous vector into the output tile:

```python
cute.copy(copy_atom, tBrB, tBgB)
```

This keeps global stores coalesced and avoids forcing TMA store to express the
in-tile transpose. A TMA store is best when the shared-memory tile already has
the layout that should be written to global memory. In this exercise, doing the
transpose through register reads from shared memory is simpler and more explicit.

Using raw `cpasync.CopyBulkTensorTileS2GOp()` directly with `cute.copy` is wrong;
that op must first be turned into a TMA atom with `make_tiled_tma_atom`. For this
hybrid kernel, a normal `cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), ...)`
is the right store primitive.

## Experiments and results

Important checkpoints from the exercise:

- Direct global-memory XOR/swizzle did not coalesce stores. The global store
  bytes-per-sector ratio was poor, because swizzling lane order does not make
  strided global addresses contiguous.
- Shared-memory staging with no padding was the fastest early variant despite
  bank conflicts. It kept instruction count low and global accesses coalesced.
- `+1` padding reduced shared-memory bank conflicts but increased instruction
  count and slowed the benchmark.
- CuTe SMEM swizzle variants on the earlier non-TMA cp.async kernel
  (`MN_SW32`, `MN_SW64`, `MN_SW128`) were correct but slower than the
  unpadded shared-memory variant for this small fixed tile.
- On the TMA-hybrid kernel, adding a SW128-shaped swizzle
  (`Swizzle<3,4,2>` composed with `(8,32):(32,1)`, tiled to `(32,32)`)
  gave a small but consistent speedup (2-10%) by reducing the column-read
  bank conflicts. This is the swizzle that is checked in.
- The final TMA-load hybrid is correct and useful as a TMA learning endpoint, but
  it is not faster than the best non-TMA shared-memory kernel on all shapes.

Recent benchmark for the final hybrid (with SW128 swizzle, GB200, fp32):

```text
shape         torch ms   cute ms   cute / torch   cute GB/s
4096x4096      0.0817    0.0594        0.73x        2260
8192x1024      0.0465    0.0469        1.01x        1432
1024x8192      0.0470    0.0483        1.03x        1390
4096x8192      0.1518    0.0825        0.54x        3253
```

The best earlier non-TMA shared-memory variant was faster for `4096 x 4096`
(`~0.037 ms`), so the current kernel should be viewed as the best TMA-learning
version, not the absolute best transpose kernel found in the exercise.

## Current recommendation

For this branch, keep the current hybrid:

- TMA load for the input tile.
- SW128-swizzled SMEM layout to reduce column-read bank conflicts.
- Mbarrier wait for load completion.
- Register transpose from shared memory.
- Thread-level `cute.copy` for the output store.

This is the best mergeable version for documenting the TMA mechanics. The next
optimization experiment would be a carefully designed TMA-store variant with a
shared-memory layout that already matches the desired global output layout, but
that is a separate exercise.
