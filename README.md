# cute-exercise
Exercises for CuTe DSL.

## Install

Fresh conda env with everything needed:

```bash
conda create -n cute-exercise python=3.12 -y && conda activate cute-exercise && pip install --extra-index-url https://download.pytorch.org/whl/cu130 torch pytest "nvidia-cutlass-dsl[cu13]==4.4.2" && pip install -e .
```

If you already have an env, we recommend installing PyTorch from the CUDA 13.0 wheel index:

```bash
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu130
```

Requires Python >= 3.12.

## Hardware

Most measurements here were taken on an NVIDIA GB200.

## Structure

Each exercise lives in its own subdirectory under `cute_exercise/`. The
subdirectory contains:

- a detailed problem statement (README)
- a CuTe DSL implementation
- a script to run it, test correctness, and benchmark

## Exercises

1. [How bad is a `for` loop in CuTe DSL?](cute_exercise/ex1_for_loop/)
2. [Load global → smem → rmem, op, store back — or straight to rmem?](cute_exercise/ex2_load_path/)
3. [TMA load → elementwise add → TMA store, no warp spec. Faster than sync copy?](cute_exercise/ex3_tma_no_warpspec/)
4. [`cp.reduce.async.bulk` for vec add — does it even make sense?](cute_exercise/ex4_reduce_bulk_vec_add/)
5. [Ship a CuTe DSL kernel as pre-compiled PTX](cute_exercise/ex5_ship_pure_ptx/)
6. [Single-tile TF32 matmul — TMEM + `tcgen05` practice](cute_exercise/ex6_single_tile_tf32_matmul/)
7. [TF32×3 matmul — does the precision actually improve?](cute_exercise/ex7_tf32x3_matmul/)
8. [Skip the `fence.proxy.async` between generic SMEM writes and TMA — what breaks?](cute_exercise/ex8_missing_proxy_fence/)
9. [How do I write a transpose in CuTe DSL?](cute_exercise/ex9_transpose/)
10. [How does SMEM / L2 swizzle work? (blockwise matrix add)](cute_exercise/ex10_swizzle_blockwise_add/)
11. [How does TMA load and store zeroing work?](cute_exercise/ex11_tma_zeroing/)
12. [How do I write an FP8 matmul, with different scaling granularities?](cute_exercise/ex12_fp8_matmul/)
13. [How do I write an MXFP8 matmul?](cute_exercise/ex13_mxfp8_matmul/)
14. [How do I write an NVFP4 matmul?](cute_exercise/ex14_nvfp4_matmul/)
15. [How does distributed shared memory work? (cluster histogram)](cute_exercise/ex15_dsmem_histogram/)
16. [How do I write code that lowers to the R2P SASS instruction?](cute_exercise/ex16_r2p_masking/)
17. [How does TMA work when one stride needs to be dynamic?](cute_exercise/ex17_tma_dynamic_stride/)
18. [Does TMA multicast actually save bandwidth on a memory-bound op?](cute_exercise/ex18_tma_multicast/)
19. [How does 2-CTA MMA (`tcgen05.mma cta_group::2`) work, and when does it help?](cute_exercise/ex19_2cta_mma/)
21. [How do I build a Blackwell `tcgen05` MMA from scratch?](cute_exercise/ex21_tcgen05_mma/)
22. [Can I fuse two kernels into one with a grid-wide barrier?](cute_exercise/ex22_grid_sync/)
23. [Where does a contended global atomic add happen?](cute_exercise/ex23_global_atomic_add/)
24. [How do I write a fast LayerNorm for different sizes?](cute_exercise/ex24_layernorm/)
25. [How do I write a fused MoE kernel?](cute_exercise/ex25_fused_moe/)
26. [How do I write a top-k kernel?](cute_exercise/ex26_topk/)
27. [Do 256-bit PTX global loads actually coalesce?](cute_exercise/ex27_256bit_load/)
28. [How do I inspect the embedded CuTe DSL CUDA compiler?](cute_exercise/ex28_cute_dsl_toolchain/)
29. [Does an MMA tile give bitwise-identical numerics when the result is offset?](cute_exercise/ex29_mma_offset_numerics/)
30. [What are the TPC groupings on GB200?](cute_exercise/ex30_gb200_tpc_groupings/)
31. [Does transposing the MMA operand roles affect numerics?](cute_exercise/ex31_mma_transpose_numerics/)
32. [Can CUDA Graphs use `if` / control flow?](cute_exercise/ex32_cuda_graph_if_node/)
