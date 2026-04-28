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
