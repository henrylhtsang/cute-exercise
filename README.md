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

1. [How bad is a `for` loop in CuTe DSL?](cute_exercise/for_loop/)
2. [Load global → smem → rmem, op, store back — or straight to rmem?](cute_exercise/load_path/)
3. [TMA load → elementwise add → TMA store, no warp spec. Faster than sync copy?](cute_exercise/tma_no_warpspec/)
