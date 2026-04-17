# cute-exercise
Exercises for CuTe DSL.

## Install

PyTorch must come from the CUDA 13.0 wheel index:

```bash
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu130
```

Requires Python >= 3.12.

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
