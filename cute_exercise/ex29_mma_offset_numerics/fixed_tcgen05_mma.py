"""Fixed-shape Blackwell BF16 GEMM probe.

This uses the known-good Blackwell CuTe dense GEMM example from ``~/cutlass``
and pins the CTA tile to MNK = (128, 128, 128). The point of this exercise is
to keep the block shape fixed while varying tensor placement.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

if "CUDA_TOOLKIT_PATH" not in os.environ:
    cuda13 = Path("/usr/local/cuda-13")
    if cuda13.exists():
        os.environ["CUDA_TOOLKIT_PATH"] = str(cuda13)
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

import cutlass  # noqa: E402
from cutlass.cute import experimental as cute_ext  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402


TILE_M = 128
TILE_N = 128
TILE_K = 128

_compile_cache: dict = {}
_dense_gemm_kernel_cls = None


def _load_dense_gemm_kernel_cls():
    """Load CUTLASS's dense GEMM example with a K128 CTA tile."""
    global _dense_gemm_kernel_cls
    if _dense_gemm_kernel_cls is not None:
        return _dense_gemm_kernel_cls

    path = (
        Path.home()
        / "cutlass"
        / "examples"
        / "python"
        / "CuTeDSL"
        / "cute_ext"
        / "blackwell"
        / "dense_gemm.py"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"expected CUTLASS dense_gemm.py at {path}; clone CUTLASS under ~/cutlass"
        )

    source = path.read_text()
    source = source.replace(
        "mma_inst_tile_k = 4  # Number of MMA K-tile subdivisions per mainloop iteration",
        "mma_inst_tile_k = 8  # Number of MMA K-tile subdivisions per mainloop iteration",
    )
    namespace = {"__name__": "_cutlass_dense_gemm_k128", "__file__": str(path)}
    exec(compile(source, str(path), "exec"), namespace)
    _dense_gemm_kernel_cls = namespace["DenseGemmKernel"]
    return _dense_gemm_kernel_cls


def _run_fixed_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    a_ = from_dlpack(a.unsqueeze(-1), assumed_align=1)
    b_ = from_dlpack(b.unsqueeze(-1), assumed_align=1)
    d_ = from_dlpack(out.unsqueeze(-1), assumed_align=1)

    key = (a.dtype, a.shape, a.stride(), b.shape, b.stride(), out.shape, out.stride())
    if key not in _compile_cache:
        kernel_cls = _load_dense_gemm_kernel_cls()
        kernel = kernel_cls(
            mn_tiler=(TILE_M, TILE_N),
            mma_dtype=(cutlass.BFloat16, cutlass.Float32),
            tmem_output_dtype=cutlass.BFloat16,
        )
        _compile_cache[key] = cute_ext.compile(kernel, a_, b_, d_)
    _compile_cache[key](a_, b_, d_)
    return out


def fixed_gemm_interface(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute BF16 GEMM with fixed 128x128x128 CuTe CTA tiles.

    Contract:
    - A: (M, K) BF16, K-major: ``stride[-1] == 1``.
    - B: (N, K) BF16, K-major canonical ``(N, K)`` form.
    - K must be a positive multiple of 128.
    - out: (M, N) BF16, N-major output: ``stride[-1] == 1``.
    """
    assert a.is_cuda and b.is_cuda, "a, b must be cuda"
    assert a.dim() == 2 and b.dim() == 2, "expected 2-D tensors"
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    assert a.shape[1] == b.shape[1], f"K mismatch: A {tuple(a.shape)}, B {tuple(b.shape)}"
    assert a.shape[1] > 0 and a.shape[1] % TILE_K == 0, (
        f"K must be a positive multiple of {TILE_K}, got {a.shape[1]}"
    )
    assert a.stride(-1) == 1, f"A must be K-major, got stride {a.stride()}"
    assert b.stride(-1) == 1, f"B must be K-major (N,K), got stride {b.stride()}"

    if out is None:
        out = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=torch.bfloat16)
    else:
        assert out.shape == (a.shape[0], b.shape[0])
        assert out.dtype == torch.bfloat16
        assert out.is_cuda
        assert out.stride(-1) == 1, f"out must be N-major, got stride {out.stride()}"

    return _run_fixed_gemm(a, b, out)


def fixed_mma_interface(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute one fixed 128x128x128 BF16 GEMM tile.

    Contract:
    - A: (128, 128) BF16, K-major: ``stride[-1] == 1``.
    - B: (128, 128) BF16, K-major canonical ``(N, K)`` form.
    - out: (128, 128) BF16, N-major output: ``stride[-1] == 1``.
    """
    assert a.is_cuda and b.is_cuda, "a, b must be cuda"
    assert a.shape == (TILE_M, TILE_K), f"expected A {(TILE_M, TILE_K)}, got {tuple(a.shape)}"
    assert b.shape == (TILE_N, TILE_K), f"expected B {(TILE_N, TILE_K)}, got {tuple(b.shape)}"
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    assert a.stride(-1) == 1, f"A must be K-major, got stride {a.stride()}"
    assert b.stride(-1) == 1, f"B must be K-major (N,K), got stride {b.stride()}"

    if out is None:
        out = torch.empty((TILE_M, TILE_N), device=a.device, dtype=torch.bfloat16)
    else:
        assert out.shape == (TILE_M, TILE_N)
        assert out.dtype == torch.bfloat16
        assert out.is_cuda
        assert out.stride(-1) == 1, f"out must be N-major, got stride {out.stride()}"

    return _run_fixed_gemm(a, b, out)
