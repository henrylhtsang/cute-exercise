"""Load a pre-compiled CuTe DSL kernel as raw PTX and launch via the CUDA
driver API. No CuTe DSL JIT happens at runtime.

Usage:

    from cute_exercise.ex5_ship_pure_ptx.ptx_runner import elementwise_add_ptx
    out = elementwise_add_ptx(a, b)        # picks dtype + shape from a
    elementwise_add_ptx(a, b, out=out)     # in-place

Behind the scenes:
  * the first call for a given (dtype, shape, device) reads the matching
    ``<artifacts>/elementwise_add_*.ptx`` + ``.manifest.json``,
  * runs ``cuModuleLoadData`` on the PTX (driver JITs to SASS — much
    cheaper than full DSL JIT),
  * caches the resulting ``CUmodule`` + ``CUfunction`` per (kernel,
    device),
  * calls ``cuLaunchKernel`` with 3 device pointers as kernel params.

Importing this module pulls in ``cuda.bindings.driver`` (~25 ms) but
*not* CuTe DSL (~1500 ms). That's the whole point.
"""

from __future__ import annotations

import ctypes
import json
import threading

import torch

import cuda.bindings.driver as cuda

from cute_exercise.ex5_ship_pure_ptx._common import (
    ARTIFACTS_DIR,
    artifact_stem,
    dtype_tag,
    sm_arch_tag,
)


_VOID_P_TUPLE_3 = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)


def _check(result) -> None:
    err = result[0] if isinstance(result, tuple) else result
    if err != cuda.CUresult.CUDA_SUCCESS:
        _, msg = cuda.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {err} ({msg.decode() if msg else ''})")


class _LoadedKernel:
    __slots__ = ("function", "module", "grid", "block", "smem_bytes")

    def __init__(self, function, module, grid, block, smem_bytes):
        self.function = function
        self.module = module
        self.grid = grid
        self.block = block
        self.smem_bytes = smem_bytes


# Cache is intentionally unbounded: in practice there's a small fixed set of
# (dtype, shape, arch, device) tuples per process.
_cache: dict[tuple[str, str, str, int], _LoadedKernel] = {}
_cache_lock = threading.Lock()


def _load(dtype_tag_str: str, shape: tuple[int, int], device: torch.device) -> _LoadedKernel:
    arch = sm_arch_tag(device)
    dev_index = device.index if device.index is not None else torch.cuda.current_device()
    key = (dtype_tag_str, f"{shape[0]}x{shape[1]}", arch, dev_index)

    cached = _cache.get(key)
    if cached is not None:
        return cached

    stem = artifact_stem(dtype_tag_str, shape, arch)
    ptx_path = ARTIFACTS_DIR / f"{stem}.ptx"
    manifest_path = ARTIFACTS_DIR / f"{stem}.manifest.json"
    try:
        ptx_bytes = ptx_path.read_bytes()
        manifest = json.loads(manifest_path.read_text())
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Missing PTX artifact for {dtype_tag_str} {shape[0]}x{shape[1]} on {arch}. "
            f"Run `python -m cute_exercise.ex5_ship_pure_ptx.dump_ptx "
            f"--dtype {dtype_tag_str} --shape {shape[0]} {shape[1]}` first."
        ) from e

    # torch.cuda has already called cuInit; just make sure the right
    # device's primary context is current before cuModuleLoadData.
    with torch.cuda.device(device):
        err, module = cuda.cuModuleLoadData(ptx_bytes)
        _check(err)
        err, function = cuda.cuModuleGetFunction(module, manifest["kernel_name"].encode())
        _check(err)

    loaded = _LoadedKernel(
        function=function,
        module=module,
        grid=tuple(manifest["grid"]),
        block=tuple(manifest["block"]),
        smem_bytes=manifest["smem_bytes"],
    )

    # Double-checked: cuModuleLoadData (~tens of ms PTX→SASS) ran outside
    # the lock; only insert under it and prefer whichever winner wrote first.
    # If we lost the race, unload our redundant CUmodule so it doesn't leak.
    with _cache_lock:
        existing = _cache.get(key)
        if existing is not None:
            with torch.cuda.device(device):
                _check(cuda.cuModuleUnload(module))
            return existing
        _cache[key] = loaded
    return loaded


def elementwise_add_ptx(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Compute ``out = a + b`` using the pre-compiled PTX kernel.

    No CuTe DSL JIT is invoked at runtime — only ``cuModuleLoadData`` (the
    first time the (dtype, shape) is seen) and ``cuLaunchKernel``.
    """
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("a and b must match in shape and dtype")
    if not (a.is_cuda and b.is_cuda and a.is_contiguous() and b.is_contiguous()):
        raise ValueError("a, b must be cuda + contiguous")
    if b.device != a.device:
        raise ValueError(f"a and b must be on the same device; got {a.device} vs {b.device}")
    if a.dim() != 2:
        raise ValueError(f"only 2D tensors supported in this exercise; got {a.dim()}D")

    if out is None:
        out = torch.empty_like(a)
    elif (
        out.shape != a.shape
        or out.dtype != a.dtype
        or not out.is_cuda
        or not out.is_contiguous()
        or out.device != a.device
    ):
        raise ValueError("out must match a in shape/dtype/device and be cuda + contiguous")

    if stream is not None and stream.device != a.device:
        raise ValueError(f"stream device {stream.device} must match tensor device {a.device}")

    loaded = _load(dtype_tag(a.dtype), (a.shape[0], a.shape[1]), a.device)
    cu_stream = (stream if stream is not None else torch.cuda.current_stream(a.device)).cuda_stream
    kernel_args = (
        (a.data_ptr(), b.data_ptr(), out.data_ptr()),
        _VOID_P_TUPLE_3,
    )
    # Make the tensors' device current so cuLaunchKernel resolves the
    # CUfunction against the right primary context. Without this, calls
    # from a process whose current device != a.device hit
    # CUDA_ERROR_INVALID_HANDLE on multi-GPU hosts.
    with torch.cuda.device(a.device):
        _check(
            cuda.cuLaunchKernel(
                loaded.function,
                loaded.grid[0], loaded.grid[1], loaded.grid[2],
                loaded.block[0], loaded.block[1], loaded.block[2],
                loaded.smem_bytes,
                cu_stream,
                kernel_args,
                0,
            )
        )
    return out
