"""Shared helpers for ex5: dtype tags, arch tags, artifact paths."""

from __future__ import annotations

from pathlib import Path

import torch


DTYPES: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

_DTYPE_TO_TAG = {v: k for k, v in DTYPES.items()}

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def dtype_tag(dtype: torch.dtype) -> str:
    return _DTYPE_TO_TAG[dtype]


def sm_arch_tag(device: torch.device | int | None = None) -> str:
    """Return e.g. ``sm_100a`` for the given (or current) CUDA device."""
    major, minor = torch.cuda.get_device_capability(device)
    suffix = "a" if major >= 9 else ""
    return f"sm_{major}{minor}{suffix}"


def artifact_stem(dtype_name: str, shape: tuple[int, int], arch: str) -> str:
    m, n = shape
    return f"elementwise_add_{dtype_name}_{m}x{n}_{arch}"
