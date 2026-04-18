"""Base class for CuTe DSL kernels."""

from abc import ABC, abstractmethod


class CuteDSLKernel(ABC):
    """Base class for a CuTe DSL kernel.

    Subclasses implement:
      * ``__call__`` decorated with ``@cute.jit`` — host-side launcher that
        does tiling/setup and launches the inner GPU kernel
      * ``kernel`` decorated with ``@cute.kernel`` — the GPU kernel itself

    Compile and run via ``cute.compile(op, *tensors)`` as usual.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        ...
