from __future__ import annotations

from contextlib import contextmanager
from typing import Literal
import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cp = None

try:
    import cupyx  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cupyx = None

Backend = Literal["auto", "cpu", "gpu"]


def is_gpu_available() -> bool:
    return cp is not None


def is_cupyx_available() -> bool:
    return cupyx is not None


def resolve_backend(backend: Backend = "auto") -> str:
    if backend == "auto":
        return "gpu" if is_gpu_available() else "cpu"
    if backend == "gpu" and not is_gpu_available():
        raise RuntimeError("backend='gpu' requested, but CuPy is not available.")
    return backend


def xp_from_backend(backend: Backend = "auto"):
    b = resolve_backend(backend)
    return cp if b == "gpu" else np


def to_numpy(x):
    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def to_backend(x, backend: Backend = "auto"):
    b = resolve_backend(backend)
    if b == "gpu":
        return cp.asarray(x)
    return np.asarray(x)


@contextmanager
def gpu_device(device: int = 0):
    if cp is None:
        raise RuntimeError("CuPy is not available.")
    with cp.cuda.Device(int(device)):
        yield
