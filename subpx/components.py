"""Connected-component labeling utilities.

Public API returns NumPy arrays regardless of backend.
"""

from __future__ import annotations

import numpy as np

from .backends import resolve_backend, to_numpy
from ._cpu.components import connected_components_stats_cpu
from ._gpu.components import connected_components_stats_gpu


def connected_components_stats(mask: np.ndarray, *, backend: str = "auto", connectivity: int = 8, device: int = 0):
    b = resolve_backend(backend)
    m = to_numpy(mask)
    if b == "gpu":
        return connected_components_stats_gpu(m, connectivity=connectivity, device=device)
    return connected_components_stats_cpu(m, connectivity=connectivity)
