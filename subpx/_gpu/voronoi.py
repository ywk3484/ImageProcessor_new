"""GPU-accelerated Voronoi partitioning via brute-force nearest neighbor.

Each pixel is assigned to the index of the closest seed, using squared
Euclidean distance.  A single CUDA kernel processes all pixels in parallel,
each checking all N seeds.  For typical photomask images (hundreds to
low-thousands of seeds) this is fast and simple.
"""
from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None

from ..backends import gpu_device


def _require_cupy():
    if cp is None:
        raise RuntimeError("CuPy is required for GPU Voronoi partitioning.")


# ── RawKernel: brute-force nearest-neighbor assignment ──────────────────────
_BRUTE_KERNEL_SRC = r"""
extern "C" __global__
void voronoi_brute(
    const double* __restrict__ seeds,   // (N, 2) [x, y]
    int* __restrict__ labels,           // (H, W) output
    const int H, const int W, const int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= H * W) return;

    int row = idx / W;
    int col = idx % W;

    double px = (double)col;
    double py = (double)row;

    double best_d2 = 1e30;
    int best_label = 0;

    for (int i = 0; i < N; i++) {
        double dx = px - seeds[2 * i];
        double dy = py - seeds[2 * i + 1];
        double d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
            best_d2 = d2;
            best_label = i;
        }
    }
    labels[idx] = best_label;
}
"""

_kernel_cache: dict[int, object] = {}


def _get_brute_kernel(device: int):
    """Cache-compiled RawKernel per device."""
    if device not in _kernel_cache:
        _kernel_cache[device] = cp.RawKernel(_BRUTE_KERNEL_SRC, "voronoi_brute")
    return _kernel_cache[device]


def compute_voronoi_labels_gpu(
    seeds_xy: np.ndarray,
    image_shape: tuple[int, int],
    *,
    device: int = 0,
) -> np.ndarray:
    """Assign each pixel to the nearest seed via Voronoi partitioning.

    Parameters
    ----------
    seeds_xy : (N, 2) float array
        Seed center positions in ``[x, y]`` order (x = column, y = row).
    image_shape : (H, W)
        Image dimensions.
    device : int
        GPU device index.

    Returns
    -------
    label_map : (H, W) int32 ndarray (NumPy)
        Each pixel value is the index of its nearest seed in *seeds_xy*.
    """
    _require_cupy()
    seeds = np.asarray(seeds_xy, dtype=np.float64)
    if seeds.ndim != 2 or seeds.shape[1] != 2:
        raise ValueError(f"seeds_xy must be (N, 2), got {seeds.shape}")

    N = seeds.shape[0]
    H, W = image_shape

    if N == 0:
        return np.full((H, W), -1, dtype=np.int32)
    if N == 1:
        return np.zeros((H, W), dtype=np.int32)

    with gpu_device(device):
        seeds_gpu = cp.asarray(seeds)  # (N, 2) float64, contiguous
        labels_gpu = cp.empty(H * W, dtype=cp.int32)

        kernel = _get_brute_kernel(device)
        total = H * W
        block = 256
        grid = (total + block - 1) // block
        kernel((grid,), (block,), (seeds_gpu, labels_gpu, H, W, N))

        labels_gpu = labels_gpu.reshape(H, W)
        return cp.asnumpy(labels_gpu)
