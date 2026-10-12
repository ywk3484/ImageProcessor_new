"""GPU-accelerated Voronoi partitioning via nearest neighbor.

Each pixel is assigned to the index of the closest seed by squared Euclidean
distance in float64; exact ties go to the lowest seed index, matching
``subpx._cpu.cells.compute_voronoi_labels_cpu``. Two strategies:

- **Brute-force** (N < 256): each pixel checks all N seeds.
- **Grid ring search** (N >= 256): seeds are binned into square grid cells of
  about one seed each; each pixel scans rings of cells outward and stops once
  every unscanned seed is provably farther than the best one found, so pixels
  far from all seeds (clustered seeds, open areas) stay exact.

Pixel indexing uses 64-bit integers to avoid overflow on large images.
"""
from __future__ import annotations

import numpy as np

from ..backends import gpu_device
from ._cuda import cp, kernel

# Distances compile without FMA contraction: they round like NumPy's, so ties
# and near-ties resolve as on the CPU.
_KERNEL_SRC = r"""
extern "C" __global__ void voronoi_brute(
    const double* __restrict__ seeds, int* __restrict__ labels, const long long HW, const int W, const int N)
{
    const long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= HW) return;
    const double px = (double)(idx % W), py = (double)(idx / W);
    double best = 0;
    int label = -1;
    for (int i = 0; i < N; ++i) {
        const double dx = px - seeds[2 * i], dy = py - seeds[2 * i + 1];
        const double d2 = dx * dx + dy * dy;
        if (label < 0 || d2 < best) { best = d2; label = i; }
    }
    labels[idx] = label;
}

extern "C" __global__ void voronoi_ring(
    const double* __restrict__ seeds, const int* __restrict__ order, const int* __restrict__ start,
    int* __restrict__ labels, const long long HW, const int W, const int gw, const int gh, const int cell)
{
    const long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= HW) return;
    const int col = (int)(idx % W), row = (int)(idx / W);
    const double px = (double)col, py = (double)row;
    const int cx = min(col / cell, gw - 1), cy = min(row / cell, gh - 1);
    double best = 0;
    int label = -1;
    for (int R = 0;; ++R) {
        for (int gy = cy - R; gy <= cy + R; ++gy) {
            if (gy < 0 || gy >= gh) continue;
            const bool edge_row = gy == cy - R || gy == cy + R;
            for (int gx = cx - R; gx <= cx + R; gx += (edge_row ? 1 : 2 * R)) {
                if (gx >= 0 && gx < gw) {
                    const int g = gy * gw + gx;
                    for (int k = start[g]; k < start[g + 1]; ++k) {
                        const int i = order[k];
                        const double dx = px - seeds[2 * i], dy = py - seeds[2 * i + 1];
                        const double d2 = dx * dx + dy * dy;
                        if (label < 0 || d2 < best || (d2 == best && i < label)) { best = d2; label = i; }
                    }
                }
                if (R == 0) break;
            }
        }
        // Seeds outside the scanned block lie beyond one of its unclipped sides.
        double bound = 1e300;
        bool all = true;
        if (cx - R > 0) { bound = fmin(bound, px - (double)(cx - R) * cell); all = false; }
        if (cy - R > 0) { bound = fmin(bound, py - (double)(cy - R) * cell); all = false; }
        if (cx + R < gw - 1) { bound = fmin(bound, (double)(cx + R + 1) * cell - px); all = false; }
        if (cy + R < gh - 1) { bound = fmin(bound, (double)(cy + R + 1) * cell - py); all = false; }
        if (all || (label >= 0 && best < bound * bound)) break;
    }
    labels[idx] = label;
}
"""

# Auto-select threshold: grid for N >= this, brute-force below
_GRID_THRESHOLD = 256


def _labels_device(seeds_xy: np.ndarray, image_shape: tuple[int, int]):
    """(H, W) int32 CuPy label map of the nearest seed (current device)."""
    seeds = np.ascontiguousarray(seeds_xy, dtype=np.float64)
    n = len(seeds)
    H, W = image_shape
    HW = H * W
    if n == 0:
        return cp.full((H, W), -1, dtype=cp.int32)
    labels = cp.empty(HW, dtype=cp.int32)
    block, grid = 256, (HW + 255) // 256
    if n < _GRID_THRESHOLD:
        kernel(_KERNEL_SRC, "voronoi_brute", exact=True)(
            (grid,), (block,), (cp.asarray(seeds), labels, np.int64(HW), np.int32(W), np.int32(n)))
        return labels.reshape(H, W)
    # About one seed per grid cell; integer binning keeps every seed inside its cell's bounds.
    cell = max(1, int(np.sqrt(HW / n)))
    gw, gh = -(-W // cell), -(-H // cell)
    bx = np.clip(np.floor(seeds[:, 0]).astype(np.int64) // cell, 0, gw - 1)
    by = np.clip(np.floor(seeds[:, 1]).astype(np.int64) // cell, 0, gh - 1)
    bins = by * gw + bx
    order = np.argsort(bins, kind="stable").astype(np.int32)
    start = np.zeros(gw * gh + 1, dtype=np.int32)
    np.cumsum(np.bincount(bins, minlength=gw * gh), out=start[1:])
    kernel(_KERNEL_SRC, "voronoi_ring", exact=True)(
        (grid,), (block,),
        (cp.asarray(seeds), cp.asarray(order), cp.asarray(start), labels, np.int64(HW), np.int32(W),
         np.int32(gw), np.int32(gh), np.int32(cell)))
    return labels.reshape(H, W)


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
        Each pixel value is the index of its nearest seed in *seeds_xy*;
        exact ties go to the lowest index. -1 everywhere when there are no seeds.
    """
    if cp is None:
        raise RuntimeError("CuPy is required for GPU Voronoi partitioning.")
    seeds = np.asarray(seeds_xy, dtype=np.float64)
    if seeds.ndim != 2 or seeds.shape[1] != 2:
        raise ValueError(f"seeds_xy must be (N, 2), got {seeds.shape}")
    with gpu_device(device):
        return cp.asnumpy(_labels_device(seeds, image_shape))
