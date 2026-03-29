"""GPU-accelerated Voronoi partitioning via nearest neighbor.

Each pixel is assigned to the index of the closest seed, using squared
Euclidean distance.  Two strategies:

- **Brute-force** (N < 256): each pixel checks all N seeds.
- **Grid-accelerated** (N >= 256): seeds are binned into a spatial grid;
  each pixel checks only seeds in nearby cells.  Reduces complexity from
  O(H*W*N) to O(H*W*K) where K is a small constant (~50).

All distance computations use float32 for throughput on GPUs with limited
FP64 units.  Pixel indexing uses 64-bit integers to avoid overflow on
large images.
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


# ── RawKernel: brute-force nearest-neighbor (small N) ───────────────────────
_BRUTE_KERNEL_SRC = r"""
extern "C" __global__
void voronoi_brute(
    const float* __restrict__ seeds,   // (N, 2) [x, y]
    int* __restrict__ labels,          // (H*W,) output
    const long long HW, const int W, const int N
) {
    long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= HW) return;

    int row = (int)(idx / W);
    int col = (int)(idx % W);

    float px = (float)col;
    float py = (float)row;

    float best_d2 = 1e30f;
    int best_label = 0;

    for (int i = 0; i < N; i++) {
        float dx = px - seeds[2 * i];
        float dy = py - seeds[2 * i + 1];
        float d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
            best_d2 = d2;
            best_label = i;
        }
    }
    labels[idx] = best_label;
}
"""

# ── RawKernel: grid-accelerated nearest-neighbor (large N) ──────────────────
_GRID_KERNEL_SRC = r"""
extern "C" __global__
void voronoi_grid(
    const float* __restrict__ seeds,          // (N, 2) [x, y]
    const int* __restrict__ sorted_indices,   // (N,) original seed index
    const int* __restrict__ cell_start,       // (grid_h*grid_w + 1,)
    int* __restrict__ labels,                 // (H*W,) output
    const long long HW, const int W,
    const int grid_w, const int grid_h,
    const int cell_size, const int search_radius
) {
    long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= HW) return;

    int row = (int)(idx / W);
    int col = (int)(idx % W);

    float px = (float)col;
    float py = (float)row;

    // Grid cell for this pixel
    int cx = col / cell_size;
    int cy = row / cell_size;

    float best_d2 = 1e30f;
    int best_label = 0;

    // Search neighborhood of grid cells
    int cx_lo = cx - search_radius;
    int cx_hi = cx + search_radius;
    int cy_lo = cy - search_radius;
    int cy_hi = cy + search_radius;
    if (cx_lo < 0) cx_lo = 0;
    if (cy_lo < 0) cy_lo = 0;
    if (cx_hi >= grid_w) cx_hi = grid_w - 1;
    if (cy_hi >= grid_h) cy_hi = grid_h - 1;

    for (int gy = cy_lo; gy <= cy_hi; gy++) {
        for (int gx = cx_lo; gx <= cx_hi; gx++) {
            int cell_idx = gy * grid_w + gx;
            int start = cell_start[cell_idx];
            int end   = cell_start[cell_idx + 1];

            for (int k = start; k < end; k++) {
                int si = sorted_indices[k];
                float dx = px - seeds[2 * si];
                float dy = py - seeds[2 * si + 1];
                float d2 = dx * dx + dy * dy;
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best_label = si;
                }
            }
        }
    }

    labels[idx] = best_label;
}
"""

_NN_DIST_KERNEL_SRC = r"""
extern "C" __global__
void nn_dist_grid(
    const float* __restrict__ seeds,
    const int* __restrict__ sorted_indices,
    const int* __restrict__ cell_start,
    float* __restrict__ nn_d2,
    const int N,
    const int grid_w, const int grid_h,
    const int cell_size, const int nn_search_radius
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;

    float sx = seeds[2 * i];
    float sy = seeds[2 * i + 1];

    int cx = (int)(sx / (float)cell_size);
    int cy = (int)(sy / (float)cell_size);
    if (cx < 0) cx = 0;
    if (cx >= grid_w) cx = grid_w - 1;
    if (cy < 0) cy = 0;
    if (cy >= grid_h) cy = grid_h - 1;

    float best_d2 = 1e30f;

    int cx_lo = cx - nn_search_radius;
    int cx_hi = cx + nn_search_radius;
    int cy_lo = cy - nn_search_radius;
    int cy_hi = cy + nn_search_radius;
    if (cx_lo < 0) cx_lo = 0;
    if (cy_lo < 0) cy_lo = 0;
    if (cx_hi >= grid_w) cx_hi = grid_w - 1;
    if (cy_hi >= grid_h) cy_hi = grid_h - 1;

    for (int gy = cy_lo; gy <= cy_hi; gy++) {
        for (int gx = cx_lo; gx <= cx_hi; gx++) {
            int cell_idx = gy * grid_w + gx;
            int start = cell_start[cell_idx];
            int end   = cell_start[cell_idx + 1];

            for (int k = start; k < end; k++) {
                int j = sorted_indices[k];
                if (j == i) continue;
                float dx = sx - seeds[2 * j];
                float dy = sy - seeds[2 * j + 1];
                float d2 = dx * dx + dy * dy;
                if (d2 < best_d2) best_d2 = d2;
            }
        }
    }

    nn_d2[i] = best_d2;
}
"""

_kernel_cache: dict[tuple[int, str], object] = {}

# Auto-select threshold: grid for N >= this, brute-force below
_GRID_THRESHOLD = 256


def _get_kernel(device: int, name: str):
    """Return a cache-compiled RawKernel for *device*."""
    key = (device, name)
    if key not in _kernel_cache:
        src = {"voronoi_brute": _BRUTE_KERNEL_SRC,
               "voronoi_grid": _GRID_KERNEL_SRC,
               "nn_dist_grid": _NN_DIST_KERNEL_SRC}[name]
        _kernel_cache[key] = cp.RawKernel(src, name)
    return _kernel_cache[key]


def _bin_seeds_to_grid(seeds_gpu, H, W, cell_size):
    """Bin seeds into a spatial grid for accelerated lookup.

    Returns
    -------
    sorted_indices : (N,) int32 CuPy — original seed indices sorted by cell
    cell_start : (total_cells+1,) int32 CuPy — cumulative start offsets
    grid_w, grid_h : int — grid dimensions
    """
    grid_w = (W + cell_size - 1) // cell_size
    grid_h = (H + cell_size - 1) // cell_size
    total_cells = grid_w * grid_h

    # Cell index for each seed
    cell_x = cp.clip((seeds_gpu[:, 0] / cell_size).astype(cp.int32),
                      0, grid_w - 1)
    cell_y = cp.clip((seeds_gpu[:, 1] / cell_size).astype(cp.int32),
                      0, grid_h - 1)
    cell_idx = cell_y * grid_w + cell_x

    # Sort seeds by cell index
    order = cp.argsort(cell_idx)
    sorted_cell_idx = cell_idx[order]

    # Build cell_start via bincount + cumsum
    counts = cp.bincount(sorted_cell_idx.astype(cp.int32),
                          minlength=total_cells)
    cell_start = cp.zeros(total_cells + 1, dtype=cp.int32)
    cp.cumsum(counts, out=cell_start[1:])

    return order.astype(cp.int32), cell_start, grid_w, grid_h


def _max_nn_dist_via_grid(seeds_gpu, sorted_idx, cell_start,
                          grid_w, grid_h, cell_size, device):
    """Max nearest-neighbor distance among seeds via grid-local search.

    Returns float.  If any seed has no neighbor within the search window
    (5x5 grid cells), returns ``inf`` so the caller falls back to
    brute-force Voronoi.
    """
    N = seeds_gpu.shape[0]
    nn_search_radius = 2  # 5×5 cells

    nn_d2 = cp.empty(N, dtype=cp.float32)
    block = 256
    grid = (N + block - 1) // block
    kernel = _get_kernel(device, "nn_dist_grid")
    kernel(
        (grid,), (block,),
        (seeds_gpu, sorted_idx, cell_start, nn_d2,
         np.int32(N), np.int32(grid_w), np.int32(grid_h),
         np.int32(cell_size), np.int32(nn_search_radius)),
    )

    max_d2 = float(nn_d2.max())
    if max_d2 >= 1e29:  # at least one seed found no neighbor
        return float("inf")
    return float(cp.sqrt(cp.float32(max_d2)))


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
    seeds = np.asarray(seeds_xy, dtype=np.float32)
    if seeds.ndim != 2 or seeds.shape[1] != 2:
        raise ValueError(f"seeds_xy must be (N, 2), got {seeds.shape}")

    N = seeds.shape[0]
    H, W = image_shape

    if N == 0:
        return np.full((H, W), -1, dtype=np.int32)
    if N == 1:
        return np.zeros((H, W), dtype=np.int32)

    HW = H * W
    block = 256
    grid = (HW + block - 1) // block

    with gpu_device(device):
        seeds_gpu = cp.asarray(seeds)  # (N, 2) float32, contiguous
        labels_gpu = cp.empty(HW, dtype=cp.int32)

        if N < _GRID_THRESHOLD:
            kernel = _get_kernel(device, "voronoi_brute")
            kernel(
                (grid,), (block,),
                (seeds_gpu, labels_gpu,
                 np.int64(HW), np.int32(W), np.int32(N)),
            )
        else:
            # Cell size ≈ average seed spacing
            cell_size = max(16, int(np.sqrt(HW / N)))

            # Bin seeds into grid (reused for both NN search and Voronoi)
            sorted_idx, cell_start, gw, gh = _bin_seeds_to_grid(
                seeds_gpu, H, W, cell_size,
            )

            # Compute max nearest-neighbor distance via grid-local
            # search: O(N*K) instead of O(N^2).
            max_nn_dist = _max_nn_dist_via_grid(
                seeds_gpu, sorted_idx, cell_start,
                gw, gh, cell_size, device,
            )

            if not np.isfinite(max_nn_dist):
                search_radius = 999  # force brute-force
            else:
                search_radius = int(np.ceil(max_nn_dist / cell_size)) + 1

            # Fall back to brute-force if search would be too wide
            if search_radius > 5:
                kernel = _get_kernel(device, "voronoi_brute")
                kernel(
                    (grid,), (block,),
                    (seeds_gpu, labels_gpu,
                     np.int64(HW), np.int32(W), np.int32(N)),
                )
            else:
                kernel = _get_kernel(device, "voronoi_grid")
                kernel(
                    (grid,), (block,),
                    (seeds_gpu, sorted_idx, cell_start, labels_gpu,
                     np.int64(HW), np.int32(W),
                     np.int32(gw), np.int32(gh),
                     np.int32(cell_size), np.int32(search_radius)),
                )

        labels_gpu = labels_gpu.reshape(H, W)
        return cp.asnumpy(labels_gpu)
