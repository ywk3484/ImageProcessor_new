from __future__ import annotations

import numpy as np

from ..types import ComponentStatsResult
from ..backends import gpu_device

try:
    import cupy as cp  # type: ignore
    from cupyx.scipy import ndimage as cnd  # type: ignore
except Exception:  # pragma: no cover
    cp = None
    cnd = None


def _require_gpu_deps():
    if cp is None or cnd is None:
        raise RuntimeError("CuPy with cupyx.scipy.ndimage is required for GPU connected components.")


# ---------------------------------------------------------------------------
# CUDA RawKernel: single-pass CC stats via GPU atomics
# ---------------------------------------------------------------------------
_CC_STATS_KERNEL_SRC = r"""
extern "C" __global__
void cc_stats_2d(const int* __restrict__ lab,
                 const int H, const int W,
                 int* __restrict__ min_x, int* __restrict__ min_y,
                 int* __restrict__ max_x, int* __restrict__ max_y,
                 unsigned int* __restrict__ area,
                 double* __restrict__ sum_x, double* __restrict__ sum_y)
{
    long long idx = (long long)blockDim.x * (long long)blockIdx.x
                  + (long long)threadIdx.x;
    long long N = (long long)H * (long long)W;
    if (idx >= N) return;
    int l = lab[idx];
    if (l <= 0) return;
    int x = (int)(idx % (long long)W);
    int y = (int)(idx / (long long)W);
    atomicAdd(&area[l], 1u);
    atomicMin(&min_x[l], x);
    atomicMin(&min_y[l], y);
    atomicMax(&max_x[l], x);
    atomicMax(&max_y[l], y);
    atomicAdd(&sum_x[l], (double)x);
    atomicAdd(&sum_y[l], (double)y);
}
"""

_cc_stats_kernel = None


def _get_cc_stats_kernel():
    global _cc_stats_kernel
    if _cc_stats_kernel is None:
        _cc_stats_kernel = cp.RawKernel(_CC_STATS_KERNEL_SRC, "cc_stats_2d")
    return _cc_stats_kernel


# ---------------------------------------------------------------------------
# Core: CuPy in, CuPy out, no device context
# ---------------------------------------------------------------------------
def _connected_components_stats_gpu_core(mask_gpu, *, connectivity=8):
    """Connected components stats — CuPy arrays in, CuPy arrays out.

    Caller must already be inside a ``cp.cuda.Device`` context.

    Returns (labels_gpu, stats_gpu, centroids_gpu, num_labels):
        labels_gpu:   (H, W) int32
        stats_gpu:    (num_labels, 5) int32 [min_x, min_y, w, h, area]
        centroids_gpu: (num_labels, 2) float64 [cx, cy]
        num_labels:   int (including background label 0)
    """
    if int(connectivity) == 4:
        structure = cp.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=cp.uint8)
    elif int(connectivity) == 8:
        structure = cp.ones((3, 3), dtype=cp.uint8)
    else:
        raise ValueError("connectivity must be 4 or 8")

    labels, num = cnd.label(mask_gpu, structure=structure)
    num = int(num)
    labels = labels.astype(cp.int32)
    H, W = labels.shape
    n_labels = num + 1  # includes background

    # Allocate small per-label arrays
    min_x = cp.full((n_labels,), W, dtype=cp.int32)
    min_y = cp.full((n_labels,), H, dtype=cp.int32)
    max_x = cp.full((n_labels,), -1, dtype=cp.int32)
    max_y = cp.full((n_labels,), -1, dtype=cp.int32)
    area = cp.zeros((n_labels,), dtype=cp.uint32)
    sum_x = cp.zeros((n_labels,), dtype=cp.float64)
    sum_y = cp.zeros((n_labels,), dtype=cp.float64)

    # Launch RawKernel — single pass over all pixels
    kernel = _get_cc_stats_kernel()
    total = H * W
    block = 256
    grid = (total + block - 1) // block
    kernel((grid,), (block,),
           (labels, np.int32(H), np.int32(W),
            min_x, min_y, max_x, max_y,
            area, sum_x, sum_y))

    # Compose stats: [min_x, min_y, w, h, area]
    area_i32 = area.astype(cp.int32)
    # Guard against empty labels (area==0): clamp width/height to 0
    w = cp.maximum(max_x - min_x + 1, 0)
    h = cp.maximum(max_y - min_y + 1, 0)
    empty = area_i32 == 0
    w = cp.where(empty, cp.int32(0), w)
    h = cp.where(empty, cp.int32(0), h)

    stats = cp.zeros((n_labels, 5), dtype=cp.int32)
    stats[:, 0] = min_x
    stats[:, 1] = min_y
    stats[:, 2] = w
    stats[:, 3] = h
    stats[:, 4] = area_i32

    # Compose centroids: [cx, cy]
    denom = cp.maximum(area.astype(cp.float64), 1.0)
    centroids = cp.zeros((n_labels, 2), dtype=cp.float64)
    centroids[:, 0] = sum_x / denom
    centroids[:, 1] = sum_y / denom

    # Background row (label 0)
    bg_area = int(H * W) - int(area[1:].sum())
    stats[0] = cp.array([0, 0, W, H, bg_area], dtype=cp.int32)
    centroids[0] = cp.array([W / 2.0, H / 2.0], dtype=cp.float64)

    return labels, stats, centroids, n_labels


# ---------------------------------------------------------------------------
# Public wrapper: NumPy in, NumPy out
# ---------------------------------------------------------------------------
def connected_components_stats_gpu(mask: np.ndarray, *, connectivity: int = 8, device: int = 0) -> ComponentStatsResult:
    _require_gpu_deps()
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError("connected_components_stats expects a 2D mask.")
    with gpu_device(device):
        gm = cp.asarray(m > 0)
        labels_gpu, stats_gpu, centroids_gpu, num_labels = (
            _connected_components_stats_gpu_core(gm, connectivity=connectivity)
        )
        return ComponentStatsResult(
            labels=cp.asnumpy(labels_gpu).astype(np.int32, copy=False),
            stats=cp.asnumpy(stats_gpu).astype(np.int32, copy=False),
            centroids=cp.asnumpy(centroids_gpu).astype(np.float64, copy=False),
            num_labels=num_labels,
            backend="gpu",
            meta={"connectivity": int(connectivity),
                  "implementation": "cupyx.scipy.ndimage.label+rawkernel",
                  "device": int(device)},
        )
