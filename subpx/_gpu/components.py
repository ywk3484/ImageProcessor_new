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


def connected_components_stats_gpu(mask: np.ndarray, *, connectivity: int = 8, device: int = 0) -> ComponentStatsResult:
    _require_gpu_deps()
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError("connected_components_stats expects a 2D mask.")
    with gpu_device(device):
        gm = cp.asarray(m > 0)
        if int(connectivity) == 4:
            structure = cp.asarray([[0,1,0],[1,1,1],[0,1,0]], dtype=cp.uint8)
        elif int(connectivity) == 8:
            structure = cp.ones((3,3), dtype=cp.uint8)
        else:
            raise ValueError("connectivity must be 4 or 8")
        labels, num = cnd.label(gm, structure=structure)
        num = int(num)
        H, W = labels.shape

        flat = labels.ravel().astype(cp.int32)
        yy = cp.repeat(cp.arange(H, dtype=cp.int32), W)
        xx = cp.tile(cp.arange(W, dtype=cp.int32), H)
        ones = cp.ones_like(flat, dtype=cp.int32)

        area = cp.bincount(flat, weights=ones, minlength=num + 1).astype(cp.int32)
        sum_x = cp.bincount(flat, weights=xx, minlength=num + 1).astype(cp.float64)
        sum_y = cp.bincount(flat, weights=yy, minlength=num + 1).astype(cp.float64)

        min_x = cp.full((num + 1,), W, dtype=cp.int32)
        min_y = cp.full((num + 1,), H, dtype=cp.int32)
        max_x = cp.full((num + 1,), -1, dtype=cp.int32)
        max_y = cp.full((num + 1,), -1, dtype=cp.int32)
        cp.minimum.at(min_x, flat, xx)
        cp.minimum.at(min_y, flat, yy)
        cp.maximum.at(max_x, flat, xx)
        cp.maximum.at(max_y, flat, yy)

        stats = cp.zeros((num + 1, 5), dtype=cp.int32)
        stats[:, 0] = min_x
        stats[:, 1] = min_y
        stats[:, 2] = cp.maximum(max_x - min_x + 1, 0)
        stats[:, 3] = cp.maximum(max_y - min_y + 1, 0)
        stats[:, 4] = area

        centroids = cp.zeros((num + 1, 2), dtype=cp.float64)
        denom = cp.maximum(area.astype(cp.float64), 1.0)
        centroids[:, 0] = sum_x / denom
        centroids[:, 1] = sum_y / denom

        if num >= 0:
            stats[0] = cp.array([0, 0, W, H, area[0]], dtype=cp.int32)
            centroids[0] = cp.array([W / 2.0, H / 2.0], dtype=cp.float64)

        return ComponentStatsResult(
            labels=cp.asnumpy(labels).astype(np.int32, copy=False),
            stats=cp.asnumpy(stats).astype(np.int32, copy=False),
            centroids=cp.asnumpy(centroids).astype(np.float64, copy=False),
            num_labels=num + 1,
            backend="gpu",
            meta={"connectivity": int(connectivity), "implementation": "cupyx.scipy.ndimage.label+bincount", "device": int(device)},
        )
