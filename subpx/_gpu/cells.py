"""GPU Voronoi cell extraction for the deblending CD path."""

from __future__ import annotations

import numpy as np

from ._cuda import cp, require_cupy
from .voronoi import _labels_device


def voronoi_cells(rows, shape: tuple[int, int], *, pad: int = 3):
    """Labels, cell masks and origins of ``_extract_voronoi_rois`` without its background fill.

    ``rows`` are ``(x, y, w, h, cx, cy)``. Masks and origins equal the CPU's
    exactly (same box arithmetic, integer comparisons); the filled ROI stack
    is not built — deblending reads the image itself.
    """
    require_cupy()
    height, width = shape
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, 6)
    labels = _labels_device(rows[:, 4:6], shape)
    n = len(rows)
    if n == 0:
        return cp.asnumpy(labels), np.zeros((0, 0, 0), dtype=bool), np.zeros((0, 2), dtype=np.int32)
    x0 = np.maximum(0, rows[:, 0].astype(np.int32) - pad)
    y0 = np.maximum(0, rows[:, 1].astype(np.int32) - pad)
    x1 = np.minimum(width, (rows[:, 0] + rows[:, 2]).astype(np.int32) + pad)
    y1 = np.minimum(height, (rows[:, 1] + rows[:, 3]).astype(np.int32) + pad)
    hs, ws = y1 - y0, x1 - x0
    hm, wm = int(hs.max()), int(ws.max())
    r, c = cp.arange(hm), cp.arange(wm)
    y0d, x0d = cp.asarray(y0)[:, None], cp.asarray(x0)[:, None]
    inside = (r[None] < cp.asarray(hs)[:, None])[:, :, None] & (c[None] < cp.asarray(ws)[:, None])[:, None, :]
    ys = cp.clip(y0d + r[None], 0, height - 1)[:, :, None]
    xs = cp.clip(x0d + c[None], 0, width - 1)[:, None, :]
    masks = inside & (labels[ys, xs] == cp.arange(n, dtype=cp.int32)[:, None, None])
    origins = np.stack((x0, y0), axis=1).astype(np.int32)
    return cp.asnumpy(labels), cp.asnumpy(masks), origins
