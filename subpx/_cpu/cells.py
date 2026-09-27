"""Shared Voronoi cell extraction and a CPU nearest-seed implementation."""

from __future__ import annotations

import numpy as np


def compute_voronoi_labels_cpu(seeds_xy: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Assign pixels to their nearest seed, resolving exact ties by seed ID."""
    from scipy.spatial import cKDTree

    seeds = np.asarray(seeds_xy, dtype=np.float64)
    labels = np.full(shape, -1, dtype=np.int32)
    if len(seeds) == 0:
        return labels
    if len(seeds) == 1:
        labels.fill(0)
        return labels
    tree = cKDTree(seeds)
    # Bound query memory independently of the image height and seed count.
    chunk_rows = max(1, 131072 // shape[1])
    for y0 in range(0, shape[0], chunk_rows):
        y1 = min(shape[0], y0 + chunk_rows)
        yy, xx = np.mgrid[y0:y1, :shape[1]]
        points = np.column_stack((xx.ravel(), yy.ravel()))
        distances, ids = tree.query(points, k=2)
        chosen = ids[:, 0].copy()
        for j in np.flatnonzero(distances[:, 0] == distances[:, 1]):
            candidates = tree.query_ball_point(points[j], np.nextafter(distances[j, 0], np.inf))
            d2 = np.sum((seeds[candidates] - points[j]) ** 2, axis=1)
            chosen[j] = np.min(np.asarray(candidates)[d2 == d2.min()])
        labels[y0:y1] = chosen.reshape(y1 - y0, shape[1])
    return labels


def _extract_voronoi_rois(
    gray: np.ndarray,
    voronoi_labels: np.ndarray,
    rows: list,
    pad: int = 3,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Extract padded cells using the existing diagnostic's background fill.

    Rows contain ``(x, y, w, h, cx, cy)``. Pixels outside each Voronoi cell
    are filled with the tenth percentile of its border pixels. The returned
    mask distinguishes measured pixels from this fill and stack padding.
    """
    H, W = gray.shape
    N = len(rows)
    if N == 0:
        return np.zeros((0, 0, 0), dtype=np.float64), np.zeros((0, 0, 0), dtype=bool), []

    rows_arr = np.asarray(rows, dtype=np.float64)
    x0 = np.maximum(0, rows_arr[:, 0].astype(np.int32) - pad)
    y0 = np.maximum(0, rows_arr[:, 1].astype(np.int32) - pad)
    x1 = np.minimum(W, (rows_arr[:, 0] + rows_arr[:, 2]).astype(np.int32) + pad)
    y1 = np.minimum(H, (rows_arr[:, 1] + rows_arr[:, 3]).astype(np.int32) + pad)
    hs = y1 - y0
    ws = x1 - x0
    Hm = int(hs.max())
    Wm = int(ws.max())
    origins = list(zip(x0.tolist(), y0.tolist()))
    rois_stack = np.zeros((N, Hm, Wm), dtype=np.float64)
    masks_stack = np.zeros((N, Hm, Wm), dtype=bool)

    for j in range(N):
        _x0, _y0 = int(x0[j]), int(y0[j])
        _x1, _y1 = int(x1[j]), int(y1[j])
        h_r, w_r = int(hs[j]), int(ws[j])
        roi = gray[_y0:_y1, _x0:_x1].astype(np.float64)
        vmask = voronoi_labels[_y0:_y1, _x0:_x1] == j
        inner = vmask.copy()
        inner[0, :] = inner[-1, :] = inner[:, 0] = inner[:, -1] = False
        inner[1:, :] &= vmask[:-1, :]
        inner[:-1, :] &= vmask[1:, :]
        inner[:, 1:] &= vmask[:, :-1]
        inner[:, :-1] &= vmask[:, 1:]
        border_vals = roi[vmask & ~inner]
        bg = float(np.percentile(border_vals, 10)) if border_vals.size > 0 else 0.0
        rois_stack[j, :, :] = bg
        rois_stack[j, :h_r, :w_r] = np.where(vmask, roi, bg)
        masks_stack[j, :h_r, :w_r] = vmask

    return rois_stack, masks_stack, origins
