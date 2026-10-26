"""Experimental raw-cell CD measurements, with bounded CPU/GPU working storage."""

from __future__ import annotations

import time

import numpy as np

from ..types import CDMethodResult
from .invariant_cd import _half_areas, _projection, _to_numpy


def cell_boxes(rows, shape, pad):
    """The existing cell extractor's clipped origins and sizes, without a stack."""
    rows = np.asarray(rows, dtype=float).reshape(-1, 6)
    height, width = shape
    x0 = np.maximum(0, rows[:, 0].astype(np.int32) - pad)
    y0 = np.maximum(0, rows[:, 1].astype(np.int32) - pad)
    x1 = np.minimum(width, (rows[:, 0] + rows[:, 2]).astype(np.int32) + pad)
    y1 = np.minimum(height, (rows[:, 1] + rows[:, 3]).astype(np.int32) + pad)
    return np.column_stack((x0, y0)), np.column_stack((y1 - y0, x1 - x0))


def _scatter(found, name, part, indices, count):
    """Assemble numerical results; all arrays crossing this boundary are NumPy."""
    def allocate(value):
        fill = False if value.dtype.kind == "b" else "no_signal" if value.dtype.kind in "US" else np.nan
        return np.full((count, *value.shape[1:]), fill, dtype=value.dtype)

    if name not in found:
        found[name] = CDMethodResult(
            method=name, centers_xy=allocate(part.centers_xy), contours_xy=allocate(part.contours_xy),
            valid=allocate(part.valid), status=allocate(part.status),
            metrics={key: allocate(value) for key, value in part.metrics.items()},
            geometry={key: allocate(value) for key, value in part.geometry.items()},
        )
    result = found[name]
    for key in ("centers_xy", "contours_xy", "valid", "status"):
        getattr(result, key)[indices] = getattr(part, key)
    for key, value in part.metrics.items():
        result.metrics[key][indices] = value
    for key, value in part.geometry.items():
        result.geometry[key][indices] = value


def measure_local(signal, labels, rows, report, *, methods, pad, n_angles,
                  half_area_level=0.5, half_area_peak_pixels=4.0, tail_area_level=0.2,
                  outlines=True, metrics_only=False, store_cells=False, xp=np):
    """Measure raw Voronoi cells; no fitted neighbour, background map, or seed revision.

    Groups have bounded storage independent of the largest cell elsewhere.
    Local intensity scaling and constant offsets cancel for fixed cell masks;
    independent neighbours and changes in segmentation can still affect CDs.
    """
    from .invariant_cd import _empty

    origins, sizes = cell_boxes(rows, signal.shape, pad)
    report = np.asarray(report, dtype=np.int64)
    origins, sizes = origins[report], sizes[report]
    count = len(report)
    max_h, max_w = sizes.max(axis=0) if count else (0, 0)
    rois = np.zeros((count, int(max_h), int(max_w))) if store_cells else np.empty((count, 0, 0))
    masks = np.zeros(rois.shape, bool)
    found, seconds = {}, {}
    if not count:
        for name in methods:
            extra = () if name == "local_projection" else ("local_level", "threshold", "mean_signal")
            found[name] = _empty(name, 0, n_angles, extra, metrics_only=metrics_only)
        return found, origins, rois, masks, {"seconds": seconds, "cell_stack_shape": [0, 0]}

    started = time.perf_counter()
    image_d, labels_d = xp.asarray(signal), xp.asarray(labels)
    # Four-pixel buckets avoid one tiny variation causing a separate GPU launch.
    bucket_sizes = ((sizes + 3) // 4) * 4
    shapes, groups = np.unique(bucket_sizes, axis=0, return_inverse=True)
    specs = []
    for name in methods:
        if name == "local_projection":
            continue
        specs.append(dict(name=name, level=tail_area_level if name == "local_tail_area" else half_area_level,
                          background="min" if name == "local_half_area_min_avg" else "median",
                          peak_pixels=None if name == "local_half_area" else half_area_peak_pixels, cuts=None))
    for group, (hb, wb) in enumerate(shapes):
        members = np.flatnonzero(groups == group)
        # Account for the 4x enlargement, not just the raw cells (about 128 MiB).
        chunk = max(1, min(4096, 128 * 2**20 // (int(hb) * int(wb) * 8 * 16)))
        for start in range(0, len(members), chunk):
            indices = members[start:start + chunk]
            ox, oy = xp.asarray(origins[indices]).T
            hs, ws = xp.asarray(sizes[indices]).T
            yy = oy[:, None, None] + xp.arange(int(hb))[None, :, None]
            xx = ox[:, None, None] + xp.arange(int(wb))[None, None, :]
            within = ((xp.arange(int(hb))[None, :, None] < hs[:, None, None])
                      & (xp.arange(int(wb))[None, None, :] < ws[:, None, None]))
            ys, xs = xp.clip(yy, 0, signal.shape[0] - 1), xp.clip(xx, 0, signal.shape[1] - 1)
            mask = within & (labels_d[ys, xs] == xp.asarray(report[indices])[:, None, None])
            cells = xp.where(mask, image_d[ys, xs], 0.0)
            # The invariant kernels accept NumPy masks for their size bookkeeping.
            mask_host = _to_numpy(mask)
            if store_cells:
                hh, ww = min(int(hb), int(max_h)), min(int(wb), int(max_w))
                rois[indices, :hh, :ww] = _to_numpy(cells[:, :hh, :ww])
                masks[indices, :hh, :ww] = mask_host[:, :hh, :ww]
            seconds["local_extract"] = seconds.get("local_extract", 0.0) + time.perf_counter() - started
            if "local_projection" in methods:
                started = time.perf_counter()
                inner = mask.copy()
                inner[:, [0, -1], :] = False
                inner[:, :, [0, -1]] = False
                inner[:, 1:] &= mask[:, :-1]
                inner[:, :-1] &= mask[:, 1:]
                inner[:, :, 1:] &= mask[:, :, :-1]
                inner[:, :, :-1] &= mask[:, :, 1:]
                level = xp.nanmedian(xp.where(mask & ~inner, cells, xp.nan).reshape(len(indices), -1), axis=1)
                cleaned = xp.where(mask, cells - level[:, None, None], 0.0)
                part = _projection(cleaned, mask_host, origins[indices], n_angles, xp=xp, metrics_only=metrics_only)
                part.metrics["background"][:] = _to_numpy(level)
                _scatter(found, "local_projection", part, indices, count)
                seconds["local_projection"] = seconds.get("local_projection", 0.0) + time.perf_counter() - started
            if specs:
                parts = _half_areas(cells, mask_host, origins[indices], n_angles, specs,
                                    outlines=outlines, metrics_only=metrics_only, xp=xp, seconds=seconds)
                for name, part in parts.items():
                    part.metrics["background"][:] = part.metrics["local_level"]
                    _scatter(found, name, part, indices, count)
            started = time.perf_counter()
    if store_cells:
        shown = dict(cells=np.where(masks, rois, np.nan), inside=masks, origins=origins)
        for part in found.values():
            part.inputs.update(shown)
    return {name: found[name] for name in methods}, origins, rois, masks, {
        "seconds": seconds, "cell_stack_shape": [int(max_h), int(max_w)], "cell_storage": "bucketed",
        "local_groups": len(shapes),
    }
