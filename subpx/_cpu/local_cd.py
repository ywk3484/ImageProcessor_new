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
                  outlines=True, metrics_only=False, store_cells=False, xp=np, _fused=True):
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

    if _fused and xp is not np and metrics_only and not store_cells and "local_projection" not in methods:
        from .._gpu.local_cd import measure_local_areas
        started = time.perf_counter()
        signal, labels = xp.asarray(signal), xp.asarray(labels)
        fallback_elapsed = 0.0
        fallback_cells = np.zeros(count, bool)
        groups = None
        for factor in (4, 1):
            names = [name for name in methods if name.startswith("pixel_") == (factor == 1)]
            if not names:
                continue
            canonical = [name.replace("pixel_", "local_", 1) if factor == 1 else name for name in names]
            options = dict(methods=canonical, factor=factor, n_angles=n_angles, half_area_level=half_area_level,
                           half_area_peak_pixels=half_area_peak_pixels, tail_area_level=tail_area_level)
            parts = measure_local_areas(signal, labels, origins, sizes, report, **options)
            if parts is None:
                # A shared-memory outlier must not send every ordinary cell
                # through the slower reference path. Retry homogeneous buckets;
                # only a bucket the kernel declines uses bounded extraction.
                if groups is None:
                    buckets = (sizes + 3) // 4 * 4
                    width_key = int(buckets[:, 1].max()) + 1
                    _, groups = np.unique(buckets[:, 0] * width_key + buckets[:, 1], return_inverse=True)
                for group in range(int(groups.max()) + 1):
                    indices = np.flatnonzero(groups == group)
                    parts = measure_local_areas(signal, labels, origins[indices], sizes[indices],
                                                report[indices], **options)
                    if parts is None:
                        fallback_started = time.perf_counter()
                        parts, _, _, _, info = measure_local(
                            signal, labels, rows, report[indices], methods=names, pad=pad, n_angles=n_angles,
                            half_area_level=half_area_level, half_area_peak_pixels=half_area_peak_pixels,
                            tail_area_level=tail_area_level, outlines=False, metrics_only=True, store_cells=False,
                            xp=xp, _fused=False,
                        )
                        fallback_elapsed += time.perf_counter() - fallback_started
                        fallback_cells[indices] = True
                        for key, value in info["seconds"].items():
                            seconds[key] = seconds.get(key, 0.0) + value
                        source_names = names
                    else:
                        source_names = canonical
                    for name, source in zip(names, source_names):
                        part = parts[source]
                        part.method = name
                        _scatter(found, name, part, indices, count)
            else:
                for name, source in zip(names, canonical):
                    part = parts[source]
                    part.method = name
                    found[name] = part
        seconds["local_fused"] = time.perf_counter() - started - fallback_elapsed
        return {name: found[name] for name in methods}, origins, rois, masks, {
            "seconds": seconds, "cell_stack_shape": [int(max_h), int(max_w)],
            "cell_storage": "fused+bucketed" if fallback_cells.any() else "fused",
            "local_fallback_cells": int(fallback_cells.sum()),
        }

    started = time.perf_counter()
    image_d, labels_d = xp.asarray(signal), xp.asarray(labels)
    # Four-pixel buckets avoid one tiny variation causing a separate GPU launch.
    bucket_sizes = ((sizes + 3) // 4) * 4
    shapes, groups = np.unique(bucket_sizes, axis=0, return_inverse=True)
    specs_by_factor = {4: [], 1: []}
    for name in methods:
        if name == "local_projection":
            continue
        factor = 1 if name.startswith("pixel_") else 4
        specs_by_factor[factor].append(dict(
            name=name, level=tail_area_level if name.endswith("tail_area") else half_area_level,
            background="min" if name.endswith("half_area_min_avg") else "median",
            peak_pixels=None if name.endswith("half_area") else half_area_peak_pixels, cuts=None))
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
            for factor, specs in specs_by_factor.items():
                if not specs:
                    continue
                parts = _half_areas(cells, mask_host, origins[indices], n_angles, specs, factor=factor,
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
