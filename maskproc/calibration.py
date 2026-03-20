"""Row/column grouping and residual-cloud helpers for calibration workflows."""

from __future__ import annotations

import numpy as np


def split_clusters_1d(values, *, threshold: float):
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return np.zeros((0,), dtype=np.int64), []
    order = np.argsort(v)
    sv = v[order]
    gaps = np.diff(sv)
    split_after = np.where(gaps > float(threshold))[0] + 1
    split_idx = split_after.astype(np.int64)
    bounds = []
    start = 0
    for stop in list(split_after) + [sv.size]:
        bounds.append((int(start), int(stop)))
        start = int(stop)
    return split_idx, bounds


def assign_clusters_1d(values, *, threshold: float):
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float64), []
    order = np.argsort(v)
    sv = v[order]
    split_idx, bounds = split_clusters_1d(sv, threshold=threshold)
    labels_sorted = np.zeros((sv.size,), dtype=np.int32)
    centers = []
    for lid, (s, e) in enumerate(bounds):
        labels_sorted[s:e] = lid
        centers.append(float(np.mean(sv[s:e])))
    labels = np.empty_like(labels_sorted)
    labels[order] = labels_sorted
    return labels, np.asarray(centers, dtype=np.float64), bounds


def find_y_cluster_split_indices(y, threshold: float = 10.0):
    """Compatibility helper preserving the old notebook workflow.

    Returns split indices and sorted-array bounds.
    """
    return split_clusters_1d(np.asarray(y, dtype=np.float64), threshold=threshold)


def residuals_vs_x_by_line(
    centers_xy,
    *,
    line_axis: str = "row",
    line_threshold: float = 10.0,
    pitch: float | None = None,
):
    pts = np.asarray(centers_xy, dtype=np.float64)
    if pts.size == 0:
        return {
            "x": np.zeros((0,), dtype=np.float64),
            "residual": np.zeros((0,), dtype=np.float64),
            "line_id": np.zeros((0,), dtype=np.int32),
            "line_coord": np.zeros((0,), dtype=np.float64),
            "pitch": np.nan if pitch is None else float(pitch),
        }

    if line_axis == "row":
        perp = pts[:, 1]
        along = pts[:, 0]
    elif line_axis == "col":
        perp = pts[:, 0]
        along = pts[:, 1]
    else:
        raise ValueError("line_axis must be 'row' or 'col'.")

    line_id, line_coord, _ = assign_clusters_1d(perp, threshold=float(line_threshold))
    if pitch is None:
        diffs_all = []
        for lid in range(line_coord.size):
            s = np.sort(along[line_id == lid])
            if s.size >= 2:
                diffs_all.append(np.diff(s))
        if diffs_all:
            pitch = float(np.median(np.concatenate(diffs_all)))
        else:
            pitch = np.nan

    residual_x = np.full(along.shape, np.nan, dtype=np.float64)
    for lid in range(line_coord.size):
        idx = np.where(line_id == lid)[0]
        if idx.size < 2 or not np.isfinite(pitch):
            continue
        x = along[idx]
        j = np.rint((x - np.min(x)) / float(pitch)).astype(np.int32)
        A = np.column_stack([j.astype(np.float64), np.ones_like(j, dtype=np.float64)])
        a, b = np.linalg.lstsq(A, x, rcond=None)[0]
        residual_x[idx] = x - (a * j + b)

    return {
        "x": along,
        "residual": residual_x,
        "line_id": line_id.astype(np.int32),
        "line_coord": line_coord,
        "pitch": float(pitch) if pitch is not None else np.nan,
    }


def line_offset_summary(residual_cloud):
    x = np.asarray(residual_cloud["x"], dtype=np.float64)
    r = np.asarray(residual_cloud["residual"], dtype=np.float64)
    lid = np.asarray(residual_cloud["line_id"], dtype=np.int32)
    valid = np.isfinite(x) & np.isfinite(r)
    x = x[valid]
    r = r[valid]
    lid = lid[valid]
    if x.size == 0:
        return {"line_id": np.zeros((0,), dtype=np.int32), "offset": np.zeros((0,), dtype=np.float64), "count": np.zeros((0,), dtype=np.int32)}
    uniq = np.unique(lid)
    offsets = np.zeros((uniq.size,), dtype=np.float64)
    counts = np.zeros((uniq.size,), dtype=np.int32)
    for i, u in enumerate(uniq):
        rv = r[lid == u]
        offsets[i] = float(np.median(rv)) if rv.size else np.nan
        counts[i] = int(rv.size)
    return {"line_id": uniq, "offset": offsets, "count": counts}


def _gaussian_kernel1d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    sigma = float(sigma)
    if sigma <= 0:
        return np.array([1.0], dtype=np.float64)
    radius = max(1, int(round(truncate * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= np.sum(k)
    return k


def _nan_gaussian_smooth_1d(values: np.ndarray, sigma: float, axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(v).astype(np.float64)
    filled = np.where(np.isfinite(v), v, 0.0)
    k = _gaussian_kernel1d(float(sigma))
    if k.size == 1:
        return np.where(valid > 0, filled, np.nan), valid
    def _conv1d(arr):
        full = np.convolve(arr, k, mode='full')
        start = (full.size - arr.size) // 2
        return full[start:start + arr.size]
    num = np.apply_along_axis(_conv1d, axis, filled * valid)
    den = np.apply_along_axis(_conv1d, axis, valid)
    out = np.where(den > 1e-12, num / den, np.nan)
    return out, den


def build_row_residual_map(
    centers_xy,
    *,
    line_axis: str = 'row',
    line_threshold: float = 10.0,
    pitch: float | None = None,
    bin_px: float = 32.0,
    min_count_per_bin: int = 1,
):
    cloud = residuals_vs_x_by_line(centers_xy, line_axis=line_axis, line_threshold=line_threshold, pitch=pitch)
    x = np.asarray(cloud['x'], dtype=np.float64)
    r = np.asarray(cloud['residual'], dtype=np.float64)
    line_id = np.asarray(cloud['line_id'], dtype=np.int32)
    line_coord = np.asarray(cloud['line_coord'], dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(r) & np.isfinite(line_id)
    x = x[valid]
    r = r[valid]
    line_id = line_id[valid]
    if x.size == 0:
        return {
            'residual_map': np.zeros((0,0), dtype=np.float64),
            'support_map': np.zeros((0,0), dtype=np.int32),
            'x_bin_centers': np.zeros((0,), dtype=np.float64),
            'line_coord': line_coord,
            'pitch': float(cloud['pitch']),
            'line_id_per_point': line_id,
            'x_per_point': x,
            'residual_per_point': r,
        }
    bin_px = float(bin_px)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    edges = np.arange(x_min, x_max + bin_px, bin_px, dtype=np.float64)
    if edges.size < 2:
        edges = np.array([x_min, x_min + bin_px], dtype=np.float64)
    x_bin = np.clip(np.digitize(x, edges) - 1, 0, edges.size - 2).astype(np.int32)
    R = int(np.max(line_id)) + 1
    K = edges.size - 1
    z = np.full((R, K), np.nan, dtype=np.float64)
    w = np.zeros((R, K), dtype=np.int32)
    for rid in range(R):
        mrow = line_id == rid
        if not np.any(mrow):
            continue
        for kb in np.unique(x_bin[mrow]):
            m = mrow & (x_bin == kb)
            if np.count_nonzero(m) >= int(min_count_per_bin):
                z[rid, kb] = float(np.median(r[m]))
                w[rid, kb] = int(np.count_nonzero(m))
    xc = 0.5 * (edges[:-1] + edges[1:])
    return {
        'residual_map': z,
        'support_map': w,
        'x_bin_centers': xc,
        'line_coord': line_coord,
        'pitch': float(cloud['pitch']),
        'line_id_per_point': line_id,
        'x_per_point': x,
        'residual_per_point': r,
    }


def smooth_row_residual_map(residual_map, support_map=None, *, sigma_x_bins: float = 3.0, sigma_row: float = 1.5, second_pass_x: bool = False):
    z = np.asarray(residual_map, dtype=np.float64)
    if z.size == 0:
        return {'smoothed_map': z.copy(), 'support_smoothed': np.zeros_like(z, dtype=np.float64)}
    if support_map is not None:
        z = np.where(np.asarray(support_map) > 0, z, np.nan)
    zx, sx = _nan_gaussian_smooth_1d(z, float(sigma_x_bins), axis=1)
    zxy, sxy = _nan_gaussian_smooth_1d(zx, float(sigma_row), axis=0)
    if second_pass_x:
        zxy, sxy = _nan_gaussian_smooth_1d(zxy, float(sigma_x_bins), axis=1)
    return {'smoothed_map': zxy, 'support_smoothed': sxy}


def interpolate_row_residual_field(model, x, line_id):
    xq = np.asarray(x, dtype=np.float64)
    rq = np.asarray(line_id, dtype=np.float64)
    z = np.asarray(model['smoothed_map'], dtype=np.float64)
    xc = np.asarray(model['x_bin_centers'], dtype=np.float64)
    if z.size == 0 or xc.size == 0:
        return np.full_like(xq, np.nan, dtype=np.float64)
    out = np.full(xq.shape, np.nan, dtype=np.float64)
    R, K = z.shape
    for i, (xx, rr) in enumerate(zip(xq, rq)):
        if not np.isfinite(xx) or not np.isfinite(rr):
            continue
        r0 = int(np.clip(np.floor(rr), 0, R - 1))
        r1 = int(np.clip(r0 + 1, 0, R - 1))
        fr = float(rr - r0) if r1 > r0 else 0.0
        v0 = np.interp(xx, xc, z[r0], left=np.nan, right=np.nan)
        v1 = np.interp(xx, xc, z[r1], left=np.nan, right=np.nan)
        if np.isfinite(v0) and np.isfinite(v1):
            out[i] = (1 - fr) * v0 + fr * v1
        elif np.isfinite(v0):
            out[i] = v0
        elif np.isfinite(v1):
            out[i] = v1
    return out


def fit_rowwise_distortion_field(
    centers_xy,
    *,
    line_axis: str = 'row',
    line_threshold: float = 10.0,
    pitch: float | None = None,
    bin_px: float = 32.0,
    sigma_x_bins: float = 3.0,
    sigma_row: float = 1.5,
    second_pass_x: bool = False,
):
    base = build_row_residual_map(centers_xy, line_axis=line_axis, line_threshold=line_threshold, pitch=pitch, bin_px=bin_px)
    smooth = smooth_row_residual_map(base['residual_map'], base['support_map'], sigma_x_bins=sigma_x_bins, sigma_row=sigma_row, second_pass_x=second_pass_x)
    out = dict(base)
    out.update(smooth)
    return out
