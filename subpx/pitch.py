
"""Pitch estimation utilities.

The public API groups lattice-pitch tasks under short task-based names rather than
historical implementation names.
"""

from __future__ import annotations

from typing import Optional, Dict, List
import numpy as np

from .types import PitchResult


def _assign_lines_1d(values: np.ndarray, tol: float):
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    if n == 0:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.int32),
        )

    order = np.argsort(v)
    v_sorted = v[order]
    labels_sorted = np.empty(n, dtype=np.int32)
    line_id = 0
    labels_sorted[0] = line_id

    run_sum = float(v_sorted[0])
    run_cnt = 1
    run_mean = run_sum / run_cnt

    for i in range(1, n):
        if abs(float(v_sorted[i]) - run_mean) <= float(tol):
            labels_sorted[i] = line_id
            run_sum += float(v_sorted[i])
            run_cnt += 1
            run_mean = run_sum / run_cnt
        else:
            line_id += 1
            labels_sorted[i] = line_id
            run_sum = float(v_sorted[i])
            run_cnt = 1
            run_mean = run_sum / run_cnt

    labels = np.empty(n, dtype=np.int32)
    labels[order] = labels_sorted

    L = line_id + 1
    line_sums = np.zeros((L,), dtype=np.float64)
    line_counts = np.zeros((L,), dtype=np.int32)
    np.add.at(line_sums, labels, v)
    np.add.at(line_counts, labels, 1)
    line_centers = line_sums / np.maximum(line_counts, 1)
    return labels, line_centers, line_counts


def _robust_pitch_from_sorted_coords(sorted_coord: np.ndarray):
    s = np.asarray(sorted_coord, dtype=np.float64)
    if s.size < 2:
        return np.nan, np.array([], dtype=np.float64)

    diffs = np.diff(s)
    if diffs.size == 0:
        return np.nan, diffs

    med = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - med)))
    if mad < 1e-12:
        keep = np.ones_like(diffs, dtype=bool)
    else:
        keep = np.abs(diffs - med) <= 3.0 * mad

    pitch = float(np.median(diffs[keep])) if np.any(keep) else med
    return pitch, diffs


def estimate_pitch(
    centers_xy: np.ndarray,
    *,
    mode: str = "global",
    method: str = "knn",
    k: int = 8,
    theta_deg: float = 15.0,
    min_sep: float = 0.5,
    max_sep: Optional[float] = None,
    refine: bool = True,
) -> dict:
    """Estimate lattice pitch.

    mode='global' returns x/y pitch estimates using nearest-neighbor vectors.
    Future line-wise or model-based methods can live behind the same API.
    """
    if mode != "global":
        raise ValueError("Currently supported modes: 'global'")
    if method != "knn":
        raise ValueError("Currently supported methods: 'knn'")

    pts = np.asarray(centers_xy, dtype=np.float64)
    n = pts.shape[0]
    if n < 2:
        base = {
            "pitch_x": np.nan,
            "pitch_y": np.nan,
            "dx_candidates": np.array([], dtype=np.float64),
            "dy_candidates": np.array([], dtype=np.float64),
        }
        return {"initial": base, "refined": None}

    kk = int(min(max(1, k), n - 1))
    dxs: List[float] = []
    dys: List[float] = []

    for i in range(n):
        d2 = np.sum((pts - pts[i]) ** 2, axis=1)
        nn = np.argsort(d2)[1:kk + 1]
        for j in nn:
            dxs.append(float(pts[j, 0] - pts[i, 0]))
            dys.append(float(pts[j, 1] - pts[i, 1]))

    dx = np.asarray(dxs, dtype=np.float64)
    dy = np.asarray(dys, dtype=np.float64)
    adx = np.abs(dx)
    ady = np.abs(dy)
    dist = np.hypot(adx, ady)

    ang = np.degrees(np.arctan2(dy, dx))
    ang = np.abs(ang)
    ang = np.minimum(ang, 180.0 - ang)

    ok = np.isfinite(dist) & (dist >= float(min_sep))
    if max_sep is not None:
        ok &= dist <= float(max_sep)

    x_like = ok & (ang <= float(theta_deg))
    y_like = ok & (ang >= (90.0 - float(theta_deg)))

    dx_cand = adx[x_like]
    dy_cand = ady[y_like]

    initial = {
        "pitch_x": float(np.median(dx_cand)) if dx_cand.size else np.nan,
        "pitch_y": float(np.median(dy_cand)) if dy_cand.size else np.nan,
        "dx_candidates": dx_cand,
        "dy_candidates": dy_cand,
    }

    refined = refine_pitch_by_index(
        pts,
        initial["pitch_x"],
        initial["pitch_y"],
    ) if refine else None

    return {"initial": initial, "refined": refined}


def refine_pitch_by_index(
    centers_xy: np.ndarray,
    pitch_x: float,
    pitch_y: float,
    *,
    max_iter: int = 2,
    trim_frac: float = 0.35,
) -> Dict[str, object]:
    pts = np.asarray(centers_xy, dtype=np.float64)
    if pts.shape[0] < 3 or not np.isfinite(pitch_x) or not np.isfinite(pitch_y):
        return {"pitch_x": float(pitch_x), "pitch_y": float(pitch_y), "kept_count": int(pts.shape[0])}

    x = pts[:, 0].copy()
    y = pts[:, 1].copy()
    x0 = float(np.percentile(x, 5))
    y0 = float(np.percentile(y, 5))
    px, py = float(pitch_x), float(pitch_y)
    keep = np.ones(len(pts), dtype=bool)

    for _ in range(int(max_iter)):
        if np.count_nonzero(keep) < 3:
            break

        ix = np.rint((x[keep] - x0) / px).astype(np.int32)
        iy = np.rint((y[keep] - y0) / py).astype(np.int32)

        A = np.column_stack([ix.astype(np.float64), np.ones_like(ix, dtype=np.float64)])
        ax, bx = np.linalg.lstsq(A, x[keep], rcond=None)[0]
        B = np.column_stack([iy.astype(np.float64), np.ones_like(iy, dtype=np.float64)])
        ay, by = np.linalg.lstsq(B, y[keep], rcond=None)[0]

        rx = x[keep] - (ax * ix + bx)
        ry = y[keep] - (ay * iy + by)
        thrx = abs(ax) * float(trim_frac)
        thry = abs(ay) * float(trim_frac)

        new_keep_local = (np.abs(rx) <= thrx) & (np.abs(ry) <= thry)
        full_keep = np.zeros_like(keep)
        full_keep[np.where(keep)[0]] = new_keep_local
        keep = full_keep
        px, py = float(ax), float(ay)

    return {"pitch_x": px, "pitch_y": py, "kept_count": int(np.count_nonzero(keep))}


def estimate_pitch_lines(
    centers_xy: np.ndarray,
    *,
    line_axis: str = "row",
    tol: float = 1.5,
    min_points_per_line: int = 3,
) -> PitchResult:
    pts = np.asarray(centers_xy, dtype=np.float64)
    if pts.size == 0:
        return PitchResult(
            values=np.zeros((0,), dtype=np.float64),
            axis=line_axis,
            method="per-line",
            backend="cpu",
            meta={
                "line_centers_perp": np.zeros((0,), dtype=np.float64),
                "line_counts": np.zeros((0,), dtype=np.int32),
                "line_ids_per_point": np.zeros((0,), dtype=np.int32),
            },
        )

    if line_axis == "row":
        perp = pts[:, 1]
        along = pts[:, 0]
    elif line_axis == "col":
        perp = pts[:, 0]
        along = pts[:, 1]
    else:
        raise ValueError("line_axis must be 'row' or 'col'.")

    labels, line_centers, line_counts = _assign_lines_1d(perp, tol=float(tol))
    L = line_centers.size
    line_pitch = np.full((L,), np.nan, dtype=np.float64)

    for lid in range(L):
        idx = np.where(labels == lid)[0]
        if idx.size < int(min_points_per_line):
            continue
        s = np.sort(along[idx])
        p, _ = _robust_pitch_from_sorted_coords(s)
        line_pitch[lid] = p

    order = np.argsort(line_centers)
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.size)
    labels_sorted = inv_order[labels]

    return PitchResult(
        values=line_pitch[order],
        axis=line_axis,
        method="per-line",
        backend="cpu",
        meta={
            "line_centers_perp": line_centers[order],
            "line_counts": line_counts[order],
            "line_ids_per_point": labels_sorted.astype(np.int32),
        },
    )


def pitch_residuals(
    centers_xy: np.ndarray,
    *,
    pitch_x: float,
    x_origin: float | None = None,
) -> dict:
    """Return a simple residual-vs-x cloud using nearest index assignment along x.

    This is a lightweight public helper for notebook analysis. More advanced row-wise
    or distortion-model workflows should be added later without changing this surface.
    """
    pts = np.asarray(centers_xy, dtype=np.float64)
    if pts.size == 0:
        return {"x": np.zeros((0,), dtype=np.float64), "residual": np.zeros((0,), dtype=np.float64)}
    if not np.isfinite(pitch_x) or pitch_x <= 0:
        raise ValueError("pitch_x must be finite and > 0")

    x = pts[:, 0]
    x0 = float(np.min(x)) if x_origin is None else float(x_origin)
    ix = np.rint((x - x0) / float(pitch_x))
    fit = x0 + ix * float(pitch_x)
    return {"x": x, "residual": x - fit, "index": ix.astype(np.int32)}
