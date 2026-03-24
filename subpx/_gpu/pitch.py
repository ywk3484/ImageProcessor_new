"""GPU-accelerated pitch estimation via histogram-based row binning."""
from __future__ import annotations

import numpy as np

from ..types import PitchResult
from ..backends import gpu_device

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None


def _require_cupy():
    if cp is None:
        raise RuntimeError("CuPy is required for GPU pitch estimation.")


def estimate_pitch_lines_gpu(
    centers_xy: np.ndarray,
    *,
    line_axis: str = "row",
    tol: float = 1.5,
    min_points_per_line: int = 3,
    device: int = 0,
    pitch_var_max: float = 0.25,
) -> PitchResult:
    """GPU estimate_pitch_lines via histogram row binning.

    Parameters
    ----------
    centers_xy : (N, 2) array, [x, y] order
    line_axis : "row" (cluster by y) or "col" (cluster by x)
    tol : tolerance for row assignment (px)
    min_points_per_line : minimum points for a valid row
    device : CUDA device index
    pitch_var_max : max coefficient of variation to accept a row
    """
    _require_cupy()
    pts = np.asarray(centers_xy, dtype=np.float64)
    if pts.size == 0:
        return PitchResult(
            values=np.zeros((0,), dtype=np.float64),
            axis=line_axis, method="per-line", backend="gpu",
            meta={
                "line_centers_perp": np.zeros((0,), dtype=np.float64),
                "line_counts": np.zeros((0,), dtype=np.int32),
                "line_ids_per_point": np.zeros((0,), dtype=np.int32),
                "valid_mask": np.zeros((0,), dtype=bool),
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

    with gpu_device(device):
        perp_gpu = cp.asarray(perp)
        along_gpu = cp.asarray(along)

        # --- Step 1: histogram row detection ---
        row_centers_gpu = _find_row_centers(perp_gpu, tol, min_points_per_line)

        if row_centers_gpu.size == 0:
            return PitchResult(
                values=np.zeros((0,), dtype=np.float64),
                axis=line_axis, method="per-line", backend="gpu",
                meta={
                    "line_centers_perp": np.zeros((0,), dtype=np.float64),
                    "line_counts": np.zeros((0,), dtype=np.int32),
                    "line_ids_per_point": -np.ones(len(pts), dtype=np.int32),
                    "valid_mask": np.zeros((0,), dtype=bool),
                },
            )

        # --- Step 2: vectorized row assignment ---
        labels_gpu, counts_gpu = _assign_to_rows(
            perp_gpu, row_centers_gpu, tol
        )

        # --- Step 3: per-row pitch via padded matrix ---
        L = int(row_centers_gpu.size)
        pitches_gpu, valid_gpu = _compute_row_pitches(
            along_gpu, labels_gpu, L,
            min_points_per_line, pitch_var_max,
        )

        # --- Transfer back to NumPy ---
        pitches = cp.asnumpy(pitches_gpu)
        valid_mask = cp.asnumpy(valid_gpu)
        row_centers = cp.asnumpy(row_centers_gpu)
        counts = cp.asnumpy(counts_gpu)
        labels = cp.asnumpy(labels_gpu)

    # Sort rows by perpendicular position
    order = np.argsort(row_centers)
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.size)
    labels_sorted = np.where(labels >= 0, inv_order[labels], -1)

    return PitchResult(
        values=pitches[order],
        axis=line_axis, method="per-line", backend="gpu",
        meta={
            "line_centers_perp": row_centers[order],
            "line_counts": counts[order],
            "line_ids_per_point": labels_sorted.astype(np.int32),
            "valid_mask": valid_mask[order],
        },
    )


# ---------------------------------------------------------------------------
# Internal helpers — all operate on CuPy arrays within gpu_device context
# ---------------------------------------------------------------------------

def _find_row_centers(perp_gpu, tol, min_count):
    """Histogram-based row center detection on GPU.

    Returns (L,) CuPy array of row center positions.
    """
    lo = float(cp.min(perp_gpu)) - tol
    hi = float(cp.max(perp_gpu)) + tol
    n_bins = max(1, int(np.ceil((hi - lo) / tol)))
    hist, edges = cp.histogram(perp_gpu, bins=n_bins, range=(lo, hi))
    bin_centers = (edges[:-1] + edges[1:]) * 0.5

    # Peak detection: bins above min_count that are local maxima
    padded = cp.pad(hist, 1, constant_values=0)
    is_peak = (
        (hist >= min_count)
        & (hist >= padded[:-2])
        & (hist >= padded[2:])
    )
    peak_centers = bin_centers[is_peak]

    if peak_centers.size == 0:
        return peak_centers

    # Merge adjacent peaks within tol
    peak_centers = _merge_adjacent_peaks(peak_centers, tol)
    return peak_centers


def _merge_adjacent_peaks(centers, tol):
    """Merge peaks that are within tol of each other.

    Uses sorted diff + cumsum to identify groups, then averages within groups.
    """
    if centers.size <= 1:
        return centers
    sorted_c = cp.sort(centers)
    gaps = cp.diff(sorted_c) > tol
    group_ids = cp.concatenate([
        cp.zeros(1, dtype=cp.int32),
        cp.cumsum(gaps).astype(cp.int32),
    ])

    n_groups = int(group_ids[-1]) + 1
    merged = cp.zeros(n_groups, dtype=cp.float64)
    counts = cp.zeros(n_groups, dtype=cp.int32)
    cp.add.at(merged, group_ids, sorted_c)
    cp.add.at(counts, group_ids, 1)
    return merged / cp.maximum(counts, 1).astype(cp.float64)


def _assign_to_rows(perp_gpu, row_centers_gpu, tol):
    """Assign each point to nearest row within tol.

    Returns:
        labels: (N,) int32, row index per point (-1 if unassigned)
        counts: (L,) int32, points per row
    """
    N = perp_gpu.size
    L = row_centers_gpu.size
    # Distance matrix (N, L)
    dist = cp.abs(perp_gpu[:, None] - row_centers_gpu[None, :])
    nearest = cp.argmin(dist, axis=1)
    nearest_dist = dist[cp.arange(N), nearest]
    labels = cp.where(nearest_dist <= tol, nearest, cp.int32(-1))
    labels = labels.astype(cp.int32)

    counts = cp.zeros(L, dtype=cp.int32)
    valid_mask = labels >= 0
    cp.add.at(counts, labels[valid_mask], 1)
    return labels, counts


def _compute_row_pitches(along_gpu, labels_gpu, L, min_points, pitch_var_max):
    """Compute per-row pitch using vectorized padded matrix approach.

    No Python for-loops — all operations are bulk CuPy.

    Returns:
        pitches: (L,) float64, per-row pitch (NaN for invalid rows)
        valid: (L,) bool, True if row passed both filters
    """
    assigned = labels_gpu >= 0
    counts = cp.zeros(L, dtype=cp.int32)
    cp.add.at(counts, labels_gpu[assigned], 1)
    max_per_row = int(cp.max(counts)) if L > 0 else 0

    if max_per_row < 2:
        return (
            cp.full(L, cp.nan, dtype=cp.float64),
            cp.zeros(L, dtype=bool),
        )

    # --- Vectorized scatter into padded (L, max_per_row) matrix ---
    valid_points = cp.where(assigned)[0]
    point_labels = labels_gpu[valid_points]
    point_along = along_gpu[valid_points]

    # Sort by (label, along_value) using composite key
    # This groups points by row and sorts within each row simultaneously
    label_offset = (cp.max(cp.abs(point_along)) + 1.0) if point_along.size > 0 else 1.0
    sort_keys = point_labels.astype(cp.float64) * label_offset + point_along
    sort_order = cp.argsort(sort_keys)
    sorted_labels = point_labels[sort_order]
    sorted_along = point_along[sort_order]

    # Compute within-row index for each point
    # group_starts[r] = index in sorted array where row r begins
    group_starts = cp.searchsorted(sorted_labels.astype(cp.float64), cp.arange(L, dtype=cp.float64))
    within_row_idx = cp.arange(sorted_labels.size, dtype=cp.int32) - group_starts[sorted_labels].astype(cp.int32)

    # Scatter into padded matrix (already sorted within rows)
    mat = cp.full((L, max_per_row), cp.nan, dtype=cp.float64)
    mat[sorted_labels, within_row_idx] = sorted_along

    # --- Vectorized diff and pitch computation ---
    diffs = mat[:, 1:] - mat[:, :-1]  # (L, max_per_row - 1)
    diff_valid = cp.isfinite(diffs)    # mask NaN padding

    # Per-row median: sort diffs (NaN goes to end), take middle valid element
    # Replace invalid diffs with +inf so they sort to end
    diffs_for_sort = cp.where(diff_valid, diffs, cp.inf)
    diffs_sorted = cp.sort(diffs_for_sort, axis=1)  # (L, max_per_row - 1)

    # Number of valid diffs per row
    n_valid = diff_valid.sum(axis=1).astype(cp.int32)  # (L,)

    # Median = middle element of sorted valid diffs
    mid_idx = n_valid // 2  # integer division for median index
    row_indices = cp.arange(L)
    # For even n_valid, take average of two middle elements
    even_mask = (n_valid > 0) & (n_valid % 2 == 0)
    odd_mask = (n_valid > 0) & (n_valid % 2 == 1)

    pitches = cp.full(L, cp.nan, dtype=cp.float64)
    if cp.any(odd_mask):
        pitches[odd_mask] = diffs_sorted[row_indices[odd_mask], mid_idx[odd_mask]]
    if cp.any(even_mask):
        mid_lo = mid_idx[even_mask] - 1
        mid_hi = mid_idx[even_mask]
        pitches[even_mask] = (
            diffs_sorted[row_indices[even_mask], mid_lo]
            + diffs_sorted[row_indices[even_mask], mid_hi]
        ) * 0.5

    # --- Row validity filters ---
    # Filter 1: minimum point count
    enough_points = counts >= min_points

    # Filter 2: pitch consistency (coefficient of variation)
    # Variance = mean of squared deviations from median
    dev = cp.where(diff_valid, diffs - pitches[:, None], 0.0)
    dev_sq = cp.where(diff_valid, dev * dev, 0.0)
    n_valid_f = cp.maximum(n_valid.astype(cp.float64), 1.0)
    variance = dev_sq.sum(axis=1) / n_valid_f
    std = cp.sqrt(variance)
    consistent = (n_valid < 2) | (pitches <= 1e-12) | (std / cp.maximum(cp.abs(pitches), 1e-12) <= pitch_var_max)

    row_valid = enough_points & consistent & (n_valid >= 1) & cp.isfinite(pitches)
    pitches = cp.where(row_valid, pitches, cp.nan)

    return pitches, row_valid
