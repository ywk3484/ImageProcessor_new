"""CD-center lattice indexing, averaged pitch profiles, and registration reports."""

from __future__ import annotations

import numpy as np

from .types import CDResult, CDSummary, PatternPitchResult, PitchProfile


def _positive_pair(value, name):
    pair = np.asarray(value, dtype=float)
    if pair.shape != (2,) or not np.isfinite(pair).all() or np.any(pair <= 0):
        raise ValueError(f"{name} must contain two finite positive values in [x, y] order")
    return pair


def _index_pattern(seeds, layout, stagger_axis, line_tolerance, pitch_hint_xy):
    from .pitch import _assign_lines_1d

    pts = np.asarray(seeds, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 4 or not np.isfinite(pts).all():
        raise ValueError("At least four finite seeds in an (N, 2) array are required")
    if layout not in {"auto", "square", "staggered"}:
        raise ValueError("layout must be 'auto', 'square', or 'staggered'")
    if stagger_axis not in {"row", "column"}:
        raise ValueError("stagger_axis must be 'row' or 'column'")
    # Work in (within-line, across-line) coordinates; transpose column staggering.
    axes = [0, 1] if stagger_axis == "row" else [1, 0]
    q = pts[:, axes]
    hint = None if pitch_hint_xy is None else _positive_pair(pitch_hint_xy, "pitch_hint_xy")[axes]
    if line_tolerance is None:
        if hint is None:
            from scipy.spatial import cKDTree
            nearest = cKDTree(q).query(q, k=2)[0][:, 1]
            scale = float(np.median(nearest))
        else:
            scale = float(np.min(hint))
        line_tolerance = 0.2 * scale
    if not np.isfinite(line_tolerance) or line_tolerance <= 0:
        raise ValueError("line_tolerance must be finite and positive (pixels)")
    lines, locations, counts = _assign_lines_1d(q[:, 1], line_tolerance)
    if len(locations) < 2 or np.count_nonzero(counts >= 2) < 2:
        raise ValueError("Need at least two populated rows and columns; check line_tolerance or image orientation")
    across = float(hint[1]) if hint is not None else float(np.median(np.diff(locations)))
    line_numbers = np.rint((locations - locations[0]) / across).astype(int)
    if np.any(np.diff(line_numbers) <= 0):
        raise ValueError("Rows overlap in lattice indexing; adjust line_tolerance or pitch_hint_xy")
    # Sort once, rather than scanning the whole array for each row.
    order = np.lexsort((q[:, 0], lines))
    a, b = order[:-1], order[1:]
    steps = q[b, 0] - q[a, 0]
    within = float(hint[0]) if hint is not None else float(np.median(steps[lines[a] == lines[b]]))
    if min(within, across) <= 0:
        raise ValueError("Duplicate seeds or nonpositive lattice pitch")
    primary = line_numbers[lines]
    phase = 2 * np.pi / within

    def fit_offset(staggered):
        corrected = q[:, 0] - (primary % 2) * within * (0.5 if staggered else 0)
        origin = np.angle(np.mean(np.exp(1j * phase * corrected))) / phase
        residual = (corrected - origin + within / 2) % within - within / 2
        # A median alone misses staggering when an odd number of rows leaves
        # a majority at one phase. Include the minority phase in model choice.
        return origin, float(np.percentile(np.abs(residual), 90)), corrected

    plain, staggered = fit_offset(False), fit_offset(True)
    if layout == "auto":
        layout = "staggered" if staggered[1] + 0.1 * within < plain[1] else "square"
    origin, residual, corrected = staggered if layout == "staggered" else plain
    secondary = np.rint((corrected - origin) / within).astype(int)
    secondary -= secondary.min()
    indices = np.column_stack((primary, secondary))
    if stagger_axis == "column":
        indices = indices[:, ::-1]
    # Reject ambiguous grids instead of silently assigning arbitrary neighbours.
    if residual > 0.2 * within or len(np.unique(indices, axis=0)) != len(pts):
        raise ValueError("Ambiguous lattice: check layout, stagger_axis, line_tolerance, or pitch_hint_xy")
    pitch_xy = np.array([within, across])[np.argsort(axes)]
    if np.any(indices.max(axis=0) > 4 * len(pts)):
        raise ValueError("Implausible lattice extent; check pitch_hint_xy")
    return indices.astype(np.int64), pitch_xy, layout, float(line_tolerance)


def _neighbor_pairs(indices, axis):
    """Only immediate lattice neighbours; a missing hole never creates 2x pitch."""
    lookup = {tuple(rc): i for i, rc in enumerate(indices)}
    dim = 1 if axis == "x" else 0
    pairs = []
    for i, rc in enumerate(indices):
        neighbor = rc.copy()
        neighbor[dim] += 1
        j = lookup.get(tuple(neighbor))
        if j is not None:
            pairs.append((i, j))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _band(count, reference, size, name):
    if count is None:
        count = size
    if isinstance(count, (bool, np.bool_)) or not np.isfinite(count) or int(count) != count or not 1 <= count <= size:
        raise ValueError(f"{name} must be an integer between 1 and {size}, or None for all")
    start = min(max(reference - (int(count) - 1) // 2, 0), size - int(count))
    return np.arange(start, start + int(count), dtype=np.int64)


def _integrate(values, reference):
    out = np.full(len(values) + 1, np.nan)
    out[reference] = 0.0
    for i in range(reference, len(values)):
        out[i + 1] = out[i] + values[i]
    for i in range(reference - 1, -1, -1):
        out[i] = out[i + 1] - values[i]
    return out


def _profile(centers, valid, seeds, indices, pairs, axis, band, reference, scale, nominal):
    coordinate, dim = (0, 1) if axis == "x" else (1, 0)
    other = 1 - dim
    n = int(indices[:, dim].max()) + 1
    selected = np.flatnonzero(np.isin(indices[:, other], band))
    use = (np.isin(indices[pairs[:, 0], other], band)
           & valid[pairs[:, 0]] & valid[pairs[:, 1]])
    pairs = pairs[use]
    bins = indices[pairs[:, 0], dim]
    values = (centers[pairs[:, 1], coordinate] - centers[pairs[:, 0], coordinate]) * scale
    counts = np.bincount(bins, minlength=n - 1)
    total = np.bincount(bins, weights=values, minlength=n - 1)
    pitch = np.divide(total, counts, out=np.full(n - 1, np.nan), where=counts > 0)
    deviations = values - pitch[bins]
    squared = np.bincount(bins, weights=deviations**2, minlength=n - 1)
    std = np.sqrt(np.divide(squared, counts - 1, out=np.full(n - 1, np.nan), where=counts > 1))
    # The abscissa is the actual mean measured coordinate of the selected holes.
    position = np.full(n, np.nan)
    for source, rows in ((seeds, selected), (centers, selected[valid[selected]])):
        labels = indices[rows, dim]
        number = np.bincount(labels, minlength=n)
        sums = np.bincount(labels, weights=source[rows, coordinate] * scale, minlength=n)
        np.divide(sums, number, out=position, where=number > 0)
    nominal = float(np.mean(values)) if nominal is None and len(values) else nominal
    nominal = float("nan") if nominal is None else float(nominal)
    ref = int(indices[reference, dim])
    return PitchProfile(
        axis=axis, indices=np.arange(n), position=position, pitch=pitch, std=std, counts=counts,
        distance=_integrate(pitch, ref), registration=_integrate(pitch - nominal, ref),
        selected_indices=selected, pairs=pairs, nominal_pitch=nominal, reference_node=ref,
    )


def analyze_pitch(
    result: CDResult | CDSummary,
    *,
    layout: str = "auto",
    stagger_axis: str = "row",
    average_rows: int | None = None,
    average_columns: int | None = None,
    reference_xy=None,
    nominal_pitch_xy=None,
    line_tolerance: float | None = None,
    pitch_hint_xy=None,
) -> dict[str, PatternPitchResult]:
    """Measure adjacent-hole x/y pitch using each CD method's own centers.

    Parameters
    ----------
    result : CDResult or CDSummary
        Output from ``measure_cds``. Seed positions define a shared lattice;
        only finite centers with ``valid=True`` contribute to pitch.
    layout : {'auto', 'square', 'staggered'}
        Axis-aligned rectangular grid (square) or alternating half-pitch offset.
    stagger_axis : {'row', 'column'}
        Rows shifted in x, or columns shifted in y. In either case y pitch is
        between adjacent logical rows, x pitch between adjacent logical columns;
        these are projected separations, not diagonal neighbour distances.
    average_rows, average_columns : int, optional
        Contiguous logical rows averaged for the x profile, and columns averaged
        for the y profile. Bands surround the reference and clamp at the edges.
        None selects the full extent. Counts give actual valid pairs per interval.
    reference_xy : (2,) array-like, optional
        Image point in pixels; nearest seed is the reference. Default: first
        lattice row/column. Integration is zero at its row/column.
    nominal_pitch_xy : (2,) array-like, optional
        Nominal x/y pitch in calibrated ``result.unit``. Default: each selected
        profile's mean pair pitch. Registration = sum(pitch - nominal), with
        signed discrete sums on either side of the reference; distance = sum(pitch).
    line_tolerance : float, optional
        Row/column grouping tolerance in pixels. Default: 20% of typical nearest
        seed spacing. Suitable for nearly axis-aligned grids with small drift.
    pitch_hint_xy : (2,) array-like, optional
        Approximate lattice pitch in pixels, useful with missing rows/columns
        or strongly unequal x/y pitch. Not a registration baseline.

    Returns
    -------
    dict of str to PatternPitchResult
        One result per CD method. Missing intervals are NaN and break cumulative
        curves; missing holes never become a doubled pitch or a fitted center.
    """
    indices, pitch_xy, layout, tolerance = _index_pattern(
        result.seeds_xy, layout, stagger_axis, line_tolerance, pitch_hint_xy)
    if not np.isfinite(result.pixel_size) or result.pixel_size <= 0:
        raise ValueError("pixel_size must be finite and positive")
    nominal = None if nominal_pitch_xy is None else _positive_pair(nominal_pitch_xy, "nominal_pitch_xy")
    if reference_xy is None:
        reference = int(np.lexsort((indices[:, 1], indices[:, 0]))[0])
    else:
        point = np.asarray(reference_xy, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("reference_xy must contain two finite pixel coordinates")
        reference = int(np.argmin(np.sum((result.seeds_xy - point)**2, axis=1)))
    rows, columns = indices.max(axis=0) + 1
    row_band = _band(average_rows, indices[reference, 0], rows, "average_rows")
    col_band = _band(average_columns, indices[reference, 1], columns, "average_columns")
    pairs = {axis: _neighbor_pairs(indices, axis) for axis in ("x", "y")}
    found = {}
    for name, measurement in result.measurements.items():
        centers = np.asarray(measurement.centers_xy, dtype=float)
        valid = np.asarray(measurement.valid, dtype=bool)
        if centers.shape != (len(indices), 2) or valid.shape != (len(indices),):
            raise ValueError(f"{name}: centers and validity must match the seed list")
        valid = valid & np.isfinite(centers).all(axis=1)
        profiles = {axis: _profile(centers, valid, result.seeds_xy, indices, pairs[axis], axis, band,
                                   reference, result.pixel_size, None if nominal is None else nominal[i])
                    for i, (axis, band) in enumerate((("x", row_band), ("y", col_band)))}
        found[name] = PatternPitchResult(
            name, centers.copy(), valid.copy(), indices.copy(), layout, stagger_axis,
            pitch_xy.copy(), reference, profiles, result.pixel_size, result.unit,
            dict(average_rows=len(row_band), average_columns=len(col_band),
                 row_band=row_band.tolist(), column_band=col_band.tolist(), line_tolerance=tolerance,
                 nominal_source="selected_pair_mean" if nominal is None else "user",
                 reference_valid=bool(valid[reference])),
        )
    return found
