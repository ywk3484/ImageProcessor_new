"""CD-center lattice indexing, averaged pitch profiles, and registration reports."""

from __future__ import annotations

import numpy as np

from .types import CDResult, CDSummary, PatternPitchResult, PitchProfile


def _positive_pair(value, name):
    pair = np.asarray(value, dtype=float)
    if pair.shape != (2,) or not np.isfinite(pair).all() or np.any(pair <= 0):
        raise ValueError(f"{name} must contain two finite positive values in [x, y] order")
    return pair


def _lattice_frame(points, hint, layout):
    """Estimate local row direction and spacing without pooling absolute y.

    Use a bounded, deterministic sample for the initial neighbour search. The
    full graph below is searched in pitch-normalized coordinates, so unequal
    x/y pitches do not hide one direction from its nearest-neighbour list.
    """
    from scipy.spatial import cKDTree

    sample = np.linspace(0, len(points) - 1, min(len(points), 12000), dtype=int)
    search = points if hint is None else points / hint
    neighbors = cKDTree(search).query(search[sample], k=min(25, len(points)))[1][:, 1:]
    vectors = points[neighbors] - points[sample, None]
    dx, dy = vectors[..., 0], vectors[..., 1]
    separation = float(np.median(np.min(np.hypot(dx, dy), axis=1)))
    if separation <= 0:
        raise ValueError("Duplicate seeds cannot be indexed as distinct holes")
    slopes = np.divide(dy, dx, out=np.full_like(dy, np.nan), where=np.abs(dx) > 0.3 * separation)
    candidates = slopes[np.isfinite(slopes) & (np.abs(slopes) < 0.35)]
    if not len(candidates):
        raise ValueError("Cannot identify the row direction from the detected centers")
    histogram, bins = np.histogram(candidates, bins=np.linspace(-0.35, 0.35, 141))
    smooth = np.convolve(histogram, np.ones(5), mode="same")
    centers = (bins[:-1] + bins[1:]) / 2
    peaks = ((smooth >= np.r_[0, smooth[:-1]]) & (smooth >= np.r_[smooth[1:], 0])
             & (smooth >= 0.25 * smooth.max()))
    # On an anisotropic grid a diagonal can have more neighbours than a row.
    # Among supported direction modes choose the one closest to the image axis.
    choices = np.flatnonzero(peaks)
    mode = centers[choices[np.argmin(np.abs(centers[choices]))]]
    slope = float(np.median(candidates[np.abs(candidates - mode) < 0.04]))
    perpendicular = dy - slope * dx

    def first_spacing(values, mask):
        nearest = np.min(np.where(mask, np.abs(values), np.inf), axis=1)
        nearest = nearest[np.isfinite(nearest)]
        if not len(nearest):
            raise ValueError("Cannot identify both lattice spacings from the detected centers")
        first = float(np.median(nearest))
        candidates = np.abs(values[mask])
        first_group = candidates[(candidates > 0.65 * first) & (candidates < 1.4 * first)]
        return float(np.median(first_group))

    within = float(hint[0]) if hint is not None else first_spacing(
        dx, (np.abs(perpendicular) < 0.25 * separation) & (np.abs(dx) > 0.3 * separation))
    across = float(hint[1]) if hint is not None else first_spacing(
        perpendicular, np.abs(perpendicular) > 0.4 * separation)
    row_steps = np.rint(perpendicular / across)
    cross = ((np.abs(row_steps) == 1) & (np.abs(perpendicular / across - row_steps) < 0.25))
    if not np.any(cross):
        # A hint can preserve missing complete rows; use any available row step
        # for shear, while layout inference still requires an odd step.
        cross = (np.abs(row_steps) > 0) & (np.abs(perpendicular / across - row_steps) < 0.25)
    odd = cross & (np.abs(row_steps) % 2 == 1)
    # Remove the transverse x displacement expected from the measured row
    # angle before deciding whether alternate rows have a half-pitch offset.
    # Even a small rotation can otherwise resemble staggering on a tall cell.
    orthogonal_shear = -slope * across / (1 + slope * slope)

    def residual(shift):
        return ((dx - orthogonal_shear * row_steps) / within - shift * row_steps + 0.5) % 1 - 0.5

    if layout == "auto":
        layout = "staggered" if np.any(odd) and (
            np.median(np.abs(residual(0.5)[odd])) + 0.1 < np.median(np.abs(residual(0)[odd]))) else "square"
    shift = 0.5 if layout == "staggered" else 0.0
    shear = orthogonal_shear
    if np.any(cross):
        shear += float(np.median(residual(shift)[cross] / row_steps[cross])) * within
    v = points[:, 1] - slope * points[:, 0]
    u = points[:, 0] - shear * v / across
    return np.column_stack((u / within - shift * v / across, v / across)), within, across, layout


def _local_indices(frame, within, across, tolerance):
    """Propagate integer lattice steps along a reliable neighbour forest.

    Accumulating short displacements lets rows bow or tilt across an image
    without splitting them into separate rows. An integer label is checked
    against redundant edges, rather than fitted to one global row origin.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import breadth_first_order, connected_components, minimum_spanning_tree
    from scipy.spatial import cKDTree

    n = len(frame)
    # Frame coordinates are [Bravais column, row]; an alternating half-pitch
    # lattice is a regular integer lattice in this frame.
    neighbors = cKDTree(frame).query(frame, k=min(13, n))[1][:, 1:]
    source = np.broadcast_to(np.arange(n)[:, None], neighbors.shape).ravel()
    target = neighbors.ravel()
    delta = frame[target] - frame[source]
    steps = np.rint(delta).astype(np.int64)
    error = delta - steps
    # A physical tolerance is applied to local displacement, not total row span.
    residual = np.max(np.abs(error) * [within, across], axis=1)
    keep = (source < target) & np.any(steps != 0, axis=1) & (residual <= tolerance)
    source, target, steps, residual = source[keep], target[keep], steps[keep], residual[keep]
    if not len(source):
        raise ValueError("Detected centers do not contain consistent lattice neighbours")
    weights = residual + 1e-6 * np.linalg.norm(steps, axis=1) + 1e-12
    graph = csr_matrix((weights, (source, target)), shape=(n, n))
    graph = graph + graph.T
    count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    main = int(np.argmax(sizes))
    if sizes[main] < 4:
        raise ValueError("Need at least four connected lattice centers")
    tree = minimum_spanning_tree(graph)
    tree = tree + tree.T
    indexed = np.zeros(n, dtype=bool)
    indices = np.zeros((n, 2), dtype=np.int64)
    # The largest connected patch determines the phase of any disconnected
    # patches. A lone off-lattice detection is retained but left unindexed.
    grouped = np.argsort(labels, kind="stable")
    groups = np.split(grouped, np.flatnonzero(np.diff(labels[grouped])) + 1)
    components = [main] + [i for i in range(count) if i != main]
    phase = None
    for component in components:
        members = groups[component]
        root = int(members[0])
        if component != main and len(members) == 1:
            continue
        order, parents = breadth_first_order(tree, root, directed=False)
        indices[root] = 0 if phase is None else np.rint(frame[root] - phase).astype(np.int64)
        for child in order[1:]:
            parent = parents[child]
            indices[child] = indices[parent] + np.rint(frame[child] - frame[parent]).astype(np.int64)
        indexed[order] = True
        if phase is None:
            phase = np.median(frame[order] - indices[order], axis=0)
    consistent = (indices[target] - indices[source] == steps).all(axis=1)
    if np.mean(consistent) < 0.98:
        raise ValueError("Detected centers have inconsistent local lattice neighbours; check segmentation or layout")
    return indices, indexed


def _index_pattern(seeds, layout, stagger_axis, line_tolerance, pitch_hint_xy):
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
    if line_tolerance is not None and (not np.isfinite(line_tolerance) or line_tolerance <= 0):
        raise ValueError("line_tolerance must be finite and positive (pixels)")
    frame, within, across, layout = _lattice_frame(q, hint, layout)
    if line_tolerance is None:
        line_tolerance = 0.35 * min(within, across)
    bravais, indexed = _local_indices(frame, within, across, line_tolerance)
    primary = bravais[:, 1] - bravais[indexed, 1].min()
    # Convert the Bravais basis (half-step every row) to alternating logical
    # rows. Normalize the first populated row so shuffled inputs keep parity.
    secondary = bravais[:, 0] + ((primary // 2) if layout == "staggered" else 0)
    secondary -= secondary[indexed].min()
    indices = np.column_stack((primary, secondary))
    indices[~indexed] = -1
    if stagger_axis == "column":
        indices = indices[:, ::-1]
    if len(np.unique(indices[indexed], axis=0)) != np.count_nonzero(indexed):
        raise ValueError("Ambiguous lattice: check layout, stagger_axis, line_tolerance, or pitch_hint_xy")
    if np.any(indices[indexed].max(axis=0) < 1):
        raise ValueError("Need at least two populated rows and columns")
    pitch_xy = np.array([within, across])[np.argsort(axes)]
    if np.any(indices.max(axis=0) > 4 * len(pts)):
        raise ValueError("Implausible lattice extent; check pitch_hint_xy")
    return indices.astype(np.int64), pitch_xy, layout, float(line_tolerance)


def _neighbor_pairs(indices, axis):
    """Only immediate lattice neighbours; a missing hole never creates 2x pitch."""
    lookup = {tuple(rc): i for i, rc in enumerate(indices) if rc[0] >= 0 and rc[1] >= 0}
    dim = 1 if axis == "x" else 0
    pairs = []
    for i, rc in enumerate(indices):
        if rc[0] < 0 or rc[1] < 0:
            continue
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
        Rectangular grid (square) or alternating half-pitch offset. Indexing
        follows local neighbours, allowing modest tilt and smooth distortion.
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
        Tolerance for local neighbour displacements in pixels. Default: 35% of
        the smaller estimated pitch. This does not limit a row's total drift.
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
    indexed = (indices >= 0).all(axis=1)
    available = np.flatnonzero(indexed)
    if reference_xy is None:
        reference = int(available[np.lexsort((indices[indexed, 1], indices[indexed, 0]))[0]])
    else:
        point = np.asarray(reference_xy, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("reference_xy must contain two finite pixel coordinates")
        reference = int(available[np.argmin(np.sum((result.seeds_xy[indexed] - point)**2, axis=1))])
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
        valid = valid & np.isfinite(centers).all(axis=1) & indexed
        profiles = {axis: _profile(centers, valid, result.seeds_xy, indices, pairs[axis], axis, band,
                                   reference, result.pixel_size, None if nominal is None else nominal[i])
                    for i, (axis, band) in enumerate((("x", row_band), ("y", col_band)))}
        found[name] = PatternPitchResult(
            name, centers.copy(), valid.copy(), indices.copy(), layout, stagger_axis,
            pitch_xy.copy(), reference, profiles, result.pixel_size, result.unit,
            dict(average_rows=len(row_band), average_columns=len(col_band),
                 row_band=row_band.tolist(), column_band=col_band.tolist(), line_tolerance=tolerance,
                 nominal_source="selected_pair_mean" if nominal is None else "user",
                 reference_valid=bool(valid[reference]), unindexed_seeds=int(np.count_nonzero(~indexed))),
        )
    return found
