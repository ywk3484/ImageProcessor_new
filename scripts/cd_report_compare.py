"""Position-matched CD comparison statistics for folder reports.

Differences are candidate minus reference, in the supplied calibrated unit.
Per-image rows describe paired holes. Aggregate rows describe paired image
means, with one equal weight per image; they do not pool holes as repeats.
Agreement limits are descriptive bias +/- 1.96 sample standard deviations,
not confidence intervals or an automatic claim of equivalence.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

import numpy as np


DEFAULT_PAIRS = tuple(
    (reference, prefix + reference)
    for reference in ("half_area", "half_area_min_avg", "tail_area")
    for prefix in ("local_", "pixel_")
)
AXES = (("cd", "cd_equivalent_px"), ("x", "cd_x_px"), ("y", "cd_y_px"))


def _mean(values):
    return float(np.mean(values)) if len(values) else None


def _std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def _fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def match_positions(reference, candidate, max_distance=1.0):
    """Return reference indices, candidate indices and distances for mutual nearest matches.

    Positions are ``[x, y]`` in the same image coordinate system. Distances
    are pixels and the bound is inclusive. Nonfinite positions stay unmatched.
    Mutual matching prevents any hole from being used more than once; no
    registration or unrestricted nearest-neighbour association is performed.
    """
    from scipy.spatial import cKDTree

    if not np.isfinite(max_distance) or max_distance < 0:
        raise ValueError("max_distance must be finite and nonnegative")
    left, right = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1:] != (2,) or right.shape[1:] != (2,):
        raise ValueError("positions must have shape (N, 2) in [x, y] order")
    li = np.flatnonzero(np.isfinite(left).all(axis=1))
    ri = np.flatnonzero(np.isfinite(right).all(axis=1))
    if not len(li) or not len(ri):
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, float)
    distance, nearest = cKDTree(right[ri]).query(left[li])
    reverse = cKDTree(left[li]).query(right[ri])[1]
    keep = (distance <= max_distance) & (reverse[nearest] == np.arange(len(li)))
    return li[keep], ri[nearest[keep]], distance[keep]


def paired_statistics(reference, candidate):
    """Descriptive statistics of finite paired values; undefined values are None.

    ``n`` is the number of finite pairs. ``std_difference`` uses N - 1;
    correlation is undefined when either input is constant or N is below two.
    """
    a, b = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    if a.ndim != 1 or b.shape != a.shape:
        raise ValueError("paired values must be one-dimensional arrays of the same length")
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    difference = b - a
    bias, spread = _mean(difference), _std(difference)
    correlation = None
    if len(a) > 1 and np.ptp(a) > 0 and np.ptp(b) > 0:
        correlation = float(np.corrcoef(a, b)[0, 1])
    return {
        "n": int(len(a)), "mean_reference": _mean(a), "mean_candidate": _mean(b),
        "bias": bias, "median_bias": float(np.median(difference)) if len(a) else None,
        "mae": _mean(np.abs(difference)),
        "rmse": float(np.sqrt(np.mean(difference ** 2))) if len(a) else None,
        "std_difference": spread,
        "p95_abs_difference": float(np.percentile(np.abs(difference), 95)) if len(a) else None,
        "correlation": correlation,
        "loa_low": bias - 1.96 * spread if spread is not None else None,
        "loa_high": bias + 1.96 * spread if spread is not None else None,
    }


def _numeric(result):
    if isinstance(result, Mapping):
        data = result
        meta = result.get("meta", {})
    else:
        data = {"seeds_xy": result.seeds_xy, "pixel_size": result.pixel_size, "unit": result.unit}
        meta = result.meta
        for name, measured in result.measurements.items():
            data[f"{name}_valid"] = measured.valid
            for _, metric in AXES:
                if metric in measured.metrics:
                    data[f"{name}_{metric}"] = measured.metrics[metric]
    seeds = np.asarray(data["seeds_xy"], dtype=float)
    if seeds.ndim != 2 or seeds.shape[1:] != (2,):
        raise ValueError("seeds_xy must have shape (N, 2)")
    scale = float(data["pixel_size"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("pixel_size must be positive and finite")
    return data, seeds, scale, str(data["unit"]), meta


def _method_values(data, name, metric, count, scale):
    values = np.asarray(data[f"{name}_{metric}"], dtype=float)
    valid = np.asarray(data[f"{name}_valid"], dtype=bool)
    if values.shape != (count,) or valid.shape != (count,):
        raise ValueError(f"{name} values and validity must align with seeds_xy")
    values = values * scale
    return values, valid & np.isfinite(values)


def _seconds(meta):
    value = meta.get("measure_s", meta.get("seconds", {}).get("total"))
    return float(value) if value is not None and np.isfinite(value) else None


def compare_image(reference, candidate, *, image_id, pairs=DEFAULT_PAIRS, max_distance=1.0,
                  reference_meta=None, candidate_meta=None):
    """Compare two independently measured results from the same image.

    Inputs may be CDResult/CDSummary objects or ``cd_compare.numeric_result``
    dictionaries. Only method pairs and axes present in both inputs are used.
    ``paired_coverage_reference/candidate`` divides the paired count by all
    finite valid holes of that method, including unmatched holes. Metadata
    overrides may supply measured wall time under ``measure_s``.
    """
    a, left, scale_a, unit_a, meta_a = _numeric(reference)
    b, right, scale_b, unit_b, meta_b = _numeric(candidate)
    if unit_a != unit_b:
        raise ValueError("comparison results must use the same unit")
    meta_a = {**meta_a, **(reference_meta or {})}
    meta_b = {**meta_b, **(candidate_meta or {})}
    ia, ib, distances = match_positions(left, right, max_distance)
    pairs = list(pairs.items()) if isinstance(pairs, Mapping) else list(pairs)
    if len(set(pairs)) != len(pairs):
        raise ValueError("method comparison pairs must be unique")
    rows = []
    for old, new in pairs:
        if f"{old}_valid" not in a or f"{new}_valid" not in b:
            continue
        different_window = old == "tail_area" and new in ("local_tail_area", "pixel_tail_area")
        notes = []
        if old in ("half_area", "half_area_min_avg", "tail_area") and new.startswith(("local_", "pixel_")):
            notes.append("model-shared wide window vs raw Voronoi cell" if different_window
                         else "deblended cell vs raw Voronoi cell")
        if new.startswith("pixel_") and not old.startswith("pixel_"):
            notes.append("native pixels vs 4x cubic grid")
        for axis, metric in AXES:
            if f"{old}_{metric}" not in a or f"{new}_{metric}" not in b:
                continue
            av, valid_a = _method_values(a, old, metric, len(left), scale_a)
            bv, valid_b = _method_values(b, new, metric, len(right), scale_b)
            va, vb = valid_a[ia], valid_b[ib]
            both = va & vb
            stats = paired_statistics(av[ia[both]], bv[ib[both]])
            mean_all_a, mean_all_b = _mean(av[valid_a]), _mean(bv[valid_b])
            convergence = meta_a.get("deblend_converged")
            iterations = meta_a.get("deblend_iterations_used")
            rows.append({
                "image_id": str(image_id), "method_reference": old, "method_candidate": new,
                "axis": axis, "unit": unit_a, "population": "paired_holes", **stats,
                "reference_holes": int(len(left)), "candidate_holes": int(len(right)),
                "matched_positions": int(len(ia)), "unmatched_reference": int(len(left) - len(ia)),
                "unmatched_candidate": int(len(right) - len(ib)), "max_distance_px": float(max_distance),
                "max_matched_distance_px": float(distances.max()) if len(distances) else None,
                "valid_reference": int(valid_a.sum()), "valid_candidate": int(valid_b.sum()),
                "paired_coverage_reference": _fraction(stats["n"], valid_a.sum()),
                "paired_coverage_candidate": _fraction(stats["n"], valid_b.sum()),
                "only_reference_valid": int((va & ~vb).sum()),
                "only_candidate_valid": int((vb & ~va).sum()), "both_invalid": int((~va & ~vb).sum()),
                "mean_all_reference": mean_all_a, "mean_all_candidate": mean_all_b,
                "unpaired_mean_difference": (mean_all_b - mean_all_a
                                             if mean_all_a is not None and mean_all_b is not None else None),
                "different_window": different_window, "definition_note": "; ".join(notes),
                "reference_converged": None if convergence is None else bool(convergence),
                "reference_iterations": None if iterations is None else int(iterations),
                "reference_measure_s": _seconds(meta_a), "candidate_measure_s": _seconds(meta_b),
            })
    return rows


def aggregate_comparisons(rows):
    """Summarize paired image means with equal image weights, grouped by methods, axis and unit.

    The common paired-statistic keys now describe image means (``n`` images),
    not individual holes. Reference/candidate standard deviations and 3 sigma
    use those same images. ``pooled_hole_*`` fields are explicitly descriptive
    hole-weighted differences; they are never used as image-repeat statistics.
    Percentiles and medians of all holes cannot be recovered from image rows
    and are deliberately not approximated by averaging image percentiles.
    """
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = tuple(row[name] for name in ("method_reference", "method_candidate", "axis", "unit"))
        unique = (key, row["image_id"])
        if unique in seen:
            raise ValueError("each image must occur once per method pair, axis and unit")
        seen.add(unique)
        groups[key].append(row)
    output = []
    for key, group in groups.items():
        paired = [r for r in group if r["n"] > 0]
        a = np.asarray([r["mean_reference"] for r in paired], dtype=float)
        b = np.asarray([r["mean_candidate"] for r in paired], dtype=float)
        stats = paired_statistics(a, b)
        count = sum(r["n"] for r in paired)
        pooled_bias = sum(r["n"] * r["bias"] for r in paired) / count if count else None
        pooled_mae = sum(r["n"] * r["mae"] for r in paired) / count if count else None
        pooled_rmse = (float(np.sqrt(sum(r["n"] * r["rmse"] ** 2 for r in paired) / count))
                       if count else None)
        pooled_std = None
        if count > 1:
            squared = sum((r["n"] - 1) * (r["std_difference"] or 0.0) ** 2
                          + r["n"] * (r["bias"] - pooled_bias) ** 2 for r in paired)
            pooled_std = float(np.sqrt(squared / (count - 1)))
        row = {
            **dict(zip(("method_reference", "method_candidate", "axis", "unit"), key)),
            "population": "image_means", "images_total": len(group), **stats,
            "paired_holes": count,
            "matched_positions": sum(r["matched_positions"] for r in group),
            "valid_reference": sum(r["valid_reference"] for r in group),
            "valid_candidate": sum(r["valid_candidate"] for r in group),
            "paired_coverage_reference": _fraction(count, sum(r["valid_reference"] for r in group)),
            "paired_coverage_candidate": _fraction(count, sum(r["valid_candidate"] for r in group)),
            "pooled_hole_bias": pooled_bias, "pooled_hole_mae": pooled_mae,
            "pooled_hole_rmse": pooled_rmse, "pooled_hole_std_difference": pooled_std,
            "different_window": any(r["different_window"] for r in group),
            "definition_note": "; ".join(dict.fromkeys(r["definition_note"] for r in group
                                                       if r["definition_note"])),
            "reference_unconverged_images": sum(r["reference_converged"] is False for r in group),
            "reference_convergence_known_images": sum(r["reference_converged"] is not None for r in group),
        }
        for name, values in (("reference", a), ("candidate", b)):
            spread, mean = _std(values), _mean(values)
            row[f"std_{name}"] = spread
            row[f"three_sigma_{name}"] = 3 * spread if spread is not None else None
            row[f"three_sigma_pct_{name}"] = (300 * spread / mean
                                             if spread is not None and mean is not None and mean != 0 else None)
        output.append(row)
    return output
