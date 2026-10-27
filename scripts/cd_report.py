"""Measure every PNG in a folder with the CD methods and write an offline HTML report.

    python scripts/cd_report.py FOLDER --output REPORT --pixel-size 21 --unit nm

Methods: gaussian (deblended Gaussian FWHM), projection (x/y profile max-slope widths), half_area (area above
half the feature's own height over its local level = median of the cell edge; assumes no shape), half_area_min
(the same with the local level = minimum of the cell edge) and half_area_min_avg (half_area_min with the peak =
mean of the brightest pixels, --peak-pixels of them) and tail_area (the same area count with a low threshold
near the background, --tail-level of the height, default 0.2, measured on a wider window of the image with the
background and every other hole removed, so the region may extend past the hole's cell; local level = median of the
hole's own area farthest from every hole) and tail_area_norm (the same window divided by the pattern's averaged
peak, area where it exceeds --tail-level, no background step) and edge_area (the same window, threshold = its mean
brightness weighted by steepness: neither the peak nor a local level is used) and foot_area (the same window,
threshold = the brightness at the foot of the edge, where the slope bends into the flat surroundings: a tail
boundary, neither the peak nor a local level is used) and foot_area_ratio (that foot found once per image, as a ratio
to each hole's steepest-slope brightness). None needs a blur or nominal CD. The cell intensity of each hole is tail_area's mean light above the local
level over its region; it gets its own statistics, map, and correlation with every CD method. The other
subpx methods (logquad, gradient, halfmax, disk, flux_ratio) are left out: they assume a round or star-shaped
outline, or need the blur, and disk/flux_ratio dominated the run time. --backend auto (default) uses the GPU when
CuPy finds a CUDA device; the backend actually used, and why the GPU is not, is printed at the start.

Open REPORT/index.html in a browser (works from the folder, no server). It has a summary tab and one tab per image.

The summary tab first answers whether the images show the same CD on average. Per method, with m_i the mean CD
of image i (over the holes that method measured validly) and N the number of images:
    M = (m_1 + ... + m_N) / N
    s = sqrt( sum_i (m_i - M)^2 / (N - 1) )
    between-image 3 sigma % = 3 s / M x 100        (also given as 3 s in --unit)
The same between-image 3 sigma % is given for two brightness values per image, in image counts: the image
intensity (simple mean of all pixels of the image) and the cell intensity (mean over the holes of each hole's
cell intensity, see below).
Below that, per image and method: valid holes, mean CD and the within-image 3 sigma % = 3 sigma / mean x 100,
sigma being the standard deviation of that image's hole CDs (N - 1 in the denominator). Every statistic is given
for the CD and separately for the x and the y width (cd_x_px / cd_y_px of each method; for the half_area family
the x / y diameters of the ellipse with the region's area and x / y spread), e.g. for an aberration along x.

Each image tab starts with the CD map: the image with every detected hole drawn at its CD (drag to pan, wheel
to zoom; method, zoom and pan stay when switching tabs; colours on one scale for all images by default). Clicking
a hole shows how the selected method measured it (the cell it worked on, projection's x/y profiles and edges,
half_area's region, local level and threshold). Below the map, a band profile averages CD, brightness,
spot peak and sub-pixel position over bands of rows or columns, to tell where banding comes from. Then the
statistics, the per-hole agreement between methods and CD histograms. The statistics also go to
REPORT/summary.csv (per image) and REPORT/between_images.csv. --save-results also keeps the full per-hole
results in REPORT/results/<image>/ (``subpx.load_cd_results``).
Hole details take about 40 MB per 1e5-hole image in REPORT/data/<image>.detail.js (--no-hole-details skips them).
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subpx import measure_cds, save_cd_results
from subpx.cd import METHOD_LABELS

ALL_METHODS = ("gaussian", "projection", "half_area", "half_area_min", "half_area_min_avg", "tail_area",
               "tail_area_norm", "edge_area", "foot_area", "foot_area_ratio")
DESCRIPTIONS = {
    "gaussian": "FWHM of a 2-D Gaussian fitted to the spot (deblended). Describes the blurred spot, not the hole; "
                "a brighter pulse can change it.",
    "projection": "Mean of the x and y widths between the steepest rise and fall of the cell's summed column and row profiles.",
    "half_area": "Area of the region around the peak brighter than the cell's local level plus half the peak height "
                 "(cell enlarged 4x by smooth interpolation), as the diameter of a circle of that area. Assumes no "
                 "shape; a pulse that brightens the feature and its surroundings together does not change it.",
    "half_area_min": "As half_area, but the local level is the minimum of the cell's outermost pixels instead of their "
                     "median, so the feature's own tail reaching the cell edge cannot raise it.",
    "half_area_min_avg": "As half_area_min, but the peak is the mean of the brightest pixels (--peak-pixels pixels' "
                         "worth of the enlarged cell) instead of the single brightest point, so one outstanding "
                         "pixel cannot set the threshold.",
    "tail_area": "Area above a low threshold near the background (--tail-level of the height, default 0.2), measured "
                 "on a wider window with the background and all other holes removed, so the area may reach past the "
                 "hole's cell. Local level = median of the hole's own area farthest from every hole (its corners), "
                 "where its own light has faded; averaged peak. The noisy bright core "
                 "barely moves the threshold; the dim tail decides the area. Its mean light above the local level "
                 "over the region is the cell intensity.",
    "tail_area_norm": "Each pattern divided by its own averaged peak (as half_area_min_avg), then the area where the "
                      "normalized pattern is above --tail-level (default 0.2), on the same wider window as tail_area. "
                      "No background level is subtracted.",
    "edge_area": "Area above the brightness where the pattern's light falls fastest, on the same wider window as "
                 "tail_area: the threshold is the window's mean brightness with each pixel weighted by its "
                 "steepness, so the flat bright top and the flat surroundings count for nothing. Uses neither the "
                 "peak nor a background level.",
    "foot_area": "Area above the brightness at the foot of the edge, where the falling slope bends into the flat "
                 "surroundings (a tail boundary), on the same wider window as tail_area. Found from the shape of "
                 "the profile (its curvature below the steepest level), not from its height: uses neither the peak "
                 "nor a background level.",
    "foot_area_ratio": "As foot_area, but the foot is found once per image: the median ratio of the foot to the "
                       "steepest-slope brightness over all holes, times each hole's own steepest-slope brightness. "
                       "Same tail boundary, less noise per hole; assumes the holes share the profile shape. Uses "
                       "neither the peak nor a background level.",
}
DESCRIPTIONS.update({
    "local_projection": "Width between the steepest edges of the raw cell's x/y projections after subtracting "
                        "its median edge level; no fitted background or neighbour subtraction.",
    "local_half_area": "Area connected to the peak above the local half-height threshold in the raw Voronoi "
                       "cell, enlarged 4x; median edge level and maximum peak; no neighbour subtraction.",
    "local_half_area_min_avg": "As local_half_area, with the minimum cell-edge level and the averaged brightest "
                               "pixels as the peak.",
    "local_tail_area": "Area connected to the peak above the low local threshold in the raw Voronoi cell, "
                       "enlarged 4x; median edge level and averaged peak; neighbour light remains in the cell.",
    "pixel_half_area": "Raw-cell half-height area counted on original pixels, using the median edge level and "
                       "maximum peak; no enlargement or neighbour subtraction.",
    "pixel_half_area_min_avg": "Raw-cell half-height area counted on original pixels, using the minimum edge "
                               "level and averaged brightest pixels; no enlargement or neighbour subtraction.",
    "pixel_tail_area": "Raw-cell area above the low local threshold counted on original pixels, using the "
                       "median edge level and averaged peak; no enlargement or neighbour subtraction.",
})
# The CD and the x / y widths: statistics key suffix, metric, label.
AXES = (("", "cd_equivalent_px", "CD"), ("_x", "cd_x_px", "x"), ("_y", "cd_y_px", "y"))


def _spread(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    mean = float(values.mean()) if values.size else float("nan")
    std = float(values.std(ddof=1)) if values.size > 1 else float("nan")
    return {"valid": int(values.size), "mean": mean, "std": std,
            "three_sigma_pct": 300 * std / mean if values.size > 1 and mean else float("nan")}


def cd_statistics(result) -> dict[str, dict]:
    """Per method: cells, valid holes, mean, sigma and 3 sigma % of the valid holes' CDs (in result.unit); the same
    for the x and y widths under the keys ``valid_x``, ``mean_x``, ... and ``..._y``."""
    stats = {}
    for name, measured in result.measurements.items():
        stats[name] = {"cells": int(len(measured.valid))}
        for suffix, metric, _ in AXES:
            spread = _spread(measured.metrics[metric][measured.valid] * result.pixel_size)
            stats[name].update({key + suffix: value for key, value in spread.items()})
    return stats


def layer_values(result) -> dict[str, np.ndarray]:
    """Per map layer: each hole's CD in px, NaN where the method failed; ``<method>_x`` / ``_y``: its x / y widths."""
    layers = {}
    for name, measured in result.measurements.items():
        for suffix, metric, _ in AXES:
            layers[name + suffix] = np.where(measured.valid, measured.metrics[metric], np.nan)
    intensity = cell_intensity(result)
    if intensity is not None:
        layers["cell_intensity"] = intensity
    return layers


def intensity_source(methods) -> str | None:
    """The tail-area method supplying cell intensity, preferring the modelled one."""
    return next((name for name in ("tail_area", "local_tail_area", "pixel_tail_area") if name in methods), None)


def cell_intensity(result) -> np.ndarray | None:
    """Per-hole mean signal in the selected tail-area region, NaN where invalid.

    Prefers tail_area, then local_tail_area, then pixel_tail_area; the latter
    two measure raw cells, without fitted background or neighbour subtraction.
    """
    measured = result.measurements.get(intensity_source(result.measurements))
    return None if measured is None else np.where(measured.valid, measured.metrics["mean_signal"], np.nan)


def agreement(result) -> tuple[list[str], np.ndarray]:
    """Correlation of the per-hole CDs of every pair of methods (and the cell intensity), over the holes both have."""
    columns = {name: np.where(m.valid, m.metrics["cd_equivalent_px"], np.nan) for name, m in result.measurements.items()}
    intensity = cell_intensity(result)
    if intensity is not None:
        columns["cell_intensity"] = intensity
    names = list(columns)
    r = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            x, y = columns[a], columns[b]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 2 and x[ok].std() > 0 and y[ok].std() > 0:
                r[i, j] = np.corrcoef(x[ok], y[ok])[0, 1]
    return names, r


def between_image_statistics(entries: list[dict], methods: tuple[str, ...]) -> dict[str, dict]:
    """Per method: spread of the per-image mean CDs over the images that method measured (see module docstring);
    the same for the mean x and y widths under the keys ``images_x``, ``mean_of_means_x``, ... and ``..._y``."""
    stats = {}
    for name in methods:
        stats[name] = {}
        for suffix, _, _ in AXES:
            means = np.array([e["stats"][name]["mean" + suffix] for e in entries
                              if not e.get("error") and name in e["stats"]])
            stats[name].update({key + suffix: value for key, value in _between_values(means).items()})
    return stats


def _between_values(values) -> dict:
    """Spread of one value per image: N, M (mean), s (standard deviation, N - 1), 3 s and 3 s / M x 100."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mean = float(values.mean()) if values.size else float("nan")
    std = float(values.std(ddof=1)) if values.size > 1 else float("nan")
    return {"images": int(values.size), "mean_of_means": mean, "std_of_means": std, "three_sigma": 3 * std,
            "three_sigma_pct": 300 * std / mean if values.size > 1 and mean else float("nan")}


def display_image(image: np.ndarray) -> np.ndarray:
    """8-bit display copy, contrast-stretched between the 0.5 and 99.5 percentiles (measurements use the original)."""
    lo, hi = np.percentile(image, [0.5, 99.5])
    return np.clip(np.rint((image.astype(float) - lo) * (255 / max(hi - lo, 1e-12))), 0, 255).astype(np.uint8)


def _b64(array: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


RAYS = 32  # directions per outline on the map


def contour_rays(measured, n: int = RAYS) -> np.ndarray:
    """Each hole's measured outline as its distance (px) from the hole's center in ``n`` directions (angle 2 pi j / n,
    x right, y down); NaN where there is none. Radial outlines (half_area family) are subsampled from their own rays;
    ellipse outlines (gaussian, projection) are read off the ellipse through their polygon."""
    geometry = measured.geometry
    if "edge_px" in geometry:
        edges = geometry["edge_px"]
        return edges[:, ::max(1, edges.shape[1] // n)][:, :n].astype(float)
    points = measured.contours_xy[:, :-1]  # drop the closing point
    center = points.mean(axis=1, keepdims=True)
    offset = points - center
    # A uniformly sampled ellipse c + A (cos t, sin t) has covariance A A^T / 2; its outline is p^T (A A^T)^-1 p = 1.
    shape_matrix = 2 * np.einsum("nki,nkj->nij", offset, offset) / offset.shape[1]
    ok = np.isfinite(shape_matrix).all(axis=(1, 2)) & (np.linalg.det(np.nan_to_num(shape_matrix)) > 1e-12)
    inverse = np.full_like(shape_matrix, np.nan)
    inverse[ok] = np.linalg.inv(shape_matrix[ok])
    angles = 2 * np.pi * np.arange(n) / n
    d = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    return 1 / np.sqrt(np.einsum("ai,nij,aj->na", d, inverse, d))


def rays_data(result, name: str) -> dict:
    """``contour_rays`` of one method as 8-bit steps of ``unit`` px, base64 (255 = no outline)."""
    rays = contour_rays(result.measurements[name])
    finite = np.isfinite(rays)
    unit = max(1 / 16, float(rays[finite].max()) / 254) if finite.any() else 1 / 16
    codes = np.where(finite, np.clip(np.rint(np.nan_to_num(rays) / unit), 0, 254), 255)
    return {"n": int(rays.shape[1]), "unit": unit, "rays": _b64(codes, "u1")}


def hole_data(result, layers: dict[str, np.ndarray], ident: str = "", image=None) -> dict:
    """Per map layer: hole centers, CD (px), validity and status as base64 arrays; plus what the band profile needs.

    The x / y width layers name their ``base`` method and share its centers and status.
    Numerical summaries require the source image explicitly. Missing contours
    use equivalent-CD circles in the viewer, without inventing measured rays.
    """
    seeds = result.seeds_xy
    methods = {}
    for name, measured in result.measurements.items():
        centers = np.where(np.isfinite(measured.centers_xy), measured.centers_xy, seeds)
        statuses, codes = np.unique(measured.status, return_inverse=True)
        methods[name] = {
            "label": METHOD_LABELS[name], "x": _b64(centers[:, 0], "<f4"), "y": _b64(centers[:, 1], "<f4"),
            "cd": _b64(layers[name], "<f4"),
            "valid": _b64(measured.valid, "u1"), "status": _b64(codes, "u1" if len(statuses) < 256 else "<u2"),
            "status_names": statuses.tolist(), "status_wide": len(statuses) >= 256,
        }
        edges = measured.geometry.get("edge_px")
        has_outlines = ((edges is not None and edges.ndim == 2 and edges.shape[1] > 0)
                        or measured.contours_xy.shape[1] > 1)
        methods[name]["outline_kind"] = "measured" if has_outlines else "equivalent_circle"
        if has_outlines:
            methods[name]["rays_file"] = f"data/{ident}.{name}.rays.js"
        for suffix, _, label in AXES[1:]:
            methods[name + suffix] = {"label": f"{METHOD_LABELS[name]}, {label} width only", "base": name,
                                      "cd": _b64(layers[name + suffix], "<f4")}
    if "cell_intensity" in layers:  # not a length: its own scale and unit
        source = intensity_source(result.measurements)
        context = "background and neighbours removed" if source == "tail_area" else "raw cell, neighbour light retained"
        methods["cell_intensity"] = {"label": f"Cell intensity: mean light above the local level in {source}'s "
                                              f"region ({context})",
                                     "base": source, "cd": _b64(layers["cell_intensity"], "<f4"),
                                     "scale": 1.0, "unit": "counts"}
    model = getattr(result, "deblend_model", None)
    if model:
        peak, peak_label = model["params"][model["report"], 0], "fitted spot amplitude"
    elif hasattr(result, "cell_labels"):
        peak, peak_label = result.mean_intensity, "mean cell intensity"
    else:
        first = next(((name, m.metrics["peak_signal"]) for name, m in result.measurements.items()
                      if "peak_signal" in m.metrics), None)
        peak = first[1] if first else np.full(len(seeds), np.nan)
        peak_label = f"{first[0]} peak above local level" if first else "unavailable peak"
    image = getattr(result, "image", None) if image is None else image
    if image is None:
        raise ValueError("hole_data requires image= with a CDSummary")
    image = np.asarray(image, dtype=float)
    pitch = result.meta.get("pitch_px")
    return {"n": int(len(seeds)), "width": int(image.shape[1]), "height": int(image.shape[0]),
            "scale": float(result.pixel_size), "unit": result.unit, "methods": methods,
            # Background brightness of each pixel row / column: its 25th percentile, i.e. the absorber between holes.
            "peak": _b64(peak, "<f4"), "peak_label": peak_label,
            "row_bg": _b64(np.percentile(image, 25, axis=1), "<f4"),
            "col_bg": _b64(np.percentile(image, 25, axis=0), "<f4"), "pitch": float(pitch) if pitch else None}


def _quantize(cells: np.ndarray, inside: np.ndarray) -> dict:
    """8-bit codes per cell pixel (255 = outside the cell) with each cell's offset and step, base64."""
    n = len(cells)
    flat = cells.reshape(n, -1).astype(float)
    inside = inside.reshape(n, -1) & np.isfinite(flat)
    lo = np.where(inside, flat, np.inf).min(axis=1)
    hi = np.where(inside, flat, -np.inf).max(axis=1)
    lo, hi = np.where(np.isfinite(lo), lo, 0.0), np.where(np.isfinite(hi), hi, 1.0)
    step = np.maximum(hi - lo, 1e-9) / 254
    codes = np.where(inside, np.rint((np.where(inside, flat, 0) - lo[:, None]) / step[:, None]), 255)
    return {"codes": _b64(codes, "u1"), "lo": _b64(lo, "<f4"), "step": _b64(step, "<f4")}


def tail_detail(result) -> dict | None:
    """tail_area's wider windows (each hole's share of the light, the middle 1.5 pitches), 8-bit, for its panel."""
    measured = next((result.measurements[n] for n in ("tail_area", "tail_area_norm", "edge_area", "foot_area",
                                                       "foot_area_ratio")
                     if n in result.measurements), None)
    if measured is None or "cells" not in measured.inputs:
        return None
    cells, inside, origins = measured.inputs["cells"], measured.inputs["inside"], measured.inputs["origins"]
    return {"h": int(cells.shape[1]), "w": int(cells.shape[2]), "ox": _b64(origins[:, 0], "<i4"),
            "oy": _b64(origins[:, 1], "<i4"), "cells": _quantize(cells, inside)}


def detail_data(result) -> dict:
    """What each method measured each hole on, for the click panel.

    ``clean``: the cell with background and neighbours' models removed (input of every method; zero outside the
    cell), 8-bit for display. ``prof_x``/``prof_y``: projection's column and row sums of that cell in full
    precision (8-bit sums move the steepest points visibly). ``half``: half_area's local level, peak height above
    it, threshold, region centroid in the cell and its outline as distances along 32 rays (1/16 px, 8-bit).
    """
    n, h, w = result.masks.shape
    out = {"h": h, "w": w, "ox": _b64(result.origins_xy[:, 0], "<i4"), "oy": _b64(result.origins_xy[:, 1], "<i4"),
           "clean": _quantize(result.rois, result.masks)}
    if "projection" in result.measurements:
        cells = np.where(result.masks, result.rois, 0.0)
        out["prof_x"], out["prof_y"] = _b64(cells.sum(axis=1), "<f4"), _b64(cells.sum(axis=2), "<f4")
    out["half"] = {}
    peak_pixels = result.meta.get("half_area_peak_pixels", 4.0)
    averaged = f"mean of the brightest {peak_pixels:g} pixels' worth of the enlarged cell"
    edge = "of the cell's outermost pixels"
    corners = ("median of the farthest tenth of the hole's own area (the pixels closer to it than to any other hole), "
               "where its light has faded, on a wider window holding its share of the light; the area may reach past "
               "the cell drawn here")
    for name, background, peak in (
            ("half_area", f"median {edge}", "highest point"), ("half_area_min", f"minimum {edge}", "highest point"),
            ("half_area_min_avg", f"minimum {edge}", averaged), ("tail_area", corners, averaged),
            ("tail_area_norm", "zero: no background step; the pattern is divided by its peak and the area is where "
             "it exceeds the level", averaged),
            ("edge_area", "zero: no background step", "not used (shown for comparison: " + averaged + ")"),
            ("foot_area", "zero: no background step", "not used (shown for comparison: " + averaged + ")"),
            ("foot_area_ratio", "zero: no background step", "not used (shown for comparison: " + averaged + ")")):
        if name not in result.measurements:
            continue
        measured = result.measurements[name]
        m, geometry = measured.metrics, measured.geometry
        edge = geometry["edge_px"]
        rays = edge[:, ::max(1, edge.shape[1] // 32)][:, :32]
        out["half"][name] = {"background": background, "peak_rule": peak, "level": _b64(m["local_level"], "<f4"), "peak": _b64(m["peak_signal"], "<f4"),
                             "threshold": _b64(m["threshold"], "<f4"), "mean": _b64(m["mean_signal"], "<f4"),
                             "cx": _b64(geometry["center_local"][:, 0], "<f4"), "cy": _b64(geometry["center_local"][:, 1], "<f4"),
                             "rays": _b64(np.clip(np.rint(np.nan_to_num(rays) * 16), 0, 255), "u1"), "n_rays": int(rays.shape[1]),
                             "level_fraction": float(result.meta.get(
                                 "tail_area_level" if name.startswith("tail") else "half_area_level",
                                 0.2 if name.startswith("tail") else 0.5))}
        if name == "foot_area":
            out["half"][name]["threshold_rule"] = ("brightness at the foot of the edge, where the falling slope bends "
                                                   "into the flat surroundings")
        if name == "foot_area_ratio":
            out["half"][name]["threshold_rule"] = ("image's foot ratio "
                                                   f"({result.meta.get('foot_ratio', float('nan')):.3f}) x this hole's "
                                                   "steepest-slope brightness")
        if name == "edge_area":
            out["half"][name]["threshold_rule"] = ("mean brightness of the window, each pixel weighted by its steepness "
                                                   "(where the light falls fastest)")
    return out


def _fmt(value: float, digits: int = 2) -> str:
    return "–" if not np.isfinite(value) else f"{value:.{digits}f}"


def _plain(value: float, digits: int = 2) -> str:
    """``_fmt`` for the console: ASCII only, since e.g. a cp949 console cannot print an en dash."""
    return "-" if not np.isfinite(value) else f"{value:.{digits}f}"


def _groups(span: int) -> str:
    """One header cell per AXES entry (CD, x width, y width), each spanning ``span`` columns."""
    return "".join(f"<th colspan='{span}' class='group'>{'CD' if not x else label + ' width'}</th>"
                   for x, _, label in AXES)


def _stats_table(stats: dict, unit: str) -> str:
    u = html.escape(unit)
    rows = "".join(
        f"<tr><td><span class='swatch' style='background:{COLORS[m]}'></span>{html.escape(METHOD_LABELS[m])} "
        f"<code>{m}</code></td><td>{s['valid']:,} / {s['cells']:,}</td>"
        + "".join(f"<td class='first'>{_fmt(s['mean' + x])}</td><td>{_fmt(s['std' + x])}</td>"
                  f"<td><b>{_fmt(s['three_sigma_pct' + x])}</b></td>" for x, _, _ in AXES) + "</tr>"
        for m, s in stats.items())
    columns = "".join(f"<th>Mean ({u})</th><th>σ ({u})</th><th>3σ (%)</th>" for _ in AXES)
    return (f"<table class='stats wide'><thead><tr><th rowspan='2'>Method</th><th rowspan='2'>Valid holes</th>"
            f"{_groups(3)}</tr><tr>{columns}</tr></thead><tbody>{rows}</tbody></table>")


def _intensity_table(stats: dict, source: str = "tail_area") -> str:
    context = "background and neighbours removed" if source == "tail_area" else "raw cell, neighbour light retained"
    return (f"<h3>Cell intensity</h3><p class='meta'>Per hole: the mean light above its local level over {html.escape(source)}'s "
            f"region ({context}), in image counts. It follows the source pulse and the hole size "
            "together (a bigger hole passes more light).</p><table class='stats'><thead><tr><th>Valid holes</th>"
            "<th>Mean (counts)</th><th>σ (counts)</th><th>3σ (%)</th></tr></thead><tbody>"
            f"<tr><td>{stats['valid']:,}</td><td>{_fmt(stats['mean'])}</td><td>{_fmt(stats['std'])}</td>"
            f"<td><b>{_fmt(stats['three_sigma_pct'])}</b></td></tr></tbody></table>")


def _agreement_table(names: list[str], r: np.ndarray) -> str:
    head = "".join(f"<th>{html.escape(n)}</th>" for n in names)
    body = "".join(f"<tr><td>{html.escape(a)}</td>" + "".join(f"<td>{_fmt(v)}</td>" for v in row) + "</tr>"
                   for a, row in zip(names, r))
    return ("<h3>Do the methods agree on which holes are big?</h3><p class='meta'>Pearson correlation of the per-hole "
            "CDs over holes both methods measured: 1 = a perfect positive linear relation; 0 = no linear relation. "
            "High correlation does not establish agreement in absolute values, and low correlation alone does not "
            "identify measurement noise. The <b>cell_intensity</b> row describes association with hole brightness; "
            "correlation alone cannot distinguish source-pulse changes, real size variation or shared input effects.</p>"
            f"<table class='stats wide'><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>")


COLORS = {"gaussian": "#5bbf2a", "projection": "#7a7a7a", "projection_x": "#2b8cbe", "projection_y": "#e6550d",
          "half_area": "#d6338a", "half_area_min": "#8e44ad", "half_area_min_avg": "#d68910", "tail_area": "#138d75",
          "tail_area_norm": "#2e86c1", "edge_area": "#c0392b",
          "foot_area": "#16a085", "foot_area_ratio": "#7d3c98"}
COLORS.update({"local_projection": "#64748b", "local_half_area": "#b83280",
               "local_half_area_min_avg": "#b7791f", "local_tail_area": "#0d9488",
               "pixel_half_area": "#a855f7", "pixel_half_area_min_avg": "#ea580c", "pixel_tail_area": "#0284c7"})
# x / y width layers: the method's colour, lighter for x and darker for y (projection keeps its own two).
COLORS["cell_intensity"] = "#b8860b"
for _name, _color in list(COLORS.items()):
    if not _name.endswith(("_x", "_y")) and _name != "cell_intensity":
        _rgb = np.array([int(_color[i:i + 2], 16) for i in (1, 3, 5)])
        COLORS.setdefault(_name + "_x", "#%02x%02x%02x" % tuple(np.rint(_rgb + (255 - _rgb) * 0.45).astype(int)))
        COLORS.setdefault(_name + "_y", "#%02x%02x%02x" % tuple(np.rint(_rgb * 0.6).astype(int)))


BRIGHTNESS = {"image_intensity": "Image intensity: simple mean of all pixels of the image",
              "cell_intensity": "Cell intensity: mean over the holes of each hole's cell intensity"}


def brightness_between(entries: list[dict]) -> dict[str, dict]:
    """Between-image spread (as between_image_statistics) of the image intensity (mean of all pixels) and, when
    tail_area ran, of the cell intensity (mean over the holes), one value per image."""
    done = [e for e in entries if not e.get("error")]
    out = {"image_intensity": _between_values([e["image_mean"] for e in done])}
    cells = [e["intensity"]["mean"] for e in done if e.get("intensity")]
    if cells:
        out["cell_intensity"] = _between_values(cells)
    return out


def _brightness_summary(entries: list[dict]) -> str:
    between = brightness_between(entries)
    rows = "".join(
        f"<tr><td>{html.escape(BRIGHTNESS[name])} <code>{name}</code></td><td>{b['images']}</td>"
        f"<td class='first'>{_fmt(b['mean_of_means'])}</td><td>{_fmt(b['std_of_means'], 3)}</td>"
        f"<td>{_fmt(b['three_sigma'], 3)}</td><td><b>{_fmt(b['three_sigma_pct'], 3)}</b></td></tr>"
        for name, b in between.items())
    cell = "cell_intensity" in between
    per_image = "".join(
        f"<tr><td><a href='#' data-tab='{e['id']}'>{html.escape(e['name'])}</a></td><td>{_fmt(e['image_mean'])}</td>"
        + (f"<td>{_fmt(e['intensity']['mean'])}</td><td>{e['intensity']['valid']:,}</td>"
           f"<td>{_fmt(e['intensity']['three_sigma_pct'])}</td>" if e.get("intensity") else
           ("<td>–</td><td>–</td><td>–</td>" if cell else "")) + "</tr>"
        for e in entries if not e.get("error"))
    cell_head = ("<th>Cell intensity, mean over holes</th><th>Valid holes</th><th>Within-image 3σ of the cell "
                 "intensity (%)</th>") if cell else ""
    return ("<h3>Between images: brightness</h3><p class='meta'>One value per image, in image counts: the image "
            "intensity is the simple mean of all its pixels; the cell intensity is the mean over its holes of each "
            "hole's mean light above the local level in the selected tail-area method's region. M, s and 3 s / M as for the CD above "
            "(s with N − 1).</p>"
            "<table class='stats'><thead><tr><th>Brightness</th><th>Images (N)</th><th>M (counts)</th>"
            "<th>s (counts)</th><th>3s (counts)</th><th>3s / M (%)</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
            "<table class='stats'><thead><tr><th>Image</th><th>Image intensity (mean of all pixels)</th>"
            f"{cell_head}</tr></thead><tbody>{per_image}</tbody></table>")


def write_report(output: Path, entries: list[dict], methods: tuple[str, ...], notes: list[str], unit: str,
                 ranges: dict | None = None, extra_summary: str = "") -> Path:
    """index.html: summary tab, one tab per image (data loaded on demand from data/<id>.js).

    ``ranges``: per map layer, the CD colour range (px) shared by all images.
    ``extra_summary``: caller-generated HTML placed before the standard statistics.
    """
    def table(title, key):
        """Per image: ``key`` of the CD, the x and the y width (one column group each) of every method."""
        names = "".join(f"<th><span class='swatch' style='background:{COLORS[m]}'></span>{html.escape(m)}</th>"
                        for _ in AXES for m in methods)
        body = []
        for e in entries:
            if e.get("error"):
                body.append(f"<tr><td>{html.escape(e['name'])}</td><td colspan='{1 + len(AXES) * len(methods)}' "
                            f"class='error'>failed: {html.escape(e['error'])}</td></tr>")
                continue
            cells = "".join(
                (f"<td{' class=first' if j == 0 else ''} title='{e['stats'][m]['valid' + x]:,} valid holes'>"
                 f"{_fmt(e['stats'][m][key + x])}</td>") if m in e["stats"] else "<td>–</td>"
                for x, _, _ in AXES for j, m in enumerate(methods))
            body.append(f"<tr><td><a href='#' data-tab='{e['id']}'>{html.escape(e['name'])}</a></td>"
                        f"<td>{e['holes']:,}</td>{cells}</tr>")
        return (f"<h3>{title}</h3><table class='stats wide'><thead><tr><th rowspan='2'>Image</th>"
                f"<th rowspan='2'>Holes</th>{_groups(len(methods))}</tr><tr>{names}</tr></thead>"
                f"<tbody>{''.join(body)}</tbody></table>")

    between = between_image_statistics(entries, methods)
    unit_html = html.escape(unit)
    rows = "".join(
        f"<tr><td><span class='swatch' style='background:{COLORS[m]}'></span>{html.escape(METHOD_LABELS[m])} "
        f"<code>{m}</code></td><td>{b['images']}</td>"
        + "".join(f"<td class='first'>{_fmt(b['mean_of_means' + x])}</td><td>{_fmt(b['std_of_means' + x], 3)}</td>"
                  f"<td>{_fmt(b['three_sigma' + x], 3)}</td><td><b>{_fmt(b['three_sigma_pct' + x], 3)}</b></td>"
                  for x, _, _ in AXES) + "</tr>"
        for m, b in between.items())
    columns = "".join(f"<th>M ({unit_html})</th><th>s ({unit_html})</th><th>3s ({unit_html})</th><th>3s / M (%)</th>"
                      for _ in AXES)
    summary = (
        "<h3>Between images: do the images show the same mean CD?</h3>"
        "<p class='meta'>Per method, m<sub>i</sub> = mean CD of image i over its validly measured holes, N = number of "
        "images. M = (m<sub>1</sub> + … + m<sub>N</sub>) / N; s = √( Σ (m<sub>i</sub> − M)² / (N − 1) ); "
        "<b>between-image 3σ % = 3 s / M × 100</b>. The same is given for each method's x width and y width "
        "(for the half_area family: the x / y diameters of the ellipse with the region's area and x / y spread), "
        "so a difference between the two directions, e.g. from an aberration along x, shows up.</p>"
        f"<table class='stats wide'><thead><tr><th rowspan='2'>Method</th><th rowspan='2'>Images (N)</th>{_groups(4)}"
        f"</tr><tr>{columns}</tr></thead><tbody>{rows}</tbody></table>"
        + table(f"Mean CD per image, m<sub>i</sub> ({unit_html})", "mean")
        + table("Within-image 3σ (%) = 3 × σ of that image's hole CDs / its mean × 100", "three_sigma_pct")
        + _brightness_summary(entries))
    legend = "".join(f"<li><span class='swatch' style='background:{COLORS[m]}'></span><b>{m}</b> — "
                     f"{html.escape(DESCRIPTIONS[m])}</li>" for m in methods)
    tabs = "".join(f"<button class='tab' data-tab='{e['id']}'{' disabled' if e.get('error') else ''}>"
                   f"{html.escape(e['name'])}</button>" for e in entries)
    panels = "".join(
        f"<section class='panel' id='panel-{e['id']}' hidden><h2>{html.escape(e['name'])}: statistics</h2>"
        f"<p class='meta'>{e['holes']:,} holes · {e['width']} × {e['height']} px · image intensity (mean of all "
        f"pixels) {_fmt(e['image_mean'])} counts · {e['seconds']:.1f} s</p>{_stats_table(e['stats'], unit)}"
        f"{_intensity_table(e['intensity'], e.get('intensity_source') or intensity_source(e['stats']) or 'tail_area') if e.get('intensity') else ''}"
        f"{_agreement_table(*e['agreement'])}"
        "<h3>CD histograms</h3><div class='hists'></div></section>"
        for e in entries if not e.get("error"))
    page = (_HTML.replace("__SUMMARY__", extra_summary + summary).replace("__LEGEND__", legend).replace("__TABS__", tabs)
            .replace("__PANELS__", panels)
            .replace("__NOTES__", "".join(f"<p class='note'>{html.escape(n)}</p>" for n in notes))
            .replace("__ENTRIES__", json.dumps({e["id"]: {"name": e["name"], "image": e["image"], "data": e["data"],
                                                          "detail": e.get("detail"), "tail": e.get("tail")}
                                                for e in entries if not e.get("error")}))
            .replace("__RANGES__", json.dumps(ranges or {}))
            .replace("__COLORS__", json.dumps(COLORS)))
    path = output / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def describe_backend(requested: str) -> str:
    """Which backend measure_cds will use for ``requested``, and why not the GPU when it falls back to the CPU."""
    if requested == "cpu":
        return "CPU (--backend cpu)"
    try:
        import cupy
    except Exception as error:  # not installed, or its CUDA libraries are missing
        reason = f"CuPy cannot be imported ({type(error).__name__}: {error})"
    else:
        try:
            if cupy.cuda.runtime.getDeviceCount():
                name = cupy.cuda.runtime.getDeviceProperties(0)["name"]
                name = name.decode() if isinstance(name, bytes) else name
                cupy.arange(4).sum().item()  # a real kernel: needs the CUDA runtime and compiler libraries
                return f"GPU ({name}, CuPy {cupy.__version__})"
            reason = "CuPy finds no CUDA device"
        except Exception as error:
            reason = f"CuPy cannot use the GPU ({type(error).__name__}: {error})"
    if requested == "gpu":
        return f"GPU requested, but {reason}"
    return f"CPU, because {reason}. GPU setup: docs/gpu.md"


def main(argv=None) -> None:
    from PIL import Image
    from scripts.analyze_contact_holes import load_image

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="Folder holding the .png images")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/cd_report")
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=list(ALL_METHODS))
    parser.add_argument("--half-level", type=float, default=0.5,
                        help="half_area: fraction of the peak height above the local level that bounds the region")
    parser.add_argument("--peak-pixels", type=float, default=4.0,
                        help="half_area_min_avg: the peak is the mean of the brightest pixels covering this many "
                             "original pixels (of the 4x enlarged cell)")
    parser.add_argument("--tail-level", type=float, default=0.2,
                        help="tail_area: fraction of the height above the local level that bounds the region "
                             "(low = near the background)")
    parser.add_argument("--tail-window", type=float, default=1.2,
                        help="tail_area: half-width of its window around each hole, in pitches (0.5 to 2)")
    parser.add_argument("--pixel-size", type=float, default=1.0)
    parser.add_argument("--unit", default="px")
    parser.add_argument("--backend", choices=("cpu", "gpu", "auto"), default="auto")
    parser.add_argument("--seeds", choices=("threshold", "peaks"), default="threshold",
                        help="peaks: local maxima, for arrays too dense to threshold")
    parser.add_argument("--area-min", type=int, default=3)
    parser.add_argument("--area-max", type=int, default=200)
    parser.add_argument("--invert", action="store_true", help="Dark holes on a bright background")
    parser.add_argument("--no-hole-details", action="store_true",
                        help="Skip the per-hole detail data (cells behind the click panel, about 40 MB per 1e5 holes)")
    parser.add_argument("--save-results", action="store_true",
                        help="Also save full per-hole results (compact NPZ + CSVs)")
    args = parser.parse_args(argv)

    images = sorted(p for p in args.folder.iterdir() if p.suffix.lower() == ".png") if args.folder.is_dir() else []
    if not images:
        raise SystemExit(f"No .png images in {args.folder}")
    methods, notes = tuple(m for m in ALL_METHODS if m in args.methods), []
    output = args.output
    for sub in ("images", "data"):
        (output / sub).mkdir(parents=True, exist_ok=True)
    print(f"Backend: {describe_backend(args.backend)}", flush=True)
    entries, rows, used = [], [], set()
    samples = {}  # per map layer: a sample of every image's CDs, for the shared colour range
    rng = np.random.default_rng(0)
    started = time.perf_counter()
    for number, path in enumerate(images, 1):
        ident = re.sub(r"[^A-Za-z0-9_-]", "_", path.stem) or "image"
        while ident in used:
            ident += "_"
        used.add(ident)
        print(f"[{number}/{len(images)}] {path.name}", flush=True)
        start = time.perf_counter()
        try:
            image = load_image(path)
            result = measure_cds(
                image, methods=methods, seeds=args.seeds, invert=args.invert, area_min=args.area_min,
                area_max=args.area_max, pixel_size=args.pixel_size, unit=args.unit, backend=args.backend,
                half_area_level=args.half_level, half_area_peak_pixels=args.peak_pixels,
                tail_area_level=args.tail_level, tail_window=args.tail_window,
            )
        except Exception as error:  # one bad image must not stop the batch
            message = str(error).replace(str(path), path.name) or type(error).__name__
            print(f"  failed: {message}", flush=True)
            entries.append({"id": ident, "name": path.name, "error": message})
            rows.append({"image": path.name, "status": "failed", "error": message})
            continue
        seconds = time.perf_counter() - start
        image_mean = float(np.mean(image, dtype=np.float64))  # simple mean of all pixels, as loaded
        stats = cd_statistics(result)
        intensity = cell_intensity(result)
        intensity = None if intensity is None else _spread(intensity)
        layers = layer_values(result)
        for layer, values in layers.items():
            values = values[np.isfinite(values)]
            samples.setdefault(layer, []).append(rng.choice(values, min(values.size, 50_000), replace=False))
        Image.fromarray(display_image(image)).save(output / "images" / f"{ident}.png")
        (output / "data" / f"{ident}.js").write_text(
            f"window.CD_DATA[{json.dumps(ident)}]=" + json.dumps(hole_data(result, layers, ident), allow_nan=False)
            + ";\n", encoding="utf-8")
        for name in result.measurements:  # measured outlines, loaded by the map when zoomed in
            (output / "data" / f"{ident}.{name}.rays.js").write_text(
                f"window.CD_RAYS[{json.dumps(ident + '/' + name)}]=" + json.dumps(rays_data(result, name)) + ";\n",
                encoding="utf-8")
        detail = tail = None
        if not args.no_hole_details:
            detail = f"data/{ident}.detail.js"
            (output / detail).write_text(f"window.CD_DETAIL[{json.dumps(ident)}]="
                                         + json.dumps(detail_data(result), allow_nan=False) + ";\n",
                                         encoding="utf-8")
            wide = tail_detail(result)
            if wide is not None:
                tail = f"data/{ident}.tail.js"
                (output / tail).write_text(f"window.CD_TAIL[{json.dumps(ident)}]=" + json.dumps(wide) + ";\n",
                                           encoding="utf-8")
        if args.save_results:
            save_cd_results(result, output / "results" / ident, compact=True)
        entries.append({"id": ident, "name": path.name, "holes": len(result.seeds_xy), "stats": stats,
                        "width": image.shape[1], "height": image.shape[0], "seconds": seconds,
                        "image": f"images/{ident}.png", "data": f"data/{ident}.js", "detail": detail, "tail": tail,
                        "agreement": agreement(result), "intensity": intensity, "image_mean": image_mean})
        for method, s in stats.items():
            rows.append({"image": path.name, "status": "ok", "method": method, "holes": len(result.seeds_xy), **s,
                         "unit": args.unit})
            print(f"  {method:17s} {s['valid']:7d} valid  mean {_plain(s['mean'], 3)} {args.unit}  "
                  f"3sigma within image {_plain(s['three_sigma_pct'])} %  "
                  f"(x {_plain(s['mean_x'], 3)}, {_plain(s['three_sigma_pct_x'])} %; "
                  f"y {_plain(s['mean_y'], 3)}, {_plain(s['three_sigma_pct_y'])} %)", flush=True)
        if intensity:
            rows.append({"image": path.name, "status": "ok", "method": "cell_intensity", "holes": len(result.seeds_xy),
                         "cells": len(result.seeds_xy), **intensity, "unit": "counts"})
            print(f"  {'cell intensity':17s} {intensity['valid']:7d} valid  mean {_plain(intensity['mean'], 3)} counts  "
                  f"3sigma within image {_plain(intensity['three_sigma_pct'])} %", flush=True)
        rows.append({"image": path.name, "status": "ok", "method": "image_intensity", "holes": len(result.seeds_xy),
                     "valid": int(image.size), "mean": image_mean, "unit": "counts"})
        print(f"  {'image intensity':17s} {image.size:7d} px     mean {_plain(image_mean, 3)} counts", flush=True)
        print(f"  {seconds:.1f} s", flush=True)
    fields = (["image", "status", "method", "holes", "cells"]
              + [key + x for x, _, _ in AXES for key in ("valid", "mean", "std", "three_sigma_pct")] + ["unit", "error"])
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    between = between_image_statistics(entries, methods)
    with (output / "between_images.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method"] + [
            key + x for x, _, _ in AXES
            for key in ("images", "mean_of_means", "std_of_means", "three_sigma", "three_sigma_pct")] + ["unit"])
        writer.writeheader()
        writer.writerows({"method": m, **b, "unit": args.unit} for m, b in between.items())
        brightness = brightness_between(entries)
        writer.writerows({"method": name, **b, "unit": "counts"} for name, b in brightness.items())
    print("Between images (3 s / M x 100, s = std of the per-image means, N - 1):", flush=True)
    for method, b in between.items():
        print(f"  {method:17s} N={b['images']}  M {_plain(b['mean_of_means'], 3)} {args.unit}  "
              f"3s {_plain(b['three_sigma'], 3)} {args.unit}  {_plain(b['three_sigma_pct'], 3)} %  "
              f"(x {_plain(b['three_sigma_pct_x'], 3)} %, y {_plain(b['three_sigma_pct_y'], 3)} %)", flush=True)
    for name, b in brightness.items():
        print(f"  {name:17s} N={b['images']}  M {_plain(b['mean_of_means'], 3)} counts  "
              f"3s {_plain(b['three_sigma'], 3)} counts  {_plain(b['three_sigma_pct'], 3)} %", flush=True)
    ranges = {}
    for layer, parts in samples.items():
        pooled = np.concatenate(parts)
        if pooled.size:
            ranges[layer] = [float(v) for v in np.quantile(pooled, [0.02, 0.98])]
    path = write_report(output, entries, methods, notes, args.unit, ranges)
    print(f"Report: {path} ({time.perf_counter() - started:.1f} s)", flush=True)


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CD Report</title>
<style>
:root{--bg:#f6f7f9;--fg:#1d2330;--muted:#5d6675;--card:#fff;--line:#d9dde3;--accent:#2f6fdb}
@media (prefers-color-scheme:dark){:root{--bg:#15181e;--fg:#e4e7ec;--muted:#9aa3b2;--card:#1e222a;--line:#343a46;--accent:#6e9ef0}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--fg)}
header{padding:14px 20px 0}h1{font-size:20px;margin:0 0 10px}h2{font-size:17px;margin:4px 0}
nav{display:flex;flex-wrap:wrap;gap:4px;padding:0 20px;border-bottom:1px solid var(--line)}
.tab{border:1px solid var(--line);border-bottom:none;background:transparent;color:var(--fg);padding:7px 12px;border-radius:6px 6px 0 0;cursor:pointer;font:inherit}
.tab.active{background:var(--card);font-weight:600;border-color:var(--accent)}.tab:disabled{opacity:.45;cursor:default}
main{padding:16px 20px}.panel{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;margin-bottom:14px}
table.stats{border-collapse:collapse;margin:8px 0;display:block;overflow-x:auto;max-width:100%}table.stats th,table.stats td{border-bottom:1px solid var(--line);padding:5px 10px;text-align:right;white-space:nowrap}
table.stats th:first-child,table.stats td:first-child{text-align:left}table.wide{display:block;overflow-x:auto;max-width:100%}
.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:baseline}
.meta,.note{color:var(--muted);margin:2px 0 6px}.error{color:#d33;text-align:left!important;white-space:normal!important;max-width:60ch}
h3{font-size:15px;margin:14px 0 2px}code{font-size:12px;color:var(--muted)}
ul.legend{padding-left:18px;color:var(--muted)}ul.legend b{color:var(--fg)}
.hists{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0}.hist{border:1px solid var(--line);border-radius:6px;padding:4px}
.hist div{font-size:12px;color:var(--muted)}a{color:var(--accent)}
#viewer{display:none}#viewer.on{display:block}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:8px}
.controls select,.controls button,.controls input{font:inherit;padding:3px 8px}
.split{display:grid;grid-template-columns:minmax(0,1fr) 430px;gap:12px;align-items:start}
@media (max-width:1100px){.split{grid-template-columns:1fr}}
#stage{position:relative;height:70vh;min-height:380px;border:1px solid var(--line);border-radius:6px;overflow:hidden;background:#000;cursor:grab}
#stage canvas{width:100%;height:100%;display:block}#bar{display:flex;align-items:center;gap:6px;font-size:12px}
#ramp{width:180px;height:10px;border-radius:2px}
#detail{font-size:13px;max-height:70vh;min-height:380px;overflow:auto;border:1px solid var(--line);border-radius:6px;padding:8px}
#detail table{border-collapse:collapse}#detail td{padding:1px 10px 1px 0}
.cells{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0}.cells figure{margin:0;font-size:12px;color:var(--muted)}
.cells canvas{display:block;image-rendering:pixelated;border:1px solid var(--line)}
canvas.plot{width:100%;display:block;margin:4px 0}
.explain{color:var(--muted);margin:4px 0 8px}
table.stats th.group{text-align:center!important}
table.stats td.first,table.stats th.group{border-left:1px solid var(--line)}
</style></head><body>
<header><h1>CD report</h1></header>
<nav><button class="tab active" data-tab="summary">Summary</button>__TABS__</nav>
<main>
<section class="panel" id="panel-summary"><h2>Summary</h2>
<p class="meta">All statistics use only the holes each method measured validly (hover a per-image value for the count). Standard deviations divide by N − 1. Click an image name to open it.</p>
__NOTES__ __SUMMARY__<h2>Methods</h2><ul class="legend">__LEGEND__</ul></section>
<section class="panel" id="viewer"><h2 id="vtitle"></h2><div class="controls">
<label>Map <select id="method"></select></label>
<label>Colours <select id="scale"><option value="shared">same for all images</option><option value="image">per image</option></select></label>
<button id="fit">Fit image</button><label><input type="checkbox" id="show" checked> Show holes</label>
<span id="bar"><span id="lo"></span><span id="ramp"></span><span id="hi"></span></span>
<span class="meta">Drag to pan · wheel to zoom · click a hole</span><span class="meta" id="outline-note"></span></div>
<div class="split"><div id="stage"><canvas id="canvas"></canvas></div><div id="detail"></div></div>
<h3>Band profile: where does banding come from?</h3>
<div class="controls"><label>Bands of <select id="axis"><option value="y">rows (along y)</option><option value="x">columns (along x)</option></select></label>
<label>Band width (px) <input id="bandw" type="number" min="1" step="1" style="width:6em"></label></div>
<p class="explain">Each band averages the holes whose centre falls in it (default width: one hole pitch, i.e. one row of holes per band). The top plot shows every map's band means, each minus its own image mean, so the maps can be compared directly; the table gives each map's band-to-band spread. How to read it:</p>
<ul class="explain"><li>Every method also has an x width (<b>…_x</b>) and a y width (<b>…_y</b>) map. Bands mainly in the y widths (or mainly in the x widths): the light changes from one pixel line to the next (scan or sensor lines), which distorts each hole's width across the lines.</li>
<li>Bands in the Deblended Gaussian but not in half_area or projection: the light level leaks into the Gaussian CD (half_area and projection ignore a change of light level).</li>
<li>The same bands in half_area and in both projection directions: most likely real CD differences on the mask.</li>
<li>Bands that follow the sub-pixel position: the method is tied to the pixel grid.</li></ul>
<p class="explain">Background brightness and spot peak are shown for reference only: bigger holes are also brighter, and the blur spreads part of their light between the holes, so a CD that follows them does not by itself prove a light leak.</p>
<div id="bandinfo"></div>
<canvas class="plot" id="band-cd" style="height:170px"></canvas><canvas class="plot" id="band-br" style="height:100px"></canvas>
</section>
__PANELS__
</main>
<script>
window.CD_DATA={};window.CD_DETAIL={};window.CD_RAYS={};window.CD_TAIL={};
const ENTRIES=__ENTRIES__, COLORS=__COLORS__, RANGES=__RANGES__, $=id=>document.getElementById(id);
const HINT="<p class='meta'>Click a hole to see how the selected method measured it.</p>";
const VIRIDIS=[[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],[31,158,137],[53,183,121],[110,206,88],[181,222,43],[253,231,37]];
function viridis(t){t=Math.min(1,Math.max(0,t))*9;const i=Math.min(8,Math.floor(t)),f=t-i,a=VIRIDIS[i],b=VIRIDIS[i+1];
  return `rgb(${a.map((v,k)=>Math.round(v+(b[k]-v)*f)).join(",")})`;}
function unpack(s,T){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new T(u.buffer);}
function quantile(sorted,q){return sorted.length?sorted[Math.min(sorted.length-1,Math.floor(q*(sorted.length-1)))]:NaN;}
function fin(v){return Number.isFinite(v);}
const canvas=$("canvas"),ctx=canvas.getContext("2d");
let cur=null,curId=null,img=null,view=null,dims=null,sel=-1,bins=null,range=[0,1];
function prepare(d){if(d.ready)return;const ms=Object.values(d.methods);
  for(const m of ms)if(!m.base){m.x=unpack(m.x,Float32Array);m.y=unpack(m.y,Float32Array);m.valid=unpack(m.valid,Uint8Array);
    m.status=unpack(m.status,m.status_wide?Uint16Array:Uint8Array);}
  for(const m of ms){m.cd=unpack(m.cd,Float32Array);if(m.base){const b=d.methods[m.base];
    m.x=b.x;m.y=b.y;m.valid=b.valid;m.status=b.status;m.status_names=b.status_names;}
    m.sorted=Float32Array.from(m.cd.filter(fin)).sort();}
  d.peak=unpack(d.peak,Float32Array);d.row_bg=unpack(d.row_bg,Float32Array);d.col_bg=unpack(d.col_bg,Float32Array);d.ready=true;}
function show(id){
  document.querySelectorAll(".tab").forEach(t=>t.classList.toggle("active",t.dataset.tab===id));
  document.querySelectorAll(".panel").forEach(p=>p.hidden=p.id!=="panel-"+id&&p.id!=="viewer");
  $("viewer").classList.toggle("on",id!=="summary");if(id==="summary"){cur=null;curId=null;return;}
  const go=()=>{const d=window.CD_DATA[id];prepare(d);cur=d;curId=id;sel=-1;$("vtitle").textContent=ENTRIES[id].name;$("detail").innerHTML=HINT;
    const keep=$("method").value;$("method").innerHTML="";
    for(const [k,m] of Object.entries(d.methods)){const o=document.createElement("option");o.value=k;o.textContent=k+" — "+m.label;$("method").appendChild(o);}
    if(keep&&d.methods[keep])$("method").value=keep;
    if(!$("bandw").value)$("bandw").value=Math.max(1,Math.round(d.pitch||10));
    histograms(id,d);
    img=new Image();img.onload=()=>{
      // Same image size as the previous tab: keep zoom and pan so the same area can be compared.
      if(view&&dims&&dims[0]===d.width&&dims[1]===d.height)sizeCanvas();else fit();
      dims=[d.width,d.height];setMethod();};img.src=ENTRIES[id].image;};
  if(window.CD_DATA[id])go();else{$("detail").textContent="Loading…";const s=document.createElement("script");s.src=ENTRIES[id].data;s.onload=go;
    s.onerror=()=>$("detail").textContent="Could not load "+ENTRIES[id].data;document.body.appendChild(s);}}
function histograms(id,d){const box=document.querySelector(`#panel-${id} .hists`);if(box.dataset.done)return;box.dataset.done=1;
  for(const [k,m] of Object.entries(d.methods)){const v=m.sorted;if(!v.length)continue;
    const lo=quantile(v,.005),hi=quantile(v,.995),n=40,h=new Array(n).fill(0),w=(hi-lo)/n||1;
    for(const x of v){const i=Math.floor((x-lo)/w);if(i>=0&&i<n)h[i]++;}
    const wrap=document.createElement("div");wrap.className="hist";const c=document.createElement("canvas");c.width=240;c.height=110;
    const label=document.createElement("div");label.textContent=`${k} (${m.unit||d.unit})`;wrap.append(label,c);box.appendChild(wrap);
    const g=c.getContext("2d"),top=Math.max(...h);g.fillStyle=COLORS[k]||"#888";
    h.forEach((y,i)=>g.fillRect(i*6,96-y/top*90,5,y/top*90));g.fillStyle=getComputedStyle(document.body).color;g.font="10px sans-serif";
    const sc=m.scale??d.scale;g.fillText((lo*sc).toPrecision(4),0,108);const t=(hi*sc).toPrecision(4);g.fillText(t,240-g.measureText(t).width,108);}}
function setMethod(){if(!cur)return;const k=$("method").value,m=cur.methods[k],v=m.sorted;
  const bm=cur.methods[m.base||k];$("outline-note").textContent=bm.outline_kind==="equivalent_circle"?
    "Zoomed in: equivalent-CD circles; measured outlines were not stored.":"Zoomed in: measured outlines, coloured by size.";
  range=$("scale").value==="shared"&&RANGES[k]?RANGES[k].slice():[quantile(v,.02),quantile(v,.98)];
  if(!(range[1]>range[0]))range=[range[0]-0.5,range[0]+0.5];
  // Colour bins: one fillStyle per bin keeps 1e5 holes fast to draw.
  bins=Array.from({length:32},()=>[]);for(let i=0;i<cur.n;i++)if(m.valid[i]&&fin(m.cd[i])){
    const b=Math.min(31,Math.max(0,Math.floor((m.cd[i]-range[0])/(range[1]-range[0])*32)));bins[b].push(i);}
  const sc=m.scale??cur.scale;$("lo").textContent=(range[0]*sc).toPrecision(4);$("hi").textContent=(range[1]*sc).toPrecision(4)+" "+(m.unit||cur.unit);
  $("ramp").style.background="linear-gradient(to right,"+[0,.25,.5,.75,1].map(viridis).join(",")+")";draw();bandPlot();if(sel>=0)renderDetail();}
function sizeCanvas(){const r=canvas.getBoundingClientRect();canvas.width=r.width*devicePixelRatio;canvas.height=r.height*devicePixelRatio;}
function fit(){sizeCanvas();const k=Math.min(canvas.width/cur.width,canvas.height/cur.height);
  view={k,tx:(canvas.width-k*cur.width)/2,ty:(canvas.height-k*cur.height)/2};draw();}
function draw(){if(!cur||!img||!view)return;ctx.setTransform(1,0,0,1,0,0);ctx.fillStyle="#000";ctx.fillRect(0,0,canvas.width,canvas.height);
  const {k,tx,ty}=view;ctx.setTransform(k,0,0,k,tx,ty);ctx.imageSmoothingEnabled=k<2;ctx.drawImage(img,0,0);
  if(!$("show").checked||!bins)return;const m=cur.methods[$("method").value],bn=m.base||$("method").value,bm=cur.methods[bn];
  const x0=-tx/k-20,y0=-ty/k-20,x1=(canvas.width-tx)/k+20,y1=(canvas.height-ty)/k+20,dots=k*4<6;
  ctx.lineWidth=1.5/k;if(!dots)loadRays(bn);const R=bm.rays,nr=bm.ray_n,u=bm.ray_unit;
  bins.forEach((list,b)=>{const col=viridis((b+.5)/32);ctx.fillStyle=col;ctx.strokeStyle=col;if(!dots)ctx.beginPath();
    for(const i of list){const x=m.x[i]+.5,y=m.y[i]+.5;if(x<x0||x>x1||y<y0||y>y1)continue;
       if(dots){const s=Math.max(1.6/k,Math.min(bm.cd[i],3));ctx.fillRect(x-s/2,y-s/2,s,s);}
      else if(R&&nr>0&&R[i*nr]!==undefined&&R[i*nr]!==255){for(let j=0;j<=nr;j++){const a=RAY_ANG[nr][j%nr],r=R[i*nr+j%nr]*u;
        if(j)ctx.lineTo(x+r*a[0],y+r*a[1]);else ctx.moveTo(x+r*a[0],y+r*a[1]);}}
       else{ctx.moveTo(x+bm.cd[i]/2,y);ctx.arc(x,y,bm.cd[i]/2,0,2*Math.PI);}}
    if(!dots)ctx.stroke();});
  if(!dots){ctx.strokeStyle="#9a9a9a";ctx.beginPath();const c=3;
    for(let i=0;i<cur.n;i++){if(m.valid[i]&&fin(m.cd[i]))continue;const x=m.x[i]+.5,y=m.y[i]+.5;if(x<x0||x>x1||y<y0||y>y1)continue;
      ctx.moveTo(x-c,y-c);ctx.lineTo(x+c,y+c);ctx.moveTo(x-c,y+c);ctx.lineTo(x+c,y-c);}ctx.stroke();}
  if(sel>=0){const x=m.x[sel]+.5,y=m.y[sel]+.5;ctx.strokeStyle="#ff3b3b";ctx.lineWidth=2.5/k;ctx.beginPath();
    ctx.arc(x,y,Math.max(fin(bm.cd[sel])?bm.cd[sel]/2+1:4,8/k),0,2*Math.PI);ctx.stroke();}}
// Measured outlines: one file per method and image, loaded once the map is zoomed in far enough to draw them.
const RAY_ANG={};
function loadRays(name){const bm=cur.methods[name];if(!bm||bm.rays||bm.raysLoading||!bm.rays_file)return;bm.raysLoading=true;
  const id=curId,s=document.createElement("script");s.src=bm.rays_file;
  s.onload=()=>{const F=window.CD_RAYS[id+"/"+name];if(!F)return;
    if(!(F.n>0)){bm.raysLoading=false;bm.rays_file=null;return;}
    bm.rays=unpack(F.rays,Uint8Array);bm.ray_n=F.n;bm.ray_unit=F.unit;
    if(!RAY_ANG[F.n])RAY_ANG[F.n]=[...Array(F.n)].map((_,j)=>[Math.cos(2*Math.PI*j/F.n),Math.sin(2*Math.PI*j/F.n)]);if(curId===id)draw();};
  s.onerror=()=>{bm.raysLoading=false;bm.rays_file=null;};document.body.appendChild(s);}
function pick(px,py){const m=cur.methods[$("method").value],x=(px-view.tx)/view.k-.5,y=(py-view.ty)/view.k-.5;
  let best=-1,bd=Infinity;for(let i=0;i<cur.n;i++){const d=(m.x[i]-x)**2+(m.y[i]-y)**2;if(d<bd){bd=d;best=i;}}
  if(best<0||bd>36)return;sel=best;renderDetail();draw();}

// ---- small plotting helpers ----
function ticks(a,b,n){const raw=(b-a)/n;if(!(raw>0))return[a];const mag=10**Math.floor(Math.log10(raw)),e=raw/mag;
  const step=(e>=5?10:e>=2?5:e>=1.5?2:1)*mag,out=[];for(let t=Math.ceil(a/step)*step;t<=b+step*1e-6;t+=step)out.push(Math.abs(t)<step*1e-9?0:t);return out;}
function tick(t){return Math.abs(t)>=1e4||(Math.abs(t)<1e-3&&t!==0)?t.toExponential(1):String(+t.toPrecision(4));}
function plot(cv,o){const dpr=devicePixelRatio||1,W=cv.clientWidth||300,H=cv.clientHeight||120;cv.width=W*dpr;cv.height=H*dpr;
  const g=cv.getContext("2d");g.setTransform(dpr,0,0,dpr,0,0);const fg=getComputedStyle(document.body).color,L=50,R=10,T=16,B=o.xl?30:18;
  let [x0,x1]=o.x,[y0,y1]=o.y;if(!(x1>x0)){x0-=1;x1+=1;}if(!(y1>y0)){y0-=1;y1+=1;}const py=(y1-y0)*.08;y0-=py;y1+=py;
  const X=x=>L+(x-x0)/(x1-x0)*(W-L-R),Y=y=>H-B-(y-y0)/(y1-y0)*(H-T-B);
  g.clearRect(0,0,W,H);g.font="10px sans-serif";g.fillStyle=fg;g.strokeStyle=fg;g.lineWidth=1;g.globalAlpha=.35;g.strokeRect(L,T,W-L-R,H-T-B);g.globalAlpha=1;
  for(const t of ticks(x0,x1,6)){const s=tick(t);g.fillText(s,X(t)-g.measureText(s).width/2,H-B+11);}
  for(const t of ticks(y0,y1,3)){const s=tick(t);g.fillText(s,L-4-g.measureText(s).width,Y(t)+3);}
  if(o.title)g.fillText(o.title,L,11);if(o.xl)g.fillText(o.xl,(L+W-R)/2-g.measureText(o.xl).width/2,H-3);
  g.save();g.beginPath();g.rect(L,T,W-L-R,H-T-B);g.clip();
  for(const h of o.h||[]){g.strokeStyle=h.color;g.setLineDash([4,3]);g.beginPath();g.moveTo(L,Y(h.y));g.lineTo(W-R,Y(h.y));g.stroke();}
  for(const v of o.v||[]){g.strokeStyle=v.color;g.setLineDash([4,3]);g.beginPath();g.moveTo(X(v.x),T);g.lineTo(X(v.x),H-B);g.stroke();}
  g.setLineDash([]);
  for(const s of o.series||[]){g.strokeStyle=g.fillStyle=s.color||fg;g.lineWidth=s.width||1.5;
    if(s.dots){for(let i=0;i<s.xs.length;i++){if(!fin(s.ys[i]))continue;g.beginPath();g.arc(X(s.xs[i]),Y(s.ys[i]),s.r||2.5,0,7);g.fill();}}
    else{g.beginPath();let on=false;for(let i=0;i<s.xs.length;i++){if(!fin(s.ys[i])){on=false;continue;}
      if(on)g.lineTo(X(s.xs[i]),Y(s.ys[i]));else g.moveTo(X(s.xs[i]),Y(s.ys[i]));on=true;}g.stroke();}}
  g.restore();
  for(const v of o.v||[])if(v.label){g.fillStyle=v.color;g.fillText(v.label,Math.min(X(v.x)+3,W-R-g.measureText(v.label).width),T+10);}
  for(const h of o.h||[])if(h.label){g.fillStyle=h.color;g.fillText(h.label,W-R-g.measureText(h.label).width-2,Y(h.y)-3);}}
function extent(...arrays){let lo=Infinity,hi=-Infinity;for(const a of arrays)for(const v of a)if(fin(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);}return[lo,hi];}
function solve(A,b){const n=b.length;A=A.map(r=>Float64Array.from(r));b=Float64Array.from(b);
  for(let c=0;c<n;c++){let p=c;for(let r=c+1;r<n;r++)if(Math.abs(A[r][c])>Math.abs(A[p][c]))p=r;[A[c],A[p]]=[A[p],A[c]];[b[c],b[p]]=[b[p],b[c]];
    if(Math.abs(A[c][c])<1e-300)return null;for(let r=c+1;r<n;r++){const f=A[r][c]/A[c][c];for(let k=c;k<n;k++)A[r][k]-=f*A[c][k];b[r]-=f*b[c];}}
  const x=new Float64Array(n);for(let r=n-1;r>=0;r--){let s=b[r];for(let k=r+1;k<n;k++)s-=A[r][k]*x[k];x[r]=s/A[r][r];}return x;}
// Cubic spline through y at x = 0..n-1 with scipy's default ("not-a-knot") ends, so the curve is the method's.
function spline(y){const n=y.length;if(n<4)return null;const A=[...Array(n)].map(()=>new Float64Array(n)),b=new Float64Array(n);
  A[0][0]=1;A[0][1]=-2;A[0][2]=1;A[n-1][n-3]=1;A[n-1][n-2]=-2;A[n-1][n-1]=1;
  for(let i=1;i<n-1;i++){A[i][i-1]=1;A[i][i]=4;A[i][i+1]=1;b[i]=6*(y[i+1]-2*y[i]+y[i-1]);}
  const M=solve(A,b);if(!M)return null;
  return x=>{const i=Math.min(n-2,Math.max(0,Math.floor(x))),t=x-i,a=y[i]-M[i]/6,c=y[i+1]-M[i+1]/6;
    return[M[i]*(1-t)**3/6+M[i+1]*t**3/6+a*(1-t)+c*t,-M[i]*(1-t)**2/2+M[i+1]*t*t/2-a+c];};}
function r2(y,cols){const n=y.length,k=cols.length+1;if(n<=k)return NaN;const A=[...Array(k)].map(()=>new Float64Array(k)),b=new Float64Array(k);
  for(let i=0;i<n;i++){const row=[1,...cols.map(c=>c[i])];for(let p=0;p<k;p++){b[p]+=row[p]*y[i];for(let q=0;q<k;q++)A[p][q]+=row[p]*row[q];}}
  const w=solve(A,b);if(!w)return NaN;const mean=y.reduce((s,v)=>s+v,0)/n;let res=0,tot=0;
  for(let i=0;i<n;i++){const f=w[0]+cols.reduce((s,c,j)=>s+w[j+1]*c[i],0);res+=(y[i]-f)**2;tot+=(y[i]-mean)**2;}return tot>0?1-res/tot:NaN;}

// ---- band profile ----
function bands(m,ax,W,size){const pos=ax==="y"?m.y:m.x,nb=Math.ceil(size/W);
  const n=new Float64Array(nb),cd=new Float64Array(nb),pk=new Float64Array(nb),cs=new Float64Array(nb),sn=new Float64Array(nb);
  for(let i=0;i<cur.n;i++){const v=m.cd[i];if(!m.valid[i]||!fin(v))continue;const p=pos[i],b=Math.floor(p/W);if(b<0||b>=nb)continue;
    const f=p-Math.floor(p);n[b]++;cd[b]+=v;pk[b]+=cur.peak[i];cs[b]+=Math.cos(2*Math.PI*f);sn[b]+=Math.sin(2*Math.PI*f);}
  const out={B:[],X:[],CD:[],PK:[],C:[],S:[]};
  for(let b=0;b<nb;b++){if(n[b]<3)continue;out.B.push(b);out.X.push((b+.5)*W);out.CD.push(cd[b]/n[b]*cur.scale);out.PK.push(pk[b]/n[b]);
    out.C.push(cs[b]/n[b]);out.S.push(sn[b]/n[b]);}return out;}
function bandPlot(){if(!cur)return;const k=$("method").value,ax=$("axis").value,W=Math.max(1,Math.round(+$("bandw").value||10));
  const size=ax==="y"?cur.height:cur.width,line=ax==="y"?cur.row_bg:cur.col_bg;
  const bandBg=b=>{let s=0,c=0;for(let j=b*W;j<Math.min((b+1)*W,size);j++){s+=line[j];c++;}return s/c;};
  const pct=v=>fin(v)?(100*v).toFixed(0)+" %":"–",series=[];let rows="",nb=0,chosen=null;
  for(const [kk,mm] of Object.entries(cur.methods)){if(mm.unit)continue;const r=bands(mm,ax,W,size);if(r.CD.length<4)continue;
    const mean=r.CD.reduce((a,v)=>a+v,0)/r.CD.length,dev=r.CD.map(v=>v-mean),sd=Math.sqrt(dev.reduce((a,v)=>a+v*v,0)/(dev.length-1));
    const BG=r.B.map(bandBg);nb=Math.max(nb,r.CD.length);if(kk===k)chosen=r;
    series.push({xs:r.X,ys:dev,color:COLORS[kk]||"#888",width:kk===k?2.8:1.2});
    rows+=`<tr${kk===k?" style='font-weight:600'":""}><td><span class="swatch" style="background:${COLORS[kk]||"#888"}"></span>${kk}</td><td>${sd.toFixed(3)}</td>`+
      `<td>${pct(r2(r.CD,[BG]))}</td><td>${pct(r2(r.CD,[r.PK]))}</td><td>${pct(r2(r.CD,[r.C,r.S]))}</td></tr>`;}
  const xr=[0,size],xl=ax==="y"?"y (px)":"x (px)";
  plot($("band-cd"),{x:xr,y:extent(...series.map(s=>s.ys)),series,h:[{y:0,color:"#888"}],title:`band mean CD minus that map's own mean (${cur.unit}); bold: the map selected above`});
  const bx=[],by=[];if(chosen)for(const b of chosen.B){bx.push((b+.5)*W);by.push(bandBg(b));}
  plot($("band-br"),{x:xr,y:extent(by),series:[{xs:bx,ys:by,color:"#d08a00",width:2}],
    title:`background brightness per band (25th percentile of each pixel ${ax==="y"?"row":"column"}: the absorber between holes)`,xl});
  $("bandinfo").innerHTML=`<table class="stats"><thead><tr><th>Map</th><th>Band-to-band spread (${cur.unit})</th><th>R² with background</th><th>R² with ${cur.peak_label||"spot peak"}</th><th>R² with sub-pixel position</th></tr></thead><tbody>${rows}</tbody></table>`+
    `<p class="meta">Band-to-band spread: standard deviation of the band means. R²: share of that band-to-band variation that goes along with the quantity (0–100 %). With ${nb} bands, chance alone gives about ${pct(1/Math.max(nb-1,1))} (sub-pixel position, fitted with two terms: ${pct(2/Math.max(nb-1,1))}).</p>`;}

// ---- hole detail ----
let detailLoading=null;
function loadTail(id){const s=document.createElement("script");s.src=ENTRIES[id].tail;
  s.onload=()=>{if(curId===id&&sel>=0)renderDetail();};document.body.appendChild(s);}
// Dashed line along the edge of the hole's ordinary cell, shifted into another window by (dx, dy).
function cellLine(D,i,dx,dy){const n=D.h*D.w,c=D.clean.codes,inC=(r,q)=>r>=0&&q>=0&&r<D.h&&q<D.w&&c[i*n+r*D.w+q]!==255;
  return(g,s)=>{g.save();g.setLineDash([4,3]);g.strokeStyle="#7fd3ff";g.lineWidth=1.5;g.beginPath();
    for(let r=0;r<D.h;r++)for(let q=0;q<D.w;q++){if(!inC(r,q))continue;const x=(q+dx)*s,y=(r+dy)*s;
      if(!inC(r-1,q)){g.moveTo(x,y);g.lineTo(x+s,y);}if(!inC(r+1,q)){g.moveTo(x,y+s);g.lineTo(x+s,y+s);}
      if(!inC(r,q-1)){g.moveTo(x,y);g.lineTo(x,y+s);}if(!inC(r,q+1)){g.moveTo(x+s,y);g.lineTo(x+s,y+s);}}
    g.stroke();g.restore();};}
function loadDetail(id){if(detailLoading===id)return;detailLoading=id;const s=document.createElement("script");s.src=ENTRIES[id].detail;
  s.onload=()=>{detailLoading=null;if(curId===id&&sel>=0)renderDetail();};
  s.onerror=()=>{detailLoading=null;$("detail").insertAdjacentHTML("beforeend",`<p class='error'>Could not load ${ENTRIES[id].detail}</p>`);};document.body.appendChild(s);}
function prepareDetail(D){if(D.ready)return;D.ox=unpack(D.ox,Int32Array);D.oy=unpack(D.oy,Int32Array);
  const C=D.clean;C.codes=unpack(C.codes,Uint8Array);C.lo=unpack(C.lo,Float32Array);C.step=unpack(C.step,Float32Array);
  if(D.prof_x){D.prof_x=unpack(D.prof_x,Float32Array);D.prof_y=unpack(D.prof_y,Float32Array);}
  for(const F of Object.values(D.half||{})){for(const k of ["level","peak","threshold","mean","cx","cy"])F[k]=unpack(F[k],Float32Array);F.rays=unpack(F.rays,Uint8Array);}D.ready=true;}
function cellValues(D,part,i){const n=D.h*D.w,out=new Float32Array(n);for(let p=0;p<n;p++){const c=part.codes[i*n+p];out[p]=c===255?NaN:part.lo[i]+c*part.step[i];}return out;}
function drawCell(id,D,vals,overlay,diverging){const cv=$(id),s=Math.max(6,Math.floor(170/Math.max(D.h,D.w)));cv.width=D.w*s;cv.height=D.h*s;
  const g=cv.getContext("2d");let [lo,hi]=extent(vals);const a=Math.max(Math.abs(lo),Math.abs(hi))||1;
  for(let r=0;r<D.h;r++)for(let c=0;c<D.w;c++){const v=vals[r*D.w+c];let col="#2a3340";
    if(fin(v)){if(diverging){const t=Math.max(-1,Math.min(1,v/a)),u=Math.round(255*(1-Math.abs(t)));col=t>0?`rgb(255,${u},${u})`:`rgb(${u},${u},255)`;}
      else{const u=Math.round(255*(v-lo)/((hi-lo)||1));col=`rgb(${u},${u},${u})`;}}
    g.fillStyle=col;g.fillRect(c*s,r*s,s,s);}
  if(overlay){g.lineWidth=2;overlay(g,s);}}
function outline(cx,cy,rx,ry,color){return(g,s)=>{if(!fin(rx)||!fin(ry))return;g.strokeStyle=color;g.beginPath();
  g.ellipse((cx+.5)*s,(cy+.5)*s,rx*s,ry*s,0,0,2*Math.PI);g.stroke();g.fillStyle=color;g.fillRect((cx+.5)*s-2,(cy+.5)*s-2,4,4);};}
function profilePlots(prefix,prof,edges,label){const sp=spline(prof),xs=[],vs=[],ss=[];const n=prof.length;
  if(sp)for(let x=0;x<=n-1+1e-9;x+=0.02){const [v,s]=sp(x);xs.push(x);vs.push(v);ss.push(s);}
  const idx=[...prof.keys()],w=edges[1]-edges[0],v=[{x:edges[0],color:"#e03b3b",label:"steepest rise"},{x:edges[1],color:"#e03b3b",label:"steepest fall"}];
  plot($(prefix+"-v"),{x:[0,n-1],y:extent(prof,vs),series:[{xs,ys:vs,color:"#7a7a7a"},{xs:idx,ys:prof,dots:true,color:getComputedStyle(document.body).color}],v,
    title:`${label}: width = ${fin(w)?w.toFixed(3)+" px = "+(w*cur.scale).toFixed(2)+" "+cur.unit:"–"}`});
  plot($(prefix+"-s"),{x:[0,n-1],y:extent(ss,[0]),series:[{xs,ys:ss,color:"#3a8fd8"}],v:v.map(e=>({x:e.x,color:e.color})),h:[{y:0,color:"#888"}],
    title:"slope of the curve (edges = its highest and lowest point)",xl:"pixel in the cell window"});}
function renderDetail(){const i=sel;if(i<0||!cur)return;const k=$("method").value,m=cur.methods[k],base=m.base||k;
  let rows="";for(const [kk,mm] of Object.entries(cur.methods)){const v=mm.cd[i];
    rows+=`<tr><td><span class="swatch" style="background:${COLORS[kk]||"#888"}"></span>${kk}</td><td>${fin(v)?(v*(mm.scale??cur.scale)).toFixed(3)+" "+(mm.unit||cur.unit):"–"}</td><td>${mm.status_names[mm.status[i]]}</td></tr>`;}
  let out=`<b>Hole ${i}</b> at x ${m.x[i].toFixed(2)}, y ${m.y[i].toFixed(2)} px<table>${rows}</table>`;
  const D=window.CD_DETAIL[curId];
  if(!ENTRIES[curId].detail){$("detail").innerHTML=out+"<p class='meta'>Cell images were not stored. Per-hole values and statuses are shown above.</p>";return;}
  if(!D){$("detail").innerHTML=out+"<p class='meta'>Loading hole details…</p>";loadDetail(curId);return;}
  prepareDetail(D);const ox=D.ox[i],oy=D.oy[i],cx=m.x[i]-ox,cy=m.y[i]-oy,cdv=m.cd[i];
  if(base==="projection"){const P=cur.methods.projection,px=cur.methods.projection_x.cd[i],py=cur.methods.projection_y.cd[i];
    const pcx=P.x[i]-ox,pcy=P.y[i]-oy,vals=cellValues(D,D.clean,i);
    const X=Array.from(D.prof_x.subarray(i*D.w,(i+1)*D.w)),Y=Array.from(D.prof_y.subarray(i*D.h,(i+1)*D.h));
    out+=`<h3>projection</h3><p class="explain">The cell with background and neighbours removed. Each column is summed into the x profile, each row into the y profile; a smooth curve (cubic spline) goes exactly through the sums, and the width is the distance between its steepest rise and steepest fall (red). Cell outline: dark pixels are outside the cell. The dots are rarely symmetric even for a symmetric feature: its centre falls between pixels, so the two sides are sampled at different points; the ragged cell outline changes how many pixels each sum holds; and noise moves the few dots near each edge. The curve follows the dots exactly, so all three move the edges.</p>
      <div class="cells"><figure><canvas id="dc-a"></canvas>cell and the x/y widths</figure></div>
      <canvas class="plot" id="pp-x-v" style="height:150px"></canvas><canvas class="plot" id="pp-x-s" style="height:90px"></canvas>
      <canvas class="plot" id="pp-y-v" style="height:150px"></canvas><canvas class="plot" id="pp-y-s" style="height:90px"></canvas>`;
    $("detail").innerHTML=out;drawCell("dc-a",D,vals,outline(pcx,pcy,px/2,py/2,"#e03b3b"));
    profilePlots("pp-x",X,[pcx-px/2,pcx+px/2],"x profile (column sums)");profilePlots("pp-y",Y,[pcy-py/2,pcy+py/2],"y profile (row sums)");return;}
  if(D.half&&D.half[base]){const F=D.half[base],b=F.level[i],pk=F.peak[i],t=F.threshold[i];let G=D,vals=cellValues(D,D.clean,i),hx=F.cx[i],hy=F.cy[i],old=null;
    if((base.startsWith("tail_area")||base==="edge_area"||base.startsWith("foot_area"))&&ENTRIES[curId].tail){const T=window.CD_TAIL[curId];
      if(!T){$("detail").innerHTML=out+"<p class='meta'>Loading the wider window…</p>";loadTail(curId);return;}
      if(!T.ready){T.ox=unpack(T.ox,Int32Array);T.oy=unpack(T.oy,Int32Array);const C=T.cells;C.codes=unpack(C.codes,Uint8Array);
        C.lo=unpack(C.lo,Float32Array);C.step=unpack(C.step,Float32Array);T.ready=true;}
      const dx=D.ox[i]-T.ox[i],dy=D.oy[i]-T.oy[i];G=T;vals=cellValues(T,T.cells,i);hx+=dx;hy+=dy;old=cellLine(D,i,dx,dy);}
    const above=vals.map(v=>fin(v)?(v>t?1:-1):NaN),nr=F.n_rays,pts=[];
    for(let r=0;r<nr;r++){const a=2*Math.PI*r/nr,q=F.rays[i*nr+r]/16;pts.push([hx+q*Math.cos(a),hy+q*Math.sin(a)]);}
    const poly=(g,s)=>{if(old)old(g,s);if(!fin(hx))return;g.strokeStyle=COLORS[base];g.beginPath();pts.forEach(([x,y],j)=>j?g.lineTo((x+.5)*s,(y+.5)*s):g.moveTo((x+.5)*s,(y+.5)*s));g.closePath();g.stroke();};
    const wideNote=old?"<p class='meta'>Shown: the wider window this method measures on, holding this hole's share of the light (each pixel's light shared among the holes in proportion to their fitted models). Dashed: the hole's ordinary cell, which the other methods are limited to; the region may reach past it.</p>":"";
    out+=`<h3>${base}</h3><p class="explain">The cell with background and neighbours removed. Its local level b is the ${F.background}; its peak is the ${F.peak_rule}. The cell is enlarged 4× by smooth (cubic spline) interpolation; the region is the connected area around the peak brighter than the threshold ${F.threshold_rule?"= the "+F.threshold_rule+"; the peak is not used":"b + "+F.level_fraction+" × (peak − b)"}. CD = diameter of a circle with the region's area. If a pulse brightens the feature and its surroundings by the same factor, b, the peak and the threshold scale together and the area stays the same.</p>
      ${wideNote}<div class="cells"><figure><canvas id="dc-a"></canvas>${old?"wider window":"cell"} and region outline</figure><figure><canvas id="dc-b"></canvas>original pixels above the threshold (red)</figure></div>
      <table><tr><td>local level b</td><td>${fin(b)?b.toPrecision(4):"–"}</td></tr><tr><td>peak height above b (${F.peak_rule})</td><td>${fin(pk)?pk.toPrecision(4):"–"}</td></tr>
      <tr><td>threshold = ${F.threshold_rule?(base.startsWith("foot_area")?"brightness at the foot of the edge":"steepness-weighted mean brightness"):"b + "+F.level_fraction+" × height"}</td><td>${fin(t)?t.toPrecision(4):"–"}</td></tr>
      <tr><td>mean light above b in the region${base==="tail_area"?" (cell intensity)":""}</td><td>${fin(F.mean[i])?F.mean[i].toPrecision(4):"–"}</td></tr>
      <tr><td>region area</td><td>${fin(cdv)?(Math.PI*cdv*cdv/4).toFixed(2)+" px²":"–"}</td></tr>
      <tr><td>CD = 2 √(area / π)</td><td>${fin(cdv)?cdv.toFixed(3)+" px = "+(cdv*cur.scale).toFixed(2)+" "+cur.unit:"–"}</td></tr></table>
      <p class="meta">The outline is drawn from 32 directions around the region's centre, so it is approximate for very irregular shapes; the area itself is counted pixel by pixel on the enlarged cell.</p>`;
    $("detail").innerHTML=out;drawCell("dc-a",G,vals,poly);drawCell("dc-b",G,above,poly,true);return;}
  const vals=cellValues(D,D.clean,i);
  out+=`<h3>${k}</h3><p class="explain">The cell this method measured (background and neighbours removed; dark pixels are outside the cell), with its outline at the CD.</p><div class="cells"><figure><canvas id="dc-a"></canvas>cell and outline</figure></div>`;
  $("detail").innerHTML=out;drawCell("dc-a",D,vals,outline(cx,cy,cdv/2,cdv/2,COLORS[k]||"#e03b3b"));}

let drag=null;
canvas.addEventListener("pointerdown",e=>{drag={x:e.clientX,y:e.clientY,tx:view.tx,ty:view.ty,moved:false};canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener("pointermove",e=>{if(!drag)return;const dx=(e.clientX-drag.x)*devicePixelRatio,dy=(e.clientY-drag.y)*devicePixelRatio;
  if(Math.abs(dx)+Math.abs(dy)>4)drag.moved=true;view.tx=drag.tx+dx;view.ty=drag.ty+dy;draw();});
canvas.addEventListener("pointerup",e=>{if(drag&&!drag.moved){const r=canvas.getBoundingClientRect();pick((e.clientX-r.left)*devicePixelRatio,(e.clientY-r.top)*devicePixelRatio);}drag=null;});
canvas.addEventListener("wheel",e=>{e.preventDefault();const r=canvas.getBoundingClientRect(),px=(e.clientX-r.left)*devicePixelRatio,py=(e.clientY-r.top)*devicePixelRatio;
  const f=Math.exp(-e.deltaY*0.0015),k=Math.min(80,Math.max(0.02,view.k*f)),g=k/view.k;view.tx=px-(px-view.tx)*g;view.ty=py-(py-view.ty)*g;view.k=k;draw();},{passive:false});
$("method").onchange=setMethod;$("scale").onchange=setMethod;$("show").onchange=draw;$("fit").onclick=fit;
$("axis").onchange=bandPlot;$("bandw").onchange=bandPlot;
window.addEventListener("resize",()=>{if(cur){sizeCanvas();draw();bandPlot();if(sel>=0)renderDetail();}});
document.querySelectorAll("[data-tab]").forEach(t=>t.addEventListener("click",e=>{e.preventDefault();show(t.dataset.tab);window.scrollTo(0,0);}));
</script></body></html>
"""


if __name__ == "__main__":
    main()
