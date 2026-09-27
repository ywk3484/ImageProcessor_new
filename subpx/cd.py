"""Contact-hole contours, critical dimensions, spatial maps, and raw exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .types import CDResult


METHOD_LABELS = {
    "logquad": "Gaussian FWHM",
    "gradient": "Maximum radial gradient",
    "halfmax": "Half-height contour",
}


def measure_cds(
    image: np.ndarray,
    *,
    methods: tuple[str, ...] = ("logquad", "gradient", "halfmax"),
    threshold: str = "otsu",
    invert: bool = False,
    area_min: int = 3,
    area_max: int = 50,
    pad: int = 3,
    n_angles: int = 128,
    radial_step: float = 0.1,
    smooth_sigma: float = 0.0,
    fit_fraction: float = 0.2,
    pixel_size: float = 1.0,
    unit: str = "px",
    cell_backend: str = "cpu",
    device: int = 0,
) -> CDResult:
    """Extract Voronoi cells and measure their contours and critical dimensions.

    Parameters
    ----------
    image : (H, W) ndarray
        Finite grayscale image. Floating-point images are scaled to uint8 for
        segmentation only; measurements retain the original intensities.
    methods : sequence of str
        ``logquad`` fits a Gaussian and returns its half-maximum ellipse;
        ``gradient`` connects radial gradient maxima; ``halfmax`` connects
        half-height crossings. These are different edge definitions.
    threshold : {'otsu', 'triangle'}
        Same segmentation as the center diagnostic.
    invert : bool
        Set True for dark holes on a bright background.
    area_min, area_max : int
        Inclusive connected-component area limits, in pixels.
    pad : int
        Padding around component bounding boxes before Voronoi masking.
    n_angles : int
        Number of contour samples; closure adds one repeated vertex.
    radial_step : float
        Ray sampling interval in original pixels (cubic interpolation).
    smooth_sigma : float
        Optional Gaussian smoothing of radial profiles' source image, in
        pixels. Broadens measured edges; does not affect the Gaussian fit.
    fit_fraction : float
        Fit pixels above this fraction of the background-subtracted peak.
    pixel_size : float
        Isotropic length per pixel, used by plots and calibrated CSV columns.
    unit : str
        Length unit, e.g. 'px' or 'nm'. Array metrics remain in pixels.
    cell_backend : {'cpu', 'gpu'}
        Backend for the Voronoi partition only. Fits run on the CPU.
    device : int
        CUDA device when cell_backend='gpu'.

    Returns
    -------
    CDResult
        Cell arrays and per-method measurements, aligned by zero-based cell
        ID. Failed cells remain present with a status and NaN geometry.
        Contours and centers are global [x, y] pixel coordinates.
    """
    from ._cpu.centers import _segment_binary
    from ._cpu.components import connected_components_stats_cpu
    from ._cpu.cells import _extract_voronoi_rois, compute_voronoi_labels_cpu
    from ._cpu.cd import measure_cells

    img = np.asarray(image)
    if img.ndim != 2 or img.size == 0 or not np.issubdtype(img.dtype, np.number):
        raise ValueError("image must be a nonempty 2D real grayscale array")
    if np.iscomplexobj(img) or not np.isfinite(img).all():
        raise ValueError("image must contain finite real intensities")
    if isinstance(methods, str):
        methods = (methods,)
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or any(m not in METHOD_LABELS for m in methods):
        raise ValueError("methods must be unique choices from logquad, gradient, halfmax")
    for name, value, lower in (("area_min", area_min, 1), ("area_max", area_max, area_min),
                               ("pad", pad, 0), ("n_angles", n_angles, 16)):
        if not np.isfinite(value) or int(value) != value or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    if not np.isfinite([radial_step, smooth_sigma, fit_fraction, pixel_size]).all():
        raise ValueError("measurement parameters must be finite")
    if not 0 < radial_step <= 1 or smooth_sigma < 0 or not 0 < fit_fraction < 1 or pixel_size <= 0:
        raise ValueError("require 0 < radial_step <= 1, smooth_sigma >= 0, 0 < fit_fraction < 1, pixel_size > 0")
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("unit must be a nonempty length-unit label")
    if unit == "px" and pixel_size != 1:
        raise ValueError("supply a physical unit when pixel_size differs from 1")
    if cell_backend not in {"cpu", "gpu"}:
        raise ValueError("cell_backend must be 'cpu' or 'gpu'")

    # Preserve the diagnostic's uint8 segmentation exactly; OpenCV's triangle
    # threshold requires uint8, whereas Otsu also supports uint16.
    segmentation = img
    rescaled = img.dtype != np.uint8 and not (img.dtype == np.uint16 and threshold == "otsu")
    if rescaled:
        gray = img.astype(np.float64)
        span = float(gray.max() - gray.min())
        segmentation = np.rint((gray - gray.min()) * (255 / span)).astype(np.uint8) if span else np.zeros(img.shape, np.uint8)
    _, binary = _segment_binary(segmentation, threshold=threshold, invert=invert)
    comp = connected_components_stats_cpu(binary, connectivity=8)
    height, width = img.shape
    rows, component_ids = [], []
    for label in range(1, comp.num_labels):
        x, y, w, h, area = comp.stats[label]
        if area < area_min or area > area_max:
            continue
        if x <= 0 or y <= 0 or x + w >= width or y + h >= height:
            continue
        rows.append([x, y, w, h, *comp.centroids[label]])
        component_ids.append(label)
    row_array = np.asarray(rows, dtype=float).reshape(-1, 6)
    seeds = row_array[:, 4:6].copy()
    if cell_backend == "gpu":
        from ._gpu.voronoi import compute_voronoi_labels_gpu
        cell_labels = compute_voronoi_labels_gpu(seeds, img.shape, device=device)
    else:
        cell_labels = compute_voronoi_labels_cpu(seeds, img.shape)
    # Negation preserves contrast and permits the same bright-feature methods
    # for dark holes, including a background estimate in the correct polarity.
    signal_image = img.astype(np.float64) * (-1 if invert else 1)
    rois, masks, origins = _extract_voronoi_rois(signal_image, cell_labels, rows, pad=int(pad))
    origins = np.asarray(origins, dtype=np.int32).reshape(-1, 2)
    measurements = measure_cells(
        rois, masks, origins, seeds, methods=methods, n_angles=int(n_angles),
        radial_step=radial_step, smooth_sigma=smooth_sigma, fit_fraction=fit_fraction,
    )
    return CDResult(
        image=img.copy(), binary=binary > 0, component_labels=comp.labels,
        component_ids=np.asarray(component_ids, dtype=np.int32),
        boxes_xywh=row_array[:, :4].astype(np.int32), seeds_xy=seeds,
        cell_labels=cell_labels,
        cell_areas_px2=np.bincount(cell_labels[cell_labels >= 0], minlength=len(rows)),
        origins_xy=origins, rois=rois, masks=masks, measurements=measurements,
        pixel_size=float(pixel_size), unit=unit,
        meta={
            "threshold": threshold, "invert": bool(invert), "area_min": int(area_min),
            "area_max": int(area_max), "pad": int(pad), "n_angles": int(n_angles),
            "radial_step": float(radial_step), "smooth_sigma": float(smooth_sigma),
            "fit_fraction": float(fit_fraction), "cell_backend": cell_backend,
            "measurement_backend": "cpu", "device": int(device),
            "segmentation_rescaled": rescaled, "components_before_filter": comp.num_labels - 1,
            "coordinates": "global [x, y] pixels; images indexed [y, x]",
            "background": "10th percentile of the polarity-normalized cell border",
        },
    )


def plot_cd_map(
    result: CDResult, *, method: str = "logquad", metric: str = "cd_equivalent_px",
    contours: bool = False, ax=None, cmap: str = "viridis", vmin=None, vmax=None,
):
    """Plot a Voronoi CD map or colored contours over the image, with colorbar.

    ``metric`` is cd_equivalent_px, cd_x_px, cd_y_px, area_px2, or mean_intensity.
    Mean intensity uses original pixels in the full Voronoi cell, including
    background, independently of CD validity and physical calibration. Gray
    cells indicate unavailable values. Axes remain in pixel coordinates.
    Returned Matplotlib axes can be zoomed or saved by the caller.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    if metric not in {"cd_equivalent_px", "cd_x_px", "cd_y_px", "area_px2", "mean_intensity"}:
        raise ValueError("metric must be cd_equivalent_px, cd_x_px, cd_y_px, area_px2, or mean_intensity")
    intensity = metric == "mean_intensity"
    power = 2 if metric == "area_px2" else 1
    if intensity:
        values = result.mean_intensity
        valid = np.isfinite(values)
    else:
        measurement = result.measurements[method]
        values = measurement.metrics[metric] * result.pixel_size**power
        valid = measurement.valid & np.isfinite(values)
    if vmin is None:
        vmin = float(values[valid].min()) if valid.any() else 0.0
    if vmax is None:
        vmax = float(values[valid].max()) if valid.any() else 1.0
    if vmax == vmin:
        delta = max(abs(vmin) * 0.01, 0.01)
        vmin, vmax = vmin - delta, vmax + delta
    norm = Normalize(vmin, vmax)
    palette = plt.get_cmap(cmap).with_extremes(bad="#bbbbbb")
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    if contours:
        measurement = result.measurements[method]
        valid = valid & measurement.valid
        ax.imshow(result.image, cmap="gray", interpolation="nearest")
        artist = LineCollection(measurement.contours_xy[valid], cmap=palette, norm=norm, linewidths=0.8)
        artist.set_array(values[valid])
        ax.add_collection(artist)
        if (~valid).any():
            ax.scatter(*result.seeds_xy[~valid].T, marker="x", c="#aaaaaa", s=12, linewidths=0.6)
    else:
        field = np.full(result.image.shape, np.nan)
        assigned = result.cell_labels >= 0
        lookup = np.where(valid, values, np.nan)
        field[assigned] = lookup[result.cell_labels[assigned]]
        artist = ax.imshow(np.ma.masked_invalid(field), cmap=palette, norm=norm, interpolation="nearest")
    label = {"cd_equivalent_px": "Area-equivalent CD", "cd_x_px": "Horizontal CD",
             "cd_y_px": "Vertical CD", "area_px2": "Contour area",
             "mean_intensity": "Mean cell intensity"}[metric]
    units = "original intensity units" if intensity else result.unit + ("²" if power == 2 else "")
    ax.figure.colorbar(artist, ax=ax, label=f"{label} ({units})", pad=0.02)
    title = "Mean cell intensity" if intensity else METHOD_LABELS[method]
    ax.set(title=f"{title} | {valid.sum()}/{len(valid)} valid",
           xlabel="x (px)", ylabel="y (px)",
           xlim=(-0.5, result.image.shape[1] - 0.5), ylim=(result.image.shape[0] - 0.5, -0.5))
    return ax


def save_cd_viewer(result: CDResult, path, *, metric: str = "cd_equivalent_px") -> Path:
    """Write a self-contained, offline HTML viewer with crisp contour zoom.

    Supports pan/zoom, method comparisons, CD and mean-intensity cell maps,
    cell selection, and optional contour vertices. No server or CDN is needed.
    The source raster retains its native resolution; contour paths are redrawn
    at the current zoom. Drawing coordinates use local float32 offsets, while
    CSV/NPZ exports retain the original float64 measurements.
    """
    from ._cd_viewer import write_viewer
    return write_viewer(result, path, metric=metric)


def save_cd_results(result: CDResult, output) -> dict[str, Path]:
    """Save measurements.csv, cells.csv, contours.csv, raw_data.npz and manifest.json.

    CSVs include pixel and calibrated values. NPZ contains the input image,
    cell extraction, all contours, validity/status arrays, and every metric;
    it loads with ``allow_pickle=False``. The JSON manifest records parameters,
    definitions, calibration, and valid/failed counts.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: output / name for name in ("measurements.csv", "cells.csv", "contours.csv", "raw_data.npz", "manifest.json")}
    scale = result.pixel_size
    metric_names = sorted({name for m in result.measurements.values() for name in m.metrics})
    calibrated = {"area_px2": "area", "cd_equivalent_px": "cd_equivalent", "cd_x_px": "cd_x",
                  "cd_y_px": "cd_y", "perimeter_px": "perimeter",
                  "fwhm_major_px": "fwhm_major", "fwhm_minor_px": "fwhm_minor"}
    header = ["cell_id", "component_id", "method", "valid", "status", "seed_x_px", "seed_y_px",
              "center_x_px", "center_y_px", "center_x", "center_y", "unit", "pixel_size", "mean_intensity"]
    with paths["measurements.csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header + metric_names + list(calibrated.values()))
        for method, measured in result.measurements.items():
            for i, seed in enumerate(result.seeds_xy):
                row = [i, int(result.component_ids[i]), method, bool(measured.valid[i]), measured.status[i],
                       *seed, *measured.centers_xy[i], *(measured.centers_xy[i] * scale), result.unit, scale,
                       float(result.mean_intensity[i])]
                row += [float(measured.metrics[name][i]) if name in measured.metrics else np.nan for name in metric_names]
                row += [float(measured.metrics[name][i]) * scale**(2 if name == "area_px2" else 1)
                        if name in measured.metrics and measured.valid[i] else np.nan for name in calibrated]
                writer.writerow(row)
    with paths["cells.csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell_id", "component_id", "seed_x_px", "seed_y_px", "cell_area_px2", "mean_intensity"])
        for i, seed in enumerate(result.seeds_xy):
            writer.writerow([i, int(result.component_ids[i]), *seed,
                             int(result.cell_areas_px2[i]), float(result.mean_intensity[i])])
    with paths["contours.csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell_id", "component_id", "method", "valid", "vertex", "x_px", "y_px", "x", "y", "unit"])
        for method, measured in result.measurements.items():
            for i, contour in enumerate(measured.contours_xy):
                for j, (x, y) in enumerate(contour):
                    writer.writerow([i, int(result.component_ids[i]), method, bool(measured.valid[i]), j,
                                     x, y, x * scale, y * scale, result.unit])
    arrays = {name: getattr(result, name) for name in (
        "image", "binary", "component_labels", "component_ids", "boxes_xywh", "seeds_xy",
        "cell_labels", "cell_areas_px2", "mean_intensity", "origins_xy", "rois", "masks",
    )}
    arrays.update(pixel_size=np.asarray(scale), unit=np.asarray(result.unit),
                  methods=np.asarray(list(result.measurements)))
    summary = {}
    for method, measured in result.measurements.items():
        for name in ("centers_xy", "contours_xy", "valid", "status"):
            arrays[f"{method}_{name}"] = getattr(measured, name)
        for name, values in measured.metrics.items():
            arrays[f"{method}_{name}"] = values
        statuses, counts = np.unique(measured.status, return_counts=True)
        values = measured.metrics["cd_equivalent_px"][measured.valid]
        summary[method] = {
            "valid": int(measured.valid.sum()), "failed": int((~measured.valid).sum()),
            "status_counts": dict(zip(statuses.tolist(), counts.tolist())),
            "median_cd_px": float(np.median(values)) if values.size else None,
            "mean_cd_px": float(values.mean()) if values.size else None,
            "std_cd_px": float(values.std()) if values.size else None,
        }
    np.savez_compressed(paths["raw_data.npz"], **arrays)
    manifest = {
        "schema_version": 2, "shape_hw": list(result.image.shape), "cells": len(result.seeds_xy),
        "pixel_size": scale, "unit": result.unit, "parameters": result.meta,
        "definitions": {
            "logquad": "Half-maximum ellipse of a background-subtracted Gaussian log-quadratic fit; FWHM = 2 sqrt(2 ln 2) sigma.",
            "gradient": "Closed polygon through subpixel maxima of the outward radial intensity decrease; Gaussian diameter is approximately 2 sigma without smoothing.",
            "halfmax": "Closed polygon through first outward crossings at half the interpolated central signal above background.",
            "cd_equivalent": "2 sqrt(contour_area / pi); analytic ellipse area for logquad, polygon area otherwise.",
            "cd_x_cd_y": "Axis-aligned contour extents; analytic ellipse projections for logquad.",
            "failed_cells": "Retained with valid=False, status, and NaN geometric metrics; candidate/partial contours may remain for diagnosis.",
            "calibration": "All NPZ geometry is in pixels. Unsuffixed CSV lengths use unit, areas use unit squared. Cell areas are Voronoi areas, not hole areas.",
            "mean_intensity": "Arithmetic mean of original image pixels over the full Voronoi cell, including background. No padding fill, background subtraction, polarity inversion, or physical-unit scaling; independent of CD fit validity.",
        },
        "methods": summary, "files": [p.name for p in paths.values()],
    }
    paths["manifest.json"].write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return paths
