"""Contact-hole contours, critical dimensions, spatial maps, and raw exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np

from .types import CDMethodResult, CDResult


METHOD_LABELS = {
    "gaussian": "Deblended Gaussian FWHM",
    "logquad": "Gaussian FWHM",
    "gradient": "Maximum radial gradient",
    "halfmax": "Half-height contour",
    "disk": "Blurred-hole fit diameter",
    "flux_ratio": "Hole light / own background",
    "projection": "x/y projection max-slope width",
    "half_area": "Area above half height",
    "half_area_min": "Area above half height (minimum level)",
    "half_area_min_avg": "Area above half height (minimum level, averaged peak)",
    "tail_area": "Area above a low threshold near the background",
    "tail_area_norm": "Area above a fraction of the peak (no background step)",
    "edge_area": "Area inside the steepest edge (no peak, no background step)",
    "foot_area": "Area inside the foot of the edge, a tail boundary (no peak, no background step)",
    "foot_area_ratio": "Area inside the foot of the edge, foot found once per image (no peak, no background step)",
}
# Opt-in methods insensitive to a per-hole source-pulse scale (subpx/_cpu/invariant_cd.py).
INVARIANT_METHODS = ("disk", "flux_ratio", "projection", "half_area", "half_area_min", "half_area_min_avg",
                     "tail_area", "tail_area_norm", "edge_area", "foot_area", "foot_area_ratio")


def measure_cds(
    image: np.ndarray,
    *,
    methods: tuple[str, ...] | None = None,
    background: str = "model",
    seeds: str = "threshold",
    threshold: str = "otsu",
    invert: bool = False,
    area_min: int = 3,
    area_max: int = 50,
    pad: int = 3,
    n_angles: int = 128,
    radial_step: float = 0.1,
    smooth_sigma: float = 0.0,
    fit_fraction: float = 0.2,
    peak_min_distance: float | None = None,
    background_level: float | None = None,
    background_tile: int | None = None,
    deblend_iterations: int = 30,
    pixel_size: float = 1.0,
    unit: str = "px",
    backend: str = "cpu",
    cell_backend: str | None = None,
    device: int = 0,
    nominal_cd: float | None = None,
    psf_fwhm: float | None = None,
    dark_level: float = 0.0,
    flux_ratio_scale: float | None = None,
    half_area_level: float = 0.5,
    half_area_peak_pixels: float = 4.0,
    tail_area_level: float = 0.2,
    tail_window: float = 1.2,
    outlines: bool = True,
) -> CDResult:
    """Extract Voronoi cells and measure their contours and critical dimensions.

    Parameters
    ----------
    image : (H, W) ndarray
        Finite grayscale image. Floating-point images are scaled to uint8 for
        segmentation only; measurements retain the original intensities.
    methods : sequence of str, optional
        ``gaussian`` returns the half-maximum ellipse of the deblended Gaussian
        fit (``background='model'`` only); ``logquad`` fits log(I - background);
        ``gradient`` connects radial gradient maxima; ``halfmax`` connects
        half-height crossings. These are different edge definitions. Default:
        all four for ``background='model'``, the last three for ``'border'``.
        Opt-in, ``background='model'`` only, insensitive to a per-hole
        illumination scale (source-pulse fluctuation): ``disk`` fits a round
        hole of diameter d imaged through a circular-pupil blur and averaged
        over each pixel, with free brightness and local background per hole
        (CD = d, the hole itself, not the spot); ``flux_ratio`` divides the
        hole's light by its own fitted local background minus ``dark_level``
        (CD = 2 sqrt(ratio / pi), proportional to the hole diameter);
        ``projection`` takes the cell's summed x and y profiles and the width
        between each one's steepest rise and fall (CD = mean of both);
        ``half_area`` takes the cell's local level b (median of its outermost
        pixels) and the region around the peak brighter than b plus
        ``half_area_level`` x the peak height above b, on a 4x cubic-spline
        enlargement of the cell (CD = 2 sqrt(area / pi), the equal-area circle;
        cd_x / cd_y = the x / y diameters of the ellipse with the region's
        area and x / y second moments). It assumes no shape.
        ``half_area_min`` is the same with b = the minimum of the outermost
        pixels instead of their median; ``half_area_min_avg`` also takes the
        peak as the mean of the brightest ``half_area_peak_pixels`` pixels'
        worth of the enlarged cell, so one outstanding pixel cannot set it.
        ``tail_area`` uses a low threshold near the background instead, on a
        wider window (``tail_window`` pitches around the hole) holding the
        hole's share of the light (image minus the background map, each pixel
        shared among the holes in proportion to their fitted models), so its
        region may extend past the hole's cell: b = median of the farthest
        tenth of the hole's own territory, averaged peak, threshold b +
        ``tail_area_level`` x the height. Its metric ``mean_signal`` is the
        mean light above b over the region. ``tail_area_norm`` is the same
        without the local level: each window divided by its averaged peak,
        region where the normalized pattern exceeds ``tail_area_level``.
        ``edge_area`` uses neither the peak nor a local level: on the same
        window, threshold = the mean brightness weighted by steepness (where
        the light falls fastest); region = the area above it. ``foot_area``
        is the tail counterpart: threshold = the brightness at the foot of
        the edge, where the falling slope bends into the flat surroundings.
        ``foot_area_ratio`` finds that foot once per image (median ratio of
        foot to steepest level) and sets each hole's threshold at that ratio
        times its own steepest level.
    background : {'model', 'border'}
        ``model`` jointly fits one Gaussian per hole and a smooth background
        map, then measures every cell with the background and all neighbouring
        holes' models removed. Background is taken from pixels far from every
        hole and interpolated across dense arrays (see ``meta['background_source']``).
        ``border`` is the previous per-cell 10th percentile of the Voronoi border,
        biased by noise and neighbour tails; kept to reproduce old results.
    seeds : {'threshold', 'peaks'}
        ``threshold`` uses connected components of the thresholded image;
        ``peaks`` uses local maxima, for arrays too dense to threshold
        (pitch/FWHM below about 1.6). ``peaks`` requires ``background='model'``.
    threshold : {'otsu', 'triangle'}
        Same segmentation as the center diagnostic.
    invert : bool
        Set True for dark holes on a bright background.
    area_min, area_max : int
        Inclusive connected-component area limits, in pixels (threshold seeds).
    pad : int
        Padding around component bounding boxes before Voronoi masking.
    n_angles : int
        Number of contour samples; closure adds one repeated vertex.
    radial_step : float
        Ray sampling interval in original pixels (cubic interpolation).
    smooth_sigma : float
        Optional Gaussian smoothing of radial profiles' source image, in
        pixels. Broadens measured edges; does not affect the Gaussian fits.
    fit_fraction : float
        ``logquad`` fits pixels above this fraction of the background-subtracted peak.
    peak_min_distance : float, optional
        Minimum seed spacing for ``seeds='peaks'``; default half the pitch
        estimated from the image autocorrelation.
    background_level : float, optional
        Known constant background in original intensity units, for images with
        no hole-free region. Overrides the background map.
    background_tile : int, optional
        Background-map tile size in pixels; default about four pitches.
    deblend_iterations : int
        Maximum deblending iterations; stops earlier on convergence.
    pixel_size : float
        Isotropic length per pixel, used by plots and calibrated CSV columns.
    unit : str
        Length unit, e.g. 'px' or 'nm'. Array metrics remain in pixels.
    backend : {'cpu', 'gpu', 'auto'}
        Where cells, deblending and measurements run. ``gpu`` (CuPy, FP64)
        reproduces each CPU step to rounding (CDs typically within ~1e-8 px)
        and is 40-70x faster on images of ~1e5 holes; threshold decisions on
        values within rounding of the threshold (marginal noise seeds in dense,
        noisy arrays) can differ. Requires ``background='model'``. ``auto``
        uses the GPU when one is available and ``background='model'``.
    cell_backend : {'cpu', 'gpu'}, optional
        Backend for the Voronoi partition alone; default follows ``backend``.
    device : int
        CUDA device for the GPU backends.
    nominal_cd : float, optional
        Typical hole diameter in ``unit``. ``disk`` then uses the blur that
        makes its median diameter equal this; ``flux_ratio`` is scaled to the
        same median. Absolute CDs follow this assumption; hole-to-hole
        differences are what these methods measure.
    psf_fwhm : float, optional
        Known FWHM of the imaging blur (incoherent Airy pattern) in ``unit``;
        overrides the ``nominal_cd`` blur calibration. ``disk`` and
        ``flux_ratio`` need one of the two.
    dark_level : float
        Detector offset in original intensity units, removed from the local
        background before ``flux_ratio`` divides by it.
    flux_ratio_scale : float, optional
        Fixed factor from ``flux_ratio``'s apparent diameter to CD, instead of
        matching its median to ``nominal_cd`` in every image. Pass
        ``result.meta["flux_ratio_scale"]`` (and ``psf_fwhm`` from
        ``result.meta["psf_fwhm_px"] * pixel_size``) of one calibrated image
        so that several images are compared on one scale.
    half_area_level : float
        Fraction of the peak height above the local level that bounds the
        ``half_area`` family's region (0 < level < 1).
    half_area_peak_pixels : float
        ``half_area_min_avg``: area, in original pixels, of the brightest part
        of the enlarged cell averaged into the peak (>= 1).
    tail_area_level : float
        ``tail_area``: fraction of the height above the local level that
        bounds the region (0 < level < 1; low = near the background);
        ``tail_area_norm``: fraction of the peak.
    tail_window : float
        ``tail_area`` / ``tail_area_norm``: half-width of the window around each hole, in pitches
        (0.5 to 2).
    outlines : bool
        False skips the display outlines of the ``half_area`` family (their
        ``contours_xy``, ``geometry['edge_px']`` and ``perimeter_px`` stay NaN),
        for runs that keep only the numbers; every CD, width, centre, status
        and other metric is unchanged.

    Returns
    -------
    CDResult
        Cell arrays and per-method measurements, aligned by zero-based cell
        ID. Failed cells remain present with a status and NaN geometry.
        Contours and centers are global [x, y] pixel coordinates.
    """
    from contextlib import nullcontext

    from .backends import gpu_device

    img = np.asarray(image)
    if img.ndim != 2 or img.size == 0 or not np.issubdtype(img.dtype, np.number):
        raise ValueError("image must be a nonempty 2D real grayscale array")
    if np.iscomplexobj(img) or not np.isfinite(img).all():
        raise ValueError("image must contain finite real intensities")
    if background not in {"model", "border"}:
        raise ValueError("background must be 'model' or 'border'")
    if seeds not in {"threshold", "peaks"}:
        raise ValueError("seeds must be 'threshold' or 'peaks'")
    if seeds == "peaks" and background == "border":
        raise ValueError("seeds='peaks' requires background='model'")
    if methods is None:
        methods = ("gaussian", "logquad", "gradient", "halfmax") if background == "model" else ("logquad", "gradient", "halfmax")
    if isinstance(methods, str):
        methods = (methods,)
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or any(m not in METHOD_LABELS for m in methods):
        raise ValueError("methods must be unique choices from " + ", ".join(METHOD_LABELS))
    if "gaussian" in methods and background != "model":
        raise ValueError("method 'gaussian' requires background='model'")
    if background != "model" and set(methods) & set(INVARIANT_METHODS):
        raise ValueError("methods " + ", ".join(INVARIANT_METHODS) + " require background='model'")
    for name, value in (("nominal_cd", nominal_cd), ("psf_fwhm", psf_fwhm), ("flux_ratio_scale", flux_ratio_scale)):
        if value is not None and not (np.isfinite(value) and value > 0):
            raise ValueError(f"{name} must be positive")
    if not np.isfinite(dark_level):
        raise ValueError("dark_level must be finite")
    if not (np.isfinite(half_area_level) and 0 < half_area_level < 1):
        raise ValueError("half_area_level must be between 0 and 1")
    if not (np.isfinite(half_area_peak_pixels) and half_area_peak_pixels >= 1):
        raise ValueError("half_area_peak_pixels must be at least 1")
    if not (np.isfinite(tail_area_level) and 0 < tail_area_level < 1):
        raise ValueError("tail_area_level must be between 0 and 1")
    if not (np.isfinite(tail_window) and 0.5 <= tail_window <= 2):
        raise ValueError("tail_window must be between 0.5 and 2 pitches")
    if not isinstance(outlines, (bool, np.bool_)):
        raise ValueError("outlines must be True or False")
    if {"disk", "flux_ratio"} & set(methods) and nominal_cd is None and psf_fwhm is None:
        raise ValueError("methods 'disk' and 'flux_ratio' need nominal_cd or psf_fwhm (the imaging blur)")
    for name, value, lower in (("area_min", area_min, 1), ("area_max", area_max, area_min),
                               ("pad", pad, 0), ("n_angles", n_angles, 16),
                               ("deblend_iterations", deblend_iterations, 1)):
        if not np.isfinite(value) or int(value) != value or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    if not np.isfinite([radial_step, smooth_sigma, fit_fraction, pixel_size]).all():
        raise ValueError("measurement parameters must be finite")
    if not 0 < radial_step <= 1 or smooth_sigma < 0 or not 0 < fit_fraction < 1 or pixel_size <= 0:
        raise ValueError("require 0 < radial_step <= 1, smooth_sigma >= 0, 0 < fit_fraction < 1, pixel_size > 0")
    for name, value, lower in (("peak_min_distance", peak_min_distance, 1.0), ("background_tile", background_tile, 4)):
        if value is not None and (not np.isfinite(value) or value < lower):
            raise ValueError(f"{name} must be >= {lower}")
    if background_level is not None and not np.isfinite(background_level):
        raise ValueError("background_level must be finite")
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("unit must be a nonempty length-unit label")
    if unit == "px" and pixel_size != 1:
        raise ValueError("supply a physical unit when pixel_size differs from 1")
    backend, cell_backend = _resolve_backends(backend, cell_backend, background)
    with gpu_device(device) if "gpu" in (backend, cell_backend) else nullcontext():
        return _measure_cds(
            img, methods=methods, background=background, seeds=seeds, threshold=threshold, invert=invert,
            area_min=area_min, area_max=area_max, pad=pad, n_angles=n_angles, radial_step=radial_step,
            smooth_sigma=smooth_sigma, fit_fraction=fit_fraction, peak_min_distance=peak_min_distance,
            background_level=background_level, background_tile=background_tile,
            deblend_iterations=deblend_iterations, pixel_size=pixel_size, unit=unit, backend=backend,
            cell_backend=cell_backend, device=device, nominal_cd=nominal_cd, psf_fwhm=psf_fwhm,
            dark_level=float(dark_level), flux_ratio_scale=flux_ratio_scale,
            half_area_level=float(half_area_level), half_area_peak_pixels=float(half_area_peak_pixels),
            tail_area_level=float(tail_area_level), tail_window=float(tail_window), outlines=bool(outlines),
        )


def _resolve_backends(backend: str, cell_backend: str | None, background: str) -> tuple[str, str]:
    """Validated (measurement, cell) backends; 'auto' picks the GPU when usable."""
    if backend not in {"cpu", "gpu", "auto"}:
        raise ValueError("backend must be 'cpu', 'gpu' or 'auto'")
    if cell_backend not in {None, "cpu", "gpu"}:
        raise ValueError("cell_backend must be 'cpu' or 'gpu'")
    if backend == "auto":
        from .backends import cp
        try:
            usable = cp is not None and cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            usable = False
        backend = "gpu" if usable and background == "model" else "cpu"
    elif backend == "gpu":
        if background != "model":
            raise ValueError("backend='gpu' requires background='model'; border mode runs on the CPU")
        from .backends import resolve_backend
        resolve_backend("gpu")
    return backend, backend if cell_backend is None else cell_backend


def _array_module(backend: str):
    if backend == "gpu":
        import cupy
        return cupy
    return np


def _unmodelled_features(binary: np.ndarray, component_labels: np.ndarray, component_ids: np.ndarray) -> np.ndarray:
    """Thresholded pixels of the components without a model, grown by 2 px: ``scipy.ndimage.binary_dilation(
    (binary > 0) & ~np.isin(component_labels, component_ids), iterations=2)`` (cross-shaped steps, nothing outside
    the image), the same pixels with a lookup table and OpenCV's dilation."""
    import cv2

    modelled = np.zeros(int(component_labels.max(initial=0)) + 1, bool)
    modelled[component_ids[(component_ids >= 0) & (component_ids < len(modelled))]] = True
    features = ((binary > 0) & ~modelled[component_labels]).astype(np.uint8)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    return cv2.dilate(features, cross, iterations=2, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0


def _measure_cds(
    img, *, methods, background, seeds, threshold, invert, area_min, area_max, pad, n_angles, radial_step,
    smooth_sigma, fit_fraction, peak_min_distance, background_level, background_tile, deblend_iterations,
    pixel_size, unit, backend, cell_backend, device, nominal_cd=None, psf_fwhm=None, dark_level=0.0,
    flux_ratio_scale=None, half_area_level=0.5, half_area_peak_pixels=4.0, tail_area_level=0.2,
    tail_window=1.2, outlines=True,
) -> CDResult:
    from ._cpu.centers import _segment_binary
    from ._cpu.components import connected_components_stats_cpu
    from ._cpu.cells import _extract_voronoi_rois, compute_voronoi_labels_cpu

    if backend == "gpu":
        from ._gpu._cuda import release_memory
        from ._gpu.cd import gaussian_method, measure_cells
        from ._gpu.deblend import deblend as deblend_gpu

        def deblend(*args, **kwargs):
            state = deblend_gpu(*args, **kwargs)
            release_memory()
            return state
    else:
        from ._cpu.cd import gaussian_method, measure_cells
        from ._cpu.deblend import deblend

        def release_memory():
            pass

    started = time.perf_counter()
    seconds = {}  # per stage, in meta["seconds"]
    height, width = img.shape
    # Negation preserves contrast and permits the same bright-feature methods
    # for dark holes, including a background estimate in the correct polarity.
    sign = -1.0 if invert else 1.0
    signal_image = img.astype(np.float64) * sign
    rescaled = False
    components_before_filter = 0
    pitch = float("nan")
    if seeds == "threshold":
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
        component_labels = comp.labels
        components_before_filter = comp.num_labels - 1
        x, y, w, h, area = (np.asarray(comp.stats[1:comp.num_labels], dtype=np.int64).reshape(-1, 5).T)
        keep = (area >= area_min) & (area <= area_max)
        inside = ~((x <= 0) | (y <= 0) | (x + w >= width) | (y + h >= height))
        # Border mode drops edge holes entirely; model mode still models
        # them so their tails are subtracted from interior neighbours.
        if background == "border":
            keep &= inside
        rows = np.column_stack((x, y, w, h, np.asarray(comp.centroids[1:comp.num_labels]).reshape(-1, 2)))[keep].tolist()
        component_ids = (np.flatnonzero(keep) + 1).astype(np.int32)
        interior = inside[keep].tolist()
    else:
        from ._cpu.deblend import detect_peaks

        peaks, pitch = detect_peaks(signal_image, min_distance=peak_min_distance)
        binary = np.zeros(img.shape, bool)
        component_labels = np.zeros(img.shape, np.int32)
        half = int(np.ceil(0.6 * pitch))
        rows = [[cx - half, cy - half, 2 * half + 1, 2 * half + 1, cx, cy] for cx, cy in peaks]
        component_ids = np.full(len(rows), -1, dtype=np.int32)
        interior = [True] * len(rows)  # decided after fitting, from each hole's width
    def build_cells(rows):
        seeds_xy = np.asarray(rows, dtype=float).reshape(-1, 6)[:, 4:6].copy()
        if cell_backend == "gpu" and background == "model":
            # Deblending reads the image itself: no background-filled ROI stack.
            from ._gpu.cells import voronoi_cells
            labels, masks, origins = voronoi_cells(rows, img.shape, pad=int(pad))
            return seeds_xy, labels, None, masks, origins
        if cell_backend == "gpu":
            from ._gpu.voronoi import compute_voronoi_labels_gpu
            labels = compute_voronoi_labels_gpu(seeds_xy, img.shape, device=device)
        else:
            labels = compute_voronoi_labels_cpu(seeds_xy, img.shape)
        rois, masks, origins = _extract_voronoi_rois(signal_image, labels, rows, pad=int(pad))
        return seeds_xy, labels, rois, masks, np.asarray(origins, dtype=np.int32).reshape(-1, 2)

    all_seeds, cell_labels, rois, masks, origins = build_cells(rows)
    seconds["segmentation"] = time.perf_counter() - started
    meta = {
        "background": background, "seeds": seeds, "threshold": threshold, "invert": bool(invert),
        "area_min": int(area_min), "area_max": int(area_max), "pad": int(pad), "n_angles": int(n_angles),
        "radial_step": float(radial_step), "smooth_sigma": float(smooth_sigma),
        "fit_fraction": float(fit_fraction), "half_area_level": float(half_area_level),
        "half_area_peak_pixels": float(half_area_peak_pixels), "tail_area_level": float(tail_area_level),
        "tail_window": float(tail_window), "outlines": bool(outlines), "cell_backend": cell_backend,
        "measurement_backend": backend, "device": int(device),
        "segmentation_rescaled": rescaled, "components_before_filter": components_before_filter,
        "coordinates": "global [x, y] pixels; images indexed [y, x]",
        "nominal_cd": nominal_cd, "psf_fwhm": psf_fwhm, "dark_level": float(dark_level), "seconds": seconds,
    }
    background_map = deblend_model = None
    start = time.perf_counter()
    if background == "border":
        measurements = measure_cells(
            rois, masks, origins, all_seeds, methods=methods, n_angles=int(n_angles),
            radial_step=radial_step, smooth_sigma=smooth_sigma, fit_fraction=fit_fraction,
        )
        seconds["radial"], start = time.perf_counter() - start, time.perf_counter()
        meta["background_definition"] = "10th percentile of the polarity-normalized cell border"
        report = np.arange(len(rows))
    else:
        from scipy.ndimage import map_coordinates
        from ._cpu.deblend import FWHM_K, auto_tile, revise_seeds, sigma_major

        component_ids = np.asarray(component_ids, dtype=np.int32)
        interior = np.asarray(interior, dtype=bool)
        # Peak seeds and seeds added from the residual have no component box;
        # their edge test uses the fitted width instead.
        by_reach = np.full(len(rows), seeds == "peaks")
        tile_px = tiled_seeds = None
        if seeds == "threshold":
            tile_px, pitch = auto_tile(all_seeds, img.shape)
            tiled_seeds = all_seeds  # deblend's own auto_tile of these seeds gives tile_px

        def tile_for(seeds_xy):
            if background_tile is not None or seeds_xy is not tiled_seeds:
                return background_tile
            return tile_px

        added = dropped = 0
        for revision in range(3):
            exclude = None
            if seeds == "threshold":
                # Thresholded features without a model (merged blobs, specks outside
                # the area limits) must not be read as background.
                exclude = _unmodelled_features(binary, component_labels, component_ids)
            final = revision == 2 or not np.isfinite(pitch)
            # Revision passes only need approximate fits to judge seeds.
            state = deblend(
                signal_image, all_seeds, masks, origins,
                iterations=int(deblend_iterations) if final else min(10, int(deblend_iterations)),
                tile=tile_for(all_seeds), exclude=exclude,
                background_level=None if background_level is None else sign * float(background_level),
            )
            if final:
                break
            # One missed or spurious hole corrupts its neighbours' fits; drop
            # insignificant seeds, add holes left in the residual, and refit.
            drop, new = revise_seeds(signal_image, state, all_seeds, pitch, xp=_array_module(backend))
            if not drop.any() and not len(new):
                if state.converged or state.iterations >= int(deblend_iterations):
                    break
                # Seeds are settled; finish these fits with the remaining budget.
                state = deblend(
                    signal_image, all_seeds, masks, origins,
                    iterations=int(deblend_iterations) - state.iterations, init=state.params,
                    tile=tile_for(all_seeds), exclude=exclude,
                    background_level=None if background_level is None else sign * float(background_level),
                )
                break
            keep = ~drop
            half = int(np.ceil(0.6 * pitch))
            rows = [r for r, k in zip(rows, keep) if k] + [
                [x - half, y - half, 2 * half + 1, 2 * half + 1, x, y] for x, y in new]
            component_ids = np.concatenate((component_ids[keep], np.full(len(new), -1, np.int32)))
            interior = np.concatenate((interior[keep], np.ones(len(new), bool)))
            by_reach = np.concatenate((by_reach[keep], np.ones(len(new), bool)))
            added, dropped = added + len(new), dropped + int(drop.sum())
            all_seeds, cell_labels, rois, masks, origins = build_cells(rows)
        if by_reach.any():
            # A hole is reported when one FWHM around its fitted center is in the image.
            params = state.params[by_reach]
            centers = params[:, 1:3] + origins[by_reach]
            reach = np.where(np.isfinite(params).all(axis=1), 2 * FWHM_K * sigma_major(params), np.inf)
            interior[by_reach] = ((centers[:, 0] - reach > 0) & (centers[:, 0] + reach < width - 1)
                                  & (centers[:, 1] - reach > 0) & (centers[:, 1] + reach < height - 1))
        report = np.flatnonzero(interior)
        cleaned = state.cleaned[report]
        seconds["deblend"], start = time.perf_counter() - start, time.perf_counter()
        measured = [m for m in methods if m != "gaussian" and m not in INVARIANT_METHODS]
        found = measure_cells(
            cleaned, masks[report], origins[report], all_seeds[report], methods=tuple(measured),
            n_angles=int(n_angles), radial_step=radial_step, smooth_sigma=smooth_sigma,
            fit_fraction=fit_fraction, backgrounds=np.zeros(len(report)),
        ) if measured else {}
        if measured:
            seconds["radial"], start = time.perf_counter() - start, time.perf_counter()
        if "gaussian" in methods:
            found["gaussian"] = gaussian_method(
                state.params[report], state.status[report], cleaned, masks[report], origins[report], int(n_angles),
            )
            seconds["gaussian"], start = time.perf_counter() - start, time.perf_counter()
        invariant = tuple(m for m in methods if m in INVARIANT_METHODS)
        if invariant:
            from ._cpu.invariant_cd import measure_invariant
            extra, info = measure_invariant(
                signal_image, state, masks, origins, report, methods=invariant, n_angles=int(n_angles),
                psf_fwhm_px=None if psf_fwhm is None else psf_fwhm / pixel_size,
                nominal_cd_px=None if nominal_cd is None else nominal_cd / pixel_size,
                dark_level=sign * float(dark_level), flux_ratio_scale=flux_ratio_scale,
                half_area_level=half_area_level, half_area_peak_pixels=half_area_peak_pixels,
                tail_area_level=tail_area_level, tail_window=tail_window,
                pitch_px=float(pitch) if np.isfinite(pitch) else None, outlines=outlines,
                xp=_array_module(backend),
            )
            found.update(extra)
            seconds.update(info.pop("seconds", {}))
            meta.update(info)
            start = time.perf_counter()
        release_memory()
        local_background = sign * map_coordinates(
            state.background_map, all_seeds[report].T[::-1], order=1, mode="nearest",
        ) if len(report) else np.zeros(0)
        measurements = {}
        for name in methods:
            found[name].metrics["background"][:] = local_background
            measurements[name] = found[name]
        rois = cleaned
        background_map = sign * state.background_map
        statuses, counts = np.unique(state.status[report], return_counts=True)
        meta.update(
            background_definition=(
                "Smooth map fitted jointly with one 2-D Gaussian per hole; measured cells have the map "
                "and all other holes' models removed"),
            background_source=state.background_info.get("source"),
            background_tiles=state.background_info.get("tiles"),
            background_measured_tiles=state.background_info.get("measured_tiles"),
            background_tile_px=int(state.tile_px), background_level=background_level,
            deblend_iterations=int(deblend_iterations), deblend_iterations_used=int(state.iterations),
            deblend_converged=bool(state.converged),
            deblend_status_counts=dict(zip(statuses.tolist(), counts.tolist())),
            modelled_holes=int(len(rows)), seeds_added_from_residual=int(added),
            seeds_dropped_as_insignificant=int(dropped),
            pitch_px=float(pitch) if np.isfinite(pitch) else None,
        )
        remap = np.full(len(rows) + 1, -1, dtype=np.int32)
        remap[report] = np.arange(len(report), dtype=np.int32)
        cell_labels = remap[cell_labels]  # -1 stays -1 through the extra slot
        params_global = state.params.copy()
        params_global[:, 1:3] += origins  # every modelled hole, as rendered by deblend
        deblend_model = {"params": params_global, "report": report, "tile_px": int(state.tile_px)}
        if "tile_medians" in state.background_info:
            deblend_model["tile_medians"] = state.background_info["tile_medians"]
        masks, origins = masks[report], origins[report]
    row_array = np.asarray(rows, dtype=float).reshape(-1, 6)
    result = CDResult(
        image=img.copy(), binary=binary > 0, component_labels=component_labels,
        component_ids=np.asarray(component_ids, dtype=np.int32)[report],
        boxes_xywh=row_array[report, :4].astype(np.int32), seeds_xy=all_seeds[report],
        cell_labels=cell_labels,
        cell_areas_px2=np.bincount(cell_labels[cell_labels >= 0], minlength=len(report)),
        origins_xy=origins, rois=rois, masks=masks, measurements=measurements,
        pixel_size=float(pixel_size), unit=unit, background_map=background_map,
        deblend_model=deblend_model, meta=meta,
    )
    seconds["result"] = time.perf_counter() - start
    seconds["total"] = time.perf_counter() - started
    return result


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


def save_cd_viewer(result: CDResult, path, *, metric: str = "cd_equivalent_px", region=None) -> Path:
    """Write a self-contained, offline HTML viewer with crisp contour zoom.

    Supports pan/zoom, method comparisons, CD and mean-intensity cell maps,
    cell selection, and optional contour vertices. No server or CDN is needed.
    The source raster retains its native resolution; contour paths are redrawn
    at the current zoom. Drawing coordinates use local float32 offsets, while
    CSV/NPZ exports retain the original float64 measurements.

    ``region=(x, y, w, h)`` limits the viewer to that crop and the cells whose
    seeds lie in it, for images too large to view whole (about 1e5 holes and
    up); cell IDs and coordinates shown remain those of the full image.
    """
    from ._cd_viewer import write_viewer
    return write_viewer(result, path, metric=metric, region=region)


_CELL_ARRAYS = ("image", "binary", "component_labels", "component_ids", "boxes_xywh", "seeds_xy",
                "cell_labels", "cell_areas_px2", "mean_intensity", "origins_xy", "rois", "masks")
_METHOD_ARRAYS = ("centers_xy", "contours_xy", "valid", "status")


def save_cd_results(result: CDResult, output, *, compact: bool = False) -> dict[str, Path]:
    """Save measurements.csv, cells.csv, contours.csv, raw_data.npz and manifest.json.

    CSVs include pixel and calibrated values. NPZ contains the input image,
    cell extraction, all contours, validity/status arrays, and every metric;
    it loads with ``allow_pickle=False``. The JSON manifest records parameters,
    definitions, calibration, and valid/failed counts.

    ``compact=True`` is for batches of large images: no contours.csv, and an
    uncompressed NPZ without the arrays that ``load_cd_results`` rebuilds
    exactly — contours (from each method's ``geometry``), ``rois``,
    ``component_labels``, and the background map (from its tile medians).
    The two halves are ``save_cd_tables`` and ``save_cd_arrays``, for
    pipelines that write the CSVs in another process.
    """
    return {**save_cd_tables(result, output, compact=compact), **save_cd_arrays(result, output, compact=compact)}


def _output_names(compact: bool) -> tuple[str, ...]:
    return ("measurements.csv", "cells.csv") + (() if compact else ("contours.csv",)) + ("raw_data.npz", "manifest.json")


def save_cd_tables(result: CDResult, output, *, compact: bool = False) -> dict[str, Path]:
    """The CSV half of ``save_cd_results``: measurements.csv, cells.csv and (unless compact) contours.csv."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: output / name for name in _output_names(compact) if name.endswith(".csv")}
    scale = result.pixel_size
    metric_names = sorted({name for m in result.measurements.values() for name in m.metrics})
    calibrated = {"area_px2": "area", "cd_equivalent_px": "cd_equivalent", "cd_x_px": "cd_x",
                  "cd_y_px": "cd_y", "perimeter_px": "perimeter",
                  "fwhm_major_px": "fwhm_major", "fwhm_minor_px": "fwhm_minor",
                  "cd_uncertainty_px": "cd_uncertainty"}
    header = ["cell_id", "component_id", "method", "valid", "status", "seed_x_px", "seed_y_px",
              "center_x_px", "center_y_px", "center_x", "center_y", "unit", "pixel_size", "mean_intensity"]
    # Whole columns go through tolist(): the csv module formats the same Python
    # floats as before, without per-element NumPy indexing (10x faster at 1e5 cells).
    n = len(result.seeds_xy)
    ids, components = list(range(n)), result.component_ids.tolist()
    seed_x, seed_y = result.seeds_xy[:, 0].tolist(), result.seeds_xy[:, 1].tolist()
    means, missing = result.mean_intensity.tolist(), [np.nan] * n
    with paths["measurements.csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header + metric_names + list(calibrated.values()))
        for method, measured in result.measurements.items():
            centers = measured.centers_xy
            columns = [ids, components, [method] * n, measured.valid.tolist(), measured.status.tolist(),
                       seed_x, seed_y, centers[:, 0].tolist(), centers[:, 1].tolist(),
                       (centers[:, 0] * scale).tolist(), (centers[:, 1] * scale).tolist(),
                       [result.unit] * n, [scale] * n, means]
            columns += [measured.metrics[name].tolist() if name in measured.metrics else missing
                        for name in metric_names]
            columns += [np.where(measured.valid, measured.metrics[name] * scale**(2 if name == "area_px2" else 1),
                                 np.nan).tolist() if name in measured.metrics else missing for name in calibrated]
            writer.writerows(zip(*columns))
    with paths["cells.csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell_id", "component_id", "seed_x_px", "seed_y_px", "cell_area_px2", "mean_intensity"])
        writer.writerows(zip(ids, components, seed_x, seed_y, result.cell_areas_px2.tolist(), means))
    if not compact:
        with paths["contours.csv"].open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["cell_id", "component_id", "method", "valid", "vertex", "x_px", "y_px", "x", "y", "unit"])
            for method, measured in result.measurements.items():
                vertices = measured.contours_xy.shape[1]
                for start in range(0, n, 4096):  # bounds the Python lists for large images
                    block = measured.contours_xy[start:start + 4096]
                    count = len(block)
                    cells = np.repeat(np.arange(start, start + count), vertices)
                    x, y = block[..., 0].ravel(), block[..., 1].ravel()
                    writer.writerows(zip(
                        cells.tolist(), result.component_ids[cells].tolist(), [method] * len(cells),
                        np.repeat(measured.valid[start:start + count], vertices).tolist(),
                        np.tile(np.arange(vertices), count).tolist(), x.tolist(), y.tolist(),
                        (x * scale).tolist(), (y * scale).tolist(), [result.unit] * len(cells)))
    return paths


def save_cd_arrays(result: CDResult, output, *, compact: bool = False) -> dict[str, Path]:
    """The array half of ``save_cd_results``: raw_data.npz and manifest.json (which lists the CSVs too)."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = _output_names(compact)
    paths = {name: output / name for name in names[-2:]}
    scale = result.pixel_size
    dropped = ("component_labels", "rois") if compact else ()
    arrays = {name: getattr(result, name) for name in _CELL_ARRAYS if name not in dropped}
    arrays.update(pixel_size=np.asarray(scale), unit=np.asarray(result.unit),
                  methods=np.asarray(list(result.measurements)))
    model = result.deblend_model or {}
    for name, value in model.items():
        arrays[f"deblend_{name}"] = np.asarray(value)
    if result.background_map is not None and not (compact and "tile_medians" in model):
        arrays["background_map"] = result.background_map
    summary = {}
    for method, measured in result.measurements.items():
        for name in _METHOD_ARRAYS:
            if not (compact and name == "contours_xy"):
                arrays[f"{method}_{name}"] = getattr(measured, name)
        for name, values in measured.metrics.items():
            arrays[f"{method}_{name}"] = values
        if compact:
            if not measured.geometry:
                raise ValueError(f"compact export needs the contour geometry of method {method!r}")
            for name, values in measured.geometry.items():
                arrays[f"{method}_geometry_{name}"] = values
        statuses, counts = np.unique(measured.status, return_counts=True)
        values = measured.metrics["cd_equivalent_px"][measured.valid]
        summary[method] = {
            "valid": int(measured.valid.sum()), "failed": int((~measured.valid).sum()),
            "status_counts": dict(zip(statuses.tolist(), counts.tolist())),
            "median_cd_px": float(np.median(values)) if values.size else None,
            "mean_cd_px": float(values.mean()) if values.size else None,
            "std_cd_px": float(values.std()) if values.size else None,
        }
    if compact and result.meta.get("background") == "model" and "params" not in model:
        raise ValueError("compact export of background='model' results needs result.deblend_model")
    if compact:
        arrays["compact"] = np.asarray(True)
        np.savez(paths["raw_data.npz"], **arrays)  # zlib would cost more time than it saves at 1e5 cells
    else:
        np.savez_compressed(paths["raw_data.npz"], **arrays)
    manifest = {
        "schema_version": 2, "compact": bool(compact), "shape_hw": list(result.image.shape), "cells": len(result.seeds_xy),
        "pixel_size": scale, "unit": result.unit, "parameters": result.meta,
        "definitions": {
            "gaussian": "Half-maximum ellipse of a least-squares 2-D Gaussian fitted jointly with a smooth background map and all neighbouring holes (deblended); FWHM = 2 sqrt(2 ln 2) sigma.",
            "logquad": "Half-maximum ellipse of a background-subtracted Gaussian log-quadratic fit; FWHM = 2 sqrt(2 ln 2) sigma.",
            "gradient": "Closed polygon through subpixel maxima of the outward radial intensity decrease; Gaussian diameter is approximately 2 sigma without smoothing.",
            "halfmax": "Closed polygon through first outward crossings at half the interpolated central signal above background.",
            "disk": "Diameter d of a round hole that, imaged through a circular-pupil (Airy) blur of FWHM psf_fwhm_px and averaged over each pixel, best fits the cell (least squares, free amplitude, center and local background per hole, neighbours' fitted models removed); the contour is the hole edge, not the spot. cd_uncertainty_px is d's standard error from the fit.",
            "flux_ratio": "Hole light (sum over the cell above the disk fit's local background, neighbours removed) divided by that local background minus dark_level; CD = 2 sqrt(ratio / pi) times flux_ratio_scale (given, else median matched to nominal_cd when given, else 1).",
            "projection": "Column and row sums of the neighbour-removed cell; cd_x/cd_y = distance between each profile's steepest rise and steepest fall (cubic spline); cd_equivalent = their mean; contour = axis-aligned ellipse with those axes.",
            "half_area": "Cell enlarged 4x (cubic spline, scipy.ndimage.zoom order 3) after subtracting its local level b (median of the cell's outermost pixels); region = pixels connected to the peak and above half_area_level x peak height (above b); cd_equivalent = 2 sqrt(region area / pi); cd_x/cd_y = x/y diameters of the ellipse with the region's area and x/y second moments (cd_x * cd_y = cd_equivalent^2); contour = the region's last pixel along rays from its centroid (display only); no shape assumed.",
            "half_area_min": "As half_area, with the local level b = the minimum (not the median) of the cell's outermost pixels.",
            "half_area_min_avg": "As half_area_min, with the peak = mean of the brightest enlarged pixels covering half_area_peak_pixels original pixels (not the single highest).",
            "tail_area": "On a window of tail_window pitches around the hole holding its share of the light (image minus the background map, each pixel shared among the holes in proportion to their fitted models; the region may extend past the cell): b = median of the farthest tenth of the hole's own territory (closer to it than to any other hole), averaged peak as half_area_min_avg, threshold at tail_area_level of the height above b; mean_signal = mean of the enlarged cell above b over the region (the cell intensity).",
            "tail_area_norm": "As tail_area without a local level: the window divided by its averaged peak; region = where that normalized pattern exceeds tail_area_level (connected to the peak).",
            "foot_area": "On tail_area's window, with neither a peak nor a local level: steepest level T = sum(I |grad I|) / sum(|grad I|); threshold = sum(I lap I) / sum(lap I) over the pixels below T in the hole's own territory (signed Laplacian, differences over +-1 original px): the brightness where the falling edge bends into the flat surroundings; region = pixels connected to the brightest point and above it. Pulse- and offset-invariant.",
            "foot_area_ratio": "As foot_area, with the threshold = r x the hole's own steepest level, r = median over the image's holes of foot level / steepest level (meta foot_ratio).",
            "edge_area": "On tail_area's window, with neither a peak nor a local level: threshold = sum(I |grad I|) / sum(|grad I|) over the enlarged window (central differences over +-1 original px), the brightness where the light falls fastest; region = pixels connected to the brightest point and above it. Pulse-invariant (a scale moves I and the threshold together) and offset-invariant.",
            "cd_equivalent": "2 sqrt(contour_area / pi); analytic ellipse area for logquad, polygon area otherwise.",
            "cd_x_cd_y": "Axis-aligned contour extents; analytic ellipse projections for logquad.",
            "failed_cells": "Retained with valid=False, status, and NaN geometric metrics; candidate/partial contours may remain for diagnosis.",
            "calibration": "All NPZ geometry is in pixels. Unsuffixed CSV lengths use unit, areas use unit squared. Cell areas are Voronoi areas, not hole areas.",
            "background": result.meta.get("background_definition", ""),
            "background_map": "Fitted background in original intensity units (background='model' only); rois in raw_data.npz are the measured, background- and neighbour-subtracted cells in polarity-normalized units.",
            "mean_intensity": "Arithmetic mean of original image pixels over the full Voronoi cell, including background. No padding fill, background subtraction, polarity inversion, or physical-unit scaling; independent of CD fit validity.",
            "deblend": "deblend_params: global [amplitude, x0, y0, u1, u2, u3] of every modelled hole (precision = L L^T, L = [[e^u1, 0], [u2, e^u3]]; polarity-normalized amplitude); deblend_report: indices of the measured cells among them; deblend_tile_medians/tile_px: the background map's tiles.",
            "compact": "Compact raw_data.npz omits contours_xy, rois, component_labels and (when tile medians exist) background_map; subpx.load_cd_results rebuilds them exactly from <method>_geometry_*, deblend_* and masks.",
        },
        "methods": summary, "files": list(names),
    }
    paths["manifest.json"].write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return paths


def load_cd_results(path, *, rois: bool = True) -> CDResult:
    """Load a results folder (or its raw_data.npz) written by ``save_cd_results``.

    Full and compact exports load to the same ``CDResult``: compact files have
    their contours, ``rois``, ``component_labels`` and background map rebuilt
    bit-identically from the stored construction data. ``meta`` comes from
    manifest.json next to the NPZ. ``rois=False`` skips rebuilding the measured
    cells of a compact file (seconds at 1e5 cells; ``result.rois`` is then None).
    """
    from ._cpu.cd import ellipse_contours, radial_contours

    path = Path(path)
    npz = path / "raw_data.npz" if path.is_dir() else path
    manifest_path = npz.parent / "manifest.json"
    meta = json.loads(manifest_path.read_text(encoding="utf-8")).get("parameters", {}) if manifest_path.exists() else {}
    with np.load(npz, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    compact = bool(arrays.pop("compact", False))
    methods = arrays.pop("methods").tolist()
    model = {name[len("deblend_"):]: arrays.pop(name) for name in list(arrays) if name.startswith("deblend_")}
    if "tile_px" in model:
        model["tile_px"] = int(model["tile_px"])
    origins = arrays["origins_xy"]
    n_angles = int(meta.get("n_angles", 128))
    # Each method's arrays, by the longest method name prefixing them ("half_area_" also prefixes "half_area_min_...").
    owned = {method: {} for method in methods}
    for name in list(arrays):
        owners = [method for method in methods if name.startswith(f"{method}_")]
        if owners:
            owner = max(owners, key=len)
            owned[owner][name[len(owner) + 1:]] = arrays.pop(name)
    measurements = {}
    for method in methods:
        own = owned[method]
        fields = {name: own.pop(name) for name in _METHOD_ARRAYS if name in own}
        geometry = {name[len("geometry_"):]: own.pop(name) for name in list(own) if name.startswith("geometry_")}
        metrics = own
        if "contours_xy" not in fields:
            if "edge_px" in geometry:
                fields["contours_xy"] = radial_contours(geometry["center_local"], geometry["edge_px"], origins, n_angles)
            elif method == "logquad":
                fields["contours_xy"] = ellipse_contours(geometry["center_local"], geometry["eigenvalues"],
                                                         geometry["axes"], n_angles, offsets=origins)
            else:
                fields["contours_xy"] = ellipse_contours(fields["centers_xy"], geometry["eigenvalues"],
                                                         geometry["axes"], n_angles)
        measurements[method] = CDMethodResult(method=method, metrics=metrics, geometry=geometry, **fields)
    shape = arrays["image"].shape
    sign = -1.0 if meta.get("invert") else 1.0
    if "background_map" not in arrays and "tile_medians" in model:
        from ._cpu.deblend import _upsample_tiles
        arrays["background_map"] = sign * _upsample_tiles(model["tile_medians"], model["tile_px"], shape)
    if compact:
        from ._cpu.components import connected_components_stats_cpu
        binary = arrays["binary"]
        arrays["component_labels"] = (connected_components_stats_cpu(binary.astype(np.uint8), connectivity=8).labels
                                      if binary.any() else np.zeros(shape, np.int32))
        arrays["rois"] = _rebuild_rois(arrays, model, meta, sign) if rois else None
    scalars = {"pixel_size": float(arrays.pop("pixel_size")), "unit": str(arrays.pop("unit"))}
    arrays.pop("mean_intensity", None)  # recomputed by CDResult from image and cell labels
    return CDResult(measurements=measurements, deblend_model=model or None, meta=meta,
                    background_map=arrays.pop("background_map", None), **scalars, **arrays)


def _rebuild_rois(arrays: dict, model: dict, meta: dict, sign: float) -> np.ndarray:
    """Measured cells exactly as ``measure_cds`` stored them in ``CDResult.rois``."""
    signal = arrays["image"].astype(np.float64) * sign
    masks, origins = arrays["masks"], arrays["origins_xy"]
    if meta.get("background") == "border":
        from ._cpu.cells import _extract_voronoi_rois
        rows = np.column_stack((arrays["boxes_xywh"], arrays["seeds_xy"])).tolist()
        return _extract_voronoi_rois(signal, arrays["cell_labels"], rows, pad=int(meta["pad"]))[0]
    from ._cpu.deblend import clean_cells, render_models, windows
    params = model["params"]
    roi_shape = masks.shape[1:]
    radius_cap = int(max(roi_shape))
    background = sign * arrays["background_map"]
    model_image = render_models(params, signal.shape, radius_cap=radius_cap)
    raw = windows(signal, origins, roi_shape)
    return clean_cells(raw, background, model_image, params[model["report"]], masks, origins, radius_cap)
