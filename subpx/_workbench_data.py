"""Qt-free analysis and diagnostic data for the desktop workbench."""

from __future__ import annotations

import csv
import json
import threading
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .types import CDMethodResult, CDResult, CDSummary, CenterResult, PatternPitchResult, PreprocessResult, ShiftResult


@dataclass
class LoadedImage:
    image: np.ndarray
    name: str
    path: str | None = None
    page: int = 0
    pages: int = 1
    conversion: str = ""


@dataclass
class DetectionReview:
    result: CenterResult
    binary: np.ndarray
    labels: np.ndarray
    boxes: np.ndarray
    kept: np.ndarray
    rescaled: bool


@dataclass
class AnalysisReview:
    image: np.ndarray
    origin_xy: np.ndarray
    parameters: dict
    detection: DetectionReview | None = None
    cd: CDResult | None = None
    patterns: dict[str, PatternPitchResult] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)  # information, e.g. component counts
    warnings: list[str] = field(default_factory=list)  # failures and limits reached
    seconds: float = 0.0

    @property
    def seeds_xy(self):
        return self.cd.seeds_xy if self.cd is not None else self.detection.result.centers_xy

    @property
    def measurements(self):
        if self.cd is not None:
            return self.cd.measurements
        points = self.detection.result.centers_xy
        return {"detection": CDMethodResult(
            "detection", points, np.empty((len(points), 0, 2)),
            np.ones(len(points), bool), np.full(len(points), "ok"), {},
        )}


@dataclass
class Preprocessed:
    """A page made with the workbench's Preprocess settings: the processed page and how it was made."""

    result: PreprocessResult
    parameters: dict  # the Setup settings it was made with (the mean hole also used the Holes settings)
    seconds: float = 0.0
    centers: int = 0  # holes found for the mean hole

    @property
    def image(self):
        return self.result.image


@dataclass
class RegistrationReview:
    result: ShiftResult
    aligned: np.ndarray
    overlap: np.ndarray
    difference_before: np.ndarray
    difference_after: np.ndarray
    correlation: np.ndarray
    rmse_before: float
    rmse_after: float
    hole_check_error: float = float("nan")
    hole_check_mode: str = "centered"


def demo_image():
    """Create a small deterministic staggered pattern for the workbench demo."""
    height, width = 192, 320
    yy, xx = np.mgrid[:height, :width]
    rng = np.random.default_rng(42)
    image = 15 + 0.5 * np.sin(2 * np.pi * xx / width)
    sigma = 5 / (2 * np.sqrt(2 * np.log(2)))
    for row, cy in enumerate(np.arange(14.45, height - 12, 16)):
        for cx in np.arange(14.35 + (row % 2) * 8, width - 12, 16):
            dx, dy = rng.normal(0, 0.04, 2)
            image += 62 * np.exp(-((xx - cx - dx)**2 + (yy - cy - dy)**2) / (2 * sigma**2))
    return np.clip(np.rint(image + rng.normal(0, 0.7, image.shape)), 0, 255).astype(np.uint8)


def grayscale(image):
    """Validate a scientific image, retaining its native grayscale precision."""
    array = np.asarray(image)
    conversion = ""
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        array = np.asarray(array[..., :3], dtype=np.float64) @ np.array([0.2126, 0.7152, 0.0722])
        conversion = "RGB converted to luminance (0.2126 R + 0.7152 G + 0.0722 B). Alpha ignored."
    if array.ndim != 2 or min(array.shape) < 2:
        raise ValueError("Choose a 2-D image or an RGB/RGBA image; stacks must be opened one page at a time.")
    if array.dtype.kind not in "buif" or not np.isfinite(array).all():
        raise ValueError("Images must contain finite, real numeric values (no NaN or infinity).")
    if array.dtype == bool:
        array = array.astype(np.uint8)
        conversion = "Boolean converted to uint8 (0 / 1 intensities)."
    return array, conversion


def load_image(path, page=0):
    path = Path(path)
    pages = 1
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        with tifffile.TiffFile(path) as tif:
            pages = len(tif.pages)
            if not 0 <= page < pages:
                raise ValueError(f"TIFF page must be between 0 and {pages - 1}.")
            data = tif.pages[page].asarray()
    elif path.suffix.lower() == ".npy":
        data = np.load(path, allow_pickle=False, mmap_mode="r")
    else:
        from PIL import Image

        with Image.open(path) as source:
            pages = getattr(source, "n_frames", 1)
            source.seek(page)
            data = np.array(source.convert("RGB") if source.mode == "P" else source)
    data, conversion = grayscale(data)
    return LoadedImage(data, path.name, str(path.resolve()), page, pages, conversion)


def load_pages(path):
    """Load all file pages separately, retaining precision and page coordinates.

    Pages may have different shapes and dtypes. They are never concatenated or
    padded into an analysis image. TIFF decoding uses one open file handle.
    """
    path = Path(path)
    frames = []
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        with tifffile.TiffFile(path) as tif:
            for index, page in enumerate(tif.pages):
                data, conversion = grayscale(page.asarray())
                frames.append(LoadedImage(data, path.name, str(path.resolve()), index, len(tif.pages), conversion))
    else:
        first = load_image(path)
        frames = [first] + [load_image(path, page) for page in range(1, first.pages)]
    return frames


class PageReader:
    """One multipage file kept open: page sizes read once, pages decoded on demand (the stacked page view).

    TIFF pages are decoded from the open file; other formats are reopened per page. ``read`` may be called
    from the worker thread and the window at the same time, so it holds a lock.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._tif = None
        if self.path.suffix.lower() in (".tif", ".tiff"):
            import tifffile

            self._tif = tifffile.TiffFile(self.path)
            self.shapes = [(int(page.imagelength), int(page.imagewidth)) for page in self._tif.pages]
        else:
            first = load_image(self.path)
            self.shapes = [first.image.shape] + [load_image(self.path, page).image.shape
                                                 for page in range(1, first.pages)]

    def read(self, page):
        if self._tif is None:
            return load_image(self.path, page)
        with self._lock:
            data = self._tif.pages[page].asarray()
        data, conversion = grayscale(data)
        return LoadedImage(data, self.path.name, str(self.path.resolve()), page, len(self.shapes), conversion)

    def close(self):
        if self._tif is not None:
            with self._lock:
                self._tif.close()


def page_preview(image, width=256):
    """Block-mean copy at most ``width`` pixels wide and its reduction factor (pages drawn small).

    The preview covers the page's first ``shape * factor`` pixels; the last partial block is left out.
    """
    factor = max(1, int(np.ceil(image.shape[1] / width)))
    if factor == 1:
        return image, 1
    h, w = image.shape[0] // factor * factor, image.shape[1] // factor * factor
    blocks = image[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
    return (np.rint(blocks).astype(image.dtype) if image.dtype.kind in "biu" else blocks.astype(np.float32)), factor


def cell_values_map(cell_labels, values):
    """Each cell's value at its pixels (float32), NaN outside the cells: the CD color map."""
    lookup = np.append(np.asarray(values, np.float32), np.float32(np.nan))
    return lookup[cell_labels]  # label -1 (no cell) picks the NaN


def cell_mask_map(cell_labels, selected):
    """1 at the pixels of the selected cells, 0 elsewhere (uint8): e.g. the failed cells."""
    lookup = np.append(np.asarray(selected, bool), False).astype(np.uint8)
    return lookup[cell_labels]


def array_bytes(*objects):
    """Memory held by the NumPy arrays in results (dataclasses, dicts, lists), each array counted once."""
    seen, total, stack = set(), 0, list(objects)
    while stack:
        value = stack.pop()
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, np.ndarray):
            total += value.nbytes
        elif isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
        elif is_dataclass(value):
            stack.extend(getattr(value, item.name, None) for item in fields(value))
    return total


def crop_image(image, bounds=None):
    """Return a crop and its [x, y] origin; bounds are half-open xyxy pixels."""
    if bounds is None:
        return image, np.zeros(2, dtype=int)
    x0, y0, x1, y1 = map(int, bounds)
    height, width = image.shape
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("The analysis region must lie inside the image and contain pixels.")
    if min(y1 - y0, x1 - x0) < 3:
        raise ValueError("The analysis region must be at least 3 x 3 pixels.")
    return image[y0:y1, x0:x1], np.array([x0, y0])


def segmentation_image(image, threshold):
    """(image, rescaled): OpenCV's triangle threshold requires uint8, so other types are scaled to 0-255 (the CD
    pipeline's rule, used for segmentation only)."""
    rescaled = image.dtype != np.uint8 and not (image.dtype == np.uint16 and threshold == "otsu")
    if not rescaled:
        return image, False
    data = np.asarray(image, dtype=np.float64)
    span = float(data.max() - data.min())
    return (np.rint((data - data.min()) * (255 / span)).astype(np.uint8) if span else np.zeros(data.shape, np.uint8),
            True)


def detect_review(image, parameters):
    from .centers import detect_centers
    from ._cpu.centers import _segment_components

    # Report the rescaling for segmentation explicitly to the reviewer.
    threshold = parameters["threshold"]
    segmentation, rescaled = segmentation_image(image, threshold)
    kwargs = {key: parameters[key] for key in ("threshold", "invert", "area_min", "area_max", "pad", "refine")}
    result = detect_centers(segmentation, backend=parameters["backend"], **kwargs)
    _, binary, _, labels, stats = _segment_components(segmentation, threshold=threshold, invert=parameters["invert"])
    boxes = stats[1:]
    kept = (boxes[:, 4] >= parameters["area_min"]) & (boxes[:, 4] <= parameters["area_max"])
    return DetectionReview(result, binary > 0, labels, boxes, kept, rescaled)


CD_KEYS = ("methods", "background", "seeds", "threshold", "invert", "area_min", "area_max", "pad", "n_angles",
           "radial_step", "smooth_sigma", "fit_fraction", "pixel_size", "unit", "backend", "half_area_level",
           "half_area_peak_pixels", "tail_area_level", "tail_window", "deblend_iterations", "deblend_tolerance",
           "nominal_cd", "psf_fwhm", "dark_level")


def cd_options(parameters):
    """The ``measure_cds`` keyword arguments of the workbench's settings (Measure CD and Measure all pages)."""
    options = {key: parameters[key] for key in CD_KEYS if key != "deblend_tolerance"}
    options["deblend_tolerance"] = parameters.get("deblend_tolerance", 1e-3)  # measure_cds's default
    return options


def pitch_numbers(profile):
    """(mean pitch over its pairs, pairs, lowest and highest placement error) of one axis' PitchProfile, in its unit;
    NaN where there are none."""
    pairs = int(np.sum(profile.counts))
    mean = float(np.nansum(profile.pitch * profile.counts) / pairs) if pairs else float("nan")
    error = profile.registration[np.isfinite(profile.registration)]
    low, high = (float(error.min()), float(error.max())) if error.size else (float("nan"), float("nan"))
    return mean, pairs, low, high


def analyze_image(image, parameters, *, operation="cd", bounds=None):
    """Run public repository methods without touching GUI state."""
    from .cd import measure_cds
    from .pattern import analyze_pitch

    started = perf_counter()
    if operation not in ("detect", "cd", "pitch"):
        raise ValueError("operation must be 'detect', 'cd' or 'pitch'")
    image, origin = crop_image(image, bounds)
    if parameters["area_max"] < parameters["area_min"]:
        raise ValueError("Maximum component area must be at least the minimum area.")
    review = AnalysisReview(image, origin, dict(parameters, bounds=bounds, operation=operation))
    if bounds is not None:
        review.notes.append("The analysis region is a crop. Features at its boundary may be clipped or excluded.")
    if operation == "detect":
        review.detection = detect_review(image, parameters)
        detected = review.detection
        review.notes.append(f"{len(detected.boxes)} components; {detected.kept.sum()} pass area limits; "
                            f"{len(detected.result.centers_xy)} finite refined centers.")
        if detected.rescaled:
            review.notes.append("Detection input was linearly rescaled for OpenCV; CD uses native image intensities.")
    else:
        if not parameters["methods"]:
            raise ValueError("Select at least one CD method.")
        review.cd = measure_cds(image, **cd_options(parameters))
        if review.cd.meta.get("deblend_converged") is False:
            review.warnings.append("Gaussian deblending reached its iteration limit. Inspect fit residuals and statuses.")
        for name, measured in review.cd.measurements.items():
            statuses, counts = np.unique(measured.status[~measured.valid], return_counts=True)
            if len(statuses):
                review.warnings.append(f"{name}: " + ", ".join(f"{s} ({c})" for s, c in zip(statuses, counts)))
    if not len(review.seeds_xy):
        review.warnings.append("No holes found. Review polarity, segmentation, area limits and region.")
    if operation == "pitch":
        source = review.cd
        if source is None:
            source = CDSummary(review.seeds_xy, review.measurements, parameters["pixel_size"], parameters["unit"])
        try:
            review.patterns = analyze_pitch(source, **parameters["pitch_options"])
        except ValueError as exc:
            # Keep the computed CD and segmentation available for debugging.
            review.warnings.append(f"Pitch indexing failed: {exc}")
    review.seconds = perf_counter() - started
    return review


def page_centers(image, parameters):
    """Hole centres [x, y] of a whole page with the Holes settings: thresholded components with a weighted
    centre, or intensity peaks for touching holes (the mean hole's boxes are placed at these)."""
    if parameters.get("seeds") == "peaks":
        from ._cpu.deblend import detect_peaks

        signal = np.asarray(image, dtype=np.float64)
        return detect_peaks(-signal if parameters["invert"] else signal)[0]
    from .centers import detect_centers

    segmentation, _ = segmentation_image(image, parameters["threshold"])
    return detect_centers(segmentation, backend=parameters["backend"], threshold=parameters["threshold"],
                          invert=parameters["invert"], area_min=parameters["area_min"],
                          area_max=parameters["area_max"], pad=parameters["pad"], refine="weighted").centers_xy


def preprocess_page(image, parameters):
    """The whole page made with ``parameters['preprocess']`` (method and its settings; 0 = automatic)."""
    from .preprocess import denoise_mean_hole, denoise_wiener, smooth_gaussian

    started = perf_counter()
    settings = parameters.get("preprocess") or {}
    method, centers = settings.get("method"), 0
    if method == "gaussian":
        result = smooth_gaussian(image, settings["sigma"])
    elif method == "wiener":
        result = denoise_wiener(image, settings.get("noise") or "auto")
    elif method == "mean_hole":
        points = page_centers(image, parameters)
        centers = len(points)
        result = denoise_mean_hole(image, points, components=settings.get("components") or "auto",
                                   radius=settings.get("radius") or "auto")
    else:
        raise ValueError(f"Unknown preprocessing method {method!r}: use gaussian, wiener or mean_hole.")
    return Preprocessed(result, dict(parameters), perf_counter() - started, centers)


def half_height_width(values, offsets):
    """Width of the central peak of a profile at half its height above the profile's lowest value (the
    background between holes; the ends may reach the neighbours), by linear interpolation; NaN if undefined."""
    values, offsets = np.asarray(values, dtype=np.float64), np.asarray(offsets, dtype=np.float64)
    if not np.isfinite(values).all() or len(values) < 3:
        return float("nan")
    peak = int(np.argmax(values))
    base = float(values.min())
    level = base + 0.5 * (values[peak] - base)
    if values[peak] <= base:
        return float("nan")
    left = np.flatnonzero(values[:peak] < level)
    right = np.flatnonzero(values[peak:] < level)
    if not len(left) or not len(right):
        return float("nan")
    i, j = left[-1], peak + right[0]
    x_left = offsets[i] + (level - values[i]) / (values[i + 1] - values[i]) * (offsets[i + 1] - offsets[i])
    x_right = offsets[j - 1] + (level - values[j - 1]) / (values[j] - values[j - 1]) * (offsets[j] - offsets[j - 1])
    return float(x_right - x_left)


# The GPU center finders of detect_centers (refine=...), offered for pitch beside the CD methods' own centers:
# refine key -> display word. Their pitch results are named "gpu_<key>".
GPU_CENTER_FINDERS = {"logquad": "log-quadratic", "weighted": "weighted", "edge_gradmoment": "edge-moment",
                      "edge_erf": "edge-erf", "radial_symmetry": "radial-symmetry",
                      "isophote_curvature": "isophote-curvature"}


def gpu_center_measurements(review, finders):
    """Centers of ``review``'s holes found by GPU center finders, as {``gpu_<finder>``: CDMethodResult}, and notes.

    Each finder runs ``detect_centers(backend="gpu", refine=finder)`` on the review's image with the settings its
    holes were found with (threshold, polarity, area limits, padding), on the segmentation input like Preview holes.
    Each hole of ``review.seeds_xy`` takes the nearest center within half the median spacing of neighbouring holes;
    a center goes only to its nearest hole. Holes without one are invalid (status ``no_center``). A finder that fails
    gets a note and no entry; the others still run.
    """
    from .centers import detect_centers

    seeds = np.asarray(review.seeds_xy, dtype=np.float64).reshape(-1, 2)
    found_all, notes = {}, []
    if not len(finders):
        return found_all, notes
    tolerance = np.inf
    if len(seeds) >= 2:
        from scipy.spatial import cKDTree

        tree = cKDTree(seeds)
        tolerance = 0.5 * float(np.median(tree.query(seeds, k=2)[0][:, 1]))
    parameters = review.parameters
    segmentation, rescaled = segmentation_image(review.image, parameters["threshold"])
    for finder in finders:
        name, label = f"gpu_{finder}", f"GPU {GPU_CENTER_FINDERS.get(finder, finder)} centers"
        started = perf_counter()
        try:
            if finder not in GPU_CENTER_FINDERS:
                raise ValueError(f"unknown finder; use one of {', '.join(GPU_CENTER_FINDERS)}")
            points = detect_centers(segmentation, backend="gpu", threshold=parameters["threshold"],
                                    invert=parameters["invert"], area_min=parameters["area_min"],
                                    area_max=parameters["area_max"], pad=parameters["pad"], refine=finder).centers_xy
        except Exception as exc:  # no CuPy, a finder that cannot run here: the others still run
            notes.append(f"{label} skipped: {exc}")
            continue
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        points = points[np.isfinite(points).all(axis=1)]
        centers = np.full(seeds.shape, np.nan)
        distance = np.full(len(seeds), np.nan)
        if len(points) and len(seeds):
            if len(seeds) >= 2:
                d, j = tree.query(points, distance_upper_bound=tolerance)
            else:
                d, j = np.hypot(*(points - seeds[0]).T), np.zeros(len(points), dtype=np.int64)
            near = np.isfinite(d)
            d, j, points = d[near], j[near], points[near]
            order = np.lexsort((d, j))  # by hole, nearest first
            holes, first = np.unique(j[order], return_index=True)
            centers[holes], distance[holes] = points[order[first]], d[order[first]]
        valid = np.isfinite(centers).all(axis=1)
        found_all[name] = CDMethodResult(name, centers, np.empty((len(seeds), 0, 2)), valid,
                                         np.where(valid, "ok", "no_center"), {"match_distance_px": distance})
        notes.append(f"{label}: {int(valid.sum()):,} of {len(seeds):,} holes have a center within "
                     f"{tolerance:.2f} px ({perf_counter() - started:.2f} s"
                     + (", on the image rescaled to 8 bits)" if rescaled else ")"))
    return found_all, notes


def analyze_patterns(review, options, extra=None):
    """Pitch of every method of ``review``; ``extra`` adds more centers of the same holes (e.g. GPU finders)."""
    from .pattern import analyze_pitch

    source = review.cd
    if source is None or extra:
        source = CDSummary(review.seeds_xy, {**review.measurements, **(extra or {})},
                           review.cd.pixel_size if review.cd is not None else review.parameters["pixel_size"],
                           review.cd.unit if review.cd is not None else review.parameters["unit"])
    return analyze_pitch(source, **options)


def cell_boundary(labels):
    """Pixel boundaries of the actual raster partition, including excluded cells."""
    edges = np.zeros(labels.shape, bool)
    edges[1:] |= labels[1:] != labels[:-1]
    edges[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    return edges


def feature_input(result, method, index):
    """Return the stored method input, mask and an honest description of its role."""
    measured = result.measurements[method]
    mask = result.masks[index]
    if "cells" in measured.inputs:
        data = measured.inputs["cells"][index]
        mask = measured.inputs.get("inside")
        mask = np.isfinite(data) if mask is None else mask[index] & np.isfinite(data)
        description = "Stored method input"
        if method.startswith("local_"):
            description = "Raw Voronoi cell (stored, polarity normalized)"
        elif method in ("tail_area", "tail_area_norm", "edge_area", "foot_area", "foot_area_ratio"):
            description = "Stored model-shared window center (display crop of the full measurement input)"
        return data, mask, description
    if method.startswith("local_"):
        ox, oy = result.origins_xy[index]
        yy, xx = np.indices(mask.shape)
        data = result.image[np.clip(yy + oy, 0, result.image.shape[0] - 1),
                            np.clip(xx + ox, 0, result.image.shape[1] - 1)].astype(float)
        if result.meta.get("invert"):
            data *= -1
        return np.where(mask, data, 0), mask, "Raw Voronoi cell (polarity normalized)"
    context_methods = ("tail_area", "tail_area_norm", "edge_area", "foot_area", "foot_area_ratio")
    description = "Cleaned cell" if result.meta.get("background") == "model" else "Border-filled cell"
    if method in context_methods:
        description += " for context; this method measures a wider model-shared window"
    return result.rois[index], mask, description


def feature_origin(result, method, index):
    origins = result.measurements[method].inputs.get("origins", result.origins_xy)
    # Wide-window origins are integer-valued but stored as float64.
    return np.rint(origins[index]).astype(np.int64)


def radial_diagnostic(result, method, index):
    """Recompute the same CPU ray samples used by the radial CD methods."""
    from ._cpu.cd import _background, _radial_profiles

    if method not in ("gradient", "halfmax"):
        return None
    roi, mask = result.rois[index], result.masks[index]
    if mask.sum() < 6:
        return None
    background = 0.0 if result.meta["background"] == "model" else _background(roi, mask)
    if np.max(roi[mask]) <= background:
        return None
    angles = np.arange(result.meta["n_angles"]) * 2 * np.pi / result.meta["n_angles"]
    center, radii, profiles, valid = _radial_profiles(
        roi, mask, background, angles, result.meta["radial_step"], result.meta["smooth_sigma"])
    return dict(center=center, radii=radii, profiles=profiles, valid=valid, angles=angles,
                edges=result.measurements[method].geometry["edge_px"][index])


def projection_diagnostic(result, method, index, axis="x"):
    """Expose the same zero-padded cubic profiles and slope picks as projection."""
    from ._cpu.invariant_cd import _spline_operators

    data, mask, _ = feature_input(result, method, index)
    background = 0.0
    if method == "local_projection":
        background = result.measurements[method].metrics["background"][index]
        if result.meta.get("invert"):
            background *= -1
    sum_axis = 0 if axis == "x" else 1
    profile = np.where(mask, data - background, 0).sum(axis=sum_axis)
    inside = mask.sum(axis=sum_axis) > 0
    if not inside.any():
        return None
    first, last = np.flatnonzero(inside)[[0, -1]]
    grid, value_op, slope_op = _spline_operators(len(profile), 0.02)
    values, slope = value_op @ profile, slope_op @ profile
    span = (grid >= first) & (grid <= last)
    peak = np.argmax(np.where(span, values, -np.inf))
    indices = np.arange(len(grid))
    left = np.argmax(np.where(span & (indices <= peak), slope, -np.inf))
    right = np.argmin(np.where(span & (indices >= peak), slope, np.inf))
    return dict(grid=grid, values=values, slope=slope, samples=profile, left=grid[left], right=grid[right], span=span)


# The batch's hole-check thresholds per mode (scripts/cd_batch.py --max-error defaults).
HOLE_CHECK_MAX_ERROR = {"centered": 0.8, "raw": 0.3}


def hole_check_error(reference, image, mode="centered"):
    """The batch's hole-check error between two images (scripts/cd_batch.py ``PccCheck``; a test pins them equal).

    scikit-image's whole-pixel phase-correlation error with ``normalization=None``: 0 when ``image`` is the
    reference at some shift, towards 1 for unrelated images. ``centered`` removes each image's mean first and
    rejects constant images (error 1); ``raw`` does not. Images of different sizes are compared on the
    centre part both share.
    """
    if mode not in HOLE_CHECK_MAX_ERROR:
        raise ValueError("hole-check mode must be 'raw' or 'centered'")
    reference, image = np.asarray(reference, dtype=np.float64), np.asarray(image, dtype=np.float64)
    h, w = min(reference.shape[0], image.shape[0]), min(reference.shape[1], image.shape[1])

    def prepare(data):
        y0, x0 = (data.shape[0] - h) // 2, (data.shape[1] - w) // 2
        data = data[y0:y0 + h, x0:x0 + w]
        if mode == "raw":
            return data, float((data * data).sum())
        mean = data.mean()
        data = data - mean
        energy = float((data * data).sum())
        scale = max(abs(float(mean)), 1.0)
        if not np.isfinite(energy) or energy <= data.size * (32 * np.finfo(float).eps * scale) ** 2:
            return data, 0.0
        return data, energy

    (ref, ref_energy), (mov, mov_energy) = prepare(reference), prepare(image)
    if mode == "centered" and (ref_energy <= 0 or mov_energy <= 0):
        return 1.0
    cc = np.fft.irfft2(np.fft.rfft2(ref) * np.conj(np.fft.rfft2(mov)), s=ref.shape)
    peak, amp = float(np.abs(cc).max()), ref_energy * mov_energy
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.sqrt(abs(1.0 - peak * peak / amp))) if amp > 0 else float("nan")


def register_images(reference, moving, *, upsample_factor=20, hole_check_mode="centered"):
    from scipy.ndimage import shift
    from .registration import estimate_shift

    if reference.shape != moving.shape:
        raise ValueError(f"Image registration requires matching shapes: {reference.shape} versus {moving.shape}.")
    if np.ptp(reference) == 0 or np.ptp(moving) == 0:
        raise ValueError("Image registration requires intensity variation in both images.")
    ref, mov = np.asarray(reference, float), np.asarray(moving, float)
    result = estimate_shift(ref, mov, backend="cpu", upsample_factor=upsample_factor)
    # NaN padding makes the overlap explicit and excludes padded borders from RMSE.
    aligned = shift(mov, result.shift_yx, order=1, mode="constant", cval=np.nan, prefilter=False)
    overlap = np.isfinite(aligned)
    if not overlap.any():
        raise ValueError("The estimated shift leaves no overlapping pixels.")
    before, after = mov - ref, aligned - ref
    cross = np.fft.fft2(ref) * np.fft.fft2(mov).conj()
    cross /= np.maximum(np.abs(cross), np.finfo(float).eps)
    correlation = np.fft.fftshift(np.abs(np.fft.ifft2(cross)))
    return RegistrationReview(result, aligned, overlap, before, after, correlation,
                              float(np.sqrt(np.mean(before[overlap] ** 2))),
                              float(np.sqrt(np.mean(after[overlap] ** 2))),
                              hole_check_error(ref, mov, hole_check_mode), hole_check_mode)


def export_review(review, directory):
    """Export native results plus a manifest describing crop-local coordinates."""
    from .cd import save_cd_results

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if review.cd is not None:
        save_cd_results(review.cd, directory)
    else:
        np.save(directory / "centers_global_xy.npy", review.seeds_xy + review.origin_xy)
        np.savez_compressed(directory / "detection.npz", binary=review.detection.binary,
                            labels=review.detection.labels, boxes=review.detection.boxes,
                            kept=review.detection.kept)
    for name, pattern in review.patterns.items():
        for axis, profile in pattern.profiles.items():
            with (directory / f"pitch_{name}_{axis}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["node", "position", "distance", "registration", "next_pitch", "next_std", "pair_count"])
                for i in range(len(profile.indices)):
                    writer.writerow([profile.indices[i], profile.position[i], profile.distance[i], profile.registration[i],
                                     profile.pitch[i] if i < len(profile.pitch) else "",
                                     profile.std[i] if i < len(profile.std) else "",
                                     profile.counts[i] if i < len(profile.counts) else ""])
    manifest = dict(parameters=review.parameters, origin_xy=review.origin_xy.tolist(),
                    coordinates="CD arrays are crop-local [x, y] pixels. Add origin_xy to obtain loaded-image coordinates.",
                    notes=review.notes, warnings=review.warnings, seconds=review.seconds)
    (directory / "workbench.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
