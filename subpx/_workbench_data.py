"""Qt-free analysis and diagnostic data for the desktop workbench."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np

from .types import CDMethodResult, CDResult, CDSummary, CenterResult, PatternPitchResult, ShiftResult


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
    notes: list[str] = field(default_factory=list)
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
class RegistrationReview:
    result: ShiftResult
    aligned: np.ndarray
    overlap: np.ndarray
    difference_before: np.ndarray
    difference_after: np.ndarray
    correlation: np.ndarray
    rmse_before: float
    rmse_after: float


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


def detect_review(image, parameters):
    from .centers import detect_centers
    from ._cpu.centers import _segment_components

    # OpenCV's triangle threshold requires uint8. Use the CD pipeline's scaling
    # rule for segmentation, and report this explicitly to the reviewer.
    threshold = parameters["threshold"]
    rescaled = image.dtype != np.uint8 and not (image.dtype == np.uint16 and threshold == "otsu")
    segmentation = image
    if rescaled:
        data = np.asarray(image, dtype=np.float64)
        span = float(data.max() - data.min())
        segmentation = np.rint((data - data.min()) * (255 / span)).astype(np.uint8) if span else np.zeros(data.shape, np.uint8)
    kwargs = {key: parameters[key] for key in ("threshold", "invert", "area_min", "area_max", "pad", "refine")}
    result = detect_centers(segmentation, backend=parameters["backend"], **kwargs)
    _, binary, _, labels, stats = _segment_components(segmentation, threshold=threshold, invert=parameters["invert"])
    boxes = stats[1:]
    kept = (boxes[:, 4] >= parameters["area_min"]) & (boxes[:, 4] <= parameters["area_max"])
    return DetectionReview(result, binary > 0, labels, boxes, kept, rescaled)


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
        keys = ("methods", "background", "seeds", "threshold", "invert", "area_min", "area_max", "pad",
                "n_angles", "radial_step", "smooth_sigma", "fit_fraction", "pixel_size", "unit", "backend",
                "half_area_level", "half_area_peak_pixels", "tail_area_level", "tail_window", "deblend_iterations",
                "nominal_cd", "psf_fwhm", "dark_level")
        review.cd = measure_cds(image, **{key: parameters[key] for key in keys})
        if review.cd.meta.get("deblend_converged") is False:
            review.notes.append("Gaussian deblending reached its iteration limit. Inspect fit residuals and statuses.")
        for name, measured in review.cd.measurements.items():
            statuses, counts = np.unique(measured.status[~measured.valid], return_counts=True)
            if len(statuses):
                review.notes.append(f"{name}: " + ", ".join(f"{s} ({c})" for s, c in zip(statuses, counts)))
    if not len(review.seeds_xy):
        review.notes.append("No features found. Review polarity, segmentation, area limits and region.")
    if operation == "pitch":
        source = review.cd
        if source is None:
            source = CDSummary(review.seeds_xy, review.measurements, parameters["pixel_size"], parameters["unit"])
        try:
            review.patterns = analyze_pitch(source, **parameters["pitch_options"])
        except ValueError as exc:
            # Keep the computed CD and segmentation available for debugging.
            review.notes.append(f"Pitch indexing failed: {exc}")
    review.seconds = perf_counter() - started
    return review


def analyze_patterns(review, options):
    from .pattern import analyze_pitch

    source = review.cd or CDSummary(review.seeds_xy, review.measurements,
                                   review.parameters["pixel_size"], review.parameters["unit"])
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


def register_images(reference, moving, *, upsample_factor=20):
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
                              float(np.sqrt(np.mean(after[overlap] ** 2))))


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
                    notes=review.notes, seconds=review.seconds)
    (directory / "workbench.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
