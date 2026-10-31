"""Display names, colours and one-line texts of the workbench (Qt-free).

One display name per method (library keys and labels stay in tooltips and exports), plain words for statuses
and metrics, and plain one-line errors.
"""

from __future__ import annotations

import re

from ..cd import METHOD_LABELS

# The default methods of scripts/cd_batch.py (ALL_METHODS of scripts/cd_report.py); a test pins them.
BATCH_METHODS = ("gaussian", "projection", "half_area", "half_area_min", "half_area_min_avg", "tail_area",
                 "tail_area_norm", "edge_area", "foot_area", "foot_area_ratio")
RECIPES = {
    "batch": ("Batch default", BATCH_METHODS),
    "spot": ("Spot widths", ("gaussian", "logquad", "gradient", "halfmax")),
    "local": ("Raw-cell widths", ("local_projection", "local_half_area")),
}
CALIBRATED_METHODS = {"disk", "flux_ratio"}
METHOD_NAMES = {
    "gaussian": "Gaussian fit", "logquad": "Log fit", "gradient": "Gradient edge", "halfmax": "Half-height",
    "disk": "Blurred-hole fit", "flux_ratio": "Flux ratio", "projection": "Projection", "half_area": "Half area",
    "half_area_min": "Half area, min level", "half_area_min_avg": "Half area, min level, avg peak",
    "tail_area": "Tail area", "tail_area_norm": "Tail area, no level", "edge_area": "Edge area",
    "foot_area": "Foot area", "foot_area_ratio": "Foot area, image ratio",
    "local_projection": "Raw-cell projection", "local_half_area": "Raw-cell half area",
    "local_half_area_min_avg": "Raw-cell half area, min level, avg peak", "local_tail_area": "Raw-cell tail area",
    "pixel_half_area": "Pixel half area", "pixel_half_area_min_avg": "Pixel half area, min level, avg peak",
    "pixel_tail_area": "Pixel tail area", "detection": "Previewed holes",
}
# Setup lists the methods in these groups, in this order.
METHOD_GROUPS = (
    ("Spot fits", ("gaussian", "logquad")),
    ("Radial edges", ("gradient", "halfmax")),
    ("Projections", ("projection",)),
    ("Area methods", ("half_area", "half_area_min", "half_area_min_avg", "tail_area", "tail_area_norm",
                      "edge_area", "foot_area", "foot_area_ratio")),
    ("Raw-cell methods", ("local_projection", "local_half_area", "local_half_area_min_avg", "local_tail_area")),
    ("Calibrated (need a nominal CD or PSF)", ("disk", "flux_ratio")),
)
METHOD_HELP = {
    "gaussian": "Joint Gaussian model. Neighbour signals and a smooth background are removed. The contour is the fitted half-maximum ellipse. Inspect fit_rmse and deblend convergence; a Gaussian shape is an assumption.",
    "logquad": "Weighted quadratic fit to log(I - background), above the fit-fraction cutoff. The negative Hessian defines the half-maximum ellipse. Inspect log_fit_rmse, fit support and whether the contour stays inside the cell.",
    "gradient": "Cubic samples along radial rays. The edge is the maximum falling slope with parabolic refinement. Missing rays make a measurement invalid.",
    "halfmax": "Cubic samples along radial rays. Each edge is the first crossing of half the center signal, interpolated between samples.",
    "projection": "Summed x/y profiles of the cleaned cell. Width is the distance between the steepest rising and falling slopes of a zero-padded cubic spline.",
    "local_projection": "Summed x/y profiles of the raw Voronoi cell after subtracting its median edge level. Width is the distance between the steepest rise and fall. Neighbours are not subtracted.",
    "disk": "A round hole convolved with an imaging blur, fitted with free brightness and background. Requires a known PSF FWHM or nominal CD. Its diameter has a different definition from spot FWHM.",
    "flux_ratio": "Hole light divided by its own fitted background minus detector dark level. Requires a nominal CD or PSF calibration. A calibration derived per image should not be used to compare image means.",
}
AREA_HELP = ("The peak-connected region above the method's threshold is measured on a 4x cubic enlargement. "
             "Equivalent CD = 2 sqrt(area / pi). The outline, threshold and local level depend on the method; "
             "x/y widths use area and second moments.")
# Method colours (contours, inspector dots): distinct on the dark background, none of them the failure red.
METHOD_COLORS = ("#4dd0e1", "#ffd166", "#b39dff", "#f78c6b", "#8bd17c", "#ff7edb", "#6fa8ff", "#e0e0e0",
                 "#d4b483", "#2fbf71", "#c792ea", "#a3b8cc")
FAILED_COLOR = "#ff5c74"
STATUS_TEXT = {
    "ok": "OK",
    "no_signal": "No signal above the background",
    "region_at_cell_border": "Region reaches the cell border",
    "edge_at_cell_border": "Edge at the cell border",
    "non_gaussian_fit": "Fit is not Gaussian-shaped",
    "insufficient_cell_pixels": "Too few cell pixels",
    "insufficient_fit_pixels": "Too few pixels for the fit",
    "at_diameter_limit": "At the diameter limit",
    "singular_fit": "Fit undetermined",
    "not_converged": "Fit did not converge",
    "incomplete_rays": "Edge not found on every ray",
    "fit_failed": "Fit failed",
    "fit_diverged": "Fit diverged",
    "center_outside_cell": "Center outside its cell",
}
METRIC_TEXT = {
    "cd_equivalent": "Equivalent CD", "cd_x": "CD x", "cd_y": "CD y", "area": "Area", "perimeter": "Perimeter",
    "peak_signal": "Peak signal", "fit_rmse": "Fit RMSE", "log_fit_rmse": "Log fit RMSE", "threshold": "Threshold",
    "cell_area": "Cell area", "mean_original_intensity": "Mean intensity (original pixels)",
}


def method_name(key):
    return METHOD_NAMES.get(key, key)


def method_line(key):
    """One line on what the method measures (the library's label)."""
    return METHOD_LABELS.get(key, "Subpixel hole centers" if key == "detection" else key)


def method_help(key):
    if key == "detection":
        return "Threshold, components, area filter and center refinement. Measure CD for widths and cell diagnostics."
    return METHOD_HELP.get(key, AREA_HELP)


def method_color(index):
    return METHOD_COLORS[index % len(METHOD_COLORS)]


def status_text(code):
    """Plain words for a library status code ('incomplete_rays' -> 'Edge not found on every ray')."""
    code = str(code)
    return STATUS_TEXT.get(code, code.replace("_", " ").capitalize())


def metric_label(key, unit):
    """(readable name with unit, factor of pixel size) for a metric key: 'cd_x_px' -> ('CD x (nm)', 1)."""
    power, base = 0, key
    if key.endswith("_px2"):
        power, base = 2, key[:-4]
    elif key.endswith("_px"):
        power, base = 1, key[:-3]
    name = METRIC_TEXT.get(base, base.replace("_", " ").capitalize())
    suffix = {0: "", 1: f" ({unit})", 2: f" ({unit}²)"}[power]
    return name + suffix, power


def plain_error(detail):
    """The last line of a traceback without the exception class, e.g. 'Area max is below area min.'"""
    lines = [line for line in str(detail).strip().splitlines() if line.strip()]
    line = lines[-1].strip() if lines else "Unknown error"
    match = re.match(r"^[\w.]*(?:Error|Exception|Exit)\w*: (.+)$", line)
    return match.group(1) if match else line
