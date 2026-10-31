"""Which settings each step and result uses, what changed since a result, and why a step cannot run (Qt-free)."""

from __future__ import annotations

from .names import CALIBRATED_METHODS, method_name

SETTINGS_KEY = "workbench/v1"
STEPS = ("image", "holes", "cd", "pitch")
STEP_TITLES = {"image": "Image", "holes": "Holes", "cd": "CD methods", "pitch": "Pitch"}
# The settings of each Setup step, as analysis parameters.
STEP_KEYS = {
    "image": ("pixel_size", "unit", "invert", "backend"),
    "holes": ("seeds", "threshold", "area_min", "area_max", "pad", "refine", "bounds"),
    "cd": ("methods", "background", "n_angles", "radial_step", "smooth_sigma", "fit_fraction", "half_area_level",
           "half_area_peak_pixels", "tail_area_level", "tail_window", "deblend_iterations", "deblend_tolerance",
           "nominal_cd", "psf_fwhm", "dark_level"),
    "pitch": ("pitch_source", "reference_selected", "pitch_options"),
}
# The settings each kind of result was made with (pitch also depends on the result it was made from).
RESULT_KEYS = {
    "detect": ("pixel_size", "unit", "invert", "backend", "threshold", "area_min", "area_max", "pad", "refine",
               "bounds"),
    "cd": tuple(key for key in STEP_KEYS["image"] + STEP_KEYS["holes"] + STEP_KEYS["cd"] if key != "refine"),
    "pitch": STEP_KEYS["pitch"],
}
LABELS = {
    "pixel_size": "pixel size", "unit": "unit", "invert": "dark holes", "backend": "compute", "seeds": "pattern",
    "threshold": "threshold", "area_min": "area min", "area_max": "area max", "pad": "padding",
    "refine": "center refinement", "bounds": "analysis region", "methods": "methods", "background": "background",
    "n_angles": "rays", "radial_step": "ray step", "smooth_sigma": "smoothing", "fit_fraction": "fit fraction",
    "half_area_level": "half-height level", "half_area_peak_pixels": "peak area", "tail_area_level": "tail level",
    "tail_window": "wide window", "deblend_iterations": "fit iterations", "deblend_tolerance": "stop tolerance",
    "nominal_cd": "nominal CD", "psf_fwhm": "PSF FWHM", "dark_level": "dark level", "pitch_source": "pitch centers",
    "reference_selected": "reference at selected hole", "layout": "lattice", "stagger_axis": "stagger direction",
    "average_rows": "average rows", "average_columns": "average columns", "nominal_pitch_xy": "design pitch",
    "pitch_hint_xy": "pitch hints",
}
VALUE_TEXT = {"otsu": "Otsu", "triangle": "Triangle", "threshold": "separate", "peaks": "touching", "auto": "Auto",
              "gpu": "GPU", "cpu": "CPU", "model": "neighbour model", "border": "cell border", "cd": "CD",
              "detection": "previewed holes", "square": "square", "staggered": "staggered", "row": "rows",
              "column": "columns", "logquad": "log fit", "weighted": "weighted", "edge_gradmoment": "edge gradient",
              "none": "none"}


def _text(value):
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_text(item) for item in value) + "]"
    return VALUE_TEXT.get(value, str(value)) if isinstance(value, str) else str(value)


def _same(old, new):
    if isinstance(old, (list, tuple)) and isinstance(new, (list, tuple)):
        return len(old) == len(new) and all(_same(a, b) for a, b in zip(old, new))
    return old == new


def changes(old, new, keys):
    """[(key, text)] of the settings in ``keys`` that differ, e.g. ('area_max', 'area max 200 -> 150')."""
    found = []
    for key in keys:
        if key == "pitch_options":
            before, after = old.get(key) or {}, new.get(key) or {}
            # The reference position follows the selection; "reference at selected hole" stands for it.
            found += [(sub, f"{LABELS.get(sub, sub)} {_text(before.get(sub))} -> {_text(after.get(sub))}")
                      for sub in sorted((set(before) | set(after)) - {"reference_xy"})
                      if not _same(before.get(sub), after.get(sub))]
            continue
        a, b = old.get(key), new.get(key)
        if key == "reference_selected":
            a, b = bool(a), bool(b)
        if _same(a, b):
            continue
        if key == "methods":
            a, b = list(a or ()), list(b or ())
            parts = [f"+{method_name(m)}" for m in b if m not in a] + [f"-{method_name(m)}" for m in a if m not in b]
            found.append((key, "methods " + (" ".join(parts[:3]) + (" ..." if len(parts) > 3 else "")
                                             if parts else "reordered")))
        else:
            found.append((key, f"{LABELS.get(key, key)} {_text(a)} -> {_text(b)}"))
    return found


def describe(found, limit=2):
    """'area max 200 -> 150, threshold Otsu -> Triangle (+1 more)'."""
    texts = [text for _, text in found]
    more = len(texts) - limit
    return ", ".join(texts[:limit]) + (f" (+{more} more)" if more > 0 else "")


def run_problems(parameters, *, loaded=None, results=None, moving=None):
    """Why each run cannot start now ('' = it can) and the setting keys at fault.

    ``results`` is the active page's PageResults (pitch uses its result on screen, never a hidden re-run).
    """
    problems = dict(detect="", cd="", pitch="", register="")
    invalid = set()
    if loaded is None:
        return dict.fromkeys(problems, "Open an image first."), invalid
    segmentation = ""
    if parameters.get("area_min", 0) > parameters.get("area_max", 0):
        segmentation = f"Area min ({parameters['area_min']}) is above area max ({parameters['area_max']})."
        invalid |= {"area_min", "area_max"}
    selected = set(parameters.get("methods") or ())
    cd = segmentation
    if not cd and not selected:
        cd = "Tick at least one CD method."
    elif not cd and selected & CALIBRATED_METHODS and not (parameters.get("nominal_cd") or parameters.get("psf_fwhm")):
        cd = ", ".join(method_name(m) for m in sorted(selected & CALIBRATED_METHODS)) + " needs a nominal CD or PSF FWHM."
        invalid |= {"nominal_cd", "psf_fwhm"}
    pitch = ""
    source = parameters.get("pitch_source", "cd")
    if source == "cd" and (results is None or results.cd is None):
        pitch = "Measure CD first: pitch uses the CD result on screen."
    elif source == "detection" and (results is None or results.detect is None):
        pitch = "Preview holes first: pitch uses the previewed holes."
    for key, what in (("nominal", "design pitches"), ("hint", "pitch hints")):
        x, y = parameters.get(f"{key}_x", 0), parameters.get(f"{key}_y", 0)
        if (x == 0) != (y == 0):
            pitch = pitch or f"Set both {what} (x and y), or neither."
            invalid |= {f"{key}_x", f"{key}_y"}
    register = ""
    if moving is None:
        register = "Pick a moving page: of this file or of another file."
    elif moving.image.shape != loaded.image.shape:
        (mh, mw), (h, w) = moving.image.shape, loaded.image.shape
        register = f"The moving image is {mw} x {mh} px; this page is {w} x {h} px."
    problems.update(detect=segmentation, cd=cd, pitch=pitch, register=register)
    return problems, invalid
