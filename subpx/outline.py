"""Outline fit: CD and centres from one sharp outline and one blur shared by every hole of the page.

    fit = subpx.fit_outline(image, seeds_xy, pixel_size=21, unit="nm")
    result = subpx.measure_cds(image, methods=("outline_fit", "half_area"), pixel_size=21, unit="nm")

A mask hole is a sharp outline; the tool's blur is smooth and the same for every hole of a page. The page is fitted
with one outline (a superellipse with harmonics: any star-convex shape) convolved with one blur, every hole at its
own position, brightness, local level and edge offset. A hole's CD is that of the outline moved by its edge offset:
the blur is fitted per page, so a focus change moves the blur, not the CD, and the brightness is free per hole.
Method, checks and limits: docs/outline_fit.md. The fit itself is ``subpx._outline`` (NumPy, or CuPy on the GPU).

    fast = subpx.fit_outline_fast(image, pixel_size=21, unit="nm")
    result = subpx.measure_cds(image, methods=("outline_fit_fast", "half_area"), pixel_size=21, unit="nm")

``fit_outline_fast`` (CD method ``outline_fit_fast``) fits the same model in one pass instead of 10 to 14 rounds: the
image's outline CD, x / y widths and blur in a fraction of a second, on the GPU only, and no size per hole (every
hole reads the image's CD). Method and measurements: docs/outline_fit_one_pass.md; the fit is
``subpx._outline.onepass`` and ``subpx._gpu.outline_onepass``.
"""

from __future__ import annotations

import math

import numpy as np

from .types import CDMethodResult, OutlineFastResult, OutlineFitResult

BLUR_FAMILIES = {"gauss2": 24, "lens": 48}  # blur family -> first template box (px); it doubles when light reaches past it
MIN_HOLES = 10  # fewer seeds than this: no fit (the page's outline and blur come from the average of its holes)
UPSAMPLE = 8  # template bins per pixel


def fit_outline(
    image: np.ndarray,
    seeds_xy: np.ndarray,
    *,
    pixel_size: float = 1.0,
    unit: str = "px",
    blur: str = "gauss2",
    rounds: int = 14,
    invert: bool = False,
    n_rays: int = 128,
    backend: str = "auto",
    device: int = 0,
    log=None,
) -> OutlineFitResult:
    """Fit one outline and one blur to the whole page and every hole's position, brightness, level and size.

    Parameters
    ----------
    image : (H, W) ndarray
        The page, bright holes (``invert=True``: dark holes on a bright background).
    seeds_xy : (N, 2) ndarray
        A starting position [x, y] for every hole on the page, including those cut by the frame (their light is part
        of the page model; they are fitted too). ``measure_cds`` passes its final seed list.
    blur : {'gauss2', 'lens'}
        The blur's family: two centred anisotropic Gaussians, or one anisotropic Gaussian times a circular pupil's
        transfer function with a free cut-off. On the real page tried both read the same CD to 0.1 nm.
    rounds : int
        Most alternations of the hole stage and the shape stage (the fit stops earlier once the outline settles;
        ``converged`` says whether it did).
    n_rays : int
        Directions of the outline returned per hole (``rays_px``), at least 16.
    backend : {'auto', 'cpu', 'gpu'}
        The GPU (CuPy) runs the same code 10-15 times faster: about 1.5-4.5 min for a 2048 x 1378 page of 30-50k holes
        on an RTX 4060 Ti, against 25-70 min on the CPU.
    log : callable, optional
        Called with one line of text per round.

    Returns
    -------
    OutlineFitResult
        NumPy arrays, lengths in pixels.
    """
    from .backends import gpu_device, resolve_backend

    img = np.asarray(image)
    if img.ndim != 2 or img.size == 0 or not np.issubdtype(img.dtype, np.number) or not np.isfinite(img).all():
        raise ValueError("image must be a nonempty 2D array of finite intensities")
    seeds = np.asarray(seeds_xy, dtype=np.float64).reshape(-1, 2)
    if len(seeds) < MIN_HOLES or not np.isfinite(seeds).all():
        raise ValueError(f"the outline fit needs at least {MIN_HOLES} finite seeds")
    if blur not in BLUR_FAMILIES:
        raise ValueError("blur must be one of " + ", ".join(BLUR_FAMILIES))
    if int(rounds) != rounds or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    if int(n_rays) != n_rays or n_rays < 16:
        raise ValueError("n_rays must be an integer of at least 16")
    if not (np.isfinite(pixel_size) and pixel_size > 0):
        raise ValueError("pixel_size must be positive")
    backend = resolve_backend(backend)
    if backend == "gpu":
        with gpu_device(device):
            return _fit(img, seeds, pixel_size, unit, blur, int(rounds), bool(invert), int(n_rays), True, log)
    return _fit(img, seeds, pixel_size, unit, blur, int(rounds), bool(invert), int(n_rays), False, log)


def _fit(img, seeds, pixel_size, unit, blur, rounds, invert, n_rays, gpu, log):
    from ._outline.model import Template
    from ._outline.page import PageFit

    # The fit models bright holes over a level with photon noise growing with the light: dark holes are turned into
    # bright ones by reflecting the grey levels (max - image), which keeps them non-negative.
    signal = img.astype(np.float64)
    top = float(signal.max())
    if invert:
        signal = top - signal
    xp = None
    if gpu:
        import cupy

        xp = cupy
    try:
        template = Template(upsample=UPSAMPLE, size_px=BLUR_FAMILIES[blur], blur=blur)
        fitter = PageFit(signal, seeds, pixel_nm=float(pixel_size), unit=unit, template=template, xp=xp)
        res = fitter.run(rounds=rounds, iterations=5, log=log)
        rounds_run = len(fitter.history)
        template = fitter.template  # possibly a doubled box
    finally:
        if gpu:
            from ._gpu._cuda import release_memory

            release_memory()
    theta = np.asarray(res["theta"], dtype=float)
    delta = np.asarray(res["edge_offset_px"], dtype=float)
    cd_x, cd_y = ellipse_widths(template, theta, delta)
    rays = outline_rays(template, theta, delta, n_rays)
    level = np.asarray(res["level"], dtype=float)
    if invert:
        level = top - level
    d = res["describe"]  # in pixel_size units
    fwhm = tuple(float(v) / pixel_size for v in d["blur_fwhm_nm"])
    summary = {
        "blur_family": blur, "rounds_max": int(rounds), "rounds_run": int(rounds_run),
        "converged": bool(res["converged"]), "per_hole_size": bool(res["per_hole_size"]),
        "outline_cd_px": float(res["cd_outline_px"]),
        "outline_extent_x_px": float(d["extent_x_nm"]) / pixel_size,
        "outline_extent_y_px": float(d["extent_y_nm"]) / pixel_size,
        "superellipse_p": float(d["superellipse"]["p"]), "outline_turn_deg": float(d["superellipse"]["phi_deg"]),
        "harmonics": {k: [float(v) for v in pair] for k, pair in d["harmonics"].items()},
        "blur_fwhm_px": [float(v) for v in fwhm],
        "blur_rms_px": [float(v) / pixel_size for v in d["blur_rms_nm"]],
        "holes_fitted": int(len(seeds)), "good_holes": int(np.sum(res["good"])),
        "spacing_px": float(res["spacing_px"]), "window_px": int(res["window_px"]),
        "template_box_px": int(template.N // template.U),
        "noise_variance": [float(v) for v in res["var_line"]],
        "edge_offset_clip_px": None if res["clip_px"] is None else float(res["clip_px"]),
        "edge_offset_mean_px": float(res["edge_offset_mean_px"]),
        "edge_offset_sd_median_px": float(np.nanmedian(res["delta_sd_px"])),
        "at_clip": int(np.sum(res["at_clip"])), "seconds": float(res["seconds"]),
        "backend": "gpu" if gpu else "cpu", "theta": [float(v) for v in theta],
        "history": [_plain(h) for h in res["history"]],
        "lengths": "px",
    }
    return OutlineFitResult(
        centers_xy=np.asarray(res["centers_xy"], dtype=float), cd_px=np.asarray(res["cd_px"], dtype=float),
        cd_x_px=cd_x, cd_y_px=cd_y, cd_se_px=np.asarray(res["cd_sd_px"], dtype=float), edge_offset_px=delta,
        brightness=np.asarray(res["brightness"], dtype=float), level=level, rays_px=rays,
        at_clip=np.asarray(res["at_clip"], dtype=bool), good=np.asarray(res["good"], dtype=bool),
        rms=np.asarray(res["rms"], dtype=float), outline_cd_px=float(res["cd_outline_px"]), blur_fwhm_px=fwhm,
        per_hole_size=bool(res["per_hole_size"]), converged=bool(res["converged"]), rounds=int(rounds_run),
        blur=blur, theta=theta, template=np.asarray(res["template"], dtype=float),
        stack=np.asarray(res["stack"], dtype=float), upsample=int(template.U), summary=summary,
        pixel_size=float(pixel_size), unit=str(unit),
    )


def _plain(value):
    """JSON-ready copy of a round's record (NumPy scalars and tuples to floats and lists)."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return value


def _tables(template, theta):
    """The outline r(angle) on the template's angle table and its normal-offset factor g = hypot(r, r') / r: the
    hole's own outline is r + delta g (``Template.indicator``), a move of delta along the outline's normal."""
    r, dr = template._radius_at(theta, template.angles)
    return template.angles, r, dr, np.hypot(r, dr) / r


def ellipse_widths(template, theta, delta):
    """x / y diameters of the equal-area ellipse with the region's x / y spread (cd_x cd_y = cd^2, as the area
    methods define them) for outlines r + delta g: the area, first and second moments are integrals of powers of
    r + delta g over the angle, so polynomials in delta, exact on the template's angle table."""
    angles, r, _, g = _tables(template, theta)
    c, s = np.cos(angles), np.sin(angles)

    def integral(k, weight):
        # int (r + delta g)^k weight d angle = sum_m C(k, m) delta^m int r^(k-m) g^m weight, highest power first
        coeffs = [math.comb(k, m) * 2 * np.pi * np.mean(r ** (k - m) * g ** m * weight) for m in range(k, -1, -1)]
        return np.polyval(coeffs, delta)

    area = integral(2, 1.0) / 2
    mx, my = integral(3, c) / 3, integral(3, s) / 3
    var_x = integral(4, c * c) / 4 / area - (mx / area) ** 2
    var_y = integral(4, s * s) / 4 / area - (my / area) ** 2
    cd = 2 * np.sqrt(np.maximum(area, 1e-12) / np.pi)
    ratio = np.sqrt(np.sqrt(np.maximum(var_x, 1e-12) / np.maximum(var_y, 1e-12)))
    return cd * ratio, cd / ratio


def outline_rays(template, theta, delta, n_rays=128):
    """Each hole's outline as its distance from the hole's centre (the outline's centroid) at angles 2 pi j / n.

    The page's outline is solved exactly per ray (Newton on the angle); a hole's offset delta moves it along the normal
    n, which lengthens the ray in direction e by delta / (e . n): exact to first order, off by about delta^2 / (2 x
    the edge's radius of curvature) (0.001 px at delta = 0.05 px on a 2.5 px radius). For drawing; the CD, x and y
    widths are exact (``ellipse_widths``)."""
    cx, cy = template.centroid_px(theta)
    phi = 2 * np.pi * np.arange(int(n_rays)) / int(n_rays)
    ex, ey = np.cos(phi), np.sin(phi)
    t = phi.copy()

    def point(t):
        r, dr = template._radius_at(theta, t)
        u, v = np.cos(t), np.sin(t)
        return r * u - cx, r * v - cy, r, dr, u, v

    for _ in range(30):
        vx, vy, r, dr, u, v = point(t)
        f = ex * vy - ey * vx  # the ray's direction crossed with (outline point - centre): zero on the ray
        # d/dt of (r u, r v) = r' u - r v, r' v + r u
        fp = ex * (dr * v + r * u) - ey * (dr * u - r * v)
        step = f / np.where(np.abs(fp) > 1e-12, fp, 1e-12)
        t = t - np.clip(step, -0.5, 0.5)
        if np.max(np.abs(step)) < 1e-13:
            break
    vx, vy, r, dr, u, v = point(t)
    rho0 = ex * vx + ey * vy
    nx, ny = r * u + dr * v, r * v - dr * u  # outward normal (unnormalised): r u - r' u_perp
    norm = np.hypot(nx, ny)
    stretch = norm / (ex * nx + ey * ny)
    return rho0[None, :] + np.asarray(delta, dtype=float)[:, None] * stretch[None, :]


def outline_measurement(fit: OutlineFitResult, rows=None, origins_xy=None) -> CDMethodResult:
    """The fit as a ``measure_cds`` method result for the holes ``rows`` (default all), aligned with the other methods.

    Valid: a finite reading whose edge offset is inside the fit's bound (``at_clip``: no reading). When the per-hole
    size was switched off (``fit.per_hole_size`` False), every hole reads the page's CD and its status is
    ``page_size``; the holes are valid (the page mean is a reading) but their spread is not one."""
    from ._cpu.cd import radial_contours

    rows = np.arange(len(fit.cd_px)) if rows is None else np.asarray(rows, dtype=np.int64)
    n = len(rows)
    centers = fit.centers_xy[rows]
    origins = np.zeros((n, 2)) if origins_xy is None else np.asarray(origins_xy, dtype=float).reshape(n, 2)
    edge = fit.rays_px[rows]
    cd = fit.cd_px[rows]
    finite = np.isfinite(cd) & np.isfinite(centers).all(axis=1) & np.isfinite(edge).all(axis=1)
    at_clip = fit.at_clip[rows]
    valid = finite & ~at_clip
    status = np.where(~finite, "not_finite", np.where(at_clip, "edge_offset_at_clip",
                                                      "ok" if fit.per_hole_size else "page_size")).astype("U32")
    center_local = centers - origins
    contours = radial_contours(center_local, edge, origins, edge.shape[1])
    segments = np.diff(contours, axis=1)

    def where_valid(values):
        return np.where(valid, values, np.nan)

    metrics = {
        "area_px2": where_valid(np.pi * cd * cd / 4), "cd_equivalent_px": where_valid(cd),
        "cd_x_px": where_valid(fit.cd_x_px[rows]), "cd_y_px": where_valid(fit.cd_y_px[rows]),
        "perimeter_px": where_valid(np.hypot(segments[..., 0], segments[..., 1]).sum(axis=1)),
        "cd_se_px": where_valid(fit.cd_se_px[rows]), "edge_offset_px": fit.edge_offset_px[rows].copy(),
        "brightness": fit.brightness[rows].copy(), "background": fit.level[rows].copy(),
        "leftover_rms": fit.rms[rows].copy(),
    }
    return CDMethodResult(method="outline_fit", centers_xy=centers.copy(), contours_xy=contours, valid=valid,
                          status=status, metrics=metrics,
                          geometry={"center_local": center_local, "edge_px": edge.copy()})


def measure_outline(image, seeds_xy, rows, origins_xy, *, pixel_size, unit, blur, rounds, invert, n_rays, backend,
                    device=0):
    """``measure_cds``'s ``outline_fit``: the fit of every seed, reported for ``rows``; (CDMethodResult, page summary).
    A page with fewer than ``MIN_HOLES`` seeds is not fitted: its holes are invalid with status ``too_few_holes``."""
    rows = np.asarray(rows, dtype=np.int64)
    if len(seeds_xy) < MIN_HOLES:
        n = len(rows)
        nan = np.full(n, np.nan)
        names = ("area_px2", "cd_equivalent_px", "cd_x_px", "cd_y_px", "perimeter_px", "cd_se_px", "edge_offset_px",
                 "brightness", "background", "leftover_rms")
        measured = CDMethodResult(
            method="outline_fit", centers_xy=np.full((n, 2), np.nan), contours_xy=np.full((n, n_rays + 1, 2), np.nan),
            valid=np.zeros(n, bool), status=np.full(n, "too_few_holes", dtype="U32"),
            metrics={name: nan.copy() for name in names},
            geometry={"center_local": np.full((n, 2), np.nan), "edge_px": np.full((n, n_rays), np.nan)})
        return measured, {"fitted": False, "reason": f"fewer than {MIN_HOLES} holes", "holes_fitted": int(len(seeds_xy))}
    fit = fit_outline(image, seeds_xy, pixel_size=pixel_size, unit=unit, blur=blur, rounds=rounds, invert=invert,
                      n_rays=n_rays, backend=backend, device=device)
    return outline_measurement(fit, rows, origins_xy), {"fitted": True, **fit.summary}


def fit_summary(fit_meta: dict | None, pixel_size: float = 1.0, unit: str = "px") -> dict | None:
    """The page readings of ``measure_cds``' ``meta["outline_fit"]`` in ``unit`` (None without the method)."""
    if not fit_meta:
        return None
    if not fit_meta.get("fitted"):
        return {"fitted": False, "reason": fit_meta.get("reason", "")}
    scale = float(pixel_size)
    return {"fitted": True, "outline_cd": fit_meta["outline_cd_px"] * scale,
            "blur_fwhm": [v * scale for v in fit_meta["blur_fwhm_px"]], "p": fit_meta["superellipse_p"],
            "rounds": fit_meta["rounds_run"], "rounds_max": fit_meta["rounds_max"], "converged": fit_meta["converged"],
            "per_hole_size": fit_meta["per_hole_size"], "blur_family": fit_meta["blur_family"],
            "seconds": fit_meta["seconds"], "holes_fitted": fit_meta["holes_fitted"], "at_clip": fit_meta["at_clip"],
            "unit": unit}


def fit_line(summary: dict | None, ascii_only: bool = False) -> str:
    """One line on a page's outline fit from ``fit_summary`` (``ascii_only`` for a console)."""
    if not summary:
        return ""
    if not summary["fitted"]:
        return f"not fitted ({summary['reason']})"
    u, x = summary["unit"], "x" if ascii_only else "×"
    state = (f"converged after {summary['rounds']} rounds" if summary["converged"]
             else f"NOT converged in {summary['rounds']} rounds (the outline was still moving)")
    size = ("a size per hole" if summary["per_hole_size"]
            else "per-hole size off: holes too small for a size each at this noise, every hole reads the image's CD")
    return (f"outline CD {summary['outline_cd']:.2f} {u}, blur FWHM {summary['blur_fwhm'][0]:.1f} {x} "
            f"{summary['blur_fwhm'][1]:.1f} {u} ({summary['blur_family']}), exponent p {summary['p']:.2f}; {state}; "
            f"{size}; {summary['holes_fitted']:,} holes fitted, {summary['at_clip']} at the edge-offset bound; "
            f"{summary['seconds']:.0f} s")


# ------------------------------------------------------------------------------------------- one-pass fit
_reading_templates: dict[str, object] = {}


def _reading_template(blur):
    """A ``Template`` for the page readings (CD, extents, blur widths) on ``fit_outline``'s first box of the blur
    form, so that the two methods' blur widths are read the same way. No coverage grid (``aa=1``): none is used."""
    from ._outline.model import Template

    if blur not in _reading_templates:
        _reading_templates[blur] = Template(upsample=UPSAMPLE, size_px=BLUR_FAMILIES[blur], blur=blur, aa=1)
    return _reading_templates[blur]


def _gpu_usable() -> bool:
    from .backends import cp

    try:
        return cp is not None and cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def fit_outline_fast(
    image: np.ndarray,
    seeds_xy: np.ndarray | None = None,
    *,
    pixel_size: float = 1.0,
    unit: str = "px",
    blur: str = "gauss2",
    invert: bool = False,
    n_rays: int = 128,
    backend: str = "auto",
    device: int = 0,
    log=None,
) -> OutlineFastResult:
    """Fit one outline and one blur to the whole image in one pass (GPU only); page values, no size per hole.

    The model, its parameters and the CD are ``fit_outline``'s. Instead of alternating a hole stage and a shape
    stage for 10 to 14 rounds, the image's holes are averaged as they are (binned at their sub-pixel offsets), the
    neighbours' light is put into the model of that average, and outline and blur are fitted to it once; the holes
    are then centred against the complete page model, and the fit is repeated with those centres. About 0.1 to
    0.3 s for a 2048 x 1378 image of 39,000 holes on an RTX 4060 Ti (the first call compiles the GPU routines:
    about half a minute, once per machine). Measurements and limits: docs/outline_fit_one_pass.md.

    Parameters
    ----------
    image : (H, W) ndarray
        The image, bright holes (``invert=True``: dark holes on a bright background).
    seeds_xy : (N, 2) ndarray, optional
        A starting position [x, y] for every hole of the image, including those cut by the frame. Default: the
        method's own seed finder (local maxima of the lightly smoothed image).
    blur : {'gauss2', 'lens'}
        The blur's family, as for ``fit_outline``.
    n_rays : int
        Directions of the page's outline returned (``rays_px``), at least 16.
    backend : {'auto', 'gpu', 'cpu'}
        There is no CPU path: 'cpu', or no usable GPU, raises ``RuntimeError``.
    log : callable, optional
        Called with one line of text per step.

    Returns
    -------
    OutlineFastResult
        NumPy arrays, lengths in pixels.
    """
    import time

    started = time.perf_counter()
    img = np.asarray(image)
    if img.ndim != 2 or img.size == 0 or not np.issubdtype(img.dtype, np.number) or not np.isfinite(img).all():
        raise ValueError("image must be a nonempty 2D array of finite intensities")
    seeds = None
    if seeds_xy is not None:
        seeds = np.asarray(seeds_xy, dtype=np.float64).reshape(-1, 2)
        if len(seeds) < MIN_HOLES or not np.isfinite(seeds).all():
            raise ValueError(f"the outline fit needs at least {MIN_HOLES} finite seeds")
    if blur not in BLUR_FAMILIES:
        raise ValueError("blur must be one of " + ", ".join(BLUR_FAMILIES))
    if int(n_rays) != n_rays or n_rays < 16:
        raise ValueError("n_rays must be an integer of at least 16")
    if not (np.isfinite(pixel_size) and pixel_size > 0):
        raise ValueError("pixel_size must be positive")
    if backend not in ("auto", "gpu", "cpu"):
        raise ValueError("backend must be 'auto', 'gpu' or 'cpu'")
    if backend == "cpu" or not _gpu_usable():
        raise RuntimeError("fit_outline_fast needs the GPU (CuPy); it has no CPU path (fit_outline has one)")
    from .backends import gpu_device

    # bright holes over a level, as fit_outline: dark holes are reflected (max - image)
    signal = img.astype(np.float64)
    top = float(signal.max())
    if invert:
        signal = top - signal
    with gpu_device(device):
        import cupy

        from ._gpu._cuda import release_memory
        from ._gpu.outline_onepass import find_seeds, fit_images, lib_template

        try:
            if seeds is None:
                seeds = cupy.asnumpy(find_seeds(cupy.asarray(signal).astype(cupy.float32)))
                if len(seeds) < MIN_HOLES:
                    raise ValueError(f"the outline fit needs at least {MIN_HOLES} holes; {len(seeds)} were found")
            res = fit_images([signal], [seeds], blur=blur, log=log)[0]
        finally:
            release_memory()
    template = lib_template(blur)
    theta = res["theta"]
    zero = np.zeros(1)
    cd = float(template.cd_px(theta))
    cd_x, cd_y = (float(v[0]) for v in ellipse_widths(template, theta, zero))
    rays = outline_rays(template, theta, zero, int(n_rays))[0]
    centers = res["centers_xy"] + np.asarray(template.centroid_px(theta))
    level = top - res["level"] if invert else res["level"]
    d = _reading_template(blur).describe(theta, 1.0)  # in pixels
    fwhm = tuple(float(v) for v in d["blur_fwhm_nm"])
    summary = {
        "blur_family": blur, "outline_cd_px": cd,
        "outline_extent_x_px": float(d["extent_x_nm"]), "outline_extent_y_px": float(d["extent_y_nm"]),
        "superellipse_p": float(d["superellipse"]["p"]), "outline_turn_deg": float(d["superellipse"]["phi_deg"]),
        "harmonics": {k: [float(v) for v in pair] for k, pair in d["harmonics"].items()},
        "blur_fwhm_px": [float(v) for v in fwhm], "blur_rms_px": [float(v) for v in d["blur_rms_nm"]],
        "holes_fitted": int(len(seeds)), "holes_stacked": int(np.sum(res["good"])),
        "spacing_px": float(res["spacing_px"]), "window_px": int(res["window_px"]),
        "pair_reach_px": float(res["pair_reach_px"]), "noise_sd": float(res["noise_sd"]),
        "leftover_chi2": float(res["chi2"]) / max(int(res["bins"]), 1),
        "first_cd_px": float(template.cd_px(res["theta_first"])),
        "fit_iterations": int(res["iterations"]), "converged": bool(res["converged"]),
        "exchanged": bool(res["exchanged"]), "per_hole_size": False,
        "seconds": time.perf_counter() - started, "backend": "gpu", "theta": [float(v) for v in theta],
        "lengths": "px",
    }
    return OutlineFastResult(
        centers_xy=centers, brightness=res["brightness"], level=level, rms=res["rms"],
        good=np.asarray(res["good"], dtype=bool), outline_cd_px=cd, cd_x_px=cd_x, cd_y_px=cd_y, rays_px=rays,
        blur_fwhm_px=fwhm, theta=theta, stack=res["stack"], stack_weight=res["stack_weight"],
        upsample=int(res["upsample"]), blur=blur, summary=summary, pixel_size=float(pixel_size), unit=str(unit),
    )


def blur_second(theta, blur: str) -> str:
    """The blur's second part of a fitted ``theta`` (of ``fit_outline`` or ``fit_outline_fast``) in words, in
    pixels; '' for a form without one. On some images the fit has more than one end point, and this is what tells
    them apart: two fits whose page CDs differ and whose second parts differ sit at different ends of the same
    model (docs/outline_fit_fast_handoff.md, 6.3)."""
    theta = np.asarray(theta, dtype=float)
    if blur == "gauss2":
        s2x, s2y = np.exp(theta[-3:-1])
        return f"second Gaussian {s2x:.1f} x {s2y:.1f} px, {100 / (1 + math.exp(-theta[-1])):.0f} % of the light"
    if blur == "lens":
        return f"pupil cut-off {math.exp(theta[-1]):.2f} cycles/px"
    return ""


FAST_METRICS = ("area_px2", "cd_equivalent_px", "cd_x_px", "cd_y_px", "perimeter_px", "brightness", "background",
                "leftover_rms")


def measure_outline_fast(image, seeds_xy, rows, origins_xy, *, pixel_size, unit, blur, invert, n_rays, backend,
                         device=0):
    """``measure_cds``'s ``outline_fit_fast``: the one-pass fit of every seed, reported for ``rows``;
    (CDMethodResult, page summary). Every reported hole reads the image's CD (status ``page_size``): the mean is a
    reading, the spread is not one. Not fitted, with every hole invalid and the reason in the summary: fewer than
    ``MIN_HOLES`` seeds (status ``too_few_holes``), a run on the CPU (``needs_gpu``), or an image the fit cannot
    stack (``not_fitted``). None of these raises: the image's other methods are kept."""
    from ._cpu.cd import radial_contours

    rows = np.asarray(rows, dtype=np.int64)
    n = len(rows)

    def unfitted(status, reason):
        measured = CDMethodResult(
            method="outline_fit_fast", centers_xy=np.full((n, 2), np.nan),
            contours_xy=np.full((n, n_rays + 1, 2), np.nan), valid=np.zeros(n, bool),
            status=np.full(n, status, dtype="U32"), metrics={name: np.full(n, np.nan) for name in FAST_METRICS},
            geometry={"center_local": np.full((n, 2), np.nan), "edge_px": np.full((n, n_rays), np.nan)})
        return measured, {"fitted": False, "reason": reason, "holes_fitted": int(len(seeds_xy))}

    if len(seeds_xy) < MIN_HOLES:
        return unfitted("too_few_holes", f"fewer than {MIN_HOLES} holes")
    if backend != "gpu":
        return unfitted("needs_gpu", "needs the GPU (CuPy); this run uses the CPU")
    try:
        fit = fit_outline_fast(image, seeds_xy, pixel_size=pixel_size, unit=unit, blur=blur, invert=invert,
                               n_rays=n_rays, backend="gpu", device=device)
    except ValueError as error:  # the options were checked by measure_cds: this is about the image
        return unfitted("not_fitted", str(error))
    centers = fit.centers_xy[rows]
    origins = np.asarray(origins_xy, dtype=float).reshape(n, 2)
    valid = np.isfinite(centers).all(axis=1) & bool(np.isfinite(fit.outline_cd_px))
    status = np.where(valid, "page_size", "not_finite").astype("U32")
    center_local = centers - origins
    edge = np.repeat(fit.rays_px[None, :], n, axis=0)
    contours = radial_contours(center_local, edge, origins, edge.shape[1])
    segments = np.diff(contours, axis=1)

    def page(value):
        return np.where(valid, value, np.nan)

    cd = fit.outline_cd_px
    metrics = {
        "area_px2": page(np.pi * cd * cd / 4), "cd_equivalent_px": page(cd), "cd_x_px": page(fit.cd_x_px),
        "cd_y_px": page(fit.cd_y_px),
        "perimeter_px": np.where(valid, np.hypot(segments[..., 0], segments[..., 1]).sum(axis=1), np.nan),
        "brightness": fit.brightness[rows].copy(), "background": fit.level[rows].copy(),
        "leftover_rms": fit.rms[rows].copy(),
    }
    measured = CDMethodResult(method="outline_fit_fast", centers_xy=centers.copy(), contours_xy=contours, valid=valid,
                              status=status, metrics=metrics,
                              geometry={"center_local": center_local, "edge_px": edge})
    return measured, {"fitted": True, **fit.summary}


def fast_summary(fit_meta: dict | None, pixel_size: float = 1.0, unit: str = "px") -> dict | None:
    """The page readings of ``measure_cds``' ``meta["outline_fit_fast"]`` in ``unit`` (None without the method)."""
    if not fit_meta:
        return None
    if not fit_meta.get("fitted"):
        return {"fitted": False, "reason": fit_meta.get("reason", "")}
    scale = float(pixel_size)
    return {"fitted": True, "outline_cd": fit_meta["outline_cd_px"] * scale,
            "blur_fwhm": [v * scale for v in fit_meta["blur_fwhm_px"]], "blur_family": fit_meta["blur_family"],
            "blur_second": blur_second(fit_meta["theta"], fit_meta["blur_family"]), "p": fit_meta["superellipse_p"],
            "leftover_chi2": fit_meta["leftover_chi2"], "first_cd": fit_meta["first_cd_px"] * scale,
            "holes_stacked": fit_meta["holes_stacked"], "holes_fitted": fit_meta["holes_fitted"],
            "converged": fit_meta["converged"], "iterations": fit_meta["fit_iterations"],
            "exchanged": fit_meta["exchanged"], "seconds": fit_meta["seconds"], "unit": unit}


def fast_line(summary: dict | None, ascii_only: bool = False) -> str:
    """One line on an image's one-pass outline fit from ``fast_summary`` (``ascii_only`` for a console)."""
    if not summary:
        return ""
    if not summary["fitted"]:
        return f"not fitted ({summary['reason']})"
    u, x = summary["unit"], "x" if ascii_only else "×"
    second = f"; {summary['blur_second']}" if summary["blur_second"] else ""
    state = "" if summary["converged"] else f"; final fit NOT converged in {summary['iterations']} iterations"
    return (f"outline CD {summary['outline_cd']:.2f} {u}, blur FWHM {summary['blur_fwhm'][0]:.1f} {x} "
            f"{summary['blur_fwhm'][1]:.1f} {u} ({summary['blur_family']}{second}), exponent p {summary['p']:.2f}; "
            f"leftover {summary['leftover_chi2']:.2f} (chi-square per bin: 1 = at the noise); first fit "
            f"{summary['first_cd']:.2f} {u}{state}; {summary['holes_stacked']:,} of {summary['holes_fitted']:,} holes "
            f"stacked; page size for every hole; {summary['seconds']:.2f} s")
