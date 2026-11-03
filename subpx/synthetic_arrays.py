"""Synthetic hole arrays with exactly known centers, for testing center and pitch methods.

Each hole's light is computed from its shape's exact Fourier transform (circle, square or ellipse), times the
blur's transfer function and the pixel's area, shifted by the hole's exact center and summed into the page. No
drawing grid is involved, so on these images a method's center error is the method's own: the rendering adds
none, at any sub-pixel position.

The defaults follow the user's ACTIS pages (21 nm pixels, 8-bit, bright holes on a dark absorber, a blur about
65 nm wide). They are test assumptions, not a calibrated ACTIS model:

- blur: a lens (circular pupil, faint rings) of ``lens_nm`` width, softened by a Gaussian smear chosen so that the
  whole blur is ``blur_nm`` wide at half height. With the default 45 nm lens the image holds no detail finer than
  the pixel grid can show (cut-off 0.48 cycles per pixel), so sampling adds no aliasing;
- light: background (absorber) ``background`` grey levels, open area ``open_level``; each hole's light is scaled by
  its own brightness factor (source flicker, sd ``brightness_sigma``), the background is not;
- per hole: CD noise (both axes change by the same nm, sd ``cd_sigma_nm``) and placement noise (x and y, sd
  ``placement_sigma_nm``);
- planted: every ``row_shift_every``-th logical row (default: the rows closest to ``row_shift_period_nm`` apart)
  moved ``row_shift_nm`` along +y (down the page);
- noise: photon noise (``photons_per_level`` photons per grey level) plus read noise, rounded to ``bits`` bits.

Coordinates are [x, y] pixels, pixel centres at integers. Logical (row, column) indices follow
``subpx.pattern.analyze_pitch``: rows run along x; in a staggered array odd rows are shifted by half the x pitch and
hole ``k`` of a row has column ``k``, so the y-pitch pairs are (r, k)-(r + 1, k).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math

import numpy as np

__all__ = ["HoleArrayImage", "render_hole_array", "blur_smear_sigma", "blur_profile", "LENS_FWHM_FACTOR"]

# Half-height width of a circular-pupil lens blur (intensity) is 0.5145 lambda / NA; its cut-off frequency is
# 2 NA / lambda, so cut-off = LENS_FWHM_FACTOR / width.
LENS_FWHM_FACTOR = 1.0290
SHAPES = ("circle", "square", "ellipse")
LAYOUTS = ("square", "staggered")


@dataclass
class HoleArrayImage:
    """A rendered page and its truth. Per-hole arrays are aligned; ``inside`` marks holes whose center is on the
    page (holes just outside are rendered too, so the page edge cuts an infinite array)."""

    image: np.ndarray
    clean: np.ndarray
    centers_xy: np.ndarray
    design_xy: np.ndarray
    indices_rc: np.ndarray
    size_px: np.ndarray
    brightness: np.ndarray
    placement_nm: np.ndarray
    row_shift_nm: np.ndarray
    inside: np.ndarray
    meta: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------------- blur
def _lens_otf(nu, cutoff, xp=np):
    """Incoherent transfer function of a circular pupil, radial frequency ``nu`` (cycles/px)."""
    s = xp.clip(nu / cutoff, 0.0, 1.0)
    return (2 / np.pi) * (xp.arccos(s) - s * xp.sqrt(1.0 - s * s))


def _otf(fx, fy, cutoff, sigma, xp=np):
    nu2 = fx * fx + fy * fy
    out = xp.exp(-2.0 * np.pi ** 2 * sigma ** 2 * nu2)
    if cutoff is not None:
        out = out * _lens_otf(xp.sqrt(nu2), cutoff, xp)
    return out


def blur_profile(radius_px, lens_px, smear_sigma_px, *, samples=4096):
    """Radial profile of the blur (peak 1) at ``radius_px``: lens of half-height width ``lens_px`` (0 = none)
    times a Gaussian smear of sd ``smear_sigma_px``, from the radial (Hankel) transform of the transfer function."""
    from scipy.special import j0

    r = np.atleast_1d(np.asarray(radius_px, dtype=np.float64))
    if lens_px > 0:
        cutoff = LENS_FWHM_FACTOR / lens_px
        nu = np.linspace(0.0, cutoff, samples)
        weight = _lens_otf(nu, cutoff) * np.exp(-2.0 * np.pi ** 2 * smear_sigma_px ** 2 * nu ** 2) * nu
    else:
        if smear_sigma_px <= 0:
            raise ValueError("a blur needs a lens or a smear")
        top = 6.0 / (2 * np.pi * smear_sigma_px)
        nu = np.linspace(0.0, top, samples)
        weight = np.exp(-2.0 * np.pi ** 2 * smear_sigma_px ** 2 * nu ** 2) * nu
    values = np.trapezoid(weight[None, :] * j0(2 * np.pi * r[:, None] * nu[None, :]), nu, axis=1)
    return values / np.trapezoid(weight, nu)


def _profile_width(lens_px, sigma):
    """Half-height width of the blur (px): the radius where the profile falls to 0.5, by root finding."""
    from scipy.optimize import brentq

    top = 2.0 * max(lens_px, 2.3548 * sigma, 0.5)
    return 2.0 * brentq(lambda r: blur_profile(r, lens_px, sigma, samples=2048)[0] - 0.5, 0.0, top, xtol=1e-10)


@lru_cache(maxsize=64)
def blur_smear_sigma(blur_px, lens_px):
    """Gaussian smear sd (px) that makes a lens of width ``lens_px`` blur ``blur_px`` wide at half height."""
    if not (np.isfinite(blur_px) and blur_px > 0 and np.isfinite(lens_px) and lens_px >= 0):
        raise ValueError("blur and lens widths must be finite, blur > 0 and lens >= 0")
    if lens_px == 0:
        return blur_px / (2 * math.sqrt(2 * math.log(2)))
    if lens_px >= blur_px:
        raise ValueError("the lens alone must be narrower than the whole blur")
    lo, hi = 0.0, blur_px / 2.3548
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _profile_width(lens_px, mid) < blur_px else (lo, mid)
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------------------------------- shapes
def _shape_ft(kind, fu, fv, ax, ay, xp, j1):
    """Fourier transform of the hole outline (unit inside), ``ax`` / ``ay`` full sizes (px) along its own axes,
    shape (B, 1, 1) against frequencies (n, n)."""
    if kind == "square":
        return ax * ay * xp.sinc(ax * fu) * xp.sinc(ay * fv)
    rho = xp.sqrt((ax * fu) ** 2 + (ay * fv) ** 2)
    t = np.pi * rho
    safe = xp.where(t > 0, t, 1.0)
    jinc = xp.where(t > 0, 2.0 * j1(safe) / safe, 1.0)
    return (np.pi / 4.0) * ax * ay * jinc


def _modules(backend):
    if backend == "gpu":
        import cupy as cp
        from cupyx.scipy.special import j1 as cj1

        return cp, cj1
    from scipy.special import j1

    return np, j1


def _resolve(backend):
    if backend not in ("auto", "cpu", "gpu"):
        raise ValueError("backend must be 'auto', 'cpu' or 'gpu'")
    if backend == "auto":
        try:
            import cupy as cp

            backend = "gpu" if cp.cuda.runtime.getDeviceCount() > 0 else "cpu"
        except Exception:
            backend = "cpu"
    return backend


def _render(shape, centers, sizes, amplitudes, kind, angle, patch, oversample, cutoff, sigma, backend):
    """Sum of every hole's pixel-integrated, blurred light on an (H, W) page (float64, host)."""
    xp, j1 = _modules(backend)
    h, w = shape
    n = oversample * patch
    f = xp.fft.fftfreq(n, d=1.0 / oversample)  # cycles/px, spacing 1/patch
    fy, fx = f[:, None], f[None, :]
    c, s = math.cos(angle), math.sin(angle)
    fu, fv = c * fx + s * fy, -s * fx + c * fy  # the hole's own axes
    common = _otf(fx, fy, cutoff, sigma, xp) * xp.sinc(fx) * xp.sinc(fy) * oversample ** 2
    page = xp.zeros(h * w, dtype=xp.float64)
    offsets = xp.arange(patch)
    batch = max(1, int((2e7 if backend == "gpu" else 3e6) // (n * n)))
    for start in range(0, len(centers), batch):
        part = slice(start, start + batch)
        cxy = xp.asarray(centers[part])
        origin = xp.rint(cxy).astype(xp.int64) - patch // 2
        local = cxy - origin  # center inside the patch, about patch / 2
        ax = xp.asarray(sizes[part, 0])[:, None, None]
        ay = xp.asarray(sizes[part, 1])[:, None, None]
        spectrum = _shape_ft(kind, fu, fv, ax, ay, xp, j1) * common
        spectrum = spectrum * xp.asarray(amplitudes[part])[:, None, None]
        phase = xp.exp(-2j * np.pi * (fx * local[:, 0, None, None] + fy * local[:, 1, None, None]))
        light = xp.fft.ifft2(spectrum * phase).real[:, ::oversample, ::oversample]
        rows = origin[:, 1, None, None] + offsets[None, :, None]
        cols = origin[:, 0, None, None] + offsets[None, None, :]
        keep = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        flat = (rows * w + cols)[keep]
        if backend == "gpu":
            xp.add.at(page, flat, light[keep])
        else:
            page += np.bincount(flat, weights=light[keep], minlength=h * w)
    page = page.reshape(h, w)
    return page.get() if backend == "gpu" else page


# ------------------------------------------------------------------------------------------------------ main
def render_hole_array(
    shape=(1378, 2048),
    *,
    hole: str = "circle",
    size_nm=(100.0, 100.0),
    pitch_nm=(200.0, 200.0),
    layout: str = "square",
    pixel_nm: float = 21.0,
    blur_nm: float = 65.0,
    lens_nm: float = 45.0,
    background: float = 15.0,
    open_level: float = 103.0,
    photons_per_level: float | None = 30.0,
    read_noise: float = 0.65,
    bits: int | None = 8,
    cd_sigma_nm: float = 1.0,
    placement_sigma_nm: float = 0.5,
    brightness_sigma: float = 0.05,
    row_shift_nm: float = 1.0,
    row_shift_period_nm: float = 1000.0,
    row_shift_every: int | None = None,
    row_shift_first: int | None = None,
    rotation_deg: float = 0.0,
    seed: int = 0,
    backend: str = "auto",
) -> HoleArrayImage:
    """Render a square or staggered array of bright holes with known centers and a planted row shift.

    Parameters
    ----------
    shape : (H, W)
        Page size in pixels.
    hole : {'circle', 'square', 'ellipse'}
        Outline; ``size_nm`` is (x, y): circle diameter (both equal), square sides or ellipse axes (full widths).
    pitch_nm : (x, y)
        x pitch along a row and y pitch between adjacent rows. ``layout='staggered'`` shifts odd rows by half the
        x pitch.
    pixel_nm, blur_nm, lens_nm
        Pixel size; the blur's half-height width; the lens's own width (0: a Gaussian blur only).
    background, open_level
        Grey level of the absorber and of a fully open area (a hole large enough to reach it).
    photons_per_level, read_noise, bits
        Photon noise (None: none), Gaussian read noise (grey levels), rounding to ``bits`` bits (None: a float
        image without rounding or clipping).
    cd_sigma_nm, placement_sigma_nm, brightness_sigma
        Per-hole random CD change (added to both axes), placement error (x and y) and relative brightness change.
    row_shift_nm, row_shift_period_nm, row_shift_every, row_shift_first
        Planted shift along +y of every ``row_shift_every``-th logical row (default: the whole number of rows
        closest to ``row_shift_period_nm``, at least 2), starting at row ``row_shift_first`` (default: half that
        number, so the first shifted row is not the top one).
    rotation_deg
        Lattice and hole rotation about the page center (0: rows parallel to the pixel rows).
    seed
        Seeds the lattice's sub-pixel origin, every per-hole draw and the noise.
    backend : {'auto', 'cpu', 'gpu'}
        Where the holes are rendered (CuPy on the GPU); noise is drawn on the host, so pages agree across backends.
    """
    h, w = (int(v) for v in shape)
    if h < 16 or w < 16:
        raise ValueError("shape must be at least 16 x 16 pixels")
    if hole not in SHAPES:
        raise ValueError(f"hole must be one of {', '.join(SHAPES)}")
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {', '.join(LAYOUTS)}")
    size = np.asarray(size_nm, dtype=np.float64).reshape(-1)
    pitch = np.asarray(pitch_nm, dtype=np.float64).reshape(-1)
    if size.shape != (2,) or pitch.shape != (2,) or not (np.isfinite(size).all() and np.isfinite(pitch).all()):
        raise ValueError("size_nm and pitch_nm need two finite values (x, y)")
    if np.any(size <= 0) or np.any(pitch <= 0) or not pixel_nm > 0:
        raise ValueError("sizes, pitches and the pixel size must be positive")
    if hole == "circle" and size[0] != size[1]:
        raise ValueError("a circle needs equal x and y sizes")
    for name, value in (("cd_sigma_nm", cd_sigma_nm), ("placement_sigma_nm", placement_sigma_nm),
                        ("brightness_sigma", brightness_sigma), ("read_noise", read_noise)):
        if not (np.isfinite(value) and value >= 0):
            raise ValueError(f"{name} must be finite and >= 0")
    if photons_per_level is not None and not (np.isfinite(photons_per_level) and photons_per_level > 0):
        raise ValueError("photons_per_level must be positive or None")
    if not np.isfinite(row_shift_nm) or not (np.isfinite(rotation_deg) and abs(rotation_deg) < 45):
        raise ValueError("row_shift_nm must be finite and |rotation_deg| < 45")
    backend = _resolve(backend)
    rng = np.random.default_rng(seed)

    px, py = pitch / pixel_nm
    size_design = size / pixel_nm
    blur_px, lens_px = blur_nm / pixel_nm, lens_nm / pixel_nm
    sigma = blur_smear_sigma(blur_px, lens_px)
    cutoff = LENS_FWHM_FACTOR / lens_px if lens_px > 0 else None
    # Q = 1 is exact when the lens passes nothing above half a cycle per pixel; else sample twice as finely.
    oversample = 1 if cutoff is not None and cutoff < 0.5 else 2
    every = int(row_shift_every) if row_shift_every is not None else max(2, int(round(row_shift_period_nm / pitch[1])))
    if every < 1:
        raise ValueError("row_shift_every must be >= 1")
    first = every // 2 if row_shift_first is None else int(row_shift_first)

    # Design lattice (logical rows r along x, columns k), rotated about the page center; generous index range.
    angle = math.radians(rotation_deg)
    reach = 0.5 * float(np.hypot(*size_design)) + 4.0 * blur_px  # light of a hole beyond this is negligible
    origin = np.array([(w - 1) / 2, (h - 1) / 2]) + rng.uniform(-0.5, 0.5, 2)
    span = 0.5 * math.hypot(w, h) + reach + max(px, py)
    grid_r, grid_k = np.mgrid[-int(span / py) - 2:int(span / py) + 3, -int(span / px) - 2:int(span / px) + 3]
    grid_r, grid_k = grid_r.ravel(), grid_k.ravel()
    c, s = math.cos(angle), math.sin(angle)

    def lattice(parity):
        lx = grid_k * px + (np.mod(grid_r - parity, 2) * 0.5 * px if layout == "staggered" else 0.0)
        ly = grid_r * py
        points = np.column_stack((origin[0] + c * lx - s * ly, origin[1] + s * lx + c * ly))
        near = ((points[:, 0] > -reach) & (points[:, 0] < w - 1 + reach)
                & (points[:, 1] > -reach) & (points[:, 1] < h - 1 + reach))
        points, r, k = points[near], grid_r[near], grid_k[near]
        order = np.lexsort((k, r))
        points, r, k = points[order], r[order], k[order]
        on_page = (points[:, 0] >= 0) & (points[:, 0] <= w - 1) & (points[:, 1] >= 0) & (points[:, 1] <= h - 1)
        return points, r, k, on_page, int(r[on_page].min())

    # The first row on the page is unshifted and the next one sits half a pitch to its right, as analyze_pitch
    # reads a staggered array; flip which rows are offset if the first row came out shifted.
    parity = 0
    design, rr, kk, on_page, r0 = lattice(parity)
    for _ in range(3):
        if layout != "staggered" or (r0 - parity) % 2 == 0:
            break
        parity = 1 - parity
        design, rr, kk, on_page, r0 = lattice(parity)
    else:
        raise ValueError("could not place an unshifted first row; try another seed or rotation")
    # Logical indices start at 0 on the page; holes just outside may get negative indices.
    rows, cols = rr - r0, kk - int(kk[on_page].min())
    n = len(design)

    # Per-hole draws, in a fixed order.
    cd_change = rng.normal(0.0, cd_sigma_nm, n) if cd_sigma_nm > 0 else np.zeros(n)
    brightness = 1.0 + (rng.normal(0.0, brightness_sigma, n) if brightness_sigma > 0 else np.zeros(n))
    placement = rng.normal(0.0, placement_sigma_nm, (n, 2)) if placement_sigma_nm > 0 else np.zeros((n, 2))
    shifted = np.mod(rows - first, every) == 0
    shift = np.where(shifted, float(row_shift_nm), 0.0)
    normal = np.array([-s, c])  # +y of the lattice
    centers = design + shift[:, None] * normal[None, :] / pixel_nm + placement / pixel_nm
    sizes = np.maximum(size_design[None, :] + cd_change[:, None] / pixel_nm, 0.05)
    amplitudes = (open_level - background) * brightness

    # Each hole is rendered on a periodic patch, so light beyond its edge folds back. A Gaussian blur leaves
    # nothing there; the lens's faint far tail does: a 64 px patch keeps the folded part's pull on a center below
    # 1e-4 px (0.002 nm) within 15 px of the hole (tests/test_synthetic_arrays.py).
    patch = 2 * int(math.ceil(reach + 1))
    if cutoff is not None:
        patch = max(patch, 64)
    clean = background + _render((h, w), centers, sizes, amplitudes, hole, angle, patch, oversample, cutoff, sigma,
                                 backend)
    if photons_per_level is not None:
        noisy = rng.poisson(np.clip(clean, 0, None) * photons_per_level) / photons_per_level
    else:
        noisy = clean.copy()
    if read_noise > 0:
        noisy = noisy + rng.normal(0.0, read_noise, clean.shape)
    if bits is None:
        image = noisy.astype(np.float32)
    else:
        top = 2 ** int(bits) - 1
        image = np.clip(np.rint(noisy), 0, top).astype(np.uint8 if bits <= 8 else np.uint16)

    inside = ((centers[:, 0] >= 0) & (centers[:, 0] <= w - 1) & (centers[:, 1] >= 0) & (centers[:, 1] <= h - 1))
    shifted_rows = sorted({int(r) for r in rows[shifted & inside]})
    meta = dict(
        description="Synthetic hole array with exactly known centers (subpx.synthetic_arrays); test data, not a "
                    "calibrated ACTIS model",
        shape_hw=[h, w], hole=hole, size_nm=size.tolist(), pitch_nm=pitch.tolist(), layout=layout,
        pixel_nm=float(pixel_nm), pitch_px=[float(px), float(py)], size_px=size_design.tolist(),
        blur_nm=float(blur_nm), lens_nm=float(lens_nm), smear_fwhm_nm=float(sigma * 2.3548 * pixel_nm),
        lens_cutoff_cycles_per_px=None if cutoff is None else float(cutoff), oversample=oversample, patch_px=patch,
        background=float(background), open_level=float(open_level), photons_per_level=photons_per_level,
        read_noise=float(read_noise), bits=bits, cd_sigma_nm=float(cd_sigma_nm),
        placement_sigma_nm=float(placement_sigma_nm), brightness_sigma=float(brightness_sigma),
        row_shift_nm=float(row_shift_nm), row_shift_every=every, row_shift_first=first, shifted_rows=shifted_rows,
        rotation_deg=float(rotation_deg), origin_xy=origin.tolist(), seed=int(seed), backend=backend,
        holes=int(n), holes_inside=int(inside.sum()), coordinates="[x, y] px, pixel centres at integers",
    )
    return HoleArrayImage(image, clean, centers, design, np.column_stack((rows, cols)).astype(np.int64),
                          sizes, brightness, placement, shift, inside, meta)
