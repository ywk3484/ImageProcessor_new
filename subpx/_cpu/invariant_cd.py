"""CD methods insensitive to a per-hole illumination (source-pulse) scale.

A hole's spot is ``a_i * S(x; d_i) + B_i``: the pulse ``a_i`` scales the spot, so
any CD read from brightness inherits it. These methods use only what the scale
cannot change, or cancel it:

``disk``
    Fits each spot with a round hole of diameter ``d`` imaged through a
    circular-pupil (Airy) blur and averaged over the pixel, with free
    amplitude, center and local background: the amplitude absorbs the pulse,
    ``d`` is set by the spot's shape alone.
``flux_ratio``
    Hole light above the local background divided by that background (minus a
    dark level). Cancels the pulse when the background is absorber light lit
    by the same pulse.
``projection``
    Summed x and y profiles of the cell; width between the steepest rise and
    fall of each; CD = mean of the two widths.
``half_area``
    Area of the region around the peak brighter than the cell's local level
    plus half the peak height above it, on a 4x cubic-spline enlargement of
    the cell; CD = equal-area circle diameter. Assumes no shape.
``half_area_min``
    ``half_area`` with the local level taken as the minimum of the cell's
    outermost pixels instead of their median.
``half_area_min_avg``
    ``half_area_min`` with the peak taken as the mean of the brightest
    pixels (a fixed area) instead of the single brightest point, so one
    outstanding pixel cannot set the threshold.

For the ``half_area`` family cd_x / cd_y are the x / y widths of the ellipse
with the region's area and its x and y second moments (the ellipse's diameter
along x and y), so cd_x * cd_y = cd_equivalent ** 2.

``projection`` and the ``half_area`` family run on NumPy or CuPy (``xp``). All work on cells with the neighbouring holes' models removed (from
``deblend``). Lengths are in pixels.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..types import CDMethodResult
from .cd import radial_contours
from .deblend import _render_on_grids, _render_radius, render_models, roi_grids, sigma_major, windows, FWHM_K


INVARIANT_METHODS = ("disk", "flux_ratio", "projection", "half_area", "half_area_min", "half_area_min_avg")
_HALF_AREA = {"half_area": ("median", False), "half_area_min": ("min", False), "half_area_min_avg": ("min", True)}
AIRY_K = 1.0290  # FWHM of the incoherent Airy pattern = AIRY_K / cutoff frequency
_CHUNK = 4096    # holes per batched fit step: (chunk, K) arrays stay ~10 MB each
_CALIBRATION_HOLES = 2000


def _otf(f: np.ndarray, fc: float) -> np.ndarray:
    nu = np.minimum(f / fc, 1.0)
    return (2 / np.pi) * (np.arccos(nu) - nu * np.sqrt(1 - nu * nu))


def _pixel_response(f: np.ndarray, n_angles: int = 64) -> np.ndarray:
    """Pixel-box transfer function averaged over direction (square pixel ~ isotropic below 0.5 cycles/px)."""
    theta = (np.arange(n_angles) + 0.5) * (np.pi / 2 / n_angles)
    return (np.sinc(np.outer(f, np.cos(theta))) * np.sinc(np.outer(f, np.sin(theta)))).mean(axis=1)


class DiskTable:
    """Spot of a unit-brightness disk vs (diameter, distance from its center), pixel-integrated.

    ``T(d, 0) -> 1`` for disks much wider than the blur, so a fitted amplitude
    is the hole's unblurred brightness above background. Hankel transform of
    disk spectrum x Airy OTF x pixel response; bilinear lookups with
    finite-difference derivatives.
    """

    def __init__(self, psf_fwhm: float, d_max: float, rho_max: float, dd: float = 0.02, drho: float = 0.02,
                 nf: int = 1024):
        from scipy.special import j0, j1

        self.psf_fwhm = float(psf_fwhm)
        fc = AIRY_K / self.psf_fwhm
        f = np.linspace(0.0, fc, nf)
        weight = np.full(nf, f[1] - f[0])
        weight[[0, -1]] /= 2
        self.d = np.arange(0.0, d_max + dd, dd)
        self.rho = np.arange(0.0, rho_max + 2 * drho, drho)
        self.dd, self.drho = dd, drho
        radius = self.d / 2
        spectrum = 2 * np.pi * radius[None] * j1(2 * np.pi * np.outer(f, radius))
        spectrum *= (_otf(f, fc) * _pixel_response(f) * weight)[:, None]
        self.T = np.ascontiguousarray((j0(2 * np.pi * np.outer(self.rho, f)) @ spectrum).T)
        self.Tr = np.gradient(self.T, drho, axis=1)
        self.Td = np.gradient(self.T, dd, axis=0)
        for table in (self.T, self.Tr, self.Td):
            table[:, -1] = 0.0  # lookups beyond rho_max clamp onto this column
        self.d_min, self.d_max = 2 * dd, self.d[-3]

    def lookup(self, d: np.ndarray, rho: np.ndarray, derivatives: bool = True):
        nd, nr = self.T.shape
        fd = d / self.dd
        i = np.clip(np.floor(fd).astype(np.int64), 0, nd - 2)
        u = np.clip(fd - i, 0.0, 1.0)
        fr = rho / self.drho
        j = np.clip(np.floor(fr).astype(np.int64), 0, nr - 2)
        v = np.clip(fr - j, 0.0, 1.0)
        k = i * nr + j

        def bilinear(table):
            t = table.ravel()
            return (1 - u) * ((1 - v) * t[k] + v * t[k + 1]) + u * ((1 - v) * t[k + nr] + v * t[k + nr + 1])

        if not derivatives:
            return bilinear(self.T)
        return bilinear(self.T), bilinear(self.Tr), bilinear(self.Td)


def _pixel_lists(masks: np.ndarray):
    """Each cell's pixels as padded (N, K) lists: flat ROI index, local x, local y, valid."""
    n, h, w = masks.shape
    flat = masks.reshape(n, -1)
    k = max(int(flat.sum(axis=1).max()), 1) if n else 1
    order = np.argsort(~flat, axis=1, kind="stable")[:, :k]
    valid = np.take_along_axis(flat, order, axis=1)
    return order, (order % w).astype(float), (order // w).astype(float), valid


def _fit_disks(table: DiskTable, gx, gy, data, valid, p0, *, max_iter: int = 50, tol: float = 1e-4):
    """Batched Levenberg-Marquardt of ``A T(d, |x - c|) + B`` on padded pixel lists.

    Returns params [A, x0, y0, d, B], status, the d uncertainty (from the
    curvature and residual) and the residual RMS.
    """
    n = len(p0)
    p = p0.astype(float).copy()
    lam = np.full(n, 1e-3)
    done = np.zeros(n, bool)
    converged = np.zeros(n, bool)
    lo =np.array([1e-9, -np.inf, -np.inf, table.d_min, -np.inf])
    hi = np.array([np.inf, np.inf, np.inf, table.d_max, np.inf])
    w = valid.astype(float)

    def evaluate(q, rows, jac):
        dx = gx[rows] - q[:, 1, None]
        dy = gy[rows] - q[:, 2, None]
        rho = np.hypot(dx, dy)
        d = np.broadcast_to(q[:, 3, None], rho.shape)
        if not jac:
            return q[:, 0, None] * table.lookup(d, rho, False) + q[:, 4, None]
        T, Tr, Td = table.lookup(d, rho)
        a = q[:, 0, None]
        inv = 1 / np.maximum(rho, 1e-9)
        J = np.stack((T, -a * Tr * dx * inv, -a * Tr * dy * inv, a * Td, np.ones_like(T)), axis=-1)
        return a * T + q[:, 4, None], J

    for _ in range(max_iter):
        active = np.flatnonzero(~done)
        if not len(active):
            break
        for start in range(0, len(active), _CHUNK):
            rows = active[start:start + _CHUNK]
            q, wr = p[rows], w[rows]
            model, J = evaluate(q, rows, True)
            r = (data[rows] - model) * wr
            cost = (r * r).sum(axis=1)
            Jw = J * wr[..., None]
            H = np.einsum("nki,nkj->nij", Jw, Jw)
            g = np.einsum("nki,nk->ni", Jw, r)
            diag = np.einsum("nii->ni", H)
            Hd = H + (lam[rows, None] * np.maximum(diag, 1e-12))[:, :, None] * np.eye(5)
            try:
                step = np.linalg.solve(Hd, g[..., None])[..., 0]
            except np.linalg.LinAlgError:
                step = np.stack([np.linalg.lstsq(m, b, rcond=None)[0] for m, b in zip(Hd, g)])
            trial = np.clip(q + step, lo, hi)
            r_new = (data[rows] - evaluate(trial, rows, False)) * wr
            cost_new = (r_new * r_new).sum(axis=1)
            accept = np.isfinite(cost_new) & (cost_new <= cost)
            moved = np.abs(trial - q)
            small = ((moved[:, 1:4] < tol).all(axis=1) & (moved[:, 0] < tol * np.abs(q[:, 0]) + 1e-12)
                     & (moved[:, 4] < tol * (np.abs(q[:, 4]) + np.abs(q[:, 0])) + 1e-12))
            p[rows[accept]] = trial[accept]
            # A tiny rejected step at low damping is the minimum itself; at high damping it is not.
            converged[rows] = small & (accept | (lam[rows] < 1e2))
            lam[rows] = np.where(accept, np.maximum(lam[rows] / 3, 1e-9), np.minimum(lam[rows] * 4, 1e9))
            done[rows] = converged[rows] | (lam[rows] >= 1e9)
    status = np.where(converged, "ok", "not_converged").astype("U32")
    sigma_d = np.full(n, np.nan)
    rmse = np.full(n, np.nan)
    for start in range(0, n, _CHUNK):
        rows = np.arange(start, min(start + _CHUNK, n))
        model, J = evaluate(p[rows], rows, True)
        wr = w[rows]
        r = (data[rows] - model) * wr
        dof = np.maximum(wr.sum(axis=1) - 5, 1)
        s2 = (r * r).sum(axis=1) / dof
        rmse[rows] = np.sqrt((r * r).sum(axis=1) / np.maximum(wr.sum(axis=1), 1))
        Jw = J * wr[..., None]
        H = np.einsum("nki,nkj->nij", Jw, Jw)
        ok = np.linalg.cond(H) < 1e12
        cov = np.full_like(H, np.nan)
        cov[ok] = np.linalg.inv(H[ok])
        sigma_d[rows] = np.sqrt(np.abs(cov[:, 3, 3]) * s2)
    finite = np.isfinite(p).all(axis=1)
    edge = (p[:, 3] <= table.d_min + table.dd) | (p[:, 3] >= table.d_max - table.dd)
    status[finite & edge] = "at_diameter_limit"
    status[finite & (p[:, 0] <= 1e-6)] = "no_signal"
    status[~finite] = "fit_failed"
    return p, status, sigma_d, rmse


def render_disks(table: DiskTable, params_global: np.ndarray, shape: tuple[int, int], radius: int) -> np.ndarray:
    """Sum of ``A T(d, rho)`` of every hole (global centers) within ``radius`` pixels, as an (H, W) image."""
    height, width = shape
    out = np.zeros(height * width)
    offsets = np.arange(-radius, radius + 1)
    chunk = max(1, 2_000_000 // len(offsets) ** 2)
    for start in range(0, len(params_global), chunk):
        p = params_global[start:start + chunk]
        ix = np.floor(p[:, 1]).astype(np.int64)[:, None, None] + offsets[None, None, :]
        iy = np.floor(p[:, 2]).astype(np.int64)[:, None, None] + offsets[None, :, None]
        ix, iy = (a.reshape(len(p), -1) for a in np.broadcast_arrays(ix, iy))
        rho = np.hypot(ix - p[:, 1, None], iy - p[:, 2, None])
        values = p[:, 0, None] * table.lookup(np.broadcast_to(p[:, 3, None], rho.shape), rho, False)
        ok = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height) & (rho <= radius)
        out += np.bincount((iy * width + ix)[ok], weights=values[ok], minlength=height * width)
    return out.reshape(shape)


def far_field(table: DiskTable, params_global: np.ndarray, shape: tuple[int, int], radius: int) -> np.ndarray:
    """Light of every hole beyond ``radius`` pixels (the slow Airy tail), as an (H, W) image.

    The tails of thousands of holes add up to a nearly uniform glow (~4 % of
    each hole's light lies beyond 16 px) that a per-hole fit would read as
    background, although the hole's own pulse does not light it. Tails barely
    depend on d, so each hole is a point of its light A pi d^2 / 4 convolved
    (FFT) with the median hole's spot minus its part within ``radius``.
    """
    height, width = shape
    if not len(params_global):
        return np.zeros(shape)
    pad = 4 * radius
    H, W = height + 2 * pad, width + 2 * pad
    d = float(np.median(params_global[:, 3]))
    flux = params_global[:, 0] * np.pi * params_global[:, 3] ** 2 / 4
    x, y = params_global[:, 1] + pad, params_global[:, 2] + pad
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    u, v = x - ix, y - iy
    points = np.zeros(H * W)
    for dx, dy, weight in ((0, 0, (1 - u) * (1 - v)), (1, 0, u * (1 - v)), (0, 1, (1 - u) * v), (1, 1, u * v)):
        jx, jy = ix + dx, iy + dy
        ok = (jx >= 0) & (jx < W) & (jy >= 0) & (jy < H)
        points += np.bincount(jy[ok] * W + jx[ok], weights=(flux * weight)[ok], minlength=H * W)
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.rfftfreq(W)[None, :]
    f = np.hypot(fx, fy)
    from scipy.special import j1
    fc = AIRY_K / table.psf_fwhm
    radius_d = d / 2
    disk = np.where(f > 0, radius_d * j1(2 * np.pi * f * radius_d) / np.where(f > 0, f, 1), np.pi * radius_d ** 2)
    full = np.where(f < fc, _otf(f, fc), 0.0) * np.sinc(fx) * np.sinc(fy) * disk / (np.pi * radius_d ** 2)
    offsets = np.arange(-radius, radius + 1)
    rho = np.hypot(offsets[None, :], offsets[:, None])
    near = np.where(rho <= radius, table.lookup(np.full(rho.shape, d), rho, False), 0.0) / (np.pi * radius_d ** 2)
    kernel = np.zeros((H, W))
    kernel[np.ix_(offsets % H, offsets % W)] = near
    tail = np.fft.irfft2(np.fft.rfft2(points.reshape(H, W)) * (full - np.fft.rfft2(kernel)), s=(H, W))
    return tail[pad:pad + height, pad:pad + width]


def _empty(method: str, n: int, n_angles: int, names: tuple[str, ...]) -> CDMethodResult:
    names = ("area_px2", "cd_equivalent_px", "cd_x_px", "cd_y_px", "perimeter_px", "background", "peak_signal") + names
    return CDMethodResult(
        method=method, centers_xy=np.full((n, 2), np.nan), contours_xy=np.full((n, n_angles + 1, 2), np.nan),
        valid=np.zeros(n, bool), status=np.full(n, "no_signal", dtype="U32"),
        metrics={name: np.full(n, np.nan) for name in names},
        geometry={"center_local": np.full((n, 2), np.nan), "edge_px": np.full((n, n_angles), np.nan)},
    )


def _set_ellipses(result: CDMethodResult, rows, centers_local, cd_x, cd_y, origins, n_angles: int) -> None:
    """Axis-aligned ellipse contours (circles when cd_x == cd_y) through ``radial_contours``."""
    angles = np.arange(n_angles) * (2 * np.pi / n_angles)
    a, b = cd_x[:, None] / 2, cd_y[:, None] / 2
    edge = a * b / np.sqrt((b * np.cos(angles)) ** 2 + (a * np.sin(angles)) ** 2)
    result.geometry["center_local"][rows] = centers_local
    result.geometry["edge_px"][rows] = edge
    result.centers_xy[rows] = centers_local + origins
    result.contours_xy[rows] = radial_contours(centers_local, edge, origins, n_angles)
    a, b = a[:, 0], b[:, 0]
    m = result.metrics
    m["cd_x_px"][rows], m["cd_y_px"][rows] = cd_x, cd_y
    m["area_px2"][rows] = np.pi * a * b
    m["cd_equivalent_px"][rows] = a + b  # mean of the x and y widths (equal for circles)
    h = ((a - b) / np.maximum(a + b, 1e-300)) ** 2
    m["perimeter_px"][rows] = np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))  # Ramanujan


@lru_cache(maxsize=16)
def _spline_operators(size: int, step: float):
    """Grid and the (grid, size) matrices giving scipy's not-a-knot ``CubicSpline`` and its slope on that grid.

    The spline through ``size`` samples is linear in them, so the curve of every profile is one matrix product.
    """
    from scipy.interpolate import CubicSpline

    grid = np.arange(0.0, size - 1 + step / 2, step)
    spline = CubicSpline(np.arange(size), np.eye(size), axis=0)
    return grid, spline(grid), spline(grid, 1)


def _to_numpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _projection(cleaned: np.ndarray, masks: np.ndarray, origins: np.ndarray, n_angles: int,
                step: float = 0.02, xp=np) -> CDMethodResult:
    """Summed x/y profiles of each neighbour-removed cell; width between the steepest rise and fall."""
    n, h, w = masks.shape
    result = _empty("projection", n, n_angles, ())
    widths, centers, statuses = np.full((n, 2), np.nan), np.full((n, 2), np.nan), np.full((n, 2), "", dtype="U32")
    for axis, (size, sum_axis) in enumerate(((w, 1), (h, 2))):
        counts = masks.sum(axis=sum_axis)
        # Sum, not mean: the cell outline is ragged by a pixel, and dividing one column by 10 pixels and the next
        # by 11 steps the profile and moves its steepest point (the cleaned cell holds only this hole's light).
        profiles = np.where(masks, cleaned, 0.0).sum(axis=sum_axis)
        inside = counts > 0
        first = np.argmax(inside, axis=1)
        last = size - 1 - np.argmax(inside[:, ::-1], axis=1)
        grid, value_op, slope_op = _spline_operators(size, step)
        value_op, slope_op, grid_d = xp.asarray(value_op.T), xp.asarray(slope_op.T), xp.asarray(grid)
        index = xp.arange(len(grid))[None]
        for start in range(0, n, _CHUNK):
            rows = np.arange(start, min(start + _CHUNK, n))
            profile = xp.asarray(profiles[rows])
            values, slope = profile @ value_op, profile @ slope_op
            lo, hi = xp.asarray(first[rows])[:, None], xp.asarray(last[rows])[:, None]
            span = (grid_d[None] >= lo) & (grid_d[None] <= hi)
            peak = xp.argmax(xp.where(span, values, -xp.inf), axis=1)
            left = xp.argmax(xp.where(span & (index <= peak[:, None]), slope, -xp.inf), axis=1)
            right = xp.argmin(xp.where(span & (index >= peak[:, None]), slope, xp.inf), axis=1)
            picked = xp.arange(len(rows))
            rising, falling = slope[picked, left] > 0, slope[picked, right] < 0
            left, right, rising, falling = (_to_numpy(a) for a in (left, right, rising, falling))
            x_left, x_right = grid[left], grid[right]
            widths[rows, axis] = x_right - x_left
            centers[rows, axis] = (x_left + x_right) / 2
            bad = (x_left <= first[rows]) | (x_right >= last[rows]) | ~rising | ~falling
            statuses[rows, axis] = np.where(bad, "edge_at_cell_border", "ok")
    result.metrics["peak_signal"][:] = np.where(masks, cleaned, -np.inf).reshape(n, -1).max(axis=1) if n else 0
    signal = result.metrics["peak_signal"] > 0
    ok = signal & (statuses == "ok").all(axis=1)
    result.status[signal] = "edge_at_cell_border"
    result.status[ok] = "ok"
    rows = np.flatnonzero(signal)
    _set_ellipses(result, rows, centers[rows], widths[rows, 0], widths[rows, 1], origins[rows], n_angles)
    for name in ("area_px2", "cd_equivalent_px", "cd_x_px", "cd_y_px", "perimeter_px"):
        result.metrics[name][~ok] = np.nan
    result.valid[:] = ok
    return result


@lru_cache(maxsize=16)
def _zoom_operators(size: int, factor: int):
    """(size * factor, size) matrices of ``scipy.ndimage.zoom`` along one axis: order 3, and order 0 for masks.

    Zoom is separable and linear, so a cell enlarges as ``Zy @ cell @ Zx.T``.
    """
    from scipy.ndimage import zoom

    eye = np.eye(size)
    return (np.stack([zoom(e, factor, order=3) for e in eye], axis=1),
            np.stack([zoom(e, factor, order=0) for e in eye], axis=1))


def _grow(region):
    """A batch of 2D boolean images dilated by one pixel in all 8 directions."""
    out = region.copy()
    out[:, 1:] |= region[:, :-1]
    out[:, :-1] |= region[:, 1:]
    grown = out.copy()
    grown[:, :, 1:] |= out[:, :, :-1]
    grown[:, :, :-1] |= out[:, :, 1:]
    return grown


def _half_area(cleaned: np.ndarray, masks: np.ndarray, origins: np.ndarray, n_angles: int,
               level: float = 0.5, factor: int = 4, background: str = "median", peak_pixels: float | None = None,
               xp=np) -> CDMethodResult:
    """Area above the cell's local level + ``level`` x the peak height, on a ``factor``x cubic enlargement.

    Local level b: ``background`` ('median' or 'min') of the cell's outermost pixels (a pulse that lights the hole lights its surroundings
    too, so b, the peak and the threshold scale together and the area does not). Peak: the highest enlarged
    pixel, or with ``peak_pixels`` the mean of the brightest enlarged pixels covering that many original pixels.
    The region is the connected part around the highest pixel above the threshold; CD = 2 sqrt(area / pi);
    cd_x / cd_y = widths of the equal-area ellipse with the region's x / y second moments.
    """
    n, h, w = masks.shape
    name = {spec: key for key, spec in _HALF_AREA.items()}[background, peak_pixels is not None]
    result = _empty(name, n, n_angles, ("local_level", "threshold"))
    if not n:
        return result
    zy, ny = _zoom_operators(h, factor)
    zx, nx = _zoom_operators(w, factor)
    H, W = zy.shape[0], zx.shape[0]
    # Original px per enlarged px: zoom maps the first and last pixel centres onto each other.
    sy, sx = (h - 1) / (H - 1), (w - 1) / (W - 1)
    zy, zx, ny, nx = (xp.asarray(a) for a in (zy, zx, ny, nx))
    angles = np.arange(n_angles) * (2 * np.pi / n_angles)
    radii = np.arange(0.0, float(np.hypot(h, w)), 0.125)  # original px
    cos_a, sin_a, radii_d = xp.asarray(np.cos(angles)), xp.asarray(np.sin(angles)), xp.asarray(radii)
    gy, gx = xp.arange(H, dtype=float), xp.arange(W, dtype=float)
    m = result.metrics
    status = np.full(n, "no_signal", dtype="U32")
    centers, edges = np.full((n, 2), np.nan), np.full((n, n_angles), np.nan)
    for start in range(0, n, _CHUNK):
        rows = np.arange(start, min(start + _CHUNK, n))
        k = len(rows)
        mask = xp.asarray(masks[rows])
        cell = xp.where(mask, xp.asarray(cleaned[rows], dtype=float), 0.0)
        inner = mask.copy()
        inner[:, 1:] &= mask[:, :-1]
        inner[:, :-1] &= mask[:, 1:]
        inner[:, :, 1:] &= mask[:, :, :-1]
        inner[:, :, :-1] &= mask[:, :, 1:]
        inner[:, [0, -1]] = False
        inner[:, :, [0, -1]] = False
        ring = mask & ~inner
        if background == "median":
            b = xp.nanmedian(xp.where(ring, cell, xp.nan).reshape(k, -1), axis=1)
        else:
            b = xp.where(ring, cell, xp.inf).reshape(k, -1).min(axis=1)
        big = zy[None] @ xp.where(mask, cell - b[:, None, None], 0.0) @ zx.T[None]
        big_mask = (ny[None] @ mask.astype(float) @ nx.T[None]) > 0.5
        flat = xp.where(big_mask, big, -xp.inf).reshape(k, -1)
        top = xp.argmax(flat, axis=1)
        peak = flat[xp.arange(k), top]
        if peak_pixels is not None:
            n_top = min(max(1, int(round(peak_pixels * factor * factor))), H * W)
            peak = xp.partition(flat, -n_top, axis=1)[:, -n_top:].mean(axis=1)  # -inf (outside the cell): no_signal
            peak = xp.where(xp.isfinite(peak), peak, -1.0)
        above = big_mask & (big > (level * peak)[:, None, None])
        region = xp.zeros((k, H * W), dtype=bool)
        region[xp.arange(k), top] = True
        region = region.reshape(k, H, W) & above
        for step in range(H + W):  # grow from the peak inside the above-threshold pixels until nothing changes
            grown = _grow(region) & above
            if step % 8 == 7 and bool((grown == region).all()):
                break
            region = grown
        count = region.sum(axis=(1, 2))
        touches = ((_grow(region) & ~big_mask).any(axis=(1, 2)) | region[:, [0, -1]].any(axis=(1, 2))
                   | region[:, :, [0, -1]].any(axis=(1, 2)))
        weight = xp.maximum(count, 1)
        cy = (region.sum(axis=2) * gy).sum(axis=1) / weight
        cx = (region.sum(axis=1) * gx).sum(axis=1) / weight
        var_x = (region.sum(axis=1) * (gx[None] - cx[:, None]) ** 2).sum(axis=1) / weight * sx * sx
        var_y = (region.sum(axis=2) * (gy[None] - cy[:, None]) ** 2).sum(axis=1) / weight * sy * sy
        # Equal-area ellipse with the region's x/y spread: its diameters along x and y (exact for an axis-aligned
        # ellipse; cd_x * cd_y = cd_equivalent^2).
        cd_eq = 2 * xp.sqrt(count * (sx * sy) / np.pi)
        ratio = xp.sqrt(xp.sqrt(xp.maximum(var_x, 1e-12) / xp.maximum(var_y, 1e-12)))
        extent_x, extent_y = cd_eq * ratio, cd_eq / ratio
        # Contour for display: the region's last pixel along each ray from its centroid. Rays stop just beyond
        # the farthest region pixel of the chunk (a few px), not at the cell diagonal.
        far = ((gx[None, None] - cx[:, None, None]) * sx) ** 2 + ((gy[None, :, None] - cy[:, None, None]) * sy) ** 2
        reach_max = float(xp.sqrt(xp.where(region, far, 0.0).max())) if k else 0.0
        steps = radii_d[:min(len(radii), int(reach_max / 0.125) + 3)]
        px = xp.rint(cx[:, None, None] + cos_a[None, :, None] * steps[None, None] / sx).astype(xp.int64)
        py = xp.rint(cy[:, None, None] + sin_a[None, :, None] * steps[None, None] / sy).astype(xp.int64)
        inside = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        index = (xp.clip(py, 0, H - 1) * W + xp.clip(px, 0, W - 1)).reshape(k, -1)
        hit = xp.take_along_axis(region.reshape(k, -1), index, axis=1).reshape(px.shape) & inside
        first_out = xp.where(hit.all(axis=2), len(steps), xp.argmin(hit, axis=2))
        reach = steps[xp.maximum(first_out - 1, 0)] + 0.0625
        out = {name: _to_numpy(v) for name, v in (
            ("b", b), ("peak", peak), ("area", count * (sx * sy)), ("ex", extent_x), ("ey", extent_y),
            ("touches", touches), ("cx", cx * sx), ("cy", cy * sy), ("edge", reach))}
        signal = out["peak"] > 0
        m["local_level"][rows] = out["b"]
        m["peak_signal"][rows] = np.where(signal, out["peak"], np.nan)
        m["threshold"][rows] = np.where(signal, out["b"] + level * out["peak"], np.nan)
        m["area_px2"][rows] = np.where(signal, out["area"], np.nan)
        m["cd_x_px"][rows] = np.where(signal, out["ex"], np.nan)
        m["cd_y_px"][rows] = np.where(signal, out["ey"], np.nan)
        status[rows] = np.where(~signal, "no_signal", np.where(out["touches"], "region_at_cell_border", "ok"))
        centers[rows] = np.column_stack((out["cx"], out["cy"]))
        edges[rows] = out["edge"]
    m["cd_equivalent_px"][:] = 2 * np.sqrt(m["area_px2"] / np.pi)
    rows = np.flatnonzero(status != "no_signal")
    result.geometry["center_local"][rows] = centers[rows]
    result.geometry["edge_px"][rows] = edges[rows]
    result.centers_xy[rows] = centers[rows] + origins[rows]
    result.contours_xy[rows] = radial_contours(centers[rows], edges[rows], origins[rows], n_angles)
    m["perimeter_px"][rows] = np.linalg.norm(np.diff(result.contours_xy[rows], axis=1), axis=2).sum(axis=1)
    result.status[:] = status
    ok = status == "ok"
    for name in ("area_px2", "cd_equivalent_px", "cd_x_px", "cd_y_px", "perimeter_px"):
        m[name][~ok] = np.nan
    result.valid[:] = ok
    return result


def measure_invariant(
    signal: np.ndarray, state, masks: np.ndarray, origins: np.ndarray, report: np.ndarray, *,
    methods: tuple[str, ...], n_angles: int, psf_fwhm_px: float | None = None,
    nominal_cd_px: float | None = None, dark_level: float = 0.0, flux_ratio_scale: float | None = None,
    rounds: int = 2, half_area_level: float = 0.5, half_area_peak_pixels: float = 4.0, xp=np,
) -> tuple[dict[str, CDMethodResult], dict]:
    """Pulse-invariant methods for the reported cells of a converged ``deblend`` state.

    ``masks``/``origins`` cover every modelled hole (as ``state``); results are
    rows of ``report``. ``disk`` and ``flux_ratio`` need ``psf_fwhm_px`` or
    ``nominal_cd_px`` (the blur is then chosen so the median fitted diameter
    equals it). ``flux_ratio`` is scaled by ``flux_ratio_scale`` when given,
    else so that its median equals ``nominal_cd_px``. ``projection`` and
    the ``half_area`` family need neither and run on ``xp`` (NumPy or CuPy).
    """
    found, info = {}, {}
    if "projection" in methods:
        found["projection"] = _projection(state.cleaned[report], masks[report], origins[report], n_angles, xp=xp)
    for name, (background, averaged) in _HALF_AREA.items():
        if name in methods:
            found[name] = _half_area(state.cleaned[report], masks[report], origins[report], n_angles,
                                     level=half_area_level, background=background,
                                     peak_pixels=half_area_peak_pixels if averaged else None, xp=xp)
    if "disk" not in methods and "flux_ratio" not in methods:
        return found, info
    if psf_fwhm_px is None and nominal_cd_px is None:
        raise ValueError("methods 'disk' and 'flux_ratio' need psf_fwhm or nominal_cd")
    n, h, w = masks.shape
    roi_shape = (h, w)
    gauss = state.params
    gauss_global = gauss.copy()
    gauss_global[:, 1:3] += origins
    order, gx, gy, valid = _pixel_lists(masks)
    raw = np.take_along_axis(windows(signal, origins, roi_shape).reshape(n, -1), order, axis=1)
    # Round 1 neighbours are the deblend Gaussians: raw minus every other hole's model.
    data = np.take_along_axis(
        (state.cleaned + windows(state.background_map, origins, roi_shape)).reshape(n, -1), order, axis=1)
    data = np.where(valid, data, 0.0)
    good = np.isfinite(gauss).all(axis=1) & (state.status == "ok") & (gauss[:, 0] > 0)
    spot_fwhm = float(np.median(2 * FWHM_K * sigma_major(gauss[good]))) if good.any() else 3.0
    rho_max = float(np.hypot(h, w)) + 1.0
    d_max = max(3.0 * spot_fwhm, 2.5 * (nominal_cd_px or 0.0))
    background0 = np.take_along_axis(windows(state.background_map, origins, roi_shape).reshape(n, -1),
                                     order, axis=1)
    background0 = np.where(valid, background0, 0).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)

    def initial(table, d_start, rows):
        p0 = np.empty((len(rows), 5))
        count = np.maximum(valid[rows].sum(1), 1)
        centroid = np.column_stack(((gx[rows] * valid[rows]).sum(1) / count, (gy[rows] * valid[rows]).sum(1) / count))
        p0[:, 1:3] = np.where(good[rows, None], gauss[rows, 1:3], centroid)
        p0[:, 3] = d_start
        peak = np.where(good[rows], gauss[rows, 0], np.maximum(data[rows].max(axis=1) - background0[rows], 1e-3))
        p0[:, 0] = np.maximum(peak / max(float(table.lookup(np.array([d_start]), np.zeros(1), False)[0]), 1e-6), 1e-3)
        p0[:, 4] = background0[rows]
        return p0

    def start_diameter(table):
        if nominal_cd_px is not None:
            return float(np.clip(nominal_cd_px, table.d_min * 2, table.d_max / 2))
        return float(np.clip(spot_fwhm, table.d_min * 2, table.d_max / 2))

    if psf_fwhm_px is None:
        # The fitted diameter falls as the assumed blur grows (the spot width is fixed by the data).
        interior = np.intersect1d(report, np.flatnonzero(good))
        rng = np.random.default_rng(0)
        subset = np.sort(rng.choice(interior, min(len(interior), _CALIBRATION_HOLES), replace=False))
        if not len(subset):
            raise ValueError("no well-fitted holes to calibrate the blur from nominal_cd")

        def median_d(blur):
            table = DiskTable(blur, d_max, rho_max)
            p, status, _, _ = _fit_disks(table, gx[subset], gy[subset], data[subset], valid[subset],
                                         initial(table, start_diameter(table), subset))
            ok = status == "ok"
            return float(np.median(p[ok, 3])) if ok.any() else np.nan

        lo, hi = np.log(0.2 * spot_fwhm), np.log(2.0 * spot_fwhm)
        d_lo, d_hi = median_d(np.exp(lo)), median_d(np.exp(hi))
        if not (np.isfinite(d_lo) and d_lo >= nominal_cd_px):
            raise ValueError(f"nominal_cd ({nominal_cd_px:.3g} px) is larger than any hole that fits these spots "
                             f"(median {d_lo:.3g} px even for a {np.exp(lo):.3g} px blur)")
        if np.isfinite(d_hi) and d_hi > nominal_cd_px:
            raise ValueError(f"nominal_cd ({nominal_cd_px:.3g} px) is too small for these spots "
                             f"(median {d_hi:.3g} px even for a {np.exp(hi):.3g} px blur)")
        for _ in range(14):
            mid = (lo + hi) / 2
            d_mid = median_d(np.exp(mid))
            if np.isfinite(d_mid) and d_mid >= nominal_cd_px:
                lo = mid
            else:
                hi = mid
        psf_fwhm_px = float(np.exp((lo + hi) / 2))
        info["psf_source"] = "calibrated: median disk diameter = nominal_cd"
    else:
        info["psf_source"] = "given"
    table = DiskTable(psf_fwhm_px, d_max, rho_max)
    info["psf_fwhm_px"] = float(psf_fwhm_px)
    everyone = np.arange(n)
    p, status, sigma_d, rmse = _fit_disks(table, gx, gy, data, valid, initial(table, start_diameter(table), everyone))
    # Later rounds: neighbours re-rendered from the previous disk fits, far tails included
    # (Gaussians where a disk fit failed).
    radius = int(max(roi_shape))
    gxg, gyg = roi_grids(origins, roi_shape)
    own_gauss = _render_on_grids(gauss_global, gxg, gyg, _render_radius(gauss_global, radius))
    own_gauss = np.take_along_axis(np.nan_to_num(own_gauss), order, axis=1)
    for _ in range(rounds - 1):
        disk_ok = status == "ok"
        p_global = p.copy()
        p_global[:, 1:3] += origins
        model = render_disks(table, p_global[disk_ok], signal.shape, radius) + far_field(
            table, p_global[disk_ok], signal.shape, radius) + render_models(
            gauss_global[~disk_ok & np.isfinite(gauss).all(axis=1)], signal.shape, radius_cap=radius)
        rho = np.hypot(gx - p[:, 1, None], gy - p[:, 2, None])
        own_disk = p[:, 0, None] * table.lookup(np.broadcast_to(p[:, 3, None], rho.shape), rho, False)
        own = np.where(disk_ok[:, None], np.where(rho <= radius, own_disk, 0.0), own_gauss)
        others = np.take_along_axis(windows(model, origins, roi_shape).reshape(n, -1), order, axis=1) - own
        data = np.where(valid, raw - others, 0.0)
        p0 = np.where(disk_ok[:, None], p, initial(table, start_diameter(table), everyone))
        p, status, sigma_d, rmse = _fit_disks(table, gx, gy, data, valid, p0)
    info["disk_median_cd_px"] = float(np.median(p[report][status[report] == "ok", 3])) if (status[report] == "ok").any() else None

    rows, local = np.arange(len(report)), p[report]
    # The fit input of each reported cell as an ROI window (NaN outside the cell), for inspection.
    cells = np.full((len(report), h * w), np.nan, dtype=np.float32)
    np.put_along_axis(cells, order[report], np.where(valid[report], data[report], np.nan).astype(np.float32), axis=1)
    inputs = {"cells": cells.reshape(len(report), h, w)}
    if "disk" in methods:
        result = _empty("disk", len(report), n_angles, ("amplitude", "local_background", "fit_rmse", "cd_uncertainty_px"))
        result.status[:] = status[report]
        fitted = np.isfinite(local).all(axis=1)
        _set_ellipses(result, rows[fitted], local[fitted, 1:3], local[fitted, 3], local[fitted, 3],
                      origins[report][fitted], n_angles)
        m = result.metrics
        m["amplitude"][:], m["local_background"][:] = local[:, 0], local[:, 4]
        m["peak_signal"][:] = local[:, 0] * table.lookup(local[:, 3], np.zeros(len(report)), False)
        m["fit_rmse"][:], m["cd_uncertainty_px"][:] = rmse[report], sigma_d[report]
        result.inputs.update(inputs)
        result.valid[:] = result.status == "ok"
        for name in ("area_px2", "cd_equivalent_px", "cd_x_px", "cd_y_px", "perimeter_px"):
            m[name][~result.valid] = np.nan
        found["disk"] = result
    if "flux_ratio" in methods:
        result = _empty("flux_ratio", len(report), n_angles, ("flux", "local_background", "flux_ratio"))
        background = local[:, 4]
        flux = np.where(valid[report], data[report] - background[:, None], 0.0).sum(axis=1)
        lit = background - dark_level
        usable = np.isin(status[report], ("ok", "at_diameter_limit")) & (lit > 0) & (flux > 0)
        result.status[:] = np.where(usable, "ok", np.where(np.isin(status[report], ("ok", "at_diameter_limit")),
                                                           "nonpositive_flux_or_background", status[report]))
        ratio = np.where(usable, flux / np.where(lit > 0, lit, np.nan), np.nan)
        apparent = 2 * np.sqrt(ratio / np.pi)
        scale = 1.0 if flux_ratio_scale is None else float(flux_ratio_scale)
        if flux_ratio_scale is None and nominal_cd_px is not None and usable.any():
            scale = float(nominal_cd_px / np.median(apparent[usable]))
        info["flux_ratio_scale"] = scale
        cd = apparent * scale
        _set_ellipses(result, rows[usable], local[usable, 1:3], cd[usable], cd[usable], origins[report][usable], n_angles)
        m = result.metrics
        m["flux"][:], m["local_background"][:], m["flux_ratio"][:] = flux, background, ratio
        m["peak_signal"][:] = np.where(valid[report], data[report], -np.inf).max(axis=1) - background
        result.valid[:] = usable
        result.inputs.update(inputs)
        found["flux_ratio"] = result
    return found, info
