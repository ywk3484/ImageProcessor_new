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
    the cell (every pixel split into 4 x 4 sub-pixels of 1/4 px; the cell is
    zero outside its mask, so its CD does not depend on the array holding
    it); CD = equal-area circle diameter. Assumes no shape.
``half_area_min``
    ``half_area`` with the local level taken as the minimum of the cell's
    outermost pixels instead of their median.
``half_area_min_avg``
    ``half_area_min`` with the peak taken as the mean of the brightest
    pixels (a fixed area) instead of the single brightest point, so one
    outstanding pixel cannot set the threshold.
``tail_area``
    The same area count with a low threshold, near the background, on a
    wider window, ``tail_window`` pitches around the hole, holding the hole's
    own share of the light (each pixel shared among the holes in proportion
    to their fitted models), so the region may extend past the hole's cell. Local level = median of the farthest tenth of the
    hole's own territory (closer to it than to any other hole), where its tail
    has faded; averaged peak; threshold at ``tail_area_level`` of the height.
    The peak's noise enters the threshold only with that weight, and the dim
    tail has less shot noise than the bright core. Its ``mean_signal`` (mean
    light above the local level over the region) is the cell intensity.
``tail_area_norm``
    ``tail_area`` without a local level: each hole's window divided by its
    averaged peak, region where that normalized pattern exceeds
    ``tail_area_level``.
``edge_area``
    On the same window, with neither the peak nor a local level: threshold =
    the window's mean brightness weighted by its steepness (where the light
    falls fastest), so a level counts as much as its outline is long and the
    flat top counts for nothing; the region is the area above it.
``foot_area``
    On the same window, with neither the peak nor a local level: threshold =
    the brightness at the foot of the edge, where the falling slope bends
    into the flat surroundings (below the steepest level, the mean brightness
    weighted by the profile's curvature, within the hole's own territory); the
    region is the area above it. A tail boundary set by the profile's shape,
    not its height.
``foot_area_ratio``
    The foot found once per image instead of per hole: r = median over the
    holes of (foot level / steepest level), each hole's threshold = r x its
    own steepest level. Same tail height, without the per-hole noise of the
    curvature (assumes the holes of an image share the profile shape).

For the ``half_area`` family cd_x / cd_y are the x / y widths of the ellipse
with the region's area and its x and y second moments (the ellipse's diameter
along x and y), so cd_x * cd_y = cd_equivalent ** 2.

``projection`` and the ``half_area`` family run on NumPy or CuPy (``xp``). All work on cells with the neighbouring holes' models removed (from
``deblend``). Lengths are in pixels.
"""

from __future__ import annotations

from functools import lru_cache
import time

import numpy as np

from ..types import CDMethodResult
from .cd import radial_contours
from .deblend import _render_on_grids, _render_radius, render_models, roi_grids, sigma_major, windows, FWHM_K


INVARIANT_METHODS = ("disk", "flux_ratio", "projection", "half_area", "half_area_min", "half_area_min_avg",
                     "tail_area", "tail_area_norm", "edge_area", "foot_area", "foot_area_ratio")
_HALF_AREA = {"half_area": ("median", False), "half_area_min": ("min", False), "half_area_min_avg": ("min", True)}
AIRY_K = 1.0290  # FWHM of the incoherent Airy pattern = AIRY_K / cutoff frequency
_CHUNK = 4096    # holes per batched fit step: (chunk, K) arrays stay ~10 MB each
_GPU_CHUNK_BYTES = 256 * 2**20  # one chunk's enlarged cells on the GPU
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


def _empty(method: str, n: int, n_angles: int, names: tuple[str, ...], *, metrics_only: bool = False) -> CDMethodResult:
    names = ("area_px2", "cd_equivalent_px", "cd_x_px", "cd_y_px", "perimeter_px", "background", "peak_signal") + names
    return CDMethodResult(
        method=method, centers_xy=np.full((n, 2), np.nan),
        contours_xy=np.full((n, 0 if metrics_only else n_angles + 1, 2), np.nan),
        valid=np.zeros(n, bool), status=np.full(n, "no_signal", dtype="U32"),
        metrics={name: np.full(n, np.nan) for name in names},
        geometry={"center_local": np.full((n, 2), np.nan),
                  "edge_px": np.full((n, 0 if metrics_only else n_angles), np.nan)},
    )


def _set_ellipses(result: CDMethodResult, rows, centers_local, cd_x, cd_y, origins, n_angles: int) -> None:
    """Axis-aligned ellipse contours (circles when cd_x == cd_y) through ``radial_contours``."""
    result.geometry["center_local"][rows] = centers_local
    result.centers_xy[rows] = centers_local + origins
    a, b = cd_x / 2, cd_y / 2
    if result.contours_xy.shape[1]:
        angles = np.arange(n_angles) * (2 * np.pi / n_angles)
        edge = a[:, None] * b[:, None] / np.sqrt((b[:, None] * np.cos(angles)) ** 2
                                                 + (a[:, None] * np.sin(angles)) ** 2)
        result.geometry["edge_px"][rows] = edge
        result.contours_xy[rows] = radial_contours(centers_local, edge, origins, n_angles)
    m = result.metrics
    m["cd_x_px"][rows], m["cd_y_px"][rows] = cd_x, cd_y
    m["area_px2"][rows] = np.pi * a * b
    m["cd_equivalent_px"][rows] = a + b  # mean of the x and y widths (equal for circles)
    h = ((a - b) / np.maximum(a + b, 1e-300)) ** 2
    m["perimeter_px"][rows] = np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))  # Ramanujan


_ZERO_PAD = 32  # zeros either side of a cell or profile before its spline: the far end's effect, 0.27^32, is rounding


@lru_cache(maxsize=16)
def _spline_operators(size: int, step: float):
    """Grid and the (grid, size) matrices giving scipy's not-a-knot ``CubicSpline`` of a profile extended with zeros
    on both sides (the cell's light outside it is zero), and its slope, on that grid over the profile.

    The spline is linear in the samples, so the curve of every profile is one matrix product; with the zeros, a
    profile's curve does not depend on how long the array holding it is (the operator of a size is the top-left
    corner of any larger one, to rounding).
    """
    from scipy.interpolate import CubicSpline

    grid = np.arange(0.0, size - 1 + step / 2, step)
    n = size + 2 * _ZERO_PAD
    samples = np.zeros((n, size))
    samples[_ZERO_PAD + np.arange(size), np.arange(size)] = 1.0
    spline = CubicSpline(np.arange(n), samples, axis=0)
    return grid, spline(grid + _ZERO_PAD), spline(grid + _ZERO_PAD, 1)


def _to_numpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _projection(cleaned: np.ndarray, masks: np.ndarray, origins: np.ndarray, n_angles: int,
                step: float = 0.02, xp=np, metrics_only: bool = False) -> CDMethodResult:
    """Summed x/y profiles of each neighbour-removed cell; width between the steepest rise and fall.

    On CuPy each cell is computed in the top-left sub-block of the stack that holds its mask (see ``_half_areas``):
    its profile is zero beyond it, so the spline of the whole stack's width gives the same values there."""
    n, h, w = masks.shape
    result = _empty("projection", n, n_angles, (), metrics_only=metrics_only)
    widths, centers, statuses = np.full((n, 2), np.nan), np.full((n, 2), np.nan), np.full((n, 2), "", dtype="U32")
    peak_signal = result.metrics["peak_signal"]
    buckets = _size_buckets(masks) if xp is not np else [(np.arange(n), h, w)]
    for members, hb, wb in buckets:
        sub = (slice(None), slice(0, hb), slice(0, wb))
        for start in range(0, len(members), _CHUNK):
            rows = members[start:start + _CHUNK]
            mask = xp.asarray(masks[sub][rows])
            cell = xp.asarray(cleaned[sub][rows], dtype=float)
            peak_signal[rows] = _to_numpy(xp.where(mask, cell, -xp.inf).reshape(len(rows), -1).max(axis=1))
            for axis, (size, size_b, sum_axis) in enumerate(((w, wb, 1), (h, hb, 2))):
                counts = _to_numpy(mask.sum(axis=sum_axis))
                # Sum, not mean: the cell outline is ragged by a pixel, and dividing one column by 10 pixels and the
                # next by 11 steps the profile and moves its steepest point (the cleaned cell holds only this hole's
                # light).
                profile = xp.where(mask, cell, 0.0).sum(axis=sum_axis)
                inside = counts > 0
                first = np.argmax(inside, axis=1)
                last = size_b - 1 - np.argmax(inside[:, ::-1], axis=1)
                grid, value_op, slope_op = _spline_operators(size, step)
                used = int(np.searchsorted(grid, size_b - 1 + step / 2))  # grid points up to the sub-block's end
                grid = grid[:used]
                value_op, slope_op = xp.asarray(value_op.T[:size_b, :used]), xp.asarray(slope_op.T[:size_b, :used])
                grid_d, index = xp.asarray(grid), xp.arange(used)[None]
                values, slope = profile @ value_op, profile @ slope_op
                lo, hi = xp.asarray(first)[:, None], xp.asarray(last)[:, None]
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
                bad = (x_left <= first) | (x_right >= last) | ~rising | ~falling
                statuses[rows, axis] = np.where(bad, "edge_at_cell_border", "ok")
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
    """(size * factor, size) matrices of the enlargement along one axis: every pixel split into ``factor``
    sub-pixels of 1/factor px, valued at their centres by the cubic spline of the cell extended with zeros
    (``scipy.ndimage.zoom`` order 3, grid mode, constant 0 outside), and order 0 for masks (each sub-pixel copies
    its pixel).

    Zoom is separable and linear, so a cell enlarges as ``Zy @ cell @ Zx.T``. With the zeros, a pixel's sub-pixels
    do not depend on the size of the array holding the cell: the operator of a size is the top-left corner of any
    larger one (to rounding).
    """
    if factor == 1:
        # Native-pixel methods do not interpolate. A cubic reconstruction of
        # the samples differs by rounding and can move exact threshold ties.
        return np.eye(size), np.eye(size)

    from scipy.ndimage import zoom

    pad = _ZERO_PAD
    eye = np.zeros((size, size + 2 * pad))
    eye[np.arange(size), pad + np.arange(size)] = 1.0
    cubic = np.stack([zoom(e, factor, order=3, mode="grid-constant", grid_mode=True)[pad * factor:(pad + size) * factor]
                      for e in eye], axis=1)
    nearest = np.zeros((size * factor, size))
    nearest[np.arange(size * factor), np.arange(size).repeat(factor)] = 1.0
    return cubic, nearest


@lru_cache(maxsize=16)
def _nearest_pixels(size: int, factor: int) -> np.ndarray:
    """For each pixel of the order-0 enlargement along one axis, the original pixel it copies.

    The order-0 zoom matrix has exactly one 1 per row, so ``mask[:, iy][:, :, ix]`` equals the matrix product
    ``(Zy @ mask @ Zx.T) > 0.5`` exactly.
    """
    nearest = _zoom_operators(size, factor)[1]
    if not (((nearest == 0) | (nearest == 1)).all() and (nearest.sum(axis=1) == 1).all()):
        raise AssertionError("order-0 zoom is not a pixel copy")
    return nearest.argmax(axis=1)


def _grow(region):
    """A batch of 2D boolean images dilated by one pixel in all 8 directions."""
    out = region.copy()
    out[:, 1:] |= region[:, :-1]
    out[:, :-1] |= region[:, 1:]
    grown = out.copy()
    grown[:, :, 1:] |= out[:, :, :-1]
    grown[:, :, :-1] |= out[:, :, 1:]
    return grown


def _connected_to_top(above, top, xp=np, max_steps: int | None = None):
    """Per image of the batch, the pixels of ``above`` connected through 8-neighbours to its ``top`` pixel (flat
    index), grown from it for at most ``max_steps`` (default H * W) steps. The default reaches every pixel even
    along a winding path; growth stops early once unchanged. On CuPy one kernel does it (same result,
    bit for bit)."""
    k, H, W = above.shape
    max_steps = H * W if max_steps is None else max_steps
    if xp is not np:
        from .._gpu.regions import grow_regions
        region = grow_regions(above, top, max_steps)
        if region is not None:
            return region
    region = xp.zeros((k, H * W), dtype=bool)
    region[xp.arange(k), top] = True
    region = region.reshape(k, H, W) & above
    for step in range(max_steps):  # grow from the peak inside the above-threshold pixels until nothing changes
        grown = _grow(region) & above
        if step % 8 == 7 and bool((grown == region).all()):
            break
        region = grown
    return region


def _top_mean(flat, n: int, xp=np):
    """Mean of each row's ``n`` largest values: ``partition(flat, -n)[:, -n:].mean(axis=1)``; on CuPy a selection
    kernel replaces sorting whole rows (same result to floating-point rounding)."""
    if xp is not np:
        from .._gpu.regions import top_mean
        return top_mean(flat, n)
    return xp.partition(flat, -n, axis=1)[:, -n:].mean(axis=1)


def _half_area(cleaned: np.ndarray, masks: np.ndarray, origins: np.ndarray, n_angles: int,
               level: float = 0.5, factor: int = 4, background: str = "median", peak_pixels: float | None = None,
               name: str | None = None, level_masks: np.ndarray | None = None, zone_masks: np.ndarray | None = None,
               cuts: np.ndarray | None = None, outlines: bool = True, xp=np,
               metrics_only: bool = False) -> CDMethodResult:
    """Area above the cell's local level + ``level`` x the peak height, on a ``factor``x cubic enlargement.

    Local level b: ``background`` ('median' or 'min') of the cell's outermost pixels, or 'zero' (none: the
    region is where the cell divided by its peak exceeds ``level``), or 'edge' (none, and no peak either: the
    threshold is the cell's mean brightness weighted by its steepness, ``level`` unused), or 'foot' (none, no
    peak: the threshold is the brightness at the foot of the edge, where the slope bends into the flat
    surroundings, found within ``zone_masks`` when given, metric ``steepest_level`` = the steepness-weighted mean
    brightness there; ``level`` unused), or 'given' (threshold = ``cuts`` per cell) (a pulse that lights the hole lights its surroundings
    too, so b, the peak and the threshold scale together and the area does not). Peak: the highest enlarged
    pixel, or with ``peak_pixels`` the mean of the brightest enlarged pixels covering that many original pixels.
    The region is the connected part around the highest pixel above the threshold; CD = 2 sqrt(area / pi);
    cd_x / cd_y = widths of the equal-area ellipse with the region's x / y second moments; ``mean_signal`` = mean
    of the enlarged cell above b over the region. ``level_masks``: per cell, the pixels b is the median of
    (instead of the outermost pixels). ``outlines=False`` skips the display outline (contours, ``edge_px`` and
    ``perimeter_px`` stay NaN); every other value is unchanged. ``metrics_only=True`` also omits storage for
    contours and angular geometry, returning zero-length angle axes.
    """
    if name is None:
        name = {spec: key for key, spec in _HALF_AREA.items()}[background, peak_pixels is not None]
    spec = {"name": name, "level": level, "background": background, "peak_pixels": peak_pixels, "cuts": cuts}
    return _half_areas(cleaned, masks, origins, n_angles, [spec], factor=factor, level_masks=level_masks,
                       zone_masks=zone_masks, outlines=outlines, xp=xp, metrics_only=metrics_only)[name]


_BUCKETS: list = []  # the last _size_buckets call: (masks, step, result); the same stack is grouped several times


def _size_buckets(masks, step: int = 4) -> list[tuple[np.ndarray, int, int]]:
    """The cells grouped by the extent of their mask (masks sit at the top left of the stack), rounded up to
    ``step`` px and at most the stack size: (rows, height, width) of each group's top-left sub-block."""
    if _BUCKETS and _BUCKETS[0][0] is masks and _BUCKETS[0][1] == step:
        return _BUCKETS[0][2]
    n, h, w = masks.shape
    if not n:
        return []
    rows_any, cols_any = _to_numpy(masks.any(axis=2)), _to_numpy(masks.any(axis=1))
    hb = np.where(rows_any.any(axis=1), h - np.argmax(rows_any[:, ::-1], axis=1), 1)
    wb = np.where(cols_any.any(axis=1), w - np.argmax(cols_any[:, ::-1], axis=1), 1)
    key = np.minimum(-(-hb // step) * step, h) * (w + 1) + np.minimum(-(-wb // step) * step, w)
    buckets = [(np.flatnonzero(key == value), int(value // (w + 1)), int(value % (w + 1))) for value in np.unique(key)]
    _BUCKETS[:] = [(masks, step, buckets)]
    return buckets


def _half_areas(cleaned, masks, origins, n_angles: int, specs: list[dict], *, factor: int = 4, level_masks=None,
                zone_masks=None, outlines: bool = True, xp=np, enlarged: dict | None = None,
                seconds: dict | None = None, rectangles: np.ndarray | None = None,
                metrics_only: bool = False) -> dict[str, CDMethodResult]:
    """``_half_area`` for several methods on the same cells (``specs``: dicts of name, level, background,
    peak_pixels, cuts), the same numbers as one call each: each chunk of cells is enlarged once per local level
    (the methods without one share it) and its averaged peak taken once per enlargement. ``enlarged`` (CuPy): a dict
    keeping those shared enlargements for a later call on the same cells, while they fit in GPU memory.
    ``seconds``: per method, the time of its own work; a shared step counts for the first method that needs it.
    ``rectangles``: (n, 4) rows r0, r1 and columns c0, c1 when every mask is that rectangle of ones (CuPy then
    enlarges a cell with a local level from the shared enlargement, to rounding).

    On CuPy each cell is computed in the top-left sub-block of the stack that holds its mask (cells grouped by that
    size), with the operators and constants of the whole stack: everything outside a mask is zero or False there,
    so a cell's values do not change, but a large cell elsewhere no longer makes every cell's work larger."""
    n, h, w = masks.shape
    outlines = outlines and not metrics_only
    results = {}
    for spec in specs:
        results[spec["name"]] = _empty(spec["name"], n, n_angles, ("local_level", "threshold", "mean_signal")
                                       + (("steepest_level",) if spec["background"] == "foot" else ()),
                                       metrics_only=metrics_only)
    if not n:
        return results
    zy, zx = _zoom_operators(h, factor)[0], _zoom_operators(w, factor)[0]
    H, W = zy.shape[0], zx.shape[0]
    # Original px per enlarged px; enlarged pixel j's centre lies at j / factor - shift original px.
    sy = sx = 1.0 / factor
    shift = (factor - 1) / (2 * factor)
    zy, zx = xp.asarray(zy), xp.asarray(zx)
    near_y, near_x = _nearest_pixels(h, factor), _nearest_pixels(w, factor)
    iy, ix = xp.asarray(near_y)[:, None], xp.asarray(near_x)[None, :]
    gpu = xp is not np
    if gpu:  # one kernel per chunk after the enlargement (subpx._gpu.regions.measure_regions)
        from .._gpu.regions import measure_regions, top_mean_masked
        iy1, ix1 = iy[:, 0].astype(xp.int32), ix[0].astype(xp.int32)
        if enlarged is not None and n * H * W * 8 > min(4 * 2**30, xp.cuda.Device().mem_info[0] // 4):
            enlarged = None  # too large to keep: a later call enlarges again
    if outlines:
        angles = np.arange(n_angles) * (2 * np.pi / n_angles)
        radii = np.arange(0.0, float(np.hypot(h, w)), 0.125)  # original px
        cos_a, sin_a, radii_d = xp.asarray(np.cos(angles)), xp.asarray(np.sin(angles)), xp.asarray(radii)
    gy, gx = xp.arange(H, dtype=float), xp.arange(W, dtype=float)
    status = {name: np.full(n, "no_signal", dtype="U32") for name in results}
    centers = {name: np.full((n, 2), np.nan) for name in results}
    edges = {name: np.full((n, n_angles), np.nan) for name in results} if outlines else {}
    buckets = _size_buckets(masks) if gpu else [(np.arange(n), h, w)]
    for bucket, (members, hb, wb) in enumerate(buckets):
        Yb, Xb = int(np.searchsorted(near_y, hb)), int(np.searchsorted(near_x, wb))  # enlarged rows / columns
        sub = (slice(None), slice(0, hb), slice(0, wb))
        zy_b, zx_b, iy_b, ix_b, gy_b, gx_b = zy[:Yb, :hb], zx[:Xb, :wb], iy[:Yb], ix[:, :Xb], gy[:Yb], gx[:Xb]
        if gpu:
            iy1_b, ix1_b = iy1[:Yb], ix1[:Xb]
            # Running sums of the operators' columns: a rectangle of ones enlarges to an outer product of their sums.
            sum_y = xp.concatenate((xp.zeros((Yb, 1)), xp.cumsum(zy_b, axis=1)), axis=1)
            sum_x = xp.concatenate((xp.zeros((Xb, 1)), xp.cumsum(zx_b, axis=1)), axis=1)
        # ~the memory of 4096 cells of 14 x 14 px; on the GPU ~256 MB per enlarged chunk (fewer, larger steps)
        chunk = max(256, int(_GPU_CHUNK_BYTES // (8 * Yb * Xb)) if gpu else _CHUNK * 196 // (hb * wb))
        for start in range(0, len(members), chunk):
            rows = members[start:start + chunk]
            k = len(rows)
            shared = {}  # this chunk's cells, local levels, enlargements and averaged peaks, made when first needed

            def cells():
                if "cell" not in shared:
                    shared["mask"] = mask = xp.asarray(masks[sub][rows])
                    shared["cell"] = xp.where(mask, xp.asarray(cleaned[sub][rows], dtype=float), 0.0)
                return shared["mask"], shared["cell"]

            def local_level(background):
                kind = background if background in ("median", "min") else "zero"
                if ("b", kind) not in shared:
                    mask, cell = cells()
                    if kind == "zero":  # no local level
                        b = xp.zeros(k)
                    else:
                        inner = mask.copy()
                        inner[:, 1:] &= mask[:, :-1]
                        inner[:, :-1] &= mask[:, 1:]
                        inner[:, :, 1:] &= mask[:, :, :-1]
                        inner[:, :, :-1] &= mask[:, :, 1:]
                        inner[:, [0, -1]] = False
                        inner[:, :, [0, -1]] = False
                        ring = mask & ~inner if level_masks is None else xp.asarray(level_masks[sub][rows]) & mask
                        if kind == "median":
                            b = xp.nanmedian(xp.where(ring, cell, xp.nan).reshape(k, -1), axis=1)
                        else:
                            b = xp.where(ring, cell, xp.inf).reshape(k, -1).min(axis=1)
                    shared["b", kind] = b
                return kind, shared["b", kind]

            def enlargement(kind, b):
                if ("big", kind) not in shared:
                    if enlarged is not None and kind == "zero" and (bucket, start) in enlarged:
                        shared["big", kind] = enlarged[bucket, start]
                    elif gpu and kind != "zero" and rectangles is not None:
                        # Zy (X0 - b M) Zx^T = (the enlargement without a level) - b (Zy M Zx^T), M a rectangle of
                        # ones: equal to rounding, without a second matrix product.
                        r0, r1, c0, c1 = xp.asarray(rectangles[rows]).T
                        ry = (sum_y[:, r1] - sum_y[:, r0]).T * b[:, None]
                        rx = (sum_x[:, c1] - sum_x[:, c0]).T
                        shared["big", kind] = enlargement("zero", xp.zeros(k)) - ry[:, :, None] * rx[:, None, :]
                    else:
                        mask, cell = cells()
                        shared["big", kind] = zy_b[None] @ xp.where(mask, cell - b[:, None, None], 0.0) @ zx_b.T[None]
                        if enlarged is not None and kind == "zero":
                            enlarged[bucket, start] = shared["big", kind]
                return shared["big", kind]

            def averaged_peak(kind, big, n_top):
                if ("peak", kind, n_top) not in shared:
                    if enlarged is not None and kind == "zero" and (bucket, start, n_top) in enlarged:
                        shared["peak", kind, n_top] = enlarged[bucket, start, n_top]
                    else:
                        if n_top > Yb * Xb:  # more pixels than the sub-block: -inf (outside the cell) among them
                            peak = xp.full(k, -xp.inf)
                        else:  # -inf (outside the cell): no_signal
                            peak = top_mean_masked(big, cells()[0], iy1_b, ix1_b, n_top, length=H * W)
                        shared["peak", kind, n_top] = xp.where(xp.isfinite(peak), peak, -1.0)
                        if enlarged is not None and kind == "zero":
                            enlarged[bucket, start, n_top] = shared["peak", kind, n_top]
                return shared["peak", kind, n_top]

            for spec in specs:
                started = time.perf_counter()
                name, level, background = spec["name"], spec["level"], spec["background"]
                peak_pixels, cuts = spec["peak_pixels"], spec["cuts"]
                n_top = None if peak_pixels is None else min(max(1, int(round(peak_pixels * factor * factor))), H * W)
                kind, b = local_level(background)
                big = enlargement(kind, b)
                mask = cells()[0]
                zone = xp.asarray(zone_masks[sub][rows]) if background == "foot" and zone_masks is not None else None
                given = xp.asarray(cuts[rows], dtype=float) if background == "given" else None
                fused = None
                if gpu:
                    fused = measure_regions(
                        big, mask, iy1_b, ix1_b, mode={"edge": "edge", "foot": "foot", "given": "given"}.get(background, "level"),
                        level=level, peak=None if n_top is None else averaged_peak(kind, big, n_top), cuts=given,
                        zone=zone, d=factor, sy=sy, sx=sx, max_steps=Yb * Xb, regions=outlines)
                if fused is not None:
                    peak, cut, steepest, count, touches = (fused[key] for key in ("peak", "cut", "steepest", "count", "touches"))
                    region, cy, cx = fused.get("region"), fused["cy"], fused["cx"]
                    weight = xp.maximum(count, 1)
                    mean_signal = fused["sum"] / weight
                    var_x = fused["sxx"] / weight * sx * sx
                    var_y = fused["syy"] / weight * sy * sy
                else:
                    peak, cut, steepest, region, count, touches, mean_signal, cy, cx, var_x, var_y = _regions_by_arrays(
                        big, mask, iy_b, ix_b, gy_b, gx_b, sy, sx, factor, level, background, n_top, zone, given, Yb * Xb, xp)
                # Equal-area ellipse with the region's x/y spread: its diameters along x and y (exact for an
                # axis-aligned ellipse; cd_x * cd_y = cd_equivalent^2).
                cd_eq = 2 * xp.sqrt(count * (sx * sy) / np.pi)
                ratio = xp.sqrt(xp.sqrt(xp.maximum(var_x, 1e-12) / xp.maximum(var_y, 1e-12)))
                extent_x, extent_y = cd_eq * ratio, cd_eq / ratio
                values = [("b", b), ("peak", peak), ("cut", cut), ("mean_signal", mean_signal),
                          ("area", count * (sx * sy)), ("ex", extent_x), ("ey", extent_y), ("touches", touches),
                          ("cx", cx * sx - shift), ("cy", cy * sy - shift)]
                if outlines:
                    # Contour for display: the region's last pixel along each ray from its centroid. Rays stop just
                    # beyond the farthest region pixel of the chunk (a few px), not at the cell diagonal; pixels past
                    # the sub-block are outside every region.
                    far = (((gx_b[None, None] - cx[:, None, None]) * sx) ** 2
                           + ((gy_b[None, :, None] - cy[:, None, None]) * sy) ** 2)
                    reach_max = float(xp.sqrt(xp.where(region, far, 0.0).max())) if k else 0.0
                    steps = radii_d[:min(len(radii), int(reach_max / 0.125) + 3)]
                    px = xp.rint(cx[:, None, None] + cos_a[None, :, None] * steps[None, None] / sx).astype(xp.int64)
                    py = xp.rint(cy[:, None, None] + sin_a[None, :, None] * steps[None, None] / sy).astype(xp.int64)
                    inside = (px >= 0) & (px < Xb) & (py >= 0) & (py < Yb)
                    index = (xp.clip(py, 0, Yb - 1) * Xb + xp.clip(px, 0, Xb - 1)).reshape(k, -1)
                    hit = xp.take_along_axis(region.reshape(k, -1), index, axis=1).reshape(px.shape) & inside
                    first_out = xp.where(hit.all(axis=2), len(steps), xp.argmin(hit, axis=2))
                    values.append(("edge", steps[xp.maximum(first_out - 1, 0)] + 0.0625))
                if gpu:  # one transfer for all per-cell numbers
                    numbers = [(key, v) for key, v in values if key != "edge"]
                    host = _to_numpy(xp.stack([xp.asarray(v, dtype=float) for _, v in numbers]))
                    out = {key: host[i] for i, (key, _) in enumerate(numbers)}
                    out["touches"] = out["touches"] != 0
                    if outlines:
                        out["edge"] = _to_numpy(values[-1][1])
                else:
                    out = {key: _to_numpy(v) for key, v in values}
                signal = (out["peak"] > 0) & np.isfinite(out["cut"]) & (out["cut"] < out["peak"])
                m = results[name].metrics
                m["local_level"][rows] = out["b"]
                m["peak_signal"][rows] = np.where(signal, out["peak"], np.nan)
                m["threshold"][rows] = np.where(signal, out["b"] + out["cut"], np.nan)
                m["mean_signal"][rows] = np.where(signal, out["mean_signal"], np.nan)
                if background == "foot":
                    m["steepest_level"][rows] = np.where(signal, _to_numpy(steepest), np.nan)
                m["area_px2"][rows] = np.where(signal, out["area"], np.nan)
                m["cd_x_px"][rows] = np.where(signal, out["ex"], np.nan)
                m["cd_y_px"][rows] = np.where(signal, out["ey"], np.nan)
                status[name][rows] = np.where(~signal, "no_signal",
                                              np.where(out["touches"], "region_at_cell_border", "ok"))
                centers[name][rows] = np.column_stack((out["cx"], out["cy"]))
                if outlines:
                    edges[name][rows] = out["edge"]
                if seconds is not None:
                    seconds[name] = seconds.get(name, 0.0) + time.perf_counter() - started  # results are on the host
    for name, result in results.items():
        m = result.metrics
        m["cd_equivalent_px"][:] = 2 * np.sqrt(m["area_px2"] / np.pi)
        rows = np.flatnonzero(status[name] != "no_signal")
        result.geometry["center_local"][rows] = centers[name][rows]
        result.centers_xy[rows] = centers[name][rows] + origins[rows]
        if outlines:
            result.geometry["edge_px"][rows] = edges[name][rows]
            result.contours_xy[rows] = radial_contours(centers[name][rows], edges[name][rows], origins[rows], n_angles)
            m["perimeter_px"][rows] = np.linalg.norm(np.diff(result.contours_xy[rows], axis=1), axis=2).sum(axis=1)
        result.status[:] = status[name]
        ok = status[name] == "ok"
        for metric in ("area_px2", "cd_equivalent_px", "cd_x_px", "cd_y_px", "perimeter_px"):
            m[metric][~ok] = np.nan
        result.valid[:] = ok
    return results


def _regions_by_arrays(big, mask, iy, ix, gy, gx, sy, sx, factor, level, background, n_top, zone, cuts, max_steps,
                       xp):
    """``_half_area``'s per-cell values from the enlarged cells ``big`` with array operations (NumPy, or CuPy when
    a cell is too large for ``measure_regions``): peak, threshold, steepest level (foot), region, pixel count,
    touches, mean signal, centroid (enlarged px) and x / y variances (original px^2). ``n_top``: pixels of the
    averaged peak (None: the highest pixel); ``zone``: the foot's zone masks; ``cuts``: the given thresholds."""
    k, H, W = big.shape
    steepest = None
    big_mask = mask[:, iy, ix]  # order-0 enlargement of the mask
    flat = xp.where(big_mask, big, -xp.inf).reshape(k, -1)
    top = xp.argmax(flat, axis=1)
    peak = flat[xp.arange(k), top]
    if n_top is not None:
        # -inf (outside the cell): no_signal; more pixels than the array: -inf (outside the cell) among them
        peak = _top_mean(flat, n_top, xp) if n_top <= H * W else xp.full(k, -xp.inf)
        peak = xp.where(xp.isfinite(peak), peak, -1.0)
    if background == "edge":
        # Threshold = the mean brightness on the slope, each pixel weighted by its steepness (central difference,
        # +-1 original px): each brightness level counts as much as its outline is long, so the flat top (the
        # peak) and the flat surroundings count for ~nothing. No peak, no local level.
        d = factor
        pad = xp.pad(big, ((0, 0), (d, d), (d, d)), mode="edge")
        steep = xp.sqrt(((pad[:, 2 * d:, d:-d] - pad[:, :-2 * d, d:-d]) / (2 * d * sy)) ** 2
                        + ((pad[:, d:-d, 2 * d:] - pad[:, d:-d, :-2 * d]) / (2 * d * sx)) ** 2)
        core = big_mask.copy()  # pixels whose whole difference stencil lies in the cell
        for axis in (1, 2):
            for shift in (d, -d):
                core &= xp.roll(big_mask, shift, axis=axis)
        core[:, :d] = core[:, -d:] = False
        core[:, :, :d] = core[:, :, -d:] = False
        wgt = xp.where(core, steep, 0.0)
        total = wgt.sum(axis=(1, 2))
        cut = xp.where(total > 0, (wgt * big).sum(axis=(1, 2)) / xp.where(total > 0, total, 1.0), xp.inf)
    elif background == "foot":
        # The foot of the edge, where the falling slope bends into the flat surroundings (no peak, no local
        # level): below the steepest level, the mean brightness with each pixel weighted by the profile's
        # (signed) curvature, which peaks at that bend. Signed weights: noise averages out instead of adding up.
        d = factor  # differences between the original pixels either side
        pad = xp.pad(big, ((0, 0), (d, d), (d, d)), mode="edge")
        up, down = pad[:, 2 * d:, d:-d], pad[:, :-2 * d, d:-d]
        right, left = pad[:, d:-d, 2 * d:], pad[:, d:-d, :-2 * d]
        steep = xp.sqrt(((up - down) / (2 * d * sy)) ** 2 + ((right - left) / (2 * d * sx)) ** 2)
        curve = (up + down - 2 * big) / (d * sy) ** 2 + (right + left - 2 * big) / (d * sx) ** 2
        core = big_mask.copy()  # pixels whose whole difference stencil lies in the cell
        for axis in (1, 2):
            for shift in (d, -d):
                core &= xp.roll(big_mask, shift, axis=axis)
        core[:, :d] = core[:, -d:] = False
        core[:, :, :d] = core[:, :, -d:] = False
        if zone is not None:  # the hole's own territory: not the dip where its share of the light ends
            core &= zone[:, iy, ix]
        wgt = xp.where(core, steep, 0.0)
        total = wgt.sum(axis=(1, 2))
        steepest = (wgt * big).sum(axis=(1, 2)) / xp.where(total > 0, total, 1.0)
        outer = core & (big < steepest[:, None, None])
        bend = xp.where(outer, curve, 0.0).sum(axis=(1, 2))
        cut = xp.where((total > 0) & (bend > 0),
                       xp.where(outer, curve * big, 0.0).sum(axis=(1, 2)) / xp.where(bend > 0, bend, 1.0), xp.inf)
    elif background == "given":
        cut = cuts
    else:
        cut = level * peak
    above = big_mask & (big > cut[:, None, None])
    region = _connected_to_top(above, top, xp, max_steps)
    count = region.sum(axis=(1, 2))
    touches = ((_grow(region) & ~big_mask).any(axis=(1, 2)) | region[:, [0, -1]].any(axis=(1, 2))
               | region[:, :, [0, -1]].any(axis=(1, 2)))
    weight = xp.maximum(count, 1)
    mean_signal = xp.where(region, big, 0.0).sum(axis=(1, 2)) / weight
    cy = (region.sum(axis=2) * gy).sum(axis=1) / weight
    cx = (region.sum(axis=1) * gx).sum(axis=1) / weight
    var_x = (region.sum(axis=1) * (gx[None] - cx[:, None]) ** 2).sum(axis=1) / weight * sx * sx
    var_y = (region.sum(axis=2) * (gy[None] - cy[:, None]) ** 2).sum(axis=1) / weight * sy * sy
    return peak, cut, steepest, region, count, touches, mean_signal, cy, cx, var_x, var_y


def _wide_cells(signal: np.ndarray, state, masks: np.ndarray, origins: np.ndarray, report: np.ndarray,
                pitch: float, radius_pitches: float, xp=np):
    """Each reported hole on a square window of half-width ``radius_pitches`` x ``pitch`` around its fitted centre,
    holding its own share of the light, so its region may extend past its cell: the image minus the background
    map, each pixel shared among the holes in proportion to their fitted models there (own model / sum of all).
    Near a neighbour's centre the neighbour's share is ~all however well its model fits, so a misfit (big, square
    or ringing spots) leaves no bright leftover for the region to spill into. Returns (cells, inside the image,
    level pixels, own territory, window origins): the level pixels are the farthest tenth (at least 4 pixels) of the hole's own
    territory (closer to it than to any other hole), the darkest place where its own tail has faded."""
    from scipy.spatial import cKDTree

    n_all, h, w = masks.shape
    cap = int(max(h, w))  # the deblend's render radius cap: the own model cancels exactly
    params = state.params.astype(float).copy()
    params[:, 1:3] += origins
    fallback = origins + np.array([(w - 1) / 2, (h - 1) / 2])
    centers_all = np.where(np.isfinite(params[:, 1:3]), params[:, 1:3], fallback)
    rows_rep = np.arange(n_all)[report]  # report: row numbers (or a mask) of the reported holes
    centers = centers_all[rows_rep]
    radius = int(np.ceil(radius_pitches * pitch))
    size = 2 * radius + 1
    win_origins = np.rint(centers).astype(np.int64) - radius
    if not len(rows_rep):
        shape = (0, size, size)
        return (np.empty(shape, np.float32), np.empty(shape, bool), np.empty(shape, bool),
                np.empty(shape, bool), win_origins.astype(float))
    k = min(9, n_all)
    neighbours = cKDTree(centers_all).query(centers, k=k)[1].reshape(len(centers), k)
    neighbours[:, 0] = rows_rep  # itself first, even when two holes share a centre
    data = xp.asarray(signal - state.background_map)
    model = xp.asarray(state.model_image)
    good_amp = params[np.isfinite(params).all(axis=1), 0]
    floor = 1e-3 * float(np.median(good_amp)) if good_amp.size else 1e-12
    height, width = signal.shape
    n = len(rows_rep)
    if xp is not np:  # one kernel; the windows stay on the GPU
        from .._gpu.windows import wide_cells
        own_params = params[rows_rep]
        good = np.isfinite(own_params).all(axis=1)
        out = wide_cells(data, model, win_origins, np.where(good[:, None], own_params, 0.0), good,
                         _render_radius(np.where(np.isfinite(own_params), own_params, 0.0), cap),
                         centers_all[neighbours], floor, size)
        if out is not None:
            return (*out, win_origins.astype(float))
    cells = np.zeros((n, size, size), np.float32)
    inside_all = np.zeros((n, size, size), bool)
    level_all = np.zeros((n, size, size), bool)
    own_all = np.zeros((n, size, size), bool)
    offsets = xp.arange(size)
    for start in range(0, n, 2048):
        sl = slice(start, min(start + 2048, n))
        o = xp.asarray(win_origins[sl])
        ys = o[:, 1, None] + offsets[None]
        xs = o[:, 0, None] + offsets[None]
        inside = ((ys >= 0) & (ys < height))[:, :, None] & ((xs >= 0) & (xs < width))[:, None, :]
        pick = (xp.clip(ys, 0, height - 1)[:, :, None], xp.clip(xs, 0, width - 1)[:, None, :])
        win, total = data[pick], model[pick]
        gp = xp.asarray(params[rows_rep[sl]])
        good = xp.isfinite(gp).all(axis=1)
        gp = xp.where(good[:, None], gp, 0.0)
        render = xp.asarray(_render_radius(np.where(np.isfinite(params[rows_rep[sl]]), params[rows_rep[sl]], 0.0), cap))
        gx = xs[:, None, :].astype(float)
        gy = ys[:, :, None].astype(float)
        dx = gx - gp[:, 1, None, None]
        dy = gy - gp[:, 2, None, None]
        a = xp.exp(gp[:, 3, None, None]) * dx + gp[:, 4, None, None] * dy
        b = xp.exp(gp[:, 5, None, None]) * dy
        box = (xp.abs(dx) <= render[:, None, None]) & (xp.abs(dy) <= render[:, None, None]) & good[:, None, None]
        own = xp.where(box, gp[:, 0, None, None] * xp.exp(-0.5 * (a * a + b * b)), 0.0)
        near = xp.asarray(centers_all[neighbours[sl]])  # (c, k, 2)
        dist = xp.sqrt((gx[:, None] - near[:, :, 0, None, None]) ** 2 + (gy[:, None] - near[:, :, 1, None, None]) ** 2)
        mine = dist[:, 0] <= dist[:, 1:].min(axis=1) if k > 1 else xp.ones_like(dist[:, 0], dtype=bool)
        # Level pixels: the farthest tenth (at least 4 pixels) of the hole's own territory, whatever the layout
        # (square, staggered, irregular): where its own light has faded most.
        own_area = inside & mine
        rows_c = own_area.shape[0]
        far = xp.sort(xp.where(own_area, dist[:, 0], -xp.inf).reshape(rows_c, -1), axis=1)[:, ::-1]
        count = own_area.reshape(rows_c, -1).sum(axis=1)
        keep = xp.clip(xp.maximum(count // 10, 4), 1, far.shape[1]) - 1
        level = own_area & (dist[:, 0] >= far[xp.arange(rows_c), keep][:, None, None])
        # Each pixel's light is shared among the holes in proportion to their models there: near a neighbour's
        # centre its share is ~all, however well its model fits, so a misfit leaves no bright leftover to spill into.
        cells[sl] = _to_numpy(xp.where(inside, win * own / (total + floor), 0.0)).astype(np.float32)
        inside_all[sl] = _to_numpy(inside)
        level_all[sl] = _to_numpy(level)
        own_all[sl] = _to_numpy(own_area)
    return cells, inside_all, level_all, own_all, win_origins.astype(float)


def measure_invariant(
    signal: np.ndarray, state, masks: np.ndarray, origins: np.ndarray, report: np.ndarray, *,
    methods: tuple[str, ...], n_angles: int, psf_fwhm_px: float | None = None,
    nominal_cd_px: float | None = None, dark_level: float = 0.0, flux_ratio_scale: float | None = None,
    rounds: int = 2, half_area_level: float = 0.5, half_area_peak_pixels: float = 4.0, tail_area_level: float = 0.2,
    tail_window: float = 1.2, pitch_px: float | None = None, outlines: bool = True, xp=np,
    metrics_only: bool = False,
) -> tuple[dict[str, CDMethodResult], dict]:
    """Pulse-invariant methods for the reported cells of a converged ``deblend`` state.

    ``masks``/``origins`` cover every modelled hole (as ``state``); results are
    rows of ``report``. ``disk`` and ``flux_ratio`` need ``psf_fwhm_px`` or
    ``nominal_cd_px`` (the blur is then chosen so the median fitted diameter
    equals it). ``flux_ratio`` is scaled by ``flux_ratio_scale`` when given,
    else so that its median equals ``nominal_cd_px``. ``projection`` and
    the ``half_area`` family need neither and run on ``xp`` (NumPy or CuPy). ``tail_area``'s window half-width is
    ``tail_window`` x ``pitch_px`` (the median nearest-neighbour distance of the holes when not given).
    ``outlines=False``: no display outlines for the ``half_area`` family (see ``_half_area``).
    ``metrics_only=True`` also skips display input windows and contour allocations for every method.
    """
    found, info = {}, {}
    seconds = info["seconds"] = {}  # per method (results are on the host when each call returns)
    start = time.perf_counter()
    half = [name for name in _HALF_AREA if name in methods]
    if "projection" in methods or half:
        cleaned_r, masks_r, origins_r = state.cleaned[report], masks[report], origins[report]
    if "projection" in methods:
        found["projection"] = _projection(cleaned_r, masks_r, origins_r, n_angles, xp=xp, metrics_only=metrics_only)
        seconds["projection"], start = time.perf_counter() - start, time.perf_counter()
    if half:  # one call: half_area_min and half_area_min_avg share their enlargement
        specs = [{"name": name, "level": half_area_level, "background": _HALF_AREA[name][0],
                  "peak_pixels": half_area_peak_pixels if _HALF_AREA[name][1] else None, "cuts": None}
                 for name in half]
        found.update(_half_areas(cleaned_r, masks_r, origins_r, n_angles, specs, outlines=outlines, xp=xp,
                                 seconds=seconds, metrics_only=metrics_only))
        start = time.perf_counter()
    tails = [name for name in ("tail_area", "tail_area_norm", "edge_area", "foot_area", "foot_area_ratio")
             if name in methods]
    if tails:
        pitch = pitch_px
        if pitch is None or not np.isfinite(pitch):
            from scipy.spatial import cKDTree
            centers = state.params[:, 1:3] + origins
            centers = centers[np.isfinite(centers).all(axis=1)]
            pitch = float(np.median(cKDTree(centers).query(centers, k=2)[0][:, 1])) if len(centers) > 1 else float(max(masks.shape[1:]))
        cells, inside, level_masks, territory, win_origins = _wide_cells(signal, state, masks, origins, report, pitch, tail_window, xp=xp)
        shown = {}
        if not metrics_only:
            # For display: the middle of each window, cut to 0.75 pitch around the hole.
            keep = min(cells.shape[1] // 2, int(np.ceil(0.75 * pitch)))
            middle = cells.shape[1] // 2
            cut = slice(middle - keep, middle + keep + 1)
            shown = dict(cells=_to_numpy(cells[:, cut, cut]), inside=_to_numpy(inside[:, cut, cut]),
                         origins=win_origins + (middle - keep))
        seconds["tail_window"], start = time.perf_counter() - start, time.perf_counter()
        backgrounds = {"tail_area": "median", "tail_area_norm": "zero", "edge_area": "edge", "foot_area": "foot",
                       "foot_area_ratio": "given"}
        # foot_area_ratio needs foot_area (computed, unreported, when not asked for); the four methods without a local
        # level share one enlargement of each window, kept for foot_area_ratio's second pass when it fits.
        first = [name for name in tails if name != "foot_area_ratio"]
        if "foot_area_ratio" in tails and "foot_area" not in first:
            first.append("foot_area")
        enlarged = {} if xp is not np else None
        # Each window's mask is the rectangle of it inside the image: rows r0:r1, columns c0:c1.
        size, (height, width) = cells.shape[1], signal.shape
        ox, oy = win_origins[:, 0].astype(np.int64), win_origins[:, 1].astype(np.int64)
        r0, c0 = np.clip(-oy, 0, size), np.clip(-ox, 0, size)
        rects = np.column_stack((r0, np.clip(height - oy, r0, size), c0, np.clip(width - ox, c0, size)))
        common = dict(level_masks=level_masks, zone_masks=territory, outlines=outlines, xp=xp, enlarged=enlarged,
                      rectangles=rects, metrics_only=metrics_only)
        tail_found = _half_areas(cells, inside, win_origins, n_angles,
                                 [{"name": name, "level": tail_area_level, "background": backgrounds[name],
                                   "peak_pixels": half_area_peak_pixels, "cuts": None} for name in first],
                                 seconds=seconds, **common)
        if "foot_area_ratio" in tails:
            foot = tail_found["foot_area"]
            steepest = foot.metrics["steepest_level"]
            ratio = foot.metrics["threshold"] / steepest
            ratio = float(np.nanmedian(ratio[foot.valid])) if foot.valid.any() else np.nan
            info["foot_ratio"] = ratio
            if "foot_area" not in tails:
                seconds["foot_area_ratio"] = seconds.pop("foot_area")
            tail_found.update(_half_areas(cells, inside, win_origins, n_angles,
                                          [{"name": "foot_area_ratio", "level": tail_area_level, "background": "given",
                                            "peak_pixels": half_area_peak_pixels, "cuts": ratio * steepest}],
                                          seconds=seconds, **common))
        for name in tails:
            found[name] = tail_found[name]
            found[name].inputs.update(shown)
            # Positions relative to the hole's own cell, as the other methods (saved results rebuild contours from them).
            local = found[name].centers_xy - origins[report]
            found[name].geometry["center_local"][:] = np.where(np.isfinite(local), local, np.nan)
        start = time.perf_counter()
        info["tail_window_px"] = int(cells.shape[1])
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
    inputs = {}
    if not metrics_only:
        cells = np.full((len(report), h * w), np.nan, dtype=np.float32)
        np.put_along_axis(cells, order[report], np.where(valid[report], data[report], np.nan).astype(np.float32), axis=1)
        inputs = {"cells": cells.reshape(len(report), h, w)}
    if "disk" in methods:
        result = _empty("disk", len(report), n_angles, ("amplitude", "local_background", "fit_rmse", "cd_uncertainty_px"),
                        metrics_only=metrics_only)
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
        result = _empty("flux_ratio", len(report), n_angles, ("flux", "local_background", "flux_ratio"),
                        metrics_only=metrics_only)
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
