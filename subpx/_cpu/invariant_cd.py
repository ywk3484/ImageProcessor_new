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
    Mean x and y profiles of the cell; width between the steepest rise and
    fall of each; CD = mean of the two widths.

All work on cells with the neighbouring holes' models removed (from
``deblend``). Lengths are in pixels.
"""

from __future__ import annotations

import numpy as np

from ..types import CDMethodResult
from .cd import radial_contours
from .deblend import _render_on_grids, _render_radius, render_models, roi_grids, sigma_major, windows, FWHM_K


INVARIANT_METHODS = ("disk", "flux_ratio", "projection")
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


def _projection(cleaned: np.ndarray, masks: np.ndarray, origins: np.ndarray, n_angles: int,
                step: float = 0.02) -> CDMethodResult:
    """Mean x/y profiles of each neighbour-removed cell; width between the steepest rise and fall."""
    from scipy.interpolate import CubicSpline

    n, h, w = masks.shape
    result = _empty("projection", n, n_angles, ())
    widths, centers, statuses = np.full((n, 2), np.nan), np.full((n, 2), np.nan), np.full((n, 2), "", dtype="U32")
    for axis, (size, sum_axis) in enumerate(((w, 1), (h, 2))):
        counts = masks.sum(axis=sum_axis)
        profiles = np.where(masks, cleaned, 0.0).sum(axis=sum_axis) / np.maximum(counts, 1)
        inside = counts > 0
        first = np.argmax(inside, axis=1)
        last = size - 1 - np.argmax(inside[:, ::-1], axis=1)
        grid = np.arange(0.0, size - 1 + step / 2, step)
        for start in range(0, n, _CHUNK):
            rows = np.arange(start, min(start + _CHUNK, n))
            spline = CubicSpline(np.arange(size), profiles[rows], axis=1)
            values, slope = spline(grid), spline(grid, 1)
            lo, hi = first[rows, None], last[rows, None]
            span = (grid[None] >= lo) & (grid[None] <= hi)
            peak = np.argmax(np.where(span, values, -np.inf), axis=1)
            index = np.arange(len(grid))[None]
            left = np.argmax(np.where(span & (index <= peak[:, None]), slope, -np.inf), axis=1)
            right = np.argmin(np.where(span & (index >= peak[:, None]), slope, np.inf), axis=1)
            x_left, x_right = grid[left], grid[right]
            widths[rows, axis] = x_right - x_left
            centers[rows, axis] = (x_left + x_right) / 2
            bad = ((x_left <= first[rows]) | (x_right >= last[rows]) | (slope[np.arange(len(rows)), left] <= 0)
                   | (slope[np.arange(len(rows)), right] >= 0))
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


def measure_invariant(
    signal: np.ndarray, state, masks: np.ndarray, origins: np.ndarray, report: np.ndarray, *,
    methods: tuple[str, ...], n_angles: int, psf_fwhm_px: float | None = None,
    nominal_cd_px: float | None = None, dark_level: float = 0.0, rounds: int = 2,
) -> tuple[dict[str, CDMethodResult], dict]:
    """Pulse-invariant methods for the reported cells of a converged ``deblend`` state.

    ``masks``/``origins`` cover every modelled hole (as ``state``); results are
    rows of ``report``. ``disk`` and ``flux_ratio`` need ``psf_fwhm_px`` or
    ``nominal_cd_px`` (the blur is then chosen so the median fitted diameter
    equals it).
    """
    found, info = {}, {}
    if "projection" in methods:
        found["projection"] = _projection(state.cleaned[report], masks[report], origins[report], n_angles)
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
        scale = 1.0
        if nominal_cd_px is not None and usable.any():
            scale = float(nominal_cd_px / np.median(apparent[usable]))
        info["flux_ratio_scale"] = scale
        cd = apparent * scale
        _set_ellipses(result, rows[usable], local[usable, 1:3], cd[usable], cd[usable], origins[report][usable], n_angles)
        m = result.metrics
        m["flux"][:], m["local_background"][:], m["flux_ratio"][:] = flux, background, ratio
        m["peak_signal"][:] = np.where(valid[report], data[report], -np.inf).max(axis=1) - background
        result.valid[:] = usable
        found["flux_ratio"] = result
    return found, info
