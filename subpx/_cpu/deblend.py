"""Joint deblending: a smooth background map plus one 2-D Gaussian per hole.

The image is modelled as ``B(x, y) + sum_k G_k(x, y)``. Alternating between a
background map estimated from the residual and per-hole Gaussian fits on
neighbour-subtracted cells removes both detector-noise bias (no percentile
background) and neighbour-tail contamination, including dense arrays where no
tail-free pixels exist.

Gaussian parameters are ROI-local ``[A, x0, y0, u1, u2, u3]``. The precision
matrix is ``L @ L.T`` with ``L = [[exp(u1), 0], [u2, exp(u3)]]``, which keeps it
positive definite throughout the fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


RENDER_SIGMAS = 4.0
FWHM_K = float(np.sqrt(2 * np.log(2)))
# Narrower than this is below pixel sampling: such fits are noise spikes, not holes.
_SIGMA_MIN = 0.5
# Bounds the (N, K, 6) Jacobian of one fit chunk to ~100 MB.
_CHUNK_ELEMENTS = 2_000_000


@dataclass
class DeblendState:
    """Converged deblending of one image.

    ``cleaned`` holds each cell's pixels with the background map and all other
    holes' models removed (zero outside the cell). ``params`` are ROI-local.
    """

    background_map: np.ndarray
    model_image: np.ndarray
    params: np.ndarray
    status: np.ndarray
    cleaned: np.ndarray
    tile_px: int
    iterations: int
    background_info: dict
    converged: bool = False


def precision_matrices(params: np.ndarray) -> np.ndarray:
    """(N, 2, 2) precision matrices from Cholesky parameters."""
    e1, u2, e3 = np.exp(params[:, 3]), params[:, 4], np.exp(params[:, 5])
    out = np.empty((len(params), 2, 2))
    out[:, 0, 0] = e1 * e1
    out[:, 0, 1] = out[:, 1, 0] = e1 * u2
    out[:, 1, 1] = u2 * u2 + e3 * e3
    return out


def sigma_major(params: np.ndarray) -> np.ndarray:
    eig = np.linalg.eigvalsh(precision_matrices(params))[:, 0]
    return 1 / np.sqrt(np.maximum(eig, 1e-300))


def _gaussian(params: np.ndarray, dx: np.ndarray, dy: np.ndarray):
    """Model values and the pieces its Jacobian reuses; dx, dy are (N, K)."""
    e1 = np.exp(params[:, 3])[:, None]
    u2 = params[:, 4][:, None]
    e3 = np.exp(params[:, 5])[:, None]
    a = e1 * dx + u2 * dy
    b = e3 * dy
    shape = np.exp(-0.5 * (a * a + b * b))
    return params[:, 0][:, None] * shape, shape, a, b, e1, u2, e3


def _render_on_grids(params: np.ndarray, gx: np.ndarray, gy: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Truncated models on (N, K) global grids; centers in params are global."""
    dx = gx - params[:, 1][:, None]
    dy = gy - params[:, 2][:, None]
    values = _gaussian(params, dx, dy)[0]
    inside = (np.abs(dx) <= radius[:, None]) & (np.abs(dy) <= radius[:, None])
    return np.where(inside, values, 0.0)


def _render_radius(params: np.ndarray, cap: int) -> np.ndarray:
    return np.clip(np.ceil(RENDER_SIGMAS * sigma_major(params)), 2, cap)


def render_models(params_global: np.ndarray, shape: tuple[int, int], *, radius_cap: int = 64) -> np.ndarray:
    """Sum all truncated Gaussians (global centers) into an (H, W) image."""
    height, width = shape
    out = np.zeros(height * width)
    if len(params_global) == 0:
        return out.reshape(shape)
    radius = _render_radius(params_global, radius_cap)
    r_max = int(radius.max())
    offsets = np.arange(-r_max, r_max + 1)
    per_hole = len(offsets) ** 2
    chunk = max(1, _CHUNK_ELEMENTS // per_hole)
    for start in range(0, len(params_global), chunk):
        p = params_global[start:start + chunk]
        ix = np.floor(p[:, 1]).astype(np.int64)[:, None, None] + offsets[None, None, :]
        iy = np.floor(p[:, 2]).astype(np.int64)[:, None, None] + offsets[None, :, None]
        ix, iy = np.broadcast_arrays(ix, iy)
        ix, iy = ix.reshape(len(p), -1), iy.reshape(len(p), -1)
        values = _render_on_grids(p, ix.astype(float), iy.astype(float), radius[start:start + chunk])
        ok = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height) & (values != 0)
        out += np.bincount((iy * width + ix)[ok], weights=values[ok], minlength=height * width)
    return out.reshape(shape)


def model_on_rois(params: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Untruncated ROI-local models on (N, h, w) grids."""
    yy, xx = np.indices(shape, dtype=float)
    dx = xx.ravel()[None] - params[:, 1, None]
    dy = yy.ravel()[None] - params[:, 2, None]
    return _gaussian(params, dx, dy)[0].reshape(len(params), *shape)


def windows(image: np.ndarray, origins_xy: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """(N, h, w) windows at integer origins; out-of-image pixels are zero."""
    height, width = image.shape
    ys = origins_xy[:, 1, None] + np.arange(shape[0])[None, :]
    xs = origins_xy[:, 0, None] + np.arange(shape[1])[None, :]
    inside = ((ys >= 0) & (ys < height))[:, :, None] & ((xs >= 0) & (xs < width))[:, None, :]
    out = image[np.clip(ys, 0, height - 1)[:, :, None], np.clip(xs, 0, width - 1)[:, None, :]]
    return np.where(inside, out, 0.0)


def initial_params(data: np.ndarray, masks: np.ndarray, seeds_local: np.ndarray) -> np.ndarray:
    """Amplitude from the cell maximum, width from second moments of the core."""
    n = len(data)
    params = np.zeros((n, 6))
    if n == 0:
        return params
    values = np.where(masks, data, -np.inf)
    peak = values.reshape(n, -1).max(axis=1)
    core = masks & (data > 0.2 * peak[:, None, None])
    weight = np.where(core, data, 0.0)
    yy, xx = np.indices(data.shape[1:], dtype=float)
    total = np.maximum(weight.sum(axis=(1, 2)), 1e-300)
    dx = xx[None] - seeds_local[:, 0, None, None]
    dy = yy[None] - seeds_local[:, 1, None, None]
    var = ((weight * (dx * dx + dy * dy)).sum(axis=(1, 2)) / total) / 2
    sigma = np.sqrt(np.clip(var, 0.25, None))
    params[:, 0] = np.maximum(peak, 1e-12)
    params[:, 1:3] = seeds_local
    params[:, 3] = params[:, 5] = -np.log(sigma)
    return params


def fit_gaussians_batch(
    data: np.ndarray, masks: np.ndarray, init: np.ndarray, *, max_iter: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Levenberg-Marquardt fits of one Gaussian per (H, W) cell, all cells at once.

    Least squares on linear intensities is unbiased under additive noise,
    unlike the log-quadratic fit. Widths are capped at each cell's equivalent
    radius: a hole cannot be wider than its cell, and the cap stops one bad fit
    (e.g. next to an undetected hole) from spreading through its neighbours.
    Returns (params, status).
    """
    n, h, w = data.shape
    params = init.astype(np.float64).copy()
    status = np.full(n, "ok", dtype="U32")
    if n == 0:
        return params, status
    yy, xx = np.indices((h, w), dtype=float)
    xx, yy = xx.ravel(), yy.ravel()
    sigma_cap = np.maximum(1.0, np.sqrt(masks.reshape(n, -1).sum(axis=1) / np.pi))
    chunk = max(1, _CHUNK_ELEMENTS // (h * w * 6))
    for start in range(0, n, chunk):
        sl = slice(start, start + chunk)
        d = data[sl].reshape(-1, h * w)
        wt = masks[sl].reshape(-1, h * w).astype(float)
        p = params[sl]
        lam = np.full(len(p), 1e-3)
        cap = sigma_cap[sl]
        # A single Gaussian cannot exceed the cell's data much; unbounded
        # amplitudes (with minimal width) would subtract spikes from neighbours.
        amp_cap = 3 * np.maximum(np.abs(np.where(wt > 0, d, 0)).max(axis=1), 1e-12)

        def cost_of(q, rows):
            residual = d[rows] - _gaussian(q, xx[None] - q[:, 1, None], yy[None] - q[:, 2, None])[0]
            return (wt[rows] * residual * residual).sum(axis=1)

        cost = cost_of(p, slice(None))
        active = np.isfinite(cost)
        for _ in range(max_iter):
            if not active.any():
                break
            q = p[active]
            dx, dy = xx[None] - q[:, 1, None], yy[None] - q[:, 2, None]
            model, shape, a, b, e1, u2, e3 = _gaussian(q, dx, dy)
            half = -0.5 * model
            jac = np.stack((
                shape,
                half * (-2 * a * e1),
                half * (-2 * a * u2 - 2 * b * e3),
                half * (2 * a * e1 * dx),
                half * (2 * a * dy),
                half * (2 * b * e3 * dy),
            ), axis=2)
            residual = d[active] - model
            jw = jac * wt[active][:, :, None]
            normal = np.einsum("nki,nkj->nij", jw, jac)
            gradient = np.einsum("nki,nk->ni", jw, residual)
            diag = np.einsum("nii->ni", normal)
            damped = normal + (lam[active][:, None] * diag + 1e-12)[:, :, None] * np.eye(6)
            try:
                step = np.linalg.solve(damped, gradient[:, :, None])[:, :, 0]
            except np.linalg.LinAlgError:
                step = np.stack([np.linalg.lstsq(m, g, rcond=None)[0] for m, g in zip(damped, gradient)])
            trial = q + step
            # Keep widths inside [_SIGMA_MIN, cell radius] so diverging fits stay bounded.
            low = -np.log(cap[active])
            trial[:, 0] = np.clip(trial[:, 0], 0.0, amp_cap[active])
            trial[:, 3] = np.clip(trial[:, 3], low, -np.log(_SIGMA_MIN))
            trial[:, 5] = np.clip(trial[:, 5], low, -np.log(_SIGMA_MIN))
            new_cost = cost_of(trial, active)
            better = np.isfinite(new_cost) & (new_cost <= cost[active])
            idx = np.flatnonzero(active)
            p[idx[better]] = trial[better]
            lam[idx[better]] /= 3
            lam[idx[~better]] *= 4
            small = np.abs(step).max(axis=1) < 1e-7
            done = (better & (cost[active] - new_cost <= 1e-12 * np.maximum(cost[active], 1e-300))) | small
            cost[idx[better]] = new_cost[better]
            active[idx[done | (lam[idx] > 1e12)]] = False
        params[sl] = p
        st = status[sl]
        sig = sigma_major(p)
        bad = ~np.isfinite(p).all(axis=1)
        st[bad] = "fit_diverged"
        st[~bad & ((p[:, 0] <= 0) | (sig >= 0.999 * cap))] = "non_gaussian_fit"
        ci = np.round(p[:, 2]).astype(int).clip(0, h - 1), np.round(p[:, 1]).astype(int).clip(0, w - 1)
        inside = (p[:, 1] >= -0.5) & (p[:, 1] <= w - 0.5) & (p[:, 2] >= -0.5) & (p[:, 2] <= h - 0.5)
        inside &= masks[sl][np.arange(len(p)), ci[0], ci[1]]
        st[(st == "ok") & ~inside] = "center_outside_cell"
    return params, status


def _tile_medians(values: np.ndarray, usable: np.ndarray, tile: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-tile median of usable pixels (NaN where none) and usable counts."""
    height, width = values.shape
    ny, nx = -(-height // tile), -(-width // tile)
    padded = np.full((ny * tile, nx * tile), np.nan)
    padded[:height, :width] = np.where(usable, values, np.nan)
    blocks = padded.reshape(ny, tile, nx, tile).transpose(0, 2, 1, 3).reshape(ny, nx, tile * tile)
    counts = np.isfinite(blocks).sum(axis=2)
    medians = np.full((ny, nx), np.nan)
    has = counts > 0
    medians[has] = np.nanmedian(blocks[has], axis=1)
    return medians, counts


def _upsample_tiles(medians: np.ndarray, tile: int, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear interpolation between tile centers, clamped at the image edges."""
    height, width = shape
    ny, nx = medians.shape
    yc, xc = _tile_centers(tile, shape)
    rows = np.stack([np.interp(np.arange(width), xc, medians[j]) for j in range(ny)])
    if ny == 1:
        return np.repeat(rows, height, axis=0)
    y = np.arange(height, dtype=float)
    j1 = np.clip(np.searchsorted(yc, y), 1, ny - 1)
    t = np.clip((y - yc[j1 - 1]) / (yc[j1] - yc[j1 - 1]), 0, 1)[:, None]
    return (1 - t) * rows[j1 - 1] + t * rows[j1]


def _tile_centers(tile: int, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    yc = np.array([(j * tile + min((j + 1) * tile, height) - 1) / 2 for j in range(-(-height // tile))])
    xc = np.array([(i * tile + min((i + 1) * tile, width) - 1) / 2 for i in range(-(-width // tile))])
    return yc, xc


def estimate_background_map(
    residual: np.ndarray, free: np.ndarray, tile: int, *, usable: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Smooth background from ``image - model``, preferring directly visible background.

    Tiles with enough ``free`` pixels (far from every hole, model negligible)
    give the median residual there. Other tiles are interpolated from them:
    inside a dense array the background cannot be separated from overlapping
    tails, because their sum is nearly flat. Only when no tile has free pixels
    is the background inferred from all ``usable`` residual pixels, which
    relies entirely on the Gaussian model.
    """
    usable = np.ones(residual.shape, bool) if usable is None else usable
    free_med, free_count = _tile_medians(residual, free & usable, tile)
    measured = free_count >= max(16, int(0.1 * tile * tile))
    info = {"tiles": int(measured.size), "measured_tiles": int(measured.sum())}
    if measured.all():
        medians, info["source"] = free_med, "measured"
    elif measured.any():
        from scipy.interpolate import griddata

        yc, xc = _tile_centers(tile, residual.shape)
        grid = np.stack(np.meshgrid(yc, xc, indexing="ij"), axis=-1)
        known, values = grid[measured], free_med[measured]
        medians = free_med.copy()
        missing = ~measured
        medians[missing] = np.nan
        if len(values) >= 3:
            try:
                medians[missing] = griddata(known, values, grid[missing], method="linear")
            except Exception:  # collinear measured tiles (e.g. one strip)
                pass
        still = missing & np.isnan(medians)
        medians[still] = griddata(known, values, grid[still], method="nearest")
        info["source"] = "interpolated"
    else:
        medians, _ = _tile_medians(residual, usable, tile)
        info["source"] = "inferred"
        if np.isnan(medians).all():
            return np.full(residual.shape, float(np.median(residual))), info
        medians = np.where(np.isnan(medians), np.nanmedian(medians), medians)
    return _upsample_tiles(medians, tile, residual.shape), info


def auto_tile(seeds_xy: np.ndarray, shape: tuple[int, int]) -> tuple[int, float]:
    """Tile size of about four pitches; pitch is the median nearest-seed distance."""
    if len(seeds_xy) < 2:
        return int(max(shape)), float("nan")
    from scipy.spatial import cKDTree

    pitch = float(np.median(cKDTree(seeds_xy).query(seeds_xy, k=2)[0][:, 1]))
    return int(np.clip(round(4 * pitch), 16, max(shape))), pitch


def deblend(
    image: np.ndarray, seeds_xy: np.ndarray, masks: np.ndarray, origins_xy: np.ndarray, *,
    iterations: int = 30, tile: int | None = None, exclude: np.ndarray | None = None,
    background_level: float | None = None, free_fraction: float = 0.05,
    free_distance_fwhm: float = 1.25, relaxation: float = 0.6, tolerance: float = 1e-3,
    max_iter: int = 30, init: np.ndarray | None = None,
) -> DeblendState:
    """Alternate background-map estimation and neighbour-subtracted Gaussian fits.

    ``image`` is polarity-normalized (holes bright). ``masks``/``origins_xy``
    are the Voronoi cell stack from ``_extract_voronoi_rois``; ``seeds_xy``
    are every modelled hole, including ones not reported later. Background
    pixels must be at least ``free_distance_fwhm`` median FWHMs from every seed
    and below ``free_fraction`` of the median amplitude in the model.
    ``background_level`` fixes a known constant background instead.
    ``init`` (ROI-local params, e.g. from a previous call) skips the
    border-percentile initialization. Iterates until 99 % of centers and log-widths move less than ``tolerance``
    (``iterations`` is the cap); ``iterations`` in the result is the count used.
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt

    image = np.asarray(image, dtype=np.float64)
    shape = image.shape
    seeds_xy = np.asarray(seeds_xy, float).reshape(-1, 2)
    n = len(seeds_xy)
    origins = np.asarray(origins_xy, dtype=np.int64).reshape(-1, 2)
    if tile is None:
        tile = auto_tile(seeds_xy, shape)[0]
    tile = int(tile)
    usable = None if exclude is None else ~exclude
    seed_pixels = np.ones(shape, bool)
    if n:
        iy = np.clip(np.rint(seeds_xy[:, 1]).astype(int), 0, shape[0] - 1)
        ix = np.clip(np.rint(seeds_xy[:, 0]).astype(int), 0, shape[1] - 1)
        seed_pixels[iy, ix] = False
    distance = distance_transform_edt(seed_pixels) if n else np.full(shape, np.inf)

    def background_of(residual, model, p, st):
        if background_level is not None:
            return np.full(shape, float(background_level)), {"source": "fixed"}
        good = (st == "ok") & (p[:, 0] > 0)
        amplitude = float(np.median(p[good, 0])) if good.any() else 1.0
        fwhm = float(np.median(2 * FWHM_K * sigma_major(p[good]))) if good.any() else 0.0
        free = (model < free_fraction * amplitude) & (distance >= free_distance_fwhm * fwhm)
        return estimate_background_map(residual, free, tile, usable=usable)

    if n == 0:
        background, info = background_of(image, np.zeros(shape), np.zeros((0, 6)), np.zeros(0, "U32"))
        return DeblendState(background, np.zeros(shape), np.zeros((0, 6)), np.zeros(0, "U32"),
                            np.zeros(masks.shape), tile, 0, info, True)
    roi_shape = masks.shape[1:]
    raw = windows(image, origins, roi_shape)
    yy, xx = np.indices(roi_shape, dtype=float)
    gx = (origins[:, 0, None, None] + xx[None]).reshape(n, -1)
    gy = (origins[:, 1, None, None] + yy[None]).reshape(n, -1)
    radius_cap = int(max(roi_shape))

    # Initialize from the cell-border background; the iterations replace it.
    if init is None:
        border = masks & ~binary_erosion(masks, structure=np.ones((1, 3, 3)), border_value=0)
        bg0 = np.array([np.percentile(r[b], 10) if b.any() else 0.0 for r, b in zip(raw, border)])
        if background_level is not None:
            bg0[:] = background_level
        data = np.where(masks, raw - bg0[:, None, None], 0.0)
        params, status = fit_gaussians_batch(data, masks, initial_params(data, masks, seeds_xy - origins),
                                             max_iter=max_iter)
    else:
        params, status = np.asarray(init, dtype=np.float64).copy(), np.full(n, "ok", dtype="U32")

    def global_params(p):
        g = p.copy()
        g[:, 1:3] += origins
        return g

    def clean(p, background, model):
        gp = global_params(p)
        own = _render_on_grids(gp, gx, gy, _render_radius(gp, radius_cap)).reshape(raw.shape)
        others = windows(model, origins, roi_shape) - own
        return np.where(masks, raw - windows(background, origins, roi_shape) - others, 0.0)

    model = render_models(global_params(params), shape, radius_cap=radius_cap)
    used, converged = 0, False
    for used in range(1, int(iterations) + 1):
        background, info = background_of(image - model, model, params, status)
        fitted, status = fit_gaussians_batch(clean(params, background, model), masks, params, max_iter=max_iter)
        # All cells update at once against last iteration's neighbours; damping
        # stops the overshoot that strongly overlapping tails otherwise cause.
        step = np.where(np.isfinite(fitted), fitted - params, 0.0)
        params = params + relaxation * step
        model = render_models(global_params(params), shape, radius_cap=radius_cap)
        ok = status == "ok"
        # The 99th percentile: a few noise-dominated holes jitter indefinitely
        # without affecting the rest, and should not hold every cell to the cap.
        if not ok.any() or np.percentile(np.abs(step[ok][:, 1:]).max(axis=1), 99) < tolerance:
            converged = True
            break
    background, info = background_of(image - model, model, params, status)
    return DeblendState(background, model, params, status, clean(params, background, model),
                        tile, used, info, converged)


def revise_seeds(
    signal: np.ndarray, state: DeblendState, seeds_xy: np.ndarray, pitch: float, *,
    drop_fraction: float = 0.3, add_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Seeds to drop and holes to add after one deblending pass.

    Dropped: fits that diverged or whose amplitude is below ``drop_fraction``
    of the median (noise maxima). Added: maxima of the smoothed residual
    ``signal - background - models`` above ``add_fraction`` of the median
    amplitude and more than half a pitch from every kept seed (holes that
    thresholding merged or missed). Returns (drop mask, new seeds_xy).
    """
    from scipy.ndimage import gaussian_filter, maximum_filter
    from scipy.spatial import cKDTree

    p = state.params
    finite = np.isfinite(p).all(axis=1)
    good = finite & (state.status == "ok") & (p[:, 0] > 0)
    if not good.any():
        return np.zeros(len(p), bool), np.zeros((0, 2))
    reference = float(np.median(p[good, 0]))
    drop = ~finite | (p[:, 0] < drop_fraction * reference)
    residual = signal - state.background_map - state.model_image
    smooth = gaussian_filter(residual, max(0.5, 0.1 * pitch), mode="nearest")
    radius = max(1, int(np.floor(0.5 * pitch)))
    peaks = (smooth == maximum_filter(smooth, size=2 * radius + 1, mode="nearest")) & (smooth > add_fraction * reference)
    ys, xs = np.nonzero(peaks)
    new = np.column_stack((xs, ys)).astype(float)
    kept = np.asarray(seeds_xy, float)[~drop]
    if len(new) and len(kept):
        new = new[cKDTree(kept).query(new)[0] > 0.5 * pitch]
    return drop, new


def estimate_pitch(signal: np.ndarray, *, max_size: int = 1024) -> float:
    """Lattice pitch from the first ring of the radially averaged autocorrelation."""
    h, w = signal.shape
    y0, x0 = max(0, (h - max_size) // 2), max(0, (w - max_size) // 2)
    crop = np.asarray(signal[y0:y0 + max_size, x0:x0 + max_size], dtype=np.float64)
    crop = crop - crop.mean()
    spectrum = np.fft.rfft2(crop, s=(2 * crop.shape[0], 2 * crop.shape[1]))
    auto = np.fft.fftshift(np.fft.irfft2(np.abs(spectrum) ** 2, s=(2 * crop.shape[0], 2 * crop.shape[1])))
    cy, cx = np.array(auto.shape) // 2
    yy, xx = np.indices(auto.shape)
    r = np.hypot(yy - cy, xx - cx)
    r_max = int(min(crop.shape) // 2)
    bins = np.rint(r).astype(int)
    keep = bins <= r_max
    profile = np.bincount(bins[keep], weights=auto[keep]) / np.maximum(np.bincount(bins[keep]), 1)
    for k in range(2, len(profile) - 1):
        if profile[k] > profile[k - 1] and profile[k] >= profile[k + 1] and profile[k] > 0:
            denom = profile[k - 1] - 2 * profile[k] + profile[k + 1]
            return float(k + (0.5 * (profile[k - 1] - profile[k + 1]) / denom if denom < 0 else 0.0))
    raise ValueError("could not estimate the hole pitch; pass peak_min_distance explicitly")


def detect_peaks(
    signal: np.ndarray, *, min_distance: float | None = None, smooth_sigma: float | None = None,
    threshold: float = 0.25,
) -> tuple[np.ndarray, float]:
    """Local-maximum hole seeds for arrays too dense to threshold.

    ``min_distance`` defaults to half the autocorrelation pitch. A maximum is kept
    when its prominence above the local minimum (within one pitch) reaches
    ``threshold`` times the 90th-percentile prominence. Returns (seeds_xy, pitch).
    """
    from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

    signal = np.asarray(signal, dtype=np.float64)
    pitch = 2.0 * float(min_distance) if min_distance is not None else estimate_pitch(signal)
    if min_distance is None:
        min_distance = 0.5 * pitch
    if smooth_sigma is None:
        smooth_sigma = max(0.5, 0.1 * pitch)
    smooth = gaussian_filter(signal, smooth_sigma, mode="nearest")
    radius = max(1, int(np.floor(min_distance)))
    is_max = smooth == maximum_filter(smooth, size=2 * radius + 1, mode="nearest")
    prominence = smooth - minimum_filter(smooth, size=2 * int(np.ceil(pitch)) + 1, mode="nearest")
    ys, xs = np.nonzero(is_max & (prominence > 0))
    if len(xs) == 0:
        return np.zeros((0, 2)), pitch
    prom = prominence[ys, xs]
    keep = prom >= threshold * np.percentile(prom, 90)
    ys, xs = ys[keep], xs[keep]
    h, w = signal.shape
    sub = np.column_stack((xs, ys)).astype(float)
    for axis, (dy, dx) in enumerate(((0, 1), (1, 0))):
        ok = (ys - dy >= 0) & (ys + dy < h) & (xs - dx >= 0) & (xs + dx < w)
        left = smooth[np.clip(ys - dy, 0, h - 1), np.clip(xs - dx, 0, w - 1)]
        mid = smooth[ys, xs]
        right = smooth[np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)]
        curvature = left - 2 * mid + right
        delta = np.where(ok & (curvature < 0), 0.5 * (left - right) / np.where(curvature < 0, curvature, -1), 0.0)
        sub[:, axis] += np.clip(delta, -0.5, 0.5)
    return sub, pitch
