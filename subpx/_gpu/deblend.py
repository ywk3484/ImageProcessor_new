"""GPU deblending: ``subpx._cpu.deblend`` with the hot loops as CUDA kernels.

Same functions and NumPy results as the CPU module. Inside ``deblend`` the
image, model, background map and cell stacks stay on the GPU between
iterations; everything that decides statuses, render radii, the background
tiles' interpolation or convergence runs on the host with the CPU code, so
those decisions agree with the CPU path. Arithmetic is FP64 throughout.
"""

from __future__ import annotations

import numpy as np

from .._cpu import deblend as cpu
from ._cuda import REDUCE_SRC, cp, kernel, optin_shared_bytes, require_cupy

_THREADS = 128
# Cleaned cells held on the GPU at once; larger stacks are fitted in chunks.
_CHUNK_BYTES = 512 * 2**20

# One block per cell runs the whole Levenberg-Marquardt loop of
# subpx._cpu.deblend.fit_gaussians_batch: same damping, clipping, acceptance
# and stopping rules; the 6x6 damped system is solved by LU with partial
# pivoting like np.linalg.solve. Cells live in shared memory unless too large.
_LM_SRC = REDUCE_SRC + r"""
__device__ __forceinline__ double clip(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);  // NaN passes through, as np.clip
}
__device__ double cost_part(const double* c, const double* d, const unsigned char* m, int h, int w) {
    const double e1 = exp(c[3]), u2 = c[4], e3 = exp(c[5]);
    double acc = 0;
    for (int k = threadIdx.x; k < h * w; k += blockDim.x) {
        if (!m[k]) continue;
        const double dx = (double)(k % w) - c[1], dy = (double)(k / w) - c[2];
        const double a = e1 * dx + u2 * dy, b = e3 * dy;
        const double r = d[k] - c[0] * exp(-0.5 * (a * a + b * b));
        acc += r * r;
    }
    return acc;
}
extern "C" __global__ void lm_fit(
    const double* __restrict__ data, const unsigned char* __restrict__ mask, double* __restrict__ params,
    const int n, const int h, const int w, const int max_iter, const double sigma_min)
{
    __shared__ double partial[32][NV];
    __shared__ double tot[NV];
    __shared__ double q[6], trial[6];
    __shared__ double s_cost, s_step;
    __shared__ int s_go;
    const int cell = blockIdx.x, tid = threadIdx.x, P = h * w;
    if (cell >= n) return;
#ifdef CELL_IN_SHARED
    extern __shared__ double cell_mem[];
    double* d = cell_mem;
    unsigned char* m = (unsigned char*)(cell_mem + P);
    for (int k = tid; k < P; k += blockDim.x) { d[k] = data[(size_t)cell * P + k]; m[k] = mask[(size_t)cell * P + k]; }
#else
    const double* d = data + (size_t)cell * P;
    const unsigned char* m = mask + (size_t)cell * P;
#endif
    if (tid < 6) q[tid] = params[cell * 6 + tid];
    __syncthreads();
    double v[NV];
    double count = 0, peak = 0;
    for (int k = tid; k < P; k += blockDim.x) if (m[k]) { count += 1; peak = fmax(peak, fabs(d[k])); }
    v[0] = count;
    block_sum(v, 1, tot, partial);
    const double cap = fmax(1.0, sqrt(tot[0] / 3.141592653589793));
    const double amp_cap = 3 * fmax(block_max(peak, partial), 1e-12);
    v[0] = cost_part(q, d, m, h, w);
    block_sum(v, 1, tot, partial);
    double lam = 1e-3;  // only thread 0's copy is used
    if (tid == 0) { s_cost = tot[0]; s_go = isfinite(tot[0]); }
    __syncthreads();
    for (int it = 0; it < max_iter && s_go; ++it) {
        const double A = q[0], x0 = q[1], y0 = q[2], e1 = exp(q[3]), u2 = q[4], e3 = exp(q[5]);
        for (int i = 0; i < 27; ++i) v[i] = 0;
        for (int k = tid; k < P; k += blockDim.x) {
            if (!m[k]) continue;
            const double dx = (double)(k % w) - x0, dy = (double)(k / w) - y0;
            const double a = e1 * dx + u2 * dy, b = e3 * dy;
            const double shape = exp(-0.5 * (a * a + b * b));
            const double model = A * shape, half = -0.5 * model;
            const double J[6] = {shape, half * (-2 * a * e1), half * (-2 * a * u2 - 2 * b * e3),
                                 half * (2 * a * e1 * dx), half * (2 * a * dy), half * (2 * b * e3 * dy)};
            const double r = d[k] - model;
            int t = 0;
            for (int i = 0; i < 6; ++i) for (int j = i; j < 6; ++j) v[t++] += J[i] * J[j];
            for (int i = 0; i < 6; ++i) v[21 + i] += J[i] * r;
        }
        block_sum(v, 27, tot, partial);
        if (tid == 0) {
            double M[6][7];
            int t = 0;
            for (int i = 0; i < 6; ++i) for (int j = i; j < 6; ++j) M[i][j] = M[j][i] = tot[t++];
            for (int i = 0; i < 6; ++i) { M[i][i] += lam * M[i][i] + 1e-12; M[i][6] = tot[21 + i]; }
            bool ok = true;
            for (int c = 0; c < 6 && ok; ++c) {
                int pivot = c;
                for (int r = c + 1; r < 6; ++r) if (fabs(M[r][c]) > fabs(M[pivot][c])) pivot = r;
                if (!(fabs(M[pivot][c]) > 0)) { ok = false; break; }
                if (pivot != c) for (int j = c; j < 7; ++j) { const double s = M[c][j]; M[c][j] = M[pivot][j]; M[pivot][j] = s; }
                for (int r = c + 1; r < 6; ++r) {
                    const double f = M[r][c] / M[c][c];
                    for (int j = c; j < 7; ++j) M[r][j] -= f * M[c][j];
                }
            }
            double step[6] = {0, 0, 0, 0, 0, 0};
            if (ok) {
                for (int i = 5; i >= 0; --i) {
                    double s = M[i][6];
                    for (int j = i + 1; j < 6; ++j) s -= M[i][j] * step[j];
                    step[i] = s / M[i][i];
                }
            }
            double largest = 0;
            for (int i = 0; i < 6; ++i) {
                trial[i] = q[i] + step[i];
                if (!(fabs(step[i]) <= largest)) largest = fabs(step[i]);  // NaN-propagating max
            }
            const double low = -log(cap), high = -log(sigma_min);
            trial[0] = clip(trial[0], 0.0, amp_cap);
            trial[3] = clip(trial[3], low, high);
            trial[5] = clip(trial[5], low, high);
            s_step = largest;
        }
        __syncthreads();
        v[0] = cost_part(trial, d, m, h, w);
        block_sum(v, 1, tot, partial);
        if (tid == 0) {
            const double new_cost = tot[0], cost = s_cost;
            const bool better = isfinite(new_cost) && new_cost <= cost;
            if (better) { for (int i = 0; i < 6; ++i) q[i] = trial[i]; lam /= 3; } else { lam *= 4; }
            const bool done = (better && cost - new_cost <= 1e-12 * fmax(cost, 1e-300)) || s_step < 1e-7;
            if (better) s_cost = new_cost;
            s_go = !(done || lam > 1e12);
        }
        __syncthreads();
    }
    if (tid < 6) params[cell * 6 + tid] = q[tid];
}
"""

# One block per hole adds its truncated Gaussian (render_models) into a fixed-
# point image: integer atomics are order-independent, so the sum is identical
# run to run, unlike floating-point atomics.
_RENDER_SRC = r"""
extern "C" __global__ void render_fixed(
    const double* __restrict__ p, const int* __restrict__ radius, unsigned long long* __restrict__ out,
    const int n, const int H, const int W, const double scale)
{
    const int hole = blockIdx.x;
    if (hole >= n) return;
    const double A = p[hole * 6], x0 = p[hole * 6 + 1], y0 = p[hole * 6 + 2];
    const double e1 = exp(p[hole * 6 + 3]), u2 = p[hole * 6 + 4], e3 = exp(p[hole * 6 + 5]);
    const int r = radius[hole], side = 2 * r + 1;
    const long long bx = (long long)floor(x0), by = (long long)floor(y0);
    for (int k = threadIdx.x; k < side * side; k += blockDim.x) {
        const long long ix = bx - r + k % side, iy = by - r + k / side;
        if (ix < 0 || ix >= W || iy < 0 || iy >= H) continue;
        const double dx = (double)ix - x0, dy = (double)iy - y0;
        if (fabs(dx) > r || fabs(dy) > r) continue;
        const double a = e1 * dx + u2 * dy, b = e3 * dy;
        const double value = A * exp(-0.5 * (a * a + b * b));
        atomicAdd(out + iy * W + ix, (unsigned long long)llrint(value * scale));
    }
}
"""

# clean_cells: each cell minus the background map and all other holes' models.
_CLEAN_SRC = r"""
extern "C" __global__ void clean_cells(
    const double* __restrict__ image, const double* __restrict__ background, const double* __restrict__ model,
    const double* __restrict__ p, const int* __restrict__ radius, const unsigned char* __restrict__ mask,
    const long long* __restrict__ origins, double* __restrict__ out,
    const long long total, const int h, const int w, const int H, const int W)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    if (!mask[t]) { out[t] = 0.0; return; }
    const long long cell = t / (h * w);
    const int k = (int)(t % (h * w));
    const long long gx = origins[2 * cell] + k % w, gy = origins[2 * cell + 1] + k / w;
    double raw = 0, bg = 0, mdl = 0;
    if (gx >= 0 && gx < W && gy >= 0 && gy < H) {
        const long long i = gy * W + gx;
        raw = image[i]; bg = background[i]; mdl = model[i];
    }
    const double* q = p + 6 * cell;
    const double dx = (double)gx - q[1], dy = (double)gy - q[2];
    double own = 0;
    if (fabs(dx) <= radius[cell] && fabs(dy) <= radius[cell]) {
        const double e1 = exp(q[3]), u2 = q[4], e3 = exp(q[5]);
        const double a = e1 * dx + u2 * dy, b = e3 * dy;
        own = q[0] * exp(-0.5 * (a * a + b * b));
    }
    out[t] = raw - bg - (mdl - own);
}
"""


def _cell_chunks(n: int, roi_shape) -> list[slice]:
    size = max(1, _CHUNK_BYTES // (8 * int(np.prod(roi_shape))))
    return [slice(start, start + size) for start in range(0, n, size)]


def _fit_device(data, masks, init: np.ndarray, *, max_iter: int) -> np.ndarray:
    """LM params for device cell stacks (N, h, w) data/uint8 masks from host ``init``."""
    n, h, w = data.shape
    params = cp.asarray(init, dtype=cp.float64)
    if n == 0:
        return cp.asnumpy(params)
    shared = h * w * 9
    budget = optin_shared_bytes()
    if shared + 8 * 1024 <= budget:  # leave room for the static reduction buffers
        fit = kernel(_LM_SRC, "lm_fit", defines=("CELL_IN_SHARED",))
        if shared > 48 * 1024:
            fit.max_dynamic_shared_size_bytes = shared
    else:
        fit, shared = kernel(_LM_SRC, "lm_fit"), 0
    fit((n,), (_THREADS,), (cp.ascontiguousarray(data), cp.ascontiguousarray(masks), params, np.int32(n),
                            np.int32(h), np.int32(w), np.int32(max_iter), np.float64(cpu._SIGMA_MIN)),
        shared_mem=shared)
    return cp.asnumpy(params)


def fit_gaussians_batch(
    data: np.ndarray, masks: np.ndarray, init: np.ndarray, *, max_iter: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU ``subpx._cpu.deblend.fit_gaussians_batch``: same caps, statuses and NumPy results."""
    require_cupy()
    masks = np.asarray(masks, dtype=bool)
    n = len(masks)
    params = np.asarray(init, dtype=np.float64).copy()
    status = np.full(n, "ok", dtype="U32")
    if n == 0:
        return params, status
    for sl in _cell_chunks(n, masks.shape[1:]):
        params[sl] = _fit_device(cp.asarray(data[sl], dtype=cp.float64), cp.asarray(masks[sl], dtype=cp.uint8),
                                 params[sl], max_iter=max_iter)
    cpu.fit_status(params, masks, cpu.sigma_caps(masks), status)
    return params, status


def _render_device(params_global: np.ndarray, shape, radius_cap: int):
    """render_models on the GPU; returns the (H, W) float64 device image."""
    height, width = shape
    n = len(params_global)
    if n == 0:
        return cp.zeros(shape)
    radius = cpu._render_radius(params_global, radius_cap).astype(np.int32)
    # Every pixel's sum is at most n * max amplitude: a power-of-two scale with
    # that headroom in 62 bits is exact to apply and cannot overflow.
    top = float(np.max(np.abs(params_global[:, 0]))) * n
    scale = 2.0 ** int(np.clip(62 - np.ceil(np.log2(top)), -1000, 1000)) if top > 0 else 1.0
    acc = cp.zeros(height * width, dtype=cp.uint64)
    kernel(_RENDER_SRC, "render_fixed")(
        (n,), (_THREADS,), (cp.asarray(params_global, dtype=cp.float64), cp.asarray(radius), acc, np.int32(n),
                            np.int32(height), np.int32(width), np.float64(scale)))
    return (acc.view(cp.int64).astype(cp.float64) / scale).reshape(shape)


def render_models(params_global: np.ndarray, shape: tuple[int, int], *, radius_cap: int = 64) -> np.ndarray:
    """GPU ``render_models``; equals the CPU sum to its fixed-point resolution, ~1e-12 of the largest amplitude."""
    require_cupy()
    return cp.asnumpy(_render_device(np.asarray(params_global, dtype=np.float64), shape, radius_cap))


def _windows(image, origins, shape):
    """``windows`` on device arrays."""
    height, width = image.shape
    ys = origins[:, 1, None] + cp.arange(shape[0])[None, :]
    xs = origins[:, 0, None] + cp.arange(shape[1])[None, :]
    inside = ((ys >= 0) & (ys < height))[:, :, None] & ((xs >= 0) & (xs < width))[:, None, :]
    out = image[cp.clip(ys, 0, height - 1)[:, :, None], cp.clip(xs, 0, width - 1)[:, None, :]]
    return cp.where(inside, out, 0.0)


def border_percentile(values, border, q: float = 10.0):
    """Per-cell ``np.percentile(values[border], q)`` (0 where no border pixel), bit for bit."""
    n = len(values)
    flat = cp.sort(cp.where(border, values, cp.inf).reshape(n, -1), axis=1)
    count = border.reshape(n, -1).sum(axis=1)
    # NumPy's 'linear' method: virtual index (n - 1) * q, then _lerp.
    virtual = (count - 1).astype(cp.float64) * (q / 100)
    below = cp.floor(virtual)
    rows = cp.arange(n)
    low = cp.clip(below.astype(cp.int64), 0, flat.shape[1] - 1)
    a, b = flat[rows, low], flat[rows, cp.minimum(low + 1, flat.shape[1] - 1)]
    gamma = virtual - below
    diff = b - a
    result = cp.where(gamma >= 0.5, b - diff * (1 - gamma), a + diff * gamma)
    result = cp.where(virtual >= count - 1, flat[rows, cp.clip(count - 1, 0, None)], result)
    return cp.where(count > 0, result, 0.0)


def _initial_params(data, masks, seeds_local):
    """``initial_params`` on device arrays."""
    n = len(data)
    params = cp.zeros((n, 6))
    peak = cp.where(masks, data, -cp.inf).reshape(n, -1).max(axis=1)
    weight = cp.where(masks & (data > 0.2 * peak[:, None, None]), data, 0.0)
    yy, xx = cp.indices(data.shape[1:], dtype=cp.float64)
    total = cp.maximum(weight.sum(axis=(1, 2)), 1e-300)
    dx = xx[None] - seeds_local[:, 0, None, None]
    dy = yy[None] - seeds_local[:, 1, None, None]
    var = ((weight * (dx * dx + dy * dy)).sum(axis=(1, 2)) / total) / 2
    sigma = cp.sqrt(cp.clip(var, 0.25, None))
    params[:, 0] = cp.maximum(peak, 1e-12)
    params[:, 1:3] = seeds_local
    params[:, 3] = params[:, 5] = -cp.log(sigma)
    return params


def _tile_medians(values, usable, tile: int):
    """``_tile_medians`` of a device image, returned as host (medians, counts); exactly NumPy's medians."""
    height, width = values.shape
    ny, nx = -(-height // tile), -(-width // tile)
    padded = cp.full((ny * tile, nx * tile), cp.inf)
    padded[:height, :width] = cp.where(usable, values, cp.inf)
    blocks = cp.sort(padded.reshape(ny, tile, nx, tile).transpose(0, 2, 1, 3).reshape(ny * nx, tile * tile), axis=1)
    counts = cp.isfinite(blocks).sum(axis=1)
    rows = cp.arange(ny * nx)
    low, high = blocks[rows, cp.maximum(counts - 1, 0) // 2], blocks[rows, counts // 2]
    medians = cp.where(counts > 0, (low + high) / 2, cp.nan)  # np.median: mean of the middle pair
    return cp.asnumpy(medians).reshape(ny, nx), cp.asnumpy(counts).reshape(ny, nx)


def _background_device(residual, free, tile: int, usable):
    """``estimate_background_map`` on device arrays: (device map, info)."""
    shape = residual.shape
    medians, info = cpu.background_tiles(lambda mask: _tile_medians(residual, mask, tile), free, usable, tile, shape)
    if medians is None:
        return cp.full(shape, float(np.median(cp.asnumpy(residual)))), info
    rows, j1, t = cpu._upsample_rows(medians, tile, shape)
    rows = cp.asarray(rows)
    if j1 is None:
        return cp.repeat(rows, shape[0], axis=0), info
    # The same separately rounded operations as _upsample_tiles.
    t, j1 = cp.asarray(t), cp.asarray(j1)
    return (1 - t) * rows[j1 - 1] + t * rows[j1], info


def estimate_background_map(residual, free, tile: int, *, usable=None) -> tuple[np.ndarray, dict]:
    """GPU ``estimate_background_map`` (inputs NumPy or CuPy); the map equals the CPU's bit for bit."""
    require_cupy()
    residual = cp.asarray(residual, dtype=cp.float64)
    usable = cp.ones(residual.shape, bool) if usable is None else cp.asarray(usable)
    background, info = _background_device(residual, cp.asarray(free), int(tile), usable)
    return cp.asnumpy(background), info


def deblend(
    image: np.ndarray, seeds_xy: np.ndarray, masks: np.ndarray, origins_xy: np.ndarray, *,
    iterations: int = 30, tile: int | None = None, exclude: np.ndarray | None = None,
    background_level: float | None = None, free_fraction: float = 0.05,
    free_distance_fwhm: float = 1.25, relaxation: float = 0.6, tolerance: float = 1e-3,
    max_iter: int = 30, init: np.ndarray | None = None,
) -> cpu.DeblendState:
    """GPU ``subpx._cpu.deblend.deblend``: same arguments, iteration rules and NumPy state."""
    from scipy.ndimage import distance_transform_edt
    from cupyx.scipy.ndimage import binary_erosion

    require_cupy()
    image = np.asarray(image, dtype=np.float64)
    shape = image.shape
    seeds_xy = np.asarray(seeds_xy, float).reshape(-1, 2)
    n = len(seeds_xy)
    masks = np.asarray(masks, dtype=bool)
    if n == 0:
        return cpu.deblend(image, seeds_xy, masks, origins_xy, iterations=iterations, tile=tile, exclude=exclude,
                           background_level=background_level, free_fraction=free_fraction,
                           free_distance_fwhm=free_distance_fwhm, relaxation=relaxation, tolerance=tolerance,
                           max_iter=max_iter, init=init)
    origins = np.asarray(origins_xy, dtype=np.int64).reshape(-1, 2)
    if tile is None:
        tile = cpu.auto_tile(seeds_xy, shape)[0]
    tile = int(tile)
    seed_pixels = np.ones(shape, bool)
    seed_pixels[np.clip(np.rint(seeds_xy[:, 1]).astype(int), 0, shape[0] - 1),
                np.clip(np.rint(seeds_xy[:, 0]).astype(int), 0, shape[1] - 1)] = False
    roi_shape = masks.shape[1:]
    radius_cap = int(max(roi_shape))
    chunks = _cell_chunks(n, roi_shape)

    image_d = cp.asarray(image)
    distance_d = cp.asarray(distance_transform_edt(seed_pixels))
    usable_d = cp.ones(shape, bool) if exclude is None else cp.asarray(~np.asarray(exclude, bool))
    masks_d = cp.asarray(masks, dtype=cp.uint8)
    origins_d = cp.asarray(origins)
    caps = cpu.sigma_caps(masks)

    def global_params(p, sl=slice(None)):
        g = p.copy()
        g[:, 1:3] += origins[sl]
        return g

    def background_of(model_d, p, st):
        if background_level is not None:
            return cp.full(shape, float(background_level)), {"source": "fixed"}
        good = (st == "ok") & (p[:, 0] > 0)
        amplitude = float(np.median(p[good, 0])) if good.any() else 1.0
        fwhm = float(np.median(2 * cpu.FWHM_K * cpu.sigma_major(p[good]))) if good.any() else 0.0
        free = (model_d < free_fraction * amplitude) & (distance_d >= free_distance_fwhm * fwhm)
        return _background_device(image_d - model_d, free, tile, usable_d)

    def clean(p, background_d, model_d, sl):
        gp = global_params(p[sl], sl)
        radius = cpu._render_radius(gp, radius_cap).astype(np.int32)
        cells = masks_d[sl]
        out = cp.empty(cells.shape, dtype=cp.float64)
        total = out.size
        kernel(_CLEAN_SRC, "clean_cells")(
            ((total + 255) // 256,), (256,),
            (image_d, background_d, model_d, cp.asarray(gp), cp.asarray(radius), cells, origins_d[sl], out,
             np.int64(total), np.int32(roi_shape[0]), np.int32(roi_shape[1]), np.int32(shape[0]), np.int32(shape[1])))
        return out

    def fit(data_of, p):
        fitted = p.copy()
        for sl in chunks:
            fitted[sl] = _fit_device(data_of(sl), masks_d[sl], p[sl], max_iter=max_iter)
        status = np.full(n, "ok", dtype="U32")
        cpu.fit_status(fitted, masks, caps, status)
        return fitted, status

    if init is None:
        # Initialize from the cell-border background; the iterations replace it.
        params = np.empty((n, 6))
        for sl in chunks:
            cells = masks_d[sl].astype(bool)
            border = cells & ~binary_erosion(cells, structure=cp.ones((1, 3, 3), bool), border_value=0)
            raw = _windows(image_d, origins_d[sl], roi_shape)
            bg0 = border_percentile(raw, border)
            if background_level is not None:
                bg0[:] = background_level
            data = cp.where(cells, raw - bg0[:, None, None], 0.0)
            start = cp.asnumpy(_initial_params(data, cells, cp.asarray(seeds_xy[sl] - origins[sl])))
            params[sl] = _fit_device(data, masks_d[sl], start, max_iter=max_iter)
        status = np.full(n, "ok", dtype="U32")
        cpu.fit_status(params, masks, caps, status)
    else:
        params, status = np.asarray(init, dtype=np.float64).copy(), np.full(n, "ok", dtype="U32")

    model_d = _render_device(global_params(params), shape, radius_cap)
    used, converged = 0, False
    for used in range(1, int(iterations) + 1):
        background_d, info = background_of(model_d, params, status)
        fitted, status = fit(lambda sl: clean(params, background_d, model_d, sl), params)
        # All cells update at once against last iteration's neighbours, damped (see the CPU module).
        step = np.where(np.isfinite(fitted), fitted - params, 0.0)
        params = params + relaxation * step
        model_d = _render_device(global_params(params), shape, radius_cap)
        ok = status == "ok"
        if not ok.any() or np.percentile(np.abs(step[ok][:, 1:]).max(axis=1), 99) < tolerance:
            converged = True
            break
    background_d, info = background_of(model_d, params, status)
    cleaned = np.empty(masks.shape)
    for sl in chunks:
        cleaned[sl] = cp.asnumpy(clean(params, background_d, model_d, sl))
    return cpu.DeblendState(cp.asnumpy(background_d), cp.asnumpy(model_d), params, status, cleaned,
                            tile, used, info, converged)
