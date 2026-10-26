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
from ._cuda import cp, kernel, optin_shared_bytes, require_cupy

_THREADS = 128
_WARPS = 4  # cells per block of the LM fit (one warp each)
# Cleaned cells held on the GPU at once; larger stacks are fitted in chunks.
_CHUNK_BYTES = 512 * 2**20

# One warp per cell (_WARPS cells per block) runs the whole Levenberg-Marquardt
# loop of subpx._cpu.deblend.fit_gaussians_batch: same damping, clipping,
# acceptance and stopping rules; the 6x6 damped system is solved by LU with
# partial pivoting like np.linalg.solve. Sums are butterfly warp reductions, so
# every lane holds the same totals and takes the same decisions (no block
# barriers). Cells live in shared memory unless too large.
_LM_SRC = r"""
__device__ __forceinline__ double clip(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);  // NaN passes through, as np.clip
}
__device__ __forceinline__ double warp_total(double x) {
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}
__device__ __forceinline__ double warp_largest(double x) {
    for (int o = 16; o > 0; o >>= 1) x = fmax(x, __shfl_xor_sync(0xffffffffu, x, o));
    return x;
}
__device__ double cost(const double* c, const double* d, const unsigned char* m, int P, int w, int lane) {
    const double e1 = exp(c[3]), u2 = c[4], e3 = exp(c[5]);
    double acc = 0;
    for (int k = lane; k < P; k += 32) {
        if (!m[k]) continue;
        const double dx = (double)(k % w) - c[1], dy = (double)(k / w) - c[2];
        const double a = e1 * dx + u2 * dy, b = e3 * dy;
        const double r = d[k] - c[0] * exp(-0.5 * (a * a + b * b));
        acc += r * r;
    }
    return warp_total(acc);
}
extern "C" __global__ void lm_fit(
    const double* __restrict__ data, const unsigned char* __restrict__ mask, double* __restrict__ params,
    const int n, const int h, const int w, const int max_iter, const double sigma_min)
{
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const long long cell = (long long)blockIdx.x * (blockDim.x >> 5) + warp;
    if (cell >= n) return;  // the whole warp
    const int P = h * w;
#ifdef CELL_IN_SHARED
    extern __shared__ double cell_mem[];
    double* d = cell_mem + (size_t)warp * P;
    unsigned char* m = (unsigned char*)(cell_mem + (size_t)(blockDim.x >> 5) * P) + (size_t)warp * P;
    for (int k = lane; k < P; k += 32) { d[k] = data[cell * P + k]; m[k] = mask[cell * P + k]; }
    __syncwarp();
#else
    const double* d = data + cell * P;
    const unsigned char* m = mask + cell * P;
#endif
    double q[6], trial[6];
    for (int i = 0; i < 6; ++i) q[i] = params[cell * 6 + i];
    double count = 0, peak = 0;
    for (int k = lane; k < P; k += 32) if (m[k]) { count += 1; peak = fmax(peak, fabs(d[k])); }
    const double cap = fmax(1.0, sqrt(warp_total(count) / 3.141592653589793));
    const double amp_cap = 3 * fmax(warp_largest(peak), 1e-12);
    double s_cost = cost(q, d, m, P, w, lane);
    double lam = 1e-3;
    bool go = isfinite(s_cost);
    for (int it = 0; it < max_iter && go; ++it) {
        const double A = q[0], x0 = q[1], y0 = q[2], e1 = exp(q[3]), u2 = q[4], e3 = exp(q[5]);
        double v[27];
        for (int i = 0; i < 27; ++i) v[i] = 0;
        for (int k = lane; k < P; k += 32) {
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
        for (int i = 0; i < 27; ++i) v[i] = warp_total(v[i]);
        double M[6][7];
        int t = 0;
        for (int i = 0; i < 6; ++i) for (int j = i; j < 6; ++j) M[i][j] = M[j][i] = v[t++];
        for (int i = 0; i < 6; ++i) { M[i][i] += lam * M[i][i] + 1e-12; M[i][6] = v[21 + i]; }
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
        const double new_cost = cost(trial, d, m, P, w, lane);
        const bool better = isfinite(new_cost) && new_cost <= s_cost;
        if (better) { for (int i = 0; i < 6; ++i) q[i] = trial[i]; lam /= 3; } else { lam *= 4; }
        const bool done = (better && s_cost - new_cost <= 1e-12 * fmax(s_cost, 1e-300)) || largest < 1e-7;
        if (better) s_cost = new_cost;
        go = !(done || lam > 1e12);
    }
    if (lane < 6) params[cell * 6 + lane] = q[lane];
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


def _sigma_major(params: np.ndarray) -> np.ndarray:
    """``cpu.sigma_major`` in closed form, equal to rounding: the precision L L^T has determinant (e^u1 e^u3)^2, so
    its smaller eigenvalue is that divided by the larger one (no cancellation)."""
    e1, u2, e3 = np.exp(params[:, 3]), params[:, 4], np.exp(params[:, 5])
    a, b, c = e1 * e1, e1 * u2, u2 * u2 + e3 * e3
    half = (a - c) / 2
    larger = (a + c) / 2 + np.sqrt(half * half + b * b)
    with np.errstate(divide="ignore", invalid="ignore"):
        smaller = np.where(larger > 0, (e1 * e3) ** 2 / larger, 0.0)
    return 1 / np.sqrt(np.maximum(smaller, 1e-300))


def _cell_parts(masks) -> list[tuple[np.ndarray, int, int]]:
    """The cells grouped by the top-left sub-block of the stack that holds their mask (rows, height, width), split
    so that one part's cleaned cells take at most _CHUNK_BYTES: everything outside a mask is zero or False, so a
    cell's fit and cleaned pixels are those of the whole stack, at the cost of its own size."""
    from .._cpu.invariant_cd import _size_buckets

    parts = []
    for rows, h, w in _size_buckets(masks):
        size = max(1, _CHUNK_BYTES // (8 * h * w))
        parts.extend((rows[start:start + size], h, w) for start in range(0, len(rows), size))
    return parts


def _fit_device(data, masks, init: np.ndarray, *, max_iter: int) -> np.ndarray:
    """LM params for device cell stacks (N, h, w) data/uint8 masks from host ``init``."""
    n, h, w = data.shape
    params = cp.asarray(init, dtype=cp.float64)
    if n == 0:
        return cp.asnumpy(params)
    shared = _WARPS * h * w * 9
    if shared <= min(optin_shared_bytes(), 96 * 1024):
        fit = kernel(_LM_SRC, "lm_fit", defines=("CELL_IN_SHARED",))
        if shared > 48 * 1024:
            fit.max_dynamic_shared_size_bytes = shared
    else:
        fit, shared = kernel(_LM_SRC, "lm_fit"), 0
    fit((-(-n // _WARPS),), (32 * _WARPS,), (cp.ascontiguousarray(data), cp.ascontiguousarray(masks), params,
                                             np.int32(n), np.int32(h), np.int32(w), np.int32(max_iter),
                                             np.float64(cpu._SIGMA_MIN)), shared_mem=shared)
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


def _render_device(params_global: np.ndarray, shape, radius_cap: int, radius: np.ndarray | None = None):
    """render_models on the GPU; returns the (H, W) float64 device image. ``radius``: the render radii when already
    known (``cpu._render_radius(params_global, radius_cap)``)."""
    height, width = shape
    n = len(params_global)
    if n == 0:
        return cp.zeros(shape)
    if radius is None:
        radius = cpu._render_radius(params_global, radius_cap)
    radius = radius.astype(np.int32)
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


# One block per background tile: its usable values (inf elsewhere and in the padding) sorted in shared memory, then
# the median of the finite ones as np.median: the mean of the middle pair. The residuals are finite.
_TILE_SRC = r"""
extern "C" __global__ void tile_medians(
    const double* __restrict__ values, const bool* __restrict__ usable, const int height, const int width,
    const int tile, const int nx, const int L, double* __restrict__ medians, long long* __restrict__ counts)
{
    extern __shared__ double keys[];
    __shared__ int s_count;
    const double INF = __longlong_as_double(0x7FF0000000000000ll);
    const int ty = blockIdx.x / nx, tx = blockIdx.x % nx;
    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        double v = INF;
        if (i < tile * tile) {
            const int y = ty * tile + i / tile, x = tx * tile + i % tile;
            if (y < height && x < width && usable[(long long)y * width + x]) v = values[(long long)y * width + x];
        }
        keys[i] = v;
        if (isfinite(v)) atomicAdd(&s_count, 1);
    }
    __syncthreads();
    for (int len = 2; len <= L; len <<= 1) {
        for (int stride = len >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < L; i += blockDim.x) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool up = (i & len) == 0;
                    const double u = keys[i], v = keys[j];
                    if (up ? u > v : u < v) { keys[i] = v; keys[j] = u; }
                }
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) {
        const int c = s_count;
        counts[blockIdx.x] = c;
        medians[blockIdx.x] = c > 0 ? (keys[(c - 1) / 2] + keys[c / 2]) / 2 : __longlong_as_double(0x7FF8000000000000ll);
    }
}
"""


def _tile_medians(values, usable, tile: int):
    """``_tile_medians`` of a device image, returned as host (medians, counts); exactly NumPy's medians."""
    height, width = values.shape
    ny, nx = -(-height // tile), -(-width // tile)
    length = 1 << max(0, (tile * tile - 1).bit_length())
    if 8 * length <= 48 * 1024:
        medians, counts = cp.empty(ny * nx, dtype=cp.float64), cp.empty(ny * nx, dtype=cp.int64)
        kernel(_TILE_SRC, "tile_medians")(
            (ny * nx,), (256,), (cp.ascontiguousarray(values, dtype=cp.float64), cp.ascontiguousarray(usable, dtype=bool),
                                 np.int32(height), np.int32(width), np.int32(tile), np.int32(nx), np.int32(length),
                                 medians, counts), shared_mem=8 * length)
        return cp.asnumpy(medians).reshape(ny, nx), cp.asnumpy(counts).reshape(ny, nx)
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
    """GPU ``subpx._cpu.deblend.deblend``: same arguments, iteration rules and NumPy state. Every cell is fitted and
    cleaned in the top-left sub-block of the stack that holds its mask; the render radius cap stays the stack's."""
    from cupyx.scipy.ndimage import binary_erosion, distance_transform_edt

    require_cupy()
    image = np.asarray(image, dtype=np.float64)
    shape = image.shape
    seeds_xy = np.asarray(seeds_xy, float).reshape(-1, 2)
    n = len(seeds_xy)
    cpu._validate_iteration_options(n, iterations, tolerance, relaxation, max_iter, init)
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
    parts = _cell_parts(masks)

    image_d = cp.asarray(image)
    distance_d = distance_transform_edt(cp.asarray(seed_pixels))  # the same distances as SciPy's
    usable_d = cp.ones(shape, bool) if exclude is None else cp.asarray(~np.asarray(exclude, bool))
    part_masks = [cp.asarray(masks[:, :h, :w][rows], dtype=cp.uint8) for rows, h, w in parts]
    part_origins = [cp.asarray(origins[rows]) for rows, _, _ in parts]
    pixels = np.empty(n)
    for part, (rows, _, _) in enumerate(parts):
        pixels[rows] = cp.asnumpy(part_masks[part].reshape(len(rows), -1).sum(axis=1))
    caps = np.maximum(1.0, np.sqrt(pixels / np.pi))  # cpu.sigma_caps(masks)
    known = {}  # sigma_major of the current parameters: render radii and the background's FWHM

    def sigmas(p):
        if known.get("params") is not p:
            known.update(params=p, sigma=_sigma_major(p))
        return known["sigma"]

    def radii(p):  # cpu._render_radius(p, radius_cap)
        return np.clip(np.ceil(cpu.RENDER_SIGMAS * sigmas(p)), 2, radius_cap)

    def global_params(p, rows=slice(None)):
        g = p[rows].copy()
        g[:, 1:3] += origins[rows]
        return g

    def background_of(model_d, p, st):
        if background_level is not None:
            return cp.full(shape, float(background_level)), {"source": "fixed"}
        good = (st == "ok") & (p[:, 0] > 0)
        amplitude = float(np.median(p[good, 0])) if good.any() else 1.0
        fwhm = float(np.median(2 * cpu.FWHM_K * sigmas(p)[good])) if good.any() else 0.0
        free = (model_d < free_fraction * amplitude) & (distance_d >= free_distance_fwhm * fwhm)
        return _background_device(image_d - model_d, free, tile, usable_d)

    def clean(p, background_d, model_d, part):
        rows, h, w = parts[part]
        cells = part_masks[part]
        out = cp.empty(cells.shape, dtype=cp.float64)
        total = out.size
        kernel(_CLEAN_SRC, "clean_cells")(
            ((total + 255) // 256,), (256,),
            (image_d, background_d, model_d, cp.asarray(global_params(p, rows)),
             cp.asarray(radii(p)[rows].astype(np.int32)), cells, part_origins[part], out, np.int64(total), np.int32(h),
             np.int32(w), np.int32(shape[0]), np.int32(shape[1])))
        return out

    def fit(data_of, p):
        fitted = p.copy()
        for part, (rows, _, _) in enumerate(parts):
            fitted[rows] = _fit_device(data_of(part), part_masks[part], p[rows], max_iter=max_iter)
        status = np.full(n, "ok", dtype="U32")
        cpu.fit_status(fitted, masks, caps, status, sigma=_sigma_major(fitted))
        return fitted, status

    if init is None:
        # Initialize from the cell-border background; the iterations replace it.
        params = np.empty((n, 6))
        for part, (rows, h, w) in enumerate(parts):
            cells = part_masks[part].astype(bool)
            border = cells & ~binary_erosion(cells, structure=cp.ones((1, 3, 3), bool), border_value=0)
            raw = _windows(image_d, part_origins[part], (h, w))
            bg0 = border_percentile(raw, border)
            if background_level is not None:
                bg0[:] = background_level
            data = cp.where(cells, raw - bg0[:, None, None], 0.0)
            start = cp.asnumpy(_initial_params(data, cells, cp.asarray(seeds_xy[rows] - origins[rows])))
            params[rows] = _fit_device(data, part_masks[part], start, max_iter=max_iter)
        status = np.full(n, "ok", dtype="U32")
        cpu.fit_status(params, masks, caps, status, sigma=_sigma_major(params))
    else:
        params, status = np.asarray(init, dtype=np.float64).copy(), np.full(n, "ok", dtype="U32")
        cpu.fit_status(params, masks, caps, status, sigma=_sigma_major(params))

    model_d = _render_device(global_params(params), shape, radius_cap, radii(params))
    used, converged = 0, False
    for used in range(1, int(iterations) + 1):
        background_d, info = background_of(model_d, params, status)
        fitted, status = fit(lambda part: clean(params, background_d, model_d, part), params)
        # All cells update at once against last iteration's neighbours, damped (see the CPU module).
        step = np.where(np.isfinite(fitted), fitted - params, 0.0)
        params = params + relaxation * step
        status = np.full(n, "ok", dtype="U32")
        cpu.fit_status(params, masks, caps, status, sigma=sigmas(params))
        model_d = _render_device(global_params(params), shape, radius_cap, radii(params))
        ok = (status == "ok") & np.isfinite(fitted).all(axis=1)
        if not ok.any():
            break
        if np.percentile(np.abs(step[ok][:, 1:]).max(axis=1), 99) < tolerance:
            converged = True
            break
    background_d, info = background_of(model_d, params, status)
    cleaned = np.zeros(masks.shape)
    for part, (rows, h, w) in enumerate(parts):
        cleaned[rows, :h, :w] = cp.asnumpy(clean(params, background_d, model_d, part))
    return cpu.DeblendState(cp.asnumpy(background_d), cp.asnumpy(model_d), params, status, cleaned,
                            tile, used, info, converged)
