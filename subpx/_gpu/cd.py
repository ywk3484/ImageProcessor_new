"""GPU CD measurements: ``subpx._cpu.cd`` for whole stacks of deblended cells.

Same ``CDMethodResult`` outputs as the CPU module. Per-pixel and per-sample
work runs on the GPU (ray sampling for halfmax/gradient, log-quadratic sums,
fit residuals); the small per-cell algebra (6x6 and 2x2 solves, eigh, ellipse
and polygon geometry) runs vectorized on the host with the CPU's operations,
so equal inputs give the CPU's contours bit for bit. Radial profiles agree
with SciPy's cubic interpolation to rounding (~1e-15 relative).
"""

from __future__ import annotations

import numpy as np

from .._cpu import cd as cpu
from .._cpu.deblend import precision_matrices
from ..types import CDMethodResult
from ._cuda import REDUCE_SRC, cp, kernel, require_cupy

# SciPy edge-pads by 12 px before the spline prefilter of map_coordinates(mode='nearest').
_PAD = 12
_CHUNK_BYTES = 256 * 2**20
_K = float(np.sqrt(2 * np.log(2)))

# One thread per (cell, ray) samples the cubic-spline profile outward and stops
# at the first sample outside the eroded cell, which is where the CPU's
# logical_and.accumulate validity ends. It returns the first half-height
# crossing and the parabolically refined maximum of the outward decrease,
# with _radial_contour's rules. Compiled without FMA contraction: sample
# coordinates and edge arithmetic round like NumPy's.
_RAY_SRC = r"""
#define PAD 12
#define NAN_D __longlong_as_double(0x7ff8000000000000LL)
#define INF_D __longlong_as_double(0x7ff0000000000000LL)
__device__ __forceinline__ double cubic_at(const double* c, int hp, int wp, double y, double x) {
    double wy[4], wx[4];
    int iy[4], ix[4];
    {
        const double f = floor(y), t = y - f, s = 1.0 - t;
        wy[1] = (t * t * (t - 2.0) * 3.0 + 4.0) / 6.0;
        wy[2] = (s * s * (s - 2.0) * 3.0 + 4.0) / 6.0;
        wy[0] = s * s * s / 6.0;
        wy[3] = 1.0 - wy[0] - wy[1] - wy[2];
        for (int k = 0; k < 4; ++k) iy[k] = min(max((int)f - 1 + k, 0), hp - 1);
    }
    {
        const double f = floor(x), t = x - f, s = 1.0 - t;
        wx[1] = (t * t * (t - 2.0) * 3.0 + 4.0) / 6.0;
        wx[2] = (s * s * (s - 2.0) * 3.0 + 4.0) / 6.0;
        wx[0] = s * s * s / 6.0;
        wx[3] = 1.0 - wx[0] - wx[1] - wx[2];
        for (int k = 0; k < 4; ++k) ix[k] = min(max((int)f - 1 + k, 0), wp - 1);
    }
    double out = 0.0;
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b) out += c[iy[a] * wp + ix[b]] * (wy[a] * wx[b]);
    return out;
}
// map_coordinates(mask, order=1, mode='constant') >= 1 - 1e-8, as _inside.
__device__ __forceinline__ bool inside_mask(const unsigned char* m, int h, int w, double y, double x) {
    if (!(y >= 0.0 && y <= h - 1.0 && x >= 0.0 && x <= w - 1.0)) return false;
    const double fy = floor(y), fx = floor(x);
    const int y0 = (int)fy, x0 = (int)fx, ny = (y == fy) ? 1 : 2, nx = (x == fx) ? 1 : 2;
    double v = 0.0;
    for (int a = 0; a < ny; ++a) {
        const double wy = a == 0 ? (fy + 1.0) - y : y - fy;
        for (int b = 0; b < nx; ++b) {
            const double wx = b == 0 ? (fx + 1.0) - x : x - fx;
            v += (double)m[(y0 + a) * w + x0 + b] * (wy * wx);
        }
    }
    return v >= 1.0 - 1e-8;
}
extern "C" __global__ void radial_edges(
    const double* __restrict__ coef, const unsigned char* __restrict__ safe, const double* __restrict__ center,
    const double* __restrict__ radii, const int L, const double* __restrict__ dirs, const int n_angles,
    double* __restrict__ peak_out, double* __restrict__ halfmax, double* __restrict__ gradient,
    const int n, const int h, const int w, const double threshold)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)n * n_angles) return;
    const int cell = (int)(t / n_angles), ray = (int)(t % n_angles);
    const int hp = h + 2 * PAD, wp = w + 2 * PAD;
    const double* c = coef + (size_t)cell * hp * wp;
    const unsigned char* m = safe + (size_t)cell * h * w;
    const double cx = center[2 * cell], cy = center[2 * cell + 1];
    const double dx = dirs[2 * ray], dy = dirs[2 * ray + 1];
    const double step = radii[1] - radii[0];
    const double peak = cubic_at(c, hp, wp, cy + PAD, cx + PAD);
    const double half = peak / 2;
    double edge_h = NAN_D, edge_g = NAN_D;
    if (peak > 0) {
        double p1 = 0, p2 = 0;            // p[k-1], p[k-2]
        double g1 = 0, g2 = 0, g3 = 0;    // g[k-1], g[k-2], g[k-3] (central differences, negated)
        bool crossed = false, left_ok = false, right_ok = false;
        double best = -INF_D, gl = 0, gm = 0, gr = 0;
        int best_j = -1;
        for (int k = 0; k < L; ++k) {
            const double r = radii[k];
            const double x = cx + dx * r, y = cy + dy * r;
            if (!inside_mask(m, h, w, y, x)) break;
            const double p = k == 0 ? peak : cubic_at(c, hp, wp, y + PAD, x + PAD);
            if (best_j >= 0 && k == best_j + 3)  // valid[j+3]: the right neighbour can be a candidate
                right_ok = best_j + 1 <= L - 3 && radii[best_j + 1] >= threshold;
            if (k >= 1 && !crossed && p1 >= half && p < half) {
                edge_h = radii[k - 1] + ((p1 - half) / (p1 - p)) * step;
                crossed = true;
            }
            if (k >= 2) {
                g3 = g2; g2 = g1; g1 = -((p - p2) / (2.0 * step));
                const int j = k - 2;  // valid[j + 2] holds: j may be a candidate
                if (j >= 2 && j <= L - 3 && radii[j] >= threshold && g2 > best) {
                    best = g2; best_j = j; gl = g3; gm = g2; gr = g1;
                    left_ok = j - 1 >= 2 && radii[j - 1] >= threshold;
                    right_ok = false;
                }
            }
            p2 = p1; p1 = p;
        }
        if (best_j >= 0) {
            const double curvature = gl - 2 * gm + gr;
            if (left_ok && right_ok && gm >= gl && gm >= gr && gm > 0.01 * peak && curvature < -1e-12) {
                const double delta = 0.5 * (gl - gr) / curvature;
                edge_g = radii[best_j] + (delta < -0.5 ? -0.5 : (delta > 0.5 ? 0.5 : delta)) * step;
            }
        }
    }
    if (ray == 0) peak_out[cell] = peak;
    halfmax[t] = edge_h;
    gradient[t] = edge_g;
}
// Points inside each cell's mask (all of a cell's K points against one mask).
extern "C" __global__ void points_inside(
    const double* __restrict__ points, const unsigned char* __restrict__ mask, bool* __restrict__ out,
    const long long total, const int K, const int h, const int w)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    const long long cell = t / K;
    out[t] = inside_mask(mask + cell * h * w, h, w, points[2 * t + 1], points[2 * t]);
}
"""

# Weighted log-quadratic least squares per cell: sums of the normal equations
# in seed-centred coordinates scaled by 1/s, then (second kernel) the weighted
# residual of the solved fit for log_fit_rmse.
_LOGQUAD_SRC = REDUCE_SRC + r"""
__device__ __forceinline__ void design(double u, double v, double* d) {
    d[0] = u * u; d[1] = v * v; d[2] = u * v; d[3] = u; d[4] = v; d[5] = 1.0;
}
extern "C" __global__ void logquad_sums(
    const double* __restrict__ roi, const unsigned char* __restrict__ mask, const double* __restrict__ background,
    const double* __restrict__ seed, const double* __restrict__ peak, const double fit_fraction,
    const double inv_scale, double* __restrict__ out, const int n, const int h, const int w)
{
    __shared__ double partial[32][NV];
    __shared__ double tot[NV];
    const int cell = blockIdx.x;
    if (cell >= n) return;
    const double bg = background[cell], top = peak[cell], sx = seed[2 * cell], sy = seed[2 * cell + 1];
    double v[NV];
    for (int i = 0; i < 28; ++i) v[i] = 0;
    for (int k = threadIdx.x; k < h * w; k += blockDim.x) {
        const size_t i = (size_t)cell * h * w + k;
        const double s = roi[i] - bg;
        if (!mask[i] || !(s > fit_fraction * top)) continue;
        const double wt = s / top, w2 = wt * wt, z = log(s);
        double d[6];
        design(((double)(k % w) - sx) * inv_scale, ((double)(k / w) - sy) * inv_scale, d);
        int t = 1;
        for (int a = 0; a < 6; ++a) for (int b = a; b < 6; ++b) v[t++] += w2 * d[a] * d[b];
        for (int a = 0; a < 6; ++a) v[22 + a] += w2 * d[a] * z;
        v[0] += 1;
    }
    block_sum(v, 28, tot, partial);
    if (threadIdx.x < 28) out[(size_t)cell * 28 + threadIdx.x] = tot[threadIdx.x];
}
extern "C" __global__ void logquad_residual(
    const double* __restrict__ roi, const unsigned char* __restrict__ mask, const double* __restrict__ background,
    const double* __restrict__ seed, const double* __restrict__ peak, const double* __restrict__ coef,
    const double fit_fraction, const double inv_scale, double* __restrict__ out, const int n, const int h, const int w)
{
    __shared__ double partial[32][NV];
    __shared__ double tot[NV];
    const int cell = blockIdx.x;
    if (cell >= n) return;
    const double bg = background[cell], top = peak[cell], sx = seed[2 * cell], sy = seed[2 * cell + 1];
    const double* c = coef + 6 * cell;
    double v[NV];
    v[0] = v[1] = 0;
    for (int k = threadIdx.x; k < h * w; k += blockDim.x) {
        const size_t i = (size_t)cell * h * w + k;
        const double s = roi[i] - bg;
        if (!mask[i] || !(s > fit_fraction * top)) continue;
        const double wt = s / top, w2 = wt * wt;
        double d[6];
        design(((double)(k % w) - sx) * inv_scale, ((double)(k / w) - sy) * inv_scale, d);
        double fit = 0;
        for (int a = 0; a < 6; ++a) fit += d[a] * c[a];
        const double r = fit - log(s);
        v[0] += w2 * r * r;
        v[1] += w2;
    }
    block_sum(v, 2, tot, partial);
    if (threadIdx.x < 2) out[(size_t)cell * 2 + threadIdx.x] = tot[threadIdx.x];
}
"""


def _chunks(n: int, cell_bytes: int) -> list[slice]:
    size = max(1, _CHUNK_BYTES // max(cell_bytes, 1))
    return [slice(start, start + size) for start in range(0, n, size)]


def _ellipses(centers, eigenvalues, axes, angles):
    """Batched ``_half_max_ellipse`` (same operations, so equal inputs give equal bits)."""
    sigma = 1 / np.sqrt(eigenvalues)
    contour = None
    if angles is not None:
        unit_circle = np.column_stack((np.cos(angles), np.sin(angles)))
        contour = centers[:, None, :] + (unit_circle[None] * (_K * sigma)[:, None, :]) @ axes.transpose(0, 2, 1)
        contour = np.concatenate((contour, contour[:, :1]), axis=1)
    covariance = (axes * (sigma * sigma)[:, None, :]) @ axes.transpose(0, 2, 1)
    fit = dict(
        sigma_major_px=sigma[:, 0], sigma_minor_px=sigma[:, 1],
        fwhm_major_px=2 * _K * sigma[:, 0], fwhm_minor_px=2 * _K * sigma[:, 1],
        angle_deg=np.degrees(np.arctan2(axes[:, 1, 0], axes[:, 0, 0])) % 180,
    )
    return contour, sigma, covariance, fit


def _polygon_geometry(contours) -> dict[str, np.ndarray]:
    """Batched ``_geometry`` of closed (N, V, 2) contours."""
    vertices = contours - contours[:, :1]
    area = np.abs(np.sum(vertices[:, :-1, 0] * vertices[:, 1:, 1] - vertices[:, 1:, 0] * vertices[:, :-1, 1],
                         axis=1)) / 2
    return {
        "area_px2": area, "cd_equivalent_px": 2 * np.sqrt(area / np.pi),
        "cd_x_px": np.ptp(contours[..., 0], axis=1), "cd_y_px": np.ptp(contours[..., 1], axis=1),
        "perimeter_px": np.linalg.norm(np.diff(contours, axis=1), axis=2).sum(axis=1),
    }


def _ellipse_geometry(contours, sigma, covariance) -> dict[str, np.ndarray]:
    """Batched ``_ellipse_geometry``: analytic area and projections, polygon perimeter."""
    geometry = _polygon_geometry(contours) if contours is not None else {"perimeter_px": np.full(len(sigma), np.nan)}
    area = np.pi * _K**2 * np.prod(sigma, axis=1)
    geometry.update(area_px2=area, cd_equivalent_px=2 * np.sqrt(area / np.pi),
                    cd_x_px=2 * _K * np.sqrt(covariance[:, 0, 0]), cd_y_px=2 * _K * np.sqrt(covariance[:, 1, 1]))
    return geometry


def _store(result: CDMethodResult, cells: np.ndarray, metrics: dict[str, np.ndarray]) -> None:
    for name, values in metrics.items():
        result.metrics[name][cells] = values


def _radial(rois, masks, backgrounds, cells, angles, radial_step, smooth_sigma):
    """Weighted centroids, peaks and per-ray halfmax/gradient edges (host arrays) of ``cells``."""
    from cupyx.scipy.ndimage import binary_erosion, gaussian_filter, spline_filter1d

    n, (h, w) = len(cells), rois.shape[1:]
    radii = np.arange(0, np.hypot(h, w) + radial_step, radial_step)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    threshold = max(0.35, 2 * (radii[1] - radii[0]))
    cross = cp.asarray([[[0, 1, 0], [1, 1, 1], [0, 1, 0]]], dtype=bool)
    yy, xx = cp.indices((h, w), dtype=cp.float64)
    radii_d, directions_d = cp.asarray(radii), cp.asarray(directions)
    centers, peaks = np.empty((n, 2)), np.empty(n)
    halfmax, gradient = np.empty((n, len(angles))), np.empty((n, len(angles)))
    for sl in _chunks(n, 8 * (h + 2 * _PAD) * (w + 2 * _PAD) * 3):
        idx = cells[sl]  # gathered per chunk: no full-stack copies on the host
        roi, mask = cp.asarray(rois[idx], dtype=cp.float64), cp.asarray(masks[idx])
        bg = cp.asarray(backgrounds[idx], dtype=cp.float64)[:, None, None]
        count = len(roi)
        signal = cp.where(mask, roi - bg, 0)
        top = signal.reshape(count, -1).max(axis=1)[:, None, None]
        weights = cp.where(mask & (signal > 0.1 * top), signal, 0)
        center = cp.stack(((weights * xx).sum(axis=(1, 2)), (weights * yy).sum(axis=(1, 2))), axis=1)
        center /= weights.sum(axis=(1, 2))[:, None]
        if smooth_sigma > 0:
            support = gaussian_filter(mask.astype(cp.float64), (0, smooth_sigma, smooth_sigma), mode="constant")
            signal = gaussian_filter(signal, (0, smooth_sigma, smooth_sigma), mode="constant") / cp.maximum(support, 1e-12)
        coef = cp.pad(signal, ((0, 0), (_PAD, _PAD), (_PAD, _PAD)), mode="edge")
        coef = spline_filter1d(coef, 3, axis=1, output=cp.float64, mode="nearest")
        coef = cp.ascontiguousarray(spline_filter1d(coef, 3, axis=2, output=cp.float64, mode="nearest"))
        safe = binary_erosion(mask, structure=cross, border_value=0).astype(cp.uint8)
        peak_d = cp.empty(count)
        half_d, grad_d = cp.empty((count, len(angles))), cp.empty((count, len(angles)))
        total = count * len(angles)
        kernel(_RAY_SRC, "radial_edges", exact=True)(
            ((total + 127) // 128,), (128,),
            (coef, safe, cp.ascontiguousarray(center), radii_d, np.int32(len(radii)), directions_d,
             np.int32(len(angles)), peak_d, half_d, grad_d, np.int32(count), np.int32(h), np.int32(w),
             np.float64(threshold)))
        centers[sl], peaks[sl] = cp.asnumpy(center), cp.asnumpy(peak_d)
        halfmax[sl], gradient[sl] = cp.asnumpy(half_d), cp.asnumpy(grad_d)
    return centers, peaks, {"halfmax": halfmax, "gradient": gradient}


def _points_inside(points: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """(N, K) points [x, y] tested against each cell's mask like ``_inside``."""
    n, k = points.shape[:2]
    out = cp.empty((n, k), dtype=bool)
    total = n * k
    kernel(_RAY_SRC, "points_inside", exact=True)(
        ((total + 255) // 256,), (256,),
        (cp.asarray(np.ascontiguousarray(points), dtype=cp.float64), cp.asarray(masks, dtype=cp.uint8), out,
         np.int64(total), np.int32(k), np.int32(masks.shape[1]), np.int32(masks.shape[2])))
    return cp.asnumpy(out)


def _logquad(rois, masks, seeds_local, backgrounds, cells, angles, fit_fraction):
    """Per-cell logquad results of ``cells``: status, centers, contours, metrics, ellipse geometry."""
    n, (h, w) = len(cells), rois.shape[1:]
    scale = max(h, w) / 2
    sums, rmse_parts, peaks = np.empty((n, 28)), np.empty((n, 2)), np.empty(n)
    sums_kernel = kernel(_LOGQUAD_SRC, "logquad_sums")
    for sl in _chunks(n, 9 * h * w):
        idx = cells[sl]
        count = len(idx)
        roi, mask = cp.asarray(rois[idx], dtype=cp.float64), cp.asarray(masks[idx])
        bg = cp.asarray(backgrounds[idx], dtype=cp.float64)
        peak = cp.where(mask, roi - bg[:, None, None], -cp.inf).reshape(count, -1).max(axis=1)
        out = cp.empty((count, 28))
        sums_kernel((count,), (128,), (
            roi, mask.astype(cp.uint8), bg, cp.asarray(seeds_local[sl], dtype=cp.float64), peak,
            np.float64(fit_fraction), np.float64(1 / scale), out, np.int32(count), np.int32(h), np.int32(w)))
        peaks[sl], sums[sl] = cp.asnumpy(peak), cp.asnumpy(out)
    status = np.full(n, "ok", dtype="U32")
    pixels = sums[:, 0]
    normal = np.empty((n, 6, 6))
    iu = np.triu_indices(6)
    normal[:, iu[0], iu[1]] = sums[:, 1:22]
    normal[:, iu[1], iu[0]] = sums[:, 1:22]
    rhs = sums[:, 22:28]
    status[pixels < 6] = "insufficient_fit_pixels"
    trying = status == "ok"
    eig = np.linalg.eigvalsh(np.where(trying[:, None, None], normal, np.eye(6)))
    # A rank-deficient design (all fit pixels on one conic) is lstsq's rank < 6.
    singular = trying & ~(eig[:, 0] > 1e-12 * eig[:, -1])
    status[singular] = "singular_fit"
    trying &= ~singular
    coef_scaled = np.zeros((n, 6))
    coef_scaled[trying] = np.linalg.solve(normal[trying], rhs[trying][..., None])[..., 0]
    coef = coef_scaled / np.array([scale**2, scale**2, scale**2, scale, scale, 1.0])  # back to pixel units
    a, b, c, d, e = coef[:, 0], coef[:, 1], coef[:, 2], coef[:, 3], coef[:, 4]
    precision = -np.stack((np.stack((2 * a, c), axis=-1), np.stack((c, 2 * b), axis=-1)), axis=-2)
    eigenvalues, axes = np.full((n, 2), np.nan), np.full((n, 2, 2), np.nan)
    eigenvalues[trying], axes[trying] = np.linalg.eigh(precision[trying])
    ev0, ev1 = eigenvalues[:, 0], eigenvalues[:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        non_gaussian = trying & ((ev0 <= 1e-10) | (ev1 / ev0 > 1e8))
    status[non_gaussian] = "non_gaussian_fit"
    trying &= ~non_gaussian
    eigenvalues[~trying], axes[~trying] = np.nan, np.nan
    centers = seeds_local.astype(np.float64).copy()
    contours = np.full((n, len(angles) + 1, 2), np.nan)
    metrics: dict[str, np.ndarray] = {"background": backgrounds[cells].astype(np.float64), "peak_signal": peaks}
    fitted = np.flatnonzero(trying)
    if len(fitted):
        rhs2 = np.stack((d[fitted], e[fitted]), axis=-1)[..., None]
        centers[fitted] = seeds_local[fitted] + np.linalg.solve(precision[fitted], rhs2)[..., 0]
        contour, sigma, covariance, fit = _ellipses(centers[fitted], eigenvalues[fitted], axes[fitted], angles)
        contours[fitted] = contour
        residual_kernel = kernel(_LOGQUAD_SRC, "logquad_residual")
        for sl in _chunks(len(fitted), 9 * h * w):
            rows = fitted[sl]
            idx = cells[rows]
            out = cp.empty((len(rows), 2))
            residual_kernel((len(rows),), (128,), (
                cp.asarray(rois[idx], dtype=cp.float64), cp.asarray(masks[idx], dtype=cp.uint8),
                cp.asarray(backgrounds[idx], dtype=cp.float64), cp.asarray(seeds_local[rows], dtype=cp.float64),
                cp.asarray(peaks[rows]), cp.asarray(coef_scaled[rows]), np.float64(fit_fraction),
                np.float64(1 / scale), out, np.int32(len(rows)), np.int32(h), np.int32(w)))
            rmse_parts[rows] = cp.asnumpy(out)
        for name, values in fit.items():
            metrics.setdefault(name, np.full(n, np.nan))[fitted] = values
        metrics.setdefault("log_fit_rmse", np.full(n, np.nan))[fitted] = np.sqrt(
            rmse_parts[fitted, 0] / rmse_parts[fitted, 1])
        inside = _points_inside(np.concatenate((centers[fitted, None, :], contour), axis=1),
                                masks[cells[fitted]]).all(axis=1)
        status[fitted[~inside]] = "contour_outside_cell"
        good = fitted[inside]
        geometry = _ellipse_geometry(contour[inside], sigma[inside], covariance[inside])
        for name, values in geometry.items():
            metrics.setdefault(name, np.full(n, np.nan))[good] = values
    return status, centers, contours, metrics, eigenvalues, axes


def measure_cells(
    rois: np.ndarray, masks: np.ndarray, origins_xy: np.ndarray, seeds_xy: np.ndarray,
    *, methods: tuple[str, ...], n_angles: int, radial_step: float,
    smooth_sigma: float, fit_fraction: float, backgrounds: np.ndarray | None = None,
) -> dict[str, CDMethodResult]:
    """GPU ``subpx._cpu.cd.measure_cells`` for cells with known ``backgrounds`` (deblended cells)."""
    require_cupy()
    if backgrounds is None:
        raise ValueError("the GPU measurement needs per-cell backgrounds (background='model')")
    rois = np.asarray(rois, dtype=np.float64)
    masks = np.asarray(masks, dtype=bool)
    backgrounds = np.asarray(backgrounds, dtype=np.float64)
    origins = np.asarray(origins_xy)
    n = len(rois)
    angles = np.arange(n_angles) * (2 * np.pi / n_angles)
    results = {}
    for method in methods:
        names = cpu.GEOMETRY_METRICS + ("background", "peak_signal")
        names += cpu.FIT_METRICS if method == "logquad" else ("ray_coverage",)
        geometry = {"center_local": np.full((n, 2), np.nan)}
        if method == "logquad":
            geometry.update(eigenvalues=np.full((n, 2), np.nan), axes=np.full((n, 2, 2), np.nan))
        else:
            geometry["edge_px"] = np.full((n, n_angles), np.nan)
        results[method] = CDMethodResult(
            method=method, centers_xy=np.full((n, 2), np.nan),
            contours_xy=np.full((n, n_angles + 1, 2), np.nan),
            valid=np.zeros(n, dtype=bool), status=np.full(n, "no_signal", dtype="U32"),
            metrics={name: np.full(n, np.nan) for name in names}, geometry=geometry,
        )
    if n == 0:
        return results
    small = masks.reshape(n, -1).sum(axis=1) < 6
    for result in results.values():
        result.status[small] = "insufficient_cell_pixels"
    peak = np.max(rois, axis=(1, 2), where=masks, initial=-np.inf)
    cells = np.flatnonzero(~small & (peak > backgrounds))
    if not len(cells):
        return results
    origin = origins[cells]
    radial = [m for m in methods if m != "logquad"]
    if radial:
        centers, peaks, edges = _radial(rois, masks, backgrounds, cells, angles, radial_step, smooth_sigma)
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        for method in radial:
            result, edge = results[method], edges[method]
            points = centers[:, None, :] + edge[:, :, None] * directions[None]
            contour = np.concatenate((points, points[:, :1]), axis=1)
            complete = np.isfinite(edge).all(axis=1)
            result.geometry["center_local"][cells] = centers
            result.geometry["edge_px"][cells] = edge
            result.centers_xy[cells] = centers + origin
            result.contours_xy[cells] = contour + origin[:, None, :]
            result.status[cells] = np.where(complete, "ok", "incomplete_rays")
            _store(result, cells, {"background": backgrounds[cells], "peak_signal": peaks,
                                   "ray_coverage": np.isfinite(edge).mean(axis=1)})
            _store(result, cells[complete], _polygon_geometry(contour[complete]))
    if "logquad" in methods:
        result = results["logquad"]
        seeds_local = np.asarray(seeds_xy, dtype=np.float64)[cells] - origin
        status, centers, contours, metrics, eigenvalues, axes = _logquad(
            rois, masks, seeds_local, backgrounds, cells, angles, fit_fraction)
        result.geometry["center_local"][cells] = centers
        result.geometry["eigenvalues"][cells] = eigenvalues
        result.geometry["axes"][cells] = axes
        result.centers_xy[cells] = centers + origin
        result.contours_xy[cells] = contours + origin[:, None, :]
        result.status[cells] = status
        _store(result, cells, metrics)
    for result in results.values():
        result.valid[:] = result.status == "ok"
    return results


def gaussian_method(
    params: np.ndarray, status: np.ndarray, cleaned: np.ndarray, masks: np.ndarray,
    origins_xy: np.ndarray, n_angles: int, *, metrics_only: bool = False,
) -> CDMethodResult:
    """GPU ``subpx._cpu.cd.gaussian_method``: fit residuals on the GPU, ellipses vectorized on the host."""
    require_cupy()
    n = len(params)
    angles = None if metrics_only else np.arange(n_angles) * (2 * np.pi / n_angles)
    names = cpu.GEOMETRY_METRICS + ("background", "peak_signal") + cpu.FIT_METRICS[:-1] + ("fit_rmse",)
    result = CDMethodResult(
        method="gaussian", centers_xy=np.full((n, 2), np.nan),
        contours_xy=np.full((n, 0 if metrics_only else n_angles + 1, 2), np.nan),
        valid=np.zeros(n, dtype=bool), status=np.asarray(status, dtype="U32").copy(),
        metrics={name: np.full(n, np.nan) for name in names},
        geometry={} if metrics_only else {"eigenvalues": np.full((n, 2), np.nan), "axes": np.full((n, 2, 2), np.nan)},
    )
    if n == 0:
        return result
    from .._cpu.invariant_cd import _size_buckets

    rmse = np.empty(n)
    # Each cell within the top-left sub-block of the stack that holds its mask: the residual is zero beyond it.
    for members, h, w in _size_buckets(masks):
        yy, xx = cp.indices((h, w), dtype=cp.float64)
        for sl in _chunks(len(members), 8 * h * w * 4):
            rows = members[sl]
            p = cp.asarray(params[rows], dtype=cp.float64)
            data = cp.asarray(cleaned[:, :h, :w][rows], dtype=cp.float64)
            mask = cp.asarray(masks[:, :h, :w][rows])
            dx = xx.ravel()[None] - p[:, 1, None]
            dy = yy.ravel()[None] - p[:, 2, None]
            e1, u2, e3 = cp.exp(p[:, 3])[:, None], p[:, 4][:, None], cp.exp(p[:, 5])[:, None]
            a, b = e1 * dx + u2 * dy, e3 * dy
            model = (p[:, 0][:, None] * cp.exp(-0.5 * (a * a + b * b))).reshape(data.shape)
            residual = cp.where(mask, data - model, 0.0)
            rmse[rows] = cp.asnumpy(cp.sqrt((residual**2).sum(axis=(1, 2)) / cp.maximum(mask.sum(axis=(1, 2)), 1)))
    result.metrics["peak_signal"][:] = params[:, 0]
    result.metrics["fit_rmse"][:] = rmse
    finite = np.flatnonzero(np.isfinite(params).all(axis=1))
    if not len(finite):
        return result
    centers = params[finite, 1:3] + origins_xy[finite]
    result.centers_xy[finite] = centers
    eigenvalues, axes = np.linalg.eigh(precision_matrices(params[finite]))
    degenerate = eigenvalues[:, 0] <= 1e-10
    result.status[finite[degenerate]] = "non_gaussian_fit"
    keep = ~degenerate
    cells = finite[keep]
    contour, sigma, covariance, fit = _ellipses(centers[keep], eigenvalues[keep], axes[keep], angles)
    if not metrics_only:
        result.contours_xy[cells] = contour
        result.geometry["eigenvalues"][cells] = eigenvalues[keep]
        result.geometry["axes"][cells] = axes[keep]
    _store(result, cells, fit)
    ok = result.status[cells] == "ok"
    _store(result, cells[ok], _ellipse_geometry(None if metrics_only else contour[ok], sigma[ok], covariance[ok]))
    result.valid[:] = result.status == "ok"
    return result
