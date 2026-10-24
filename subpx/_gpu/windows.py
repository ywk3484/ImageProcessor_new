"""GPU kernel behind ``subpx._cpu.invariant_cd._wide_cells`` on CuPy: each reported hole's wide window in one pass.

Same elementwise arithmetic as the array code (compiled without FMA contraction, squares as products like CuPy's
``** 2``), so every window pixel, mask and level pixel is the same, bit for bit; the arrays stay on the GPU.
"""

from __future__ import annotations

import numpy as np

from ._cuda import cp, kernel, require_cupy

_THREADS = 256

# One block per reported hole, window pixel (y, x) at image row oy + y, column ox + x:
#   cell = win * own / (total + floor) inside the image (0 outside), win = data, total = model there,
#   own = the hole's Gaussian within its render box, mine = closer to it than to its other neighbours,
#   level = mine & inside & distance >= the (keep + 1)-th largest such distance, keep = clip(max(count // 10, 4),
#   1, size^2) - 1 (-inf when fewer pixels: all of them), found by sorting those distances in shared memory.
_WIDE_SRC = r"""
extern "C" __global__ void wide_cells(
    const double* __restrict__ data, const double* __restrict__ model, const long long* __restrict__ origins,
    const double* __restrict__ params, const bool* __restrict__ good, const double* __restrict__ render,
    const double* __restrict__ near, const int k, const double floor_, const int size, const int height,
    const int width, const int sorted_len, float* __restrict__ cells, bool* __restrict__ inside_out,
    bool* __restrict__ level_out, bool* __restrict__ own_out)
{
    extern __shared__ double smem[];
    double* dist = smem;                // distance of each window pixel to the hole's own centre
    double* keys = smem + size * size;  // own-territory distances, sorted descending
    __shared__ int s_count;
    const double NEG_INF = __longlong_as_double(0xFFF0000000000000ll);
    const long long hole = blockIdx.x;
    const int P = size * size;
    const long long ox = origins[2 * hole], oy = origins[2 * hole + 1];
    const double* p = params + 6 * hole;
    const bool g = good[hole];
    const double A = p[0], x0 = p[1], y0 = p[2], e1 = exp(p[3]), u2 = p[4], e3 = exp(p[5]), r = render[hole];
    const double* nb = near + (long long)hole * k * 2;
    if (threadIdx.x == 0) s_count = 0;
    for (int i = threadIdx.x; i < sorted_len; i += blockDim.x) keys[i] = NEG_INF;
    __syncthreads();
    for (int i = threadIdx.x; i < P; i += blockDim.x) {
        const long long ys = oy + i / size, xs = ox + i % size;
        const bool in = ys >= 0 && ys < height && xs >= 0 && xs < width;
        const long long cy = ys < 0 ? 0 : (ys > height - 1 ? height - 1 : ys);
        const long long cx = xs < 0 ? 0 : (xs > width - 1 ? width - 1 : xs);
        const double win = data[cy * width + cx], total = model[cy * width + cx];
        const double gx = (double)xs, gy = (double)ys;
        const double dx = gx - x0, dy = gy - y0;
        const double a = e1 * dx + u2 * dy, b = e3 * dy;
        const bool box = fabs(dx) <= r && fabs(dy) <= r && g;
        const double own = box ? A * exp(-0.5 * (a * a + b * b)) : 0.0;
        double d0 = 0.0, nearest = __longlong_as_double(0x7FF0000000000000ll);
        for (int j = 0; j < k; ++j) {
            const double ex = gx - nb[2 * j], ey = gy - nb[2 * j + 1];
            const double d = sqrt(ex * ex + ey * ey);
            if (j == 0) d0 = d; else nearest = fmin(nearest, d);
        }
        const bool mine = k > 1 ? d0 <= nearest : true;
        const bool own_area = in && mine;
        dist[i] = d0;
        if (own_area) keys[atomicAdd(&s_count, 1)] = d0;
        cells[hole * P + i] = (float)(in ? win * own / (total + floor_) : 0.0);
        inside_out[hole * P + i] = in;
        own_out[hole * P + i] = own_area;
    }
    __syncthreads();
    // Bitonic sort of keys, descending (unused slots are -inf and sink to the end).
    for (int len = 2; len <= sorted_len; len <<= 1) {
        for (int stride = len >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < sorted_len; i += blockDim.x) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool down = (i & len) == 0;
                    const double u = keys[i], v = keys[j];
                    if (down ? u < v : u > v) { keys[i] = v; keys[j] = u; }
                }
            }
            __syncthreads();
        }
    }
    const int count = s_count;
    int keep = count / 10 > 4 ? count / 10 : 4;
    keep = (keep < 1 ? 1 : (keep > P ? P : keep)) - 1;
    const double threshold = keep < count ? keys[keep] : NEG_INF;
    for (int i = threadIdx.x; i < P; i += blockDim.x) level_out[hole * P + i] = own_out[hole * P + i] && dist[i] >= threshold;
}
"""


def wide_cells(data, model, origins, params, good, render, near, floor: float, size: int):
    """Windows (n, size, size) of float32 cells, inside, level and own-territory masks, on the GPU (see _WIDE_SRC);
    None when a window does not fit in shared memory. ``params`` (n, 6) are zero where not ``good``."""
    require_cupy()
    n, k = near.shape[:2]
    height, width = data.shape
    pixels = size * size
    sorted_len = 1 << max(0, (pixels - 1).bit_length())
    shared = 8 * (pixels + sorted_len)
    if shared > 48 * 1024:
        return None
    cells = cp.empty((n, size, size), dtype=cp.float32)
    inside, level, own = (cp.empty((n, size, size), dtype=bool) for _ in range(3))
    if n:
        kernel(_WIDE_SRC, "wide_cells", exact=True)(
            (n,), (_THREADS,), (cp.ascontiguousarray(data, dtype=cp.float64), cp.ascontiguousarray(model, dtype=cp.float64),
                                cp.asarray(origins, dtype=cp.int64), cp.asarray(params, dtype=cp.float64),
                                cp.asarray(good, dtype=bool), cp.asarray(render, dtype=cp.float64),
                                cp.asarray(near, dtype=cp.float64), np.int32(k), np.float64(floor), np.int32(size),
                                np.int32(height), np.int32(width), np.int32(sorted_len), cells, inside, level, own),
            shared_mem=shared)
    return cells, inside, level, own
