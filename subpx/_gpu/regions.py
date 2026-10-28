"""GPU kernels behind ``subpx._cpu.invariant_cd._half_area`` on CuPy.

``grow_regions`` replaces the flood-fill loop (hundreds of small array operations per chunk) and ``top_mean`` the
full sort behind the averaged peak. Regions are identical; peak means agree to floating-point rounding.
``measure_regions`` does everything ``_half_area`` derives from the enlarged cells in one kernel (one block per
cell, no (k, H, W) temporaries): the same elementwise arithmetic and region; only the order of floating-point sums
differs (thresholds of edge / foot, mean_signal and second moments agree to rounding).
"""

from __future__ import annotations

import numpy as np

from ._cuda import REDUCE_SRC, cp, kernel, optin_shared_bytes, require_cupy

_THREADS = 256

# One synchronous step of the region growth: N = (8-neighbour dilation of C) & A, rows bit-packed: pixel x of row y
# is bit (x & 31) of word y * WW + (x >> 5). Returns whether this thread's words changed.
_GROW_STEP_SRC = r"""
__device__ __forceinline__ unsigned int dilate_word(const unsigned int* C, int i, int y, int j, int H, int WW) {
    // Rows y - 1 .. y + 1 (words j - 1 .. j + 1), then one pixel to the left and to the right.
    unsigned int v = C[i], l = j > 0 ? C[i - 1] : 0u, r = j < WW - 1 ? C[i + 1] : 0u;
    if (y > 0) { v |= C[i - WW]; if (j > 0) l |= C[i - WW - 1]; if (j < WW - 1) r |= C[i - WW + 1]; }
    if (y < H - 1) { v |= C[i + WW]; if (j > 0) l |= C[i + WW - 1]; if (j < WW - 1) r |= C[i + WW + 1]; }
    return v | (v << 1) | (l >> 31) | (v >> 1) | (r << 31);
}
// Grows C inside A until a step changes nothing or max_steps steps are done; returns the buffer holding the region.
__device__ unsigned int* grow(unsigned int* C, unsigned int* N, const unsigned int* A, int H, int WW, int max_steps) {
    const int P = H * WW;
    for (int step = 0; step < max_steps; ++step) {
        int changed = 0;
        for (int i = threadIdx.x; i < P; i += blockDim.x) {
            const int y = i / WW, j = i - y * WW;
            const unsigned int grown = dilate_word(C, i, y, j, H, WW) & A[i];
            N[i] = grown;
            changed |= grown != C[i];
        }
        changed = __syncthreads_or(changed);
        unsigned int* swap = C; C = N; N = swap;
        if (!changed) break;
    }
    return C;
}
"""

# One block per cell. The region grows from the cell's top pixel in synchronous steps, each adding the 8 neighbours
# of the region that are above the threshold (the loop's _grow(region) & above), until a step changes nothing or
# max_steps steps are done: the loop's result. Bits past the row end stay 0 because they are 0 in A.
_GROW_SRC = _GROW_STEP_SRC + r"""
extern "C" __global__ void grow_regions(
    const bool* __restrict__ above, const long long* __restrict__ top, bool* __restrict__ out,
    const int H, const int W, const int max_steps)
{
    extern __shared__ unsigned int words[];
    const int WW = (W + 31) >> 5, P = H * WW;
    unsigned int* A = words;          // above the threshold
    unsigned int* C = words + P;      // the region now
    unsigned int* N = words + 2 * P;  // the region one step later
    const long long cell = blockIdx.x;
    const bool* a = above + cell * (long long)H * W;
    for (int i = threadIdx.x; i < P; i += blockDim.x) {
        const int y = i / WW, x0 = (i - y * WW) << 5, n = min(32, W - x0);
        const bool* row = a + (long long)y * W + x0;
        unsigned int word = 0u;
        for (int b = 0; b < n; ++b) word |= (unsigned int)row[b] << b;
        A[i] = word;
        C[i] = 0u;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const long long t = top[cell];
        const int y = (int)(t / W), x = (int)(t % W), i = y * WW + (x >> 5);
        C[i] = A[i] & (1u << (x & 31));
    }
    __syncthreads();
    C = grow(C, N, A, H, WW, max_steps);
    bool* o = out + cell * (long long)H * W;
    for (int p = threadIdx.x; p < H * W; p += blockDim.x) {
        const int y = p / W, x = p - y * W;
        o[p] = (C[y * WW + (x >> 5)] >> (x & 31)) & 1u;
    }
}
"""

# One block per row: radix-select the key of the n-th largest value 8 bits at a time, then write the n largest
# values (in no particular order) to out[row, :n]. Keys order the doubles as CuPy's sort does (NaN largest).
# top_values_masked reads an enlarged cell stack instead, -inf where the order-0 enlarged mask is False (the
# values of ``where(big_mask, big, -inf)``).
_TOP_SRC = r"""
__device__ __forceinline__ unsigned long long order_key(double v) {
    if (v != v) return 0xFFFFFFFFFFFFFFFFull;
    const unsigned long long b = (unsigned long long)__double_as_longlong(v);
    return (b >> 63) ? ~b : (b | 0x8000000000000000ull);
}
__device__ __forceinline__ double key_value(unsigned long long k) {
    return __longlong_as_double((long long)((k >> 63) ? (k & 0x7FFFFFFFFFFFFFFFull) : ~k));
}
struct Plain {
    const double* r;
    __device__ double operator()(long long i) const { return r[i]; }
};
struct Masked {
    const double* r; const bool* m; const int* iy; const int* ix; int w, W;
    __device__ double operator()(long long i) const {
        const int y = (int)(i / W), x = (int)(i - (long long)y * W);
        return m[iy[y] * w + ix[x]] ? r[i] : __longlong_as_double(0xFFF0000000000000ll);
    }
};
template <class Value> __device__ void select_top(const Value value, const long long L, const int n, double* o)
{
    __shared__ unsigned int hist[256];
    __shared__ unsigned long long s_prefix;
    __shared__ int s_need, s_complete;
    __shared__ unsigned int s_count;
    unsigned long long prefix = 0ull, known = 0ull;
    int need = n;  // how many of the values matching the prefix belong to the n largest
    bool complete = false;
    for (int shift = 56; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0u;
        __syncthreads();
        for (long long base = 0; base < L; base += blockDim.x) {
            const long long i = base + threadIdx.x;
            unsigned int digit = 256u;
            if (i < L) {
                const unsigned long long k = order_key(value(i));
                if ((k & known) == prefix) digit = (unsigned int)((k >> shift) & 255ull);
            }
            // One atomic per distinct digit and warp: most cells share a few digits (e.g. -inf outside the cell).
            const unsigned int peers = __match_any_sync(0xFFFFFFFFu, digit);
            if (digit < 256u && (int)(threadIdx.x & 31) == __ffs(peers) - 1) atomicAdd(&hist[digit], __popc(peers));
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int d = 255, left = need;
            while (d > 0 && (int)hist[d] < left) { left -= (int)hist[d]; --d; }
            s_prefix = prefix | ((unsigned long long)d << shift);
            s_need = left;
            // The whole boundary bin belongs to the answer. Its lower key
            // bits cannot change which n values are selected.
            s_complete = (int)hist[d] == left;
        }
        __syncthreads();
        prefix = s_prefix;
        need = s_need;
        complete = s_complete != 0;
        known |= 255ull << shift;
        __syncthreads();
        if (complete) break;
    }
    // Complete bin: include every key at or above its lower bound. Otherwise
    // prefix is the exact n-th key; fill the required duplicates after it.
    if (threadIdx.x == 0) s_count = 0u;
    __syncthreads();
    for (long long i = threadIdx.x; i < L; i += blockDim.x) {
        const double v = value(i);
        const unsigned long long k = order_key(v);
        if (k > prefix || (complete && k == prefix)) o[atomicAdd(&s_count, 1u)] = v;
    }
    __syncthreads();
    if (!complete) {
        const double kth = key_value(prefix);
        for (int j = (int)s_count + threadIdx.x; j < n; j += blockDim.x) o[j] = kth;
    }
}
extern "C" __global__ void top_values(const double* __restrict__ x, const long long L, const int n,
                                      double* __restrict__ out)
{
    select_top(Plain{x + (long long)blockIdx.x * L}, L, n, out + (long long)blockIdx.x * n);
}
extern "C" __global__ void top_values_masked(
    const double* __restrict__ big, const bool* __restrict__ mask, const int* __restrict__ iy,
    const int* __restrict__ ix, const int h, const int w, const int H, const int W, const int n,
    double* __restrict__ out)
{
    const long long cell = blockIdx.x, L = (long long)H * W;
    select_top(Masked{big + cell * L, mask + cell * (long long)h * w, iy, ix, w, W}, L, n, out + cell * n);
}
"""

# One block per cell: what _half_area computes from the enlarged cell ``big`` (H, W), its mask (h, w) and the
# original pixel (iy, ix) each enlarged row / column copies (order-0 enlargement of the mask). Modes: 0 threshold =
# level x peak, 1 edge (steepness-weighted mean brightness), 2 foot (curvature-weighted mean below the steepest level,
# within the zone mask), 3 given threshold. Same elementwise operations as the array code (compiled without FMA
# contraction): pass 1 the highest pixel (CuPy's argmax: NaN largest, first index) and the steepness sums, pass 2 the
# foot's curvature sums, then the pixels above the threshold, the region grown from the top pixel, its touches flag,
# pixel count, light sum, centroid and second moments.
_MEASURE_SRC = REDUCE_SRC + _GROW_STEP_SRC + r"""
__device__ __forceinline__ bool ahead(double v, long long i, double bv, long long bi) {
    const bool vn = v != v, bn = bv != bv;
    if (vn || bn) return vn && (!bn || i < bi);
    return v > bv || (v == bv && i < bi);
}
extern "C" __global__ void measure_regions(
    const double* __restrict__ big, const bool* __restrict__ mask, const bool* __restrict__ zone,
    const int* __restrict__ iy_g, const int* __restrict__ ix_g, const double* __restrict__ peak_in,
    const double* __restrict__ cut_in, const int h, const int w, const int H, const int W, const int mode,
    const int has_zone, const int has_peak, const double level, const int d, const double dy2, const double dx2,
    const double sy2, const double sx2, const int max_steps, long long* __restrict__ top_out,
    long long* __restrict__ count_out, bool* __restrict__ touches_out, double* __restrict__ out,
    bool* __restrict__ region_out, const int want_region)
{
    extern __shared__ unsigned int smem[];
    const int WW = (W + 31) >> 5, P = H * WW, tid = threadIdx.x, bd = blockDim.x;
    unsigned int* M = smem;       // the enlarged mask
    unsigned int* A = M + P;      // above the threshold
    unsigned int* C = A + P;      // the region now
    unsigned int* N = C + P;      // the region one step later
    int* iy = (int*)(N + P);      // original row / column of each enlarged row / column
    int* ix = iy + H;
    int* rows = ix + W;           // region pixels per row / column
    int* cols = rows + H;
    unsigned char* cm = (unsigned char*)(cols + W);  // the cell's mask, then its zone mask
    unsigned char* zm = cm + h * w;
    __shared__ double partial[32][NV];
    __shared__ double tot[NV];
    __shared__ double best_v[32];
    __shared__ long long best_i[32];
    __shared__ double s_peak, s_cut, s_steepest, s_total, s_sum;
    __shared__ long long s_top;
    __shared__ int s_touch;
    const double NEG_INF = __longlong_as_double(0xFFF0000000000000ll);
    const double POS_INF = __longlong_as_double(0x7FF0000000000000ll);
    const long long cell = blockIdx.x;
    const double* b = big + cell * (long long)H * W;
    for (int i = tid; i < h * w; i += bd) {
        cm[i] = mask[cell * (long long)(h * w) + i];
        zm[i] = has_zone ? zone[cell * (long long)(h * w) + i] : 1;
    }
    for (int i = tid; i < H; i += bd) iy[i] = iy_g[i];
    for (int i = tid; i < W; i += bd) ix[i] = ix_g[i];
    for (int i = tid; i < P; i += bd) { M[i] = 0u; A[i] = 0u; C[i] = 0u; }
    if (tid == 0) s_touch = 0;
    __syncthreads();

    double v[4] = {0.0, 0.0, 0.0, 0.0};
    double bv = NEG_INF;
    long long bi = 0x7FFFFFFFFFFFFFFFll;
    for (int p = tid; p < H * W; p += bd) {
        const int Y = p / W, X = p - Y * W;
        const bool in = cm[iy[Y] * w + ix[X]];
        if (in) atomicOr(&M[Y * WW + (X >> 5)], 1u << (X & 31));
        const double c = in ? b[p] : NEG_INF;
        if (ahead(c, p, bv, bi)) { bv = c; bi = p; }
        if (mode == 1 || mode == 2) {
            // Pixels whose whole difference stencil (+-d) lies in the cell, away from the array border.
            bool core = in && Y >= d && Y < H - d && X >= d && X < W - d && cm[iy[Y - d] * w + ix[X]]
                        && cm[iy[Y + d] * w + ix[X]] && cm[iy[Y] * w + ix[X - d]] && cm[iy[Y] * w + ix[X + d]];
            if (mode == 2) core = core && zm[iy[Y] * w + ix[X]];
            if (core) {
                const double t1 = (b[p + d * W] - b[p - d * W]) / dy2, t2 = (b[p + d] - b[p - d]) / dx2;
                const double steep = sqrt(t1 * t1 + t2 * t2);
                v[0] += steep;
                v[1] += steep * c;
            }
        }
    }
    const int lane = tid & 31, warp = tid >> 5, warps = bd >> 5;
    for (int o = 16; o > 0; o >>= 1) {
        const double ov = __shfl_down_sync(0xFFFFFFFFu, bv, o);
        const long long oi = __shfl_down_sync(0xFFFFFFFFu, bi, o);
        if (ahead(ov, oi, bv, bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { best_v[warp] = bv; best_i[warp] = bi; }
    block_sum(v, 2, tot, partial);  // ends with a barrier: best_v / best_i are visible
    if (tid == 0) {
        for (int k = 1; k < warps; ++k) if (ahead(best_v[k], best_i[k], best_v[0], best_i[0])) {
            best_v[0] = best_v[k]; best_i[0] = best_i[k];
        }
        s_top = best_i[0];
        const double peak = has_peak ? peak_in[cell] : best_v[0];
        s_peak = peak;
        s_total = tot[0];
        s_steepest = 0.0;
        if (mode == 0) s_cut = level * peak;
        else if (mode == 3) s_cut = cut_in[cell];
        else if (mode == 1) s_cut = tot[0] > 0 ? tot[1] / tot[0] : POS_INF;
        else { s_steepest = tot[1] / (tot[0] > 0 ? tot[0] : 1.0); s_cut = POS_INF; }
    }
    __syncthreads();
    if (mode == 2) {
        // The foot: curvature-weighted mean brightness over the core pixels below the steepest level.
        const double steepest = s_steepest;
        v[0] = 0.0; v[1] = 0.0;
        for (int p = tid; p < H * W; p += bd) {
            const int Y = p / W, X = p - Y * W;
            const bool core = cm[iy[Y] * w + ix[X]] && zm[iy[Y] * w + ix[X]] && Y >= d && Y < H - d && X >= d
                              && X < W - d && cm[iy[Y - d] * w + ix[X]] && cm[iy[Y + d] * w + ix[X]]
                              && cm[iy[Y] * w + ix[X - d]] && cm[iy[Y] * w + ix[X + d]];
            if (core && b[p] < steepest) {
                const double c = b[p];
                const double curve = (b[p + d * W] + b[p - d * W] - 2 * c) / sy2 + (b[p + d] + b[p - d] - 2 * c) / sx2;
                v[0] += curve;
                v[1] += curve * c;
            }
        }
        block_sum(v, 2, tot, partial);
        if (tid == 0) s_cut = (s_total > 0 && tot[0] > 0) ? tot[1] / tot[0] : POS_INF;
        __syncthreads();
    }
    const double cut = s_cut;
    for (int p = tid; p < H * W; p += bd) {
        const int Y = p / W, X = p - Y * W, i = Y * WW + (X >> 5);
        const unsigned int bit = 1u << (X & 31);
        if ((M[i] & bit) && b[p] > cut) atomicOr(&A[i], bit);
    }
    __syncthreads();
    if (tid == 0) {
        const long long t = s_top;
        const int Y = (int)(t / W), X = (int)(t % W), i = Y * WW + (X >> 5);
        C[i] = A[i] & (1u << (X & 31));
    }
    __syncthreads();
    C = grow(C, N, A, H, WW, max_steps);

    // Touches: the region reaches a pixel outside the enlarged mask (one more dilation) or the array border.
    const unsigned int last_valid = (W & 31) ? ((1u << (W & 31)) - 1u) : 0xFFFFFFFFu;
    for (int i = tid; i < P; i += bd) {
        const int y = i / WW, j = i - y * WW;
        const unsigned int valid = j == WW - 1 ? last_valid : 0xFFFFFFFFu;
        bool touch = (dilate_word(C, i, y, j, H, WW) & valid & ~M[i]) != 0u;
        touch |= (y == 0 || y == H - 1) && C[i] != 0u;
        touch |= j == 0 && (C[i] & 1u);
        touch |= j == ((W - 1) >> 5) && ((C[i] >> ((W - 1) & 31)) & 1u);
        if (touch) s_touch = 1;
    }
    for (int y = tid; y < H; y += bd) {
        int s = 0;
        for (int j = 0; j < WW; ++j) s += __popc(C[y * WW + j]);
        rows[y] = s;
    }
    for (int x = tid; x < W; x += bd) {
        int s = 0;
        for (int y = 0; y < H; ++y) s += (C[y * WW + (x >> 5)] >> (x & 31)) & 1u;
        cols[x] = s;
    }
    __syncthreads();
    v[0] = 0.0; v[1] = 0.0; v[2] = 0.0; v[3] = 0.0;
    for (int y = tid; y < H; y += bd) { v[0] += (double)rows[y]; v[1] += (double)rows[y] * (double)y; }
    for (int x = tid; x < W; x += bd) v[2] += (double)cols[x] * (double)x;
    for (int p = tid; p < H * W; p += bd) {
        const int Y = p / W, X = p - Y * W;
        if ((C[Y * WW + (X >> 5)] >> (X & 31)) & 1u) v[3] += b[p];
    }
    block_sum(v, 4, tot, partial);
    const double count = tot[0], weight = count > 1.0 ? count : 1.0;  // pixel count and sums are exact integers
    const double cy = tot[1] / weight, cx = tot[2] / weight;
    if (tid == 0) s_sum = tot[3];
    v[0] = 0.0; v[1] = 0.0;
    for (int x = tid; x < W; x += bd) { const double t = (double)x - cx; v[0] += (double)cols[x] * (t * t); }
    for (int y = tid; y < H; y += bd) { const double t = (double)y - cy; v[1] += (double)rows[y] * (t * t); }
    block_sum(v, 2, tot, partial);
    if (tid == 0) {
        top_out[cell] = s_top;
        count_out[cell] = (long long)count;
        touches_out[cell] = s_touch != 0;
        double* o = out + cell * 8;
        o[0] = s_peak; o[1] = s_cut; o[2] = s_steepest; o[3] = s_sum; o[4] = cy; o[5] = cx; o[6] = tot[0]; o[7] = tot[1];
    }
    if (want_region) {
        bool* r = region_out + cell * (long long)H * W;
        for (int p = tid; p < H * W; p += bd) {
            const int Y = p / W, X = p - Y * W;
            r[p] = (C[Y * WW + (X >> 5)] >> (X & 31)) & 1u;
        }
    }
}
"""

_MODES = {"level": 0, "edge": 1, "foot": 2, "given": 3}
_STATIC_SHARED = 9 * 1024  # measure_regions' reduction buffers and scalars


def grow_regions(above, top, max_steps: int):
    """Per cell, the pixels of ``above`` (k, H, W) connected through 8-neighbours to its ``top`` pixel (flat index),
    grown for at most ``max_steps`` steps; None when a cell does not fit in shared memory (use the loop then)."""
    require_cupy()
    k, H, W = above.shape
    shared = 3 * H * ((W + 31) // 32) * 4
    if shared > optin_shared_bytes():
        return None
    out = cp.empty((k, H, W), dtype=bool)
    if k == 0:
        return out
    grow = kernel(_GROW_SRC, "grow_regions")
    if shared > 48 * 1024:
        grow.max_dynamic_shared_size_bytes = shared
    grow((k,), (_THREADS,), (cp.ascontiguousarray(above, dtype=bool), cp.ascontiguousarray(top, dtype=cp.int64), out,
                             np.int32(H), np.int32(W), np.int32(max_steps)), shared_mem=shared)
    return out


def _tail_mean(picked, k: int, length: int, n: int):
    """Average only the picked values, without a padded copy of every original pixel.

    The contiguous reduction can round differently from the original strided partition view; its error is at
    floating-point roundoff, while storage scales with the selected count instead of the enlarged window.
    """
    return cp.sort(picked, axis=1).mean(axis=1)


def top_mean(flat, n: int):
    """``cupy.partition(flat, -n, axis=1)[:, -n:].mean(axis=1)`` to roundoff, without sorting whole rows.

    The n largest values of each row are picked, sorted (CuPy's partition sorts whole rows, so its last n columns
    are the sorted n largest). Only those n values are retained for the reduction.
    """
    require_cupy()
    k, length = flat.shape
    if not 1 <= n <= length:
        raise ValueError("n must be between 1 and the row length")
    flat = cp.ascontiguousarray(flat, dtype=cp.float64)
    picked = cp.empty((k, n), dtype=cp.float64)
    if k:
        kernel(_TOP_SRC, "top_values")((k,), (_THREADS,), (flat, np.int64(length), np.int32(n), picked))
    return _tail_mean(picked, k, length, n)


def top_mean_masked(big, mask, iy, ix, n: int, length: int | None = None):
    """``top_mean(where(big_mask, big, -inf).reshape(k, -1), n)`` without building that array: ``big`` (k, H, W),
    ``mask`` (k, h, w), big_mask[:, Y, X] = mask[:, iy[Y], ix[X]]. ``length``: the row length ``top_mean`` would
    see (default H * W; larger when ``big`` is the top-left part of a larger enlargement, -inf beyond it)."""
    require_cupy()
    k, H, W = big.shape
    h, w = mask.shape[1:]
    if not 1 <= n <= H * W:
        raise ValueError("n must be between 1 and the row length")
    picked = cp.empty((k, n), dtype=cp.float64)
    if k:
        kernel(_TOP_SRC, "top_values_masked")(
            (k,), (_THREADS,), (cp.ascontiguousarray(big, dtype=cp.float64), cp.ascontiguousarray(mask, dtype=bool),
                                cp.ascontiguousarray(iy, dtype=cp.int32), cp.ascontiguousarray(ix, dtype=cp.int32),
                                np.int32(h), np.int32(w), np.int32(H), np.int32(W), np.int32(n), picked))
    return _tail_mean(picked, k, H * W if length is None else length, n)


def measure_regions(big, mask, iy, ix, *, mode: str, level: float = 0.0, peak=None, cuts=None, zone=None,
                    d: int = 4, sy: float = 1.0, sx: float = 1.0, max_steps: int = 0, regions: bool = False):
    """``_half_area``'s per-cell results from the enlarged cells ``big`` (k, H, W) and their masks (k, h, w), with
    big_mask[:, Y, X] = mask[:, iy[Y], ix[X]]. ``mode``: 'level' (threshold = ``level`` x peak), 'edge', 'foot'
    (``zone`` (k, h, w) limits its pixels), 'given' (threshold = ``cuts``); ``peak``: the averaged peak per cell, else
    the highest enlarged pixel. Returns device arrays top, count, touches, peak, cut, steepest, sum (light over the
    region), cy, cx (centroid, enlarged px), sxx, syy (sums of squared offsets per pixel) and, with ``regions``, the
    regions; None when a cell does not fit in shared memory."""
    require_cupy()
    k, H, W = big.shape
    h, w = mask.shape[1:]
    words = H * ((W + 31) // 32)
    shared = -(-(16 * words + 8 * (H + W) + 2 * h * w) // 8) * 8  # 4 bit planes, 4 index rows, 2 cell masks
    if shared + _STATIC_SHARED > optin_shared_bytes():
        return None
    out = {"top": cp.empty(k, dtype=cp.int64), "count": cp.empty(k, dtype=cp.int64),
           "touches": cp.empty(k, dtype=bool)}
    values = cp.empty((k, 8), dtype=cp.float64)
    region = cp.empty((k, H, W) if regions else (1,), dtype=bool)
    if k:
        dummy = cp.zeros(1, dtype=cp.float64)
        fn = kernel(_MEASURE_SRC, "measure_regions", exact=True)
        if shared + _STATIC_SHARED > 48 * 1024:
            fn.max_dynamic_shared_size_bytes = shared
        fn((k,), (_THREADS,), (
            cp.ascontiguousarray(big, dtype=cp.float64), cp.ascontiguousarray(mask, dtype=bool),
            cp.ascontiguousarray(zone, dtype=bool) if zone is not None else cp.zeros(1, dtype=bool),
            cp.ascontiguousarray(iy, dtype=cp.int32), cp.ascontiguousarray(ix, dtype=cp.int32),
            cp.ascontiguousarray(peak, dtype=cp.float64) if peak is not None else dummy,
            cp.ascontiguousarray(cuts, dtype=cp.float64) if cuts is not None else dummy,
            np.int32(h), np.int32(w), np.int32(H), np.int32(W), np.int32(_MODES[mode]), np.int32(zone is not None),
            np.int32(peak is not None), np.float64(level), np.int32(d), np.float64(2 * d * sy), np.float64(2 * d * sx),
            np.float64((d * sy) ** 2), np.float64((d * sx) ** 2), np.int32(max_steps), out["top"], out["count"],
            out["touches"], values, region, np.int32(bool(regions))), shared_mem=shared)
    for i, name in enumerate(("peak", "cut", "steepest", "sum", "cy", "cx", "sxx", "syy")):
        out[name] = values[:, i]
    if regions:
        out["region"] = region
    return out
