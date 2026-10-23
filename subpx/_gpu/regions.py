"""GPU kernels behind ``subpx._cpu.invariant_cd._half_area`` on CuPy.

``grow_regions`` replaces the flood-fill loop (hundreds of small array operations per chunk) and ``top_mean`` the
full sort behind the averaged peak. Both return the same arrays as the code they replace, bit for bit.
"""

from __future__ import annotations

import numpy as np

from ._cuda import cp, kernel, optin_shared_bytes, require_cupy

_THREADS = 256

# One block per cell. The region grows from the cell's top pixel in synchronous steps, each adding the 8 neighbours
# of the region that are above the threshold (the loop's _grow(region) & above), until a step changes nothing or
# max_steps steps are done: the loop's result. Rows are bit-packed in shared memory: pixel x of row y is bit (x & 31)
# of word y * WW + (x >> 5); bits past the row end stay 0 because they are 0 in A.
_GROW_SRC = r"""
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
    for (int step = 0; step < max_steps; ++step) {
        int changed = 0;
        for (int i = threadIdx.x; i < P; i += blockDim.x) {
            const int y = i / WW, j = i - y * WW;
            // Rows y - 1 .. y + 1 (words j - 1 .. j + 1), then one pixel to the left and to the right.
            unsigned int v = C[i], l = j > 0 ? C[i - 1] : 0u, r = j < WW - 1 ? C[i + 1] : 0u;
            if (y > 0) { v |= C[i - WW]; if (j > 0) l |= C[i - WW - 1]; if (j < WW - 1) r |= C[i - WW + 1]; }
            if (y < H - 1) { v |= C[i + WW]; if (j > 0) l |= C[i + WW - 1]; if (j < WW - 1) r |= C[i + WW + 1]; }
            const unsigned int grown = (v | (v << 1) | (l >> 31) | (v >> 1) | (r << 31)) & A[i];
            N[i] = grown;
            changed |= grown != C[i];
        }
        changed = __syncthreads_or(changed);
        unsigned int* swap = C; C = N; N = swap;
        if (!changed) break;
    }
    bool* o = out + cell * (long long)H * W;
    for (int p = threadIdx.x; p < H * W; p += blockDim.x) {
        const int y = p / W, x = p - y * W;
        o[p] = (C[y * WW + (x >> 5)] >> (x & 31)) & 1u;
    }
}
"""

# One block per row: radix-select the key of the n-th largest value 8 bits at a time, then write the n largest
# values (in no particular order) to out[row, :n]. Keys order the doubles as CuPy's sort does (NaN largest).
_TOP_SRC = r"""
__device__ __forceinline__ unsigned long long order_key(double v) {
    if (v != v) return 0xFFFFFFFFFFFFFFFFull;
    const unsigned long long b = (unsigned long long)__double_as_longlong(v);
    return (b >> 63) ? ~b : (b | 0x8000000000000000ull);
}
__device__ __forceinline__ double key_value(unsigned long long k) {
    return __longlong_as_double((long long)((k >> 63) ? (k & 0x7FFFFFFFFFFFFFFFull) : ~k));
}
extern "C" __global__ void top_values(const double* __restrict__ x, const long long L, const int n,
                                      double* __restrict__ out)
{
    __shared__ unsigned int hist[256];
    __shared__ unsigned long long s_prefix;
    __shared__ int s_need;
    __shared__ unsigned int s_count;
    const double* r = x + (long long)blockIdx.x * L;
    unsigned long long prefix = 0ull, known = 0ull;
    int need = n;  // how many of the values matching the prefix belong to the n largest
    for (int shift = 56; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0u;
        __syncthreads();
        for (long long base = 0; base < L; base += blockDim.x) {
            const long long i = base + threadIdx.x;
            unsigned int digit = 256u;
            if (i < L) {
                const unsigned long long k = order_key(r[i]);
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
        }
        __syncthreads();
        prefix = s_prefix;
        need = s_need;
        known |= 255ull << shift;
        __syncthreads();
    }
    // prefix is now the key of the n-th largest value; `need` of the values equal to it are among the n largest.
    if (threadIdx.x == 0) s_count = 0u;
    __syncthreads();
    double* o = out + (long long)blockIdx.x * n;
    for (long long i = threadIdx.x; i < L; i += blockDim.x) {
        const double v = r[i];
        if (order_key(v) > prefix) o[atomicAdd(&s_count, 1u)] = v;
    }
    __syncthreads();
    const double kth = key_value(prefix);
    for (int j = (int)s_count + threadIdx.x; j < n; j += blockDim.x) o[j] = kth;
}
"""


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


def top_mean(flat, n: int):
    """``cupy.partition(flat, -n, axis=1)[:, -n:].mean(axis=1)`` bit for bit, without sorting whole rows.

    The n largest values of each row are picked, sorted (CuPy's partition sorts whole rows, so its last n columns
    are the sorted n largest) and written into the last n columns of an array of the old shape, so that the mean
    runs over the same view and adds the same values in the same order.
    """
    require_cupy()
    k, length = flat.shape
    if not 1 <= n <= length:
        raise ValueError("n must be between 1 and the row length")
    flat = cp.ascontiguousarray(flat, dtype=cp.float64)
    picked = cp.empty((k, n), dtype=cp.float64)
    if k:
        kernel(_TOP_SRC, "top_values")((k,), (_THREADS,), (flat, np.int64(length), np.int32(n), picked))
    tail = cp.empty((k, length), dtype=cp.float64)
    tail[:, length - n:] = cp.sort(picked, axis=1)
    return tail[:, length - n:].mean(axis=1)
