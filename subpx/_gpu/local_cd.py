"""Fused numerical-only measurements of raw Voronoi cells.

Factor 4 retains the cubic grid of ``_cpu.local_cd``; factor 1 counts native
pixels without interpolation. Local levels, connected regions and validity
rules match the corresponding reference. One block holds a cell's small
working arrays in shared memory. Enlarged cells use bounded device
scratch storage; no enlarged stack or mask crosses the host.
"""

from __future__ import annotations

import numpy as np

from ._cuda import REDUCE_SRC, cp, kernel, optin_shared_bytes, require_cupy
from .regions import _GROW_STEP_SRC, _TOP_SRC
from .spline import CUBIC_SRC


AREA_METHODS = ("local_half_area", "local_half_area_min_avg", "local_tail_area")
_WORK_BYTES = 128 * 2**20

_PEAK_SORT_SRC = r"""
__device__ double peak_exchange(double value, double other, bool low, bool ascending) {
    const double lower = low ? value : other, upper = low ? other : value;
    const bool exchange = ascending ? lower > upper : lower < upper;
    return exchange ? other : value;
}
__device__ void sort_peak(double* values, int n, int capacity) {
    const int tid = threadIdx.x, lane = tid & 31;
    if (n <= 8) {
        if (tid == 0) {
            for (int j = 1; j < n; ++j) {
                const double value = values[j]; int k = j;
                while (k > 0 && values[k - 1] > value) { values[k] = values[k - 1]; --k; }
                values[k] = value;
            }
        }
    } else {
        if (n != capacity) {
            for (int j = n + tid; j < capacity; j += blockDim.x)
                values[j] = __longlong_as_double(0x7FF0000000000000ll);
            __syncthreads();
        }
        if (capacity <= 64) {
            if (tid < 32) {
                double a = values[lane];
                double b = capacity == 64 ? values[lane + 32] : 0.0;
                for (int length = 2; length <= 32; length <<= 1) {
                    for (int stride = length >> 1; stride; stride >>= 1) {
                        const double other_a = __shfl_xor_sync(0xffffffffu, a, stride);
                        a = peak_exchange(a, other_a, (lane & stride) == 0, (lane & length) == 0);
                        if (capacity == 64) {
                            const double other_b = __shfl_xor_sync(0xffffffffu, b, stride);
                            b = peak_exchange(b, other_b, (lane & stride) == 0, ((lane + 32) & length) == 0);
                        }
                    }
                }
                if (capacity == 64) {
                    const double old_a = a;
                    a = peak_exchange(a, b, true, true);
                    b = peak_exchange(b, old_a, false, true);
                    for (int stride = 16; stride; stride >>= 1) {
                        const double other_a = __shfl_xor_sync(0xffffffffu, a, stride);
                        const double other_b = __shfl_xor_sync(0xffffffffu, b, stride);
                        a = peak_exchange(a, other_a, (lane & stride) == 0, true);
                        b = peak_exchange(b, other_b, (lane & stride) == 0, true);
                    }
                    values[lane + 32] = b;
                }
                values[lane] = a;
            }
        } else {
            for (int length = 2; length <= capacity; length <<= 1) {
                for (int stride = length >> 1; stride; stride >>= 1) {
                    for (int i = tid; i < capacity; i += blockDim.x) {
                        const int j = i ^ stride;
                        if (j > i) {
                            const double a = values[i], b = values[j];
                            if (((i & length) == 0) ? a > b : a < b) { values[i] = b; values[j] = a; }
                        }
                    }
                    __syncthreads();
                }
            }
        }
    }
    __syncthreads();
}
"""

_SOURCE = REDUCE_SRC + _GROW_STEP_SRC + _TOP_SRC + CUBIC_SRC + _PEAK_SORT_SRC + r"""
extern "C" __global__ void local_areas(
    const double* image, const int* labels, const long long* metadata,
    const double* zy, const double* zx, double* output,
    const int image_h, const int image_w, const int h, const int w, const int sort_length,
    const int n_top, const int methods, const double half_level, const double tail_level, double* enlarged)
{
    const int tid = threadIdx.x, bd = blockDim.x, cell = blockIdx.x;
    const int F = LOCAL_FACTOR, H = F * h, W = F * w, raw_n = h * w, big_n = H * W;
    const int WW = (W + 31) >> 5, words_n = H * WW;
    extern __shared__ double smem[];
    double* raw = smem;
    double* intermediate = raw + raw_n;
#if LOCAL_FACTOR > 1
    double* big = enlarged + (long long)cell * big_n;
    double* sorted = intermediate + H * w;
#else
    double* big = intermediate + H * w;
    double* sorted = big + big_n;
#endif
    double* picked = sorted + sort_length;
    const int top_capacity = n_top <= 8 ? n_top : (1 << (32 - __clz((unsigned int)max(32, n_top) - 1)));
    unsigned int* M = (unsigned int*)(picked + top_capacity);
    unsigned int* A = M + words_n;
    unsigned int* C0 = A + words_n;
    unsigned int* N0 = C0 + words_n;
    unsigned char* mask = (unsigned char*)(N0 + words_n);
    __shared__ int border_count;
    __shared__ double local_levels[2], max_peak, average_peak;
    __shared__ long long max_index;
    __shared__ double partial[4][NV], total[NV];
    __shared__ double best[4];
    __shared__ long long best_index[4];
    __shared__ long long picked_index[8];
    const double INF = __longlong_as_double(0x7FF0000000000000ll);
    const double NAN_ = __longlong_as_double(0x7FF8000000000000ll);
    const long long ox = metadata[5 * cell], oy = metadata[5 * cell + 1];
    const long long rh = metadata[5 * cell + 2], rw = metadata[5 * cell + 3], label = metadata[5 * cell + 4];
    if (tid == 0) border_count = 0;
    for (int i = tid; i < raw_n; i += bd) {
        const int y = i / w, x = i % w;
        const long long gy = oy + y, gx = ox + x;
        const bool inside = y < rh && x < rw && gy >= 0 && gy < image_h && gx >= 0 && gx < image_w;
        const bool own = inside && labels[gy * image_w + gx] == label;
        mask[i] = own;
        raw[i] = own ? image[gy * image_w + gx] : 0.0;
    }
    __syncthreads();
    for (int i = tid; i < sort_length; i += bd) {
        bool border = false;
        if (i < raw_n && mask[i]) {
            const int y = i / w, x = i % w;
            border = y == 0 || y == h - 1 || x == 0 || x == w - 1 ||
                     !mask[i - w] || !mask[i + w] || !mask[i - 1] || !mask[i + 1];
        }
        sorted[i] = border ? raw[i] : INF;
        if (border) atomicAdd(&border_count, 1);
    }
    __syncthreads();
    for (int length = 2; length <= sort_length; length <<= 1) {
        for (int stride = length >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < sort_length; i += bd) {
                const int j = i ^ stride;
                if (j > i) {
                    const double a = sorted[i], b = sorted[j];
                    if (((i & length) == 0) ? a > b : a < b) { sorted[i] = b; sorted[j] = a; }
                }
            }
            __syncthreads();
        }
    }
    if (tid == 0) {
        local_levels[0] = border_count ? (sorted[(border_count - 1) / 2] + sorted[border_count / 2]) / 2 : NAN_;
        local_levels[1] = border_count ? sorted[0] : INF;
    }
    for (int i = tid; i < words_n; i += bd) {
        const int y = i / WW, x0 = (i % WW) * 32;
        unsigned int bits = 0;
        for (int b = 0; b < 32 && x0 + b < W; ++b)
            bits |= (unsigned int)mask[(y / F) * w + (x0 + b) / F] << b;
        M[i] = bits;
    }
    __syncthreads();
    for (int level_kind = 0; level_kind < 2; ++level_kind) {
        if (level_kind == 0 ? !(methods & 5) : !(methods & 2)) continue;
        const double baseline = local_levels[level_kind];
        // Complete zero-extended cubic spline on the same 4x sample grid.
#if LOCAL_FACTOR > 1
        prepare_cubic_rows(raw, mask, baseline, h, w, intermediate, big);
#endif
        double bv = -INF;
        long long bi = 0x7FFFFFFFFFFFFFFFll;
        for (int i = tid; i < big_n; i += bd) {
            const int Y = i / W, X = i % W;
            double sum = 0;
#if LOCAL_FACTOR > 1
            sum = cubic_sample4(intermediate + Y * w, w, 1, X);
#else
            sum = raw[i] - baseline;
#endif
            const double value = mask[(Y / F) * w + X / F] ? sum : -INF;
            big[i] = value;
            if (value > bv || (value == bv && i < bi)) { bv = value; bi = i; }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            const double value = __shfl_down_sync(0xffffffffu, bv, offset);
            const long long index = __shfl_down_sync(0xffffffffu, bi, offset);
            if (value > bv || (value == bv && index < bi)) { bv = value; bi = index; }
        }
        const int lane = tid & 31, warp = tid >> 5;
        if (lane == 0) { best[warp] = bv; best_index[warp] = bi; }
        __syncthreads();
        if (tid == 0) {
            for (int j = 1; j < (bd >> 5); ++j)
                if (best[j] > best[0] || (best[j] == best[0] && best_index[j] < best_index[0])) {
                    best[0] = best[j]; best_index[0] = best_index[j];
                }
            max_peak = best[0]; max_index = best_index[0];
        }
        __syncthreads();
        if (level_kind == 1 || (methods & 4)) {
#if LOCAL_FACTOR == 1
            if (n_top <= 8) {
                // Native pixels normally average four samples. Four block
                // maxima avoid eight radix passes and their shared atomics.
                for (int rank = 0; rank < n_top; ++rank) {
                    double value = -INF;
                    long long index = 0x7FFFFFFFFFFFFFFFll;
                    for (int i = tid; i < big_n; i += bd)
                        if (big[i] > value || (big[i] == value && i < index)) { value = big[i]; index = i; }
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        const double other = __shfl_down_sync(0xffffffffu, value, offset);
                        const long long at = __shfl_down_sync(0xffffffffu, index, offset);
                        if (other > value || (other == value && at < index)) { value = other; index = at; }
                    }
                    if (lane == 0) { best[warp] = value; best_index[warp] = index; }
                    __syncthreads();
                    if (tid == 0) {
                        for (int j = 1; j < (bd >> 5); ++j)
                            if (best[j] > best[0] || (best[j] == best[0] && best_index[j] < best_index[0])) {
                                best[0] = best[j]; best_index[0] = best_index[j];
                            }
                        picked[rank] = best[0];
                        picked_index[rank] = isfinite(best[0]) ? best_index[0] : -1;
                        if (picked_index[rank] >= 0) big[picked_index[rank]] = -INF;
                    }
                    __syncthreads();
                }
                if (tid < n_top && picked_index[tid] >= 0) big[picked_index[tid]] = picked[tid];
                __syncthreads();
            } else
#endif
            select_top(Plain{big}, big_n, n_top, picked);
            __syncthreads();  // select_top finishes with writes from every warp
            // Preserve the same ascending FP64 sum after parallel sorting.
            sort_peak(picked, n_top, top_capacity);
            if (tid == 0) {
                double sum = 0;
                for (int j = 0; j < n_top; ++j) sum += picked[j];
                average_peak = isfinite(sum) ? sum / n_top : -1.0;
            }
            __syncthreads();
        }
        for (int method = 0; method < 3; ++method) {
            if (!(methods & (1 << method)) || ((method == 1) != (level_kind == 1))) continue;
            const double peak = method == 0 ? max_peak : average_peak;
            const double cut = (method == 2 ? tail_level : half_level) * peak;
            const bool signal = peak > 0 && isfinite(cut) && cut < peak;
            for (int i = tid; i < words_n; i += bd) {
                const int y = i / WW, x0 = (i % WW) * 32;
                unsigned int bits = 0;
                for (int b = 0; b < 32 && x0 + b < W; ++b)
                    bits |= (unsigned int)(big[y * W + x0 + b] > cut) << b;
                A[i] = bits & M[i]; C0[i] = 0;
            }
            __syncthreads();
            if (tid == 0 && max_index < big_n) {
                const int y = max_index / W, x = max_index % W, i = y * WW + x / 32;
                C0[i] = A[i] & (1u << (x & 31));
            }
            __syncthreads();
            unsigned int* C = grow(C0, N0, A, H, WW, H * W);
            double sums[7] = {0, 0, 0, 0, 0, 0, 0};
            const unsigned int last_valid = (W & 31) ? ((1u << (W & 31)) - 1u) : 0xffffffffu;
            for (int i = tid; i < words_n; i += bd) {
                const int y = i / WW, j = i % WW;
                const unsigned int valid = j == WW - 1 ? last_valid : 0xffffffffu;
                bool touch = (dilate_word(C, i, y, j, H, WW) & valid & ~M[i]) != 0u;
                touch |= (y == 0 || y == H - 1) && C[i] != 0u;
                touch |= j == 0 && (C[i] & 1u);
                touch |= j == ((W - 1) >> 5) && ((C[i] >> ((W - 1) & 31)) & 1u);
                if (touch) sums[5] = 1;
            }
            for (int i = tid; i < big_n; i += bd) {
                const int y = i / W, x = i % W;
                if ((C[y * WW + x / 32] >> (x & 31)) & 1u) {
                    sums[0] += 1; sums[1] += x; sums[2] += y; sums[3] += big[i];
                    sums[4] += (double)(x * x); sums[6] += (double)(y * y);
                }
            }
            block_sum(sums, 7, total, partial);
            const double count = total[0], weight = fmax(count, 1.0), cx = total[1] / weight, cy = total[2] / weight;
            const double mean_signal = total[3] / weight, touch = total[5];
            if (tid == 0) {
                double* o = output + ((long long)cell * 3 + method) * 10;
                const double area = count / (F * F);
                // Exact integer raw moments avoid a second scan and cancellation
                // in the central moments. Supported cell sizes fit in 64 bits.
                const unsigned long long n = (unsigned long long)count;
                const unsigned long long x = (unsigned long long)total[1], y = (unsigned long long)total[2];
                const unsigned long long xx = (unsigned long long)total[4], yy = (unsigned long long)total[6];
                const double denominator = weight * weight * (F * F);
                const double ratio = sqrt(sqrt(fmax((double)(n * xx - x * x) / denominator, 1e-12) /
                                                fmax((double)(n * yy - y * y) / denominator, 1e-12)));
                const double cd = 2 * sqrt(area / 3.141592653589793);
                o[0] = baseline; o[1] = peak; o[2] = baseline + cut; o[3] = mean_signal;
                o[4] = area; o[5] = cd * ratio; o[6] = cd / ratio;
                o[7] = cx / F - (F - 1.0) / (2 * F); o[8] = cy / F - (F - 1.0) / (2 * F);
                o[9] = !signal ? 0 : (touch > 0 ? 1 : 2);
            }
            __syncthreads();
        }
        __syncthreads();
    }
}
"""


def _shared_bytes(h, w, n_top, factor=4):
    sort_length = 1 << max(0, (h * w - 1).bit_length())
    words = factor * h * ((factor * w + 31) // 32)
    peak_capacity = n_top if n_top <= 8 else 1 << (max(32, n_top) - 1).bit_length()
    enlarged = factor * factor * h * w if factor == 1 else 0
    return 8 * (h * w + factor * h * w + enlarged + sort_length + peak_capacity) + 16 * words + h * w


def measure_local_areas(signal, labels, origins, sizes, report, *, methods, n_angles=16,
                        half_area_level=0.5, half_area_peak_pixels=4.0, tail_area_level=0.2, factor=4):
    """Numerical-only local area results, or None when any cell is too large.

    Origins/sizes contain only reported cells; ``report`` maps them to labels.
    Returning None leaves oversized inputs to the existing bounded fallback.
    All result arrays are NumPy arrays. Reduction order can differ by rounding.
    ``factor=1`` is a distinct pixel-count estimator; ``factor=4`` keeps the
    established cubic grid. Method names here are the canonical local names.
    """
    from .._cpu.invariant_cd import _empty

    require_cupy()
    if any(name not in AREA_METHODS for name in methods):
        raise ValueError("the fused local path supports area methods only")
    if factor not in (1, 4):
        raise ValueError("factor must be 1 or 4")
    count = len(report)
    buckets = (np.asarray(sizes, dtype=np.int64) + 3) // 4 * 4
    width_key = int(buckets[:, 1].max()) + 1 if count else 1
    keys, groups = np.unique(buckets[:, 0] * width_key + buckets[:, 1], return_inverse=True)
    shapes = np.column_stack((keys // width_key, keys % width_key))
    top = max(1, int(round(half_area_peak_pixels * factor * factor)))
    capacity = min(optin_shared_bytes(), 96 * 1024)
    if any(_shared_bytes(int(h), int(w), min(top, int(h * w * factor * factor)), factor) + 3072 > capacity
           for h, w in shapes):
        return None
    found = {name: _empty(name, count, n_angles, ("local_level", "threshold", "mean_signal"), metrics_only=True)
             for name in methods}
    if not count:
        return found
    image_d = cp.asarray(signal, dtype=cp.float64)
    labels_d = cp.asarray(labels, dtype=cp.int32)
    bits = sum(1 << AREA_METHODS.index(name) for name in methods)
    execute = kernel(_SOURCE, "local_areas", exact=True, defines=(f"LOCAL_FACTOR={factor}",))
    indices = np.argsort(groups, kind="stable")
    boundaries = np.concatenate(([0], np.bincount(groups).cumsum()))
    metadata = cp.asarray(np.column_stack((origins, sizes, report))[indices], dtype=cp.int64)
    out = cp.empty((count, 3, 10), dtype=cp.float64)
    if factor > 1:
        # Reuse one allocation across every bucket, including shape changes.
        # Allocating a replacement before releasing the previous bucket would
        # transiently double the scratch budget.
        work_cells = factor * factor * shapes[:, 0] * shapes[:, 1]
        work_counts = np.minimum(np.diff(boundaries), np.maximum(1, _WORK_BYTES // (8 * work_cells)))
        scratch = cp.empty(int(np.max(work_counts * work_cells)), dtype=cp.float64)
    else:
        scratch = image_d  # unused by the native-pixel kernel

    for group, (hb, wb) in enumerate(shapes):
        h, w = int(hb), int(wb)
        start, stop = boundaries[group:group + 2]
        n_top = min(top, h * w * factor * factor)
        shared = _shared_bytes(h, w, n_top, factor)
        if shared + 3072 > 48 * 1024:
            execute.max_dynamic_shared_size_bytes = shared
        chunk = max(1, _WORK_BYTES // (factor * factor * h * w * 8)) if factor > 1 else int(stop - start)
        for begin in range(int(start), int(stop), chunk):
            end = min(begin + chunk, int(stop))
            execute((end - begin,), (128,), (
                image_d, labels_d, metadata[begin:end], image_d, image_d, out[begin:end],
                np.int32(signal.shape[0]), np.int32(signal.shape[1]),
                np.int32(h), np.int32(w), np.int32(1 << max(0, (h * w - 1).bit_length())), np.int32(n_top),
                np.int32(bits), np.float64(half_area_level), np.float64(tail_area_level), scratch), shared_mem=shared)
    host = cp.asnumpy(out)  # one synchronization and download for every bucket and method
    for name, result in found.items():
        numbers = host[:, AREA_METHODS.index(name)]
        signal_ok, valid = numbers[:, 9] > 0, numbers[:, 9] == 2
        result.status[indices] = np.where(~signal_ok, "no_signal", np.where(valid, "ok", "region_at_cell_border"))
        result.valid[indices] = valid
        m = result.metrics
        m["local_level"][indices] = m["background"][indices] = numbers[:, 0]
        for column, key in enumerate(("peak_signal", "threshold", "mean_signal"), 1):
            m[key][indices] = np.where(signal_ok, numbers[:, column], np.nan)
        for column, key in enumerate(("area_px2", "cd_x_px", "cd_y_px"), 4):
            m[key][indices] = np.where(valid, numbers[:, column], np.nan)
        m["cd_equivalent_px"][indices] = 2 * np.sqrt(m["area_px2"][indices] / np.pi)
        centers = np.where(signal_ok[:, None], numbers[:, 7:9], np.nan)
        result.geometry["center_local"][indices] = centers
        result.centers_xy[indices] = centers + origins[indices]
    return found
