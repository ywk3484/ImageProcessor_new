"""Batched FP64 cubic enlargement using the existing spline operators."""

from __future__ import annotations

import numpy as np

from ._cuda import cp, kernel, optin_shared_bytes, require_cupy


_SOURCE = r"""
extern "C" __global__ void enlarge_cells(
    const double* cells, const double* zy, const double* zx, double* out,
    const int h, const int w, const int H, const int W)
{
    const int tid = threadIdx.x, cell = blockIdx.x;
    extern __shared__ double work[];
    double* raw = work;
    double* middle = raw + h * w;
    for (int i = tid; i < h * w; i += blockDim.x)
        raw[i] = cells[(long long)cell * h * w + i];
    __syncthreads();
    for (int i = tid; i < H * w; i += blockDim.x) {
        const int y = i / w, x = i % w;
        double sum = 0.0;
        for (int j = 0; j < h; ++j) sum += zy[y * h + j] * raw[j * w + x];
        middle[i] = sum;
    }
    __syncthreads();
    for (int i = tid; i < H * W; i += blockDim.x) {
        const int y = i / W, x = i % W;
        double sum = 0.0;
        for (int j = 0; j < w; ++j) sum += middle[y * w + j] * zx[x * w + j];
        out[(long long)cell * H * W + i] = sum;
    }
}
"""


def enlarge_cells(cells, zy, zx):
    """Return ``zy[None] @ cells @ zx.T[None]`` in FP64.

    The operators, all their coefficients, and both matrix products are
    unchanged. A block reuses a cell and its intermediate product in shared
    memory instead of allocating a batch of intermediate matrices. Floating
    point summation order can differ from cuBLAS. Oversized cells use cuBLAS.
    """
    require_cupy()
    n, h, w = cells.shape
    H, W = zy.shape[0], zx.shape[0]
    shared = (h + H) * w * 8
    if shared > min(optin_shared_bytes(), 48 * 1024):
        return zy[None] @ cells @ zx.T[None]
    out = cp.empty((n, H, W), dtype=cp.float64)
    if n:
        execute = kernel(_SOURCE, "enlarge_cells")
        execute((n,), (128,), (cp.ascontiguousarray(cells, dtype=cp.float64),
                              cp.ascontiguousarray(zy, dtype=cp.float64),
                              cp.ascontiguousarray(zx, dtype=cp.float64), out,
                              np.int32(h), np.int32(w), np.int32(H), np.int32(W)), shared_mem=shared)
    return out
