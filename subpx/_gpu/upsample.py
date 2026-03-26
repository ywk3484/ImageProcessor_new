"""Batch bicubic and nearest-neighbor upsampling via CUDA RawKernels.

Processes (N, H, W) → (N, H*factor, W*factor) in a single kernel launch,
replacing per-ROI ``cndi.zoom`` loops that required 2N launches.

The bicubic kernel uses Keys' cubic (Catmull-Rom, a=-0.5), an interpolating
cubic that requires no pre-filter — unlike scipy's B-spline default.  Center
estimation difference vs. scipy is < 0.02 px for Gaussian-like blobs.
"""
from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None


# ── Batch bicubic upsample (Catmull-Rom, a=-0.5) ───────────────────────────
_BICUBIC_KERNEL_SRC = r"""
extern "C" __global__
void batch_bicubic_upsample(
    const double* __restrict__ src,    // (N, Hin, Win) contiguous
    double* __restrict__ dst,          // (N, Hout, Wout) contiguous
    const int N, const int Hin, const int Win,
    const int Hout, const int Wout, const int factor
) {
    long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long total = (long long)N * Hout * Wout;
    if (idx >= total) return;

    int n  = (int)(idx / ((long long)Hout * Wout));
    int rem = (int)(idx % ((long long)Hout * Wout));
    int oy = rem / Wout;
    int ox = rem % Wout;

    // Center-aligned coordinate mapping (matches ndimage.zoom default)
    double sy = ((double)oy + 0.5) / (double)factor - 0.5;
    double sx = ((double)ox + 0.5) / (double)factor - 0.5;

    int iy = (int)floor(sy);
    int ix = (int)floor(sx);
    double fy = sy - (double)iy;
    double fx = sx - (double)ix;

    // Keys' cubic weights (a = -0.5)
    double wy[4], wx[4];
    {
        double t;
        // wy[0] = W(-1-fy) → W(1+fy)
        t = 1.0 + fy;
        wy[0] = ((-0.5*t + 2.5)*t - 4.0)*t + 2.0;
        // wy[1] = W(-fy)   → W(fy)
        t = fy;
        wy[1] = ((1.5*t - 2.5)*t)*t + 1.0;
        // wy[2] = W(1-fy)
        t = 1.0 - fy;
        wy[2] = ((1.5*t - 2.5)*t)*t + 1.0;
        // wy[3] = W(2-fy)
        t = 2.0 - fy;
        wy[3] = ((-0.5*t + 2.5)*t - 4.0)*t + 2.0;
    }
    {
        double t;
        t = 1.0 + fx;
        wx[0] = ((-0.5*t + 2.5)*t - 4.0)*t + 2.0;
        t = fx;
        wx[1] = ((1.5*t - 2.5)*t)*t + 1.0;
        t = 1.0 - fx;
        wx[2] = ((1.5*t - 2.5)*t)*t + 1.0;
        t = 2.0 - fx;
        wx[3] = ((-0.5*t + 2.5)*t - 4.0)*t + 2.0;
    }

    // Gather 4x4 neighborhood with clamped boundary
    const double* src_n = src + (long long)n * Hin * Win;
    double val = 0.0;
    for (int dy = 0; dy < 4; dy++) {
        int yy = iy - 1 + dy;
        if (yy < 0) yy = 0;
        if (yy >= Hin) yy = Hin - 1;
        for (int dx = 0; dx < 4; dx++) {
            int xx = ix - 1 + dx;
            if (xx < 0) xx = 0;
            if (xx >= Win) xx = Win - 1;
            val += wy[dy] * wx[dx] * src_n[yy * Win + xx];
        }
    }

    dst[idx] = val;
}
"""

# ── Batch nearest-neighbor upsample ─────────────────────────────────────────
_NN_KERNEL_SRC = r"""
extern "C" __global__
void batch_nn_upsample(
    const double* __restrict__ src,    // (N, Hin, Win) contiguous
    double* __restrict__ dst,          // (N, Hout, Wout) contiguous
    const int N, const int Hin, const int Win,
    const int Hout, const int Wout, const int factor
) {
    long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long total = (long long)N * Hout * Wout;
    if (idx >= total) return;

    int n  = (int)(idx / ((long long)Hout * Wout));
    int rem = (int)(idx % ((long long)Hout * Wout));
    int oy = rem / Wout;
    int ox = rem % Wout;

    int sy = oy / factor;
    int sx = ox / factor;
    if (sy >= Hin) sy = Hin - 1;
    if (sx >= Win) sx = Win - 1;

    dst[idx] = src[(long long)n * Hin * Win + sy * Win + sx];
}
"""

_kernel_cache: dict[tuple[int, str], object] = {}


def _get_kernel(device: int, name: str):
    key = (device, name)
    if key not in _kernel_cache:
        src = {"batch_bicubic_upsample": _BICUBIC_KERNEL_SRC,
               "batch_nn_upsample": _NN_KERNEL_SRC}[name]
        _kernel_cache[key] = cp.RawKernel(src, name)
    return _kernel_cache[key]


def batch_bicubic_upsample(rois_gpu: "cp.ndarray", factor: int) -> "cp.ndarray":
    """Upsample (N, H, W) batch by *factor* using Catmull-Rom bicubic.

    Parameters
    ----------
    rois_gpu : (N, Hin, Win) CuPy float64 array, contiguous
    factor : int upsampling factor

    Returns
    -------
    (N, Hin*factor, Win*factor) CuPy float64 array
    """
    N, Hin, Win = rois_gpu.shape
    Hout, Wout = Hin * factor, Win * factor
    dst = cp.empty((N, Hout, Wout), dtype=cp.float64)

    total = N * Hout * Wout
    block = 256
    grid = (total + block - 1) // block

    device = int(rois_gpu.device.id)
    kernel = _get_kernel(device, "batch_bicubic_upsample")
    kernel(
        (grid,), (block,),
        (cp.ascontiguousarray(rois_gpu), dst,
         np.int32(N), np.int32(Hin), np.int32(Win),
         np.int32(Hout), np.int32(Wout), np.int32(factor)),
    )
    return dst


def batch_nn_upsample(rois_gpu: "cp.ndarray", factor: int) -> "cp.ndarray":
    """Upsample (N, H, W) batch by *factor* using nearest-neighbor.

    Parameters
    ----------
    rois_gpu : (N, Hin, Win) CuPy float64 array, contiguous
    factor : int upsampling factor

    Returns
    -------
    (N, Hin*factor, Win*factor) CuPy float64 array
    """
    N, Hin, Win = rois_gpu.shape
    Hout, Wout = Hin * factor, Win * factor
    dst = cp.empty((N, Hout, Wout), dtype=cp.float64)

    total = N * Hout * Wout
    block = 256
    grid = (total + block - 1) // block

    device = int(rois_gpu.device.id)
    kernel = _get_kernel(device, "batch_nn_upsample")
    kernel(
        (grid,), (block,),
        (cp.ascontiguousarray(rois_gpu), dst,
         np.int32(N), np.int32(Hin), np.int32(Win),
         np.int32(Hout), np.int32(Wout), np.int32(factor)),
    )
    return dst
