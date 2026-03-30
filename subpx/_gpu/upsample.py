"""Batch bicubic and nearest-neighbor upsampling via CUDA RawKernels.

Processes (N, H, W) → (N, H*factor, W*factor) in a single kernel launch,
replacing per-ROI ``cndi.zoom`` loops that required 2N launches.

The bicubic kernel uses Keys' cubic (Catmull-Rom, a=-0.5), an interpolating
cubic that requires no pre-filter — unlike scipy's B-spline default.  Both
use the same edge-aligned coordinate mapping as ``scipy.ndimage.zoom``
(``output[0] → input[0]``, ``output[last] → input[last]``).
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

    // Edge-aligned coordinate mapping (matches scipy ndimage.zoom default):
    // output[0] -> input[0], output[last] -> input[last]
    double scale_y = (Hout > 1) ? ((double)Hin - 1.0) / ((double)Hout - 1.0) : 0.0;
    double scale_x = (Wout > 1) ? ((double)Win - 1.0) / ((double)Wout - 1.0) : 0.0;
    double sy = (double)oy * scale_y;
    double sx = (double)ox * scale_x;

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

    // Edge-aligned coordinate mapping matching scipy ndimage.zoom order=0
    double scale_y = (Hout > 1) ? ((double)Hin - 1.0) / ((double)Hout - 1.0) : 0.0;
    double scale_x = (Wout > 1) ? ((double)Win - 1.0) / ((double)Wout - 1.0) : 0.0;
    int sy = (int)((double)oy * scale_y + 0.5);
    int sx = (int)((double)ox * scale_x + 0.5);
    if (sy >= Hin) sy = Hin - 1;
    if (sx >= Win) sx = Win - 1;

    dst[idx] = src[(long long)n * Hin * Win + sy * Win + sx];
}
"""

# ── Batch bicubic upsample — float32 variant ────────────────────────────────
_BICUBIC_F32_KERNEL_SRC = r"""
extern "C" __global__
void batch_bicubic_upsample_f32(
    const float* __restrict__ src,    // (N, Hin, Win) contiguous
    float* __restrict__ dst,          // (N, Hout, Wout) contiguous
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

    // Edge-aligned coordinate mapping (matches scipy ndimage.zoom default):
    // output[0] -> input[0], output[last] -> input[last]
    float scale_y = (Hout > 1) ? ((float)Hin - 1.0f) / ((float)Hout - 1.0f) : 0.0f;
    float scale_x = (Wout > 1) ? ((float)Win - 1.0f) / ((float)Wout - 1.0f) : 0.0f;
    float sy = (float)oy * scale_y;
    float sx = (float)ox * scale_x;

    int iy = (int)floorf(sy);
    int ix = (int)floorf(sx);
    float fy = sy - (float)iy;
    float fx = sx - (float)ix;

    // Keys' cubic weights (a = -0.5)
    float wy[4], wx[4];
    {
        float t;
        // wy[0] = W(-1-fy) → W(1+fy)
        t = 1.0f + fy;
        wy[0] = ((-0.5f*t + 2.5f)*t - 4.0f)*t + 2.0f;
        // wy[1] = W(-fy)   → W(fy)
        t = fy;
        wy[1] = ((1.5f*t - 2.5f)*t)*t + 1.0f;
        // wy[2] = W(1-fy)
        t = 1.0f - fy;
        wy[2] = ((1.5f*t - 2.5f)*t)*t + 1.0f;
        // wy[3] = W(2-fy)
        t = 2.0f - fy;
        wy[3] = ((-0.5f*t + 2.5f)*t - 4.0f)*t + 2.0f;
    }
    {
        float t;
        t = 1.0f + fx;
        wx[0] = ((-0.5f*t + 2.5f)*t - 4.0f)*t + 2.0f;
        t = fx;
        wx[1] = ((1.5f*t - 2.5f)*t)*t + 1.0f;
        t = 1.0f - fx;
        wx[2] = ((1.5f*t - 2.5f)*t)*t + 1.0f;
        t = 2.0f - fx;
        wx[3] = ((-0.5f*t + 2.5f)*t - 4.0f)*t + 2.0f;
    }

    // Gather 4x4 neighborhood with clamped boundary
    const float* src_n = src + (long long)n * Hin * Win;
    float val = 0.0f;
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

# ── Batch nearest-neighbor upsample — float32 variant ───────────────────────
_NN_F32_KERNEL_SRC = r"""
extern "C" __global__
void batch_nn_upsample_f32(
    const float* __restrict__ src,    // (N, Hin, Win) contiguous
    float* __restrict__ dst,          // (N, Hout, Wout) contiguous
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

    // Edge-aligned coordinate mapping matching scipy ndimage.zoom order=0
    float scale_y = (Hout > 1) ? ((float)Hin - 1.0f) / ((float)Hout - 1.0f) : 0.0f;
    float scale_x = (Wout > 1) ? ((float)Win - 1.0f) / ((float)Wout - 1.0f) : 0.0f;
    int sy = (int)((float)oy * scale_y + 0.5f);
    int sx = (int)((float)ox * scale_x + 0.5f);
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
               "batch_nn_upsample": _NN_KERNEL_SRC,
               "batch_bicubic_upsample_f32": _BICUBIC_F32_KERNEL_SRC,
               "batch_nn_upsample_f32": _NN_F32_KERNEL_SRC}[name]
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


def batch_bicubic_upsample_f32(rois_gpu: "cp.ndarray", factor: int) -> "cp.ndarray":
    """Upsample (N, H, W) batch by *factor* using Catmull-Rom bicubic (float32).

    Parameters
    ----------
    rois_gpu : (N, Hin, Win) CuPy float32 array, contiguous
    factor : int upsampling factor

    Returns
    -------
    (N, Hin*factor, Win*factor) CuPy float32 array
    """
    N, Hin, Win = rois_gpu.shape
    Hout, Wout = Hin * factor, Win * factor
    dst = cp.empty((N, Hout, Wout), dtype=cp.float32)

    total = N * Hout * Wout
    block = 256
    grid = (total + block - 1) // block

    device = int(rois_gpu.device.id)
    kernel = _get_kernel(device, "batch_bicubic_upsample_f32")
    kernel(
        (grid,), (block,),
        (cp.ascontiguousarray(rois_gpu), dst,
         np.int32(N), np.int32(Hin), np.int32(Win),
         np.int32(Hout), np.int32(Wout), np.int32(factor)),
    )
    return dst


def batch_nn_upsample_f32(rois_gpu: "cp.ndarray", factor: int) -> "cp.ndarray":
    """Upsample (N, H, W) batch by *factor* using nearest-neighbor (float32).

    Parameters
    ----------
    rois_gpu : (N, Hin, Win) CuPy float32 array, contiguous
    factor : int upsampling factor

    Returns
    -------
    (N, Hin*factor, Win*factor) CuPy float32 array
    """
    N, Hin, Win = rois_gpu.shape
    Hout, Wout = Hin * factor, Win * factor
    dst = cp.empty((N, Hout, Wout), dtype=cp.float32)

    total = N * Hout * Wout
    block = 256
    grid = (total + block - 1) // block

    device = int(rois_gpu.device.id)
    kernel = _get_kernel(device, "batch_nn_upsample_f32")
    kernel(
        (grid,), (block,),
        (cp.ascontiguousarray(rois_gpu), dst,
         np.int32(N), np.int32(Hin), np.int32(Win),
         np.int32(Hout), np.int32(Wout), np.int32(factor)),
    )
    return dst
