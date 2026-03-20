from __future__ import annotations

import numpy as np

from ..types import ShiftResult
from ..backends import gpu_device

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None


def _require_cupy():
    if cp is None:
        raise RuntimeError("CuPy is required for GPU phase cross correlation.")


def _parabolic_offset_1d(y_m1, y_0, y_p1):
    denom = y_m1 - 2 * y_0 + y_p1
    if abs(float(denom)) < 1e-12:
        return 0.0
    return 0.5 * float(y_m1 - y_p1) / float(denom)


def estimate_shift_gpu(ref: np.ndarray, mov: np.ndarray, *, upsample_factor: int = 20, device: int = 0, use_float64: bool = True) -> ShiftResult:
    """GPU phase cross-correlation with parabolic sub-pixel refinement.

    .. warning::
       *upsample_factor* is accepted for API compatibility but is **not used**.
       Sub-pixel precision comes from a parabolic peak fit (~0.1 px typical),
       not the iterative DFT refinement used by the CPU backend.
    """
    _require_cupy()
    import warnings
    if upsample_factor > 1:
        warnings.warn(
            "GPU estimate_shift ignores upsample_factor; sub-pixel precision "
            "comes from parabolic peak fit (~0.1 px). Use backend='cpu' for "
            "iterative DFT refinement.",
            stacklevel=2,
        )
    dtype = cp.float64 if use_float64 else cp.float32

    with gpu_device(device):
        a = cp.asarray(ref, dtype=dtype)
        b = cp.asarray(mov, dtype=dtype)
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("estimate_shift(..., backend='gpu') currently expects 2D arrays.")
        if a.shape != b.shape:
            raise ValueError("ref and mov must have the same shape for GPU phase correlation.")

        Fa = cp.fft.fftn(a)
        Fb = cp.fft.fftn(b)
        cps = Fa * cp.conj(Fb)
        cps /= cp.maximum(cp.abs(cps), dtype(1e-12))
        cc = cp.fft.ifftn(cps)
        mag = cp.abs(cc)

        peak_flat = int(cp.asnumpy(cp.argmax(mag)))
        peak = np.unravel_index(peak_flat, mag.shape)
        peak_y, peak_x = int(peak[0]), int(peak[1])
        H, W = mag.shape

        shift_y = float(peak_y)
        shift_x = float(peak_x)
        if shift_y > H // 2:
            shift_y -= H
        if shift_x > W // 2:
            shift_x -= W

        mag_np = cp.asnumpy(mag)
        y0 = peak_y
        x0 = peak_x
        ym1 = (y0 - 1) % H
        yp1 = (y0 + 1) % H
        xm1 = (x0 - 1) % W
        xp1 = (x0 + 1) % W

        dy = _parabolic_offset_1d(mag_np[ym1, x0], mag_np[y0, x0], mag_np[yp1, x0])
        dx = _parabolic_offset_1d(mag_np[y0, xm1], mag_np[y0, x0], mag_np[y0, xp1])

        shift = np.array([shift_y + dy, shift_x + dx], dtype=np.float64)

    return ShiftResult(
        shift_yx=shift,
        error=None,
        phasediff=None,
        backend="gpu",
        method="phase_xcorr",
        meta={
            "upsample_factor_requested": int(upsample_factor),
            "subpixel_method": "parabolic_peak",
            "implementation": "cupy_fft_cross_power_spectrum",
            "device": int(device),
            "use_float64": bool(use_float64),
        },
    )
