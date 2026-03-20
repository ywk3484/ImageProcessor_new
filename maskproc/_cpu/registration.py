from __future__ import annotations

import numpy as np

from ..types import ShiftResult

try:
    from skimage.registration import phase_cross_correlation  # type: ignore
except Exception:  # pragma: no cover
    phase_cross_correlation = None


def estimate_shift_cpu(ref: np.ndarray, mov: np.ndarray, *, upsample_factor: int = 20) -> ShiftResult:
    if phase_cross_correlation is None:
        raise RuntimeError("scikit-image is required for CPU phase cross correlation.")

    shift, error, phasediff = phase_cross_correlation(
        np.asarray(ref),
        np.asarray(mov),
        upsample_factor=int(upsample_factor),
    )
    return ShiftResult(
        shift_yx=np.asarray(shift, dtype=np.float64),
        error=float(error) if np.isscalar(error) else None,
        phasediff=float(phasediff) if np.isscalar(phasediff) else None,
        backend="cpu",
        method="phase_xcorr",
        meta={"upsample_factor": int(upsample_factor), "implementation": "skimage.phase_cross_correlation"},
    )
