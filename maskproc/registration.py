"""Image registration utilities.

This module exposes a stable registration API while dispatching to CPU or GPU
implementations behind the scenes.
"""

from __future__ import annotations

import numpy as np
from .backends import resolve_backend, to_numpy
from ._cpu.registration import estimate_shift_cpu
from ._gpu.registration import estimate_shift_gpu

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


def estimate_shift(
    ref: np.ndarray,
    mov: np.ndarray,
    *,
    method: str = "phase_xcorr",
    backend: str = "auto",
    upsample_factor: int = 20,
    device: int = 0,
    use_float64: bool = True,
):
    """Estimate sub-pixel shift between *ref* and *mov* images.

    Returns a :class:`ShiftResult` whose ``shift_yx`` is in **[y, x] order**
    (row, column), which differs from the project-wide [x, y] convention used
    for point arrays.  This matches the NumPy/image convention where axis-0 is
    the row (y) direction.

    Note: The GPU backend uses a parabolic peak fit for sub-pixel refinement and
    does not honour *upsample_factor*.  The CPU backend (scikit-image) performs
    iterative DFT refinement controlled by *upsample_factor*.
    """
    if method != "phase_xcorr":
        raise ValueError("Currently supported method: 'phase_xcorr'")

    b = resolve_backend(backend)
    ref_np = to_numpy(ref)
    mov_np = to_numpy(mov)

    if b == "gpu":
        return estimate_shift_gpu(
            ref_np,
            mov_np,
            upsample_factor=upsample_factor,
            device=device,
            use_float64=use_float64,
        )
    return estimate_shift_cpu(ref_np, mov_np, upsample_factor=upsample_factor)


def apply_shift(image: np.ndarray, shift_yx, *, order: int = 1, cval: float = 0.0) -> np.ndarray:
    img = np.asarray(image)
    sy, sx = map(float, shift_yx)

    if cv2 is not None:
        M = np.array([[1.0, 0.0, sx], [0.0, 1.0, sy]], dtype=np.float32)
        interp = cv2.INTER_LINEAR if int(order) == 1 else cv2.INTER_NEAREST
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=interp, borderValue=float(cval))

    out = np.full_like(img, fill_value=cval)
    iy, ix = int(round(sy)), int(round(sx))
    y0_src = max(0, -iy)
    y1_src = min(img.shape[0], img.shape[0] - iy)
    x0_src = max(0, -ix)
    x1_src = min(img.shape[1], img.shape[1] - ix)
    y0_dst = max(0, iy)
    y1_dst = y0_dst + (y1_src - y0_src)
    x0_dst = max(0, ix)
    x1_dst = x0_dst + (x1_src - x0_src)
    out[y0_dst:y1_dst, x0_dst:x1_dst] = img[y0_src:y1_src, x0_src:x1_src]
    return out


def crop_overlap(ref: np.ndarray, mov: np.ndarray, shift_yx) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(ref)
    mov = np.asarray(mov)
    sy, sx = map(int, np.round(shift_yx))

    y0 = max(0, sy)
    y1 = min(ref.shape[0], mov.shape[0] + sy)
    x0 = max(0, sx)
    x1 = min(ref.shape[1], mov.shape[1] + sx)

    ref_crop = ref[y0:y1, x0:x1]
    mov_crop = mov[y0 - sy:y1 - sy, x0 - sx:x1 - sx]
    return ref_crop, mov_crop
