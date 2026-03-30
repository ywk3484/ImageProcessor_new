import numpy as np
import pytest

def _has_cupy():
    try:
        import cupy
        return True
    except Exception:
        return False

skipno_gpu = pytest.mark.skipif(not _has_cupy(), reason="CuPy not available")


@skipno_gpu
def test_bicubic_upsample_f32_matches_f64():
    """Float32 bicubic upsample must match float64 within 1e-4."""
    import cupy as cp
    from subpx._gpu.upsample import batch_bicubic_upsample, batch_bicubic_upsample_f32

    rng = np.random.RandomState(42)
    rois = rng.rand(5, 10, 12).astype(np.float64)
    factor = 4

    with cp.cuda.Device(0):
        ref = batch_bicubic_upsample(cp.asarray(rois, dtype=cp.float64), factor)
        out = batch_bicubic_upsample_f32(cp.asarray(rois, dtype=cp.float32), factor)
        diff = float(cp.max(cp.abs(ref - out.astype(cp.float64))))
    assert diff < 1e-4, f"f32 vs f64 max diff: {diff}"


@skipno_gpu
def test_nn_upsample_f32_matches_f64():
    """Float32 NN upsample must match float64 exactly (no interpolation)."""
    import cupy as cp
    from subpx._gpu.upsample import batch_nn_upsample, batch_nn_upsample_f32

    rng = np.random.RandomState(42)
    rois = rng.rand(5, 8, 10).astype(np.float64)
    factor = 4

    with cp.cuda.Device(0):
        ref = batch_nn_upsample(cp.asarray(rois, dtype=cp.float64), factor)
        out = batch_nn_upsample_f32(cp.asarray(rois, dtype=cp.float32), factor)
        diff = float(cp.max(cp.abs(ref - out.astype(cp.float64))))
    assert diff < 1e-6, f"f32 NN vs f64 max diff: {diff}"
