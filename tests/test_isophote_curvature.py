import numpy as np
import pytest

def _has_cupy():
    try:
        import cupy
        return True
    except Exception:
        return False

skipno_gpu = pytest.mark.skipif(not _has_cupy(), reason="CuPy not available")

def _make_gaussian_blob(cx, cy, sigma, shape):
    H, W = shape
    yy, xx = np.mgrid[:H, :W]
    return np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2)).astype(np.float64)

@skipno_gpu
def test_isophote_curvature_isolated_blob():
    from subpx._gpu.centers import _isophote_curvature_batch_gpu
    true_cx, true_cy = 7.3, 6.7
    blob = _make_gaussian_blob(true_cx, true_cy, sigma=1.5, shape=(15, 15))
    rois = np.stack([blob])
    masks = np.ones_like(rois, dtype=bool)
    centers, spreads = _isophote_curvature_batch_gpu(rois, masks, upsample_factor=4, device=0)
    err = np.sqrt((centers[0, 0] - true_cx)**2 + (centers[0, 1] - true_cy)**2)
    assert err < 0.15, f"Isophote center error {err:.4f} px"

@skipno_gpu
def test_isophote_curvature_spread_low_for_symmetric():
    from subpx._gpu.centers import _isophote_curvature_batch_gpu
    blob = _make_gaussian_blob(7.0, 7.0, sigma=1.5, shape=(15, 15))
    rois = np.stack([blob])
    masks = np.ones_like(rois, dtype=bool)
    _, spreads = _isophote_curvature_batch_gpu(rois, masks, upsample_factor=4, device=0)
    assert spreads[0] < 1.0, f"Spread {spreads[0]:.4f} too high for symmetric blob"

@skipno_gpu
def test_isophote_agrees_with_radial_symmetry():
    from subpx._gpu.centers import _radial_symmetry_batch_gpu, _isophote_curvature_batch_gpu
    true_cx, true_cy = 7.3, 6.7
    blob = _make_gaussian_blob(true_cx, true_cy, sigma=1.5, shape=(15, 15))
    rois = np.stack([blob])
    masks = np.ones_like(rois, dtype=bool)
    rs_centers, _ = _radial_symmetry_batch_gpu(rois, masks, upsample_factor=4, device=0)
    ic_centers, _ = _isophote_curvature_batch_gpu(rois, masks, upsample_factor=4, device=0)
    dist = np.sqrt(np.sum((rs_centers - ic_centers)**2))
    assert dist < 0.15, f"Methods disagree by {dist:.4f} px"

@skipno_gpu
def test_isophote_curvature_batch():
    from subpx._gpu.centers import _isophote_curvature_batch_gpu
    rng = np.random.RandomState(42)
    N = 5
    rois = np.stack([
        _make_gaussian_blob(7 + rng.uniform(-1, 1), 7 + rng.uniform(-1, 1), 1.2, (15, 15))
        for _ in range(N)
    ])
    masks = np.ones_like(rois, dtype=bool)
    centers, spreads = _isophote_curvature_batch_gpu(rois, masks, upsample_factor=4, device=0)
    assert centers.shape == (N, 2)
    assert spreads.shape == (N,)
    assert np.all(np.isfinite(centers))
