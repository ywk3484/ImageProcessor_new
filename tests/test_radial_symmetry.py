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
    blob = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    return blob.astype(np.float64)

@skipno_gpu
def test_radial_symmetry_isolated_blob_accuracy():
    from subpx._gpu.centers import _radial_symmetry_batch_gpu
    true_cx, true_cy = 7.3, 6.7
    blob = _make_gaussian_blob(true_cx, true_cy, sigma=1.5, shape=(15, 15))
    rois = np.stack([blob])
    masks = np.ones_like(rois, dtype=bool)
    centers, residuals = _radial_symmetry_batch_gpu(rois, masks, upsample_factor=4, device=0)
    assert centers.shape == (1, 2)
    err = np.sqrt((centers[0, 0] - true_cx)**2 + (centers[0, 1] - true_cy)**2)
    assert err < 0.05, f"Center error {err:.4f} px exceeds 0.05 px"

@skipno_gpu
def test_radial_symmetry_subpixel_positions():
    from subpx._gpu.centers import _radial_symmetry_batch_gpu
    offsets = [0.1, 0.25, 0.5, 0.75, 0.9]
    errors = []
    for dx in offsets:
        cx, cy = 7.0 + dx, 7.0 + dx
        blob = _make_gaussian_blob(cx, cy, sigma=1.5, shape=(15, 15))
        rois = np.stack([blob])
        masks = np.ones_like(rois, dtype=bool)
        centers, _ = _radial_symmetry_batch_gpu(rois, masks, upsample_factor=4, device=0)
        err = np.sqrt((centers[0, 0] - cx)**2 + (centers[0, 1] - cy)**2)
        errors.append(err)
    assert max(errors) < 0.1, f"Pixel-locking detected: max error {max(errors):.4f}"
    assert np.std(errors) < 0.03, f"Pixel-locking bias: std of errors {np.std(errors):.4f}"

@skipno_gpu
def test_radial_symmetry_small_blob_3px():
    from subpx._gpu.centers import _radial_symmetry_batch_gpu
    true_cx, true_cy = 1.4, 1.6
    blob = _make_gaussian_blob(true_cx, true_cy, sigma=0.8, shape=(3, 3))
    rois = np.stack([blob])
    masks = np.ones_like(rois, dtype=bool)
    centers, residuals = _radial_symmetry_batch_gpu(rois, masks, upsample_factor=4, device=0)
    err = np.sqrt((centers[0, 0] - true_cx)**2 + (centers[0, 1] - true_cy)**2)
    assert err < 0.2, f"3px blob error {err:.4f} px exceeds 0.2 px"

@skipno_gpu
def test_radial_symmetry_batch():
    from subpx._gpu.centers import _radial_symmetry_batch_gpu
    rng = np.random.RandomState(42)
    N = 10
    size = 11
    rois = []
    true_centers = []
    for _ in range(N):
        cx = size / 2 + rng.uniform(-1, 1)
        cy = size / 2 + rng.uniform(-1, 1)
        blob = _make_gaussian_blob(cx, cy, sigma=1.2, shape=(size, size))
        rois.append(blob)
        true_centers.append([cx, cy])
    rois = np.stack(rois)
    masks = np.ones_like(rois, dtype=bool)
    true_centers = np.array(true_centers)
    centers, residuals = _radial_symmetry_batch_gpu(rois, masks, upsample_factor=4, device=0)
    assert centers.shape == (N, 2)
    assert residuals.shape == (N,)
    errors = np.sqrt(np.sum((centers - true_centers)**2, axis=1))
    assert np.all(errors < 0.1), f"Max batch error: {errors.max():.4f}"

@skipno_gpu
def test_radial_symmetry_voronoi_mask():
    from subpx._gpu.centers import _radial_symmetry_batch_gpu
    true_cx, true_cy = 7.3, 7.3
    blob = _make_gaussian_blob(true_cx, true_cy, sigma=1.5, shape=(15, 15))
    mask = np.ones((15, 15), dtype=bool)
    mask[:, 12:] = False
    rois = np.stack([blob])
    masks = np.stack([mask])
    centers, _ = _radial_symmetry_batch_gpu(rois, masks, upsample_factor=4, device=0)
    err = np.sqrt((centers[0, 0] - true_cx)**2 + (centers[0, 1] - true_cy)**2)
    assert err < 0.15, f"Masked ROI error {err:.4f} px"

@skipno_gpu
def test_radial_symmetry_no_upsample():
    from subpx._gpu.centers import _radial_symmetry_batch_gpu
    true_cx, true_cy = 7.3, 6.7
    blob = _make_gaussian_blob(true_cx, true_cy, sigma=1.5, shape=(15, 15))
    rois = np.stack([blob])
    masks = np.ones_like(rois, dtype=bool)
    centers, _ = _radial_symmetry_batch_gpu(rois, masks, upsample_factor=1, device=0)
    err = np.sqrt((centers[0, 0] - true_cx)**2 + (centers[0, 1] - true_cy)**2)
    assert err < 0.1, f"No-upsample error {err:.4f} px"
