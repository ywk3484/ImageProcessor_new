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


@skipno_gpu
def test_radial_symmetry_batched_matches_unbatched():
    """Batched processing (gpu_batch=3) must produce same results as full batch."""
    from subpx._gpu.centers import _radial_symmetry_batch_gpu

    rng = np.random.RandomState(42)
    N = 10
    size = 11
    rois = []
    for _ in range(N):
        cx = size / 2 + rng.uniform(-1, 1)
        cy = size / 2 + rng.uniform(-1, 1)
        yy, xx = np.mgrid[:size, :size]
        blob = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 1.2**2))
        rois.append(blob)
    rois = np.stack(rois)
    masks = np.ones_like(rois, dtype=bool)

    # Full batch (default gpu_batch is large enough for N=10)
    c_full, r_full = _radial_symmetry_batch_gpu(
        rois, masks, upsample_factor=4, device=0, gpu_batch=10000,
    )
    # Small batches
    c_batched, r_batched = _radial_symmetry_batch_gpu(
        rois, masks, upsample_factor=4, device=0, gpu_batch=3,
    )
    # Results must be identical (same computation, just chunked)
    np.testing.assert_allclose(c_full, c_batched, atol=1e-10)
    np.testing.assert_allclose(r_full, r_batched, atol=1e-10)


@skipno_gpu
def test_isophote_batched_matches_unbatched():
    """Batched isophote curvature (gpu_batch=3) matches full batch."""
    from subpx._gpu.centers import _isophote_curvature_batch_gpu

    rng = np.random.RandomState(42)
    N = 10
    size = 15
    rois = []
    for _ in range(N):
        cx = size / 2 + rng.uniform(-1, 1)
        cy = size / 2 + rng.uniform(-1, 1)
        yy, xx = np.mgrid[:size, :size]
        blob = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 1.5**2))
        rois.append(blob)
    rois = np.stack(rois)
    masks = np.ones_like(rois, dtype=bool)

    c_full, s_full = _isophote_curvature_batch_gpu(
        rois, masks, upsample_factor=4, device=0, gpu_batch=10000,
    )
    c_batched, s_batched = _isophote_curvature_batch_gpu(
        rois, masks, upsample_factor=4, device=0, gpu_batch=3,
    )
    np.testing.assert_allclose(c_full, c_batched, atol=1e-10)
    np.testing.assert_allclose(s_full, s_batched, atol=1e-10)


@skipno_gpu
def test_detect_single_tile_gpu_exists():
    """_detect_single_tile_gpu must be importable and produce centers."""
    from subpx._gpu.centers import _detect_single_tile_gpu

    img = np.zeros((64, 64), dtype=np.uint8)
    img[10:14, 10:14] = 200
    img[30:34, 40:44] = 200

    centers, meta = _detect_single_tile_gpu(
        img, threshold="triangle", invert=False,
        area_min=4, area_max=100, morph_open=0, morph_close=0,
        pad=3, refine="weighted", connectivity=8,
        gpu_batch=4096, device=0, use_float64=True,
        components_backend="cpu", small_feature_max=12.0,
        upsample_factor=4, pre_threshold=None,
    )
    assert centers.shape == (2, 2)
    assert isinstance(meta, dict)


@skipno_gpu
def test_detect_centers_gpu_tiled_basic():
    """detect_centers_gpu with tile_h should produce centers from a tall image."""
    from subpx._gpu.centers import detect_centers_gpu
    from subpx.types import CenterResult

    def _make_dot_grid(rows, cols, spacing, dot_size, margin):
        H = 2 * margin + (rows - 1) * spacing + dot_size
        W = 2 * margin + (cols - 1) * spacing + dot_size
        img = np.zeros((H, W), dtype=np.uint8)
        expected = []
        for r in range(rows):
            for c in range(cols):
                y0 = margin + r * spacing
                x0 = margin + c * spacing
                img[y0:y0+dot_size, x0:x0+dot_size] = 255
                expected.append([x0 + (dot_size-1)/2.0, y0 + (dot_size-1)/2.0])
        return img, np.array(expected, dtype=np.float64)

    img, expected = _make_dot_grid(rows=10, cols=4, spacing=20, dot_size=3, margin=10)
    H = img.shape[0]
    result = detect_centers_gpu(
        img, tile_h=H // 3, overlap=30, threshold_mode="global",
        area_min=1, area_max=50, refine="weighted",
    )
    assert isinstance(result, CenterResult)
    assert abs(result.centers_xy.shape[0] - expected.shape[0]) <= 2


@skipno_gpu
def test_detect_centers_gpu_tiled_threshold_modes():
    """Both 'global' and 'per_tile' threshold modes should work."""
    from subpx._gpu.centers import detect_centers_gpu

    img = np.zeros((200, 100), dtype=np.uint8)
    img[20:24, 20:24] = 200
    img[120:124, 60:64] = 200

    for mode in ("global", "per_tile"):
        result = detect_centers_gpu(
            img, tile_h=100, overlap=20, threshold_mode=mode,
            area_min=4, area_max=100, refine="weighted",
        )
        assert result.centers_xy.shape[0] == 2, f"mode={mode}: expected 2 centers"


def test_detect_centers_cpu_tiled():
    """CPU detect_centers_cpu with tile_h should tile and dedupe."""
    from subpx._cpu.centers import detect_centers_cpu

    img = np.zeros((200, 100), dtype=np.uint8)
    img[20:24, 20:24] = 200
    img[120:124, 60:64] = 200

    result = detect_centers_cpu(
        img, tile_h=100, overlap=20, dedupe_eps=1.5,
        area_min=4, area_max=100, refine="weighted",
    )
    assert result.centers_xy.shape[0] == 2
