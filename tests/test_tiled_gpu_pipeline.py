"""Tests for the dedicated tiled GPU pipeline."""
import numpy as np
import pytest

from subpx.types import CenterResult


def _make_dot_grid(rows=4, cols=4, spacing=20, dot_size=3, margin=10):
    """Create a synthetic image with a grid of bright dots on black background."""
    H = 2 * margin + (rows - 1) * spacing + dot_size
    W = 2 * margin + (cols - 1) * spacing + dot_size
    img = np.zeros((H, W), dtype=np.uint8)
    expected = []
    for r in range(rows):
        for c in range(cols):
            y0 = margin + r * spacing
            x0 = margin + c * spacing
            img[y0:y0 + dot_size, x0:x0 + dot_size] = 255
            expected.append([x0 + (dot_size - 1) / 2.0, y0 + (dot_size - 1) / 2.0])
    return img, np.array(expected, dtype=np.float64)


# --- GPU tests (skip if no CuPy) ---

try:
    import cupy as _cp
    HAS_CUPY = True
except Exception:
    HAS_CUPY = False

gpu = pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")


# --- CC RawKernel tests ---


@gpu
def test_cc_rawkernel_correctness():
    """RawKernel CC stats must match known expected values."""
    import cupy as cp
    from subpx._gpu.components import _connected_components_stats_gpu_core

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 255   # 10x10 block
    mask[50:60, 50:60] = 255   # 10x10 block
    with cp.cuda.Device(0):
        mask_gpu = cp.asarray(mask > 0)
        labels, stats, cents, num = _connected_components_stats_gpu_core(mask_gpu)
        assert num == 3  # bg + 2 components
        areas = cp.asnumpy(stats[1:, 4])
        assert set(areas.tolist()) == {100}  # both 10x10


@gpu
def test_cc_public_returns_numpy():
    """Public connected_components_stats_gpu must still return NumPy."""
    from subpx._gpu.components import connected_components_stats_gpu

    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    comp = connected_components_stats_gpu(mask)
    assert isinstance(comp.labels, np.ndarray)
    assert isinstance(comp.stats, np.ndarray)
    assert isinstance(comp.centroids, np.ndarray)


# --- Tiled pipeline tests ---


@gpu
def test_tiled_gpu_pipeline_basic():
    """Basic: small image, single tile covers everything."""
    from subpx._gpu.centers import _detect_centers_tiled_gpu

    img, expected = _make_dot_grid(rows=3, cols=3, spacing=15, dot_size=3, margin=10)
    result = _detect_centers_tiled_gpu(
        img,
        area_min=1,
        area_max=50,
        tile_h=8192,
        overlap=16,
        refine="edge_gradmoment",
    )
    assert isinstance(result, CenterResult)
    assert result.centers_xy.shape[1] == 2
    assert result.centers_xy.shape[0] == expected.shape[0]


@gpu
def test_tiled_gpu_pipeline_multi_tile():
    """Multi-tile: image taller than tile_h, tests tiling + band de-dup."""
    from subpx._gpu.centers import _detect_centers_tiled_gpu

    img, expected = _make_dot_grid(rows=10, cols=4, spacing=20, dot_size=3, margin=10)
    H = img.shape[0]
    tile_h = H // 3
    result = _detect_centers_tiled_gpu(
        img,
        area_min=1,
        area_max=50,
        tile_h=tile_h,
        overlap=30,
        refine="edge_gradmoment",
    )
    assert isinstance(result, CenterResult)
    assert result.centers_xy.shape[1] == 2
    assert abs(result.centers_xy.shape[0] - expected.shape[0]) <= 2


@gpu
def test_tiled_gpu_pipeline_empty_image():
    """All-black image should return zero centers."""
    from subpx._gpu.centers import _detect_centers_tiled_gpu

    img = np.zeros((100, 100), dtype=np.uint8)
    result = _detect_centers_tiled_gpu(img, area_min=1, area_max=50)
    assert result.centers_xy.shape == (0, 2)


@gpu
def test_tiled_gpu_pipeline_overlap_validation():
    """overlap >= tile_h should raise ValueError."""
    from subpx._gpu.centers import _detect_centers_tiled_gpu

    img = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match="overlap must be < tile_h"):
        _detect_centers_tiled_gpu(img, tile_h=64, overlap=64)


@gpu
def test_tiled_gpu_pipeline_unknown_refine():
    """Unknown refine method should raise ValueError."""
    from subpx._gpu.centers import _detect_centers_tiled_gpu

    img, _ = _make_dot_grid(rows=2, cols=2, spacing=15, dot_size=3, margin=10)
    with pytest.raises(ValueError, match="Unknown refine method"):
        _detect_centers_tiled_gpu(img, refine="nonexistent", area_min=1, area_max=50)


@gpu
def test_tiled_gpu_pipeline_edge_erf():
    """edge_erf refine should work in tiled pipeline."""
    from subpx._gpu.centers import _detect_centers_tiled_gpu

    img, expected = _make_dot_grid(rows=3, cols=3, spacing=15, dot_size=5, margin=10)
    result = _detect_centers_tiled_gpu(img, area_min=1, area_max=100, refine="edge_erf")
    assert isinstance(result, CenterResult)
    assert result.centers_xy.shape[0] == expected.shape[0]


@gpu
def test_tiled_gpu_pipeline_logquad():
    """logquad refine should work in tiled pipeline."""
    from subpx._gpu.centers import _detect_centers_tiled_gpu

    img, expected = _make_dot_grid(rows=3, cols=3, spacing=15, dot_size=3, margin=10)
    result = _detect_centers_tiled_gpu(img, area_min=1, area_max=50, refine="logquad")
    assert isinstance(result, CenterResult)
    assert result.centers_xy.shape[0] == expected.shape[0]


@gpu
def test_tiled_gpu_pipeline_weighted():
    from subpx._gpu.centers import _detect_centers_tiled_gpu
    img, expected = _make_dot_grid(rows=3, cols=3, spacing=15, dot_size=3, margin=10)
    result = _detect_centers_tiled_gpu(img, area_min=1, area_max=50, refine="weighted")
    assert isinstance(result, CenterResult)
    assert result.centers_xy.shape[0] == expected.shape[0]


@gpu
def test_tiled_gpu_pipeline_auto():
    from subpx._gpu.centers import _detect_centers_tiled_gpu
    img, expected = _make_dot_grid(rows=3, cols=3, spacing=15, dot_size=3, margin=10)
    result = _detect_centers_tiled_gpu(img, area_min=1, area_max=50, refine="auto")
    assert isinstance(result, CenterResult)
    assert result.centers_xy.shape[0] == expected.shape[0]


@gpu
def test_tiled_gpu_pipeline_auto_mixed_sizes():
    """auto should handle both small and large features."""
    from subpx._gpu.centers import _detect_centers_tiled_gpu
    H, W = 200, 200
    img = np.zeros((H, W), dtype=np.uint8)
    img[20:23, 20:23] = 255      # small 3x3
    img[20:23, 50:53] = 255      # small 3x3
    img[100:115, 100:115] = 255  # large 15x15
    img[100:115, 150:165] = 255  # large 15x15
    result = _detect_centers_tiled_gpu(
        img, area_min=1, area_max=300, refine="auto", small_feature_max=12.0,
    )
    assert isinstance(result, CenterResult)
    assert result.centers_xy.shape[0] == 4


@gpu
def test_tiled_gpu_pipeline_weighted_multi_tile():
    from subpx._gpu.centers import _detect_centers_tiled_gpu
    img, expected = _make_dot_grid(rows=10, cols=4, spacing=20, dot_size=3, margin=10)
    H = img.shape[0]
    result = _detect_centers_tiled_gpu(
        img, area_min=1, area_max=50, tile_h=H // 3, overlap=30, refine="weighted",
    )
    assert isinstance(result, CenterResult)
    assert abs(result.centers_xy.shape[0] - expected.shape[0]) <= 2


# --- Integration tests for detect_centers_tiled public API ---


@gpu
def test_detect_centers_tiled_dispatches_to_gpu_pipeline():
    """detect_centers_tiled(backend='gpu') should use the new tiled GPU pipeline."""
    from subpx import detect_centers_tiled

    img, expected = _make_dot_grid(rows=3, cols=3, spacing=15, dot_size=3, margin=10)
    result = detect_centers_tiled(
        img,
        backend="gpu",
        tile_h=8192,
        overlap=16,
        refine="edge_gradmoment",
        area_min=1,
        area_max=50,
    )
    assert isinstance(result, CenterResult)
    assert result.centers_xy.shape[1] == 2
    assert result.centers_xy.shape[0] == expected.shape[0]
    assert "tiled_gpu" in result.method


def test_detect_centers_tiled_cpu_still_works():
    """CPU path should be unchanged."""
    from subpx import detect_centers_tiled

    img, _ = _make_dot_grid(rows=3, cols=3, spacing=15, dot_size=3, margin=10)
    result = detect_centers_tiled(
        img,
        backend="cpu",
        tile_h=8192,
        overlap=16,
        refine="weighted",
        area_min=1,
        area_max=50,
    )
    assert isinstance(result, CenterResult)
    assert result.centers_xy.shape[1] == 2
    assert "tiled(cpu)" in result.method


def test_detect_centers_tiled_cpu_overlap_validation():
    """CPU path: overlap >= tile_h should raise ValueError."""
    from subpx import detect_centers_tiled

    img = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match="overlap must be < tile_h"):
        detect_centers_tiled(img, backend="cpu", tile_h=64, overlap=64)
