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


@skipno_gpu
def test_detect_centers_radial_symmetry_integration():
    """Full pipeline: detect_centers with refine='radial_symmetry'."""
    from subpx.centers import detect_centers
    img = np.zeros((64, 64), dtype=np.uint8)
    # Two small blobs
    for cy, cx in [(15, 15), (15, 45)]:
        yy, xx = np.mgrid[:64, :64]
        img += (200 * np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2*2.0**2))).astype(np.uint8)
    res = detect_centers(
        img, backend="gpu", refine="radial_symmetry",
        area_min=4, area_max=200, upsample_factor=4,
    )
    assert res.centers_xy.shape[0] == 2
    assert "radial_symmetry_residual" in res.meta
    assert "voronoi_cell_area" in res.meta
    assert res.meta["upsample_factor"] == 4


def test_detect_centers_radial_symmetry_cpu_raises():
    """CPU backend must raise NotImplementedError (no GPU needed for this test)."""
    from subpx.centers import detect_centers
    img = np.zeros((32, 32), dtype=np.uint8)
    img[10:14, 10:14] = 200
    with pytest.raises(NotImplementedError):
        detect_centers(img, backend="cpu", refine="radial_symmetry")


@skipno_gpu
def test_detect_centers_tiled_radial_symmetry():
    """Tiled detection should work for radial_symmetry method."""
    from subpx.centers import detect_centers_tiled
    # Two blobs: one in top half, one in bottom half
    img = np.zeros((128, 64), dtype=np.uint8)
    img[20:26, 20:26] = 200
    img[80:86, 40:46] = 200
    res = detect_centers_tiled(
        img, backend="gpu", refine="radial_symmetry",
        tile_h=64, overlap=16, area_min=4, area_max=100,
    )
    assert res.centers_xy.shape[0] == 2
    # Verify both blobs detected (one near y=23, one near y=83)
    ys = sorted(res.centers_xy[:, 1])
    assert 15 < ys[0] < 30
    assert 75 < ys[1] < 90


def test_detect_centers_tiled_global_otsu_radial_symmetry_raises():
    """tiled_global_otsu must also raise NotImplementedError."""
    from subpx.centers import detect_centers_tiled_global_otsu
    img = np.zeros((64, 64), dtype=np.uint8)
    img[10:14, 10:14] = 200
    with pytest.raises(NotImplementedError):
        detect_centers_tiled_global_otsu(img, backend="gpu", refine="radial_symmetry")


@skipno_gpu
def test_radial_symmetry_no_neighbor_bias():
    """Closely-packed blobs: radial symmetry should not have neighbor-induced bias.

    Create two Gaussian blobs separated by only 2*sigma. Without Voronoi
    partitioning, a centroid method would be pulled toward the neighbor.
    With Voronoi partitioning + radial symmetry, the center should be unbiased.
    """
    from subpx.centers import detect_centers
    sigma = 2.0
    sep = 2.5 * sigma  # 5 px separation — closely packed
    cx1, cy1 = 20.0, 20.0
    cx2, cy2 = 20.0 + sep, 20.0

    img = np.zeros((40, 50), dtype=np.float64)
    yy, xx = np.mgrid[:40, :50]
    img += 200 * np.exp(-((xx - cx1)**2 + (yy - cy1)**2) / (2 * sigma**2))
    img += 200 * np.exp(-((xx - cx2)**2 + (yy - cy2)**2) / (2 * sigma**2))
    img = np.clip(img, 0, 255).astype(np.uint8)

    res = detect_centers(
        img, backend="gpu", refine="radial_symmetry",
        area_min=4, area_max=500, upsample_factor=4,
    )
    assert res.centers_xy.shape[0] == 2

    # Sort by x to identify blob 1 and blob 2
    pts = res.centers_xy[np.argsort(res.centers_xy[:, 0])]

    err1 = np.sqrt((pts[0, 0] - cx1)**2 + (pts[0, 1] - cy1)**2)
    err2 = np.sqrt((pts[1, 0] - cx2)**2 + (pts[1, 1] - cy2)**2)

    # The key assertion: no systematic bias toward the neighbor
    # Both errors should be small (< 0.3 px) despite close packing
    assert err1 < 0.3, f"Blob 1 error {err1:.4f} px — possible neighbor bias"
    assert err2 < 0.3, f"Blob 2 error {err2:.4f} px — possible neighbor bias"


def test_segment_binary_triangle_threshold():
    """_segment_binary accepts threshold='triangle' and produces a valid binary mask."""
    from subpx._cpu.centers import _segment_binary

    # Create image with faint blobs on dark background
    img = np.zeros((64, 64), dtype=np.uint8)
    # Bright blob
    yy, xx = np.mgrid[:64, :64]
    img += (100 * np.exp(-((xx - 32)**2 + (yy - 32)**2) / (2 * 2.0**2))).astype(np.uint8)

    g, bw = _segment_binary(img, threshold="triangle")
    assert bw.shape == (64, 64)
    assert bw.dtype == np.uint8
    assert bw.max() == 255  # at least some foreground pixels


@skipno_gpu
def test_radial_symmetry_rejects_out_of_bounds_center():
    """A heavily asymmetric ROI should produce an out-of-bounds fit that gets rejected (NaN)."""
    from subpx._gpu.centers import _radial_symmetry_batch_gpu

    # Create a blob at the right edge of a 15x15 ROI — center is outside
    # the left portion, simulating a clipped edge blob
    blob = _make_gaussian_blob(14.0, 7.0, sigma=1.5, shape=(15, 15))
    # Mask out right half to simulate asymmetric clipping
    mask = np.ones((15, 15), dtype=bool)
    mask[:, 10:] = False

    rois = np.stack([blob])
    masks = np.stack([mask])
    centers, residuals = _radial_symmetry_batch_gpu(rois, masks, upsample_factor=4, device=0)

    # The fit should be NaN (rejected) because the true center is outside
    # the valid masked region
    assert np.isnan(centers[0, 0]) or np.isnan(centers[0, 1]), (
        f"Expected NaN for out-of-bounds fit, got center ({centers[0, 0]:.2f}, {centers[0, 1]:.2f})"
    )


@skipno_gpu
def test_radial_symmetry_rejects_edge_blobs():
    """Blobs at the image boundary should be rejected by quality gates.

    Creates an image with one center blob (should be detected) and four
    edge blobs (one on each side, partially outside image). The edge blobs
    should be rejected because their asymmetric ROIs produce poor fits.
    """
    from subpx.centers import detect_centers

    img = np.zeros((64, 64), dtype=np.uint8)
    yy, xx = np.mgrid[:64, :64]

    # Center blob — should survive
    img += (200 * np.exp(-((xx - 32)**2 + (yy - 32)**2) / (2 * 2.0**2))).astype(np.uint8)
    # Edge blobs — placed so bbox doesn't touch boundary but padded ROI is clipped
    # Blob near left edge (x=2)
    img += (200 * np.exp(-((xx - 2)**2 + (yy - 32)**2) / (2 * 2.0**2))).astype(np.uint8)
    # Blob near top edge (y=2)
    img += (200 * np.exp(-((xx - 32)**2 + (yy - 2)**2) / (2 * 2.0**2))).astype(np.uint8)

    res = detect_centers(
        img, backend="gpu", refine="radial_symmetry",
        area_min=4, area_max=200, upsample_factor=4, threshold="triangle",
    )
    # Only the center blob should survive quality gates.
    # The edge blobs may be detected as connected components but should
    # be filtered by the boundary check (bbox touches edge) or quality gates.
    # At minimum, the center blob should be in the results.
    assert res.centers_xy.shape[0] >= 1

    # Check that center blob is detected near (32, 32)
    dists_to_center = np.sqrt(np.sum((res.centers_xy - [32, 32])**2, axis=1))
    assert np.min(dists_to_center) < 1.0, "Center blob not detected"

    # Edge blobs should NOT be in the results
    for edge_pos in [[2, 32], [32, 2]]:
        dists = np.sqrt(np.sum((res.centers_xy - edge_pos)**2, axis=1))
        if len(dists) > 0:
            assert np.min(dists) > 3.0, (
                f"Edge blob near {edge_pos} should have been rejected, "
                f"but found center at distance {np.min(dists):.2f} px"
            )


def test_triangle_detects_more_faint_blobs_than_otsu():
    """Triangle threshold captures faint blobs that Otsu misses.

    Creates a grid of blobs with varying brightness on a large dark background.
    The faintest blobs should be detected by Triangle but missed by Otsu
    because Otsu's threshold is pulled too high by the dominant background.
    """
    from subpx._cpu.centers import _segment_binary
    from subpx._cpu.components import connected_components_stats_cpu

    img = np.zeros((200, 200), dtype=np.uint8)
    yy, xx = np.mgrid[:200, :200]
    # 4x4 grid of blobs with decreasing brightness
    positions = [(30 + 40*i, 30 + 40*j) for i in range(4) for j in range(4)]
    brightnesses = [200, 180, 150, 120, 100, 80, 70, 60,
                    55, 50, 45, 40, 35, 30, 25, 20]
    for (cx, cy), brightness in zip(positions, brightnesses):
        img += (brightness * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 1.5**2))).astype(np.uint8)

    _, bw_otsu = _segment_binary(img, threshold="otsu")
    _, bw_tri = _segment_binary(img, threshold="triangle")

    comp_otsu = connected_components_stats_cpu(bw_otsu, connectivity=8)
    comp_tri = connected_components_stats_cpu(bw_tri, connectivity=8)
    # num_labels includes background (label 0), so subtract 1
    n_otsu = comp_otsu.num_labels - 1
    n_tri = comp_tri.num_labels - 1

    assert n_tri > n_otsu, (
        f"Triangle ({n_tri} blobs) should detect more blobs than Otsu ({n_otsu} blobs)"
    )
