"""Optimized half-area calculations preserve regions and numerical values to roundoff."""

import numpy as np
import pytest

from subpx._cpu.invariant_cd import _grow, _nearest_pixels, _zoom_operators


def _loop(region_seed_above, top, xp):
    """Reference flood fill with a safe path-length bound and early termination."""
    above = region_seed_above
    k, H, W = above.shape
    region = xp.zeros((k, H * W), dtype=bool)
    region[xp.arange(k), top] = True
    region = region.reshape(k, H, W) & above
    for step in range(H * W):
        grown = _grow(region) & above
        if step % 8 == 7 and bool((grown == region).all()):
            break
        region = grown
    return region


def test_mask_enlargement_by_pixel_copy_equals_the_order0_zoom_product():
    rng = np.random.default_rng(0)
    for h, w in ((13, 13), (25, 25), (19, 47), (1, 5)):
        masks = rng.random((20, h, w)) < 0.5
        ny, nx = _zoom_operators(h, 4)[1], _zoom_operators(w, 4)[1]
        product = (ny[None] @ masks.astype(float) @ nx.T[None]) > 0.5
        gathered = masks[:, _nearest_pixels(h, 4)[:, None], _nearest_pixels(w, 4)[None, :]]
        np.testing.assert_array_equal(gathered, product)


def _snake(size=21):
    snake = np.zeros((size, size), bool)
    snake[::2] = True
    for y in range(1, size, 2):
        snake[y, size - 1 if (y // 2) % 2 == 0 else 0] = True
    return snake


def test_region_kernel_equals_complete_flood_fill():
    from subpx.backends import cp

    if cp is None:
        pytest.skip("CuPy not available")
    from subpx._gpu.regions import grow_regions

    rng = np.random.default_rng(1)
    for k, H, W in ((200, 100, 100), (100, 37, 53), (30, 188, 188), (64, 8, 70)):
        for density in (0.45, 0.6, 0.75):
            above = cp.asarray(rng.random((k, H, W)) < density)
            top = cp.asarray(rng.integers(0, H * W, k))
            np.testing.assert_array_equal(cp.asnumpy(grow_regions(above, top, H * W)), cp.asnumpy(_loop(above, top, cp)))
    # A snake can require more steps than the height plus width of its box.
    above = cp.asarray(np.repeat(_snake()[None], 3, axis=0))
    top = cp.asarray(np.array([0, 5, 21 * 21 - 1]))
    grown = cp.asnumpy(grow_regions(above, top, 21 * 21))
    np.testing.assert_array_equal(grown, cp.asnumpy(_loop(above, top, cp)))
    np.testing.assert_array_equal(grown[0], _snake())


def test_top_mean_kernel_equals_partition_mean_to_roundoff():
    from subpx.backends import cp

    if cp is None:
        pytest.skip("CuPy not available")
    from subpx._gpu.regions import top_mean

    rng = np.random.default_rng(2)
    for length in (64, 600, 10000, 35344):
        for n in (1, 16, 64):
            x = np.round(rng.normal(30, 10, (300, length)), 1)  # many ties
            x[rng.random(x.shape) < 0.4] = -np.inf               # outside the cell
            x[:3] = -np.inf
            x[3, :length - 2] = -np.inf                           # fewer finite values than n
            x[4, 7] = np.nan
            x[5] = 0.0
            x[5, ::2] = -0.0
            data = cp.asarray(x)
            old = cp.asnumpy(cp.partition(data, -n, axis=1)[:, -n:].mean(axis=1))
            new = cp.asnumpy(top_mean(data, n))
            # A contiguous reduction may add in a different order; this bound is a few FP64 ulps, not a CD tolerance.
            np.testing.assert_allclose(new, old, rtol=2e-15, atol=0, equal_nan=True, err_msg=f"{length=}, {n=}")


def test_default_region_growth_reaches_winding_component():
    from scipy.ndimage import label
    from subpx._cpu.invariant_cd import _connected_to_top

    snake = _snake(31)
    top = np.array([0])
    labels, _ = label(snake, structure=np.ones((3, 3)))
    expected = labels == labels.flat[top[0]]
    region = _connected_to_top(snake[None], top)[0]
    np.testing.assert_array_equal(region, expected)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_half_area_winding_shape_matches_connected_component_area(backend):
    from scipy.ndimage import label
    from subpx._cpu.invariant_cd import _half_area

    xp = np
    if backend == "gpu":
        from subpx.backends import cp
        if cp is None:
            pytest.skip("CuPy not available")
        xp = cp
    # Seven horizontal runs linked at alternate ends; the entire shape is inside the cell mask.
    cell = np.zeros((31, 31))
    for i, y in enumerate(range(3, 28, 4)):
        cell[y, 3:28] = 1
        if y < 27:
            cell[y:y + 5, 27 if i % 2 == 0 else 3] = 1
    cell[3, 3] = 1.01  # deterministic unique peak at one end of the path
    mask = np.ones_like(cell, dtype=bool)
    zoom = _zoom_operators(31, 4)[0]
    enlarged = zoom @ cell @ zoom.T
    above = enlarged > 0.5 * enlarged.max()
    labels, _ = label(above, structure=np.ones((3, 3)))
    expected_area = np.count_nonzero(labels == labels.flat[enlarged.argmax()]) / 16
    result = _half_area(cell[None], mask[None], np.zeros((1, 2)), 16, background="zero",
                        name="winding", outlines=False, xp=xp)
    assert result.status[0] == "ok"
    np.testing.assert_allclose(result.metrics["area_px2"], [expected_area], rtol=0, atol=0)
