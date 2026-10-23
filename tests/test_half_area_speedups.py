"""The faster paths inside _half_area give the same arrays as the code they replaced, bit for bit."""

import numpy as np
import pytest

from subpx._cpu.invariant_cd import _grow, _nearest_pixels, _zoom_operators


def _loop(region_seed_above, top, xp):
    """The flood-fill loop _half_area used before (its result: H + W growth steps or the fixed point)."""
    above = region_seed_above
    k, H, W = above.shape
    region = xp.zeros((k, H * W), dtype=bool)
    region[xp.arange(k), top] = True
    region = region.reshape(k, H, W) & above
    for step in range(H + W):
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


def test_region_kernel_equals_the_flood_fill_loop_including_its_step_limit():
    from subpx.backends import cp

    if cp is None:
        pytest.skip("CuPy not available")
    from subpx._gpu.regions import grow_regions

    rng = np.random.default_rng(1)
    for k, H, W in ((200, 100, 100), (100, 37, 53), (30, 188, 188), (64, 8, 70)):
        for density in (0.45, 0.6, 0.75):
            above = cp.asarray(rng.random((k, H, W)) < density)
            top = cp.asarray(rng.integers(0, H * W, k))
            np.testing.assert_array_equal(cp.asnumpy(grow_regions(above, top, H + W)), cp.asnumpy(_loop(above, top, cp)))
    # A snake longer than H + W steps: both stop growing at the same place.
    above = cp.asarray(np.repeat(_snake()[None], 3, axis=0))
    top = cp.asarray(np.array([0, 5, 21 * 21 - 1]))
    grown = cp.asnumpy(grow_regions(above, top, 42))
    np.testing.assert_array_equal(grown, cp.asnumpy(_loop(above, top, cp)))
    assert grown[0].sum() < _snake().sum()


def test_top_mean_kernel_equals_partition_mean_bit_for_bit():
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
            same = (old.view(np.int64) == new.view(np.int64)) | (np.isnan(old) & np.isnan(new))
            assert same.all(), (length, n, np.flatnonzero(~same)[:5])
