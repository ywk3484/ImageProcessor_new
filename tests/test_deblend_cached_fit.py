"""The accelerated LM fit preserves the original FP64 iterations."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    _DEVICES = cp.cuda.runtime.getDeviceCount()
except Exception:
    _DEVICES = 0
pytestmark = pytest.mark.skipif(_DEVICES == 0, reason="CUDA device unavailable")


@pytest.mark.parametrize("shape", [(9, 13), (16, 16), (20, 19), (64, 70)])
def test_cached_gaussian_evaluations_preserve_fit(shape):
    from subpx._gpu.deblend import _fit_device

    rng = np.random.default_rng(5461)
    count = 19  # a partial final warp group
    yy, xx = np.indices(shape)
    centers = np.tile([(shape[1] - 1) / 2, (shape[0] - 1) / 2], (count, 1))
    centers += rng.uniform(-0.4, 0.4, centers.shape)
    dx, dy = xx[None] - centers[:, 0, None, None], yy[None] - centers[:, 1, None, None]
    cells = 50 * np.exp(-0.5 * ((dx / 2.1) ** 2 + (dy / 1.7) ** 2))
    cells += rng.normal(0, 1.5, cells.shape)
    masks = cp.asarray((dx / (shape[1] / 2)) ** 2 + (dy / (shape[0] / 2)) ** 2 < 1, dtype=cp.uint8)
    initial = np.zeros((count, 6))
    initial[:, 0] = rng.uniform(25, 60, count)
    initial[:, 1:3] = centers + rng.uniform(-1, 1, centers.shape)
    initial[:, 3] = -np.log(rng.uniform(0.6, 3, count))
    initial[:, 5] = -np.log(rng.uniform(0.6, 3, count))
    device_initial = cp.asarray(initial)
    data = cp.asarray(cells)
    expected = _fit_device(data, masks, initial, max_iter=30, _cache_shapes=False)
    actual = _fit_device(data, masks, device_initial, max_iter=30, _download=False)
    assert isinstance(actual, cp.ndarray)
    np.testing.assert_array_equal(cp.asnumpy(actual), expected)
    np.testing.assert_array_equal(cp.asnumpy(device_initial), initial)


def test_empty_device_fit_keeps_output_location():
    from subpx._gpu.deblend import _fit_device

    data, masks, initial = cp.empty((0, 8, 8)), cp.empty((0, 8, 8), dtype=cp.uint8), cp.empty((0, 6))
    assert isinstance(_fit_device(data, masks, initial, max_iter=30), np.ndarray)
    actual = _fit_device(data, masks, initial, max_iter=30, _download=False)
    assert isinstance(actual, cp.ndarray) and actual.shape == (0, 6)
