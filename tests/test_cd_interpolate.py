"""The fast GPU enlargement retains the full original cubic calculation."""

import numpy as np
import pytest

from subpx._cpu.invariant_cd import _zoom_operators


def gpu():
    from subpx.backends import cp

    if cp is None:
        pytest.skip("CuPy not available")
    return cp


@pytest.mark.parametrize("shape,whole", [
    ((0, 12, 12), (12, 12)),
    ((7, 1, 5), (1, 5)),
    ((13, 12, 16), (25, 29)),
    ((5, 25, 25), (25, 25)),
    ((3, 31, 31), (31, 31)),
    ((2, 47, 53), (47, 53)),
])
def test_enlargement_matches_original_full_cubic_products(shape, whole):
    cp = gpu()
    from subpx._gpu.interpolate import enlarge_cells

    n, h, w = shape
    rng = np.random.default_rng(214)
    data = rng.normal(40, 12, (n, h, w))
    data[rng.random(data.shape) < 0.3] = 0.0
    cells = cp.asarray(data)
    # Sub-block operators are sliced from the whole stack, just as _half_areas
    # does. This also exercises noncontiguous operator storage.
    zy = cp.asarray(_zoom_operators(whole[0], 4)[0])[:4 * h, :h]
    zx = cp.asarray(_zoom_operators(whole[1], 4)[0])[:4 * w, :w]
    expected = zy[None] @ cells @ zx.T[None]
    actual = enlarge_cells(cells, zy, zx)
    assert actual.dtype == cp.float64
    assert actual.shape == (n, 4 * h, 4 * w)
    # Different CUDA/cuBLAS versions may accumulate FP64 products differently;
    # this permits only rounding, not any reduced interpolation precision.
    np.testing.assert_allclose(cp.asnumpy(actual), cp.asnumpy(expected), rtol=3e-14, atol=3e-13)


def test_enlargement_retains_subpixel_response_and_cubic_overshoot():
    cp = gpu()
    from subpx._gpu.interpolate import enlarge_cells

    cells = np.zeros((2, 13, 17))
    cells[0, 6, 8] = 1.0
    cells[1, 6, 9] = 1.0
    zy, zx = (cp.asarray(_zoom_operators(size, 4)[0]) for size in cells.shape[1:])
    actual = cp.asnumpy(enlarge_cells(cp.asarray(cells), zy, zx))
    expected = cp.asnumpy(zy[None] @ cp.asarray(cells) @ zx.T[None])
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-15)
    assert actual.shape == (2, 52, 68)
    assert actual.min() < 0  # cubic ringing is retained, not clipped or smoothed
    assert np.unique(actual[0]).size > 100
    np.testing.assert_allclose(actual[0, :, :-4], actual[1, :, 4:], rtol=0, atol=3e-15)


def test_large_cells_keep_the_original_fallback(monkeypatch):
    cp = gpu()
    from subpx._gpu import interpolate

    cells = cp.arange(2 * 7 * 9, dtype=cp.float64).reshape(2, 7, 9)
    zy, zx = (cp.asarray(_zoom_operators(size, 4)[0]) for size in cells.shape[1:])
    monkeypatch.setattr(interpolate, "optin_shared_bytes", lambda: 0)
    expected = zy[None] @ cells @ zx.T[None]
    np.testing.assert_array_equal(cp.asnumpy(interpolate.enlarge_cells(cells, zy, zx)), cp.asnumpy(expected))
