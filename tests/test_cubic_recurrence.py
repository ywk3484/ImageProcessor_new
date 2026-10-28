"""The linear-work cubic implementation retains every original 4x sample."""

import numpy as np
import pytest

from subpx._cpu.invariant_cd import _zoom_operators

cp = pytest.importorskip("cupy")
try:
    _DEVICES = cp.cuda.runtime.getDeviceCount()
except Exception:
    _DEVICES = 0
pytestmark = pytest.mark.skipif(_DEVICES == 0, reason="CUDA device unavailable")


def enlarge(cells, masks, levels):
    from subpx._gpu._cuda import kernel
    from subpx._gpu.spline import CUBIC_SRC

    n, h, w = cells.shape
    H, W = h * 4, w * 4
    source = CUBIC_SRC + r"""
    extern "C" __global__ void sample_test(
        const double* data, const unsigned char* mask, const double* level,
        double* rows, double* scratch, double* result, const int h, const int w)
    {
        const int c = blockIdx.x, H = 4 * h, W = 4 * w;
        double* r = rows + (long long)c * H * w;
        double* s = scratch + (long long)c * H * w;
        prepare_cubic_rows(data + (long long)c * h * w, mask + (long long)c * h * w,
                           level[c], h, w, r, s);
        for (int i = threadIdx.x; i < H * W; i += blockDim.x)
            result[(long long)c * H * W + i] = cubic_sample4(r + (i / W) * w, w, 1, i % W);
    }
    """
    result = cp.empty((n, H, W), dtype=cp.float64)
    kernel(source, "sample_test", exact=True)((n,), (128,), (
        cp.asarray(cells), cp.asarray(masks, dtype=cp.uint8), cp.asarray(levels),
        cp.empty((n, H, w)), cp.empty((n, H, w)), result, np.int32(h), np.int32(w)))
    return cp.asnumpy(result)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (2, 3), (7, 5), (12, 12),
                                  (16, 13), (25, 31), (47, 53), (97, 101)])
def test_all_cubic_samples_match_original_zero_extended_operators(shape):
    rng = np.random.default_rng(682)
    cells = rng.normal(50, 100, (7, *shape))
    masks = rng.random(cells.shape) > 0.15
    masks[0] = False
    levels = rng.uniform(-50, 50, len(cells))
    zy, zx = (_zoom_operators(n, 4)[0] for n in shape)
    expected = zy[None] @ np.where(masks, cells - levels[:, None, None], 0.0) @ zx.T[None]
    actual = enlarge(cells, masks, levels)
    assert actual.dtype == np.float64 and actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=7e-13)


def test_cubic_interpolation_keeps_boundary_tails_and_negative_lobes():
    cells = np.zeros((5, 12, 16), dtype=np.float64)
    for c, (y, x) in enumerate(((0, 0), (0, 15), (11, 0), (11, 15), (5, 8))):
        cells[c, y, x] = 1.0
    masks = np.ones_like(cells, dtype=bool)
    zy, zx = (_zoom_operators(n, 4)[0] for n in cells.shape[1:])
    expected = zy[None] @ cells @ zx.T[None]
    actual = enlarge(cells, masks, np.zeros(len(cells)))
    np.testing.assert_allclose(actual, expected, rtol=5e-14, atol=7e-16)
    assert actual.min() < -0.09
    # Far samples retain the complete exponentially decaying spline tail.
    np.testing.assert_allclose(actual[0, -4:, -4:], expected[0, -4:, -4:], rtol=2e-13, atol=0)
