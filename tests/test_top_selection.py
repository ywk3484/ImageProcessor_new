"""GPU peak selection returns the exact sample multiset, including ties."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    _DEVICES = cp.cuda.runtime.getDeviceCount()
except Exception:
    _DEVICES = 0
pytestmark = pytest.mark.skipif(_DEVICES == 0, reason="CUDA device unavailable")


@pytest.mark.parametrize("selected", [1, 4, 64, 257, 513])
def test_radix_selection_keeps_exact_samples_at_complete_and_partial_bins(selected):
    from subpx._gpu._cuda import kernel
    from subpx._gpu.regions import _TOP_SRC

    rng = np.random.default_rng(9284)
    length = 513
    values = rng.normal(30, 10, (8, length))
    values[1] = np.round(values[1])  # a partial bin of repeated boundary values
    values[2] = -np.inf
    values[2, :selected] = 100  # a whole boundary bin, resolved at a coarse key
    values[3] = np.nextafter(1.0, 2.0)
    values[3, ::2] = 1.0  # differences only in the final key byte
    values[4] = rng.choice([-np.inf, -1e100, -1.0, -0.0, 0.0, 1.0, 1e100, np.inf], length)
    values[5, :7] = np.nan
    values[6] = rng.uniform(-1, 1, length) * np.logspace(-200, 200, length)
    values[7] = np.nan
    rng.shuffle(values, axis=1)
    # Guards catch an incorrect early-exit count writing beyond the answer.
    storage = cp.full(values.shape[0] * selected + 16, -779.25, dtype=cp.float64)
    kernel(_TOP_SRC, "top_values")(
        (len(values),), (256,), (cp.asarray(values), np.int64(length), np.int32(selected), storage[8:-8]))
    actual = cp.asnumpy(storage)
    np.testing.assert_array_equal(actual[:8], -779.25)
    np.testing.assert_array_equal(actual[-8:], -779.25)
    picked = np.sort(actual[8:-8].reshape(len(values), selected), axis=1)
    expected = np.sort(values, axis=1)[:, -selected:]
    np.testing.assert_array_equal(picked, expected)
