"""GPU edge-erf finder: an edge is found the same wherever it sits on the page (fixed 2026-11-03).

Its straight-line background term used absolute image coordinates; the normal matrix's condition number grew with
the square of the position and every edge beyond ~140 px from the page's left / top edge was refused.
"""

import math

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device unavailable", allow_module_level=True)
except Exception:  # CUDA runtime missing
    pytest.skip("CUDA runtime unavailable", allow_module_level=True)

from subpx import detect_centers  # noqa: E402
from subpx._gpu.centers import _edge_from_profile_erf_gpu  # noqa: E402
from subpx.synthetic_arrays import render_hole_array  # noqa: E402


def test_same_edge_found_at_any_position():
    starts = np.array([0.0, 100.0, 150.0, 500.0, 1300.0, 2000.0])
    k = np.arange(24.0)
    edge, sigma = 10.37, 1.3
    erf = np.vectorize(math.erf)
    profile = 15.0 + 70.0 * 0.5 * (1 + erf((k - edge) / (math.sqrt(2) * sigma)))
    profiles = cp.asarray(np.tile(profile, (len(starts), 1)))
    found, ok = _edge_from_profile_erf_gpu(profiles, cp.asarray(starts), cp.asarray(starts + 10.0))
    assert bool(cp.all(ok))
    np.testing.assert_allclose(cp.asnumpy(found) - starts, edge, atol=1e-6)


def test_large_squares_found_all_over_the_page():
    page = render_hole_array((420, 700), hole="square", size_nm=(300.0, 300.0), pitch_nm=(600.0, 600.0), seed=3,
                             backend="gpu")
    points = detect_centers(page.image, backend="gpu", threshold="otsu", area_min=40, area_max=1100, pad=3,
                            refine="edge_erf").centers_xy
    truth = page.centers_xy[page.inside]
    h, w = page.image.shape
    margin = 30.0
    far = truth[(truth[:, 0] > margin) & (truth[:, 0] < w - 1 - margin) & (truth[:, 1] > margin)
                & (truth[:, 1] < h - 1 - margin)]
    distance = np.hypot(*(far[:, None, :] - np.asarray(points)[None, :, :]).transpose(2, 0, 1)).min(axis=1)
    assert len(far) > 100
    assert (distance < 0.1).mean() >= 0.99  # before the fix: only the holes within ~140 px of the left edge
