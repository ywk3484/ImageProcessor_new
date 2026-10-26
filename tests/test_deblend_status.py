"""Accepted fit statuses and metrics-only Gaussian measurements."""

import numpy as np
import pytest

from subpx._cpu import cd as cpu_cd
from subpx._cpu import deblend as cpu


@pytest.fixture(params=("cpu", "gpu"))
def backend(request):
    if request.param == "cpu":
        return cpu, cpu_cd
    cp = pytest.importorskip("cupy")
    try:
        available = cp.cuda.runtime.getDeviceCount()
    except Exception:
        available = 0
    if not available:
        pytest.skip("CUDA device unavailable")
    from subpx._gpu import cd, deblend
    return deblend, cd


def single_cell(sigma=10.0, amplitude=100.0):
    y, x = np.indices((7, 7))
    image = 10 + amplitude * np.exp(-0.5 * ((x - 3) ** 2 + (y - 3) ** 2) / sigma**2)
    masks = np.ones((1, 7, 7), bool)
    init = np.array([[amplitude, 3, 3, -np.log(0.5), 0, -np.log(0.5)]])
    return image, np.array([[3.0, 3.0]]), masks, np.array([[0, 0]]), init


def test_status_describes_the_accepted_damped_fit(backend):
    fit, _ = backend
    image, seeds, masks, origins, init = single_cell()
    state = fit.deblend(image, seeds, masks, origins, init=init, iterations=1, background_level=10)
    expected = np.full(1, "ok", dtype="U32")
    cpu.fit_status(state.params, masks, cpu.sigma_caps(masks), expected)
    np.testing.assert_array_equal(state.status, expected)
    assert state.status[0] == "ok"
    assert not state.converged


def test_no_usable_fits_is_not_convergence(backend):
    fit, _ = backend
    image, seeds, masks, origins, init = single_cell(amplitude=0)
    state = fit.deblend(image, seeds, masks, origins, init=init, iterations=3, background_level=10)
    assert state.status[0] == "non_gaussian_fit"
    assert not state.converged


def test_continuation_initial_status_is_checked(backend):
    fit, _ = backend
    image, seeds, masks, origins, init = single_cell()
    init[:, 3] = init[:, 5] = -np.log(10)
    state = fit.deblend(image, seeds, masks, origins, init=init, iterations=0, background_level=10)
    assert state.status[0] == "non_gaussian_fit"
    assert state.iterations == 0 and not state.converged


@pytest.mark.parametrize("change", [
    {"tolerance": np.nan}, {"tolerance": -1}, {"relaxation": 0}, {"iterations": -1},
    {"init": np.zeros((1, 5))}, {"init": np.full((1, 6), np.nan)},
])
def test_invalid_continuation_options_fail_clearly(change):
    image, seeds, masks, origins, init = single_cell()
    options = dict(init=init, iterations=1, background_level=10)
    options.update(change)
    with pytest.raises(ValueError):
        cpu.deblend(image, seeds, masks, origins, **options)


def test_empty_fit_is_not_reported_as_convergence():
    state = cpu.deblend(np.zeros((7, 7)), np.empty((0, 2)), np.zeros((0, 7, 7), bool), np.empty((0, 2)))
    assert state.iterations == 0 and not state.converged


def test_metrics_only_gaussian_preserves_measurement_numbers(backend):
    _, measure = backend
    params = np.array([[100, 3.2, 3.1, -np.log(1.2), 0.13, -np.log(0.9)],
                       [80, 2.9, 3.3, -np.log(1.1), -0.2, -np.log(1.3)]])
    masks = np.ones((2, 7, 7), bool)
    origins = np.array([[0, 0], [12, 8]])
    cleaned = cpu.model_on_rois(params, (7, 7)) + np.random.default_rng(7).normal(0, 0.1, (2, 7, 7))
    status = np.array(["ok", "center_outside_cell"], dtype="U32")
    full = measure.gaussian_method(params, status, cleaned, masks, origins, 128)
    small = measure.gaussian_method(params, status, cleaned, masks, origins, 128, metrics_only=True)
    assert small.contours_xy.shape == (2, 0, 2)
    assert small.geometry == {}
    assert np.isnan(small.metrics["perimeter_px"]).all()
    np.testing.assert_array_equal(small.status, full.status)
    np.testing.assert_array_equal(small.valid, full.valid)
    np.testing.assert_array_equal(small.centers_xy, full.centers_xy)
    for name in full.metrics:
        if name != "perimeter_px":
            np.testing.assert_array_equal(small.metrics[name], full.metrics[name], err_msg=name)
