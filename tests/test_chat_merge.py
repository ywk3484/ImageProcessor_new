import numpy as np

from maskproc import (
    fft_periodicity_uniform,
    fft_periodicity_resample,
    fit_rowwise_distortion_field,
    map_centers_to_global,
    cell_runs_to_global_rects,
)


def test_fft_helpers_return_period_peak():
    x = np.linspace(0, 100, 256)
    y = 2.0 * np.sin(2 * np.pi * x / 20.0)
    out = fft_periodicity_uniform(x, y, detrend_order=None)
    k = np.argmax(out['amplitude'][1:]) + 1
    assert abs(out['period_um'][k] - 20.0) < 1.0


def test_fft_resample_smoke():
    x = np.sort(np.random.default_rng(0).uniform(0, 100, size=200))
    y = np.sin(2 * np.pi * x / 25.0)
    out = fft_periodicity_resample(x, y)
    assert out['amplitude'].size > 0


def test_fit_rowwise_distortion_field_smoke():
    xs = np.tile(np.arange(10, dtype=float) * 10.0, 3)
    ys = np.repeat(np.array([0.0, 20.0, 40.0]), 10)
    xs = xs + np.repeat([0.0, 0.3, 0.6], 10)
    centers = np.column_stack([xs, ys])
    model = fit_rowwise_distortion_field(centers, line_threshold=5.0, pitch=10.0, bin_px=20.0)
    assert 'smoothed_map' in model
    assert model['smoothed_map'].shape[0] == 3


def test_map_centers_to_global_and_rects():
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    stripe = {'x0': 100, 'y0': 200, 'Hprime': 1000}
    board = {'x0': 10, 'width': 50}
    g = map_centers_to_global(pts, stripe, board, k_canon=2, H=10, crop_top=3, flipud=False)
    assert np.allclose(g[0], [111.0, 219.0])
    rects = cell_runs_to_global_rects([False, True, True, False, True], stripe, board, H=10, crop_top=3)
    assert len(rects) == 2
