"""Metrics-only invariant measurements retain numbers without display arrays."""

from types import SimpleNamespace

import numpy as np
import pytest

from subpx._cpu import invariant_cd as cd


def cells():
    yy, xx = np.mgrid[:25, :25]
    windows = np.stack([80 * np.exp(-((xx - x) ** 2 / 15 + (yy - y) ** 2 / 9))
                        for x, y in ((11.7, 12.2), (12.3, 11.8))])
    masks = np.ones_like(windows, dtype=bool)
    origins = np.array([[10.0, 20.0], [40.0, 30.0]])
    return windows, masks, origins


def same_numbers(full, metrics):
    np.testing.assert_array_equal(metrics.status, full.status)
    np.testing.assert_array_equal(metrics.valid, full.valid)
    np.testing.assert_allclose(metrics.centers_xy, full.centers_xy, rtol=0, atol=0)
    assert metrics.metrics.keys() == full.metrics.keys()
    for name in full.metrics:
        np.testing.assert_allclose(metrics.metrics[name], full.metrics[name], rtol=0, atol=0,
                                   equal_nan=True, err_msg=name)
    assert metrics.contours_xy.shape == (len(metrics.valid), 0, 2)
    assert metrics.geometry["edge_px"].shape == (len(metrics.valid), 0)
    assert metrics.inputs == {}


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_metrics_only_projection_and_area_match_full_numbers(backend):
    xp = np
    if backend == "gpu":
        from subpx.backends import cp
        if cp is None:
            pytest.skip("CuPy not available")
        xp = cp
    windows, masks, origins = cells()
    full = cd._projection(windows, masks, origins, 64, xp=xp)
    metrics = cd._projection(windows, masks, origins, 64, xp=xp, metrics_only=True)
    same_numbers(full, metrics)
    for background in ("median", "min", "zero", "edge", "foot"):
        options = dict(name=background, background=background, peak_pixels=4.0, outlines=False, xp=xp)
        full = cd._half_area(windows, masks, origins, 64, **options)
        metrics = cd._half_area(windows, masks, origins, 64, metrics_only=True, **options)
        same_numbers(full, metrics)


def test_metrics_only_tail_methods_do_not_download_display_windows(monkeypatch):
    windows, masks, origins = cells()
    ring = masks.copy()
    ring[:, 3:-3, 3:-3] = False
    monkeypatch.setattr(cd, "_wide_cells", lambda *args, **kwargs: (windows, masks, ring, masks, origins))
    methods = ("tail_area", "tail_area_norm", "edge_area", "foot_area", "foot_area_ratio")
    args = (np.zeros((80, 80)), SimpleNamespace(), masks, origins, np.arange(2))
    options = dict(methods=methods, n_angles=64, pitch_px=10, outlines=False)
    full, _ = cd.measure_invariant(*args, **options)
    original = cd._to_numpy

    def no_display_arrays(value):
        assert value.ndim < 3, "metrics-only measurement transferred a display window"
        return original(value)

    monkeypatch.setattr(cd, "_to_numpy", no_display_arrays)
    metrics, info = cd.measure_invariant(*args, metrics_only=True, **options)
    assert info["tail_window_px"] == windows.shape[1]
    for name in methods:
        same_numbers(full[name], metrics[name])
        assert full[name].inputs["cells"].ndim == 3


def test_metrics_only_skips_contour_calculation_even_with_outlines_enabled(monkeypatch):
    windows, masks, origins = cells()

    def no_contours(*args, **kwargs):
        raise AssertionError("metrics-only measurement calculated display contours")

    monkeypatch.setattr(cd, "radial_contours", no_contours)
    assert cd._projection(windows, masks, origins, 64, metrics_only=True).valid.all()
    assert cd._half_area(windows, masks, origins, 64, metrics_only=True).valid.all()
