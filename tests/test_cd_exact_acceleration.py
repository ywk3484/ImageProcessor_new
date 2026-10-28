"""Acceleration preserves the original deblended, interpolated measurements."""

import numpy as np
import pytest

from scripts import cd_report as report
from scripts.benchmark_cd import lattice
from subpx import measure_cds


METHODS = ("gaussian", "projection", "half_area", "half_area_min", "half_area_min_avg",
           "tail_area", "tail_area_norm", "edge_area", "foot_area", "foot_area_ratio")
AREA_METHODS = METHODS[2:]


@pytest.fixture(scope="module")
def gpu():
    from subpx.backends import cp

    if cp is None:
        pytest.skip("CuPy not available")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device not available")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime not available")
    return cp


def _same_values(actual, expected, path=""):
    """All exported numbers agree to FP64 rounding; discrete fields are exact."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for name in expected:
            _same_values(actual[name], expected[name], f"{path}/{name}")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, (value, reference) in enumerate(zip(actual, expected)):
            _same_values(value, reference, f"{path}/{index}")
    elif isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray), path
        if expected.dtype.kind in "biuUS":
            np.testing.assert_array_equal(actual, expected, err_msg=path)
        else:
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-10,
                                       equal_nan=True, err_msg=path)
    elif isinstance(expected, (float, np.floating)):
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-10,
                                   equal_nan=True, err_msg=path)
    else:
        assert actual == expected, path


@pytest.mark.parametrize("summary_only", [False, True], ids=["full_report", "batch_summary"])
@pytest.mark.parametrize("pitch,fwhm,snr,invert,seeds", [
    (10.0, 4.5, 35.0, False, "threshold"),
    (8.0, 5.8, 22.0, False, "peaks"),
    (10.5, 5.0, 18.0, True, "peaks"),
], ids=["shifted_noisy", "overlapping", "dark_holes"])
def test_original_methods_match_cublas_reference_end_to_end(
        gpu, monkeypatch, summary_only, pitch, fwhm, snr, invert, seeds):
    from subpx._gpu import interpolate

    image, _ = lattice((96, 128), pitch, fwhm, snr, seed=713)
    if invert:
        image = 255 - image
    options = dict(methods=METHODS, backend="gpu", seeds=seeds, invert=invert,
                   area_max=200, pixel_size=21, unit="nm", n_angles=64,
                   deblend_iterations=30, deblend_tolerance=1e-4,
                   half_area_level=0.47, half_area_peak_pixels=3.5,
                   tail_area_level=0.23, tail_window=1.3, summary_only=summary_only)
    accelerated = interpolate.enlarge_cells
    calls = []

    def fast(cells, zy, zx):
        calls.append((cells.shape, zy.shape, zx.shape))
        return accelerated(cells, zy, zx)

    monkeypatch.setattr(interpolate, "enlarge_cells", fast)
    actual = measure_cds(image, **options)
    assert calls, "The optimized enlargement must be exercised by the public API"
    fast_calls = calls.copy()
    calls.clear()

    def reference(cells, zy, zx):
        calls.append((cells.shape, zy.shape, zx.shape))
        return zy[None] @ cells @ zx.T[None]

    monkeypatch.setattr(interpolate, "enlarge_cells", reference)
    expected = measure_cds(image, **options)
    assert calls == fast_calls
    assert len(actual.seeds_xy) > 20
    np.testing.assert_array_equal(actual.seeds_xy, expected.seeds_xy)
    assert actual.pixel_size == expected.pixel_size and actual.unit == expected.unit
    assert actual.meta["background"] == "model"
    assert actual.meta["deblend_iterations_used"] > 0
    for key in ("deblend_iterations_used", "deblend_passes", "deblend_pass_iterations", "deblend_converged",
                "deblend_tolerance", "deblend_status_counts", "tail_window_px"):
        _same_values(actual.meta[key], expected.meta[key], key)
    assert tuple(actual.measurements) == METHODS
    for name in METHODS:
        measured, original = actual.measurements[name], expected.measurements[name]
        assert measured.valid.any(), f"{name} needs valid measurements to check"
        for field in ("valid", "status", "centers_xy", "contours_xy", "metrics", "geometry", "inputs"):
            _same_values(getattr(measured, field), getattr(original, field), f"{name}/{field}")
        if name in AREA_METHODS:
            # Each area is an exact count of 1/16 px² samples: require the
            # identical region size, rather than accepting a close CD alone.
            np.testing.assert_array_equal(measured.metrics["area_px2"], original.metrics["area_px2"])
            areas = measured.metrics["area_px2"][measured.valid]
            np.testing.assert_array_equal(areas * 16, np.rint(areas * 16))
        if summary_only:
            assert measured.inputs == {} and measured.contours_xy.shape[1] == 0

    _same_values(report.cd_statistics(actual), report.cd_statistics(expected), "report/statistics")
    _same_values(report.layer_values(actual), report.layer_values(expected), "report/layers")
    if not summary_only:
        for field in ("image", "binary", "component_labels", "component_ids", "boxes_xywh", "cell_labels",
                      "cell_areas_px2", "origins_xy", "rois", "masks", "background_map", "deblend_model",
                      "mean_intensity"):
            _same_values(getattr(actual, field), getattr(expected, field), field)
        _same_values(report.hole_data(actual, report.layer_values(actual)),
                     report.hole_data(expected, report.layer_values(expected)), "report/hole_data")
        for name in METHODS:
            _same_values(report.rays_data(actual, name), report.rays_data(expected, name), f"report/{name}/rays")
