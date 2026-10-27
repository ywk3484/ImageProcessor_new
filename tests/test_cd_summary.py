"""Batch summaries preserve measurements without full image result construction."""

import numpy as np
import pytest

from scripts.benchmark_cd import lattice
from subpx import measure_cds, plot_cd_map, save_cd_results, save_cd_viewer
from subpx.cd import PIXEL_METHODS
from subpx.types import CDResult, CDSummary


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("methods", [PIXEL_METHODS, ("local_half_area",), ("gaussian",),
                                     ("gaussian", "pixel_half_area")])
def test_summary_preserves_measurements_without_full_result(backend, methods, monkeypatch):
    if backend == "gpu":
        cp = pytest.importorskip("cupy")
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA unavailable")
    image, _ = lattice((50, 70), 10, 4.4, 20)
    options = dict(methods=methods, backend=backend, area_max=200, n_angles=16,
                   outlines=False, metrics_only=True, pixel_size=21, unit="nm")
    full = measure_cds(image, **options)

    def forbidden(*args):
        raise AssertionError("summary must not construct full image statistics")

    monkeypatch.setattr(CDResult, "__post_init__", forbidden)
    summary = measure_cds(image, summary_only=True, **options)
    assert isinstance(summary, CDSummary)
    assert summary.pixel_size == 21 and summary.unit == "nm"
    assert summary.meta["cell_stack_shape"] == full.meta["cell_stack_shape"]
    np.testing.assert_array_equal(summary.seeds_xy, full.seeds_xy)
    for name in methods:
        a, b = full.measurements[name], summary.measurements[name]
        assert not b.inputs and b.contours_xy.shape[1:] == (0, 2)
        np.testing.assert_array_equal(a.status, b.status)
        np.testing.assert_array_equal(a.valid, b.valid)
        np.testing.assert_allclose(a.centers_xy, b.centers_xy, atol=1e-10)
        for key in a.metrics:
            assert isinstance(b.metrics[key], np.ndarray)
            np.testing.assert_allclose(a.metrics[key], b.metrics[key], atol=1e-10, rtol=1e-10)


def test_empty_summary_and_export_contract(tmp_path):
    summary = measure_cds(np.zeros((20, 20), np.uint8), methods=PIXEL_METHODS,
                           summary_only=True, backend="cpu")
    assert summary.seeds_xy.shape == (0, 2)
    assert summary.meta["metrics_only"] and summary.meta["summary_only"]
    for name in PIXEL_METHODS:
        assert summary.measurements[name].valid.size == 0
    for writer in (save_cd_results, save_cd_viewer):
        with pytest.raises(ValueError, match="summary_only=False|metrics_only=False"):
            writer(summary, tmp_path / "output")
    with pytest.raises(ValueError, match="summary_only=False"):
        plot_cd_map(summary)
    assert not list(tmp_path.iterdir())


def test_summary_border_omits_contours_and_inputs():
    image, _ = lattice((50, 70), 10, 4.4, 20)
    summary = measure_cds(image, methods=("logquad",), background="border", area_max=200,
                           summary_only=True, backend="cpu")
    measured = summary.measurements["logquad"]
    assert measured.contours_xy.shape == (len(measured.valid), 0, 2)
    assert not measured.inputs


def test_summary_cpu_measurement_with_gpu_cells():
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA unavailable")
    image, _ = lattice((50, 70), 10, 4.4, 20)
    result = measure_cds(image, methods=PIXEL_METHODS, summary_only=True,
                         area_max=200, backend="cpu", cell_backend="gpu")
    assert isinstance(result.seeds_xy, np.ndarray)
    assert result.measurements["pixel_half_area"].valid.any()
