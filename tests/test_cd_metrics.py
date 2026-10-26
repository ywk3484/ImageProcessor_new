"""Public metrics-only results and local compact export contracts."""

import numpy as np
import pytest

from scripts.benchmark_cd import lattice
from scripts.cd_report import ALL_METHODS
from subpx import load_cd_results, measure_cds, save_cd_results, save_cd_viewer
from subpx.cd import LOCAL_METHODS


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_metrics_only_preserves_all_report_measurements(backend):
    if backend == "gpu":
        cp = pytest.importorskip("cupy")
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device unavailable")
    image, _ = lattice((70, 100), 10, 4.4, 20)
    options = dict(methods=ALL_METHODS, backend=backend, area_max=200, outlines=False, n_angles=16)
    full = measure_cds(image, **options)
    small = measure_cds(image, metrics_only=True, **options)
    np.testing.assert_array_equal(full.seeds_xy, small.seeds_xy)
    assert small.rois.shape == small.masks.shape == (len(small.seeds_xy), 0, 0)
    assert small.meta["cell_stack_shape"] == list(full.masks.shape[1:])
    assert small.meta["deblend_iterations_used"] == sum(small.meta["deblend_pass_iterations"])
    for name in ALL_METHODS:
        a, b = full.measurements[name], small.measurements[name]
        assert b.contours_xy.shape == (len(b.valid), 0, 2) and not b.inputs
        np.testing.assert_array_equal(a.valid, b.valid)
        np.testing.assert_array_equal(a.status, b.status)
        np.testing.assert_allclose(a.centers_xy, b.centers_xy, atol=1e-10, equal_nan=True)
        for metric in a.metrics:
            if metric != "perimeter_px":
                np.testing.assert_allclose(a.metrics[metric], b.metrics[metric], rtol=1e-12, atol=1e-10, equal_nan=True)


def test_local_compact_export_reconstructs_raw_cells(tmp_path):
    image, _ = lattice((70, 100), 10, 4.4, 20)
    original = measure_cds(image, methods=LOCAL_METHODS, area_max=200)
    save_cd_results(original, tmp_path, compact=True)
    loaded = load_cd_results(tmp_path)
    for field in ("rois", "masks", "origins_xy", "cell_labels", "seeds_xy"):
        np.testing.assert_array_equal(getattr(original, field), getattr(loaded, field))
    for name in LOCAL_METHODS:
        np.testing.assert_allclose(original.measurements[name].contours_xy,
                                   loaded.measurements[name].contours_xy, atol=1e-12, equal_nan=True)


def test_metrics_only_rejects_geometry_exports_before_writing(tmp_path):
    image, _ = lattice((70, 100), 10, 4.4, 20)
    result = measure_cds(image, methods=LOCAL_METHODS, area_max=200, metrics_only=True)
    output = tmp_path / "compact"
    with pytest.raises(ValueError, match="metrics_only=False"):
        save_cd_results(result, output, compact=True)
    assert not output.exists()
    with pytest.raises(ValueError, match="metrics_only=False"):
        save_cd_viewer(result, tmp_path / "viewer.html")


def test_metrics_only_full_npz_roundtrip(tmp_path):
    image, _ = lattice((70, 100), 10, 4.4, 20)
    original = measure_cds(image, methods=LOCAL_METHODS, area_max=200, metrics_only=True)
    save_cd_results(original, tmp_path)
    loaded = load_cd_results(tmp_path)
    assert loaded.meta["metrics_only"] and loaded.rois.shape[1:] == (0, 0)
    for name in LOCAL_METHODS:
        np.testing.assert_array_equal(original.measurements[name].valid, loaded.measurements[name].valid)
        np.testing.assert_array_equal(original.measurements[name].metrics["cd_equivalent_px"],
                                      loaded.measurements[name].metrics["cd_equivalent_px"])
