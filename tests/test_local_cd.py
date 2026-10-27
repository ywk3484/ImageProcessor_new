"""The experimental local estimators stay separate from deblending and retain cell identity."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from scripts.benchmark_cd import lattice
from subpx import measure_cds
from subpx.cd import LOCAL_METHODS
from subpx._cpu import local_cd


def _backend(name):
    if name == "cpu":
        return np
    from subpx.backends import cp
    if cp is None:
        pytest.skip("CuPy unavailable")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device unavailable")
    except Exception:
        pytest.skip("CUDA device unavailable")
    return cp


def _image():
    return lattice((100, 130), 15, 4.4, 35, seed=4)[0]


def _fixed_cells():
    """Three unrelated shapes with fixed territories, so intensity invariance is tested independently of detection."""
    size = 25
    shapes = np.zeros((3, size, size))
    shapes[0, 8:17, 7:18] = 1  # rectangle
    shapes[1, 7:18, 10:15] = 1
    shapes[1, 12:18, 7:19] = 1  # asymmetric L
    yy, xx = np.mgrid[:size, :size]
    shapes[2] = ((xx - 12) ** 2 / 35 + (yy - 12) ** 2 / 12 < 1)
    cells = np.stack([20 + 80 * gaussian_filter(shape, 1.2) for shape in shapes])
    image = np.concatenate(cells, axis=1)
    labels = np.repeat(np.arange(3)[None], size, axis=0).repeat(size, axis=1)
    rows = [[i * size, 0, size, size, i * size + 12, 12] for i in range(3)]
    return image, labels, rows


def _compare(a, b, *, tolerance=1e-10):
    np.testing.assert_array_equal(a.valid, b.valid)
    np.testing.assert_array_equal(a.status, b.status)
    np.testing.assert_allclose(a.centers_xy, b.centers_xy, rtol=0, atol=tolerance, equal_nan=True)
    for name in ("area_px2", "cd_equivalent_px", "cd_x_px", "cd_y_px"):
        np.testing.assert_allclose(a.metrics[name], b.metrics[name], rtol=0, atol=tolerance, equal_nan=True)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_local_methods_ignore_independent_cell_gain_and_offset(backend):
    xp = _backend(backend)
    image, labels, rows = _fixed_cells()
    gain, offset = np.array([0.7, 2.5, 1.4]), np.array([40.0, -13.0, 5.0])
    transformed = image * gain[labels] + offset[labels]
    options = dict(methods=LOCAL_METHODS, pad=0, n_angles=16, metrics_only=True, xp=xp)
    before, *_ = local_cd.measure_local(image, labels, rows, np.arange(3), **options)
    after, *_ = local_cd.measure_local(transformed, labels, rows, np.arange(3), **options)
    for name in LOCAL_METHODS:
        assert before[name].valid.all(), name
        _compare(before[name], after[name])


def test_local_methods_gpu_cpu_parity():
    _backend("gpu")
    image = _image()
    options = dict(methods=LOCAL_METHODS, area_max=200, n_angles=16, metrics_only=True)
    cpu = measure_cds(image, backend="cpu", **options)
    gpu = measure_cds(image, backend="gpu", **options)
    np.testing.assert_array_equal(cpu.seeds_xy, gpu.seeds_xy)
    np.testing.assert_array_equal(cpu.cell_labels, gpu.cell_labels)
    for name in LOCAL_METHODS:
        _compare(cpu.measurements[name], gpu.measurements[name], tolerance=1e-8)
        for value in gpu.measurements[name].metrics.values():
            assert isinstance(value, np.ndarray)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_local_only_never_calls_deblend(backend, monkeypatch):
    _backend(backend)
    from subpx._cpu import deblend as cpu

    def unexpected(*args, **kwargs):
        raise AssertionError("local-only measurement called deblend")

    monkeypatch.setattr(cpu, "deblend", unexpected)
    if backend == "gpu":
        from subpx._gpu import deblend as gpu
        monkeypatch.setattr(gpu, "deblend", unexpected)
    result = measure_cds(_image(), methods=LOCAL_METHODS, backend=backend, area_max=200, metrics_only=True)
    assert len(result.seeds_xy) > 0
    assert result.meta["measurement_scheme"] == "local"
    assert result.meta["deblend_iterations_used"] == 0
    assert result.meta["deblend_converged"] is None


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_local_zero_seeds_returns_empty_arrays(backend):
    _backend(backend)
    result = measure_cds(np.zeros((40, 50), np.uint8), methods=LOCAL_METHODS, backend=backend, metrics_only=True)
    assert result.seeds_xy.shape == (0, 2)
    assert result.rois.shape == result.masks.shape == (0, 0, 0)
    assert np.all(result.cell_labels == -1)
    for value in result.measurements.values():
        assert value.centers_xy.shape == (0, 2)
        assert value.contours_xy.shape == (0, 0, 2)
        assert value.valid.shape == value.metrics["cd_equivalent_px"].shape == (0,)


def test_large_outlier_does_not_pad_all_local_cells(monkeypatch):
    image = np.zeros((220, 220))
    labels = np.full(image.shape, -1, np.int32)
    rows = []
    for i in range(20):
        x, y = 4 + i % 10 * 16, 4 + i // 10 * 16
        rows.append([x, y, 9, 9, x + 4, y + 4])
        labels[y:y + 9, x:x + 9] = i
        image[y:y + 9, x:x + 9] = np.exp(-((np.arange(9)[:, None] - 4) ** 2
                                          + (np.arange(9)[None] - 4) ** 2) / 4)
    rows.append([1, 50, 201, 160, 100, 130])
    labels[50:210, 1:202] = 20
    seen = []
    original = local_cd._projection

    def record(cells, masks, *args, **kwargs):
        seen.append(cells.shape)
        return original(cells, masks, *args, **kwargs)

    monkeypatch.setattr(local_cd, "_projection", record)
    found, origins, rois, masks, info = local_cd.measure_local(
        image, labels, rows, np.arange(len(rows)), methods=("local_projection",), pad=0,
        n_angles=16, metrics_only=True)
    assert rois.shape == masks.shape == (21, 0, 0)
    assert info["local_groups"] == 2
    assert (20, 12, 12) in seen
    assert all(n == 1 for n, h, w in seen if h > 12 or w > 12)
    assert found["local_projection"].valid[:20].all()


def test_gpu_oversized_cell_keeps_ordinary_cells_on_fused_path(monkeypatch):
    xp = _backend("gpu")
    from subpx._gpu import local_cd as gpu_local

    image = np.zeros((110, 220))
    labels = np.full(image.shape, -1, np.int32)
    rows = []
    for i in range(20):
        x, y, size = 4 + i % 10 * 16, 4 + i // 10 * 16, 9
        yy, xx = np.mgrid[:size, :size]
        image[y:y + size, x:x + size] = 20 + 90 * np.exp(-((xx - 4.2) ** 2 + (yy - 4.4) ** 2) / 4)
        labels[y:y + size, x:x + size] = i
        rows.append([x, y, size, size, x + 4.2, y + 4.4])
    # Too large for shared memory even at factor 1, but only one such cell.
    size, x, y = 60, 5, 45
    yy, xx = np.mgrid[:size, :size]
    image[y:y + size, x:x + size] = 30 + 120 * np.exp(-((xx - 29.3) ** 2 + (yy - 30.2) ** 2) / 180)
    labels[y:y + size, x:x + size] = 20
    rows.append([x, y, size, size, x + 29.3, y + 30.2])
    report = np.random.default_rng(12).permutation(len(rows))
    cubic = tuple(name for name in LOCAL_METHODS if name.startswith("local_") and name != "local_projection")
    methods = cubic + tuple(name.replace("local_", "pixel_") for name in cubic)
    options = dict(methods=methods, pad=0, n_angles=16, metrics_only=True, outlines=False)
    expected, *_ = local_cd.measure_local(image, labels, rows, report, **options)
    fallback_calls, fused_calls = [], []
    original_half, original_fused = local_cd._half_areas, gpu_local.measure_local_areas

    def record_fallback(cells, masks, *args, **kwargs):
        fallback_calls.append((cells.shape, kwargs["factor"]))
        return original_half(cells, masks, *args, **kwargs)

    def record_fused(*args, **kwargs):
        result = original_fused(*args, **kwargs)
        fused_calls.append((len(args[4]), kwargs["factor"], result is not None))
        return result

    monkeypatch.setattr(local_cd, "_half_areas", record_fallback)
    monkeypatch.setattr(gpu_local, "measure_local_areas", record_fused)
    found, _, rois, masks, info = local_cd.measure_local(image, labels, rows, report, xp=xp, **options)
    for name in methods:
        _compare(expected[name], found[name], tolerance=1e-8)
    assert (20, 4, True) in fused_calls
    assert (20, 1, True) in fused_calls
    assert len(fallback_calls) == 2
    assert all(shape == (1, 60, 60) for shape, factor in fallback_calls)
    assert {factor for shape, factor in fallback_calls} == {1, 4}
    assert rois.shape == masks.shape == (21, 0, 0)
    assert info["local_fallback_cells"] == 1
    assert info["cell_storage"] == "fused+bucketed"


def test_mixed_methods_keep_common_cell_alignment():
    image = _image()
    options = dict(backend="cpu", area_max=200, n_angles=16, deblend_iterations=3, metrics_only=True)
    model = measure_cds(image, methods=("gaussian", "half_area"), **options)
    mixed = measure_cds(image, methods=("gaussian", "half_area", *LOCAL_METHODS), **options)
    np.testing.assert_array_equal(mixed.seeds_xy, model.seeds_xy)
    np.testing.assert_array_equal(mixed.component_ids, model.component_ids)
    np.testing.assert_array_equal(mixed.origins_xy, model.origins_xy)
    for name in ("gaussian", "half_area"):
        _compare(model.measurements[name], mixed.measurements[name])
    for name in LOCAL_METHODS:
        measured = mixed.measurements[name]
        assert len(measured.valid) == len(mixed.seeds_xy)
        assert np.max(np.abs(measured.centers_xy[measured.valid] - mixed.seeds_xy[measured.valid])) < 1


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_local_metrics_only_matches_full_output(backend):
    _backend(backend)
    image = _image()
    options = dict(methods=LOCAL_METHODS, backend=backend, area_max=200, n_angles=16)
    full = measure_cds(image, **options)
    metrics = measure_cds(image, metrics_only=True, **options)
    np.testing.assert_array_equal(full.seeds_xy, metrics.seeds_xy)
    assert full.rois.shape[1] > 0
    assert metrics.rois.shape == metrics.masks.shape == (len(metrics.seeds_xy), 0, 0)
    for name in LOCAL_METHODS:
        _compare(full.measurements[name], metrics.measurements[name])
        assert metrics.measurements[name].contours_xy.shape == (len(metrics.seeds_xy), 0, 2)


@pytest.mark.parametrize("shape", ["rectangle", "rotated_L", "irregular"])
def test_local_area_methods_follow_size_changes_in_nonround_shapes(shape):
    # Fixed blur and fixed territories: enlarging the actual object must move the result. This checks sensitivity,
    # not agreement with a physical CD calibration. Area-equivalent true diameter scales linearly with scale.
    size, supersample = 41, 8
    yy, xx = np.mgrid[:size * supersample, :size * supersample]
    xx, yy = (xx + 0.5) / supersample - size / 2, (yy + 0.5) / supersample - size / 2
    theta = 0.43
    images = []
    scales = (0.8, 1.0, 1.2)
    for scale in scales:
        u = (np.cos(theta) * xx + np.sin(theta) * yy) / scale
        v = (-np.sin(theta) * xx + np.cos(theta) * yy) / scale
        if shape == "rectangle":
            region = (np.abs(u) < 5) & (np.abs(v) < 3)
        elif shape == "rotated_L":
            region = (np.abs(u) < 5) & (np.abs(v) < 5) & ((u < -1) | (v > 1))
        else:
            region = ((np.abs(u) < 5) & (np.abs(v) < 3)) | ((u - 1) ** 2 + (v - 3) ** 2 < 9)
        blurred = gaussian_filter(region.astype(float), 0.9 * supersample)
        pixels = blurred.reshape(size, supersample, size, supersample).mean(axis=(1, 3))
        images.append(20 + 80 * pixels)
    image = np.concatenate(images, axis=1)
    labels = np.repeat(np.arange(3)[None], size, axis=0).repeat(size, axis=1)
    rows = [[i * size, 0, size, size, i * size + size // 2, size // 2] for i in range(3)]
    # Projection can switch between different edges of a concave shape; it is intentionally outside this
    # nonround-shape sensitivity check and remains an optional diagnostic method.
    methods = tuple(name for name in LOCAL_METHODS if name != "local_projection")
    found, *_ = local_cd.measure_local(image, labels, rows, np.arange(3), methods=methods,
                                     pad=0, n_angles=16, metrics_only=True)
    for name, result in found.items():
        assert result.valid.all(), name
        measured = result.metrics["cd_equivalent_px"]
        assert np.all(np.diff(measured) > 0), name
        # Normalize both changes at the middle size, so differing edge definitions cannot hide a flat response.
        response = (measured[-1] - measured[0]) / measured[1] / (scales[-1] - scales[0])
        assert 0.75 < response < 1.25, (name, shape, response)
