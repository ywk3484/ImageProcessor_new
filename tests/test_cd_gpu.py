"""GPU CD backend against the CPU implementation (skipped without CuPy or a CUDA device)."""

import numpy as np
import pytest
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

cp = pytest.importorskip("cupy")
try:
    _DEVICES = cp.cuda.runtime.getDeviceCount()
except Exception:
    _DEVICES = 0
pytestmark = pytest.mark.skipif(_DEVICES == 0, reason="CUDA device unavailable")

from subpx import load_cd_results, measure_cds, save_cd_results  # noqa: E402
from subpx._cpu import cd as cpu_cd  # noqa: E402
from subpx._cpu import deblend as cpu  # noqa: E402
from subpx._cpu.cells import _extract_voronoi_rois, compute_voronoi_labels_cpu  # noqa: E402

FWHM = 4.0


def lattice(ratio, *, snr=20, shape=(120, 240), seed=0):
    """Staggered Gaussian holes (amplitude 50 on 20) with a 15 px hole-free margin."""
    rng = np.random.default_rng(seed)
    pitch, sigma = ratio * FWHM, FWHM / (2 * np.sqrt(2 * np.log(2)))
    xx, yy = np.meshgrid(np.arange(15, shape[1] - 15, pitch), np.arange(15, shape[0] - 15, pitch))
    xx[1::2] += pitch / 2
    centers = np.column_stack((xx.ravel(), yy.ravel()))
    centers = centers[centers[:, 0] < shape[1] - 15]
    y, x = np.indices(shape, dtype=float)
    image = np.full(shape, 20.0)
    for cx, cy in centers:
        image += 50 * np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / sigma**2)
    return image + (rng.normal(0, 50 / snr, shape) if snr else 0), centers


def cells(ratio=2.0):
    image, centers = lattice(ratio)
    centers = centers + 0.2
    labels = compute_voronoi_labels_cpu(centers, image.shape)
    half = int(np.ceil(0.6 * ratio * FWHM))
    rows = [[x - half, y - half, 2 * half + 1, 2 * half + 1, x, y] for x, y in centers]
    _, masks, origins = _extract_voronoi_rois(image, labels, rows, pad=3)
    return image, centers, rows, masks, np.asarray(origins)


@pytest.mark.parametrize("kind", ["uniform", "clustered", "integer-ties", "few"])
def test_voronoi_labels_equal_cpu(kind):
    from subpx._gpu.voronoi import compute_voronoi_labels_gpu

    rng = np.random.default_rng(3)
    shape = (130, 170)
    if kind == "uniform":
        seeds = rng.uniform(0, [170, 130], (500, 2))
    elif kind == "clustered":
        seeds = rng.uniform(0, [60, 40], (400, 2))
    elif kind == "integer-ties":
        seeds = np.column_stack(np.nonzero(np.ones((20, 20))))[:, ::-1] * 6.0 + 3  # equidistant pixels
    else:
        seeds = rng.uniform(0, [170, 130], (5, 2))
    np.testing.assert_array_equal(compute_voronoi_labels_gpu(seeds, shape), compute_voronoi_labels_cpu(seeds, shape))


def test_cell_masks_equal_cpu():
    from subpx._gpu.cells import voronoi_cells

    image, centers, rows, masks, origins = cells(1.6)
    rows = [[x - 0.4, y - 0.7, w, h, cx, cy] for x, y, w, h, cx, cy in rows]  # fractional (peak-seed) boxes
    rows[0][:2] = [-2.6, -1.2]  # truncation toward zero at the image edge
    labels = compute_voronoi_labels_cpu(np.asarray(rows)[:, 4:6], image.shape)
    _, masks, origins = _extract_voronoi_rois(image, labels, rows, pad=3)
    gpu_labels, gpu_masks, gpu_origins = voronoi_cells(rows, image.shape, pad=3)
    np.testing.assert_array_equal(gpu_labels, labels)
    np.testing.assert_array_equal(gpu_masks, masks)
    np.testing.assert_array_equal(gpu_origins, np.asarray(origins))


def test_background_pieces_are_bit_identical():
    from subpx._gpu import deblend as gpu

    rng = np.random.default_rng(1)
    values = rng.normal(size=(3000, 9, 9)).round(rng.integers(0, 3))
    border = rng.random((3000, 9, 9)) < rng.uniform(0, 0.5, (3000, 1, 1))
    border[:3] = False
    border[3, 0, 0] = border[4, :1, :2] = True
    expected = np.array([np.percentile(v[b], 10) if b.any() else 0.0 for v, b in zip(values, border)])
    np.testing.assert_array_equal(cp.asnumpy(gpu.border_percentile(cp.asarray(values), cp.asarray(border))), expected)
    for tile in (16, 40):  # both of NumPy's nanmedian code paths
        residual = rng.normal(size=(250, 700)).round(2)
        free = rng.random(residual.shape) < 0.3
        free[:, :120] = False  # unmeasured tiles: interpolated
        for usable in (None, rng.random(residual.shape) < 0.9):
            expected_map, expected_info = cpu.estimate_background_map(residual, free, tile, usable=usable)
            got_map, got_info = gpu.estimate_background_map(residual, free, tile, usable=usable)
            np.testing.assert_array_equal(got_map, expected_map)
            assert got_info["source"] == expected_info["source"]


def test_fits_and_rendering_match_cpu():
    from subpx._gpu import deblend as gpu

    image, centers, rows, masks, origins = cells(1.4)
    raw = cpu.windows(image, origins, masks.shape[1:])
    border = masks & ~binary_erosion(masks, structure=np.ones((1, 3, 3)), border_value=0)
    data = np.where(masks, raw - np.array([np.percentile(r[b], 10) for r, b in zip(raw, border)])[:, None, None], 0.0)
    init = cpu.initial_params(data, masks, centers - origins)
    expected, expected_status = cpu.fit_gaussians_batch(data, masks, init)
    params, status = gpu.fit_gaussians_batch(data, masks, init)
    np.testing.assert_array_equal(status, expected_status)
    np.testing.assert_allclose(params, expected, rtol=0, atol=1e-6)
    global_params = expected.copy()
    global_params[:, 1:3] += origins
    cap = max(masks.shape[1:])
    model = gpu.render_models(global_params, image.shape, radius_cap=cap)
    np.testing.assert_allclose(model, cpu.render_models(global_params, image.shape, radius_cap=cap), rtol=0, atol=1e-8 * 50)
    np.testing.assert_array_equal(gpu.render_models(global_params, image.shape, radius_cap=cap), model)  # deterministic
    huge = global_params.copy()
    huge[:, 0] *= 1e9  # the fixed-point scale adapts: no overflow, resolution ~1e-12 of the brightest hole
    expected = cpu.render_models(huge, image.shape, radius_cap=cap)
    np.testing.assert_allclose(gpu.render_models(huge, image.shape, radius_cap=cap), expected, rtol=0,
                               atol=1e-12 * expected.max())


def test_measurements_match_cpu_on_identical_cells():
    from subpx._gpu import cd as gpu_cd

    reference = measure_cds(lattice(2.0)[0], area_max=400)
    args = (reference.rois, reference.masks, reference.origins_xy, reference.seeds_xy)
    kwargs = dict(methods=("logquad", "gradient", "halfmax"), n_angles=128, radial_step=0.1, smooth_sigma=0.0,
                  fit_fraction=0.2, backgrounds=np.zeros(len(reference.rois)))
    expected, got = cpu_cd.measure_cells(*args, **kwargs), gpu_cd.measure_cells(*args, **kwargs)
    for method in expected:
        np.testing.assert_array_equal(got[method].status, expected[method].status)
        np.testing.assert_allclose(got[method].metrics["cd_equivalent_px"], expected[method].metrics["cd_equivalent_px"],
                                   rtol=0, atol=1e-9, equal_nan=True)
        np.testing.assert_allclose(got[method].contours_xy, expected[method].contours_xy, rtol=0, atol=1e-9, equal_nan=True)
    params = reference.deblend_model["params"][reference.deblend_model["report"]].copy()
    params[:, 1:3] -= reference.origins_xy
    status = reference.measurements["gaussian"].status
    expected = cpu_cd.gaussian_method(params, status, reference.rois, reference.masks, reference.origins_xy, 128)
    got = gpu_cd.gaussian_method(params, status, reference.rois, reference.masks, reference.origins_xy, 128)
    np.testing.assert_array_equal(got.contours_xy, expected.contours_xy)  # host ellipses: same bits
    np.testing.assert_array_equal(got.status, expected.status)


@pytest.mark.parametrize("ratio, seeds", [(2.0, "threshold"), (1.4, "peaks")])
def test_measure_cds_gpu_matches_cpu(ratio, seeds):
    image, _ = lattice(ratio, shape=(160, 320))
    expected = measure_cds(image, seeds=seeds, area_max=400)
    got = measure_cds(image, seeds=seeds, area_max=400, backend="gpu")
    assert got.meta["measurement_backend"] == got.meta["cell_backend"] == "gpu"
    np.testing.assert_array_equal(got.seeds_xy, expected.seeds_xy)
    np.testing.assert_array_equal(got.cell_labels, expected.cell_labels)
    for method, measured in expected.measurements.items():
        other = got.measurements[method]
        assert np.mean(other.status == measured.status) >= 0.999
        both = other.valid & measured.valid
        delta = np.abs(other.metrics["cd_equivalent_px"][both] - measured.metrics["cd_equivalent_px"][both])
        assert delta.max(initial=0) <= 2e-3


@pytest.mark.parametrize("ratio, seeds", [(2.67, "threshold"), (2.0, "peaks"), (1.6, "peaks"), (1.4, "peaks")])
def test_gpu_gaussian_cd_is_unbiased_through_dense_arrays(ratio, seeds):
    image, centers = lattice(ratio)
    result = measure_cds(image, seeds=seeds, methods=("gaussian",), area_max=400, backend="gpu")
    measured = result.measurements["gaussian"]
    distance = cKDTree(centers).query(result.seeds_xy)[0]
    inner = np.all((result.seeds_xy > 20) & (result.seeds_xy < np.array(image.shape[::-1]) - 20), axis=1)
    use = measured.valid & inner & (distance < 1)
    assert use.sum() >= 0.99 * inner.sum()
    values = measured.metrics["cd_equivalent_px"][use]
    assert abs(values.mean() - FWHM) < 0.05 and values.std() < 0.25


def test_gpu_results_round_trip_compact(tmp_path):
    result = measure_cds(lattice(2.0)[0], area_max=400, backend="gpu")
    save_cd_results(result, tmp_path, compact=True)
    loaded = load_cd_results(tmp_path)
    for method, measured in result.measurements.items():
        np.testing.assert_array_equal(loaded.measurements[method].contours_xy, measured.contours_xy)
    np.testing.assert_array_equal(loaded.background_map, result.background_map)
    # Cells are rebuilt with the CPU renderer: equal to the GPU's fixed-point sum to its resolution.
    np.testing.assert_allclose(loaded.rois, result.rois, rtol=0, atol=1e-7)


def test_gpu_backend_requires_model_background():
    with pytest.raises(ValueError, match="requires background='model'"):
        measure_cds(lattice(2.67)[0], background="border", backend="gpu", area_max=400)
    assert measure_cds(lattice(2.67)[0], background="border", backend="auto", area_max=400).meta["measurement_backend"] == "cpu"
