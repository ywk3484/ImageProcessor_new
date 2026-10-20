import numpy as np
import pytest

from subpx import load_cd_results, measure_cds, save_cd_results
from subpx._cpu.invariant_cd import DiskTable, far_field, render_disks
from scripts.validate_invariant_cd import against_truth as _fit_against_truth, exact_disks as _exact_disks
from scripts.validate_invariant_cd import matched as _matched, pulsed_lattice

PSF, D0, DARK = 3.3, 4.76, 6.0  # ~70 nm blur, ~100 nm holes on ~21 nm pixels


def _lattice_image(*, seed=0, pulse=0.05, noise=True, shape=(110, 150)):
    """ACTIS-like holes: per-hole pulse a_i lights the hole and its own cell's absorber background."""
    return pulsed_lattice(psf_fwhm=PSF, cd=D0, shape=shape, pulse=pulse, dark=DARK, noise=noise, seed=seed)


@pytest.fixture(scope="module")
def pulsed():
    image, centers, d, a = _lattice_image()
    result = measure_cds(image, methods=("gaussian", "disk", "flux_ratio", "projection", "half_area"), area_max=200,
                         psf_fwhm=PSF, dark_level=DARK)
    return result, _matched(result, centers, d, a)


def test_disk_table_matches_exact_rendering():
    table = DiskTable(PSF, 3 * D0, 24)
    center = np.array([[31.3, 30.6]])
    # The FFT rendering wraps around: render 256 x 256 and crop, so the window holds no wrapped tails.
    exact = 100 * _exact_disks((256, 256), center + 96, [D0], [1.0], PSF)[96:160, 96:160]
    params = np.array([[100.0, *center[0], D0, 0.0]])
    model = render_disks(table, params, (64, 64), 16) + far_field(table, params, (64, 64), 16)
    near = np.hypot(*(np.mgrid[:64, :64] - center[0, ::-1, None, None])) < 12
    # The table averages the square pixel's response over direction; that costs < 1 % of the peak.
    assert np.abs(model - exact)[near].max() < 0.01 * exact.max()
    # The far tail carries the light the 16 px rendering truncates (~4 % of the hole's light).
    assert abs(model.sum() - exact.sum()) < 0.01 * exact.sum()


def test_disk_fit_recovers_hole_diameter_and_ignores_the_pulse(pulsed):
    result, (use, truth, pulse) = pulsed
    disk = result.measurements["disk"]
    assert disk.valid[use].mean() > 0.97
    slope, scatter, sensitivity = _fit_against_truth(disk.metrics["cd_equivalent_px"], truth, pulse, use & disk.valid)
    assert 0.9 < slope < 1.1
    assert scatter < 0.035 * D0
    assert abs(sensitivity) < 0.1
    assert abs(np.median(disk.metrics["cd_equivalent_px"][use]) - D0) < 0.03 * D0
    # The reported uncertainty is of the right size (noise only; scatter adds model and neighbour errors).
    assert 0.4 * scatter < np.nanmedian(disk.metrics["cd_uncertainty_px"][use]) < 1.5 * scatter
    # A circle of the fitted diameter around the fitted center.
    i = np.flatnonzero(use & disk.valid)[0]
    radii = np.linalg.norm(disk.contours_xy[i] - disk.centers_xy[i], axis=1)
    assert np.allclose(radii, disk.metrics["cd_equivalent_px"][i] / 2)


def test_disk_keeps_the_cells_it_was_fitted_to(pulsed):
    result, _ = pulsed
    disk = result.measurements["disk"]
    cells = disk.inputs["cells"]
    assert cells.shape == result.masks.shape
    np.testing.assert_array_equal(np.isfinite(cells), result.masks)
    assert result.measurements["flux_ratio"].inputs["cells"] is cells
    table = DiskTable(PSF, 3 * D0, 24)
    yy, xx = np.indices(cells.shape[1:])
    for i in np.flatnonzero(disk.valid)[:20]:
        cx, cy = disk.geometry["center_local"][i]
        rho = np.hypot(xx - cx, yy - cy)
        m = disk.metrics
        model = m["amplitude"][i] * table.lookup(np.full(rho.shape, m["cd_equivalent_px"][i]), rho, False)
        residual = (cells[i] - model - m["local_background"][i])[result.masks[i]]
        assert np.isclose(np.sqrt(np.mean(residual ** 2)), m["fit_rmse"][i], rtol=1e-4)


def test_lit_background_ratio_and_projection_ignore_the_pulse_while_the_gaussian_does_not(pulsed):
    result, (use, truth, pulse) = pulsed
    sensitivity = {}
    for name, measured in result.measurements.items():
        assert measured.valid[use].mean() > 0.9, name
        _, _, sensitivity[name] = _fit_against_truth(measured.metrics["cd_equivalent_px"], truth, pulse,
                                                     use & measured.valid)
    assert abs(sensitivity["flux_ratio"]) < 0.2  # plain flux would be 0.5: CD ~ sqrt(flux)
    assert abs(sensitivity["projection"]) < 0.5
    # The smooth background map cannot follow each cell's own pulse-lit background.
    assert sensitivity["gaussian"] > 0.3
    projection = result.measurements["projection"]
    ok = use & projection.valid
    assert np.allclose(projection.metrics["cd_equivalent_px"][ok],
                       (projection.metrics["cd_x_px"][ok] + projection.metrics["cd_y_px"][ok]) / 2)


def test_projection_is_not_disturbed_by_the_ragged_cell_outline():
    # Identical noise-free holes; rows 10.5 px apart, so every other row's cell outline falls differently on the
    # pixels. Averaging each column over the cell's pixels (10 in one column, 11 in the next) spread these by 0.15 px.
    image, *_ = pulsed_lattice(psf_fwhm=PSF, cd=D0, pitch=10.5, shape=(110, 150), pulse=0.0, cd_spread=0.0,
                               dark=DARK, noise=False)
    result = measure_cds(image, methods=("projection",), area_max=300)
    projection = result.measurements["projection"]
    cd = projection.metrics["cd_equivalent_px"][projection.valid]
    assert projection.valid.mean() > 0.9
    assert cd.std() < 0.1


def test_nominal_cd_calibrates_the_blur():
    image, centers, d, a = _lattice_image(seed=1)
    result = measure_cds(image, methods=("disk", "flux_ratio"), area_max=200, nominal_cd=D0 * 21, pixel_size=21,
                         unit="nm")
    assert result.meta["psf_source"].startswith("calibrated")
    assert abs(result.meta["psf_fwhm_px"] - PSF) < 0.05 * PSF
    use, _, _ = _matched(result, centers, d, a)
    for name in ("disk", "flux_ratio"):
        median = np.median(result.measurements[name].metrics["cd_equivalent_px"][use & result.measurements[name].valid])
        assert abs(median - D0) < 0.03 * D0, name


def test_invalid_requests_raise():
    image, *_ = _lattice_image(noise=False, shape=(60, 60))
    with pytest.raises(ValueError, match="nominal_cd or psf_fwhm"):
        measure_cds(image, methods=("disk",), area_max=200)
    with pytest.raises(ValueError, match="background='model'"):
        measure_cds(image, methods=("projection",), background="border", area_max=200)
    with pytest.raises(ValueError, match="nominal_cd"):
        measure_cds(image, methods=("disk",), area_max=200, nominal_cd=30.0)
    with pytest.raises(ValueError, match="half_area_level"):
        measure_cds(image, methods=("half_area",), area_max=200, half_area_level=1.0)
    measure_cds(image, methods=("projection", "half_area"), area_max=200)  # need no blur


def test_compact_export_rebuilds_the_new_contours(tmp_path, pulsed):
    result, _ = pulsed
    save_cd_results(result, tmp_path, compact=True)
    loaded = load_cd_results(tmp_path)
    for name in ("disk", "flux_ratio", "projection", "half_area"):
        original, other = result.measurements[name], loaded.measurements[name]
        np.testing.assert_array_equal(other.contours_xy, original.contours_xy)
        np.testing.assert_array_equal(other.status, original.status)
        for metric, values in original.metrics.items():
            np.testing.assert_array_equal(other.metrics[metric], values)


def test_cli_runs_the_new_methods(tmp_path):
    from PIL import Image
    from scripts.analyze_contact_holes import main

    image, *_ = _lattice_image(shape=(60, 80))
    Image.fromarray(image.astype(np.uint8)).save(tmp_path / "holes.png")
    main([str(tmp_path / "holes.png"), "--methods", "disk", "flux_ratio", "projection", "--psf-fwhm", str(PSF * 21),
          "--pixel-size", "21", "--unit", "nm", "--dark-level", str(DARK), "--area-max", "200",
          "--outputs", "compact", "--output", str(tmp_path / "out")])
    loaded = load_cd_results(tmp_path / "out")
    assert set(loaded.measurements) == {"disk", "flux_ratio", "projection"}
    assert loaded.meta["psf_fwhm"] == pytest.approx(PSF * 21)


def test_projection_spline_operators_reproduce_scipy():
    from scipy.interpolate import CubicSpline
    from subpx._cpu.invariant_cd import _spline_operators

    profiles = np.random.default_rng(0).normal(size=(6, 11))
    grid, values, slopes = _spline_operators(11, 0.02)
    spline = CubicSpline(np.arange(11), profiles, axis=1)
    np.testing.assert_allclose(profiles @ values.T, spline(grid), atol=1e-9)
    np.testing.assert_allclose(profiles @ slopes.T, spline(grid, 1), atol=1e-9)


def _gaussian_cells(centers, sigma=1.5, size=15):
    y, x = np.mgrid[:size, :size]
    return np.stack([np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2)) for cx, cy in centers])


def test_half_area_of_a_gaussian_spot_is_its_fwhm_and_ignores_scale_and_pedestal():
    from subpx._cpu.invariant_cd import _half_area

    cells = _gaussian_cells([(7.0, 7.0), (7.3, 6.6), (1.0, 7.0)])  # the third spot runs over the cell edge
    masks = np.ones(cells.shape, bool)
    origins = np.zeros((3, 2))
    plain = _half_area(cells, masks, origins, 64)
    # A pulse: the spot and its surroundings scaled together (here a pedestal that scales with it).
    lit = _half_area(3.0 * (cells + 0.2), masks, origins, 64)
    fwhm = 2 * np.sqrt(2 * np.log(2)) * 1.5
    assert list(plain.status) == ["ok", "ok", "region_at_cell_border"]
    np.testing.assert_allclose(plain.metrics["cd_equivalent_px"][:2], fwhm, rtol=0.02)
    np.testing.assert_allclose(lit.metrics["cd_equivalent_px"][:2], plain.metrics["cd_equivalent_px"][:2], rtol=1e-9)
    np.testing.assert_allclose(plain.centers_xy[:2], [[7.0, 7.0], [7.3, 6.6]], atol=0.15)
    assert np.isnan(plain.metrics["cd_equivalent_px"][2]) and not plain.valid[2]


def _shape_lattice(seed=0, gain=1.0, grow=1.0, shape=(110, 150), pitch=10.5, size=D0, pad=16, s=6):
    """Randomly rotated rectangles and ellipses of equal-area size ``size`` (px), Airy blur PSF, per-feature pulse
    lighting the feature and its own cell; rasterized by supersampling. The methods know no shape."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    height, width = shape[0] + 2 * pad, shape[1] + 2 * pad
    ys, xs = np.mgrid[pitch / 2:height:pitch, pitch / 2:width:pitch]
    centers = np.column_stack((xs.ravel(), ys.ravel())) + rng.uniform(-0.3, 0.3, (xs.size, 2))
    shapes = np.random.default_rng(99)
    kind, angle = shapes.integers(0, 2, len(centers)), shapes.uniform(0, np.pi, len(centers))
    d = size * grow * (1 + 0.03 * rng.standard_normal(len(centers)))
    a = 1 + 0.05 * rng.standard_normal(len(centers))
    fine = np.zeros((height * s, width * s))
    for (x, y), k, t, di, ai in zip(centers, kind, angle, d, a):
        x0, y0 = max(int((x - di) * s), 0), max(int((y - di) * s), 0)
        x1, y1 = min(int((x + di) * s) + 1, width * s), min(int((y + di) * s) + 1, height * s)
        gy, gx = (np.mgrid[y0:y1, x0:x1] + 0.5) / s
        u = (gx - x) * np.cos(t) + (gy - y) * np.sin(t)
        v = -(gx - x) * np.sin(t) + (gy - y) * np.cos(t)
        if k:  # rectangle 1.6 : 1 of area di^2
            half_long = di * np.sqrt(1.6) / 2
            inside = (np.abs(u) <= half_long) & (np.abs(v) <= half_long / 1.6)
        else:  # ellipse 1.5 : 1 of area di^2
            r = di / np.sqrt(np.pi)
            inside = (u / (r * np.sqrt(1.5))) ** 2 + (v * np.sqrt(1.5) / r) ** 2 <= 1
        fine[y0:y1, x0:x1] += ai * inside
    image = fine.reshape(height, s, width, s).mean(axis=(1, 3))
    fy, fx = np.fft.fftfreq(height)[:, None], np.fft.rfftfreq(width)[None, :]
    f, fc = np.hypot(fx, fy), 1.029 / PSF
    nu = np.minimum(f / fc, 1)
    otf = np.where(f < fc, (2 / np.pi) * (np.arccos(nu) - nu * np.sqrt(1 - nu * nu)), 0)
    image = np.fft.irfft2(np.fft.rfft2(image) * otf, s=(height, width))
    yy, xx = np.mgrid[:height, :width]
    _, cell = cKDTree(centers).query(np.column_stack((xx.ravel(), yy.ravel())))
    image = gain * (110 * image + 10 * a[cell].reshape(height, width)) + DARK
    image += rng.standard_normal(image.shape) * np.sqrt(1 + 0.05 * np.maximum(image - DARK, 0))
    return np.clip(np.rint(image), 0, 255)[pad:-pad, pad:-pad], centers - pad, d, a


@pytest.fixture(scope="module")
def shaped():
    def run(**kw):
        image, centers, d, a = _shape_lattice(**kw)
        result = measure_cds(image, methods=("projection", "half_area", "half_area_min"), area_max=300)
        return result, _matched(result, centers, d, a)
    return run(), run(gain=1.2), run(grow=1.05)


def test_half_area_separates_the_pulse_from_size_for_unknown_shapes(shaped):
    (result, (use, truth, pulse)), (bright, (use_b, *_)), (grown, (use_g, *_)) = shaped
    half, projection = result.measurements["half_area"], result.measurements["projection"]
    assert half.valid[use].mean() > 0.97
    _, scatter, sensitivity = _fit_against_truth(half.metrics["cd_equivalent_px"], truth, pulse, use & half.valid)
    _, scatter_projection, _ = _fit_against_truth(projection.metrics["cd_equivalent_px"], truth, pulse,
                                                  use & projection.valid)
    assert abs(sensitivity) < 0.1
    assert scatter < 0.6 * scatter_projection
    mean = np.nanmean(half.metrics["cd_equivalent_px"][use])
    # The whole image 20 % brighter: the same CD. Every feature 5 % larger: a clearly larger CD.
    assert abs(np.nanmean(bright.measurements["half_area"].metrics["cd_equivalent_px"][use_b]) / mean - 1) < 0.005
    assert np.nanmean(grown.measurements["half_area"].metrics["cd_equivalent_px"][use_g]) / mean - 1 > 0.015


def test_half_area_and_projection_agree_between_cpu_and_gpu(shaped):
    from subpx.backends import cp

    if cp is None:
        pytest.skip("CuPy not available")
    result, _ = shaped[0]
    gpu = measure_cds(result.image, methods=("projection", "half_area", "half_area_min"), area_max=300, backend="gpu")
    for name in ("projection", "half_area", "half_area_min"):
        np.testing.assert_array_equal(gpu.measurements[name].status, result.measurements[name].status)
        np.testing.assert_allclose(gpu.measurements[name].metrics["cd_equivalent_px"],
                                   result.measurements[name].metrics["cd_equivalent_px"], rtol=1e-9)


def test_half_area_min_takes_the_lowest_cell_edge_pixel_and_ignores_scale_and_pedestal():
    from subpx._cpu.invariant_cd import _half_area

    cells = _gaussian_cells([(7.0, 7.0), (7.3, 6.6)])
    cells[:, 0, 5] -= 0.05  # one low pixel on the cell edge sets the minimum level
    masks = np.ones(cells.shape, bool)
    origins = np.zeros((2, 2))
    median, low = _half_area(cells, masks, origins, 64), _half_area(cells, masks, origins, 64, background="min")
    lit = _half_area(3.0 * (cells + 0.2), masks, origins, 64, background="min")
    assert low.method == "half_area_min" and list(low.status) == ["ok", "ok"]
    ring = np.concatenate([cells[:, 0], cells[:, -1], cells[:, 1:-1, 0], cells[:, 1:-1, -1]], axis=1)
    np.testing.assert_allclose(low.metrics["local_level"], ring.min(axis=1))
    assert np.all(low.metrics["local_level"] < median.metrics["local_level"])
    # A lower level puts the half-height threshold lower: a larger region.
    assert np.all(low.metrics["cd_equivalent_px"] > median.metrics["cd_equivalent_px"])
    np.testing.assert_allclose(lit.metrics["cd_equivalent_px"], low.metrics["cd_equivalent_px"], rtol=1e-9)
