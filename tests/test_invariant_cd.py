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
    result = measure_cds(image, methods=("gaussian", "disk", "flux_ratio", "projection"), area_max=200,
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
    measure_cds(image, methods=("projection",), area_max=200)  # needs no blur


def test_compact_export_rebuilds_the_new_contours(tmp_path, pulsed):
    result, _ = pulsed
    save_cd_results(result, tmp_path, compact=True)
    loaded = load_cd_results(tmp_path)
    for name in ("disk", "flux_ratio", "projection"):
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
