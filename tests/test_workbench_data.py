"""Integration checks for the workbench's native-precision analysis data."""

import json

import numpy as np
import pytest

from subpx._workbench_data import (
    analyze_image, analyze_patterns, cell_boundary, crop_image, export_review,
    feature_input, feature_origin, grayscale, load_image, load_pages, projection_diagnostic, radial_diagnostic, register_images,
)


def parameters(**changes):
    values = dict(threshold="otsu", invert=False, area_min=3, area_max=200, pad=5,
                  refine="weighted", backend="cpu", methods=("logquad", "gradient", "halfmax"),
                  background="border", seeds="threshold", pixel_size=2.5, unit="nm",
                  n_angles=48, radial_step=0.1, smooth_sigma=0.0, fit_fraction=0.2,
                  half_area_level=0.5, half_area_peak_pixels=4.0, tail_area_level=0.2,
                  tail_window=1.2, deblend_iterations=30, nominal_cd=None, psf_fwhm=None,
                  dark_level=0.0, pitch_source="cd", pitch_options=dict(layout="auto"))
    values.update(changes)
    return values


@pytest.fixture
def spots():
    y, x = np.mgrid[:100, :140]
    image = np.full((100, 140), 12.0)
    for cy in range(15, 90, 16):
        for cx in range(15, 130, 16):
            image += 160 * np.exp(-((x - cx - .2)**2 + (y - cy + .15)**2) / (2 * 2**2))
    return image.astype(np.float32)


def test_tiff_pages_and_native_precision(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    image = np.arange(120, dtype=np.uint16).reshape(10, 12) * 32
    path = tmp_path / "pages.tif"
    with tifffile.TiffWriter(path) as writer:
        writer.write(image)
        writer.write(image + 101)
    loaded = load_image(path, page=1)
    assert loaded.pages == 2 and loaded.page == 1
    np.testing.assert_array_equal(loaded.image, image + 101)
    assert loaded.image.dtype == np.uint16
    with pytest.raises(ValueError, match="page"):
        load_image(path, page=2)
    np.save(tmp_path / "image.npy", image.astype(np.float64) + .000123)
    loaded = load_image(tmp_path / "image.npy")
    np.testing.assert_array_equal(loaded.image, image.astype(np.float64) + .000123)


def test_rgb_and_invalid_images():
    image = np.zeros((8, 10, 4), np.uint8)
    image[..., 0] = 100
    result, description = grayscale(image)
    np.testing.assert_allclose(result, 21.26)
    assert "luminance" in description
    binary, description = grayscale(np.ones((8, 10), bool))
    assert binary.dtype == np.uint8 and binary.max() == 1
    assert "Boolean" in description
    for invalid in (np.zeros((2, 5, 7)), np.full((5, 7), np.nan), np.ones((5, 7), complex)):
        with pytest.raises(ValueError):
            grayscale(invalid)


def test_load_all_tiff_pages_keeps_shapes_dtypes_and_order(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    images = [np.arange(120, dtype=np.uint16).reshape(10, 12) * 32,
              np.arange(48, dtype=np.float64).reshape(8, 6) + .000123,
              np.full((5, 9, 3), [100, 20, 0], dtype=np.uint8)]
    path = tmp_path / "mixed_pages.tiff"
    with tifffile.TiffWriter(path) as writer:
        for image in images:
            writer.write(image, photometric="rgb" if image.ndim == 3 else "minisblack")
    pages = load_pages(path)
    assert len(pages) == 3
    for index, (page, image) in enumerate(zip(pages, images)):
        expected, conversion = grayscale(image)
        np.testing.assert_array_equal(page.image, expected)
        assert page.image.dtype == expected.dtype
        assert (page.page, page.pages) == (index, 3)
        assert page.conversion == conversion
        assert page.path == str(path.resolve())


def test_crop_detection_and_export_keep_global_origin(spots, tmp_path):
    review = analyze_image(spots, parameters(), operation="detect", bounds=[8, 9, 130, 93])
    assert review.detection.rescaled
    assert review.detection.binary.dtype == bool
    assert len(review.seeds_xy) == 40
    # Check the coordinate transform independently of the refinement's bias.
    np.testing.assert_array_equal(np.rint(review.seeds_xy[0] + review.origin_xy), [15, 15])
    patterns = analyze_patterns(review, dict(layout="square", nominal_pitch_xy=(40, 40)))
    # The last column is cropped through its tails; check the interior intervals.
    np.testing.assert_allclose(patterns["detection"].profiles["x"].pitch[:-1], 40, atol=.02)
    review.patterns = patterns
    export_review(review, tmp_path)
    np.testing.assert_allclose(np.load(tmp_path / "centers_global_xy.npy"), review.seeds_xy + [8, 9])
    manifest = json.loads((tmp_path / "workbench.json").read_text())
    assert manifest["origin_xy"] == [8, 9]
    assert (tmp_path / "pitch_detection_x.csv").is_file()
    with pytest.raises(ValueError):
        crop_image(spots, [-1, 0, 20, 30])


def test_cd_native_input_and_radial_diagnostics(spots, tmp_path):
    review = analyze_image(spots, parameters(), operation="pitch")
    assert review.cd.image.dtype == np.float32
    assert review.patterns
    halfmax = review.cd.measurements["halfmax"]
    index = int(np.flatnonzero(halfmax.valid)[0])
    diagnostic = radial_diagnostic(review.cd, "halfmax", index)
    from subpx._cpu.cd import _radial_contour

    _, _, _, status, edges = _radial_contour("halfmax", diagnostic["center"], diagnostic["radii"],
        diagnostic["profiles"], diagnostic["valid"], diagnostic["angles"], 0)
    assert status == "ok"
    np.testing.assert_allclose(edges, diagnostic["edges"], atol=1e-12)
    measured = halfmax.contours_xy[index, :-1]
    center = diagnostic["center"] + review.cd.origins_xy[index]
    np.testing.assert_allclose(np.linalg.norm(measured - center, axis=1), edges, atol=1e-12)
    export_review(review, tmp_path)
    assert (tmp_path / "workbench.json").is_file()
    assert (tmp_path / "pitch_halfmax_y.csv").is_file()


@pytest.mark.parametrize("invert", [False, True])
def test_local_method_input_is_raw_even_in_mixed_run(spots, invert):
    image = 255 - spots if invert else spots
    review = analyze_image(image, parameters(methods=("logquad", "local_half_area"), background="model", invert=invert))
    result = review.cd
    data, mask, description = feature_input(result, "local_half_area", 0)
    ox, oy = result.origins_xy[0]
    yy, xx = np.nonzero(mask)
    np.testing.assert_array_equal(data[mask], image[yy + oy, xx + ox] * (-1 if invert else 1))
    assert "Raw" in description


def test_registration_shift_order_and_overlap():
    from scipy.ndimage import gaussian_filter

    image = gaussian_filter(np.random.default_rng(42).normal(size=(64, 80)), 1)
    moving = np.roll(image, (3, -4), axis=(0, 1))
    review = register_images(image, moving, upsample_factor=40)
    np.testing.assert_allclose(review.result.shift_yx, [-3, 4], atol=.025)
    assert review.rmse_after < 1e-12
    assert review.rmse_before > .1
    assert np.isnan(review.aligned[~review.overlap]).all()
    assert review.correlation.shape == image.shape
    with pytest.raises(ValueError, match="matching shapes"):
        register_images(image, moving[:-1])
    with pytest.raises(ValueError, match="variation"):
        register_images(np.ones((5, 5)), np.ones((5, 5)))


def test_projection_samples_and_wide_window_origin(spots):
    review = analyze_image(spots, parameters(methods=("local_projection", "tail_area"), background="model"))
    result = review.cd
    measured = result.measurements["local_projection"]
    index = int(np.flatnonzero(measured.valid)[0])
    for axis in ("x", "y"):
        diagnostic = projection_diagnostic(result, "local_projection", index, axis)
        np.testing.assert_allclose(diagnostic["right"] - diagnostic["left"], measured.metrics[f"cd_{axis}_px"][index], atol=1e-10)
    tail = result.measurements["tail_area"]
    _, mask, description = feature_input(result, "tail_area", index)
    np.testing.assert_array_equal(feature_origin(result, "tail_area", index), tail.inputs["origins"][index])
    assert mask.shape == tail.inputs["cells"].shape[1:]
    assert "display crop" in description


def test_empty_detection_and_label_boundaries():
    review = analyze_image(np.ones((32, 32), np.uint8), parameters(), operation="detect")
    assert len(review.seeds_xy) == 0
    assert any("No features" in note for note in review.notes)
    edges = cell_boundary(np.array([[0, 0, 1], [0, 2, 1]]))
    np.testing.assert_array_equal(edges, [[False, False, True], [False, True, True]])
