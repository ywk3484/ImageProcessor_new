import csv
import json

import numpy as np
import pytest

from subpx import measure_cds, plot_cd_map, save_cd_results, save_cd_viewer
from subpx._cpu.cells import compute_voronoi_labels_cpu


def gaussian_image(shape=(64, 72), center=(35.3, 30.65), sigmas=(2.8, 1.8), angle=0.53):
    y, x = np.indices(shape, dtype=float)
    x, y = x - center[0], y - center[1]
    u = x * np.cos(angle) + y * np.sin(angle)
    v = -x * np.sin(angle) + y * np.cos(angle)
    return 12 + 60 * np.exp(-0.5 * ((u / sigmas[0])**2 + (v / sigmas[1])**2))


def test_rotated_gaussian_widths_contours_and_global_coordinates():
    center = np.array([35.3, 30.65])
    sigmas = np.array([2.8, 1.8])
    result = measure_cds(gaussian_image(), area_max=200, pad=12, n_angles=256, radial_step=0.05)
    assert len(result.seeds_xy) == 1
    gaussian = result.measurements["logquad"]
    assert gaussian.valid[0], gaussian.status[0]
    np.testing.assert_allclose(gaussian.centers_xy[0], center, atol=1e-5)
    np.testing.assert_allclose(
        [gaussian.metrics["sigma_major_px"][0], gaussian.metrics["sigma_minor_px"][0]], sigmas, atol=1e-5,
    )
    np.testing.assert_allclose(gaussian.metrics["angle_deg"], np.degrees(0.53), atol=1e-5)
    k = np.sqrt(2 * np.log(2))
    for name, expected_cd in (("logquad", 2 * k * np.sqrt(np.prod(sigmas))),
                              ("gradient", 2 * np.sqrt(np.prod(sigmas))),
                              ("halfmax", 2 * k * np.sqrt(np.prod(sigmas)))):
        measured = result.measurements[name]
        assert measured.valid[0], measured.status[0]
        assert abs(measured.metrics["cd_equivalent_px"][0] - expected_cd) < 0.035
        np.testing.assert_allclose(measured.contours_xy[:, 0], measured.contours_xy[:, -1])
        np.testing.assert_allclose(measured.centers_xy[0], center, atol=0.06)
    # Horizontal extent of a rotated ellipse is not its major-axis width.
    expected_x = 2 * k * np.sqrt((sigmas[0] * np.cos(0.53))**2 + (sigmas[1] * np.sin(0.53))**2)
    assert abs(gaussian.metrics["cd_x_px"][0] - expected_x) < 1e-5


def test_polarity_and_intensity_scale_preserve_widths():
    bright = gaussian_image()
    first = measure_cds(bright, area_max=200, pad=10)
    dark = measure_cds(1000 - 3 * bright, invert=True, area_max=200, pad=10)
    for method in first.measurements:
        a, b = first.measurements[method], dark.measurements[method]
        assert a.valid.all() and b.valid.all()
        np.testing.assert_allclose(a.contours_xy, b.contours_xy, atol=2e-6)
        np.testing.assert_allclose(a.metrics["cd_equivalent_px"], b.metrics["cd_equivalent_px"], atol=2e-6)


def test_cd_variation_is_spatially_aligned_with_cells():
    y, x = np.indices((48, 120))
    centers = np.array([[20.2, 24.1], [60.3, 24.1], [100.4, 24.1]])
    widths = np.array([2.8, 4.0, 5.2])
    image = np.full(x.shape, 10.0)
    for (cx, cy), width in zip(centers, widths):
        sigma = width / (2 * np.sqrt(2 * np.log(2)))
        image += 65 * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    result = measure_cds(image, area_max=200, pad=9)
    assert len(result.seeds_xy) == 3
    nearest = np.linalg.norm(result.seeds_xy[:, None] - centers, axis=2).argmin(axis=1)
    for method in result.measurements.values():
        assert method.valid.all(), method.status
    np.testing.assert_allclose(result.measurements["logquad"].metrics["cd_equivalent_px"], widths[nearest], atol=1e-5)
    for i, (cx, cy) in enumerate(result.seeds_xy):
        assert result.cell_labels[int(round(cy)), int(round(cx))] == i
    assert result.cell_areas_px2.sum() == image.size


def test_clipped_contours_stay_in_results_with_invalid_status():
    result = measure_cds(gaussian_image(), area_max=200, pad=0)
    assert len(result.seeds_xy) == 1
    for name in ("gradient", "halfmax"):
        method = result.measurements[name]
        assert not method.valid[0]
        assert method.status[0] == "incomplete_rays"
        assert method.metrics["ray_coverage"][0] < 1
        assert np.isnan(method.metrics["cd_equivalent_px"][0])


def test_blurred_disk_gradient_and_halfheight_recover_diameter():
    from scipy.ndimage import gaussian_filter

    y, x = np.indices((80, 80))
    image = 10 + 70 * gaussian_filter((((x - 40)**2 + (y - 40)**2) < 12**2).astype(float), 1.0)
    result = measure_cds(image, area_max=700, pad=6, methods=("gradient", "halfmax"))
    for measured in result.measurements.values():
        assert measured.valid.all(), measured.status
        assert abs(measured.metrics["cd_equivalent_px"][0] - 24) < 0.5


def test_empty_export_and_map(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = measure_cds(np.zeros((24, 40), dtype=np.uint8))
    assert result.cell_labels.shape == (24, 40)
    assert np.all(result.cell_labels == -1)
    assert result.origins_xy.shape == (0, 2)
    assert result.mean_intensity.shape == (0,)
    paths = save_cd_results(result, tmp_path)
    with np.load(paths["raw_data.npz"], allow_pickle=False) as raw:
        assert raw["logquad_contours_xy"].shape == (0, 129, 2)
    for contours in (True, False):
        ax = plot_cd_map(result, contours=contours)
        ax.figure.canvas.draw()
        plt.close(ax.figure)
    ax = plot_cd_map(result, metric="mean_intensity")
    ax.figure.canvas.draw()
    plt.close(ax.figure)
    manifest = json.loads(paths["manifest.json"].read_text())
    assert manifest["methods"]["logquad"]["median_cd_px"] is None
    assert len(paths["measurements.csv"].read_text().splitlines()) == 1


def test_calibrated_exports_reload_without_pickle(tmp_path):
    result = measure_cds(gaussian_image(), area_max=200, pad=10, pixel_size=2.5, unit="nm")
    paths = save_cd_results(result, tmp_path)
    with paths["measurements.csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    for row in rows:
        assert row["valid"] == "True"
        assert row["unit"] == "nm"
        assert float(row["cd_equivalent"]) == pytest.approx(float(row["cd_equivalent_px"]) * 2.5)
        assert float(row["area"]) == pytest.approx(float(row["area_px2"]) * 2.5**2)
        assert float(row["mean_intensity"]) == pytest.approx(result.image.mean())
    with np.load(paths["raw_data.npz"], allow_pickle=False) as raw:
        for name in raw.files:
            assert raw[name].dtype != object
        np.testing.assert_array_equal(raw["image"], result.image)
        np.testing.assert_allclose(raw["mean_intensity"], result.mean_intensity)
        np.testing.assert_allclose(raw["halfmax_contours_xy"], result.measurements["halfmax"].contours_xy)
    with paths["contours.csv"].open(newline="", encoding="utf-8") as handle:
        contours = list(csv.DictReader(handle))
    assert len(contours) == 3 * 129
    for row in contours[::128]:
        assert float(row["x"]) == pytest.approx(float(row["x_px"]) * 2.5)
    with paths["cells.csv"].open(newline="", encoding="utf-8") as handle:
        cells = list(csv.DictReader(handle))
    assert len(cells) == 1
    assert float(cells[0]["mean_intensity"]) == pytest.approx(result.image.mean())


def test_mean_intensity_uses_original_full_cells_and_ignores_cd_validity():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = gaussian_image() + gaussian_image(center=(14.2, 16.3), sigmas=(1.5, 1.5))
    result = measure_cds(image, methods=("gradient",), area_max=200, pad=0, pixel_size=2.5, unit="nm")
    assert len(result.seeds_xy) == 2
    assert not result.measurements["gradient"].valid.all()
    expected = [image[result.cell_labels == i].mean() for i in range(2)]
    np.testing.assert_allclose(result.mean_intensity, expected, atol=1e-12)
    # Include the full cell's background, rather than only the bright CC/ROI.
    assert result.mean_intensity[0] < result.rois[0][result.masks[0]].mean()
    dark = measure_cds(200 - image, invert=True, methods=("gradient",), area_max=200, pad=0)
    np.testing.assert_array_equal(dark.cell_labels, result.cell_labels)
    np.testing.assert_allclose(dark.mean_intensity, 200 - result.mean_intensity, atol=1e-12)
    ax = plot_cd_map(result, metric="mean_intensity")
    np.testing.assert_allclose(ax.images[0].get_array(), result.mean_intensity[result.cell_labels])
    ax.figure.canvas.draw()
    plt.close(ax.figure)


@pytest.mark.parametrize("empty", [False, True])
def test_offline_viewer_preserves_contours_and_calibration(tmp_path, empty):
    import base64
    import re

    image = np.zeros((64, 72), dtype=np.uint8) if empty else gaussian_image()
    result = measure_cds(image, area_max=200, pad=10, pixel_size=2.5, unit="nm")
    path = save_cd_viewer(result, tmp_path / "viewer.html")
    html = path.read_text(encoding="utf-8")
    data = json.loads(re.search(r'<script id="cd-data" type="application/json">(.*?)</script>', html, re.S).group(1))
    assert data["count"] == len(result.seeds_xy)
    assert data["pixel_size"] == 2.5
    assert data["unit"] == "nm"
    assert '<script src=' not in html
    assert 'https://' not in html
    means = np.frombuffer(base64.b64decode(data["means"]), dtype="<f8")
    np.testing.assert_array_equal(means, result.mean_intensity)
    for name, measured in result.measurements.items():
        packed = np.frombuffer(base64.b64decode(data["methods"][name]["points"]), dtype="<f4")
        reconstructed = packed.reshape(measured.contours_xy.shape) + result.origins_xy[:, None, :]
        np.testing.assert_allclose(reconstructed, measured.contours_xy, atol=1e-6)


def test_cpu_voronoi_exact_ties_choose_lowest_id():
    seeds = np.array([[2, 2], [6, 2], [2, 6], [6, 6]], dtype=float)
    labels = compute_voronoi_labels_cpu(seeds, (9, 9))
    assert labels[4, 4] == 0
    y, x = np.indices(labels.shape)
    truth = np.sum((np.stack((x, y), axis=-1)[..., None, :] - seeds)**2, axis=-1).argmin(axis=-1)
    np.testing.assert_array_equal(labels, truth)


def test_gpu_cell_partition_parity():
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device unavailable")
    centers = [(19.2, 19.3), (51.6, 19.4), (19.2, 45.5), (51.6, 45.3)]
    image = np.rint(12 + sum(gaussian_image(center=c) - 12 for c in centers)).astype(np.uint8)
    cpu = measure_cds(image, area_max=200, pad=8)
    gpu = measure_cds(image, area_max=200, pad=8, cell_backend="gpu")
    assert len(cpu.seeds_xy) == 4
    np.testing.assert_array_equal(cpu.cell_labels, gpu.cell_labels)
    np.testing.assert_array_equal(cpu.rois, gpu.rois)
    for method in cpu.measurements:
        np.testing.assert_allclose(cpu.measurements[method].contours_xy, gpu.measurements[method].contours_xy)


def test_cli_writes_maps_and_provenance(tmp_path):
    from PIL import Image
    from scripts.analyze_contact_holes import main

    image_path = tmp_path / "input.png"
    Image.fromarray(np.rint(gaussian_image()).astype(np.uint8)).save(image_path)
    output = tmp_path / "results"
    main([str(image_path), "--output", str(output), "--area-max", "200", "--pad", "10",
          "--methods", "logquad", "--pixel-size", "2", "--unit", "nm"])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["cells"] == 1
    assert manifest["methods"]["logquad"]["valid"] == 1
    assert manifest["input_image"] == str(image_path.resolve())
    assert len(manifest["input_sha256"]) == 64
    assert manifest["unit"] == "nm"
    assert manifest["interactive_viewer"] == "contours.html"
    assert "intensity_map.png" in manifest["figures"]
    for name in manifest["files"]:
        assert (output / name).is_file()


@pytest.mark.parametrize("kwargs", [
    {"methods": ()}, {"methods": ("unknown",)}, {"methods": ("logquad", "logquad")},
    {"pad": -1}, {"n_angles": 8}, {"radial_step": 0}, {"radial_step": np.nan},
    {"fit_fraction": 1}, {"pixel_size": 0}, {"pixel_size": 2}, {"smooth_sigma": -1},
    {"area_min": 10, "area_max": 2}, {"cell_backend": "typo"},
])
def test_invalid_parameters_fail_early(kwargs):
    with pytest.raises(ValueError):
        measure_cds(np.zeros((8, 8), dtype=np.uint8), **kwargs)
