"""Measure all pages (subpx/_workbench_pages.py): real worker processes on a small multipage TIFF."""

import csv
import json

import numpy as np
import pytest

from subpx._workbench_data import analyze_image, analyze_patterns, pitch_numbers
from subpx._workbench_pages import PagesRun, page_values, run_pages

tifffile = pytest.importorskip("tifffile")
METHODS = ("gaussian", "projection", "half_area", "tail_area")


def parameters(**changes):
    values = dict(threshold="otsu", invert=False, area_min=3, area_max=200, pad=3, refine="weighted", backend="cpu",
                  methods=METHODS, background="model", seeds="threshold", pixel_size=2.5, unit="nm", n_angles=48,
                  radial_step=0.1, smooth_sigma=0.0, fit_fraction=0.2, half_area_level=0.5, half_area_peak_pixels=4.0,
                  tail_area_level=0.2, tail_window=1.2, deblend_iterations=30, deblend_tolerance=1e-3,
                  nominal_cd=None, psf_fwhm=None, dark_level=0.0, pitch_source="cd", pitch_finders=(),
                  pitch_options=dict(layout="auto", stagger_axis="row", average_rows=None, average_columns=None,
                                     reference_xy=None, nominal_pitch_xy=None, pitch_hint_xy=None),
                  bounds=None, preprocess=None)
    values.update(changes)
    return values


def spots(shape=(120, 160), width=2.0, offset=(0.0, 0.0), seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:shape[0], :shape[1]]
    image = np.full(shape, 12.0)
    for cy in range(14, shape[0] - 8, 16):
        for cx in range(14, shape[1] - 8, 16):
            image += 160 * np.exp(-((x - cx - offset[0]) ** 2 + (y - cy - offset[1]) ** 2) / (2 * width ** 2))
    return (image + rng.normal(0, 1.5, shape)).astype(np.float32)


@pytest.fixture
def pages(tmp_path):
    """Four pages: holes, holes a little wider, a blank page (no holes), holes on a smaller page."""
    images = [spots(), spots(width=2.2, offset=(.3, -.2), seed=1),
              np.random.default_rng(2).normal(12, 1.5, (120, 160)).astype(np.float32),
              spots(shape=(80, 112), seed=3)]
    path = tmp_path / "pages.tif"
    with tifffile.TiffWriter(path) as writer:
        for image in images:
            writer.write(image)
    return path, images


def direct(image, settings):
    """What Measure CD and Measure pitch give on the page in the window: the CD and pitch results."""
    review = analyze_image(image, settings, operation="cd", bounds=settings["bounds"])
    return review, analyze_patterns(review, dict(settings["pitch_options"], reference_xy=None))


def test_every_page_measured_as_measure_cd_and_cd_batch(pages, tmp_path):
    from scripts.cd_batch import page_values as batch_values
    from subpx.cd import measure_cds

    path, images = pages
    settings = parameters()
    run = run_pages(path, range(4), settings, tmp_path / "out", reference=images[0], reference_page=0)
    assert run.state == "done" and run.counts == {"ok": 3, "no_holes": 1, "failed": 0}
    assert run.rows[2]["status"] == "no_holes" and run.rows[2]["pcc_error"] >= 0.8
    for page in (0, 1, 3):
        row, image = run.rows[page], images[page]
        assert row["status"] == "ok" and row["pcc_error"] < 0.8
        # the same values as scripts/cd_batch.py writes for this page
        summary = measure_cds(image, methods=METHODS, area_min=3, area_max=200, pixel_size=2.5, unit="nm",
                              backend="cpu", n_angles=48, outlines=False, summary_only=True)
        for key, value in batch_values(summary, image, METHODS).items():
            assert row[key] == value or (np.isnan(value) and np.isnan(row[key])), key
        # the same CDs and pitch as Measure CD and Measure pitch in the window
        review, patterns = direct(image, settings)
        for name in METHODS:
            measured = review.cd.measurements[name]
            cds = measured.metrics["cd_equivalent_px"][measured.valid] * 2.5
            assert row[f"{name}_valid"] == np.isfinite(cds).sum()
            assert row[f"{name}_cd"] == pytest.approx(np.nanmean(cds), rel=1e-12)
            for axis in ("x", "y"):
                mean, _, low, high = pitch_numbers(patterns[name].profiles[axis])
                assert row[f"{name}_pitch_{axis}"] == pytest.approx(mean, rel=1e-12)
                assert (row[f"{name}_placement_{axis}_min"], row[f"{name}_placement_{axis}_max"]) == \
                    pytest.approx((low, high), rel=1e-12, abs=1e-12)
        # each hole's CD in the page's CSV
        with (tmp_path / "out" / "holes" / f"page_{page:04d}.csv").open(encoding="utf-8") as stream:
            holes = list(csv.DictReader(stream))
        assert len(holes) == row["holes"] == len(review.seeds_xy)
        measured = review.cd.measurements["half_area"]
        written = np.array([float(h["half_area_cd"]) if h["half_area_cd"] else np.nan for h in holes])
        expected = np.where(measured.valid, measured.metrics["cd_equivalent_px"] * 2.5, np.nan)
        np.testing.assert_allclose(written, expected, rtol=1e-6)
        assert [h["half_area_status"] for h in holes] == list(measured.status)
        np.testing.assert_allclose([[float(h["x"]), float(h["y"])] for h in holes], review.seeds_xy, rtol=1e-6)
    assert not (tmp_path / "out" / "holes" / "page_0002.csv").exists()
    with (tmp_path / "out" / "pages.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(r["page"]) for r in rows] == [0, 1, 2, 3]
    assert [r["status"] for r in rows] == ["ok", "ok", "no_holes", "ok"]
    assert list(rows[0])[:10] == ["file", "page", "status", "pcc_error", "holes", "image_intensity", "cell_intensity",
                                  "deblend_converged", "deblend_iterations", "deblend_passes"]
    manifest = json.loads((tmp_path / "out" / "run.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "done" and manifest["hole_check"]["reference_page"] == 0
    assert manifest["counts"] == {"ok": 3, "no_holes": 1, "failed": 0} and manifest["compute"]["backend"] == "cpu"


def test_region_preprocessing_and_failed_pages(pages, tmp_path):
    path, images = pages
    # a region that fits the large pages only: the small page fails, the others are measured in the region
    settings = parameters(bounds=[8, 6, 150, 110], preprocess=dict(method="gaussian", sigma=0.6))
    run = run_pages(path, [0, 3], settings, tmp_path / "out")
    assert run.rows[0]["status"] == "ok" and run.rows[3]["status"] == "failed"
    assert "analysis region" in run.rows[3]["error"]
    from subpx._workbench_data import preprocess_page

    review, _ = direct(preprocess_page(images[0], settings).image, settings)
    assert run.rows[0]["holes"] == len(review.seeds_xy)
    assert run.rows[0]["half_area_cd"] == pytest.approx(
        np.nanmean(review.cd.measurements["half_area"].metrics["cd_equivalent_px"][
            review.cd.measurements["half_area"].valid]) * 2.5, rel=1e-12)
    with (tmp_path / "out" / "holes" / "page_0000.csv").open(encoding="utf-8") as stream:
        first = next(csv.DictReader(stream))
    assert float(first["x"]) == pytest.approx(review.seeds_xy[0, 0] + 8, rel=1e-6)  # page pixels, not the crop's


def test_stop_keeps_the_rows_so_far(pages, tmp_path):
    path, images = pages
    run = PagesRun(path, range(4), parameters(), tmp_path / "out")
    run.start()
    import time

    deadline = time.monotonic() + 120
    while not run.rows and time.monotonic() < deadline:
        run.step()
        time.sleep(0.02)
    run.stop()
    assert run.state == "stopped" and run.rows and not run.workers
    with (tmp_path / "out" / "pages.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert sorted(int(r["page"]) for r in rows) == sorted(run.rows)
    assert json.loads((tmp_path / "out" / "run.json").read_text(encoding="utf-8"))["state"] == "stopped"
    with pytest.raises(ValueError, match="pages.csv"):  # never overwrites an earlier run's results
        PagesRun(path, range(4), parameters(), tmp_path / "out").start()


def test_gpu_pages_equal_measure_cd_and_add_the_center_finders(pages, tmp_path):
    from subpx.backends import is_gpu_available

    if not is_gpu_available():
        pytest.skip("no GPU")
    from subpx.cd import measure_cds

    path, images = pages
    settings = parameters(backend="gpu", pitch_finders=("logquad",))
    run = run_pages(path, [0, 1], settings, tmp_path / "out")
    assert run.backend == "gpu" and run.threads == 2 and run.counts["ok"] == 2
    for page in (0, 1):
        summary = measure_cds(images[page], methods=METHODS, area_min=3, area_max=200, pixel_size=2.5, unit="nm",
                              backend="gpu", n_angles=48, outlines=False, summary_only=True)
        for key, value in page_values(summary, images[page], METHODS).items():
            assert run.rows[page][key] == value or (np.isnan(value) and np.isnan(run.rows[page][key])), key
        assert np.isfinite(run.rows[page]["gpu_logquad_pitch_x"])
