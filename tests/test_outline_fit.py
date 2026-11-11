"""Outline fit (subpx.fit_outline and measure_cds' outline_fit): geometry of the per-hole outline, a small page fit,
and its place among the other CD methods."""

import json
import math

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("cv2")

from subpx import fit_outline, load_cd_results, measure_cds, save_cd_results
from subpx._outline.model import Template
from subpx.outline import ellipse_widths, outline_rays
from subpx.synthetic_arrays import render_hole_array

PIXEL_NM = 21.0


def ellipse_theta(template, a, b, phi=0.0, coeffs=None):
    coeffs = np.zeros((len(template.ks), 2)) if coeffs is None else np.asarray(coeffs, float)
    return template.pack(a, b, 2.0, phi, coeffs, template.blur_pack(s1x=1.0, s1y=1.0))


def test_ellipse_widths_are_the_axes_and_the_area_is_exact():
    t = Template(upsample=8, size_px=24)
    theta = ellipse_theta(t, 4.0, 2.0)
    cd_x, cd_y = ellipse_widths(t, theta, np.array([0.0]))
    assert cd_x[0] == pytest.approx(8.0, rel=1e-6) and cd_y[0] == pytest.approx(4.0, rel=1e-6)
    # an offset outline: the widths keep cd_x cd_y = cd^2, with cd from the exact polar area
    delta = np.array([-0.3, 0.2])
    cd_x, cd_y = ellipse_widths(t, theta, delta)
    cd = np.array([t.cd_px(theta, float(d)) for d in delta])
    np.testing.assert_allclose(cd_x * cd_y, cd ** 2, rtol=1e-9)


def test_rays_follow_the_outline_about_its_centroid():
    t = Template(upsample=8, size_px=24)
    # a lopsided outline (first harmonic) moves its centroid off the origin; rays are measured from the centroid
    theta = ellipse_theta(t, 3.0, 2.5, phi=0.3, coeffs=[[0.08, -0.05], [0.02, 0.0]])
    rays = outline_rays(t, theta, np.array([0.0, 0.1]), n_rays=64)
    assert rays.shape == (2, 64) and (rays > 0).all()
    cx, cy = t.centroid_px(theta)
    angles = 2 * np.pi * np.arange(64) / 64
    for j in (0, 13, 40):
        # the ray's end lies on the outline: its polar radius about the origin equals r(angle)
        x, y = cx + rays[0, j] * np.cos(angles[j]), cy + rays[0, j] * np.sin(angles[j])
        r, _ = t._radius_at(theta, np.array([math.atan2(y, x)]))
        assert math.hypot(x, y) == pytest.approx(r[0], abs=1e-9)
    # an offset of 0.1 px along the normal lengthens every ray by about 0.1 px (more where the ray is oblique)
    assert np.all(rays[1] - rays[0] > 0.099) and np.all(rays[1] - rays[0] < 0.13)


@pytest.fixture(scope="module")
def page():
    return render_hole_array((120, 170), hole="circle", size_nm=(100, 100), pitch_nm=(210, 210), layout="square",
                             cd_sigma_nm=3.0, seed=5, backend="cpu")


@pytest.fixture(scope="module")
def measured(page):
    options = dict(pixel_size=PIXEL_NM, unit="nm", backend="cpu", area_max=200)
    plain = measure_cds(page.image, methods=("half_area",), **options)
    both = measure_cds(page.image, methods=("half_area", "outline_fit"), **options)
    return plain, both


def matched_truth(page, centers):
    from scipy.spatial import cKDTree

    distance, index = cKDTree(page.centers_xy).query(centers)
    return distance, np.sqrt(page.size_px[index, 0] * page.size_px[index, 1]) * PIXEL_NM, page.centers_xy[index]


def test_measure_cds_outline_fit_reads_the_size_at_slope_one(page, measured):
    plain, both = measured
    # adding the method changes neither the seeds nor the other methods' numbers
    np.testing.assert_array_equal(plain.seeds_xy, both.seeds_xy)
    np.testing.assert_array_equal(plain.measurements["half_area"].metrics["cd_equivalent_px"],
                                  both.measurements["half_area"].metrics["cd_equivalent_px"])
    m = both.measurements["outline_fit"]
    summary = both.meta["outline_fit"]
    assert summary["fitted"] and summary["per_hole_size"]
    assert m.valid.mean() > 0.95 and set(np.unique(m.status)) <= {"ok", "edge_offset_at_clip"}
    distance, true_cd, true_xy = matched_truth(page, m.centers_xy)
    ok = m.valid & (distance < 1)
    cd = m.metrics["cd_equivalent_px"][ok] * PIXEL_NM
    slope = np.polyfit(true_cd[ok], cd, 1)[0]
    # 100 nm holes under a 65 nm blur: the outline fit follows the true size; the half-height area does not
    assert abs(cd.mean() - true_cd[ok].mean()) < 1.5
    assert 0.85 < slope < 1.15
    half = both.measurements["half_area"]
    good = ok & half.valid
    assert np.polyfit(true_cd[good], half.metrics["cd_equivalent_px"][good] * PIXEL_NM, 1)[0] < 0.8
    # centres to a fraction of a pixel, and a standard error of about a nanometre
    assert np.sqrt(np.mean(np.sum((m.centers_xy[ok] - true_xy[ok]) ** 2, axis=1))) * PIXEL_NM < 1.5
    assert 0.3 < np.nanmedian(m.metrics["cd_se_px"]) * PIXEL_NM < 3.0


def test_outline_fit_outline_matches_its_cd_and_survives_a_compact_export(measured, tmp_path):
    _, both = measured
    m = both.measurements["outline_fit"]
    points = m.contours_xy[:, :-1]
    x, y = points[..., 0], points[..., 1]
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1))
    np.testing.assert_allclose(2 * np.sqrt(area[m.valid] / np.pi), m.metrics["cd_equivalent_px"][m.valid], atol=0.005)
    json.dumps(both.meta)
    save_cd_results(both, tmp_path / "results", compact=True)
    back = load_cd_results(tmp_path / "results").measurements["outline_fit"]
    np.testing.assert_array_equal(back.contours_xy, m.contours_xy)
    np.testing.assert_array_equal(back.status, m.status)
    for name, values in m.metrics.items():
        np.testing.assert_array_equal(back.metrics[name], values)


def test_fit_outline_leaves_the_seeds_alone_and_checks_its_input(page):
    seeds = page.centers_xy[(page.centers_xy > 1).all(axis=1)][:12].copy()
    before = seeds.copy()
    fit = fit_outline(page.image, seeds, pixel_size=PIXEL_NM, unit="nm", rounds=1, backend="cpu")
    np.testing.assert_array_equal(seeds, before)
    assert fit.cd_px.shape == (12,) and fit.rays_px.shape == (12, 128) and fit.rounds == 1
    with pytest.raises(ValueError, match="at least"):
        fit_outline(page.image, seeds[:3], backend="cpu")
    with pytest.raises(ValueError, match="blur"):
        fit_outline(page.image, seeds, blur="airy", backend="cpu")
    with pytest.raises(ValueError, match="outline_blur"):
        measure_cds(page.image, methods=("outline_fit",), outline_blur="airy", backend="cpu")
    with pytest.raises(ValueError, match="background='model'"):
        measure_cds(page.image, methods=("outline_fit",), background="border", backend="cpu")


def test_a_page_without_holes_gives_no_fit_and_no_error():
    result = measure_cds(np.zeros((16, 16), np.uint8), methods=("outline_fit",), backend="cpu")
    assert result.meta["outline_fit"]["fitted"] is False
    assert len(result.measurements["outline_fit"].valid) == len(result.seeds_xy)


def test_gpu_fit_matches_the_cpu_fit(page):
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("no CUDA device")
    except Exception:
        pytest.skip("CUDA unavailable")
    seeds = measure_cds(page.image, methods=("half_area",), backend="cpu", area_max=200).seeds_xy
    cpu = fit_outline(page.image, seeds, rounds=4, backend="cpu")
    gpu = fit_outline(page.image, seeds, rounds=4, backend="gpu")
    assert isinstance(gpu.cd_px, np.ndarray)
    # The GPU's sums use atomic adds (their order varies from run to run; CPU runs repeat exactly). Over 6 GPU runs
    # of 4 rounds the hole CDs fell in two groups, about 1e-5 px and about 3e-4 px from the CPU's (at most 3.7e-4 px
    # apart, 0.008 nm at 21 nm pixels), the centres within 7e-5 px, the outline CD within 2e-5 px. The limits are
    # about three times that.
    assert gpu.outline_cd_px == pytest.approx(cpu.outline_cd_px, abs=1e-4)
    np.testing.assert_allclose(gpu.cd_px, cpu.cd_px, atol=1e-3)
    np.testing.assert_allclose(gpu.centers_xy, cpu.centers_xy, atol=2e-4)


def test_report_marks_the_spread_unmeasured_when_every_hole_got_the_page_size(measured):
    from dataclasses import replace

    from scripts import cd_report as report

    _, both = measured
    stats = report.cd_statistics(both)["outline_fit"]
    assert np.isfinite(stats["std"]) and np.isfinite(stats["three_sigma_pct"])
    page_size = replace(both, meta={**both.meta, "outline_fit": {**both.meta["outline_fit"], "per_hole_size": False}})
    stats = report.cd_statistics(page_size)["outline_fit"]
    assert np.isfinite(stats["mean"]) and np.isnan(stats["std"]) and np.isnan(stats["three_sigma_pct_x"])
    summary = report.outline_summary(page_size, "nm")
    assert "per-hole size off" in report.outline_line(summary)
    assert "within-image sigma is not measured" in report.outline_note("page.png", summary)
    assert report.outline_line(report.outline_summary(both, "nm"), ascii_only=True).isascii()


@pytest.mark.parametrize("script", ["cd_report", "cd_fast_report"])
def test_reports_measure_the_outline_fit_on_request(page, tmp_path, script):
    import csv
    import importlib

    from PIL import Image

    module = importlib.import_module(f"scripts.{script}")
    folder = tmp_path / "in"
    folder.mkdir()
    Image.fromarray(page.image.astype(np.uint8)).save(folder / "page.png")
    output = tmp_path / "out"
    options = ["--methods", "half_area"] if script == "cd_report" else []
    module.main([str(folder), "--output", str(output), "--pixel-size", "21", "--unit", "nm", "--backend", "cpu",
                 "--no-hole-details", "--outline-fit", "--outline-rounds", "2", *options])
    with (output / "between_images.csv").open(newline="", encoding="utf-8") as handle:
        between = {row["method"]: row for row in csv.DictReader(handle)}
    with (output / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["method"]: row for row in csv.DictReader(handle)}
    assert float(between["outline_fit"]["mean_of_means"]) == pytest.approx(float(rows["outline_fit"]["mean"]))
    assert 0 < float(rows["outline_fit"]["three_sigma_pct"]) < 30
    page_html = (output / "index.html").read_text(encoding="utf-8")
    assert "Outline fit:" in page_html and "did not converge in 2 rounds" in page_html
