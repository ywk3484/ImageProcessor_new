"""One-pass outline fit (subpx.fit_outline_fast and measure_cds' outline_fit_fast): the page fit's pieces under
NumPy, the method's place among the other CD methods without a GPU, and the fit itself on generated images (GPU).

The limits of the GPU tests are errors of the page CD on generated images, not targets: with 135 stacked holes the
fit's own standard error is about 0.25 nm. Numbers in brackets are the prototype's (docs/outline_fit_fast_handoff.md).
"""

import csv
import importlib
import re

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("cv2")

from subpx import fit_outline_fast, load_cd_results, measure_cds, save_cd_results
from subpx._outline.model import Template
from subpx._outline.onepass import Band, ShapeFitter, Spec, exchanged, second_is_narrower
from subpx.outline import blur_second, fast_line, fast_summary
from subpx.synthetic_arrays import render_hole_array

PIXEL_NM = 21.0


# ------------------------------------------------------------------------------------------- no GPU needed
@pytest.mark.parametrize("blur, box", [("gauss2", 24), ("lens", 48)])
def test_analytic_template_matches_the_rendered_one(blur, box):
    t = Template(upsample=8, size_px=box, blur=blur)
    spec, band = Spec(t, n_poly=512), Band(box_px=box, f_max=1.0, upsample=8)
    rng = np.random.default_rng(1)
    for _ in range(2):
        a, b = rng.uniform(1.6, 3.2, 2)
        blur_params = t.blur_pack(s1x=rng.uniform(1.0, 1.5), s1y=rng.uniform(1.0, 1.5), s2x=rng.uniform(2, 3),
                                  s2y=rng.uniform(2, 3), w2=rng.uniform(0.02, 0.2), sx=rng.uniform(0.6, 1.2),
                                  sy=rng.uniform(0.6, 1.2), cutoff=rng.uniform(0.35, 0.6))
        theta = t.pack(a, b, rng.uniform(1.7, 3.5), rng.uniform(-0.3, 0.3), rng.uniform(-0.03, 0.03, (2, 2)), blur_params)
        for delta, bin_px in ((0.0, 0.0), (0.1, 0.0), (-0.1, 0.125)):  # with and without an edge offset and the bin box
            rendered = t.render(theta, delta, box_px=bin_px)
            analytic = band.to_grid(spec.template_hat(theta, band.fx, band.fy, delta, box_px=bin_px))
            assert np.abs(analytic - rendered).max() < 3e-4 * rendered.max()  # [1.2e-4]


@pytest.mark.parametrize("blur", ["gauss2", "lens"])
def test_model_derivatives_match_central_differences(blur):
    t = Template(upsample=8, size_px=24, blur=blur)
    sf = ShapeFitter(t, box_px=48, f_max=0.6, U=4, R=7, n_poly=256, xp=np)
    # clearly not round: the turn angle has no effect on a round outline
    theta = t.pack(2.9, 2.1, 2.6, 0.3, [[0.03, -0.02], [0.02, 0.015]],
                   t.blur_pack(s1x=1.2, s1y=1.35, s2x=2.4, s2y=2.1, w2=0.08, sx=1.1, sy=1.25, cutoff=0.45))
    gamma = np.ones((1, sf.U, sf.U, sf.nky, sf.nkx), dtype=complex)
    model = sf.synth(sf.template_hat(theta[None], np.eye(sf.P)), gamma)[0]  # (1 + P, U, U, bins, bins)
    for k, name in enumerate(t.names):
        step = np.zeros(sf.P)
        step[k] = 1e-5
        numeric = (sf.synth(sf.template_hat((theta + step)[None]), gamma)[0, 0]
                   - sf.synth(sf.template_hat((theta - step)[None]), gamma)[0, 0]) / 2e-5
        assert np.abs(model[1 + k] - numeric).max() < 1e-3 * np.abs(numeric).max(), name  # [2e-4 at most]


@pytest.mark.parametrize("blur", ["gauss2", "lens"])
def test_page_fit_recovers_a_known_outline(blur):
    t = Template(upsample=8, size_px=24, blur=blur)
    sf = ShapeFitter(t, box_px=48, f_max=0.6, U=4, R=7, n_poly=256, xp=np)
    truth = t.pack(2.4, 2.3, 2.2, 0.2, np.zeros((2, 2)),
                   t.blur_pack(s1x=1.27, s1y=1.27, s2x=2.5, s2y=2.5, w2=0.05, sx=1.2, sy=1.2, cutoff=0.45))
    gamma = np.ones((1, sf.U, sf.U, sf.nky, sf.nkx), dtype=complex)  # a hole without neighbours
    mask = sf.pad[:, None, :, None] & sf.pad[None, :, None, :]
    sd = 0.16  # grey levels per bin on a hole of 180
    clean = 180.0 * sf.synth(sf.template_hat(truth[None]), gamma)[0, 0] + 40.0
    y = np.where(mask, clean + np.random.default_rng(0).normal(0, sd, clean.shape), 0.0)
    w = np.where(mask, 1.0 / sd ** 2, 0.0)
    true_cd = t.cd_px(truth) * PIXEL_NM
    found = []
    for start, grid in ((t.start(1.7, 1.0), True), (truth, False)):
        theta, scale, const, cost, info = sf.fit(start[None], gamma, y[None], w[None], grid=grid)
        found.append(t.cd_px(theta[0]) * PIXEL_NM)
        assert abs(found[-1] - true_cd) < 0.05  # [-0.016 gauss2, -0.020 lens]
        assert 0.9 < cost[0] / mask.sum() < 1.1 and scale[0] == pytest.approx(180, rel=0.02)
        assert info[-1][0] == "full" and info[-1][2].shape == (1,)
    assert found[0] == pytest.approx(found[1], abs=1e-3)  # the grid of starts and a start at the truth


def test_gauss2_exchange_is_the_same_blur_under_the_other_labels():
    t = Template(upsample=8, size_px=24, blur="gauss2")
    sf = ShapeFitter(t, box_px=56, f_max=0.6, U=4, R=8, n_poly=256, xp=np)
    theta = t.pack(2.4, 2.3, 2.2, 0.1, [[0.01, -0.02], [0.015, 0.005]],
                   t.blur_pack(s1x=1.6, s1y=1.5, s2x=1.2, s2y=1.1, w2=0.45))
    other = exchanged(t, theta)
    assert second_is_narrower(t, theta) and not second_is_narrower(t, other)
    np.testing.assert_allclose(sf.template_hat(other[None]), sf.template_hat(theta[None]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(exchanged(t, other), theta, atol=1e-12)
    # a weak second Gaussian has no exchanged twin inside the bounds: its weight stops at 0.6
    edge = exchanged(t, t.pack(2.4, 2.3, 2.2, 0.1, np.zeros((2, 2)), t.blur_pack(w2=0.05)))
    assert 1 / (1 + np.exp(-edge[-1])) == pytest.approx(0.6, abs=1e-3)
    assert blur_second(theta, "gauss2") == "second Gaussian 1.2 x 1.1 px, 45 % of the light"
    lens = Template(upsample=8, size_px=24, blur="lens")
    assert blur_second(lens.start(2.0), "lens") == "pupil cut-off 0.45 cycles/px"


@pytest.fixture(scope="module")
def page():
    return render_hole_array((120, 170), hole="circle", size_nm=(100, 100), pitch_nm=(210, 210), layout="square",
                             cd_sigma_nm=3.0, seed=5, backend="cpu")


def test_without_a_gpu_or_without_holes_the_method_reports_no_fit_and_loses_nothing(page):
    options = dict(pixel_size=PIXEL_NM, unit="nm", backend="cpu", area_max=200)
    plain = measure_cds(page.image, methods=("half_area",), **options)
    both = measure_cds(page.image, methods=("half_area", "outline_fit_fast"), **options)
    np.testing.assert_array_equal(plain.seeds_xy, both.seeds_xy)
    np.testing.assert_array_equal(plain.measurements["half_area"].metrics["cd_equivalent_px"],
                                  both.measurements["half_area"].metrics["cd_equivalent_px"])
    fast = both.measurements["outline_fit_fast"]
    assert both.meta["outline_fit_fast"]["fitted"] is False and "GPU" in both.meta["outline_fit_fast"]["reason"]
    assert not fast.valid.any() and set(fast.status) == {"needs_gpu"} and len(fast.valid) == len(plain.measurements["half_area"].valid)
    summary = fast_summary(both.meta["outline_fit_fast"], PIXEL_NM, "nm")
    assert fast_line(summary, ascii_only=True) == "not fitted (needs the GPU (CuPy); this run uses the CPU)"
    empty = measure_cds(np.zeros((16, 16), np.uint8), methods=("outline_fit_fast",), backend="cpu")
    assert empty.meta["outline_fit_fast"]["fitted"] is False and "fewer than" in empty.meta["outline_fit_fast"]["reason"]
    assert len(empty.measurements["outline_fit_fast"].valid) == len(empty.seeds_xy)


def test_fit_outline_fast_checks_its_input(page):
    seeds = page.centers_xy[(page.centers_xy > 1).all(axis=1)][:12]
    with pytest.raises(ValueError, match="at least"):
        fit_outline_fast(page.image, seeds[:3])
    with pytest.raises(ValueError, match="blur"):
        fit_outline_fast(page.image, seeds, blur="airy")
    with pytest.raises(RuntimeError, match="needs the GPU"):
        fit_outline_fast(page.image, seeds, backend="cpu")
    with pytest.raises(ValueError, match="outline_blur"):
        measure_cds(page.image, methods=("outline_fit_fast",), outline_blur="airy", backend="cpu")
    with pytest.raises(ValueError, match="background='model'"):
        measure_cds(page.image, methods=("outline_fit_fast",), background="border", backend="cpu")


def test_report_on_the_cpu_says_the_fast_fit_needs_the_gpu(page, tmp_path):
    from PIL import Image

    from scripts import cd_report

    folder = tmp_path / "in"
    folder.mkdir()
    Image.fromarray(page.image.astype(np.uint8)).save(folder / "page.png")
    cd_report.main([str(folder), "--output", str(tmp_path / "out"), "--pixel-size", "21", "--unit", "nm", "--backend",
                    "cpu", "--no-hole-details", "--methods", "half_area", "--outline-fit-fast"])
    page_html = (tmp_path / "out" / "index.html").read_text(encoding="utf-8")
    assert "page.png: outline_fit_fast not fitted (needs the GPU" in page_html
    assert "Outline fit (fast):" in page_html and "outline_fit_fast - outline_fit" not in page_html


# ------------------------------------------------------------------------------------------------- GPU
@pytest.fixture(scope="module")
def gpu():
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("no CUDA device")
    except Exception:
        pytest.skip("CUDA unavailable")


def stacked_truth(image_page, fit):
    """Mean true CD (nm) of the stacked holes and the rms distance (nm) of their centres from the true ones."""
    from scipy.spatial import cKDTree

    distance, index = cKDTree(image_page.centers_xy).query(fit.centers_xy)
    ok = fit.good & (distance < 1.5)
    true_cd = np.sqrt(image_page.size_px[index[ok], 0] * image_page.size_px[index[ok], 1]) * PIXEL_NM
    return float(true_cd.mean()), float(np.sqrt(np.mean(distance[ok] ** 2)) * PIXEL_NM)


CASES = {  # image: shape, pitch (nm), cd sigma (nm), seed, limit of the page CD's error (nm), holes stacked
    "library test page": ((120, 170), 210, 3.0, 5, 1.5, 135),  # [+0.63 gauss2, +0.31 lens]; pitch exactly 10 px
    "300 x 400": ((300, 400), 210, 3.0, 6, 0.8, 999),  # [+0.34, +0.16]
    "600 x 800": ((600, 800), 200, 2.5, 7, 0.5, 4941),  # [+0.09, -0.06]
}


@pytest.fixture(scope="module")
def fits(gpu, page):
    """Every case fitted with both blur forms, from measure_cds' seeds and from the method's own."""
    out = {}
    for name, (shape, pitch, sigma, seed, _, _) in CASES.items():
        image_page = page if seed == 5 else render_hole_array(
            shape, hole="circle", size_nm=(100, 100), pitch_nm=(pitch, pitch), layout="square", cd_sigma_nm=sigma,
            seed=seed, backend="cpu")
        seeds = measure_cds(image_page.image, methods=("half_area",), backend="cpu", area_max=200).seeds_xy
        for blur in ("gauss2", "lens"):
            for source, given in (("report", seeds), ("own", None)):
                out[name, blur, source] = fit_outline_fast(image_page.image, given, pixel_size=PIXEL_NM, unit="nm",
                                                           blur=blur)
        out[name, "page"] = image_page
    return out


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("blur", ["gauss2", "lens"])
def test_page_cd_on_generated_images(fits, name, blur):
    limit, stacked = CASES[name][4:]
    fit = fits[name, blur, "report"]
    true_cd, centre_rms = stacked_truth(fits[name, "page"], fit)
    assert isinstance(fit.centers_xy, np.ndarray) and fit.rays_px.shape == (128,)
    assert abs(fit.outline_cd_px * PIXEL_NM - true_cd) < limit
    assert 0.5 < fit.summary["leftover_chi2"] < 2  # [0.9 to 1.1]
    assert fit.summary["holes_stacked"] == int(fit.good.sum()) == stacked
    assert centre_rms < 1.5  # [0.53 nm, from seeds 3.1 nm off]
    assert fit.cd_x_px * fit.cd_y_px == pytest.approx(fit.outline_cd_px ** 2, rel=1e-9)
    # The seeds handed in must not decide the page CD. gauss2 without the exchange step: 0.62, 0.62, 0.40 nm apart;
    # with it [0.005, 0.005, 0.004 nm]; lens [0.02 nm at most]
    own = fits[name, blur, "own"]
    assert abs(own.outline_cd_px - fit.outline_cd_px) * PIXEL_NM < 0.05
    assert fit.summary["exchanged"] in (True, False) and (blur == "gauss2" or not fit.summary["exchanged"])


def test_the_same_image_fitted_twice_reads_the_same(fits, page):
    again = fit_outline_fast(page.image, None, pixel_size=PIXEL_NM, unit="nm")
    assert again.outline_cd_px == pytest.approx(fits["library test page", "gauss2", "own"].outline_cd_px, abs=1e-3)


def test_dark_holes_read_the_same_as_bright_ones(gpu, page):
    seeds = measure_cds(page.image, methods=("half_area",), backend="cpu", area_max=200).seeds_xy
    bright = fit_outline_fast(page.image, seeds)
    dark = fit_outline_fast(255 - page.image.astype(np.int32), seeds, invert=True)
    assert dark.outline_cd_px == pytest.approx(bright.outline_cd_px, abs=1e-3)
    # the level comes back in the dark image's own grey levels
    np.testing.assert_allclose(dark.level[dark.good], 255 - bright.level[bright.good], atol=0.05)


@pytest.fixture(scope="module")
def measured(gpu, page):
    return {blur: measure_cds(page.image, methods=("half_area", "outline_fit_fast"), pixel_size=PIXEL_NM, unit="nm",
                              backend="gpu", area_max=200, outline_blur=blur) for blur in ("gauss2", "lens")}


@pytest.mark.parametrize("blur", ["gauss2", "lens"])
def test_measure_cds_gives_every_hole_the_page_cd(page, measured, blur):
    from scipy.spatial import cKDTree

    from scripts import cd_report as report

    result = measured[blur]
    m, meta = result.measurements["outline_fit_fast"], result.meta["outline_fit_fast"]
    assert meta["fitted"] and meta["per_hole_size"] is False and meta["blur_family"] == blur
    assert m.valid.mean() > 0.95 and set(m.status[m.valid]) == {"page_size"}
    cd = m.metrics["cd_equivalent_px"][m.valid]
    assert np.ptp(cd) == 0 and cd[0] == pytest.approx(meta["outline_cd_px"])
    distance, index = cKDTree(page.centers_xy).query(m.centers_xy)
    ok = m.valid & (distance < 1)
    true_cd = np.sqrt(page.size_px[index[ok], 0] * page.size_px[index[ok], 1]) * PIXEL_NM
    assert abs(cd[0] * PIXEL_NM - true_cd.mean()) < 1.5
    assert np.sqrt(np.mean(distance[ok] ** 2)) * PIXEL_NM < 1.5
    # the outline drawn per hole has the page's area
    x, y = m.contours_xy[m.valid][:, :-1, 0], m.contours_xy[m.valid][:, :-1, 1]
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1))
    np.testing.assert_allclose(2 * np.sqrt(area / np.pi), cd, atol=0.005)
    # the report: a mean, no spread; an ASCII line for the console
    stats = report.cd_statistics(result)["outline_fit_fast"]
    assert np.isfinite(stats["mean"]) and np.isnan(stats["std"]) and np.isnan(stats["three_sigma_pct_x"])
    assert np.isfinite(report.cd_statistics(result)["half_area"]["std"])
    line = report.outline_fast_line(report.outline_fast_summary(result, "nm"), ascii_only=True)
    assert line.isascii() and "page size for every hole" in line and "outline CD" in line


def test_results_survive_a_compact_export(measured, tmp_path):
    result = measured["gauss2"]
    m = result.measurements["outline_fit_fast"]
    save_cd_results(result, tmp_path / "results", compact=True)
    back = load_cd_results(tmp_path / "results").measurements["outline_fit_fast"]
    np.testing.assert_array_equal(back.contours_xy, m.contours_xy)
    np.testing.assert_array_equal(back.status, m.status)
    for name, values in m.metrics.items():
        np.testing.assert_array_equal(back.metrics[name], values)


@pytest.mark.parametrize("script", ["cd_report", "cd_fast_report"])
def test_reports_compare_the_two_outline_fits(gpu, page, tmp_path, script, capsys):
    from PIL import Image

    module = importlib.import_module(f"scripts.{script}")
    folder = tmp_path / "in"
    folder.mkdir()
    Image.fromarray(page.image.astype(np.uint8)).save(folder / "page.png")
    output = tmp_path / "out"
    options = ["--methods", "half_area"] if script == "cd_report" else []
    module.main([str(folder), "--output", str(output), "--pixel-size", "21", "--unit", "nm", "--backend", "gpu",
                 "--no-hole-details", "--outline-fit", "--outline-rounds", "2", "--outline-fit-fast", *options])
    printed = capsys.readouterr().out.strip().splitlines()
    with (output / "between_images.csv").open(newline="", encoding="utf-8") as handle:
        between = {row["method"]: row for row in csv.DictReader(handle)}
    with (output / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["method"]: row for row in csv.DictReader(handle)}
    for method in ("outline_fit", "outline_fit_fast"):
        assert float(between[method]["mean_of_means"]) == pytest.approx(float(rows[method]["mean"]))
    assert rows["outline_fit_fast"]["std"] in ("", "nan") and float(rows["outline_fit"]["std"]) > 0
    # the last line printed is the comparison: short, ASCII, the difference of the two outline CDs (outline_fit's
    # mean over its holes, the summary's number, is within a few hundredths of a nanometre of its outline CD)
    last = printed[-1]
    match = re.fullmatch(r"outline_fit_fast - outline_fit: mean ([+-]\d+\.\d\d), largest (\d+\.\d\d) nm over 1 image; "
                         r"\d+\.\d\d s against \d+ s per image", last)
    assert match, last
    difference = float(rows["outline_fit_fast"]["mean"]) - float(rows["outline_fit"]["mean"])
    assert float(match[1]) == pytest.approx(difference, abs=0.05) and float(match[2]) == abs(float(match[1]))
    assert last.isascii() and len(last) <= 150
    assert any(line.strip().startswith("blur second part: outline_fit second Gaussian") for line in printed)
    page_html = (output / "index.html").read_text(encoding="utf-8")
    for text in ("Outline fit:", "Outline fit (fast):", "blur second part: outline_fit second Gaussian",
                 "; outline_fit_fast second Gaussian", last, "outline_fit_fast gives every hole the image&#x27;s size"):
        assert text in page_html, text
