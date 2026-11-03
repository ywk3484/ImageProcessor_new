"""scripts/pitch_signature_test.py scores (true centers, shared and independent errors) and the template-fit prototype."""

import json

import numpy as np
import pytest

pytest.importorskip("scipy")

from scripts.pitch_signature_test import PIXEL_NM, evaluate, evaluation_set, main, match_points, pitch_pairs  # noqa: E402
from scripts.template_fit_prototype import template_fit_centers  # noqa: E402
from subpx.synthetic_arrays import render_hole_array  # noqa: E402

QUIET = dict(photons_per_level=None, bits=None, read_noise=0.0, backend="cpu")


def _score(page, centers):
    use = evaluation_set(page)
    return evaluate(page, centers, use, pitch_pairs(page, use))


def test_true_centers_score_perfect():
    page = render_hole_array((160, 240), placement_sigma_nm=0.0, seed=4, **QUIET)
    r = _score(page, page.centers_xy.copy())
    assert r["found"] == 1.0 and r["center_sd_y"] == 0.0 and r["phase_error"] == 0.0
    assert r["spike"] == pytest.approx(1.0, abs=1e-9)  # the planted 1 nm step, nothing else
    assert r["need_seen"] == 1 and r["need_method"] == 1
    assert max(r["noise_seen"].values()) < 1e-9


def test_row_shared_error_never_averages_away():
    """An error shared by a whole row (as a pixel-position error is when rows run along the pixel rows) stays at
    any number of columns; the same size of independent per-hole error shrinks with sqrt(columns)."""
    page = render_hole_array((300, 400), seed=5, **QUIET)
    rng = np.random.default_rng(0)
    rows = page.indices_rc[:, 0]
    labels, inverse = np.unique(rows, return_inverse=True)
    shared, independent = page.centers_xy.copy(), page.centers_xy.copy()
    shared[:, 1] += rng.normal(0.0, 0.02, len(labels))[inverse]
    independent[:, 1] += rng.normal(0.0, 0.02, len(rows))
    a, b = _score(page, shared), _score(page, independent)
    assert a["noise_method"]["all"] == pytest.approx(a["noise_method"]["1"], rel=1e-6)
    assert a["need_method"] is None
    assert b["noise_method"]["all"] < b["noise_method"]["1"] / np.sqrt(b["columns_total"]) * 1.3
    assert b["need_method"] is not None


def test_pixel_position_error_is_measured():
    page = render_hole_array((300, 400), seed=6, rotation_deg=2.0, **QUIET)  # tilted: holes at every height in a pixel
    centers = page.centers_xy.copy()
    centers[:, 1] += 0.01 * np.sin(2 * np.pi * (centers[:, 1] - np.rint(centers[:, 1])))
    r = _score(page, centers)
    expected = 0.01 * PIXEL_NM * np.sinc(1 / 16)  # half the peak-to-peak of the 16-bin means
    assert r["phase_error"] == pytest.approx(expected, abs=0.01)


def test_match_points_gives_each_hole_its_nearest_point():
    truth = np.array([[10.0, 10.0], [20.0, 10.0], [30.0, 10.0]])
    points = np.array([[20.3, 10.0], [10.2, 10.1], [19.9, 10.0], [50.0, 50.0], [np.nan, 1.0]])
    out = match_points(points, truth, tolerance=1.0)
    np.testing.assert_allclose(out[:2], [[10.2, 10.1], [19.9, 10.0]])
    assert np.isnan(out[2]).all()


def test_template_prototype_recovers_true_centers():
    page = render_hole_array((200, 300), seed=8, row_shift_nm=0.0, cd_sigma_nm=0.0, **QUIET)
    rng = np.random.default_rng(1)
    start = page.centers_xy + rng.uniform(-0.15, 0.15, page.centers_xy.shape)
    start[5] = np.nan  # an unusable start keeps its place in the output
    fit = template_fit_centers(page.clean, start, backend="cpu")
    assert fit.shape == start.shape and np.isnan(fit[5]).all()
    use = evaluation_set(page)
    use[5] = False
    error = fit[use] - page.centers_xy[use]
    assert np.isfinite(error).all()
    assert np.abs(error.mean(axis=0)).max() < 0.005  # px; one shared offset does not change any pitch
    assert (error - error.mean(axis=0)).std(axis=0).max() < 0.004  # px per hole, 0.08 nm


def test_end_to_end_on_cpu(tmp_path):
    data = main(["--images", "circle100_square", "--shape", "160", "240", "--cd-methods", "gaussian", "--finders",
                 "--no-template", "--backend", "cpu", "--render-backend", "cpu", "--output", str(tmp_path)])
    for name in ("results.json", "settings.json", "pitch_signature.csv", "report.html"):
        assert (tmp_path / name).stat().st_size > 0
    image = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))["images"][0]
    assert set(image["methods"]) == {"gaussian", "truth"}
    truth, gaussian = image["methods"]["truth"], image["methods"]["gaussian"]
    assert truth["center_sd_y"] == 0.0 and truth["need_method"] == 1
    assert gaussian["found"] >= 0.99 and 0.1 < gaussian["center_sd_y"] < 1.5
    assert image["best_cd"] == "gaussian" and image["best_gpu"] is None
    assert 0.2 < image["limit_nm"] < 0.6
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "<title>Pitch Signature Test</title>" in html and "circle100_square" in html
    assert data["images"][0]["key"] == "circle100_square"
