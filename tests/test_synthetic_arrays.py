"""The test-image generator renders holes whose true centers are exact (subpx/synthetic_arrays.py)."""

import math

import numpy as np
import pytest

from subpx import analyze_pitch
from subpx.synthetic_arrays import (LENS_FWHM_FACTOR, _profile_width, _render, blur_smear_sigma,
                                    render_hole_array)
from subpx.types import CDMethodResult, CDSummary

QUIET = dict(photons_per_level=None, read_noise=0.0, bits=None, cd_sigma_nm=0.0, placement_sigma_nm=0.0,
             brightness_sigma=0.0, row_shift_nm=0.0, backend="cpu")


def single(hole, size, **options):
    """One noiseless hole alone on a page (pitch far larger than the page)."""
    return render_hole_array((96, 96), hole=hole, size_nm=size, pitch_nm=(4000.0, 4000.0), **{**QUIET, **options})


@pytest.mark.parametrize("hole, size", [("circle", (100.0, 100.0)), ("square", (300.0, 300.0)),
                                        ("ellipse", (160.0, 80.0))])
def test_gaussian_blur_hole_has_exact_flux_centroid_and_spread(hole, size):
    page = single(hole, size, lens_nm=0.0, seed=4)
    (center,) = page.centers_xy[page.inside]
    light = page.clean - 15.0
    yy, xx = np.indices(light.shape)
    total = light.sum()
    ax, ay = np.asarray(size) / 21.0
    area = ax * ay if hole == "square" else math.pi * ax * ay / 4
    np.testing.assert_allclose(total, (103.0 - 15.0) * area, rtol=1e-12)
    np.testing.assert_allclose([(light * xx).sum() / total, (light * yy).sum() / total], center, atol=1e-9)
    # Spread = outline + blur + pixel area (variances add): circle/ellipse axis a -> a^2/16, side w -> w^2/12.
    sigma = (65.0 / 21.0) / (2 * math.sqrt(2 * math.log(2)))
    outline = (np.array([ax, ay]) ** 2) / (12 if hole == "square" else 16)
    measured = [(light * (xx - center[0]) ** 2).sum() / total, (light * (yy - center[1]) ** 2).sum() / total]
    np.testing.assert_allclose(measured, outline + sigma ** 2 + 1 / 12, rtol=1e-9)


def test_blur_is_as_wide_as_asked():
    lens, blur = 45.0 / 21.0, 65.0 / 21.0
    assert _profile_width(lens, blur_smear_sigma(blur, lens)) == pytest.approx(blur, abs=1e-8)
    assert blur_smear_sigma(blur, 0.0) == pytest.approx(blur / 2.3548200450309493)


@pytest.mark.parametrize("hole, size", [("circle", (100.0 / 21, 100.0 / 21)), ("square", (300.0 / 21, 300.0 / 21))])
def test_lens_tail_folded_by_the_64px_patch_does_not_move_centers(hole, size):
    """The lens's faint far tail folds back into each hole's periodic patch; near the hole it is negligible."""
    lens, blur = 45.0 / 21.0, 65.0 / 21.0
    sigma, cutoff = blur_smear_sigma(blur, lens), LENS_FWHM_FACTOR / lens
    center = np.array([[120.3, 120.37]])
    pages = {p: _render((241, 241), center, np.array([size]), np.array([88.0]), hole, 0.0, p, 1, cutoff, sigma, "cpu")
             for p in (64, 256)}
    yy, xx = np.indices((241, 241))
    near = np.hypot(xx - center[0, 0], yy - center[0, 1]) < 15
    difference = pages[64][near] - pages[256][near]
    pull = np.array([(difference * (xx[near] - center[0, 0])).sum(), (difference * (yy[near] - center[0, 1])).sum()])
    assert np.abs(pull / pages[256][near].sum()).max() < 1e-4  # px, i.e. < 0.002 nm at 21 nm


@pytest.mark.parametrize("layout, pitch", [("square", (200.0, 200.0)), ("staggered", (200.0, 175.0))])
def test_planted_row_shift_and_indices_follow_analyze_pitch(layout, pitch):
    page = render_hole_array((260, 420), hole="circle", size_nm=(100.0, 100.0), pitch_nm=pitch, layout=layout,
                             seed=7, **{**QUIET, "row_shift_nm": 1.0})
    rows = page.indices_rc[:, 0]
    every, first = page.meta["row_shift_every"], page.meta["row_shift_first"]
    assert every == round(1000.0 / pitch[1])
    expected = np.where((rows - first) % every == 0, 1.0, 0.0)
    np.testing.assert_allclose(page.row_shift_nm, expected)
    np.testing.assert_allclose((page.centers_xy - page.design_xy) * 21.0, np.column_stack((0 * expected, expected)),
                               atol=1e-12)
    inside = page.inside
    seeds = page.centers_xy[inside]
    n = len(seeds)
    truth = CDMethodResult("truth", seeds, np.empty((n, 0, 2)), np.ones(n, bool), np.full(n, "ok"), {})
    found = analyze_pitch(CDSummary(seeds, {"truth": truth}, pixel_size=21.0, unit="nm"), layout=layout)["truth"]
    assert found.layout == layout
    np.testing.assert_array_equal(found.indices_rc, page.indices_rc[inside])
    deviation = found.profiles["y"].pitch - pitch[1]
    shifted = np.array(page.meta["shifted_rows"])
    np.testing.assert_allclose(deviation[shifted[shifted > 0] - 1], 1.0, atol=1e-9)  # into a shifted row
    np.testing.assert_allclose(deviation[shifted[shifted < len(deviation)]], -1.0, atol=1e-9)  # out of it


def test_same_seed_same_page_and_noise_statistics():
    a = render_hole_array((128, 160), seed=11, backend="cpu")
    b = render_hole_array((128, 160), seed=11, backend="cpu")
    np.testing.assert_array_equal(a.image, b.image)
    assert a.image.dtype == np.uint8
    # Photon noise (30 photons per grey level) + read noise 0.65 + rounding, pixel by pixel.
    expected = a.clean / 30.0 + 0.65 ** 2 + 1 / 12
    assert np.mean((a.image - a.clean) ** 2 / expected) == pytest.approx(1.0, abs=0.03)
    c = render_hole_array((128, 160), seed=12, backend="cpu")
    assert not np.array_equal(a.centers_xy, c.centers_xy)


def test_gpu_renders_the_same_page():
    pytest.importorskip("cupy")
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device unavailable")
    options = dict(hole="ellipse", size_nm=(160.0, 80.0), pitch_nm=(320.0, 200.0), seed=5)
    cpu = render_hole_array((160, 200), backend="cpu", **options)
    gpu = render_hole_array((160, 200), backend="gpu", **options)
    np.testing.assert_allclose(gpu.clean, cpu.clean, atol=1e-9)
    np.testing.assert_array_equal(gpu.image, cpu.image)


def test_rejects_bad_settings():
    with pytest.raises(ValueError, match="circle needs equal"):
        render_hole_array((64, 64), hole="circle", size_nm=(100, 80))
    with pytest.raises(ValueError, match="hole must be"):
        render_hole_array((64, 64), hole="star")
    with pytest.raises(ValueError, match="narrower"):
        render_hole_array((64, 64), blur_nm=40.0, lens_nm=45.0)
