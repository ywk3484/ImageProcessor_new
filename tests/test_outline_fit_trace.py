"""The record of outline_fit_fast's steps (``fit_outline_fast(trace=...)``) and the image redrawn from its answer.

The record is for looking at the calculation (scripts/outline_fit_steps.py); it must not change it. The page fit is
tested under NumPy, where it is the same from run to run; the whole fit on the GPU, where two runs of one image
differ in the sixth digit of the CD (the device adds in another order every time), so "the same" there has that
width. Numbers in brackets are measured values.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("cv2")

from subpx import fit_outline_fast, measure_cds
from subpx._outline.model import Template
from subpx._outline.onepass import ShapeFitter
from subpx.synthetic_arrays import render_hole_array

PIXEL_NM = 21.0
STEPS = ("window", "average_all", "average_1", "fit_1", "holes", "kept_2", "average_2", "average_2_parts", "fit_2",
         "exchange", "redrawn", "image")


# ------------------------------------------------------------------------------------------- no GPU needed
def known_stack(blur="gauss2"):
    """A stack made from a known outline and blur, without neighbours, and its fitter (NumPy)."""
    t = Template(upsample=8, size_px=24, blur=blur)
    sf = ShapeFitter(t, box_px=48, f_max=0.6, U=4, R=7, n_poly=256, xp=np)
    truth = t.pack(2.4, 2.3, 2.2, 0.2, np.zeros((2, 2)), t.blur_pack(s1x=1.27, s1y=1.27, s2x=2.5, s2y=2.5, w2=0.05))
    gamma = np.ones((1, sf.U, sf.U, sf.nky, sf.nkx), dtype=complex)
    mask = sf.pad[:, None, :, None] & sf.pad[None, :, None, :]
    clean = 180.0 * sf.synth(sf.template_hat(truth[None]), gamma)[0, 0] + 40.0
    y = np.where(mask, clean + np.random.default_rng(0).normal(0, 0.16, clean.shape), 0.0)
    return t, sf, gamma, y, np.where(mask, 1.0 / 0.16 ** 2, 0.0)


def test_recording_a_page_fit_leaves_it_as_it_is():
    t, sf, gamma, y, w = known_stack()
    start = t.start(1.7, 1.0)[None]
    plain = sf.fit(start, gamma, y[None], w[None])
    trace = {}
    recorded = sf.fit(start, gamma, y[None], w[None], trace=trace)
    for a, b in zip(plain[:4], recorded[:4]):
        np.testing.assert_array_equal(a, b)
    assert [(s, n) for s, n, _ in plain[4]] == [(s, n) for s, n, _ in recorded[4]]

    scan, = trace["grid"]  # one page
    assert scan["sizes_px"].shape == (15,) and scan["widths_px"].shape == (12,) and scan["cost"].shape == (15, 12)
    assert np.all(np.diff(scan["sizes_px"]) > 0) and np.all(np.diff(scan["widths_px"]) > 0)
    assert scan["pick"] == np.unravel_index(scan["cost"].argmin(), (15, 12)) and scan["weight"] > 0

    stages = [name for name, _, _ in recorded[4]]
    assert stages == ["round", "super", "full"]
    for name, iterations, stopped in recorded[4]:
        rows = [r for r in trace["iterations"] if r["stage"] == name]
        assert [r["iteration"] for r in rows] == list(range(iterations + 1))  # the start, then every iteration
        cost = np.array([r["cost"][0] for r in rows])
        assert np.all(np.diff(cost) <= 0)  # the fit never moves to a worse point
        for before, row in zip(rows, rows[1:]):
            assert row["theta"].shape == (1, t.n_params) and row["tried"].shape == (1,)
            if not row["accepted"][0]:  # a rejected point: the fit stays where it was
                np.testing.assert_array_equal(row["theta"], before["theta"])
                assert row["cost"][0] == before["cost"][0]
            else:
                assert row["cost"][0] == row["tried"][0] < before["cost"][0]
        assert rows[-1]["done"][0] == stopped[0]
    np.testing.assert_array_equal(trace["iterations"][-1]["theta"], recorded[0])
    assert trace["iterations"][-1]["cost"][0] == recorded[3][0]


def test_from_classes_is_the_way_back_from_to_classes():
    _, sf, _, _, _ = known_stack()
    side = 2 * int(np.ceil((sf.R + 0.5) * sf.U)) + 1  # the stack's grid, as GpuPage makes it
    grid = np.random.default_rng(1).normal(size=(side, side))
    by_class, mask = sf.to_classes(grid)
    back = sf.from_classes(by_class)
    low = side // 2 - sf.nmax
    assert back.shape == (2 * sf.nmax + 1,) * 2 and np.isfinite(back).all()  # every bin belongs to one class
    np.testing.assert_array_equal(back, grid[low:low + back.shape[0], low:low + back.shape[1]])
    np.testing.assert_array_equal(sf.from_classes(mask.astype(float)), np.ones_like(back))


# ------------------------------------------------------------------------------------------------- GPU
@pytest.fixture(scope="module")
def gpu():
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("no CUDA device")
    except Exception:
        pytest.skip("CUDA unavailable")


def report_seeds(image):
    """The seeds the folder reports hand to the fit: every hole found, with those cut by the frame (a hole that is
    not in the list is missing from the method's picture of the image)."""
    from scripts.outline_fast_check import tapped_seeds

    box = {}
    with tapped_seeds(box):
        measure_cds(image, methods=("outline_fit_fast",), pixel_size=PIXEL_NM, unit="nm", backend="gpu", area_max=200)
    return box["seeds"]


@pytest.fixture(scope="module")
def fitted(gpu):
    """A generated image that is what the method assumes (one hole size, a Gaussian blur, the same noise of 1.2
    grey levels at every pixel, 8 bit; pitch 10.3 px; 4,642 holes found, 4,389 of them a window away from the
    frame), fitted plainly and with the record and the redrawn picture."""
    page = render_hole_array((600, 800), hole="circle", size_nm=(100, 100), pitch_nm=(216.3, 216.3), layout="square",
                             blur_nm=65, lens_nm=0, photons_per_level=None, read_noise=1.2, cd_sigma_nm=0,
                             row_shift_nm=0, seed=11, backend="cpu")
    seeds = report_seeds(page.image)
    plain = fit_outline_fast(page.image, seeds, pixel_size=PIXEL_NM, unit="nm")
    trace = {}
    traced = fit_outline_fast(page.image, seeds, pixel_size=PIXEL_NM, unit="nm", trace=trace, redrawn=True)
    return page, seeds, plain, traced, trace


def test_recording_changes_nothing(fitted):
    _, _, plain, traced, _ = fitted
    # two plain runs of this image: CD 2e-5 px apart, centres 1e-5 px, brightness 2e-3 grey levels
    assert traced.outline_cd_px == pytest.approx(plain.outline_cd_px, abs=1e-3)
    np.testing.assert_allclose(traced.centers_xy, plain.centers_xy, atol=1e-3)
    np.testing.assert_allclose(traced.brightness, plain.brightness, rtol=1e-3)
    np.testing.assert_allclose(traced.level, plain.level, atol=0.05)
    assert int(np.sum(traced.good != plain.good)) <= 1
    np.testing.assert_allclose(traced.stack, plain.stack, atol=1e-3, equal_nan=True)


def test_every_step_is_recorded(fitted):
    page, seeds, _, traced, trace = fitted
    image = page.image
    assert tuple(trace) == STEPS  # in the order of the calculation
    n = len(seeds)
    window = trace["window"]
    np.testing.assert_array_equal(window["seeds"], seeds)
    assert window["nearest_px"].shape == (n,) and window["spacing_px"] == pytest.approx(10.3, abs=0.2)
    assert window["window_px"] == 8 and window["bins_per_px"] == 4 and window["stack_bins"] == 69

    first, used = trace["average_all"], trace["average_1"]
    assert first["stack"].shape == first["pixels"].shape == used["stack"].shape == (69, 69)
    for key in ("inside", "brightness", "level", "leftover", "compared", "kept"):
        assert first[key].shape == (n,)
    assert not (first["kept"] & ~first["inside"]).any() and first["kept"].sum() <= first["inside"].sum()
    low, high = first["brightness_limits"]
    usual = (first["brightness"] > low) & (first["brightness"] < high)
    quiet = first["leftover"] < first["leftover_limit"] * first["typical_leftover"]
    np.testing.assert_array_equal(first["kept"], first["inside"] & usual & quiet & first["compared"])
    np.testing.assert_array_equal(used["kept"], first["kept"])
    assert used["noise"] == pytest.approx(np.median(used["leftover"][used["kept"]]))

    for name, stages in (("fit_1", ["round", "super"]), ("fit_2", ["full"])):
        fit = trace[name]
        assert [s for s, _, _ in fit["stages"]] == stages
        assert (fit["grid"] is not None) == (name == "fit_1")
        side = 2 * int(np.floor((window["window_px"] - 0.5) * 4)) + 1
        for key in ("average", "pixels", "fitted", "match", "alone"):
            assert fit[key].shape == (side, side)
        assert fit["fitted"].sum() == fit["bins"] and np.isfinite(fit["match"]).all()
        assert fit["neighbour_map"].ndim == 2 and fit["holes_per_class"].shape == (4, 4)
        assert fit["holes_per_class"].sum() == pytest.approx(trace["average_1" if name == "fit_1" else "average_2"]["kept"].sum())
        assert fit["iterations"][0]["iteration"] == 0 and fit["iterations"][-1]["theta"].shape == fit["theta"].shape
        np.testing.assert_array_equal(fit["iterations"][-1]["theta"], fit["theta"])
        # the hole alone is the smaller part of the best match's light: the rest comes from the neighbours
        light = fit["match"] - fit["const"]
        assert 0.5 < fit["alone"].sum() / light.sum() < 1.0

    holes = trace["holes"]
    for key in ("brightness", "level", "leftover"):
        assert holes[key].shape == (n,)
    assert holes["centers_xy"].shape == (n, 2) and len(holes["passes"]) == 2
    assert holes["picture"].shape == image.shape and holes["picture"].dtype == np.float32
    assert np.abs(holes["centers_xy"] - seeds).max() <= 0.5 + 1e-9  # two passes of at most a quarter pixel
    assert float(np.sqrt(np.mean((holes["picture"] - image) ** 2))) < 2.0  # [1.41 grey levels; noise 1.2]

    kept = trace["kept_2"]
    np.testing.assert_array_equal(trace["average_2"]["kept"],
                                  kept["inside"] & kept["usual_brightness"] & kept["usual_leftover"])
    np.testing.assert_array_equal(trace["average_2"]["kept"], traced.good)
    last = trace["exchange"] if trace["exchange"] and trace["exchange"]["kept"] else trace["fit_2"]
    np.testing.assert_array_equal(last["theta"], traced.theta)
    # the nine parts of the image, stacked each by itself, add up to the whole image's stack
    parts, whole = trace["average_2_parts"], trace["average_2"]
    assert parts["stacks"].shape == (9, 69, 69) and parts["holes"].sum() == whole["kept"].sum() and parts["holes"].min() > 300
    np.testing.assert_allclose(parts["pixels"].sum(axis=0), whole["pixels"])
    total = np.nansum(parts["stacks"] * parts["pixels"], axis=0)
    filled = whole["pixels"] > 0
    np.testing.assert_allclose(total[filled] / whole["pixels"][filled], whole["stack"][filled], rtol=1e-9)
    assert trace["image"] == {"inverted": False, "top": float(image.max()), "own_seeds": False}


def test_the_pictures_add_up_to_the_fits_own_cost(fitted):
    _, _, _, traced, trace = fitted
    for name in ("fit_1", "fit_2"):
        fit = trace[name]
        weight = np.where(fit["fitted"], fit["pixels"], 0.0) / fit["noise"] ** 2
        distance = np.where(fit["fitted"], fit["match"] - fit["average"], 0.0)
        assert float((weight * distance ** 2).sum()) == pytest.approx(fit["data_cost"], rel=1e-9)
        assert fit["data_cost"] <= fit["cost"] <= fit["data_cost"] + 50  # the weak priors add a little [1.8]
    last = trace["exchange"] if trace["exchange"] and trace["exchange"]["kept"] else trace["fit_2"]
    assert traced.summary["leftover_chi2"] == pytest.approx(last["cost"] / last["bins"], rel=1e-12)
    assert traced.summary["fit_iterations"] == last["stages"][-1][1]


def test_the_redrawn_picture_is_the_image_without_its_noise(fitted):
    page, seeds, _, _, trace = fitted
    copy = trace["redrawn"]
    assert copy["picture"].shape == page.image.shape and copy["picture"].dtype == np.float32
    assert copy["centers_xy"].shape == (len(seeds), 2) and copy["brightness"].shape == (len(seeds),)
    # away from the frame (holes beyond it are in the image and not in the picture) the picture is the generator's
    # noise-free image to a fifth of the noise: what is left of 1.2 grey levels after fitting a hole's four numbers
    inner = (slice(20, -20), slice(20, -20))
    assert float(np.sqrt(np.mean((copy["picture"] - page.clean)[inner] ** 2))) < 0.35  # [0.22]
    assert float(np.sqrt(np.mean((trace["holes"]["picture"] - page.clean)[inner] ** 2))) < 0.35  # [0.22], step B's
    # the pixel noise found is the noise put in (1.2) with the rounding to whole grey levels (1 / 12 of variance)
    assert copy["pixel_noise"] == pytest.approx(np.sqrt(1.2 ** 2 + 1 / 12), rel=0.03)  # [1.235]
    assert copy["leftover"] == pytest.approx(copy["pixel_noise"], rel=0.1)  # nothing but noise is left of a hole


def test_an_image_redrawn_from_the_answer_reads_back_the_cd_it_was_drawn_with(fitted):
    page, _, _, traced, trace = fitted
    copy = trace["redrawn"]
    grain = np.sqrt(copy["pixel_noise"] ** 2 - 1 / 12)  # the rounding below adds its twelfth again
    drawn = copy["picture"] + grain * np.random.default_rng(3).standard_normal(copy["picture"].shape)
    stored = np.clip(np.rint(drawn), 0, 255).astype(np.uint8)
    again = fit_outline_fast(stored, report_seeds(stored), pixel_size=PIXEL_NM, unit="nm")
    # four noise seeds: +0.00 to +0.06 nm here; -0.10 to 0.00 on an image with a size spread and a lens blur
    assert abs(again.outline_cd_px - traced.outline_cd_px) * PIXEL_NM < 0.2
    assert 0.7 < again.summary["leftover_chi2"] < 1.4  # [0.95 to 1.04: at the noise, as the method expects]


def test_dark_holes_are_recorded_turned_over(gpu, fitted):
    page, seeds, _, traced, _ = fitted
    image = page.image
    trace = {}
    dark = fit_outline_fast(255 - image, seeds, pixel_size=PIXEL_NM, unit="nm", invert=True, trace=trace, redrawn=True)
    assert trace["image"] == {"inverted": True, "top": float((255 - image).max()), "own_seeds": False}
    assert dark.outline_cd_px == pytest.approx(traced.outline_cd_px, abs=1e-3)
    turned = trace["image"]["top"] - (255.0 - image)  # the fit's own frame: bright holes
    assert float(np.sqrt(np.mean((trace["redrawn"]["picture"] - turned) ** 2))) < 2.0
    assert float(np.sqrt(np.mean((trace["holes"]["picture"] - turned) ** 2))) < 2.0


def test_a_fit_that_fails_leaves_the_steps_it_reached(gpu):
    image = np.random.default_rng(0).integers(10, 60, size=(24, 40)).astype(np.uint8)
    seeds = np.column_stack([np.arange(2.0, 22.0, 2.0), np.full(10, 2.0)])  # ten holes in a row along the frame
    trace = {}
    with pytest.raises(ValueError, match="inside the image"):
        fit_outline_fast(image, seeds, trace=trace)
    assert tuple(trace) == ("window", "image") and trace["window"]["window_px"] == 3
