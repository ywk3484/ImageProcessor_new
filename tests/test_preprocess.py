"""Preprocessing (subpx.preprocess) against synthetic pages with known truth: square and staggered lattices,
mixed rotated shapes, per-feature brightness flicker, noise."""

import numpy as np
import pytest

from subpx.preprocess import average_hole, denoise_mean_hole, denoise_wiener, median_spacing, smooth_gaussian


def spot(xx, yy, cx, cy, kind, size):
    dx, dy = xx - cx, yy - cy
    if kind == "round":
        return np.exp(-(dx ** 2 + dy ** 2) / (2 * size ** 2))
    angle = np.deg2rad(30 if kind == "ellipse" else 20)
    u, v = dx * np.cos(angle) + dy * np.sin(angle), -dx * np.sin(angle) + dy * np.cos(angle)
    if kind == "ellipse":
        return np.exp(-(u / (1.3 * size)) ** 2 / 2 - (v / (0.8 * size)) ** 2 / 2)
    return np.exp(-((np.abs(u) / (1.3 * size)) ** 4 + (np.abs(v) / (1.3 * size)) ** 4))  # rounded square


def page(h=150, w=190, pitch=8.2, staggered=False, mixed=False, flicker=0.1, noise=5.0, seed=0):
    """Truth, noisy page (8-bit like), the feature centres and brightness factors; random subpixel positions."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w].astype(float)
    truth = np.full((h, w), 12.0)
    centers, gains = [], []
    for row, cy in enumerate(np.arange(5.2, h - 4, pitch)):
        for cx in np.arange(5.1 + (pitch / 2) * (row % 2) * staggered, w - 4, pitch):
            x, y = cx + rng.uniform(-0.5, 0.5), cy + rng.uniform(-0.5, 0.5)
            kind = ("round", "ellipse", "square")[len(centers) % 3] if mixed else "round"
            gains.append(1 + flicker * rng.normal())
            truth += 150 * gains[-1] * spot(xx, yy, x, y, kind, 1.6)
            centers.append((x, y))
    noisy = np.clip(np.rint(truth + rng.normal(0, noise, truth.shape)), 0, 255)
    return truth, noisy, np.array(centers), np.array(gains)


def rmse(a, b, margin=5):
    return float(np.sqrt(np.mean((np.asarray(a, float) - b)[margin:-margin, margin:-margin] ** 2)))


def test_smooth_gaussian_matches_scipy_and_leaves_flat_images_alone():
    ndimage = pytest.importorskip("scipy.ndimage")
    image = np.random.default_rng(1).normal(100, 10, (40, 57))
    result = smooth_gaussian(image, 0.6)
    assert result.image.dtype == np.float32 and result.image.shape == image.shape
    np.testing.assert_allclose(result.image, ndimage.gaussian_filter(image, 0.6, mode="reflect"), atol=1e-4)
    np.testing.assert_allclose(smooth_gaussian(np.full((9, 9), 7), 1.3).image, 7, atol=1e-5)
    np.testing.assert_array_equal(smooth_gaussian(image, 0).image, image.astype(np.float32))
    assert result.parameters["sigma"] == 0.6 and result.meta["kernel_radius"] == 2


def test_wiener_picks_the_noise_level_itself_and_is_linear():
    truth, noisy, _, _ = page(noise=6.0)
    result = denoise_wiener(noisy)
    assert 4.0 < result.meta["noise"] < 9.0 and not result.meta["at_edge"]  # true noise 6 (plus rounding)
    assert rmse(result.image, truth) < 0.5 * rmse(noisy, truth)
    errors = [rmse(denoise_wiener(noisy, sigma).image, truth) for sigma in (2, 4, 6, 8, 12)]
    assert rmse(result.image, truth) < 1.15 * min(errors)  # close to the best level, found without the truth
    # Scaling the image and the noise level together scales the output: one linear filter per image.
    a = denoise_wiener(noisy, 5.0).image
    b = denoise_wiener(2.0 * noisy, 10.0).image
    np.testing.assert_allclose(b, 2.0 * a, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(denoise_wiener(noisy, 0.0).image, noisy, atol=1e-3)  # no noise: nothing removed


@pytest.mark.parametrize("staggered, mixed", [(False, False), (True, False), (True, True)])
def test_mean_hole_removes_noise_on_any_layout_and_shape(staggered, mixed):
    truth, noisy, centers, _ = page(staggered=staggered, mixed=mixed)
    result = denoise_mean_hole(noisy, centers)
    assert np.isfinite(result.image).all()
    assert rmse(result.image, truth) < 0.6 * rmse(noisy, truth)
    assert 2 <= result.meta["components"] < result.meta["limit"]
    assert result.meta["covered_share"] > 0.9
    loss = result.meta["loss"]
    assert np.argmin(loss) == result.meta["components"]  # Noise2Self's pick is the minimum of its curve


def test_mean_hole_passes_brightness_through_and_leaves_open_areas_alone():
    truth, noisy, centers, _ = page(w=230)
    noisy[:, 200:] = np.random.default_rng(3).normal(40, 5, (noisy.shape[0], 30))  # no features here
    centers = centers[centers[:, 0] < 190]
    first = denoise_mean_hole(noisy, centers, components=6)
    brighter = denoise_mean_hole(1.3 * noisy, centers, components=6)
    np.testing.assert_allclose(brighter.image, 1.3 * first.image, rtol=1e-5, atol=1e-3)
    far = slice(None), slice(205, None)  # farther than radius + 1 px from every centre
    np.testing.assert_array_equal(first.image[far], noisy[far].astype(np.float32))
    # More components keep more of each feature's own pixels: the output moves towards the input.
    many = denoise_mean_hole(noisy, centers, components=40)
    assert rmse(many.image, noisy) < rmse(first.image, noisy)


def test_mean_hole_reports_each_features_brightness():
    _, noisy, centers, gains = page(noise=1.0, flicker=0.15)
    result = denoise_mean_hole(noisy, centers, components=4)
    brightness = result.meta["brightness"]
    assert np.isfinite(brightness).all() and abs(np.median(brightness) - 1) < 1e-9
    assert np.corrcoef(brightness, gains)[0, 1] > 0.98


def test_average_hole_recovers_the_true_shape_on_a_finer_grid():
    rng = np.random.default_rng(2)
    h, w, size = 200, 260, 1.4
    yy, xx = np.mgrid[:h, :w].astype(float)
    image = np.full((h, w), 10.0)
    centers, gains = [], []
    for cy in np.arange(6, h - 5, 9.3):
        for cx in np.arange(6, w - 5, 9.3):
            x, y = cx + rng.uniform(-.5, .5), cy + rng.uniform(-.5, .5)
            gains.append(1 + 0.2 * rng.normal())
            image += 100 * gains[-1] * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * size ** 2))
            centers.append((x, y))
    centers = np.array(centers)
    result = average_hole(image, centers, upsample=4, radius=3.0)
    assert result.image.shape == (25, 25) and result.offsets[0] == -3.0 and result.offsets[12] == 0
    assert result.meta["empty_bins"] == 0 and 0.15 < result.meta["brightness_spread"] < 0.25
    # Each bin is 1/4 px wide: the truth is the profile averaged over that width, at the mean brightness.
    width = np.linspace(-0.125, 0.125, 11)
    profile = np.mean([np.exp(-(result.offsets + d) ** 2 / (2 * size ** 2)) for d in width], axis=0)
    truth = 10 + 100 * np.mean(gains) * profile[:, None] * profile[None, :]
    error = result.image - truth  # grey levels, peak ~100; ~40 features per bin, each at its own sub-bin offset
    assert np.sqrt(np.mean(error ** 2)) < 0.6 and np.abs(error).max() < 2.0
    plain = average_hole(image, centers, upsample=4, radius=3.0, normalize=False)
    assert plain.meta["holes"] == len(centers) and np.isnan(plain.meta["brightness_spread"])
    assert np.abs(plain.image - truth).max() > np.abs(error).max()  # flicker left in blurs the plain average


def test_median_spacing_and_input_checks():
    _, noisy, centers, _ = page()
    assert abs(median_spacing(centers) - 8.2) < 0.6
    with pytest.raises(ValueError):
        smooth_gaussian(np.zeros((3, 3, 3)))
    with pytest.raises(ValueError):
        denoise_wiener(np.full((5, 5), np.nan))
    with pytest.raises(ValueError):
        denoise_mean_hole(noisy, centers[:2])
    with pytest.raises(ValueError):
        average_hole(noisy, np.empty((0, 2)))
