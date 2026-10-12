import numpy as np
import pytest
from scipy.spatial import cKDTree

from subpx import detect_centers, measure_cds
from subpx._cpu.cells import _extract_voronoi_rois, compute_voronoi_labels_cpu
from subpx._cpu.deblend import deblend, detect_peaks, estimate_pitch


FWHM = 4.0
BACKGROUND = 20.0
AMPLITUDE = 50.0


def lattice(ratio, *, snr=None, shape=(120, 240), missing=0.0, seed=0):
    """Staggered Gaussian holes with a 15 px hole-free margin."""
    rng = np.random.default_rng(seed)
    pitch = ratio * FWHM
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    xs = np.arange(15, shape[1] - 15, pitch)
    ys = np.arange(15, shape[0] - 15, pitch)
    xx, yy = np.meshgrid(xs, ys)
    xx[1::2] += pitch / 2
    centers = np.column_stack((xx.ravel(), yy.ravel()))
    centers = centers[centers[:, 0] < shape[1] - 15]
    centers = centers[rng.random(len(centers)) >= missing]
    y, x = np.indices(shape, dtype=float)
    image = np.full(shape, BACKGROUND)
    for cx, cy in centers:
        image += AMPLITUDE * np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / sigma**2)
    if snr is not None:
        image += rng.normal(0, AMPLITUDE / snr, shape)
    return image, centers


def interior_cd(result, centers, method):
    measured = result.measurements[method]
    distance, _ = cKDTree(centers).query(result.seeds_xy)
    inner = (result.seeds_xy[:, 0] > 20) & (result.seeds_xy[:, 0] < result.image.shape[1] - 20)
    inner &= (result.seeds_xy[:, 1] > 20) & (result.seeds_xy[:, 1] < result.image.shape[0] - 20)
    use = measured.valid & inner & (distance < 1)
    return measured.metrics["cd_equivalent_px"][use], use.sum(), inner.sum()


@pytest.mark.parametrize("ratio", [2.67, 2.0, 1.6, 1.4])
def test_deblend_recovers_background_and_widths_noise_free(ratio):
    image, centers = lattice(ratio)
    labels = compute_voronoi_labels_cpu(centers, image.shape)
    half = int(np.ceil(0.6 * ratio * FWHM))
    rows = [[x - half, y - half, 2 * half + 1, 2 * half + 1, x, y] for x, y in centers]
    _, masks, origins = _extract_voronoi_rois(image, labels, rows, pad=3)
    state = deblend(image, centers + 0.2, masks, np.asarray(origins))
    assert state.background_info["source"] in {"measured", "interpolated"}
    np.testing.assert_allclose(state.background_map, BACKGROUND, atol=0.01 * AMPLITUDE)
    assert (state.status == "ok").all()
    np.testing.assert_allclose(state.params[:, 0], AMPLITUDE, rtol=0.01)
    np.testing.assert_allclose(state.params[:, 1:3] + origins, centers, atol=0.01)


@pytest.mark.parametrize("ratio, seeds", [(2.67, "threshold"), (2.0, "peaks"), (1.6, "peaks"), (1.4, "peaks")])
def test_gaussian_cd_is_unbiased_through_dense_arrays(ratio, seeds):
    image, centers = lattice(ratio, snr=20)
    result = measure_cds(image, seeds=seeds, methods=("gaussian",), area_max=400)
    values, valid, inner = interior_cd(result, centers, "gaussian")
    assert valid >= 0.99 * inner
    assert abs(values.mean() - FWHM) < 0.05
    assert values.std() < 0.25


def test_model_background_removes_border_bias_for_halfmax():
    image, centers = lattice(2.67, snr=10, seed=3)
    border = measure_cds(image, background="border", methods=("halfmax",), area_max=400)
    model = measure_cds(image, methods=("halfmax",), area_max=400)
    border_bias = interior_cd(border, centers, "halfmax")[0].mean() - FWHM
    model_bias = interior_cd(model, centers, "halfmax")[0].mean() - FWHM
    assert border_bias > 0.25
    assert abs(model_bias) < 0.15


def test_missing_and_spurious_seeds_are_revised():
    # Missing holes leave asymmetric neighbours; noise maxima in the margin give spurious seeds.
    image, centers = lattice(1.6, snr=20, missing=0.05, seed=5)
    result = measure_cds(image, seeds="peaks", methods=("gaussian",), area_max=400)
    values, valid, inner = interior_cd(result, centers, "gaussian")
    assert valid >= 0.98 * inner
    assert abs(values.mean() - FWHM) < 0.05


def test_known_background_level_and_dark_holes():
    image, centers = lattice(2.0)
    dark = 100 - image
    result = measure_cds(dark, invert=True, seeds="peaks", methods=("gaussian",),
                         background_level=100 - BACKGROUND)
    assert result.meta["background_source"] == "fixed"
    np.testing.assert_allclose(result.background_map, 100 - BACKGROUND)
    values, valid, inner = interior_cd(result, centers, "gaussian")
    assert valid == inner
    np.testing.assert_allclose(values, FWHM, atol=1e-3)


def test_peak_detection_and_pitch_estimate():
    image, centers = lattice(1.4, snr=20)
    assert estimate_pitch(image) == pytest.approx(1.4 * FWHM, rel=0.1)
    peaks, _ = detect_peaks(image)
    distance, ids = cKDTree(centers).query(peaks)
    assert len(np.unique(ids[distance < 1])) == len(centers)


def test_border_mode_keeps_previous_defaults_and_rejects_new_options():
    image, _ = lattice(2.67)
    assert list(measure_cds(image, background="border", area_max=400).measurements) == ["logquad", "gradient", "halfmax"]
    assert list(measure_cds(image, area_max=400).measurements) == ["gaussian", "logquad", "gradient", "halfmax"]
    with pytest.raises(ValueError, match="requires background='model'"):
        measure_cds(image, background="border", seeds="peaks")
    with pytest.raises(ValueError, match="requires background='model'"):
        measure_cds(image, background="border", methods=("gaussian",))


def test_detect_centers_default_is_unchanged_and_model_is_opt_in():
    image, centers = lattice(2.0, snr=None)
    image = np.clip(np.rint(image * 50), 0, 65535).astype(np.uint16)
    default = detect_centers(image, backend="cpu", refine="logquad", threshold="otsu", area_max=200)
    again = detect_centers(image, backend="cpu", refine="logquad", threshold="otsu", area_max=200, background=None)
    np.testing.assert_array_equal(default.centers_xy, again.centers_xy)
    assert "background" not in default.meta
    model = detect_centers(image, backend="cpu", refine="logquad", threshold="otsu", area_max=200, background="model")
    assert model.meta["background"] == "model"
    error = cKDTree(centers).query(model.centers_xy)[0]
    assert len(model.centers_xy) == len(default.centers_xy)
    assert np.median(error) < 0.005
    with pytest.raises(NotImplementedError, match="edge methods"):
        detect_centers(image, backend="cpu", refine="edge_gradmoment", background="model")
