from dataclasses import replace

import numpy as np
import pytest

from subpx import analyze_pitch, measure_cds
from subpx.types import CDMethodResult, CDSummary


def lattice(rows=6, columns=8, layout="square", stagger_axis="row", pitch=(10.0, 12.0)):
    rr, cc = np.mgrid[:rows, :columns]
    points = np.column_stack((cc.ravel() * pitch[0] + 15, rr.ravel() * pitch[1] + 15)).astype(float)
    if layout == "staggered":
        points[:, 0 if stagger_axis == "row" else 1] += (
            rr.ravel() if stagger_axis == "row" else cc.ravel()) % 2 * pitch[0 if stagger_axis == "row" else 1] / 2
    return points, np.column_stack((rr.ravel(), cc.ravel()))


def summary(seeds, centers=None, valid=None, scale=1.0):
    n = len(seeds)
    centers = seeds.copy() if centers is None else centers
    method = CDMethodResult("test", centers, np.empty((n, 0, 2)), np.ones(n, bool) if valid is None else valid,
                            np.full(n, "ok"), {})
    return CDSummary(seeds, {"test": method}, pixel_size=scale, unit="nm" if scale != 1 else "px")


@pytest.mark.parametrize("layout", ["square", "staggered"])
@pytest.mark.parametrize("stagger_axis", ["row", "column"])
def test_adjacent_pitch_on_square_and_staggered_grids(layout, stagger_axis):
    seeds, rc = lattice(layout=layout, stagger_axis=stagger_axis)
    rng = np.random.default_rng(12)
    order = rng.permutation(len(seeds))
    measured = analyze_pitch(summary(seeds[order]), stagger_axis=stagger_axis)["test"]
    np.testing.assert_array_equal(measured.indices_rc, rc[order])
    assert measured.layout == layout
    np.testing.assert_allclose(measured.profiles["x"].pitch, 10, atol=1e-12)
    np.testing.assert_allclose(measured.profiles["y"].pitch, 12, atol=1e-12)
    np.testing.assert_allclose(measured.profiles["x"].registration, 0, atol=1e-12)
    np.testing.assert_allclose(measured.profiles["y"].distance, np.arange(6) * 12, atol=1e-12)


def test_directional_means_selection_reference_calibration_and_discrete_integrals():
    seeds, rc = lattice(layout="staggered")
    r, c = rc.T
    centers = seeds + np.column_stack((0.03 * c**2 + 0.01 * r * c, 0.04 * r**2 + 0.02 * c * r))
    source = summary(seeds, centers, scale=2.5)
    result = analyze_pitch(source, average_rows=3, average_columns=4, reference_xy=seeds[3 * 8 + 4])["test"]
    assert result.meta["row_band"] == [2, 3, 4]
    assert result.meta["column_band"] == [3, 4, 5, 6]
    x, y = result.profiles["x"], result.profiles["y"]
    expected_x = (10 + 0.03 * (2 * np.arange(7) + 1) + 0.01 * 3) * 2.5
    expected_y = (12 + 0.04 * (2 * np.arange(5) + 1) + 0.02 * 4.5) * 2.5
    np.testing.assert_allclose(x.pitch, expected_x, atol=1e-12)
    np.testing.assert_allclose(y.pitch, expected_y, atol=1e-12)
    np.testing.assert_array_equal(x.counts, 3)
    np.testing.assert_array_equal(y.counts, 4)
    for p, expected in ((x, expected_x), (y, expected_y)):
        accumulated = np.r_[0, np.cumsum(expected)]
        np.testing.assert_allclose(p.distance, accumulated - accumulated[p.reference_node], atol=1e-12)
        accumulated = np.r_[0, np.cumsum(expected - expected.mean())]
        np.testing.assert_allclose(p.registration, accumulated - accumulated[p.reference_node], atol=1e-12)
    explicit = analyze_pitch(source, nominal_pitch_xy=(25, 30))["test"]
    assert explicit.profiles["x"].nominal_pitch == 25
    assert explicit.meta["nominal_source"] == "user"
    np.testing.assert_allclose(explicit.profiles["y"].registration, np.cumsum(np.r_[0, explicit.profiles["y"].pitch - 30]))


def test_methods_use_their_own_centers_and_share_indices_without_seed_fallback():
    seeds, rc = lattice()
    source = summary(seeds)
    changed = seeds + np.column_stack((rc[:, 1] * 0.1, rc[:, 0] * 0.2))
    valid = np.ones(len(seeds), bool)
    valid[0] = False
    changed[9] = np.nan
    source.measurements["other"] = replace(source.measurements["test"], method="other", centers_xy=changed, valid=valid)
    results = analyze_pitch(source, average_rows=1, average_columns=1)
    original, other = results["test"], results["other"]
    np.testing.assert_array_equal(original.indices_rc, other.indices_rc)
    assert not other.meta["reference_valid"]
    assert not other.valid[9]
    assert np.isnan(other.profiles["x"].pitch[0])
    assert np.isnan(other.profiles["x"].distance[1:]).all()
    np.testing.assert_allclose(other.profiles["x"].pitch[1:], 10.1, atol=1e-12)
    assert np.all(other.profiles["x"].pairs != 0)
    assert np.all(other.profiles["x"].pairs != 9)


def test_missing_holes_and_entire_lines_never_create_doubled_pitch():
    seeds, rc = lattice(layout="staggered")
    keep = (rc[:, 1] != 3) & (rc[:, 0] != 2) & ~((rc[:, 0] == 4) & (rc[:, 1] == 5))
    result = analyze_pitch(summary(seeds[keep]), layout="staggered", pitch_hint_xy=(10, 12))["test"]
    np.testing.assert_array_equal(result.indices_rc, rc[keep])
    x, y = result.profiles["x"], result.profiles["y"]
    assert np.isnan(x.pitch[2:4]).all()
    assert np.isnan(y.pitch[1:3]).all()
    np.testing.assert_allclose(x.pitch[np.isfinite(x.pitch)], 10)
    np.testing.assert_allclose(y.pitch[np.isfinite(y.pitch)], 12)
    assert np.isnan(x.registration[3:]).all()


def test_noisy_seed_grid_and_edge_clamped_band():
    seeds, rc = lattice(rows=20, columns=30, layout="staggered", pitch=(8, 9))
    seeds += np.random.default_rng(3).normal(0, 0.04, seeds.shape)
    result = analyze_pitch(summary(seeds), average_rows=5, average_columns=7, reference_xy=seeds[-1])["test"]
    np.testing.assert_array_equal(result.indices_rc, rc)
    assert result.meta["row_band"] == list(range(15, 20))
    assert result.meta["column_band"] == list(range(23, 30))
    np.testing.assert_allclose(result.profiles["x"].pitch, 8, atol=0.1)
    np.testing.assert_allclose(result.profiles["y"].pitch, 9, atol=0.1)


@pytest.mark.parametrize("options,match", [
    ({"average_rows": 0}, "average_rows"), ({"average_columns": 9}, "average_columns"),
    ({"average_rows": 2.5}, "average_rows"), ({"nominal_pitch_xy": (1, 0)}, "nominal_pitch_xy"),
    ({"reference_xy": (np.nan, 2)}, "reference_xy"), ({"layout": "hex"}, "layout"),
    ({"stagger_axis": "diagonal"}, "stagger_axis"), ({"line_tolerance": -1}, "line_tolerance"),
    ({"pitch_hint_xy": (1, -1)}, "pitch_hint_xy"),
])
def test_invalid_options_are_rejected(options, match):
    with pytest.raises(ValueError, match=match):
        analyze_pitch(summary(lattice()[0]), **options)


def gaussian_image(layout="square"):
    centers, rc = lattice(rows=5, columns=7, layout=layout, pitch=(11, 12))
    yy, xx = np.mgrid[:80, :100]
    image = np.full(xx.shape, 8.0)
    for x, y in centers:
        image += 130 * np.exp(-((xx - x - 0.23)**2 + (yy - y - 0.17)**2) / (2 * 1.35**2))
    return image, centers + [0.23, 0.17]


@pytest.mark.parametrize("layout", ["square", "staggered"])
def test_cd_centers_on_synthetic_images_have_accurate_pitch(layout):
    image, centers = gaussian_image(layout)
    source = measure_cds(image, methods=("gaussian", "projection", "half_area", "local_projection", "pixel_half_area"),
                         area_max=100, outlines=False, metrics_only=True)
    assert len(source.seeds_xy) == len(centers)
    for m in analyze_pitch(source).values():
        assert m.layout == layout
        np.testing.assert_allclose(m.profiles["x"].pitch, 11, atol=0.08)
        np.testing.assert_allclose(m.profiles["y"].pitch, 12, atol=0.08)


def test_gpu_cd_center_pitch_parity():
    from subpx.backends import is_gpu_available
    if not is_gpu_available():
        pytest.skip("CuPy/CUDA unavailable")
    image, _ = gaussian_image("staggered")
    options = dict(methods=("gaussian", "local_projection", "local_half_area", "pixel_half_area"), area_max=100,
                   outlines=False, metrics_only=True)
    cpu = analyze_pitch(measure_cds(image, backend="cpu", **options))
    gpu = analyze_pitch(measure_cds(image, backend="gpu", **options))
    for name in cpu:
        np.testing.assert_array_equal(gpu[name].indices_rc, cpu[name].indices_rc)
        for axis in ("x", "y"):
            np.testing.assert_allclose(gpu[name].profiles[axis].pitch, cpu[name].profiles[axis].pitch, atol=2e-5)
