"""scripts/deblend_variants.py: with every switch off the variant loop is the library's, bit for bit; each switch does
what it says (skipped without CuPy or a CUDA device)."""

import math

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    _DEVICES = cp.cuda.runtime.getDeviceCount()
except Exception:
    _DEVICES = 0
pytestmark = pytest.mark.skipif(_DEVICES == 0, reason="CUDA device unavailable")

from subpx import measure_cds  # noqa: E402
from subpx._cpu.cells import _extract_voronoi_rois, compute_voronoi_labels_cpu  # noqa: E402
from subpx._cpu.deblend import auto_tile, precision_matrices  # noqa: E402
from subpx._gpu import deblend as gpu  # noqa: E402
from scripts.deblend_variants import Patch, Variant, deblend  # noqa: E402

FWHM = 4.0
KEYS = {"round", "seconds", "ok", "step_p99", "centre_p99", "width_p99", "edge_p99", "bg_mean", "bg_max", "bg_rms",
        "free_px", "measured_tiles", "source", "cos", "update"}


def lattice(ratio, *, snr=20, shape=(120, 240), small=0, seed=0):
    """Staggered Gaussian holes (amplitude 50 on 20) with a 15 px hole-free margin. ``small`` interior holes are
    narrow (sigma 1.1 px): thresholding finds them below area_min=10, and the seed re-check adds them back."""
    rng = np.random.default_rng(seed)
    pitch = ratio * FWHM
    xx, yy = np.meshgrid(np.arange(15, shape[1] - 15, pitch), np.arange(15, shape[0] - 15, pitch))
    xx[1::2] += pitch / 2
    centers = np.column_stack((xx.ravel(), yy.ravel()))
    centers = centers[centers[:, 0] < shape[1] - 15]
    sigma = np.full(len(centers), FWHM / (2 * np.sqrt(2 * np.log(2))))
    if small:
        inner = np.flatnonzero((np.abs(centers[:, 0] - shape[1] / 2) < shape[1] / 2 - 40)
                               & (np.abs(centers[:, 1] - shape[0] / 2) < shape[0] / 2 - 30))
        sigma[rng.choice(inner, small, replace=False)] = 1.1
    y, x = np.indices(shape, dtype=float)
    image = np.full(shape, 20.0)
    for (cx, cy), s in zip(centers, sigma):
        image += 50 * np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / s**2)
    return image + rng.normal(0, 50 / snr, shape), centers


def cells(image, centers, ratio):
    """Seeds 0.2 px off the true centres and their Voronoi cell stack."""
    seeds = centers + 0.2
    labels = compute_voronoi_labels_cpu(seeds, image.shape)
    half = int(np.ceil(0.6 * ratio * FWHM))
    rows = [[x - half, y - half, 2 * half + 1, 2 * half + 1, x, y] for x, y in seeds]
    _, masks, origins = _extract_voronoi_rois(image, labels, rows, pad=3)
    return seeds, masks, np.asarray(origins)


def assert_same_state(got, expected):
    for name in ("background_map", "model_image", "params", "cleaned"):
        assert np.array_equal(getattr(got, name), getattr(expected, name), equal_nan=True), name
    assert np.array_equal(got.status, expected.status)
    assert (got.tile_px, got.iterations, got.converged) == (expected.tile_px, expected.iterations, expected.converged)
    assert got.background_info.keys() == expected.background_info.keys()
    for key, value in expected.background_info.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(got.background_info[key], value, equal_nan=True), key
        else:
            assert got.background_info[key] == value, key


def assert_same_fixed_point(got, expected):
    """Centres and log-widths within 2e-3 (the stop rule moves them less than 1e-3 per round); amplitudes, in image
    units, within 2e-3 of their value."""
    ok = (got.status == "ok") & (expected.status == "ok")
    assert ok.mean() > 0.99 and np.isfinite(got.params).all()
    np.testing.assert_allclose(got.params[ok, 1:], expected.params[ok, 1:], rtol=0, atol=2e-3)
    np.testing.assert_allclose(got.params[ok, 0], expected.params[ok, 0], rtol=2e-3)


@pytest.mark.parametrize("traced", [False, True])
def test_switches_off_is_the_library_bit_for_bit(traced):
    image, centers = lattice(1.6)
    seeds, masks, origins = cells(image, centers, 1.6)
    expected = gpu.deblend(image, seeds, masks, origins)
    assert expected.iterations >= 6
    assert_same_state(deblend(image, seeds, masks, origins, trace=[] if traced else None), expected)
    start = gpu.deblend(image, seeds, masks, origins, iterations=3).params
    assert_same_state(deblend(image, seeds, masks, origins, init=start, trace=[] if traced else None),
                      gpu.deblend(image, seeds, masks, origins, init=start))


@pytest.mark.parametrize("seeds, small, ratio", [("peaks", 0, 1.6), ("threshold", 8, 1.8)])
def test_patch_with_switches_off_changes_no_number(seeds, small, ratio):
    # peaks: pass 1, then the continuation with init. threshold: pass 1, the re-check adds the small holes, a refit.
    image, _ = lattice(ratio, small=small, seed=2)
    kwargs = dict(seeds=seeds, area_min=10, area_max=400, backend="gpu")
    expected = measure_cds(image, **kwargs)
    original = gpu.deblend
    with Patch() as patch:
        got = measure_cds(image, **kwargs)
    assert gpu.deblend is original
    assert len(patch.passes) == 2 and patch.passes[1]["continued"] == (seeds == "peaks")
    assert np.array_equal(got.seeds_xy, expected.seeds_xy)
    for name, measured in expected.measurements.items():
        other = got.measurements[name]
        assert np.array_equal(other.status, measured.status), name
        assert other.metrics.keys() == measured.metrics.keys()
        for key, values in measured.metrics.items():
            assert np.array_equal(other.metrics[key], values, equal_nan=True), (name, key)
    assert np.array_equal(got.deblend_model["params"], expected.deblend_model["params"], equal_nan=True)
    assert np.array_equal(got.background_map, expected.background_map)
    assert got.meta["deblend_iterations_used"] == expected.meta["deblend_iterations_used"]


@pytest.mark.parametrize("limit", [1e-4, 1e-2])
def test_tolerance_is_the_library_stop_rule(limit):
    image, centers = lattice(1.6)
    seeds, masks, origins = cells(image, centers, 1.6)
    expected = gpu.deblend(image, seeds, masks, origins, tolerance=limit)
    assert expected.iterations != gpu.deblend(image, seeds, masks, origins).iterations
    assert_same_state(deblend(image, seeds, masks, origins, variant=Variant(tolerance=limit)), expected)
    # Patch applies it to every call: pass 1 of measure_cds as well as the last pass.
    with Patch(Variant(tolerance=limit)) as patch:
        measure_cds(image, seeds="peaks", backend="gpu")
    assert all(p["variant"]["tolerance"] == limit for p in patch.passes)
    with Patch() as plain:
        measure_cds(image, seeds="peaks", backend="gpu")
    tighter = limit < 1e-3
    assert (sum(p["used"] for p in patch.passes) > sum(p["used"] for p in plain.passes)) == tighter


@pytest.mark.parametrize("steps", [100, 3])
def test_lm_steps_is_the_library_max_iter(steps):
    image, centers = lattice(1.4, snr=10)  # a page where the LM step count changes the result
    seeds, masks, origins = cells(image, centers, 1.4)
    expected = gpu.deblend(image, seeds, masks, origins, max_iter=steps)
    assert not np.array_equal(expected.params, gpu.deblend(image, seeds, masks, origins).params)
    assert_same_state(deblend(image, seeds, masks, origins, variant=Variant(lm_steps=steps)), expected)


def test_freeze_background_keeps_the_first_rounds_map():
    image, centers = lattice(1.6)
    seeds, masks, origins = cells(image, centers, 1.6)
    rows = []
    state = deblend(image, seeds, masks, origins, variant=Variant(freeze_background=True), trace=rows)
    assert state.converged and len(rows) == state.iterations > 2
    assert all(row["bg_max"] == 0 and row["bg_rms"] == 0 for row in rows[1:])
    first = deblend(image, seeds, masks, origins, iterations=1, variant=Variant(freeze_background=True))
    assert np.array_equal(state.background_map, first.background_map)  # round 1's map, to the end
    assert not np.array_equal(state.background_map, gpu.deblend(image, seeds, masks, origins).background_map)


def test_freeze_free_keeps_the_free_pixels_and_still_updates_the_map():
    image, centers = lattice(1.6)
    seeds, masks, origins = cells(image, centers, 1.6)
    plain, frozen = [], []
    deblend(image, seeds, masks, origins, trace=plain)
    state = deblend(image, seeds, masks, origins, variant=Variant(freeze_free=True), trace=frozen)
    assert state.converged
    assert len({row["free_px"] for row in plain}) > 1 and len({row["free_px"] for row in frozen}) == 1
    assert any(row["bg_max"] > 0 for row in frozen[1:])


def test_tile_free_px_measures_tiles_with_fewer_free_pixels():
    image, centers = lattice(1.6)
    image, centers = image[8:-8, 8:-8], centers - 8  # a 1-3 px hole-free band along the edges: few free pixels
    seeds, masks, origins = cells(image, centers, 1.6)
    tile = auto_tile(seeds, image.shape)[0]
    expected = gpu.deblend(image, seeds, masks, origins)
    rule = max(16, int(0.1 * tile * tile))  # the library's: background_tiles measures a tile from this many
    assert_same_state(deblend(image, seeds, masks, origins, variant=Variant(tile_free_px=rule)), expected)
    looser = deblend(image, seeds, masks, origins, variant=Variant(tile_free_px=16))
    assert looser.converged
    assert looser.background_info["measured_tiles"] > expected.background_info["measured_tiles"]
    assert looser.background_info["source"] in ("measured", "interpolated")


def test_anderson_settles_sooner_at_the_same_fixed_point():
    image, centers = lattice(1.8, snr=10)
    seeds, masks, origins = cells(image, centers, 1.8)
    plain = deblend(image, seeds, masks, origins)
    assert plain.converged and plain.iterations >= 6
    rows = []
    fast = deblend(image, seeds, masks, origins, variant=Variant(anderson=5), trace=rows)
    assert fast.converged and fast.iterations <= plain.iterations
    assert_same_fixed_point(fast, plain)
    updates = [row["update"] for row in rows]
    assert set(updates) <= {"plain", "anderson", "fallback"} and "anderson" in updates
    assert updates[0] == "plain"  # no history yet


def test_widen_starts_wider_and_ends_at_the_same_fixed_point():
    image, centers = lattice(1.6)
    seeds, masks, origins = cells(image, centers, 1.6)
    start = deblend(image, seeds, masks, origins, iterations=0)
    wide = deblend(image, seeds, masks, origins, iterations=0, variant=Variant(widen=0.02))
    np.testing.assert_allclose(start.params[:, [3, 5]] - wide.params[:, [3, 5]], np.log(1.02), rtol=0, atol=1e-12)
    assert np.array_equal(wide.params[:, 4], start.params[:, 4] / (1 + 0.02)) and (start.params[:, 4] != 0).any()
    assert np.array_equal(start.params[:, :3], wide.params[:, :3])
    # so every direction is 2 % wider: the precision matrices (inverse covariances) shrink by 1.02 ** 2
    np.testing.assert_allclose(precision_matrices(wide.params), precision_matrices(start.params) / 1.02**2, rtol=1e-12)
    given = deblend(image, seeds, masks, origins, iterations=0, init=start.params, variant=Variant(widen=0.02))
    assert np.array_equal(given.params, wide.params)  # a given init is widened the same way
    huge = deblend(image, seeds, masks, origins, iterations=0, variant=Variant(widen=10.0))
    assert (huge.status == "non_gaussian_fit").all()  # the widened start's status is recomputed
    assert_same_fixed_point(deblend(image, seeds, masks, origins, variant=Variant(widen=0.02)),
                            deblend(image, seeds, masks, origins))


def test_patch_applies_the_switches_in_the_last_pass_only():
    image, _ = lattice(1.6)
    with Patch(Variant(anderson=5, lm_steps=40, tile_free_px=16), full_iterations=30) as patch:
        measure_cds(image, seeds="peaks", methods=("gaussian",), area_max=400, backend="gpu")
    first, last = patch.passes
    assert (first["asked"], first["last"], first["continued"]) == (10, False, False)
    assert first["variant"]["anderson"] == 0 and first["variant"]["tile_free_px"] is None
    assert first["variant"]["lm_steps"] == 40
    assert {row["update"] for row in first["rounds"]} == {"plain"}
    assert last["last"] and last["continued"] and last["variant"] == {
        "lm_steps": 40, "tolerance": None, "freeze_background": False, "freeze_free": False, "anderson": 5,
        "widen": 0.0, "tile_free_px": 16, "continue_after_recheck": False}


def test_continue_after_recheck_starts_kept_holes_from_their_fits():
    # Thresholding misses the narrow holes; the re-check after pass 1 adds them and measure_cds rebuilds the cells.
    image, _ = lattice(1.8, small=8, seed=2)
    kwargs = dict(seeds="threshold", methods=("gaussian",), area_min=10, area_max=400, backend="gpu")
    with Patch() as plain_patch:
        plain = measure_cds(image, **kwargs)
    with Patch(Variant(continue_after_recheck=True)) as patch:
        continued = measure_cds(image, **kwargs)
        count = len(patch.passes)
        again = measure_cds(image, **kwargs)  # another measurement: nothing carries over from the first
    assert plain.meta["seeds_added_from_residual"] > 0
    assert patch.passes[0]["continued_holes"] == 0 and patch.passes[1]["continued_holes"] == patch.passes[0]["holes"]
    assert sum(p["used"] for p in patch.passes[1:count]) <= sum(p["used"] for p in plain_patch.passes[1:])
    assert patch.passes[count]["continued_holes"] == 0
    assert np.array_equal(again.deblend_model["params"], continued.deblend_model["params"])
    index = {key: i for i, key in enumerate(map(tuple, plain.seeds_xy.tolist()))}
    a, b = np.array([(index[key], j) for j, key in enumerate(map(tuple, continued.seeds_xy.tolist()))
                     if key in index]).T
    first, second = plain.measurements["gaussian"], continued.measurements["gaussian"]
    both = first.valid[a] & second.valid[b]
    assert both.sum() > 0.99 * len(plain.seeds_xy)
    delta = first.metrics["cd_equivalent_px"][a][both] - second.metrics["cd_equivalent_px"][b][both]
    assert np.abs(delta).max() < 0.02


def test_trace_rows():
    image, centers = lattice(2.0)
    image, centers = image[12:, 12:], centers - 12  # holes 3 px from the top and left edges
    seeds, masks, origins = cells(image, centers, 2.0)
    edge = np.minimum(np.minimum(seeds[:, 0], image.shape[1] - 1 - seeds[:, 0]),
                      np.minimum(seeds[:, 1], image.shape[0] - 1 - seeds[:, 1])) / (2.0 * FWHM)
    assert all(band.any() for band in (edge < 1, (edge >= 1) & (edge < 2), edge >= 2))
    rows = []
    state = deblend(image, seeds, masks, origins, trace=rows)
    assert state.converged and len(rows) == state.iterations >= 3
    for number, row in enumerate(rows, 1):
        assert row.keys() == KEYS and row["round"] == number and row["update"] == "plain"
        for value in row.values():
            assert type(value) in (int, float, str, list, type(None))
        assert all(type(value) is float and math.isfinite(value) for value in row["edge_p99"])
        assert type(row["ok"]) is type(row["free_px"]) is type(row["measured_tiles"]) is int
        assert abs(row["bg_mean"] - 20) < 0.5  # the page's background level
        assert all(math.isnan(row[key]) == (number == 1) for key in ("cos", "bg_max", "bg_rms"))
        assert row["seconds"] > 0 and row["width_p99"] <= row["step_p99"] and row["centre_p99"] <= row["step_p99"]
    assert rows[-1]["ok"] == (state.status == "ok").sum()
    assert all(row["step_p99"] >= 1e-3 for row in rows[:-1]) and rows[-1]["step_p99"] < 1e-3  # the stop rule's number
