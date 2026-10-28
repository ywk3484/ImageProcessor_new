"""Fused local estimators retain the reference grid and connectivity rules."""

import numpy as np
import pytest

from subpx.backends import cp
from subpx._cpu.invariant_cd import _half_areas
from subpx._gpu.local_cd import AREA_METHODS, measure_local_areas


pytestmark = pytest.mark.skipif(cp is None, reason="CuPy unavailable")


def _reference(cells, masks, origins, factor, methods=AREA_METHODS, peak_pixels=4.0):
    specs = [dict(name=name, level=0.2 if name == "local_tail_area" else 0.5,
                  background="min" if name == "local_half_area_min_avg" else "median",
                  peak_pixels=None if name == "local_half_area" else peak_pixels, cuts=None)
             for name in methods]
    return _half_areas(cells, masks, origins, 16, specs, factor=factor,
                       outlines=False, metrics_only=True)


def _compare(reference, found):
    assert found is not None
    for name, expected in reference.items():
        actual = found[name]
        np.testing.assert_array_equal(actual.status, expected.status)
        np.testing.assert_array_equal(actual.valid, expected.valid)
        np.testing.assert_array_equal(actual.metrics["area_px2"], expected.metrics["area_px2"])
        np.testing.assert_allclose(actual.centers_xy, expected.centers_xy, rtol=0, atol=1e-10)
        for key in expected.metrics:
            if key == "background":  # raw-cell API gives the actual local level here
                np.testing.assert_allclose(actual.metrics[key], expected.metrics["local_level"], atol=1e-10)
            else:
                np.testing.assert_allclose(actual.metrics[key], expected.metrics[key], rtol=1e-10, atol=1e-10)
        assert actual.contours_xy.shape == (len(actual.valid), 0, 2)


@pytest.mark.parametrize("factor", [1, 4])
@pytest.mark.parametrize("size", [9, 12, 15])
def test_fused_local_matches_raw_reference_on_irregular_masks(factor, size):
    rng = np.random.default_rng(24 + size)
    n = 19
    yy, xx = np.mgrid[:size, :size]
    cells = np.array([30 + 130 * np.exp(-((xx - rng.uniform(0.3, 0.7) * size) ** 2
                                         + (yy - rng.uniform(0.3, 0.7) * size) ** 2)
                                       / (2 * (0.14 * size) ** 2))
                      + rng.normal(0, 0.4, (size, size)) for _ in range(n)])
    masks = rng.random((n, size, size)) > 0.04
    masks[0] = False
    masks[1] = False
    masks[1, size // 2, size // 2] = True
    cells[2] = 14  # no signal
    # Different shapes clipped at the end of the bucket exercise zero padding.
    sizes = np.full((n, 2), size, dtype=np.int64)
    sizes[3:8, 0] -= 2
    masks[3:8, -2:] = False
    origins = np.column_stack((np.arange(n) * size, np.zeros(n))).astype(np.int64)
    image = np.concatenate(cells, axis=1)
    labels = np.concatenate([np.where(masks[i], i, -1) for i in range(n)], axis=1).astype(np.int32)
    found = measure_local_areas(image, labels, origins, sizes, np.arange(n),
                               methods=AREA_METHODS, factor=factor)
    _compare(_reference(cells, masks, origins, factor), found)


@pytest.mark.parametrize("methods", [AREA_METHODS[:1], AREA_METHODS[1:2], AREA_METHODS[2:]])
def test_fused_local_method_subset_and_peak_pixel_setting(methods):
    yy, xx = np.mgrid[:12, :12]
    cells = (20 + 100 * np.exp(-((xx - 5.4) ** 2 + (yy - 6.2) ** 2) / 6))[None]
    masks = np.ones(cells.shape, bool)
    origins = np.zeros((1, 2), np.int64)
    found = measure_local_areas(cells[0], np.zeros((12, 12), np.int32), origins, np.array([[12, 12]]),
                               np.array([0]), methods=methods, factor=1, half_area_peak_pixels=2.5)
    _compare(_reference(cells, masks, origins, 1, methods, peak_pixels=2.5), found)


def test_native_flood_fill_follows_long_winding_component():
    size = 20
    image = np.zeros((size, size))
    for i, y in enumerate(range(1, size - 1, 2)):
        image[y, 1:size - 1] = 100
        if y + 2 < size - 1:
            image[y + 1, size - 2 if i % 2 == 0 else 1] = 100
    image[1, 1] = 120
    origins = np.zeros((1, 2), np.int64)
    found = measure_local_areas(image, np.zeros_like(image, dtype=np.int32), origins,
                               np.array([[size, size]]), np.array([0]), methods=AREA_METHODS, factor=1)
    reference = _reference(image[None], np.ones((1, size, size), bool), origins, 1)
    _compare(reference, found)
    assert found["local_half_area"].metrics["area_px2"][0] == np.count_nonzero(image)


def test_native_quantized_threshold_ties_are_excluded_exactly():
    # Peak 100, border 20: the half-height threshold is exactly 60. Every
    # 60-valued pixel must stay outside the strict > region, without a spline
    # reconstruction nudging selected pixels just above the threshold.
    size = 9
    image = np.full((size, size), 20.0)
    image[2:7, 2:7] = 60
    image[4, 4] = 100
    origins = np.zeros((1, 2), np.int64)
    found = measure_local_areas(image, np.zeros_like(image, dtype=np.int32), origins,
                               np.array([[size, size]]), np.array([0]), methods=AREA_METHODS, factor=1)
    reference = _reference(image[None], np.ones((1, size, size), bool), origins, 1)
    _compare(reference, found)
    assert reference["local_half_area"].metrics["area_px2"][0] == 1
    assert found["local_half_area"].metrics["threshold"][0] == 60
    assert found["local_half_area_min_avg"].metrics["area_px2"][0] == 25


def test_native_radix_top_mean_waits_for_ties_from_every_warp():
    # More than eight peak samples selects the radix path. Four samples are
    # brighter than the remaining sixty tied samples, so multiple warps must
    # finish writing the selected values before thread zero sorts them.
    size, count = 12, 257
    cells = np.full((count, size, size), 20.0)
    cells[:, 2:10, 2:10] = 100
    cells[:, 5:7, 5:7] = 120
    cells += (np.arange(count) % 7)[:, None, None]
    masks = np.ones_like(cells, dtype=bool)
    origins = np.column_stack((np.arange(count) * size, np.zeros(count))).astype(np.int64)
    sizes = np.full((count, 2), size, dtype=np.int64)
    image = np.concatenate(cells, axis=1)
    labels = np.concatenate([np.full((size, size), i, np.int32) for i in range(count)], axis=1)
    reference = _reference(cells, masks, origins, 1, peak_pixels=64)
    for _ in range(4):
        found = measure_local_areas(image, labels, origins, sizes, np.arange(count),
                                   methods=AREA_METHODS, factor=1, half_area_peak_pixels=64)
        _compare(reference, found)
        np.testing.assert_array_equal(found["local_half_area_min_avg"].metrics["peak_signal"],
                                      np.full(count, 81.25))


def test_oversized_fused_cell_requests_bounded_fallback():
    image = np.ones((100, 100))
    found = measure_local_areas(image, np.zeros_like(image, dtype=np.int32), np.zeros((1, 2), np.int64),
                               np.array([[100, 100]]), np.array([0]), methods=AREA_METHODS, factor=1)
    assert found is None


@pytest.mark.parametrize("factor", [1, 4])
@pytest.mark.parametrize("n_top", [1, 7, 21, 33, 64, 97, 160])
def test_parallel_peak_sort_preserves_arbitrary_sample_counts(factor, n_top):
    size, count = 12, 7
    rng = np.random.default_rng(914)
    yy, xx = np.mgrid[:size, :size]
    cells = np.array([20 + 100 * np.exp(-((xx - 5.4 - i / 20) ** 2 + (yy - 6.1) ** 2) / 8)
                      + rng.normal(0, 0.2, (size, size)) for i in range(count)])
    masks = np.ones(cells.shape, bool)
    masks[0] = False
    masks[1] = False
    masks[1, 5:7, 5:7] = True  # fewer usable samples than several requested peaks
    cells[2] = 20
    cells[3, 3:9, 3:9] = 100  # ties in the selected tail
    origins = np.column_stack((np.arange(count) * size, np.zeros(count))).astype(np.int64)
    image = np.concatenate(cells, axis=1)
    labels = np.concatenate([np.where(masks[i], i, -1) for i in range(count)], axis=1).astype(np.int32)
    peak_pixels = n_top / (factor * factor)
    found = measure_local_areas(image, labels, origins, np.full((count, 2), size), np.arange(count),
                               methods=AREA_METHODS, factor=factor, half_area_peak_pixels=peak_pixels)
    _compare(_reference(cells, masks, origins, factor, peak_pixels=peak_pixels), found)


def test_interpolated_scratch_chunks_preserve_large_cell_results(monkeypatch):
    from subpx._gpu import local_cd

    size, count = 36, 7  # these cells exceed the old all-shared-memory implementation
    yy, xx = np.mgrid[:size, :size]
    cells = np.array([30 + 90 * np.exp(-((xx - 17.4 - i / 10) ** 2 + (yy - 18.3) ** 2) / 35)
                      for i in range(count)])
    masks = np.ones(cells.shape, bool)
    masks[2, :4] = False
    masks[4, :, -3:] = False
    sizes = np.full((count, 2), size)
    for index in (1, 5):
        sizes[index] = 24
        masks[index, 24:] = False
        masks[index, :, 24:] = False
    origins = np.column_stack((np.arange(count) * size, np.zeros(count))).astype(np.int64)
    image = np.concatenate(cells, axis=1)
    labels = np.concatenate([np.where(masks[i], i, -1) for i in range(count)], axis=1).astype(np.int32)
    original_empty = cp.empty
    scratch_shapes = []

    def empty(shape, *args, **kwargs):
        if isinstance(shape, int):
            scratch_shapes.append(shape)
        return original_empty(shape, *args, **kwargs)

    monkeypatch.setattr(local_cd, "_WORK_BYTES", 3 * (4 * size) ** 2 * 8)
    monkeypatch.setattr(cp, "empty", empty)
    found = measure_local_areas(image, labels, origins, sizes, np.arange(count),
                               methods=AREA_METHODS, factor=4)
    assert scratch_shapes == [3 * (4 * size) ** 2]
    _compare(_reference(cells, masks, origins, 4), found)
