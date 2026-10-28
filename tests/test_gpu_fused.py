"""The GPU kernels that replace array code in the CD methods give the same per-hole results as that code.

Each kernel is compared with the CuPy array code it replaces (the kernel switched off): masks, regions, counts and
statuses must be equal; values built from sums may differ by rounding only (the kernels add in another order).
"""

import numpy as np
import pytest

from subpx.backends import cp

pytestmark = pytest.mark.skipif(cp is None, reason="CuPy not available")


def _cells(rng, k, h, w):
    """Gaussian spots with noise on random masks; a few empty, full and one-pixel masks."""
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = rng.uniform(0.3 * h, 0.7 * h, k), rng.uniform(0.3 * w, 0.7 * w, k)
    sigma = rng.uniform(0.8, 0.25 * min(h, w), k)
    amp = rng.uniform(0.0, 100.0, k)
    cells = (amp[:, None, None] * np.exp(-((yy - cy[:, None, None]) ** 2 + (xx - cx[:, None, None]) ** 2)
                                        / (2 * sigma[:, None, None] ** 2)) + rng.normal(0, 2.0, (k, h, w)))
    y0, x0 = rng.integers(0, h // 4 + 1, k), rng.integers(0, w // 4 + 1, k)
    y1, x1 = h - rng.integers(0, h // 4 + 1, k), w - rng.integers(0, w // 4 + 1, k)
    masks = (yy >= y0[:, None, None]) & (yy < y1[:, None, None]) & (xx >= x0[:, None, None]) & (xx < x1[:, None, None])
    masks &= rng.random((k, h, w)) > 0.05
    masks[0] = False
    masks[1] = True
    masks[2] = False
    masks[2, h // 2, w // 2] = True
    return cells, masks


def _compare(a, b, tol=1e-11):
    """Per-hole results of two runs: equal statuses and valid flags, equal pixel-count areas (half_area family);
    other values equal to ``tol`` (relative, and px for positions)."""
    assert a.keys() == b.keys()
    for name in a:
        ra, rb = a[name], b[name]
        np.testing.assert_array_equal(ra.status, rb.status, err_msg=name)
        np.testing.assert_array_equal(ra.valid, rb.valid, err_msg=name)
        if "threshold" in ra.metrics:
            np.testing.assert_array_equal(ra.metrics["area_px2"], rb.metrics["area_px2"], err_msg=name)
        for key, value in ra.metrics.items():
            np.testing.assert_allclose(value, rb.metrics[key], rtol=tol, atol=tol, err_msg=f"{name} {key}")
        np.testing.assert_allclose(ra.centers_xy, rb.centers_xy, rtol=0, atol=tol, err_msg=name)
        np.testing.assert_allclose(ra.contours_xy, rb.contours_xy, rtol=0, atol=tol, err_msg=name)


def _without_kernels(monkeypatch):
    """The array code: the kernels report that a cell does not fit."""
    from subpx._gpu import regions, windows

    monkeypatch.setattr(regions, "measure_regions", lambda *a, **k: None)
    monkeypatch.setattr(windows, "wide_cells", lambda *a, **k: None)


@pytest.mark.parametrize("outlines", [False, True])
def test_region_kernel_matches_the_array_code_for_every_threshold(monkeypatch, outlines):
    from subpx._cpu.invariant_cd import _half_areas

    rng = np.random.default_rng(3)
    for k, h, w in ((300, 13, 12), (120, 25, 25), (40, 7, 31)):
        cells, masks = _cells(rng, k, h, w)
        zone = masks & (rng.random(masks.shape) > 0.2)
        cuts = rng.uniform(-5, 60, k)
        specs = [{"name": f"{background}{'_avg' if averaged else ''}", "level": 0.35, "background": background,
                  "peak_pixels": 4.0 if averaged else None, "cuts": cuts if background == "given" else None}
                 for background in ("median", "min", "zero", "edge", "foot", "given") for averaged in (False, True)]
        origins = rng.integers(0, 500, (k, 2)).astype(float)
        run = lambda: _half_areas(cp.asarray(cells), cp.asarray(masks), origins, 16, specs, zone_masks=cp.asarray(zone),
                                  outlines=outlines, xp=cp)
        fused = run()
        with monkeypatch.context() as m:
            _without_kernels(m)
            arrays = run()
        _compare(fused, arrays)
        measured = sum((r.status != "no_signal").sum() for r in fused.values())
        assert measured > len(specs) * k // 2 and sum(r.valid.sum() for r in fused.values()) > 0


def test_masked_top_mean_equals_top_mean_of_the_masked_array():
    from subpx._cpu.invariant_cd import _nearest_pixels, _zoom_operators
    from subpx._gpu.regions import top_mean, top_mean_masked

    rng = np.random.default_rng(4)
    for k, h, w in ((200, 13, 12), (50, 25, 25)):
        cells, masks = _cells(rng, k, h, w)
        zy, zx = _zoom_operators(h, 4)[0], _zoom_operators(w, 4)[0]
        big = cp.asarray(zy[None] @ cells @ zx.T[None])
        iy, ix = _nearest_pixels(h, 4), _nearest_pixels(w, 4)
        flat = cp.where(cp.asarray(masks)[:, iy[:, None], ix[None, :]], big, -cp.inf).reshape(k, -1)
        for n in (1, 64, 200):
            expected = cp.asnumpy(top_mean(flat, n))
            got = cp.asnumpy(top_mean_masked(big, cp.asarray(masks), cp.asarray(iy, dtype=cp.int32),
                                             cp.asarray(ix, dtype=cp.int32), n))
            np.testing.assert_array_equal(got, expected)


def _blob_page():
    """One long thin blob (area under area_max) makes the stack, every hole's square, large."""
    from scripts.benchmark_cd import lattice

    image, _ = lattice((300, 400), 9.5, 4.2, 20.0, seed=5)
    i = np.arange(30)
    image[100 + i, 150 + i] = 190
    image[100 + i, 151 + i] = 190
    return image


def _one_bucket(masks, step=4):
    return [(np.arange(len(masks)), *masks.shape[1:])]


def test_cells_measured_in_their_own_box_match_the_whole_stack(monkeypatch):
    """projection, the half_area family and the gaussian residual of each hole, computed in the sub-block of the
    stack that holds its mask, equal the whole-stack computation."""
    from subpx import measure_cds
    from subpx._cpu import invariant_cd
    from subpx._gpu.cd import gaussian_method

    result = measure_cds(_blob_page(), methods=("gaussian",), area_min=3, area_max=200, backend="gpu", n_angles=16)
    assert result.masks.shape[1] > 30 and len(invariant_cd._size_buckets(result.masks)) > 1
    cleaned, masks, origins = result.rois, result.masks, result.origins_xy.astype(float)
    params = result.deblend_model["params"][result.deblend_model["report"]].copy()
    params[:, 1:3] -= origins
    specs = [{"name": name, "level": 0.5, "background": background, "peak_pixels": peak, "cuts": None}
             for name, (background, peak) in {"half_area": ("median", None), "half_area_min": ("min", None),
                                               "half_area_min_avg": ("min", 4.0)}.items()]

    def run():
        found = invariant_cd._half_areas(cleaned, masks, origins, 32, specs, outlines=True, xp=cp)
        found["projection"] = invariant_cd._projection(cleaned, masks, origins, 32, xp=cp)
        found["gaussian"] = gaussian_method(params, result.measurements["gaussian"].status, cleaned, masks, origins, 32)
        return found

    own = run()
    with monkeypatch.context() as m:
        m.setattr(invariant_cd, "_size_buckets", _one_bucket)
        whole = run()
    _compare(own, whole)


def test_deblend_in_own_boxes_matches_the_whole_stack(monkeypatch):
    """Fits in each hole's own box: the same fits up to the order of the sums (to 1e-6), the same statuses."""
    from subpx import measure_cds
    from subpx._cpu import invariant_cd

    kw = dict(methods=("gaussian", "half_area", "tail_area"), area_min=3, area_max=200, backend="gpu", n_angles=16)
    own = measure_cds(_blob_page(), **kw)
    with monkeypatch.context() as m:
        m.setattr(invariant_cd, "_size_buckets", _one_bucket)
        whole = measure_cds(_blob_page(), **kw)
    np.testing.assert_allclose(own.deblend_model["params"], whole.deblend_model["params"], rtol=1e-6, atol=1e-6)
    _compare(own.measurements, whole.measurements, tol=1e-6)


def test_measure_cds_with_kernels_matches_the_array_code(monkeypatch):
    from scripts.benchmark_cd import lattice
    from subpx import measure_cds

    image, _ = lattice((330, 430), 9.5, 4.2, 20.0, seed=2)
    image = image[7:300, 5:400].copy()  # holes cut by the border
    image[150:152, 180:200] = 170  # two holes merged into one component (a bigger box)
    methods = ("gaussian", "projection", "half_area", "half_area_min", "half_area_min_avg", "tail_area",
               "tail_area_norm", "edge_area", "foot_area", "foot_area_ratio")
    kw = dict(methods=methods, area_min=3, area_max=200, backend="gpu", n_angles=16, outlines=False)
    fused = measure_cds(image, **kw).measurements
    with monkeypatch.context() as m:
        _without_kernels(m)
        arrays = measure_cds(image, **kw).measurements
    _compare(fused, arrays)
    # foot_area_ratio alone computes foot_area without reporting it: the same numbers.
    alone = measure_cds(image, **dict(kw, methods=("foot_area_ratio",))).measurements
    _compare({"foot_area_ratio": alone["foot_area_ratio"]}, {"foot_area_ratio": fused["foot_area_ratio"]})


@pytest.mark.parametrize("outlines", [False, True])
def test_batched_download_preserves_all_rows_and_methods(monkeypatch, outlines):
    """Deferred copies retain every number and outline across differently sized cells and chunks."""
    from subpx._cpu import invariant_cd
    from subpx._gpu import interpolate

    cells, masks = _cells(np.random.default_rng(42), 530, 13, 12)
    masks[::3, 8:] = False
    masks[1::3, :, 8:] = False
    specs = [{"name": background, "level": 0.35, "background": background,
              "peak_pixels": None if background == "median" else 4.0,
              "cuts": np.linspace(-5, 60, len(cells)) if background == "given" else None}
             for background in ("median", "min", "zero", "edge", "foot", "given")]
    origins = np.column_stack((np.arange(len(cells)), np.arange(len(cells)) * 2)).astype(float)
    monkeypatch.setattr(invariant_cd, "_GPU_CHUNK_BYTES", 1)  # exercise several bounded chunks
    copied = []
    original_copy = invariant_cd._to_numpy

    def copy(value):
        if isinstance(value, cp.ndarray):
            copied.append(value.size)
        return original_copy(value)

    monkeypatch.setattr(invariant_cd, "_to_numpy", copy)
    run = lambda: invariant_cd._half_areas(cells, masks, origins, 16, specs, zone_masks=masks,
                                           outlines=outlines, xp=cp)
    batched = run()
    assert len(copied) == 1  # masks are already on the host; one copy includes all methods and outlines
    with monkeypatch.context() as m:
        m.setattr(invariant_cd, "_GPU_RESULT_BYTES", 1)  # eager copy after every individual chunk and method
        m.setattr(interpolate, "enlarge_cells", lambda cell, zy, zx: zy[None] @ cell @ zx.T[None])
        eager = run()
    assert len(copied) > 10
    _compare(batched, eager)
