# Radial Symmetry Accuracy Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix inaccurate radial symmetry center detection caused by Otsu thresholding missing blobs and lack of post-hoc quality filtering.

**Architecture:** Three independent fixes: (1) add Triangle threshold option to `_segment_binary`, (2) add in-bounds + residual quality gates to `_radial_symmetry_batch_gpu`, (3) validation tests. The Triangle fix addresses the root cause (incomplete Voronoi seeds); the quality gates address the secondary issue (edge blob contamination).

**Tech Stack:** Python, NumPy, OpenCV (`cv2.THRESH_TRIANGLE`), CuPy, pytest

**Spec:** `docs/superpowers/specs/2026-03-26-radial-symmetry-accuracy-fix-design.md`

---

## File Structure

| File | Responsibility | Change Type |
|---|---|---|
| `subpx/_cpu/centers.py` | Binary segmentation (shared by CPU+GPU paths) | Modify `_segment_binary()` |
| `subpx/_gpu/centers.py` | Radial symmetry batch refinement | Modify `_radial_symmetry_batch_gpu()` |
| `tests/test_radial_symmetry.py` | Radial symmetry unit + integration tests | Add new test functions |

No new files created. The GPU path imports `_segment_binary` from the CPU module (line 7 of `subpx/_gpu/centers.py`), so the threshold change propagates automatically.

---

### Task 1: Add Triangle Threshold Support

**Files:**
- Modify: `subpx/_cpu/centers.py:247-264` (`_segment_binary` function)
- Test: `tests/test_radial_symmetry.py`

- [ ] **Step 1: Write the failing test for triangle threshold**

Add to `tests/test_radial_symmetry.py`:

```python
def test_segment_binary_triangle_threshold():
    """_segment_binary accepts threshold='triangle' and produces a valid binary mask."""
    from subpx._cpu.centers import _segment_binary

    # Create image with faint blobs on dark background
    img = np.zeros((64, 64), dtype=np.uint8)
    # Bright blob
    yy, xx = np.mgrid[:64, :64]
    img += (100 * np.exp(-((xx - 32)**2 + (yy - 32)**2) / (2 * 2.0**2))).astype(np.uint8)

    g, bw = _segment_binary(img, threshold="triangle")
    assert bw.shape == (64, 64)
    assert bw.dtype == np.uint8
    assert bw.max() == 255  # at least some foreground pixels
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_radial_symmetry.py::test_segment_binary_triangle_threshold -v`
Expected: FAIL with `ValueError: Currently supported threshold values: 'otsu'`

- [ ] **Step 3: Implement triangle threshold support**

In `subpx/_cpu/centers.py`, replace the validation and thresholding logic in `_segment_binary()`:

```python
def _segment_binary(image: np.ndarray, *, threshold: str = "otsu", invert: bool = False, morph_open: int = 0, morph_close: int = 0):
    _require_cv2()
    g = np.asarray(image)
    if g.ndim != 2:
        raise ValueError("Expected 2D grayscale image.")

    _THRESHOLD_METHODS = {"otsu": cv2.THRESH_OTSU, "triangle": cv2.THRESH_TRIANGLE}
    if threshold not in _THRESHOLD_METHODS:
        raise ValueError(f"Unsupported threshold method: {threshold!r}. "
                         f"Supported: {sorted(_THRESHOLD_METHODS)}")

    thr_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, bw = cv2.threshold(g, 0, 255, thr_type | _THRESHOLD_METHODS[threshold])

    if morph_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=int(morph_open))
    if morph_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=int(morph_close))
    return g, bw
```

Key changes:
- Lines 252-253: Replace single-value check with dictionary lookup
- Line 256: Use `_THRESHOLD_METHODS[threshold]` instead of hardcoded `cv2.THRESH_OTSU`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_radial_symmetry.py::test_segment_binary_triangle_threshold -v`
Expected: PASS

- [ ] **Step 5: Write test that Triangle detects more faint blobs than Otsu**

Add to `tests/test_radial_symmetry.py`:

```python
def test_triangle_detects_more_faint_blobs_than_otsu():
    """Triangle threshold captures faint blobs that Otsu misses.

    Creates a grid of blobs with varying brightness on a large dark background.
    The faintest blobs should be detected by Triangle but missed by Otsu
    because Otsu's threshold is pulled too high by the dominant background.
    """
    import cv2
    from subpx._cpu.centers import _segment_binary
    from subpx._cpu.components import connected_components_stats_cpu

    img = np.zeros((200, 200), dtype=np.uint8)
    yy, xx = np.mgrid[:200, :200]
    # 4x4 grid of blobs with decreasing brightness
    positions = [(30 + 40*i, 30 + 40*j) for i in range(4) for j in range(4)]
    brightnesses = [200, 180, 150, 120, 100, 80, 70, 60,
                    55, 50, 45, 40, 35, 30, 25, 20]
    for (cx, cy), brightness in zip(positions, brightnesses):
        img += (brightness * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 1.5**2))).astype(np.uint8)

    _, bw_otsu = _segment_binary(img, threshold="otsu")
    _, bw_tri = _segment_binary(img, threshold="triangle")

    comp_otsu = connected_components_stats_cpu(bw_otsu, connectivity=8)
    comp_tri = connected_components_stats_cpu(bw_tri, connectivity=8)
    # num_labels includes background (label 0), so subtract 1
    n_otsu = comp_otsu.num_labels - 1
    n_tri = comp_tri.num_labels - 1

    assert n_tri > n_otsu, (
        f"Triangle ({n_tri} blobs) should detect more blobs than Otsu ({n_otsu} blobs)"
    )
```

- [ ] **Step 6: Run the new test**

Run: `pytest tests/test_radial_symmetry.py::test_triangle_detects_more_faint_blobs_than_otsu -v`
Expected: PASS (Triangle's lower threshold should capture faint blobs that Otsu misses)

- [ ] **Step 7: Run existing tests to confirm no regression**

Run: `pytest tests/test_radial_symmetry.py -v`
Expected: All existing tests PASS. The `_segment_binary` change only adds a new option; existing code still passes `threshold="otsu"` (the default).

- [ ] **Step 8: Commit**

```bash
git add subpx/_cpu/centers.py tests/test_radial_symmetry.py
git commit -m "feat: add triangle threshold support to _segment_binary

Add threshold='triangle' option using cv2.THRESH_TRIANGLE for images
where background dominates the histogram and Otsu sets threshold too
high. Keeps 'otsu' as default for backward compatibility."
```

---

### Task 2: Add Post-hoc Quality Gates to Radial Symmetry

**Files:**
- Modify: `subpx/_gpu/centers.py:974-1174` (`_radial_symmetry_batch_gpu` function)
- Test: `tests/test_radial_symmetry.py`

- [ ] **Step 1: Write the failing test for in-bounds rejection**

Add to `tests/test_radial_symmetry.py`:

```python
@skipno_gpu
def test_radial_symmetry_rejects_out_of_bounds_center():
    """A heavily asymmetric ROI should produce an out-of-bounds fit that gets rejected (NaN)."""
    from subpx._gpu.centers import _radial_symmetry_batch_gpu

    # Create a blob at the right edge of a 15x15 ROI — center is outside
    # the left portion, simulating a clipped edge blob
    blob = _make_gaussian_blob(14.0, 7.0, sigma=1.5, shape=(15, 15))
    # Mask out right half to simulate asymmetric clipping
    mask = np.ones((15, 15), dtype=bool)
    mask[:, 10:] = False

    rois = np.stack([blob])
    masks = np.stack([mask])
    centers, residuals = _radial_symmetry_batch_gpu(rois, masks, upsample_factor=4, device=0)

    # The fit should be NaN (rejected) because the true center is outside
    # the valid masked region
    assert np.isnan(centers[0, 0]) or np.isnan(centers[0, 1]), (
        f"Expected NaN for out-of-bounds fit, got center ({centers[0, 0]:.2f}, {centers[0, 1]:.2f})"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_radial_symmetry.py::test_radial_symmetry_rejects_out_of_bounds_center -v`
Expected: FAIL — currently the function returns a (possibly incorrect but finite) center without any bounds check.

- [ ] **Step 3: Write the failing test for edge blob rejection in full pipeline**

Add to `tests/test_radial_symmetry.py`:

```python
@skipno_gpu
def test_radial_symmetry_rejects_edge_blobs():
    """Blobs at the image boundary should be rejected by quality gates.

    Creates an image with one center blob (should be detected) and four
    edge blobs (one on each side, partially outside image). The edge blobs
    should be rejected because their asymmetric ROIs produce poor fits.
    """
    from subpx.centers import detect_centers

    img = np.zeros((64, 64), dtype=np.uint8)
    yy, xx = np.mgrid[:64, :64]

    # Center blob — should survive
    img += (200 * np.exp(-((xx - 32)**2 + (yy - 32)**2) / (2 * 2.0**2))).astype(np.uint8)
    # Edge blobs — placed so bbox doesn't touch boundary but padded ROI is clipped
    # Blob near left edge (x=2)
    img += (200 * np.exp(-((xx - 2)**2 + (yy - 32)**2) / (2 * 2.0**2))).astype(np.uint8)
    # Blob near top edge (y=2)
    img += (200 * np.exp(-((xx - 32)**2 + (yy - 2)**2) / (2 * 2.0**2))).astype(np.uint8)

    res = detect_centers(
        img, backend="gpu", refine="radial_symmetry",
        area_min=4, area_max=200, upsample_factor=4, threshold="triangle",
    )
    # Only the center blob should survive quality gates.
    # The edge blobs may be detected as connected components but should
    # be filtered by the boundary check (bbox touches edge) or quality gates.
    # At minimum, the center blob should be in the results.
    assert res.centers_xy.shape[0] >= 1

    # Check that center blob is detected near (32, 32)
    dists_to_center = np.sqrt(np.sum((res.centers_xy - [32, 32])**2, axis=1))
    assert np.min(dists_to_center) < 1.0, "Center blob not detected"

    # Edge blobs should NOT be in the results
    for edge_pos in [[2, 32], [32, 2]]:
        dists = np.sqrt(np.sum((res.centers_xy - edge_pos)**2, axis=1))
        if len(dists) > 0:
            assert np.min(dists) > 3.0, (
                f"Edge blob near {edge_pos} should have been rejected, "
                f"but found center at distance {np.min(dists):.2f} px"
            )
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_radial_symmetry.py::test_radial_symmetry_rejects_edge_blobs -v`
Expected: FAIL — edge blobs currently pass through without rejection.

- [ ] **Step 5: Implement quality gates in `_radial_symmetry_batch_gpu`**

In `subpx/_gpu/centers.py`, modify the coordinate transform loop (step 13, lines 1154-1174) to add quality gates:

Replace the existing loop at lines 1154-1174 with:

```python
    # Convert from centered upsampled coords to original ROI pixel coords
    # xc is relative to center of upsampled grid: xc_pix_up = xc + (w_up - 1) / 2
    # Then map back: xc_orig = xc_pix_up * (w_orig - 1) / (w_up - 1)
    centers = np.empty((N, 2), dtype=np.float64)
    for i in range(N):
        w_up_i = float(up_ws[i])
        h_up_i = float(up_hs[i])
        w_orig_i = float(orig_ws[i])
        h_orig_i = float(orig_hs[i])

        xc_pix_up = xc_cpu[i] + (w_up_i - 1.0) / 2.0
        yc_pix_up = yc_cpu[i] + (h_up_i - 1.0) / 2.0

        if factor > 1 and w_up_i > 1 and h_up_i > 1:
            xc_orig = xc_pix_up * (w_orig_i - 1.0) / (w_up_i - 1.0)
            yc_orig = yc_pix_up * (h_orig_i - 1.0) / (h_up_i - 1.0)
        else:
            xc_orig = xc_pix_up
            yc_orig = yc_pix_up

        # -- Quality gate 1: in-bounds check --
        if not (0 <= xc_orig < w_orig_i and 0 <= yc_orig < h_orig_i):
            xc_orig = np.nan
            yc_orig = np.nan
            residual_cpu[i] = np.inf

        centers[i, 0] = xc_orig
        centers[i, 1] = yc_orig

    return centers, residual_cpu
```

Key change: After computing `xc_orig` and `yc_orig`, check if they fall within `[0, w_orig)` and `[0, h_orig)`. If not, set to NaN. This mirrors logquad's `in_bounds` check at line 108 of the same file.

Note: The residual gate is intentionally deferred to a separate step. The in-bounds check alone should handle the edge rejection case. The residual threshold requires empirical calibration.

- [ ] **Step 6: Run both new tests to verify they pass**

Run: `pytest tests/test_radial_symmetry.py::test_radial_symmetry_rejects_out_of_bounds_center tests/test_radial_symmetry.py::test_radial_symmetry_rejects_edge_blobs -v`
Expected: PASS

- [ ] **Step 7: Run ALL existing tests to confirm no regression**

Run: `pytest tests/test_radial_symmetry.py -v`
Expected: All tests PASS, including existing ones. The quality gates set NaN only for out-of-bounds fits — well-centered Gaussian blobs in the existing test suite will have in-bounds centers and should not be affected.

If any existing test fails, the in-bounds check may be too strict. Check which test failed and inspect the fitted center coordinates to determine if the threshold needs adjustment.

- [ ] **Step 8: Commit**

```bash
git add subpx/_gpu/centers.py tests/test_radial_symmetry.py
git commit -m "feat: add in-bounds quality gate to radial symmetry

Reject radial symmetry fits where the center falls outside the ROI
bounds. Mirrors logquad's in_bounds check. Prevents edge-adjacent
blobs with clipped/asymmetric ROIs from producing biased results."
```

---

### Task 3: Integration Validation

**Files:**
- Test: `tests/test_radial_symmetry.py`

- [ ] **Step 1: Write Voronoi correctness test**

Add to `tests/test_radial_symmetry.py`:

```python
@skipno_gpu
def test_voronoi_each_cell_contains_one_seed():
    """After Triangle thresholding, each Voronoi cell should contain exactly one seed.

    This validates that the Voronoi kernel is correct when given complete seeds.
    If this test fails, the Voronoi kernel has a bug independent of thresholding.
    """
    from subpx._gpu.voronoi import compute_voronoi_labels_gpu

    # Create a regular grid of seeds (simulating well-detected blob centroids)
    spacing = 10
    seeds = []
    for y in range(spacing, 100, spacing):
        for x in range(spacing, 100, spacing):
            seeds.append([float(x), float(y)])
    seeds = np.array(seeds, dtype=np.float64)
    N = len(seeds)

    labels = compute_voronoi_labels_gpu(seeds, (100, 100), device=0)

    # Each cell should contain exactly one seed center
    for i in range(N):
        sx, sy = int(round(seeds[i, 0])), int(round(seeds[i, 1]))
        # Clamp to image bounds
        sx = min(max(sx, 0), 99)
        sy = min(max(sy, 0), 99)
        assert labels[sy, sx] == i, (
            f"Seed {i} at ({seeds[i, 0]:.1f}, {seeds[i, 1]:.1f}) is in Voronoi cell "
            f"{labels[sy, sx]}, expected cell {i}"
        )

    # No cell should contain a different seed's center
    for i in range(N):
        cell_mask = labels == i
        other_seeds_in_cell = 0
        for j in range(N):
            if j == i:
                continue
            sx, sy = int(round(seeds[j, 0])), int(round(seeds[j, 1]))
            sx = min(max(sx, 0), 99)
            sy = min(max(sy, 0), 99)
            if cell_mask[sy, sx]:
                other_seeds_in_cell += 1
        assert other_seeds_in_cell == 0, (
            f"Voronoi cell {i} contains {other_seeds_in_cell} other seed center(s)"
        )
```

- [ ] **Step 2: Write full pipeline integration test with Triangle threshold**

Add to `tests/test_radial_symmetry.py`:

```python
@skipno_gpu
def test_radial_symmetry_triangle_full_pipeline():
    """Full pipeline: triangle threshold + radial symmetry on closely-packed blobs.

    This is the end-to-end test that validates the entire fix chain:
    triangle threshold → complete seeds → correct Voronoi → accurate centers.
    """
    from subpx.centers import detect_centers

    # Create 3x3 grid of closely-packed blobs (6px spacing, ~3px diameter)
    img = np.zeros((64, 64), dtype=np.uint8)
    yy, xx = np.mgrid[:64, :64]
    true_centers = []
    for row in range(3):
        for col in range(3):
            cx = 20.0 + col * 8.0
            cy = 20.0 + row * 8.0
            img += (150 * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 1.5**2))).astype(np.uint8)
            true_centers.append([cx, cy])
    true_centers = np.array(true_centers)

    res = detect_centers(
        img, backend="gpu", refine="radial_symmetry",
        area_min=1, area_max=200, upsample_factor=4, threshold="triangle",
    )

    # All 9 blobs should be detected
    assert res.centers_xy.shape[0] == 9, (
        f"Expected 9 centers, got {res.centers_xy.shape[0]}"
    )

    # Match detected to true centers (nearest neighbor)
    from scipy.spatial import cKDTree
    tree = cKDTree(true_centers)
    dists, _ = tree.query(res.centers_xy)
    assert np.all(dists < 0.5), (
        f"Max center error {dists.max():.3f} px exceeds 0.5 px"
    )
```

- [ ] **Step 3: Run validation tests**

Run: `pytest tests/test_radial_symmetry.py::test_voronoi_each_cell_contains_one_seed tests/test_radial_symmetry.py::test_radial_symmetry_triangle_full_pipeline -v`
Expected: PASS

If `test_voronoi_each_cell_contains_one_seed` fails, the Voronoi kernel has a bug independent of thresholding — file a separate issue.

If `test_radial_symmetry_triangle_full_pipeline` fails on blob count, the triangle threshold may still be too high for the test image. Try adjusting blob brightness (increase from 150) or `area_min` (decrease to 1).

- [ ] **Step 4: Run the full test suite**

Run: `pytest tests/test_radial_symmetry.py -v`
Expected: ALL tests PASS (existing + new).

- [ ] **Step 5: Commit**

```bash
git add tests/test_radial_symmetry.py
git commit -m "test: add validation tests for radial symmetry accuracy fix

- Voronoi cell correctness (one seed per cell)
- Full pipeline with triangle threshold on closely-packed blobs
- Edge blob rejection via quality gates
- Triangle vs Otsu blob count comparison"
```

---

### Deferred: Residual Threshold Gate

The spec defines a Gate 2 (residual threshold) for rejecting poor radial symmetry fits. This is intentionally deferred from this plan because:

1. The residual threshold requires empirical calibration from real data — we need to observe typical residual values for good fits vs. bad fits after the in-bounds gate is in place.
2. The in-bounds gate alone should handle the primary edge-contamination issue.
3. Adding a poorly calibrated residual threshold risks rejecting valid fits.

**Follow-up**: After this plan is implemented and validated on real photomask images, measure the residual distribution and add a residual gate if needed.
