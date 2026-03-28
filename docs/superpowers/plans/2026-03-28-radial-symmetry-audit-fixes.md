# Radial Symmetry Audit Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 8 issues found in the radial symmetry code review: Voronoi grid correctness, quality gates, numerical robustness, background estimation, and test coverage.

**Architecture:** All fixes are localized edits to existing functions in `voronoi.py` and `centers.py`. No new files. Two new test functions. Each fix is independent — tasks can be implemented in any order, though Task 1 (Voronoi fix) and Task 7 (its test) should be paired.

**Tech Stack:** CuPy, NumPy, pytest

**Spec:** `docs/superpowers/specs/2026-03-28-radial-symmetry-audit-fixes-design.md`

---

### Task 1: Fix Voronoi Grid Search Radius (Fix A — Critical)

**Files:**
- Modify: `subpx/_gpu/voronoi.py:225-240`

- [ ] **Step 1: Implement adaptive search_radius with brute-force fallback**

In `compute_voronoi_labels_gpu`, replace the fixed `search_radius = 2` block (lines 225-240) with adaptive computation:

```python
        else:
            # Cell size ≈ average seed spacing
            cell_size = max(16, int(np.sqrt(HW / N)))

            # Compute max nearest-neighbor distance to set search_radius
            # O(N^2) on GPU — N is small (typically <50k seeds)
            diffs = seeds_gpu[:, None, :] - seeds_gpu[None, :, :]  # (N, N, 2)
            d2 = (diffs * diffs).sum(axis=2)  # (N, N)
            # Set self-distance to inf
            d2[cp.arange(N), cp.arange(N)] = cp.float32(1e30)
            max_nn_dist = float(cp.sqrt(d2.min(axis=1).max()))

            search_radius = int(np.ceil(max_nn_dist / cell_size)) + 1

            # Fall back to brute-force if search would be too wide
            if search_radius > 5:
                kernel = _get_kernel(device, "voronoi_brute")
                kernel(
                    (grid,), (block,),
                    (seeds_gpu, labels_gpu,
                     np.int64(HW), np.int32(W), np.int32(N)),
                )
            else:
                sorted_idx, cell_start, gw, gh = _bin_seeds_to_grid(
                    seeds_gpu, H, W, cell_size,
                )
                kernel = _get_kernel(device, "voronoi_grid")
                kernel(
                    (grid,), (block,),
                    (seeds_gpu, sorted_idx, cell_start, labels_gpu,
                     np.int64(HW), np.int32(W),
                     np.int32(gw), np.int32(gh),
                     np.int32(cell_size), np.int32(search_radius)),
                )
```

- [ ] **Step 2: Run existing Voronoi tests**

Run: `pytest tests/test_voronoi.py -v`
Expected: All 5 tests PASS (grid_matches_brute_force test especially)

- [ ] **Step 3: Commit**

```bash
git add subpx/_gpu/voronoi.py
git commit -m "fix: adaptive Voronoi grid search radius for non-uniform seeds"
```

---

### Task 2: Fix Degenerate Determinant Handling (Fix D)

**Files:**
- Modify: `subpx/_gpu/centers.py:1145-1148`

- [ ] **Step 1: Replace clamped determinant with direct NaN**

In `_radial_symmetry_batch_gpu`, replace lines 1145-1148:

Old:
```python
        det = smmw * sw - smw * smw  # (N,)
        det_safe = cp.where(cp.abs(det) < 1e-30, cp.float64(1e-30), det)
        xc = (-smbw * sw + smw * sbw) / det_safe   # (N,)
        yc = (smmw * sbw - smbw * smw) / det_safe   # (N,)
```

New:
```python
        det = smmw * sw - smw * smw  # (N,)
        bad_det = cp.abs(det) < cp.float64(1e-30)
        det_safe = cp.where(bad_det, cp.float64(1.0), det)
        xc = cp.where(bad_det, cp.nan, (-smbw * sw + smw * sbw) / det_safe)
        yc = cp.where(bad_det, cp.nan, (smmw * sbw - smbw * smw) / det_safe)
```

- [ ] **Step 2: Run radial symmetry tests**

Run: `pytest tests/test_radial_symmetry.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add subpx/_gpu/centers.py
git commit -m "fix: use NaN for degenerate determinant in radial symmetry WLS"
```

---

### Task 3: Add Residual Quality Gate (Fix B)

**Files:**
- Modify: `subpx/_gpu/centers.py:974-981` (signature) and `subpx/_gpu/centers.py:1190-1192` (before return)

- [ ] **Step 1: Add residual_max parameter to signature**

Change the function signature at line 974-981 from:

```python
def _radial_symmetry_batch_gpu(
    rois: np.ndarray,
    masks: np.ndarray,
    *,
    upsample_factor: int = 4,
    boundary_margin: int | None = None,
    device: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
```

To:

```python
def _radial_symmetry_batch_gpu(
    rois: np.ndarray,
    masks: np.ndarray,
    *,
    upsample_factor: int = 4,
    boundary_margin: int | None = None,
    residual_max: float | None = None,
    device: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
```

- [ ] **Step 2: Add residual filtering before return**

Before the `return centers, residual_cpu` at line 1192, add:

```python
    # -- Quality gate 2: residual threshold --
    if residual_max is not None:
        bad_residual = residual_cpu > residual_max
        centers[bad_residual] = np.nan
        residual_cpu[bad_residual] = np.inf
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_radial_symmetry.py -v`
Expected: All tests PASS (residual_max=None by default, so no behavior change)

- [ ] **Step 4: Commit**

```bash
git add subpx/_gpu/centers.py
git commit -m "feat: add residual_max quality gate to radial symmetry"
```

---

### Task 4: Fix Background Estimation and Zero-Pad Fill (Fixes C + E)

**Files:**
- Modify: `subpx/_gpu/centers.py:1429` (Fix C) and `subpx/_gpu/centers.py:1440-1444` (Fix E)

- [ ] **Step 1: Change median to 10th percentile**

In `_extract_voronoi_rois`, at line 1429, change:

```python
        bg = float(np.median(border_vals)) if border_vals.size > 0 else 0.0
```

To:

```python
        bg = float(np.percentile(border_vals, 10)) if border_vals.size > 0 else 0.0
```

- [ ] **Step 2: Fill padded stack with per-ROI background**

Replace lines 1440-1444:

Old:
```python
    rois_stack = np.zeros((len(rois_list), Hm, Wm), dtype=np.float64)
    masks_stack = np.zeros((len(rois_list), Hm, Wm), dtype=bool)
    for j, (roi, vmask) in enumerate(zip(rois_list, masks_list)):
        h_r, w_r = roi.shape
        rois_stack[j, :h_r, :w_r] = roi
        masks_stack[j, :h_r, :w_r] = vmask
```

New:
```python
    rois_stack = np.zeros((len(rois_list), Hm, Wm), dtype=np.float64)
    masks_stack = np.zeros((len(rois_list), Hm, Wm), dtype=bool)
    for j, (roi, vmask) in enumerate(zip(rois_list, masks_list)):
        h_r, w_r = roi.shape
        # Fill entire slice with background to avoid zero-pad gradient cliffs
        bg_val = roi[~vmask].mean() if (~vmask).any() else 0.0
        rois_stack[j, :, :] = bg_val
        rois_stack[j, :h_r, :w_r] = roi
        masks_stack[j, :h_r, :w_r] = vmask
```

Note: We use `roi[~vmask].mean()` here because `roi` is already background-filled (outside the Voronoi mask was set to `bg` in line 1430). So `roi[~vmask]` contains the fill value, and `.mean()` of a constant is that constant. This is equivalent to using `bg` but doesn't require threading the per-ROI `bg` through to this loop. Alternatively, store bg values in a list — but this is simpler.

Actually, the simpler approach: store bg values alongside rois. Let me revise. Instead, modify the first loop to collect bg values, then use them in the stacking loop.

Replace the full function body from line 1407 onward:

Old (lines 1407-1447):
```python
    H, W = gray.shape
    rois_list = []
    masks_list = []
    origins = []

    for j, (x, y, w, h, cx, cy) in enumerate(rows):
        x0 = max(0, int(x) - int(pad))
        y0 = max(0, int(y) - int(pad))
        x1 = min(W, int(x + w) + int(pad))
        y1 = min(H, int(y + h) + int(pad))
        roi = gray[y0:y1, x0:x1].astype(np.float64)
        vmask = voronoi_labels[y0:y1, x0:x1] == j

        # Background-fill: median of border pixels (numpy-only erosion)
        inner = vmask.copy()
        inner[0, :] = inner[-1, :] = inner[:, 0] = inner[:, -1] = False
        inner[1:, :] &= vmask[:-1, :]
        inner[:-1, :] &= vmask[1:, :]
        inner[:, 1:] &= vmask[:, :-1]
        inner[:, :-1] &= vmask[:, 1:]
        border_mask = vmask & ~inner
        border_vals = roi[border_mask]
        bg = float(np.median(border_vals)) if border_vals.size > 0 else 0.0
        roi_filled = np.where(vmask, roi, bg)

        rois_list.append(roi_filled)
        masks_list.append(vmask)
        origins.append((x0, y0))

    # Pad to uniform size and stack
    hs = [r.shape[0] for r in rois_list]
    ws = [r.shape[1] for r in rois_list]
    Hm, Wm = max(hs), max(ws)
    rois_stack = np.zeros((len(rois_list), Hm, Wm), dtype=np.float64)
    masks_stack = np.zeros((len(rois_list), Hm, Wm), dtype=bool)
    for j, (roi, vmask) in enumerate(zip(rois_list, masks_list)):
        h_r, w_r = roi.shape
        rois_stack[j, :h_r, :w_r] = roi
        masks_stack[j, :h_r, :w_r] = vmask

    return rois_stack, masks_stack, origins
```

New:
```python
    H, W = gray.shape
    rois_list = []
    masks_list = []
    bg_list = []
    origins = []

    for j, (x, y, w, h, cx, cy) in enumerate(rows):
        x0 = max(0, int(x) - int(pad))
        y0 = max(0, int(y) - int(pad))
        x1 = min(W, int(x + w) + int(pad))
        y1 = min(H, int(y + h) + int(pad))
        roi = gray[y0:y1, x0:x1].astype(np.float64)
        vmask = voronoi_labels[y0:y1, x0:x1] == j

        # Background-fill: 10th percentile of border pixels (numpy-only erosion)
        inner = vmask.copy()
        inner[0, :] = inner[-1, :] = inner[:, 0] = inner[:, -1] = False
        inner[1:, :] &= vmask[:-1, :]
        inner[:-1, :] &= vmask[1:, :]
        inner[:, 1:] &= vmask[:, :-1]
        inner[:, :-1] &= vmask[:, 1:]
        border_mask = vmask & ~inner
        border_vals = roi[border_mask]
        bg = float(np.percentile(border_vals, 10)) if border_vals.size > 0 else 0.0
        roi_filled = np.where(vmask, roi, bg)

        rois_list.append(roi_filled)
        masks_list.append(vmask)
        bg_list.append(bg)
        origins.append((x0, y0))

    # Pad to uniform size and stack — fill padding with per-ROI background
    hs = [r.shape[0] for r in rois_list]
    ws = [r.shape[1] for r in rois_list]
    Hm, Wm = max(hs), max(ws)
    rois_stack = np.zeros((len(rois_list), Hm, Wm), dtype=np.float64)
    masks_stack = np.zeros((len(rois_list), Hm, Wm), dtype=bool)
    for j, (roi, vmask, bg) in enumerate(zip(rois_list, masks_list, bg_list)):
        h_r, w_r = roi.shape
        rois_stack[j, :, :] = bg
        rois_stack[j, :h_r, :w_r] = roi
        masks_stack[j, :h_r, :w_r] = vmask

    return rois_stack, masks_stack, origins
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_radial_symmetry.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add subpx/_gpu/centers.py
git commit -m "fix: use 10th percentile background and fill padding with bg value"
```

---

### Task 5: Add Isophote Curvature In-Bounds Gate (Fix F)

**Files:**
- Modify: `subpx/_gpu/centers.py:1363-1381`

- [ ] **Step 1: Add in-bounds check to isophote curvature coordinate transform**

In `_isophote_curvature_batch_gpu`, replace lines 1363-1381:

Old:
```python
    centers = np.empty((N, 2), dtype=np.float64)
    for i in range(N):
        w_up_i = float(up_ws[i])
        h_up_i = float(up_hs[i])
        w_orig_i = float(orig_ws[i])
        h_orig_i = float(orig_hs[i])

        # xc is relative to center of upsampled grid: xc_pix_up = xc + (w_up - 1) / 2
        xc_pix_up = xc_cpu[i] + (w_up_i - 1.0) / 2.0
        yc_pix_up = yc_cpu[i] + (h_up_i - 1.0) / 2.0

        if factor > 1 and w_up_i > 1 and h_up_i > 1:
            xc_orig = xc_pix_up * (w_orig_i - 1.0) / (w_up_i - 1.0)
            yc_orig = yc_pix_up * (h_orig_i - 1.0) / (h_up_i - 1.0)
        else:
            xc_orig = xc_pix_up
            yc_orig = yc_pix_up

        centers[i, 0] = xc_orig
        centers[i, 1] = yc_orig

    return centers, spread_cpu
```

New:
```python
    centers = np.empty((N, 2), dtype=np.float64)
    for i in range(N):
        w_up_i = float(up_ws[i])
        h_up_i = float(up_hs[i])
        w_orig_i = float(orig_ws[i])
        h_orig_i = float(orig_hs[i])

        # xc is relative to center of upsampled grid: xc_pix_up = xc + (w_up - 1) / 2
        xc_pix_up = xc_cpu[i] + (w_up_i - 1.0) / 2.0
        yc_pix_up = yc_cpu[i] + (h_up_i - 1.0) / 2.0

        if factor > 1 and w_up_i > 1 and h_up_i > 1:
            xc_orig = xc_pix_up * (w_orig_i - 1.0) / (w_up_i - 1.0)
            yc_orig = yc_pix_up * (h_orig_i - 1.0) / (h_up_i - 1.0)
        else:
            xc_orig = xc_pix_up
            yc_orig = yc_pix_up

        # Quality gate: in-bounds check
        if not (0 <= xc_orig < w_orig_i and 0 <= yc_orig < h_orig_i):
            xc_orig = np.nan
            yc_orig = np.nan
            spread_cpu[i] = np.inf

        centers[i, 0] = xc_orig
        centers[i, 1] = yc_orig

    return centers, spread_cpu
```

- [ ] **Step 2: Run isophote curvature tests**

Run: `pytest tests/test_isophote_curvature.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add subpx/_gpu/centers.py
git commit -m "fix: add in-bounds quality gate to isophote curvature"
```

---

### Task 6: Add Non-Uniform Voronoi Test (Fix G1)

**Files:**
- Modify: `tests/test_voronoi.py` (append new test)

- [ ] **Step 1: Write the test**

Append to `tests/test_voronoi.py`:

```python
@skipno_gpu
def test_voronoi_grid_matches_brute_clustered():
    """Grid-accelerated kernel matches brute-force for clustered (non-uniform) seeds.

    This is a regression test for the adaptive search_radius fix.
    With a fixed search_radius=2, clustered seeds produce wrong labels
    because the max nearest-neighbor distance exceeds 2*cell_size.
    """
    from subpx._gpu.voronoi import compute_voronoi_labels_gpu
    import subpx._gpu.voronoi as voronoi_mod

    rng = np.random.default_rng(99)
    N = 400  # above _GRID_THRESHOLD=256
    H, W = 200, 300
    # Cluster seeds in the top-left quadrant — large empty region in bottom-right
    seeds = rng.uniform(0, [W / 2, H / 2], size=(N, 2)).astype(np.float32)

    # Grid-accelerated (N=400 > threshold)
    grid_labels = compute_voronoi_labels_gpu(seeds, (H, W), device=0)

    # Force brute-force by raising threshold temporarily
    old = voronoi_mod._GRID_THRESHOLD
    voronoi_mod._GRID_THRESHOLD = 10_000
    try:
        brute_labels = compute_voronoi_labels_gpu(seeds, (H, W), device=0)
    finally:
        voronoi_mod._GRID_THRESHOLD = old

    assert np.array_equal(grid_labels, brute_labels), (
        f"Grid and brute-force labels differ for clustered seeds: "
        f"{(grid_labels != brute_labels).sum()} pixels disagree"
    )
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_voronoi.py::test_voronoi_grid_matches_brute_clustered -v`
Expected: PASS (after Task 1 fix)

- [ ] **Step 3: Commit**

```bash
git add tests/test_voronoi.py
git commit -m "test: add non-uniform seed test for Voronoi grid correctness"
```

---

### Task 7: Add Coordinate Roundtrip Accuracy Test (Fix G2)

**Files:**
- Modify: `tests/test_radial_symmetry.py` (append new test)

- [ ] **Step 1: Write the test**

Append to `tests/test_radial_symmetry.py`:

```python
@skipno_gpu
def test_radial_symmetry_roundtrip_accuracy():
    """Full pipeline roundtrip: known blob positions → detect_centers → verify accuracy.

    Places a 3x3 grid of Gaussian blobs at known subpixel positions in a full image,
    runs the complete detect_centers pipeline with radial_symmetry, and verifies
    the returned positions match within 0.15 px.
    """
    from subpx.centers import detect_centers
    from scipy.spatial import cKDTree

    sigma = 1.8
    spacing = 12.0
    img = np.zeros((80, 80), dtype=np.float64)
    yy, xx = np.mgrid[:80, :80]
    true_centers = []
    for row in range(3):
        for col in range(3):
            cx = 20.3 + col * spacing
            cy = 20.7 + row * spacing
            img += 180.0 * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
            true_centers.append([cx, cy])
    true_centers = np.array(true_centers)
    img = np.clip(img, 0, 255).astype(np.uint8)

    res = detect_centers(
        img, backend="gpu", refine="radial_symmetry",
        area_min=1, area_max=300, upsample_factor=4, threshold="triangle",
    )

    assert res.centers_xy.shape[0] == 9, (
        f"Expected 9 centers, got {res.centers_xy.shape[0]}"
    )

    tree = cKDTree(true_centers)
    dists, _ = tree.query(res.centers_xy)
    assert np.all(dists < 0.15), (
        f"Roundtrip accuracy failed: max error {dists.max():.4f} px, "
        f"errors: {dists}"
    )
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_radial_symmetry.py::test_radial_symmetry_roundtrip_accuracy -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_radial_symmetry.py
git commit -m "test: add coordinate roundtrip accuracy test for radial symmetry"
```

---

### Task 8: Run Full Test Suite

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Final commit if any fixups needed**

If any tests fail, fix and commit. Otherwise, no action needed.
