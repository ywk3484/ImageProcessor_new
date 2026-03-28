# Radial Symmetry Audit Fixes — Design Spec

**Date**: 2026-03-28
**Status**: Draft

## Problem Statement

An adversarial code review of the radial symmetry center detection pipeline identified 8 issues across correctness, quality filtering, numerical robustness, and test coverage. This spec addresses all of them.

## Issues and Fixes

### Fix A: Adaptive Voronoi Grid Search Radius (Critical — C1)

**File**: `subpx/_gpu/voronoi.py`, function `compute_voronoi_labels_gpu`

**Problem**: The grid-accelerated nearest-neighbor kernel uses `cell_size = max(16, int(sqrt(HW/N)))` with a fixed `search_radius = 2`. This assumes uniform seed distribution. When area filtering removes some connected components, gaps appear in the seed array. A pixel in a gap can have its nearest seed more than `2 * cell_size` away, producing wrong Voronoi labels.

**Fix**: After binning seeds to the grid, compute the actual maximum nearest-neighbor distance across all seeds. Derive `search_radius` dynamically:

```python
# Compute max nearest-neighbor distance among seeds (O(N^2) on GPU, N is small)
dists = cp.linalg.norm(seeds_gpu[:, None, :] - seeds_gpu[None, :, :], axis=2)
cp.fill_diagonal(dists, cp.inf)
max_nn_dist = float(cp.min(dists, axis=1).max())

search_radius = int(np.ceil(max_nn_dist / cell_size)) + 1
```

If `search_radius > 5` (11x11 cell search — very sparse/clustered seeds), fall back to brute-force since it will be faster. This guarantees correctness regardless of seed distribution.

### Fix B: Residual Quality Gate (Important — I1)

**File**: `subpx/_gpu/centers.py`, function `_radial_symmetry_batch_gpu`

**Problem**: The design spec requires residual-based rejection, but the implementation only does in-bounds filtering. The residual is computed and returned but never used for rejection.

**Fix**: Add `residual_max: float | None = None` parameter. When not None, mark centers as NaN where `residual > residual_max`:

```python
if residual_max is not None:
    bad_residual = residual_cpu > residual_max
    centers[bad_residual] = np.nan
    residual_cpu[bad_residual] = np.inf
```

The `detect_centers_gpu` caller passes this through without imposing a default. The parameter is exposed but opt-in — users can filter by the returned residual array or set the threshold explicitly.

### Fix C: Background Estimation Bias (Important — I3)

**File**: `subpx/_gpu/centers.py`, function `_extract_voronoi_rois`

**Problem**: `np.median(border_vals)` includes blob signal when the Voronoi cell border is close to the blob center. This biases the background estimate high, creating larger intensity cliffs at cell boundaries.

**Fix**: Change from median to 10th percentile:

```python
bg = float(np.percentile(border_vals, 10)) if border_vals.size > 0 else 0.0
```

For blobs brighter than background, the 10th percentile better approximates the true background. For dark blobs on bright background (inverted images), the 10th percentile is conservative — it slightly underestimates background, which creates a smaller intensity cliff.

### Fix D: Degenerate Determinant Handling (Important — I4)

**File**: `subpx/_gpu/centers.py`, function `_radial_symmetry_batch_gpu`

**Problem**: When `det ≈ 0`, clamping to `1e-30` produces coordinates ~`1e30`. These are eventually rejected by in-bounds, but the intermediate large values are numerically unclean.

**Fix**: Set xc, yc to NaN directly when det is degenerate:

```python
bad_det = cp.abs(det) < 1e-30
det_safe = cp.where(bad_det, cp.float64(1.0), det)  # safe for division
xc = cp.where(bad_det, cp.nan, (-smbw * sw + smw * sbw) / det_safe)
yc = cp.where(bad_det, cp.nan, (smmw * sbw - smbw * smw) / det_safe)
```

### Fix E: Zero-Pad Background Fill (Suggestion — S4)

**File**: `subpx/_gpu/centers.py`, function `_extract_voronoi_rois`

**Problem**: ROIs padded to uniform `(Hmax, Wmax)` use zero-fill. After bicubic upsampling, the zero-to-signal transition creates gradient cliffs that extend into the valid region.

**Fix**: Initialize padded stack with per-ROI background value:

```python
rois_stack[j, :, :] = bg        # fill entire slice with background
rois_stack[j, :h_r, :w_r] = roi # overlay actual ROI
```

### Fix F: Isophote Curvature In-Bounds Gate (Suggestion — S6)

**File**: `subpx/_gpu/centers.py`, function `_isophote_curvature_batch_gpu`

**Problem**: Unlike `_radial_symmetry_batch_gpu`, the isophote curvature method has no in-bounds check. Biased fits can produce out-of-bounds centers that pass the `np.isfinite()` check downstream.

**Fix**: Add the same in-bounds check after coordinate transform (identical pattern to radial symmetry lines 1183-1187):

```python
if not (0 <= xc_orig < w_orig_i and 0 <= yc_orig < h_orig_i):
    xc_orig = np.nan
    yc_orig = np.nan
    spread_cpu[i] = np.inf
```

### Fix G: Test Coverage (Suggestions — S2, S3)

**File**: `tests/test_voronoi.py` and `tests/test_radial_symmetry.py`

**G1 — Non-uniform Voronoi test**: Create seeds clustered in one region of the image. Verify the grid-accelerated path produces identical labels to brute-force. This directly validates Fix A.

```python
def test_voronoi_grid_matches_brute_clustered():
    # Seeds clustered in top-left quadrant, N > 256
    rng = np.random.default_rng(99)
    N = 400
    H, W = 200, 300
    seeds = rng.uniform(0, [W/2, H/2], size=(N, 2)).astype(np.float32)
    # Grid labels should match brute-force
```

**G2 — Coordinate roundtrip test**: Place Gaussian blobs at known subpixel positions, run full `detect_centers(..., refine="radial_symmetry")`, verify returned positions match within 0.15 px.

```python
def test_radial_symmetry_roundtrip_accuracy():
    # 3x3 grid, known subpixel positions
    # Run detect_centers, match to true positions, check error < 0.15 px
```

## Files Modified

| File | Changes |
|------|---------|
| `subpx/_gpu/voronoi.py` | Fix A: adaptive search_radius, brute-force fallback |
| `subpx/_gpu/centers.py` | Fixes B, C, D, E, F in respective functions |
| `tests/test_voronoi.py` | Fix G1: non-uniform seed test |
| `tests/test_radial_symmetry.py` | Fix G2: coordinate roundtrip test |

## Testing Strategy

1. Run existing test suite — all tests must pass (fixes are behavioral improvements, not breaking changes)
2. New tests G1 and G2 must pass
3. Fix A is verified by G1 (grid matches brute-force for clustered seeds)
4. Fixes B-F are verified by existing tests (they add strictness, not change behavior for good fits)
