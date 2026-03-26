# Radial Symmetry Accuracy Fix — Design Spec

**Date**: 2026-03-26
**Status**: Draft

## Problem Statement

The radial symmetry center detection pipeline (`refine="radial_symmetry"`) produces inaccurate centers due to three cascading failures:

1. **Otsu threshold too high** — For images where small blobs (3-6px) occupy a tiny fraction of total pixels, Otsu's global threshold is pulled too high by the dominant background peak. Many blobs are missed in the binary segmentation step.

2. **Incomplete Voronoi seeds** — The missed blobs mean fewer connected components, so fewer Voronoi seeds. Voronoi cells expand to cover the territory of multiple actual features, making the mask useless within the padded ROI (it's all-True).

3. **No post-hoc quality filtering** — Unlike logquad (which rejects fits via in-bounds + negative-definite checks), radial symmetry's WLS solve always produces a result. The only filter is `np.isfinite()`, which almost never triggers. Edge-adjacent and degenerate fits pass through unchecked.

## Root Cause Analysis

### Thresholding (Primary)

`_segment_binary()` (`subpx/_cpu/centers.py:247`) uses only `cv2.THRESH_OTSU`. For the photomask blob use case:
- Background pixels vastly outnumber foreground pixels
- Otsu optimizes total pixel classification accuracy, favoring the background class
- Threshold ends up too high, clipping faint blobs
- Result: incomplete connected components → incomplete Voronoi seeds → incorrect partitioning

### Quality Gates (Secondary)

`_radial_symmetry_batch_gpu()` (`subpx/_gpu/centers.py:974`) computes a residual but never uses it for filtering. The fitted center has no bounds check. Compare to logquad at line 107-109:

```python
negdef = (a < -1e-12) & (b < -1e-12) & (detH > 1e-12)
in_bounds = (cx >= 0) & (cy >= 0) & (cx < ws) & (cy < hs)
ok = (counts >= 6) & negdef & in_bounds & isfinite(cx) & isfinite(cy)
```

Radial symmetry has none of these gates, so edge-adjacent blobs with clipped/asymmetric ROIs produce biased fits that are not rejected.

## Design

### Fix 1: Add Triangle Threshold Support

**File**: `subpx/_cpu/centers.py`
**Function**: `_segment_binary()`

Add `"triangle"` as a supported value for the `threshold` parameter:
- Map `"triangle"` to `cv2.THRESH_TRIANGLE` (built into OpenCV)
- Keep `"otsu"` as the default for backward compatibility
- Update the validation check at line 252-253 to accept both values

Triangle thresholding (Zack's method) finds the histogram "elbow" — the point of maximum distance from a line connecting the histogram peak to the tail. Designed for unimodal histograms with a dominant peak and a small tail, which is precisely the photomask blob scenario.

The GPU path (`detect_centers_gpu`) imports `_segment_binary` from the CPU module, so this change propagates automatically to both backends.

### Fix 2: Post-hoc Quality Gates for Radial Symmetry

**File**: `subpx/_gpu/centers.py`
**Function**: `_radial_symmetry_batch_gpu()`

Add two quality gates after the WLS solve (step 13, coordinate transform), applied per-ROI:

**Gate 1 — In-bounds**: After converting to original ROI coordinates, verify the fitted center is within the ROI:
```
valid = (xc_orig >= 0) & (xc_orig < w_orig) & (yc_orig >= 0) & (yc_orig < h_orig)
```
Mark out-of-bounds fits as NaN. This mirrors logquad's `in_bounds` check.

**Gate 2 — Residual threshold**: The WLS residual (step 12) measures the weighted mean perpendicular distance squared from the fitted center to the symmetry lines. High residual indicates a poor fit (e.g., asymmetric ROI, edge contamination, or a non-radially-symmetric feature). Reject fits where the residual exceeds a threshold by marking them as NaN.

The residual threshold value should be determined empirically from the test suite. A conservative starting point: reject residuals above a fixed value that allows all current test cases to pass.

**Integration**: In `detect_centers_gpu()`, the existing `np.isfinite(gx) and np.isfinite(gy)` check at line 1535 already handles NaN rejection, so no changes are needed at the output stage.

### Fix 3: Validation

Implemented as test functions (not production code):

1. **Triangle vs Otsu blob count**: Create a test image with closely-packed blobs. Verify Triangle detects more blobs than Otsu.

2. **Voronoi correctness**: After Triangle thresholding with complete seeds, verify each Voronoi cell contains exactly one seed center. If this test fails, the Voronoi kernel has a bug independent of the threshold.

3. **Edge rejection**: Create blobs at various distances from the image boundary. Verify the quality gates reject edge-adjacent fits.

4. **Regression**: All existing `test_radial_symmetry.py` tests must pass with the quality gates in place (gates should not reject good fits).

## Files Changed

| File | Change |
|---|---|
| `subpx/_cpu/centers.py` | Add `"triangle"` support to `_segment_binary()` |
| `subpx/_gpu/centers.py` | Add in-bounds + residual quality gates to `_radial_symmetry_batch_gpu()` |
| `tests/test_radial_symmetry.py` | Add validation tests for triangle threshold, edge rejection, Voronoi correctness |

## Non-Goals

- Replacing the Voronoi kernel algorithm (should self-correct with complete seeds)
- Changing radial symmetry's WLS solve algorithm
- Adding new refinement methods
- Modifying the ROI extraction logic

## Risk

- Triangle threshold may be too aggressive (threshold too low), capturing noise as blobs. Mitigated by the existing `area_min` / `area_max` filters.
- Residual threshold requires empirical calibration. Starting conservative (high threshold) and tightening based on validation results.
