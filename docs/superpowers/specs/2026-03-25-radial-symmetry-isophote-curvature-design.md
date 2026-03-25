# Design: Radial Symmetry & Isophote Curvature Center Estimation

**Date**: 2026-03-25
**Status**: Draft
**Scope**: Two new GPU-accelerated refinement methods for small, closely-packed PSF-like blobs, with Voronoi geometric partitioning.

---

## 1. Problem Statement

### What is happening

The current center detection pipeline (`detect_centers`) uses connected-component bounding boxes as ROIs for refinement. Each blob's ROI is its component bbox, and the refinement methods (weighted centroid, log-quadratic fit) operate on all pixels within that bbox.

When blobs are closely packed, adjacent blob PSF wings leak into each other's ROIs. This contaminates:
- **Background estimation**: border pixels of the ROI may belong to a neighbor, biasing the background level.
- **Weighted centroid**: intensity from a neighbor pulls the centroid toward that neighbor.
- **Log-quadratic fit**: the quadratic surface warps to accommodate neighbor intensity, shifting the fitted peak.

### Why it matters

For photomask metrology, center positions must be accurate to subpixel precision. Neighbor contamination introduces systematic bias that scales with pattern density — exactly the regime where accuracy matters most.

### What we are building

Two new refinement methods that address this through **geometric partitioning** (Voronoi) and **geometry-first center estimation** (radial symmetry, isophote curvature), plus bicubic upsampling to handle very small blobs (3-6 px).

---

## 2. Pipeline Reasoning

This section explains *why* each pipeline stage exists and what problem it solves.

### 2.1 Why Voronoi Partitioning

**Problem**: In a closely-packed array, the bounding box of one blob overlaps with neighboring blobs. Any method that uses the full bbox will see contaminating signal.

**Solution**: Assign each pixel to exactly one blob using nearest-seed Voronoi partitioning. Each blob's refinement domain is its Voronoi cell — the set of all pixels closer to that blob's seed than to any other seed.

**Why Voronoi and not pitch-based midlines**: The user's patterns are staggered arrays. Axis-aligned midlines (which assume rectangular grid layout) would cut through cells in a staggered arrangement. Voronoi partitioning is agnostic to array geometry — it works for regular grids, staggered arrays, and irregular arrangements.

**Why not sector masking**: Sector masking (excluding angular directions facing neighbors) was considered but rejected as unnecessary complexity. Voronoi partitioning already excludes neighbor pixels geometrically. Sector masking could be added later if Voronoi proves insufficient for extremely tight packing.

**What the Voronoi cell guarantees**: Every pixel within a blob's cell is closer to that blob than to any neighbor. For a roughly periodic array, the cell boundary sits approximately at the midpoint between adjacent blob centers — exactly where the contamination transition occurs.

### 2.2 Why Radial Symmetry (Parthasarathy Method)

**Problem**: Existing methods (weighted centroid, log-quadratic) estimate the center from **intensity amplitudes**. This makes them sensitive to:
- Absolute background level
- Illumination gradients
- Any intensity asymmetry (including residual neighbor contamination at cell boundaries)

**Solution**: The Parthasarathy radial symmetry method estimates the center from **gradient directions**, not intensity values. For a radially symmetric object, every gradient vector points radially away from (or toward) the center. The center is where these gradient normals intersect in a least-squares sense.

**Why gradient geometry is more robust**:
- A multiplicative illumination gradient changes gradient magnitudes but not directions (to first order).
- An additive background shifts all intensities but doesn't change gradient directions (gradient of a constant is zero).
- The weighting scheme (|grad|^2 / distance-to-centroid) naturally concentrates weight on the steepest part of the PSF slope — the mid-ring — which is the most informative region for direction estimation and the least affected by neighbor overlap.

**Why not Gaussian fitting**: Gaussian fitting achieves near-optimal accuracy for isolated blobs, but it assumes a specific functional form. The radial symmetry method works for *any* radially symmetric profile — Gaussian, Airy, or experimentally measured PSFs. It is also non-iterative (closed-form solve), which eliminates convergence issues and is ideal for GPU batch processing.

**The closed-form solve**: The method reduces to a weighted linear system with 5 scalar sums and a 2×2 determinant. No iteration, no initial guess, no convergence criteria. Every blob gets an answer in exactly the same number of operations — perfect for SIMD/GPU execution.

### 2.3 Why Bicubic 4x Upsampling

**Problem**: For a 3px blob, the Parthasarathy gradient array is 2×2 = 4 midpoints. This is the absolute minimum for a 2D least-squares solve (4 equations, 2 unknowns). There is essentially no redundancy — a single noisy gradient dominates the result.

**What upsampling does**: Bicubic interpolation fits a piecewise cubic polynomial to the discrete pixel values and evaluates it on a finer grid. This does NOT add new frequency content or fabricate information. What it does is provide a **smooth, differentiable surface** whose gradients are mathematically well-defined at arbitrary subpixel positions.

Computing finite-difference gradients on the upsampled grid is equivalent to evaluating the gradient of the interpolating polynomial at those positions. This gives:
- More gradient vectors for the least-squares fit (see table below)
- Better-conditioned least-squares problem
- Smoother gradient field (fewer noise spikes from single-pixel differences)

**Data point counts** (gradient midpoints = (H_up-1) × (W_up-1)):

| Original blob | Upsampled size (4×) | Gradient midpoints | Without upsampling |
|---------------|--------------------|--------------------|-------------------|
| 3×3 px | 12×12 | 11×11 = 121 | 2×2 = 4 |
| 4×4 px | 16×16 | 15×15 = 225 | 3×3 = 9 |
| 5×5 px | 20×20 | 19×19 = 361 | 4×4 = 16 |
| 6×6 px | 24×24 | 23×23 = 529 | 5×5 = 25 |

**Why 4× fixed factor**:
- At 3px, 4× gives 121 gradient midpoints — well-conditioned.
- At 6px, 4× gives 529 midpoints — comfortably over-determined.
- Fixed factor avoids edge cases where blobs near a size boundary get different treatment, which could introduce systematic position-dependent bias.
- The computational cost is negligible on GPU (tiny ROIs, batch processing).

**Why bicubic specifically**:
- Bilinear interpolation is C0 (discontinuous gradients) — defeats the purpose.
- Bicubic is C1 (continuous first derivatives) — the minimum order that gives meaningful gradients.
- Lanczos-3 kernel support (6px) nearly spans a 3px ROI — problematic.
- Sinc/FFT has severe truncation ringing on 3px ROIs.

**Caveat**: Upsampled gradient values are correlated (derived from the same original pixels). The 121 midpoints from a 4× upsampled 3px blob represent perhaps 8-12 effective degrees of freedom, not 121 independent measurements. But this is still a large improvement over 4 raw gradient values.

**Future improvement**: Replace upsampling + discrete gradients with analytic gradient computation from bicubic polynomial coefficients. This is mathematically equivalent to infinite upsampling with zero discretization error. Deferred for now as it's more complex to implement on GPU.

### 2.4 Why Isophote Curvature as a Diagnostic

**Problem**: Any single center estimation method can produce a number without indicating whether that number is trustworthy. A contaminated or asymmetric blob will still yield *some* center — but it may be biased.

**Solution**: The isophote curvature method provides an **independent cross-check** using fundamentally different mathematics. If the two methods agree, confidence is high. If they disagree, it flags the blob for inspection.

**How it works**: At each pixel, the local intensity isophotes (level curves) have a center of curvature. For a radially symmetric blob, all isophote curvature centers coincide at the blob center. Each pixel "votes" for the center position using its gradient direction and local curvature radius. The vote is weighted by curvedness (how strongly curved the isophotes are at that pixel).

**Why this is complementary to radial symmetry**:
- Radial symmetry uses only **first derivatives** (gradient directions).
- Isophote curvature uses **first and second derivatives** (gradient direction + curvature).
- They fail differently: radial symmetry degrades when gradients are noisy; isophote curvature degrades when second derivatives are noisy.
- Agreement between methods is a strong quality signal.

**The curvature-consistency spread metric**: The standard deviation of per-pixel center votes quantifies how "well-behaved" the blob is. A small spread means all pixels agree on the center (good symmetry). A large spread indicates contamination, asymmetry, or insufficient sampling.

**Reliability note**: Isophote curvature computes second derivatives of the interpolated surface. Bicubic interpolation is C1 (continuous first derivatives) but C0 for second derivatives — second derivatives have discontinuities at bicubic knot points (every 4 upsampled pixels for 4× upsampling). For blobs < 5px original, the `isophote_curvature_spread` metric will be elevated and should be interpreted with caution. A light Gaussian pre-smoothing (sigma=0.5 upsampled pixels) of the upsampled ROI before computing second derivatives is recommended to suppress these artifacts.

**Why not multi-level contour fitting**: The original proposal was to extract iso-contours at multiple intensity levels via marching squares and fit circles to each. This is GPU-hostile (irregular output lengths) and fragile for small blobs (too few contour points at extreme levels). The isophote curvature method captures the same insight — inner intensity regions are more reliable — using only local derivatives, which map cleanly to GPU batch operations.

---

## 3. Algorithm Specifications

### 3.1 Voronoi Partitioning

**Input**:
- `seeds_xy`: (N, 2) float array — seed center positions [x, y]
- `image_shape`: (H, W) — image dimensions

**Output**:
- `label_map`: (H, W) int32 array — pixel `(r, c)` is assigned to `argmin_i ||[c, r] - seeds_xy[i]||²`

**GPU Implementation — Grid-Accelerated Nearest Neighbor**:

This is the only implementation approach (the naive broadcast approach requires O(H×W×N) memory, which is prohibitive for typical images).

1. **Compute grid cell size**: `cell_size = median_nearest_neighbor_distance(seeds_xy) * 1.5`. The 1.5 factor ensures that a seed's Voronoi cell is fully covered by its own grid cell plus adjacent cells. For staggered arrays with different pitch along x/y, a single isotropic cell size is simpler and sufficient (the 1.5 factor provides adequate margin).

2. **Assign seeds to grid cells**: Hash each seed to `(floor(x / cell_size), floor(y / cell_size))`. Store as a flat array with a cell-to-seed index mapping.

3. **For each pixel**, check seeds in the same + 8 adjacent grid cells (9 cells total). Compute Euclidean distance to each candidate seed. Assign pixel to nearest.

4. **Compute cell areas**: `voronoi_cell_area = cp.bincount(label_map.ravel(), minlength=N)` — used as a diagnostic for crowdedness.

Complexity: O(H × W × k) where k is the average candidates per pixel (typically 4-9 for a regular array).

**Visualization function** (in `subpx/visualization.py`):
```python
def draw_voronoi_boundaries(image, centers_xy, label_map=None, color=1.0, thickness=1):
    """Overlay Voronoi cell boundaries on an image.

    Boundaries are pixels where the label differs from at least one
    4-connected neighbor.

    Parameters
    ----------
    image : (H, W) array
        Background image.
    centers_xy : (N, 2) array
        Seed positions.
    label_map : (H, W) int array, optional
        Pre-computed Voronoi labels. Computed if not provided.
    color : float or tuple
        Boundary color/intensity.
    thickness : int
        Boundary line width (dilation iterations).

    Returns
    -------
    overlay : (H, W) or (H, W, 3) array
        Image with boundaries drawn.
    """
```

### 3.2 Parthasarathy Radial Symmetry

**Reference**: Parthasarathy, R. "Rapid, accurate particle tracking by calculation of radial symmetry centers." *Nature Methods* 9, 724-726 (2012).

**Input** (per blob, after Voronoi masking + upsampling):
- `roi`: (H_up, W_up) float64 array — upsampled ROI

**Algorithm**:

1. **Diagonal gradients at midpoints** (computed on (H_up-1, W_up-1) grid):
   ```
   dIdu[i,j] = I[i, j+1] - I[i+1, j]      # upper-right minus lower-left
   dIdv[i,j] = I[i, j]   - I[i+1, j+1]     # upper-left minus lower-right
   ```
   These are gradients along 45°-rotated axes, evaluated at midpoints between pixels.

2. **Gradient magnitude**:
   ```
   grad_mag² = dIdu² + dIdv²
   ```

3. **Gradient slope in (x, y) frame** (45° rotation from (u, v)):
   ```
   m = -(dIdv + dIdu) / (dIdu - dIdv)
   ```
   When `|dIdu - dIdv| < epsilon` (near-vertical gradient), clamp `m` to `sign(m) * 1e9` and keep weight nonzero (the line is still valid, just near-vertical). When `grad_mag² < epsilon²` (flat region), set weight to 0 — no directional information.

4. **Midpoint coordinates** (centered on ROI center):
   ```
   xm = -(W_up-1)/2 + 0.5, ..., (W_up-1)/2 - 0.5    # shape (W_up-1,)
   ym = -(H_up-1)/2 + 0.5, ..., (H_up-1)/2 - 0.5     # shape (H_up-1,)
   ```

5. **Line intercepts**:
   ```
   b = ym - m * xm
   ```

6. **Weights** — gradient magnitude squared × inverse distance to gradient-weighted centroid:
   ```
   x_gc = sum(grad_mag² * xm) / sum(grad_mag²)
   y_gc = sum(grad_mag² * ym) / sum(grad_mag²)
   w = grad_mag² / sqrt((xm - x_gc)² + (ym - y_gc)²)
   ```

7. **Boundary gradient masking**: After computing weights, set `w = 0` for gradient midpoints within `margin` upsampled pixels of the Voronoi cell boundary (see Section 3.4 for details). Default `margin = upsample_factor` (i.e., 1 original pixel worth of margin). This suppresses spurious gradients from the background-fill transition at cell edges.

8. **Analytic least-squares solve** — minimize weighted perpendicular distance to all lines `y = mx + b`:
   ```
   w' = w / (m² + 1)

   sw   = Σ w'
   smmw = Σ m² w'
   smw  = Σ m w'
   smbw = Σ m b w'
   sbw  = Σ b w'

   det = smw² - smmw * sw

   xc = (smbw * sw  - smw * sbw) / det
   yc = (smbw * smw - smmw * sbw) / det
   ```

9. **Goodness-of-fit metric**:
   ```
   d² = (b - (yc - m * xc))² / (m² + 1)
   residual = Σ(d² * grad_mag²) / Σ(grad_mag²)
   ```

10. **Coordinate transform** (see Section 3.5 for derivation):
    ```
    x_image = xc * scale + roi_center_x
    y_image = yc * scale + roi_center_y
    ```
    where `scale` and `roi_center` depend on the `zoom` coordinate convention used.

**GPU batch processing**:
- Stack all ROIs into (N_blobs, H_max, W_max) float64, zero-pad smaller ROIs
- Create corresponding Voronoi mask stack (N_blobs, H_max, W_max) bool
- Upsample batch: `zoom` each ROI
- All gradient, slope, weight, and sum operations are element-wise or reductions over spatial dims
- The 5 scalar sums and 2×2 solve are per-blob (vectorized over batch dim)
- All intermediate computations in float64 for subpixel precision

### 3.3 Isophote Curvature Center

**Reference**: Based on isophote curvature analysis (Valenti & Gevers, CVPR 2008), adapted for subpixel blob localization.

**Input** (per blob, after Voronoi masking + upsampling):
- `roi`: (H_up, W_up) float64 array — upsampled ROI

**Algorithm**:

0. **Optional Gaussian pre-smoothing**: Apply Gaussian filter with sigma=0.5 upsampled pixels to suppress C0 second-derivative discontinuities at bicubic knot points. This is recommended for blobs < 5px original size. Configurable via parameter.

1. **First derivatives** via central finite differences on upsampled ROI:
   ```
   Ix = (I[r, c+1] - I[r, c-1]) / 2
   Iy = (I[r+1, c] - I[r-1, c]) / 2
   ```

2. **Second derivatives**:
   ```
   Ixx = I[r, c+1] - 2*I[r, c] + I[r, c-1]
   Iyy = I[r+1, c] - 2*I[r, c] + I[r-1, c]
   Ixy = (I[r+1, c+1] - I[r+1, c-1] - I[r-1, c+1] + I[r-1, c-1]) / 4
   ```

3. **Curvature denominator** (related to isophote curvature):
   ```
   denom = Iy² * Ixx - 2 * Ix * Ixy * Iy + Ix² * Iyy
   ```

4. **Displacement to curvature center** from each pixel (r, c):
   ```
   grad_sq = Ix² + Iy²
   Dx = -Ix * grad_sq / denom
   Dy = -Iy * grad_sq / denom
   ```

   Skip pixels where `|denom| < epsilon` (flat regions, inflection points).

5. **Curvedness weight**:
   ```
   curvedness = sqrt(Ixx² + 2*Ixy² + Iyy²)
   ```

6. **Voted center from each pixel**:
   ```
   vote_x = x_pixel + Dx
   vote_y = y_pixel + Dy
   ```

7. **Outlier rejection**: Reject votes where `sqrt(Dx² + Dy²) > ROI_size/2` (displacement larger than half the ROI — likely noise or boundary artifact). Also apply the boundary gradient mask (same margin as radial symmetry) to exclude votes from Voronoi cell edge pixels.

8. **Weighted average of votes**:
   ```
   xc = Σ(curvedness * vote_x) / Σ(curvedness)
   yc = Σ(curvedness * vote_y) / Σ(curvedness)
   ```

9. **Curvature-consistency spread** (quality metric):
   ```
   spread = sqrt(Σ(curvedness * ((vote_x - xc)² + (vote_y - yc)²)) / Σ(curvedness))
   ```

10. **Coordinate transform**: Same as radial symmetry (Section 3.5).

**GPU batch processing**: Same stacking pattern as radial symmetry. All operations are element-wise or reductions. The derivative computation uses standard stencil operations on 2D arrays. All computations in float64.

### 3.4 Voronoi Mask Handling and Boundary Gradient Suppression

**Problem**: Pixels outside the Voronoi cell must be handled before bicubic upsampling. Setting them to zero creates a sharp intensity cliff. This cliff, after bicubic interpolation, produces a gradient ramp extending `~2 upsampled pixels` into the cell interior — these are spurious gradients that point away from the cliff, not toward the blob center.

**Solution — two-layer defense**:

1. **Background fill before upsampling**: Set out-of-cell pixels to the local background estimate (median of cell-interior border pixels). This minimizes the intensity step at the cell boundary, reducing (but not eliminating) boundary gradient artifacts.

2. **Boundary gradient masking after upsampling**: After computing gradients on the upsampled ROI, set weight = 0 for gradient midpoints within `margin` upsampled pixels of the Voronoi cell boundary. Default: `margin = upsample_factor` (1 original pixel of margin). The boundary mask is computed by upsampling the Voronoi boolean mask (nearest-neighbor, NOT bicubic) and eroding it by `margin` pixels.

This two-layer approach means:
- The background fill reduces the gradient artifact magnitude
- The margin mask excludes any remaining artifacts from the least-squares solve
- For very tightly packed blobs (cell radius ~ 2-3 original pixels), this margin removes a significant fraction of gradient data. The residual metric will reflect this (fewer data points → noisier solve → higher residual).

### 3.5 Coordinate Transform Convention

**Critical precision detail**: The relationship between original and upsampled pixel coordinates depends on the `zoom` function's alignment convention.

**We use `cupyx.scipy.ndimage.zoom` with default settings** (center-aligned, `grid_mode=False`). In this mode, the centers of the first and last input pixels map to the centers of the first and last output pixels. The output pixel spacing in terms of input coordinates is:

```
spacing = (N_in - 1) / (N_out - 1)
```

where `N_out = N_in * upsample_factor`.

**The Parthasarathy algorithm works in ROI-centered coordinates** where the midpoint grid spans from `-(N_out-2)/2` to `+(N_out-2)/2` in steps of 1. The center result `(xc, yc)` is in these units (upsampled pixels, origin at ROI center).

**Transform to image coordinates**:

```python
# Upsampled ROI-centered coords → upsampled ROI pixel coords
xc_pixel_up = xc + (W_up - 1) / 2.0
yc_pixel_up = yc + (H_up - 1) / 2.0

# Upsampled pixel coords → original pixel coords (center-aligned zoom)
# output_pixel_i corresponds to input_pixel = i * (N_in - 1) / (N_out - 1)
xc_pixel_orig = xc_pixel_up * (W_orig - 1) / (W_up - 1)
yc_pixel_orig = yc_pixel_up * (H_orig - 1) / (H_up - 1)

# Original ROI pixel coords → image coords
x_image = xc_pixel_orig + roi_x0
y_image = yc_pixel_orig + roi_y0
```

where `roi_x0, roi_y0` are the top-left corner coordinates of the original ROI in image space.

**For 1× upsampling** (disabled), `W_up = W_orig` and the transform reduces to `x_image = xc + (W_orig-1)/2 + roi_x0`, which is the standard Parthasarathy coordinate conversion.

---

## 4. API Design

### 4.1 Public Interface Changes

```python
def detect_centers(
    image,
    *,
    refine="auto",           # existing + "radial_symmetry", "isophote_curvature"
    upsample_factor=4,       # NEW: bicubic upsampling factor (1 = disabled)
    backend="auto",
    # ... existing parameters unchanged ...
) -> CenterResult:
```

New `refine=` values:
- `"radial_symmetry"` — Parthasarathy method with Voronoi partitioning
- `"isophote_curvature"` — Isophote curvature diagnostic with Voronoi partitioning

The `upsample_factor` parameter applies to both new methods. It is ignored by existing methods (`"weighted"`, `"logquad"`, `"edge_gradmoment"`, `"edge_erf"`).

**`refine="auto"` behavior**: The auto-selection heuristic (`choose_refine_method_for_bbox`) is **unchanged** — it never selects the new methods. `"radial_symmetry"` and `"isophote_curvature"` are opt-in only. This preserves backward compatibility.

**CPU backend error handling**: When `backend="cpu"` (or `backend="auto"` with no GPU) and `refine` is `"radial_symmetry"` or `"isophote_curvature"`, `detect_centers` raises `NotImplementedError` with message: `"refine='{method}' requires GPU backend. Set backend='gpu' or install CuPy."`. This is checked early, before any processing begins.

### 4.2 Modified GPU Dispatch Flow

The existing GPU dispatch in `detect_centers_gpu()` processes blobs independently via a method-to-rows dictionary. The new methods require a **different dispatch flow** because Voronoi partitioning needs all seed positions simultaneously.

**New dispatch architecture** (inside `detect_centers_gpu`):

```
Phase 1: Segmentation (unchanged)
    Otsu threshold → connected components → bbox/stats extraction
    → coarse_centers_xy (N, 2)  [component centroids]

Phase 2: Method dispatch (modified)
    if refine in ("radial_symmetry", "isophote_curvature"):
        → NEW PATH: Voronoi-partitioned refinement
            1. Compute Voronoi label map from coarse_centers_xy
            2. Extract ROIs masked by Voronoi cells
            3. Background-fill out-of-cell pixels
            4. Pad to uniform size → stack (N, H_max, W_max)
            5. Bicubic 4× upsample each ROI
            6. Compute boundary gradient mask
            7. Batch-vectorized refinement (radial_symmetry or isophote_curvature)
            8. Coordinate transform to image space
            9. Compute quality metrics + cell areas

    else:
        → EXISTING PATH: per-blob independent refinement
            (weighted, logquad, edge_gradmoment, edge_erf — unchanged)

Phase 3: Result assembly (unchanged)
    → CenterResult(centers_xy, method, backend, meta)
```

The key architectural difference: the new path computes the Voronoi label map **once** from all coarse centers, then passes both the label map and per-blob ROIs to the batch refinement function. Existing methods never see the Voronoi map.

### 4.3 Modified Function Signatures

```python
# In subpx/_gpu/centers.py
def detect_centers_gpu(
    image,
    *,
    refine="logquad",
    upsample_factor=4,          # NEW
    # ... existing parameters ...
) -> CenterResult:

# NEW function in subpx/_gpu/centers.py
def _refine_voronoi_batch_gpu(
    image,                       # (H, W) full image on GPU
    labels,                      # (H, W) connected component labels
    stats,                       # component statistics
    coarse_centers_xy,           # (N, 2) from CC centroids
    voronoi_labels,              # (H, W) Voronoi label map
    method,                      # "radial_symmetry" or "isophote_curvature"
    upsample_factor=4,
    boundary_margin=None,        # defaults to upsample_factor
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns (refined_centers, ok_mask, quality_meta)."""
```

### 4.4 Tiled Detection Compatibility

The tiled detection variants (`detect_centers_tiled`, `detect_centers_tiled_global_otsu`) process tiles independently. Voronoi partitioning requires global seed positions.

**For now**: The new methods are **not compatible with tiled detection**. If `refine="radial_symmetry"` is used with tiled detection, raise `NotImplementedError("Voronoi-partitioned methods are not supported with tiled detection")`. This is documented as a known limitation.

**Future**: Support tiled detection by collecting all coarse centers across tiles first, computing the global Voronoi map, then refining per-tile with boundary-aware overlap.

### 4.5 CenterResult.meta Additions

Quality metrics are returned as `(N_kept,)` arrays, filtered in parallel with `centers_xy`. Only blobs that pass the existing `ok` filter appear in both arrays.

For `refine="radial_symmetry"`:
```python
meta = {
    # ... existing fields ...
    "radial_symmetry_residual": np.ndarray,  # (N_kept,) per-blob weighted mean d²
    "voronoi_cell_area": np.ndarray,         # (N_kept,) per-blob cell area in pixels
    "upsample_factor": int,                  # upsampling factor used
}
```

For `refine="isophote_curvature"`:
```python
meta = {
    # ... existing fields ...
    "isophote_curvature_spread": np.ndarray,  # (N_kept,) per-blob vote spread
    "voronoi_cell_area": np.ndarray,          # (N_kept,) per-blob cell area in pixels
    "upsample_factor": int,
}
```

### 4.6 Visualization Function

In new file `subpx/visualization.py`:

```python
def draw_voronoi_boundaries(
    image,
    centers_xy,
    label_map=None,
    color=1.0,
    thickness=1,
    backend="auto",
) -> np.ndarray:
    """Overlay Voronoi partition boundaries on an image.

    Parameters
    ----------
    image : (H, W) ndarray
        Background image to overlay on.
    centers_xy : (N, 2) ndarray
        Seed center positions [x, y].
    label_map : (H, W) ndarray, optional
        Pre-computed Voronoi label map. Computed from centers_xy if None.
    color : float
        Intensity value for boundary pixels.
    thickness : int
        Boundary line width (morphological dilation iterations).
    backend : str
        "auto", "cpu", or "gpu".

    Returns
    -------
    overlay : (H, W) ndarray
        Copy of image with Voronoi boundaries drawn.
    """
```

### 4.7 Internal Module Structure

New/modified files:
```
subpx/
├── centers.py              # Add dispatch + NotImplementedError for CPU
├── _gpu/
│   ├── centers.py          # Add GPU implementations + new dispatch path
│   └── voronoi.py          # NEW: Voronoi partitioning (GPU)
└── visualization.py        # NEW: draw_voronoi_boundaries
```

---

## 5. GPU Implementation Strategy

### 5.1 Full Pipeline Sequence

```
detect_centers_gpu(image, refine="radial_symmetry", upsample_factor=4)
│
├─ [1] Segment: Otsu → morphology → connected components
│      → labels (H, W), stats (N, 5), coarse_centers (N, 2)
│
├─ [2] Voronoi: compute_voronoi_labels_gpu(coarse_centers, (H, W))
│      → voronoi_labels (H, W)
│      → cell_areas (N,)
│
├─ [3] Extract ROIs: for each blob, crop bbox from image + voronoi mask
│      → roi_list [(h_i, w_i)], mask_list [(h_i, w_i)]
│
├─ [4] Background fill: for each ROI, set ~mask pixels to median(border & mask)
│
├─ [5] Pad + Stack: pad to (H_max, W_max), stack → (N, H_max, W_max)
│      → roi_stack, mask_stack on GPU
│
├─ [6] Upsample: zoom each ROI 4× bicubic
│      → roi_up (N, H_max*4, W_max*4)
│
├─ [7] Boundary mask: upsample mask (nearest), erode by margin
│      → boundary_ok (N, H_max*4-1, W_max*4-1)  [for gradient midpoints]
│
├─ [8] Refine: batch radial symmetry or isophote curvature
│      → centers_upsampled (N, 2), residuals (N,)
│
├─ [9] Coordinate transform: upsampled → original image coords
│      → centers_xy (N, 2)
│
├─ [10] Filter: remove invalid centers (NaN, out-of-bounds)
│       → centers_xy (N_kept, 2), residuals (N_kept,), cell_areas (N_kept,)
│
└─ [11] Return: CenterResult(cp.asnumpy(centers_xy), method, "gpu", meta)
```

### 5.2 Voronoi on GPU — Grid-Accelerated Nearest Neighbor

1. **Compute grid cell size**: `cell_size = median_nearest_neighbor_distance(seeds_xy) * 1.5`.
   - Compute KNN with k=1 for all seeds (CuPy batch pairwise distance + argmin, excluding self)
   - Take median of the N nearest-neighbor distances
   - Multiply by 1.5 for margin

2. **Build grid index**: Assign each seed to grid cell `(floor(x/cell_size), floor(y/cell_size))`. Build a CSR-like structure: for each grid cell, store the list of seed indices.

3. **For each pixel** (parallelized over pixel grid):
   - Compute grid cell of pixel
   - Gather candidate seeds from 9 neighboring cells
   - Compute squared distances to candidates
   - Assign to nearest (argmin)

4. **Output**: (H, W) int32 label map

For moderate seed counts (< 50k), this is efficient on GPU. The bottleneck is the irregular gather (different pixels may have different numbers of candidates), which can be handled by padding candidates to a fixed max per cell.

### 5.3 Upsampling on GPU

`cupyx.scipy.ndimage.zoom` with `order=3` (bicubic).

**Start with per-ROI loop** (simple, correct):
```python
for i in range(N):
    roi_up[i] = cupyx.scipy.ndimage.zoom(roi_stack[i], upsample_factor, order=3)
```

Profile. If this is a bottleneck (unlikely for small ROIs), implement a batched bicubic kernel.

### 5.4 Memory Budget

For 10,000 blobs of 6×6 pixels, upsampled 4×:
- Original ROIs: 10k × 6 × 6 × 8 bytes (float64) = 2.9 MB
- Upsampled ROIs: 10k × 24 × 24 × 8 bytes = 46 MB
- Gradient arrays: 10k × 23 × 23 × 8 bytes × 5 intermediates = ~212 MB
- Voronoi label map (4096×4096): 64 MB (int32)
- **Total**: ~325 MB — well within 44 GB VRAM

---

## 6. Edge Cases and Failure Modes

### 6.1 Blobs at image boundary
Voronoi cells at the image edge are clipped by the image boundary. The ROI may be asymmetric. The radial symmetry method handles this naturally (fewer gradient vectors on one side, but the least-squares still works). Flag these blobs via the residual metric.

### 6.2 Single-pixel or 2-pixel blobs
Below 3×3, even 4× upsampling gives a tiny gradient array. The goodness-of-fit residual will be high, signaling unreliability. Do not reject — let the user filter by residual.

### 6.3 Elongated (non-circular) blobs
The radial symmetry method assumes radial symmetry. Elongated PSFs (astigmatism, coma) will produce biased results. The residual metric will be elevated, indicating poor radial symmetry. The isophote curvature spread will also be large.

### 6.4 Single blob detected
If only one blob exists, Voronoi assigns the entire image to it. This is harmless — the ROI is just the full bbox (or clipped by a reasonable radius), and refinement proceeds normally.

### 6.5 Zero-gradient regions
Flat regions (background, blob peak) produce zero gradients. The Parthasarathy weighting (|grad|² in numerator) naturally gives these zero weight. No special handling needed.

### 6.6 Degenerate least-squares (det ≈ 0)
If `det = smw² - smmw * sw ≈ 0`, the system is degenerate (all gradient lines are parallel or all weights are zero). Mark this blob as failed (`ok = False`). Report NaN center.

---

## 7. Testing Strategy

### 7.1 Synthetic tests
- **Isolated Gaussian blob**: Known center at various subpixel positions, verify accuracy < 0.05 px for sigma >= 1.0 px
- **Closely-packed Gaussian array**: Known centers at pitch 3× and 2× sigma, verify no systematic neighbor-induced bias > 0.02 px
- **Staggered array**: Verify Voronoi partitioning produces correct cell assignments against scipy.spatial.Voronoi reference
- **Varying blob sizes** (3-6 px): Verify upsampling improves accuracy relative to no upsampling (measure RMS error at each size)
- **Noise sweep**: Gaussian noise at SNR 5, 10, 20, 50 — verify graceful degradation and monotonic correlation of residual metric with actual error

### 7.2 Regression tests
- Existing `detect_centers` tests must pass unchanged (new methods are additive)
- New methods must return valid `CenterResult` with correct meta fields and shapes
- `refine="radial_symmetry"` with `backend="cpu"` must raise `NotImplementedError`
- `refine="radial_symmetry"` with tiled detection must raise `NotImplementedError`

### 7.3 Diagnostic tests
- Isophote curvature center should agree with radial symmetry center within 0.1 px for isolated blobs
- Curvature spread should increase when neighbors are artificially brought closer
- Voronoi visualization should produce closed cell boundaries matching scipy.spatial.Voronoi

---

## 8. Dependencies

- **CuPy** (existing optional dependency) — all GPU operations
- **cupyx.scipy.ndimage.zoom** — bicubic upsampling
- **No new dependencies**

---

## 9. Limitations and Future Work

### Current limitations
- GPU-only implementation (CPU fallback not in scope)
- Fixed upsampling factor (no adaptive per-blob sizing)
- No sector masking (Voronoi partitioning is the only contamination defense)
- Isophote curvature diagnostic is marginal for blobs < 5px original (second derivatives on small ROIs are noisy even after upsampling)
- Not compatible with tiled detection (needs global Voronoi map)

### Future improvements
- **Analytic gradient computation**: Replace bicubic upsampling + discrete gradients with analytic gradient evaluation from bicubic polynomial coefficients. Mathematically equivalent to infinite upsampling with zero discretization error.
- **CPU fallback**: Implement NumPy versions of both methods for environments without CuPy.
- **Sector masking**: For extremely tight packing where Voronoi still includes contaminated boundary pixels.
- **Adaptive upsampling**: Per-blob factor based on actual bbox size, if edge-case concerns are resolved.
- **Tiled detection support**: Collect all coarse centers across tiles, compute global Voronoi, refine per-tile with overlap.
