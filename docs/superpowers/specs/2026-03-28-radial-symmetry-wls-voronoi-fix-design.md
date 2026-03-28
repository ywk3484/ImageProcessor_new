# Radial Symmetry WLS + Voronoi Fix Design Spec

## Problem Statement

Four issues identified in the radial symmetry center detection pipeline:

1. **Center detected outside the circle** despite gradients pointing inward
2. **Gradient mask clips parts of the circle** — erosion too aggressive
3. **Strip-shaped Voronoi cells** in diagnostic visualization
4. **Non-converging gradient lines** — need better understanding of WLS center determination

## Root Cause Analysis

### RC1: Missing `1/(m² + 1)` normalization in WLS (Critical)

**Verified against Parthasarathy reference MATLAB implementation** ([GitHub](https://github.com/rplab/TrackingGUI_and_Localization_Public/blob/master/radialcenter.m)).

The reference code normalizes weights by the perpendicular-distance factor:

```matlab
wm2p1 = w./(m.*m+1);
sw  = sum(sum(wm2p1));
smmw = sum(sum(m.*m.*wm2p1));
smw  = sum(sum(m.*wm2p1));
smbw = sum(sum(m.*b.*wm2p1));
sbw  = sum(sum(b.*wm2p1));
```

Our implementation (`_radial_symmetry_batch_gpu`, lines 1126-1130) uses raw `w` without normalization:

```python
sw = w.sum(axis=(1, 2))           # MISSING: / (slope**2 + 1)
smmw = (slope**2 * w).sum(...)    # MISSING: / (slope**2 + 1)
```

**Impact**: Without `1/(m² + 1)`, the WLS minimizes weighted *vertical* distance instead of *perpendicular* distance. Near-vertical gradient lines get disproportionate influence. When the gradient distribution is non-uniform (e.g., from partial masking), this pulls the center estimate outside the feature.

### RC2: Missing 3×3 gradient smoothing (High)

The reference applies a 3×3 averaging filter to the diagonal gradients before computing slopes:

```matlab
h = ones(3)/9;
fdu = conv2(dIdu, h, 'same');
fdv = conv2(dIdv, h, 'same');
```

Our implementation uses raw unsmoothed gradients. The original paper notes this smoothing reduces mean localization error from 0.04 to 0.027 pixels.

### RC3: Erosion margin too aggressive (High)

Default `boundary_margin = upsample_factor = 4` upsampled pixels. For a 5px blob upsampled 4×:
- Upsampled diameter: ~20px
- Erosion from each side: 4px
- Valid diameter: ~12px (40% loss)

This clips gradients at the feature edge, especially when the Voronoi cell is narrow in one direction (small pitch).

### RC4: Diagnostic colormap creates false visual merging (Medium)

`inspect_blob` uses `tab20` (20 discrete colors) for Voronoi labels. With N > 20 blobs, cells whose indices differ by a multiple of 20 get identical colors. For a regular grid, vertically adjacent cells often satisfy this condition, creating apparent "horizontal strips" that look like merged cells. The actual Voronoi partitioning is correct — only the visualization is misleading.

## Design

### Fix A: WLS perpendicular-distance normalization

**File**: `subpx/_gpu/centers.py`, function `_radial_symmetry_batch_gpu`, lines 1126-1135

Change the WLS sums to use normalized weights:

```python
# Perpendicular distance normalization (matches Parthasarathy reference)
wm2p1 = w / (slope**2 + 1.0)        # (N, Hp, Wp)
sw   = wm2p1.sum(axis=(1, 2))
smmw = (slope**2 * wm2p1).sum(axis=(1, 2))
smw  = (slope * wm2p1).sum(axis=(1, 2))
smbw = (slope * b * wm2p1).sum(axis=(1, 2))
sbw  = (b * wm2p1).sum(axis=(1, 2))
```

The determinant, xc, yc formulas remain the same. The residual computation at line 1141 already uses `1/(slope**2 + 1.0)` — no change needed there.

### Fix B: 3×3 gradient smoothing

**File**: `subpx/_gpu/centers.py`, function `_radial_symmetry_batch_gpu`, after lines 1065-1066

Apply a `ones(3)/9` averaging filter to dIdu and dIdv:

```python
if Hp >= 3 and Wp >= 3:
    kern = cp.ones((1, 3, 3), dtype=cp.float64) / 9.0
    dIdu = _conv2d_same(dIdu, kern)
    dIdv = _conv2d_same(dIdv, kern)
```

Helper `_conv2d_same` uses CuPy pad + convolution (or direct indexing for a 3×3 kernel for simplicity — manual stencil avoids FFT overhead for such a small kernel).

### Fix C: Reduce erosion margin

**File**: `subpx/_gpu/centers.py`, function `_radial_symmetry_batch_gpu`, line 1007

Change default from `upsample_factor` to `max(1, upsample_factor // 2)`:

```python
if boundary_margin is None:
    boundary_margin = max(1, upsample_factor // 2)
```

For 4× upsample: erosion goes from 4 to 2 upsampled pixels (0.5 original pixels). Combined with the 3×3 gradient smoothing (Fix B), boundary artifacts are sufficiently attenuated at this reduced margin.

### Fix D: Diagnostic Voronoi colormap

**File**: `scripts/diagnose_radial_symmetry.py`, function `inspect_blob`

Replace `tab20` with a shuffled colormap that avoids color collisions:

```python
from matplotlib.colors import ListedColormap
rng = np.random.default_rng(0)
n_labels = voronoi_labels.max() + 1
colors = rng.random((n_labels, 3))
cmap = ListedColormap(colors)
ax.imshow(vlabels, cmap=cmap, origin="upper", interpolation="nearest")
```

### Fix E: Update diagnostic `_inspect_radial_symmetry_steps`

**File**: `scripts/diagnose_radial_symmetry.py`, function `_inspect_radial_symmetry_steps`

Apply fixes A, B, C to the manual re-implementation of the radial symmetry algorithm so diagnostic output matches the production pipeline.

## Testing Strategy

1. **Regression test**: Run existing tests to verify no breakage
2. **Accuracy test**: Run `diagnose_radial_symmetry.py` on test image — verify:
   - Distance median and 95th percentile decrease (especially max)
   - No centers detected outside features
   - Gradient mask no longer clips features aggressively
3. **Visual verification**: Voronoi labels in `inspect_blob` show distinct colors per cell

## Files Modified

| File | Change |
|------|--------|
| `subpx/_gpu/centers.py` | Fixes A, B, C in `_radial_symmetry_batch_gpu` |
| `scripts/diagnose_radial_symmetry.py` | Fixes D, E in `inspect_blob` and `_inspect_radial_symmetry_steps` |
