# Center Detection API Unification & GPU Memory Fix

**Date:** 2026-03-30
**Status:** Draft

## Problem Statement

The center detection module has three public entry points (`detect_centers`, `detect_centers_tiled`, `detect_centers_tiled_global_otsu`) that each implement overlapping but divergent logic. This causes:

1. **Code divergence:** Each function has its own implementation of segmentation, filtering, refinement dispatch, and coordinate transforms. Bug fixes applied to one may not propagate to the others.
2. **API confusion:** `gpu_batch` (refinement batching) vs `tile_h` (spatial tiling) serve different purposes but users must understand both to choose the right function.
3. **GPU OOM for Voronoi methods:** `_radial_symmetry_batch_gpu` and `_isophote_curvature_batch_gpu` process ALL ROIs simultaneously without batching, causing OOM on large images (~30 GB for 10k blobs with 4× upsampling).

## Design

### 1. Unified Public API

Merge all three functions into a single `detect_centers()`:

```python
def detect_centers(
    image: np.ndarray,
    *,
    # Core detection params
    backend: str = "auto",
    threshold: str = "triangle",
    invert: bool = False,
    area_min: int = 1,
    area_max: int = 50,
    morph_open: int = 0,
    morph_close: int = 0,
    pad: int = 3,
    refine: str = "logquad",
    connectivity: int = 8,
    gpu_batch: int = 4096,
    device: int = 0,
    use_float64: bool = True,
    components_backend: str = "cpu",
    small_feature_max: float = 12.0,
    upsample_factor: int = 4,
    # Tiling params (new — defaults preserve current non-tiled behavior)
    tile_h: int | None = None,
    overlap: int = 128,
    threshold_mode: str = "auto",
    otsu_downsample: int = 4,
    thr_scale: float = 0.8,
    dedupe_eps: float = 1.5,
) -> CenterResult:
```

**`tile_h` behavior:**
- `None` (default): process the entire image at once (current `detect_centers` behavior)
- `int`: split image into vertical tiles of this height, process each, dedup at boundaries

**`threshold_mode` behavior:**
- `"auto"`: when `tile_h is None`, use per-image threshold (respects `threshold` param); when tiling, use `"global"`
- `"per_tile"`: each tile gets its own threshold using the method specified by `threshold` param (useful when illumination varies across the image)
- `"global"`: compute one Otsu threshold on downsampled image (ignores `threshold` param since global mode always uses Otsu), apply `thr_scale` multiplier, binarize all tiles with this fixed threshold

### 2. Deprecated Aliases

Add to `legacy.py`:

```python
def detect_centers_tiled(image, *, tile_h=8192, **kwargs):
    """Deprecated: use detect_centers(image, tile_h=...) instead."""
    warnings.warn("detect_centers_tiled is deprecated, use detect_centers(tile_h=...)", DeprecationWarning)
    return detect_centers(image, tile_h=tile_h, **kwargs)
```

Similarly for `detect_centers_tiled_global_otsu` (maps `tile_h` + `threshold_mode="global"`).

Remove `detect_centers_tiled` and `detect_centers_tiled_global_otsu` from `centers.py`. Remove `detect_centers_tiled_batch` from `batch.py` (or make it call `detect_centers_batch` with `tile_h`).

### 3. Internal Architecture

#### 3.1 GPU Backend (`_gpu/centers.py`)

Merge `detect_centers_gpu()` and `_detect_centers_tiled_gpu()` into one function:

```
detect_centers_gpu(image, *, tile_h=None, threshold_mode="auto", ...)
  ├─ if tile_h is not None:
  │    ├─ compute global threshold if threshold_mode in {"global", "auto"}
  │    └─ for each tile:
  │         └─ _detect_single_tile_gpu(tile, threshold=thr, ...)
  │              ├─ binarize (using pre-computed or per-tile threshold)
  │              ├─ connected components (GPU or CPU)
  │              ├─ vectorized area filter
  │              └─ refinement dispatch → shared refine functions
  └─ if tile_h is None:
       └─ _detect_single_tile_gpu(image, ...)
```

`_detect_single_tile_gpu()` is the shared core extracted from the current non-tiled path of `detect_centers_gpu()`. It handles:
- Segmentation (binary threshold)
- Connected components
- Area filtering
- Method dispatch (logquad, weighted, edge_gradmoment, edge_erf, radial_symmetry, isophote_curvature)
- Returns tile-local centers

The tiling loop handles:
- Global threshold computation
- Tile extraction
- Coordinate offset (tile-local → image-global)
- Band-based dedup at tile boundaries

#### 3.2 CPU Backend (`_cpu/centers.py`)

`detect_centers_cpu()` gains `tile_h` parameter. When set, it loops over tiles internally (same pattern as current CPU path in `detect_centers_tiled`).

### 4. GPU Memory Optimization for Voronoi Methods

#### 4.1 Batch Loop

Add `gpu_batch` parameter to `_radial_symmetry_batch_gpu()` and `_isophote_curvature_batch_gpu()`:

```python
def _radial_symmetry_batch_gpu(rois, masks, *, gpu_batch=2048, ...):
    N = rois.shape[0]
    centers = np.empty((N, 2), dtype=np.float64)
    residuals = np.empty((N,), dtype=np.float64)

    for s in range(0, N, gpu_batch):
        e = min(N, s + gpu_batch)
        # Process batch [s:e] on GPU
        c, r = _radial_symmetry_batch_gpu_inner(rois[s:e], masks[s:e], ...)
        centers[s:e] = c
        residuals[s:e] = r

    return centers, residuals
```

Memory per batch of B=2048, 20×20 ROIs, factor=4:
- Input: B × 80 × 80 × 4 bytes (float32) = 50 MB
- Mask: B × 80 × 80 × 1 byte = 13 MB
- ~6 intermediate arrays (gradients, slopes, weights): ~300 MB
- Total per batch: ~400 MB (vs ~30 GB unbatched)

#### 4.2 Float32 Internal Precision

Use float32 for:
- Bicubic upsampled ROIs
- Gradient computation (dIdu, dIdv)
- Slope and intercept computation
- Weight computation

Keep float64 for:
- The 2×2 WLS solve (small, precision matters)
- Final center coordinates (output)

Requires adding a float32 variant of the CUDA upsample kernel (currently hardcoded to `double`). Make kernel dtype-generic by having both float and double versions or templating.

#### 4.3 Vectorize Python Loops

**`_extract_voronoi_rois()` (lines 1401-1465):** Currently a Python loop over every blob doing array slicing. Vectorize the ROI extraction:
- Compute all bounding boxes at once with numpy
- Use a single padded array allocation
- Background estimation can use vectorized border-pixel extraction

**Coordinate transform loops** (lines 1169-1194 in radial symmetry, lines 1372-1397 in isophote curvature): Replace per-ROI Python loops with vectorized numpy operations:
```python
# Before: for i in range(N): centers[i,0] = xc_cpu[i] + (w_up-1)/2 ...
# After:
xc_pix_up = xc_cpu + (up_ws - 1.0) / 2.0
yc_pix_up = yc_cpu + (up_hs - 1.0) / 2.0
centers[:, 0] = xc_pix_up * (orig_ws - 1.0) / (up_ws - 1.0)
centers[:, 1] = yc_pix_up * (orig_hs - 1.0) / (up_hs - 1.0)
```

### 5. Files Changed

| File | Change |
|------|--------|
| `subpx/centers.py` | Remove `detect_centers_tiled`, `detect_centers_tiled_global_otsu`. Add tiling params to `detect_centers`. |
| `subpx/_gpu/centers.py` | Merge `detect_centers_gpu` + `_detect_centers_tiled_gpu` into one. Add batching to Voronoi methods. Float32 internals. Vectorize loops. |
| `subpx/_cpu/centers.py` | Add `tile_h` tiling loop to `detect_centers_cpu`. |
| `subpx/_gpu/upsample.py` | Add float32 kernel variant alongside existing float64. |
| `subpx/batch.py` | Remove `detect_centers_tiled_batch` (or alias to `detect_centers_batch(tile_h=...)`). |
| `subpx/legacy.py` | Add `detect_centers_tiled` and `detect_centers_tiled_global_otsu` deprecated aliases. |
| `tests/` | Update tests that call `detect_centers_tiled` to use `detect_centers(tile_h=...)`. |

### 6. Known Issue (Not Addressed)

Voronoi cells at image edges may include adjacent partial circles that are clipped. This is a minor issue handled through post-processing of the final centers (e.g., `filter_centers` with margin). No fix planned for the Voronoi partitioning itself.

### 7. Testing Strategy

1. **Regression tests:** Existing tests for `detect_centers` must pass unchanged.
2. **Tiling equivalence:** `detect_centers(img, tile_h=8192)` should produce results within tolerance of current `detect_centers_tiled(img, tile_h=8192)`.
3. **GPU memory:** Profile `_radial_symmetry_batch_gpu` on a 10k-blob image to confirm peak memory stays within 4 GB (vs 30 GB before).
4. **Float32 accuracy:** Compare radial symmetry centers computed in float32 vs float64 internals. Acceptable if max deviation < 0.01 px.
5. **Deprecated aliases:** Verify `detect_centers_tiled()` raises `DeprecationWarning` and produces identical results.
