# Dedicated Tiled GPU Pipeline for `detect_centers_tiled`

**Date:** 2026-03-23
**Status:** Proposed
**Scope:** `subpx/_gpu/centers.py`, `subpx/centers.py`

## Problem

`detect_centers_tiled` is ~10x slower than the user's original `find_subpixel_centers_tiled_hybrid_gpu` function despite being intended as its replacement. Root causes:

1. **Per-tile Otsu** — each tile computes its own threshold via `_segment_binary`, vs one global threshold in the reference.
2. **Python for-loop over components** — `detect_centers_gpu` iterates every component in Python (line 855-869) instead of vectorized filtering.
3. **CPU connected components by default** — `components_backend="cpu"` forces unnecessary CPU→GPU transfer.
4. **O(N^2) deduplication** — `dedupe_centers` uses all-pairs distance in Python.
5. **Per-tile GPU context overhead** — each tile independently enters GPU context, uploads image, creates arrays.
6. **Different overlap stepping** — `step = tile_h - 2*overlap` vs reference's `step = tile_h - overlap`.

These issues compound: the generic `detect_centers` API was designed for single-image use, not for tight tiled loops.

## Solution

Add `_detect_centers_tiled_gpu()` in `_gpu/centers.py` — a dedicated tiled pipeline that mirrors the reference's structure. Wire `detect_centers_tiled()` to dispatch to it when `backend="gpu"`.

## Architecture

```
detect_centers_tiled(image, backend="gpu", ...)
    └─► _detect_centers_tiled_gpu(image, ...)
            ├─ Global Otsu (once, downsampled, scaled by thr_scale)
            ├─ Per-tile: fixed threshold + morphology (CPU, OpenCV)
            ├─ Per-tile: connected_components_stats_gpu (GPU CC)
            ├─ Per-tile: vectorized area filter (NumPy, no Python loop)
            ├─ Per-tile: refinement dispatch (batched GPU)
            └─ Per-tile: band-based de-dup (O(N))

detect_centers_tiled(image, backend="gpu" but CuPy unavailable)
    └─► resolve_backend raises RuntimeError (consistent with rest of codebase)

detect_centers_tiled(image, backend="auto", ...)
    └─► resolve_backend("auto") returns "gpu" if CuPy available, else "cpu"
        ├─ "gpu" → _detect_centers_tiled_gpu (same as above)
        └─ "cpu" → existing tile-by-tile detect_centers() path (unchanged)

detect_centers_tiled(image, backend="cpu", ...)
    └─► existing tile-by-tile detect_centers() path (unchanged)
```

Note: `detect_centers_tiled` defaults to `backend="gpu"`. On a CPU-only machine, callers must use `backend="auto"` or `backend="cpu"` explicitly.

## New Function: `_detect_centers_tiled_gpu`

### Location

`subpx/_gpu/centers.py`

### Signature

```python
def _detect_centers_tiled_gpu(
    image: np.ndarray,
    *,
    invert: bool = False,
    area_min: int = 1,
    area_max: int = 50,
    morph_open: int = 0,
    morph_close: int = 0,
    tile_h: int = 8192,
    overlap: int = 128,
    otsu_downsample: int = 4,
    thr_scale: float = 0.8,
    refine: str = "edge_gradmoment",
    connectivity: int = 8,
    device: int = 0,
    # edge_gradmoment params
    band_rad: int = 4,
    edge_rad: int = 10,
    smooth_passes: int = 2,
    loc_rad: int = 3,
    grad_power: float = 8.0,
    iters: int = 2,
    # logquad/weighted params (used only by those methods, ignored by edge_gradmoment)
    pad: int = 3,
    gpu_batch: int = 4096,
    use_float64: bool = True,
) -> CenterResult:
```

`use_float64` is only forwarded to `logquad` and `weighted` refinement branches. `refine_centers_edge_moment_gpu` always operates in float32 internally. This parameter is ignored when `refine="edge_gradmoment"`.

### Input validation

```python
if overlap >= tile_h:
    raise ValueError("overlap must be < tile_h")
```

This check is the responsibility of `_detect_centers_tiled_gpu` itself, not just the public wrapper.

### Pipeline Steps

#### Step 0: Global threshold (once)

```python
thr_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
g_ds = image[::otsu_downsample, ::otsu_downsample]
otsu_thresh, _ = cv2.threshold(g_ds, 0, 255, thr_type | cv2.THRESH_OTSU)
thr = float(otsu_thresh) * thr_scale
```

#### Step 1: Tile loop

```python
step = tile_h - overlap
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) if (morph_open > 0 or morph_close > 0) else None

# No outer gpu_device() context here. Each GPU function (connected_components_stats_gpu,
# refine_centers_edge_moment_gpu) manages its own device context internally.
# An outer context would be redundant since CuPy device contexts are re-entrant.
y0 = 0
while y0 < H:
    y1 = min(H, y0 + tile_h)
    tile = image[y0:y1]
    # ... per-tile processing (Steps 2-6) ...
    if y1 == H:
        break
    y0 += step
```

**Single-tile case**: When `H <= tile_h`, the loop runs exactly once with `y0=0, y1=H`. The band logic sets `keep_lo=0, keep_hi=H`, keeping all centers. This is correct and requires no special handling.

#### Step 2: Per-tile threshold + morphology (CPU)

```python
_, bw = cv2.threshold(tile, thr, 255, thr_type)
if morph_open > 0:
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=morph_open)
if morph_close > 0:
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=morph_close)
```

#### Step 3: GPU connected components

```python
comp = connected_components_stats_gpu(bw, connectivity=connectivity, device=device)
```

#### Step 4: Vectorized area filtering (no Python loop)

```python
stats = comp.stats          # (num_labels, 5) NumPy int32: [min_x, min_y, w, h, area]
centroids = comp.centroids  # (num_labels, 2) NumPy float64: [cx, cy]
num = comp.num_labels

# num_labels includes background (label 0). num_labels == 1 means zero foreground components.
if num <= 1:
    continue

area = stats[1:, 4]  # skip background label 0
mask = (area >= area_min) & (area <= area_max)
if not np.any(mask):
    continue

filtered_stats = stats[1:][mask]       # (K, 5): min_x, min_y, w, h, area
filtered_cents = centroids[1:][mask]   # (K, 2): cx, cy  (float64 from GPU CC)

# Build (K, 6) array: [x, y, w, h, cx, cy] — drop area column (index 4) via [:, :4]
stats_xywh_cc = np.concatenate([
    filtered_stats[:, :4].astype(np.float32),  # x, y, w, h (int32 → float32)
    filtered_cents.astype(np.float32),          # cx, cy (float64 → float32, acceptable
                                                # since refine_edge_moment_gpu converts to
                                                # cp.float32 anyway)
], axis=1)  # (K, 6)
```

#### Step 5: Refinement dispatch

All methods use the vectorized tiled pipeline. No fallback to per-tile `detect_centers`.

```python
if refine == "edge_gradmoment":
    refined, ok = refine_centers_edge_moment_gpu(
        tile, stats_xywh_cc,
        band_rad=band_rad, edge_rad=edge_rad,
        smooth_passes=smooth_passes, loc_rad=loc_rad,
        grad_power=grad_power, iters=iters, device=device,
    )
elif refine in ("logquad", "logquadratic"):
    # Build GPU arrays, call refine_centers_logquad_gpu_match_cpu
    raise NotImplementedError(
        "refine='logquad' is not yet supported in the tiled GPU pipeline. "
        "Use refine='edge_gradmoment' or the non-tiled detect_centers_gpu()."
    )
elif refine == "weighted":
    raise NotImplementedError(
        "refine='weighted' is not yet supported in the tiled GPU pipeline. "
        "Use refine='edge_gradmoment' or the non-tiled detect_centers_gpu()."
    )
elif refine == "edge_erf":
    raise NotImplementedError(
        "refine='edge_erf' is not yet supported in the tiled GPU pipeline. "
        "Use refine='edge_gradmoment' or the non-tiled detect_centers_gpu()."
    )
elif refine == "auto":
    raise NotImplementedError(
        "refine='auto' is not yet supported in the tiled GPU pipeline. "
        "Use refine='edge_gradmoment' or the non-tiled detect_centers_gpu()."
    )
else:
    raise ValueError(f"Unknown refine method: {refine}")
```

**This round:** `edge_gradmoment` fully implemented and tested. All other methods raise `NotImplementedError` with clear guidance. They will be implemented in future rounds using the same shared tiling/filtering/dedup infrastructure.

#### Step 6: Global coordinate offset + band-based de-dup

```python
refined = np.asarray(refined, dtype=np.float64)
ok = np.asarray(ok, dtype=bool)
good = ok & np.all(np.isfinite(refined), axis=1)
refined = refined[good]
refined[:, 1] += y0  # tile-local → global

half = overlap // 2
keep_lo = y0 if y0 == 0 else (y0 + half)
keep_hi = y1 if y1 == H else (y1 - half)
m = (refined[:, 1] >= keep_lo) & (refined[:, 1] < keep_hi)
if np.any(m):
    all_centers.append(refined[m])
```

## Changes to `detect_centers_tiled`

### File: `subpx/centers.py`, function at line 184

**New parameters added:**
- `thr_scale: float = 0.8`

**Dispatch logic:**
```python
def detect_centers_tiled(image, *, backend="gpu", tile_h=8192, overlap=128,
                         otsu_downsample=4, thr_scale=0.8, dedupe_eps=1.5, **kwargs):
    ...
    b = resolve_backend(backend)
    if b == "gpu":
        from ._gpu.centers import _detect_centers_tiled_gpu
        return _detect_centers_tiled_gpu(
            img, tile_h=tile_h, overlap=overlap,
            otsu_downsample=otsu_downsample, thr_scale=thr_scale, **kwargs,
        )
    # CPU path: unchanged tile-by-tile detect_centers approach
    ...
```

**Overlap stepping for CPU path** also updated to `step = tile_h - overlap` with band-based de-dup for consistency.

## Known Limitations (This Round)

- **`dedupe_eps` ignored in GPU path**: The GPU tiled pipeline uses band-based de-dup (O(N) per tile) instead of the O(N^2) `dedupe_centers` union-find. The `dedupe_eps` parameter on `detect_centers_tiled` is only used by the CPU path. This is intentional — band-based de-dup is how the reference function works. The parameter remains in the public signature for CPU-path callers.
- **`refine="auto"` not yet supported**: The non-tiled `detect_centers_gpu` supports `refine="auto"` (dispatching per-component by bbox size), but the tiled GPU pipeline raises `NotImplementedError` for it. This will be implemented alongside the other refine methods in a future round.

## What Does NOT Change

- `detect_centers_gpu` — non-tiled single-image API stays as-is
- `refine_centers_edge_moment_gpu` — reused as-is
- `connected_components_stats_gpu` — reused as-is
- `detect_centers_tiled_global_otsu` — stays as separate variant (may be deprecated later)
- `dedupe_centers` — stays available for other callers, just not used in GPU tiled path

## Performance Expectations

| Bottleneck | Before | After |
|---|---|---|
| Threshold | Per-tile Otsu | 1 global Otsu |
| Component filtering | Python for-loop | Vectorized NumPy |
| Connected components | CPU by default | GPU |
| Deduplication | O(N^2) union-find | O(N) band-based |
| GPU context | Per-tile setup | Single context |
| Refinement | logquad (different algo) | edge_gradmoment (matches reference) |

Expected speedup: ~10x, bringing performance in line with the reference function.

## Testing Strategy

1. **Correctness**: Compare output of `detect_centers_tiled(backend="gpu", refine="edge_gradmoment")` against the reference `find_subpixel_centers_tiled_hybrid_gpu` on the same image. Centers should match within subpixel tolerance (~0.1 px).
2. **Regression**: Existing `test_detect_centers_tiled_smoke` must still pass.
3. **Performance**: Time comparison on a real photomask image to verify the 10x gap is closed.
4. **NotImplementedError paths**: Test that `refine="logquad"`, `"weighted"`, `"edge_erf"`, and `"auto"` raise `NotImplementedError` with a helpful message.

## Implementation Order

1. Add `_detect_centers_tiled_gpu` in `_gpu/centers.py` with `edge_gradmoment` fully working
2. Update `detect_centers_tiled` in `centers.py` to dispatch to new function for GPU backend
3. Update CPU path overlap stepping for consistency
4. Add tests
5. (Future) Implement remaining refine methods in the tiled pipeline
